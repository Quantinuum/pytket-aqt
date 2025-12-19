# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backendinfo import BackendInfo, fully_connected_backendinfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType, Qubit
from pytket.passes import (
    AutoRebase,
    BasePass,
    DecomposeBoxes,
    EulerAngleReduction,
    FlattenRegisters,
    FullPeepholeOptimise,
    RenameQubitsPass,
    SequencePass,
    SynthesiseTket,
)
from pytket.predicates import (
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)

from .._metadata import __extension_version__
from ..backends.config import AQTConfig
from ..multi_zone_architecture.circuit.multizone_circuit import (
    MultiZoneCircuit,
)
from ..multi_zone_architecture.circuit_routing.legacy.route_circuit import (
    route_circuit_legacy,
)
from ..multi_zone_architecture.circuit_routing.route_circuit import route_circuit
from ..multi_zone_architecture.compilation_settings import CompilationSettings
from ..multi_zone_architecture.initial_placement.initial_placement_generators import (
    get_initial_placement,
)
from ..multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
)

if TYPE_CHECKING:
    from pytket.unit_id import UnitID

AQT_URL_PREFIX = "https://gateway.aqt.eu/marmot/"

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"

_GATE_SET = {
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.XXPhase,
    OpType.Measure,
    OpType.Barrier,
}

AQTResult = tuple[int, list[int]]  # (n_qubits, list of readouts)

# TODO add more
_STATUS_MAP = {
    "finished": StatusEnum.COMPLETED,
    "error": StatusEnum.ERROR,
    "queued": StatusEnum.QUEUED,
}

_DEFAULT_COMPILATION_SETTINGS = CompilationSettings()


class AqtAuthenticationError(Exception):
    """Raised when there is no AQT access token available."""

    def __init__(self) -> None:
        super().__init__("No AQT access token provided or found in config file.")


class AQTMultiZoneBackend(Backend):
    """
    Interface to an AQT device with multiple compute zones.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        architecture: MultiZoneArchitectureSpec,
        device_name: str = "multi_zone",
        access_token: str | None = None,
        label: str = "",
    ):
        """
        Construct a new AQT backend for multi zone architectures.

        This backend currently only supports compilation of circuits
         of type MultiZoneCircuit.
        Submission of circuits to a quantum machine is not supported.

        The backend is currently for experimental purposes only.

        :param      device_name:  device name (suffix of URL, e.g. "sim/noise-model-1")
        :type       device_name:  string
        :param      access_token: AQT access token, default None
        :type       access_token: string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        """
        super().__init__()
        self._url = AQT_URL_PREFIX + device_name
        self._label = label
        config = AQTConfig.from_default_config_file()

        if access_token is None:
            access_token = config.access_token
        if access_token is None:
            raise AqtAuthenticationError

        self._header = {"Ocp-Apim-Subscription-Key": access_token, "SDK": "pytket"}
        self._backend_info: BackendInfo | None = None
        self._qm: dict[Qubit, Qubit] = {}
        self._architecture = architecture
        self._backend_info = fully_connected_backendinfo(
            type(self).__name__,
            device_name,
            __extension_version__,
            architecture.n_qubits_max,
            _GATE_SET,
        )
        self._qm = {
            Qubit(i): cast("Qubit", node)
            for i, node in enumerate(self._backend_info.nodes)
        }
        self._MACHINE_DEBUG = True

    def rebase_pass(self) -> BasePass:
        return _aqt_rebase()

    @property
    def backend_info(self) -> BackendInfo | None:
        return self._backend_info

    @classmethod
    def available_devices(cls, **kwargs: Any) -> list[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: none.
        """
        return []

    @property
    def required_predicates(self) -> list[Predicate]:
        preds = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            NoMidMeasurePredicate(),
            NoSymbolsPredicate(),
            GateSetPredicate(_GATE_SET),
        ]
        if self._backend_info is not None:
            preds.append(MaxNQubitsPredicate(self._backend_info.n_nodes))
        return preds

    def default_compilation_pass(self, optimisation_level: int = 2) -> SequencePass:
        assert optimisation_level in range(3)
        if optimisation_level == 0:
            return SequencePass(
                [
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    DecomposeBoxes(),
                    self.rebase_pass(),
                ]
            )
        if optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseTket(),
                    FlattenRegisters(),
                    RenameQubitsPass(self._qm),
                    self.rebase_pass(),
                    EulerAngleReduction(OpType.Ry, OpType.Rx),
                ]
            )
        return SequencePass(
            [
                DecomposeBoxes(),
                FullPeepholeOptimise(),
                FlattenRegisters(),
                RenameQubitsPass(self._qm),
                self.rebase_pass(),
                EulerAngleReduction(OpType.Ry, OpType.Rx),
            ]
        )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, str, str)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[int | None] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: none.
        """
        raise NotImplementedError

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: dict[str, BackendResult]
    ) -> None:
        raise NotImplementedError

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        raise NotImplementedError

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        raise NotImplementedError

    def compile_circuit(
        self,
        circuit: Circuit,
        compilation_settings: CompilationSettings = _DEFAULT_COMPILATION_SETTINGS,
    ) -> Circuit:
        """
        Compile a pytket Circuit assuming all to all connectivity

        Returns a pytket Circuit that conforms to the backend
         architectures gate set but does not necessarily respect connectivity

        """
        if not circuit.is_simple:
            raise ValueError(f"{type(self).__name__} only supports simple circuits")
        compiled = super().get_compiled_circuit(
            circuit, compilation_settings.pytket_optimisation_level
        )
        # compilation renames qbit register to "fcNode" so rename back to "q"
        qubit_map = {
            cast("UnitID", qubit): cast("UnitID", Qubit(qubit.index[0]))
            for qubit in compiled.qubits
        }
        compiled.rename_units(qubit_map)
        return compiled

    def route_compiled(
        self,
        compiled_circuit: Circuit,
        compilation_settings: CompilationSettings = _DEFAULT_COMPILATION_SETTINGS,
    ) -> MultiZoneCircuit:
        """
        Route a pytket Circuit to the backend architecture

        Does not perform gate optimization and inserts only the necessary shuttles
         and swaps. Returns a MultiZoneCircuit.

        """
        initial_placement = get_initial_placement(
            compilation_settings.initial_placement, compiled_circuit, self._architecture
        )
        if compilation_settings.routing.use_legacy_greedy_method:
            routed = route_circuit_legacy(
                compiled_circuit,
                self._architecture,
                initial_placement,
            )
        else:
            routed = route_circuit(
                compiled_circuit,
                self._architecture,
                initial_placement,
                compilation_settings.routing,
            )

        routed.is_compiled = True
        routed.validate()
        return routed

    def compile_and_route_circuit(
        self,
        circuit: Circuit,
        compilation_settings: CompilationSettings = _DEFAULT_COMPILATION_SETTINGS,
        *,
        use_legacy_routing: bool = False,
    ) -> MultiZoneCircuit:
        """
        Compile a pytket Circuit and route it to the backend architecture

        Returns a MultiZoneCircuit that conforms to the backend architecture

        """
        if not circuit.is_simple:
            raise ValueError(f"{type(self).__name__} only supports simple circuits")
        compiled = super().get_compiled_circuit(
            circuit, compilation_settings.pytket_optimisation_level
        )
        # compilation renames qbit register to "fcNode" so rename back to "q"
        qubit_map = {
            cast("UnitID", qubit): cast("UnitID", Qubit(qubit.index[0]))
            for qubit in compiled.qubits
        }
        compiled.rename_units(qubit_map)

        return self.route_compiled(compiled, compilation_settings)

    def compile_manually_routed_multi_zone_circuit(
        self,
        circuit: MultiZoneCircuit,
        optimisation_level: int = 2,
    ) -> MultiZoneCircuit:
        """Compile a MultiZoneCircuit to run on an AQT multi-zone architecture

        Compiles the underlying PyTKET Circuit first according to the chosen
        optimisation level. The barriers within the Circuit mark move points
        which should not be optimised through.

        Afterward, the precomputed "PSWAP" and "SHUTTLE" operations are added
        at the appropriate barrier points.
        """

        circuit.validate()
        new_initial_zone_to_qubits = deepcopy(circuit.initial_zone_to_qubits)
        new_circuit = MultiZoneCircuit(
            circuit.architecture,
            new_initial_zone_to_qubits,
            circuit.pytket_circuit.n_qubits,
            circuit.pytket_circuit.n_bits,
        )
        compiled_circuit = super().get_compiled_circuit(
            circuit.pytket_circuit, optimisation_level
        )

        new_circuit.zone_to_qubits = deepcopy(circuit.zone_to_qubits)
        new_circuit.multi_zone_operations = deepcopy(circuit.multi_zone_operations)

        current_multiop_index_per_qubit: dict[int, int] = dict.fromkeys(
            new_circuit.multi_zone_operations, 0
        )
        for cmd in compiled_circuit:
            op = cmd.op
            if op.type == OpType.Barrier:
                if len(cmd.args) == len(circuit.all_qubit_list):
                    continue
                qubit = cmd.args[0].index[0]
                current_multiop_index = current_multiop_index_per_qubit[qubit]
                mult_op_bundles = new_circuit.multi_zone_operations[qubit]
                multi_ops = mult_op_bundles[current_multiop_index]
                for multi_op in multi_ops:
                    multi_op.append_to_circuit(new_circuit)
                new_circuit.add_move_barrier()
                current_multiop_index_per_qubit[qubit] = current_multiop_index + 1
            else:
                qubits = [q.index[0] for q in cmd.args]
                new_circuit.add_gate(cmd.op.type, qubits, op.params)

        new_circuit.is_compiled = True
        return new_circuit


def get_aqt_json_syntax_for_compiled_circuit(
    circuit: MultiZoneCircuit | Circuit,
) -> list[list]:
    """Get python List object containing circuit instructions in AQT JSON Syntax"""
    aqt_syntax_operation_list: list[list[Any]] = []
    if isinstance(circuit, MultiZoneCircuit):
        if not circuit.is_compiled:
            raise Exception(
                "AQT json syntax can only be generated from a compiled circuit"
            )
        aqt_syntax_operation_list = _translate_aqt(circuit.pytket_circuit)[0]
    elif isinstance(circuit, Circuit):
        first_op = circuit.get_commands()[0].op
        optype = first_op.type
        op_string = f"{first_op}"
        if optype != OpType.CustomGate or "INIT" not in op_string:
            raise Exception(
                "Missing INIT in circuit, AQT json syntax can"
                " only be generated from a compiled circuit"
            )
        aqt_syntax_operation_list = _translate_aqt(circuit)[0]

    return aqt_syntax_operation_list


def _get_initial_zone_to_qubit_data(
    circ: Circuit,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """
    From the initialization operations at the beginning of a circuit
    routed to a MultiZoneArchitecture, determine the initial mapping of
    qubits to a (zone, position) tuple and the initial mapping of zones
    to an (occupancy, offset) tuple.

    These mappings are used in the translations
    from Circuit commands to AQT API commands
    """
    qubit_to_zone_position: dict[int, tuple[int, int]] = {}
    zone_to_occupancy_offset: dict[int, tuple[int, int]] = {}
    for cmd in circ.get_commands():
        op = cmd.op
        optype = op.type
        op_string = f"{op}"
        if optype == OpType.CustomGate and "INIT" in op_string:
            target_zone = int(op.params[0])
            qubits = [qubit.index[0] for qubit in cmd.args]
            zone_to_occupancy_offset[target_zone] = (len(qubits), 0)
            for position, qubit in enumerate(qubits):
                qubit_to_zone_position[qubit] = (target_zone, position)
        else:
            # INITs should always be the very first commands
            break
    return qubit_to_zone_position, zone_to_occupancy_offset


def _translate_aqt(circ: Circuit) -> tuple[list[list], str]:  # noqa: PLR0912, PLR0915
    """Convert a circuit in the AQT gate set to AQT list representation,
    along with a JSON string describing the measure result permutations."""
    gates: list = list()  # noqa: C408
    measures: list = list()  # noqa: C408
    qubit_to_zone_position, zone_to_occupancy_offset = _get_initial_zone_to_qubit_data(
        circ
    )

    def zop(qubit_: int) -> list[int]:
        (zone_, position_) = qubit_to_zone_position[qubit_]
        (occupancy_, offset_) = zone_to_occupancy_offset[zone_]
        return [zone_, occupancy_, position_ - offset_]

    def swap_position(qubit_1_: int, qubit_2_: int) -> None:
        (zone_1, position_1) = qubit_to_zone_position[qubit_1_]
        (zone_2, position_2) = qubit_to_zone_position[qubit_2_]
        if zone_1 != zone_2:
            raise Exception
        qubit_to_zone_position[qubit_1_] = (zone_2, position_2)
        qubit_to_zone_position[qubit_2_] = (zone_1, position_1)

    for cmd in circ.get_commands():
        op = cmd.op
        optype = op.type
        op_string = f"{op}"
        # https://www.aqt.eu/aqt-gate-definitions/
        if optype == OpType.Rx:
            gates.append(["X", op.params[0], [zop(q.index[0]) for q in cmd.args]])
        elif optype == OpType.Ry:
            gates.append(["Y", op.params[0], [zop(q.index[0]) for q in cmd.args]])
        elif optype == OpType.Rz:
            gates.append(["Z", op.params[0], [zop(q.index[0]) for q in cmd.args]])
        elif optype == OpType.XXPhase:
            gates.append(["MS", op.params[0], [zop(q.index[0]) for q in cmd.args]])
        elif optype == OpType.CustomGate:
            if "MOVE" in op_string:
                pass
            elif "INIT" in op_string:
                target_zone = int(op.params[0])
                gates.append(["INIT", [target_zone, len(cmd.args)]])
            elif "PSWAP" in op_string:
                qubit_1 = cmd.args[0].index[0]
                qubit_2 = cmd.args[1].index[0]
                gates.append(["PSWAP", [zop(qubit_1), zop(qubit_2)]])
                swap_position(qubit_1, qubit_2)
            elif "SHUTTLE" in op_string:
                qubits = [q.index[0] for q in cmd.args]
                n_shuttle = len(qubits)
                qubit_zones_positions = [
                    qubit_to_zone_position[qubit] for qubit in qubits
                ]
                if not all(
                    i[0] == j[0] and i[1] == j[1] - 1
                    for i, j in itertools.pairwise(qubit_zones_positions)
                ):
                    raise Exception(
                        "Multi-Shuttle only possible for adjacent qubits in same zone"
                    )
                source_zone = qubit_zones_positions[0][0]
                (source_occupancy, source_offset) = zone_to_occupancy_offset[
                    source_zone
                ]
                source_zone_spec = int(op.params[0])
                if source_zone != source_zone_spec:
                    raise Exception("Faulty source zone spec")
                target_zone = int(op.params[1])
                (target_occupancy, target_offset) = zone_to_occupancy_offset[
                    target_zone
                ]
                source_edge_encoding = round(op.params[2])
                target_edge_encoding = round(op.params[3])
                if source_edge_encoding == 0:
                    zone_to_occupancy_offset[source_zone] = (
                        source_occupancy - n_shuttle,
                        source_offset + n_shuttle,
                    )
                elif source_edge_encoding == 1:
                    zone_to_occupancy_offset[source_zone] = (
                        source_occupancy - n_shuttle,
                        source_offset,
                    )
                else:
                    raise Exception("Error in Shuttle command source port")
                if target_edge_encoding == 0:
                    zone_to_occupancy_offset[target_zone] = (
                        target_occupancy + n_shuttle,
                        target_offset - n_shuttle,
                    )
                    for i, qubit in enumerate(qubits):
                        qubit_to_zone_position[qubit] = (
                            target_zone,
                            target_offset - n_shuttle + i,
                        )
                elif target_edge_encoding == 1:
                    zone_to_occupancy_offset[target_zone] = (
                        target_occupancy + n_shuttle,
                        target_offset,
                    )
                    for i, qubit in enumerate(qubits):
                        qubit_to_zone_position[qubit] = (
                            target_zone,
                            target_occupancy + target_offset + i,
                        )
                else:
                    raise Exception("Error in Shuttle command target port")

                zops_source = [
                    source_zone,
                    source_occupancy,
                    [
                        source_position - source_offset
                        for _, source_position in qubit_zones_positions
                    ],
                ]
                zops_target_l = [zop(qubit) for qubit in qubits]
                if not all(
                    i[0] == j[0] and i[1] == j[1] and i[2] == j[2] - 1
                    for i, j in itertools.pairwise(zops_target_l)
                ):
                    raise Exception("Error in target definition")
                zops_target = [
                    zops_target_l[0][0],
                    zops_target_l[0][1],
                    [zops_target_l[i][2] for i, _ in enumerate(qubits)],
                ]
                gates.append(["SHUTTLE", n_shuttle, [zops_source, zops_target]])

        elif optype == OpType.Measure:
            # predicate has already checked format is correct, so
            # errors are not handled here
            qb_id = cmd.qubits[0].index[0]
            bit_id = cmd.bits[0].index[0]
            while len(measures) <= bit_id:
                measures.append(None)
            measures[bit_id] = qb_id
        else:
            assert optype in {OpType.noop, OpType.Barrier}
    if None in measures:
        raise IndexError("Bit index not written to by a measurement.")
    return gates, json.dumps(measures)


def _aqt_rebase() -> BasePass:
    return AutoRebase({OpType.XXPhase, OpType.Rx, OpType.Ry})
