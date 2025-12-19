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
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from pytket.circuit import Circuit, CustomGateDef, OpType, UnitID
from sympy import Expr, symbols

from ..circuit_routing.routing_ops import PSwap, RoutingBarrier, RoutingOp
from ..circuit_routing.routing_ops import Shuttle as ShuttleOp
from ..trap_architecture.architecture import MultiZoneArchitectureSpec, PortId
from ..trap_architecture.macro_architecture_graph import MultiZoneArch
from .helpers import get_qubit_to_zone

ParamType: TypeAlias = Expr | float


class QubitPlacementError(Exception):
    pass


class MoveError(Exception):
    pass


class AcrossZoneOperationError(Exception):
    pass


class ValidationError(Exception):
    pass


class VirtualZonePosition(Enum):
    VirtualLeft = 0
    VirtualRight = 1


sz, tz, se, te = symbols("source_zone target_zone source_edge target_edge")

move_def_circ = Circuit(1)
move_def_circ.add_barrier([0])
move_gate = CustomGateDef("MOVE", move_def_circ, [tz])
"""Custom `MOVE` Gate

Added to the circuit during manual routing to indicate that the qubit
it acts on needs to be moved to the zone specified by the parameter.
The compiler should reduce these gates to SHUTTLES and PSWAPS before
submission to aqt.
"""

swap_def_circ = Circuit(2)
swap_gate = CustomGateDef("PSWAP", swap_def_circ, [sz])
"""Custom `PSWAP` Gate

Added to the circuit during routing to indicate that a physical swap
of the two qubits it acts on should take place.

For a PSWAP to be valid, the qubits it acts
on must be located next to each other in the same zone.
"""


@dataclass
class SwapWithinZone:
    """This class holds all information for defining a PSWAP"""

    qubit_0: int
    qubit_1: int
    zone: int

    def __str__(self) -> str:
        return f"{self.qubit_0}: {self.qubit_1}"

    def append_to_circuit(self, circuit: "MultiZoneCircuit") -> None:
        circuit.pytket_circuit.add_custom_gate(
            swap_gate, [self.zone], [self.qubit_0, self.qubit_1]
        )


@dataclass
class Shuttle:
    """This class holds all information for defining a SHUTTLE operation"""

    qubits: list[int]
    source_zone: int
    target_zone: int

    source_port: int
    target_port: int

    def __str__(self) -> str:
        return f"{self.qubits}: {self.target_zone}"

    def append_to_circuit(self, circuit: "MultiZoneCircuit") -> None:
        shuttle_def_circ = Circuit(len(self.qubits))
        shuttle_gate = CustomGateDef("SHUTTLE", shuttle_def_circ, [sz, tz, se, te])
        circuit.pytket_circuit.add_custom_gate(
            shuttle_gate,
            [self.source_zone, self.target_zone, self.source_port, self.target_port],
            self.qubits,
        )


@dataclass
class Init:
    qubit: int
    zone: int


MZAOperation = SwapWithinZone | Shuttle


def _swap_left_to_right_through_list(
    qubit: int, qubit_list: list[int], zone: int
) -> list[MZAOperation]:
    """Generate a list of swap operations moving an ion from left to right
    through a zone"""
    return [SwapWithinZone(qubit, swap_qubit, zone) for swap_qubit in qubit_list]


def _swap_right_to_left_through_list(
    qubit: int, qubit_list: list[int], zone: int
) -> list[MZAOperation]:
    """Generate a list of swap operations moving an ion from right to left
    through a zone"""
    return [
        SwapWithinZone(swap_qubit, qubit, zone) for swap_qubit in reversed(qubit_list)
    ]


def _move_from_zone_position_to_connected_zone_edge(  # noqa: PLR0913
    qubit: int,
    zone_qubit_list: list[int],
    position_in_zone: int | VirtualZonePosition,
    move_source_edge_port: PortId,
    move_target_edge_port: PortId,
    source_zone: int,
    target_zone: int,
) -> list[MZAOperation]:
    """Generate a list of swap and shuttle operations moving an ion from a
    given position within a zone to the edge of a target zone"""
    move_operations = []
    match (move_source_edge_port, position_in_zone):
        case (PortId.p1, VirtualZonePosition.VirtualLeft):
            move_operations.extend(
                _swap_left_to_right_through_list(qubit, zone_qubit_list, source_zone)
            )
        case (PortId.p0, VirtualZonePosition.VirtualRight):
            move_operations.extend(
                _swap_right_to_left_through_list(qubit, zone_qubit_list, source_zone)
            )
        case (PortId.p1, VirtualZonePosition.VirtualRight):
            pass
        case (PortId.p0, VirtualZonePosition.VirtualLeft):
            pass
        case (PortId.p1, position) if isinstance(position, int):
            move_operations.extend(
                _swap_left_to_right_through_list(
                    qubit, zone_qubit_list[position + 1 :], source_zone
                )
            )
        case (PortId.p0, position) if isinstance(position, int):
            move_operations.extend(
                _swap_right_to_left_through_list(
                    qubit, zone_qubit_list[:position], source_zone
                )
            )
    move_operations.append(
        Shuttle(
            [qubit],
            source_zone,
            target_zone,
            move_source_edge_port.value,
            move_target_edge_port.value,
        )
    )
    return move_operations


class MultiZoneCircuit:
    """Circuit for AQT Multi-Zone architectures

    Adds operations for initialisation of ions within zones and
    movement of ions between zones.

    Also validates correctness of circuit with respect to the
    architecture constraints

    """

    architecture: MultiZoneArchitectureSpec
    macro_arch: MultiZoneArch
    qubit_to_zones: list[list[int]]
    zone_to_qubits: list[list[int]]
    initial_zone_to_qubits: list[list[int]]
    multi_zone_operations: dict[int, list[list[MZAOperation]]]
    pytket_circuit: Circuit
    _is_compiled: bool = False

    def __init__(
        self,
        multi_zone_arch: MultiZoneArchitectureSpec,
        initial_zone_to_qubits: list[list[int]] | dict[int, list[int]],
        *args: int,
        **kwargs: str,
    ):
        self.architecture = multi_zone_arch
        self.macro_arch = MultiZoneArch(multi_zone_arch)
        self.pytket_circuit = Circuit(*args, **kwargs)
        if isinstance(initial_zone_to_qubits, list):
            self.initial_zone_to_qubits = deepcopy(initial_zone_to_qubits)
        elif isinstance(initial_zone_to_qubits, dict):
            self.initial_zone_to_qubits = [[] for _ in range(self.architecture.n_zones)]
            for zone, qubits in initial_zone_to_qubits.items():
                self.initial_zone_to_qubits[zone] = qubits
        self.zone_to_qubits = deepcopy(self.initial_zone_to_qubits)
        zone_free_space = [
            self.architecture.get_zone_max_ions_gates(zone)
            - len(self.zone_to_qubits[zone])
            for zone in range(self.architecture.n_zones)
        ]
        violating_zones = [
            zone for zone, free_space in enumerate(zone_free_space) if free_space < 0
        ]
        if violating_zones:
            raise QubitPlacementError(
                f"The initial placement of qubits into zones {violating_zones} violates"
                f"their specified maximum ion capacity"
            )
        initial_qubit_to_zone = get_qubit_to_zone(
            self.pytket_circuit.n_qubits, self.initial_zone_to_qubits
        )
        unplaced_qubits = [
            qubit for qubit, zone in enumerate(initial_qubit_to_zone) if zone == -1
        ]
        if unplaced_qubits:
            raise QubitPlacementError(
                f"Qubits {unplaced_qubits} was not placed in initial placement"
            )
        self.qubit_to_zones = [
            [initial_qubit_to_zone[qubit]]
            for qubit in range(self.pytket_circuit.n_qubits)
        ]

        self.multi_zone_operations = {
            qubit: [] for qubit in range(multi_zone_arch.n_qubits_max)
        }
        self.all_qubit_list = list(range(len(self.pytket_circuit.qubits)))

        move_barrier_def_circ = Circuit(len(self.all_qubit_list))
        move_barrier_def_circ.add_barrier(self.all_qubit_list)
        self.move_barrier_gate = CustomGateDef(
            "MOVE_BARRIER", move_barrier_def_circ, []
        )
        """A `MOVE_BARRIER` is used during manual routing.

        It prevents compiling through custom `MOVE` operations,
        which could invalidate the manual routing
        """
        for zone, qubit_list in enumerate(self.initial_zone_to_qubits):
            init_def_circ = Circuit(len(qubit_list))
            custom_init = CustomGateDef("INIT", init_def_circ, [tz])
            """An `INIT` gate.

            A custom gate that represents the initialization of the qubits it acts on
            within the zone whose id is provided as a gate parameter.
            """
            self.pytket_circuit.add_custom_gate(custom_init, [zone], qubit_list)

        self._n_shuttles = 0
        self._n_pswaps = 0

    def __iter__(self) -> Iterator:
        return self.pytket_circuit.__iter__()

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    @is_compiled.setter
    def is_compiled(self, new_value: bool) -> None:
        self._is_compiled = new_value

    def add_move_barrier(self) -> None:
        """Add custom gate MOVE_BARRIER

        This is internally a barrier over all qubits
        It is necessary to prevent reordering of shuttling
        during compilation
        """
        self.pytket_circuit.add_custom_gate(
            self.move_barrier_gate, [], self.all_qubit_list
        )

    def move_qubit(
        self,
        qubit: int,
        new_zone: int,
        precompiled: bool = False,
        use_transport_limit: bool = False,
        path_override: list[int] | None = None,
    ) -> None:
        """Move a qubit from its current zone to new_zone

        Calculates the needs "PSWAP" and "SHUTTLE" operations to implement move.
        Adds custom gates to underlying Circuit to signify move and prevent optimisation
        through the move.
        Raises error is move is not possible

        If precompiled=False, the needed "PSWAP" and "SHUTTLE" operations are added to
        lists for each qubit and "MoveBarriers" are added to underlying pytket circuit.
        The "MoveBarriers" serve as markers to add the physical operations to the
        circuit after compilation

        If precompiled=True (should not be used for manual routing), the underlying
        circuit is assumed to already be compiled (but not yet routed). the "PSWAP"
        and "SHUTTLE" operations will be added to the circuit directly.

        :param qubit: qubit to move
        :param new_zone: zone to move it too
        :param precompiled: whether the underlying pytket circuit has already been
         compiled (but not yet routed)
        :param use_transport_limit: If False will use the maximum ion limit for gate operations for new_zone,
         if True, use maximum transport limit (assuming any overflow will be corrected before gates are performed)
         :param path_override: Path to take for move
        """
        if qubit >= self.pytket_circuit.n_qubits:
            raise QubitPlacementError("Requested move on out-of-range qubit")
        old_zone = self.qubit_to_zones[qubit][-1]
        if old_zone == new_zone:
            raise MoveError(
                f"Requested move has no effect,"
                f" qubit {qubit} is already in zone {new_zone}"
            )
        move_operations = []
        if path_override:
            shortest_path = path_override
        else:
            shortest_path = self.macro_arch.shortest_path(int(old_zone), int(new_zone))
            if not shortest_path:
                raise MoveError(
                    f"Cannot move ion to zone {new_zone},"
                    f" no path found from current zone {old_zone}"
                )

        old_zone_qubits = self.zone_to_qubits[old_zone]
        position_in_zone: int | VirtualZonePosition = old_zone_qubits.index(qubit)

        new_zone_limit = (
            self.architecture.get_zone_max_ions_gates(new_zone)
            if not use_transport_limit
            else self.architecture.get_zone_max_ions_transport(new_zone)
        )

        for source_zone, target_zone in itertools.pairwise(shortest_path):
            n_qubits_in_target_zone = len(self.zone_to_qubits[target_zone])
            if target_zone == new_zone and n_qubits_in_target_zone >= new_zone_limit:
                raise MoveError(
                    f"Cannot move ion to zone {target_zone},"
                    f" maximum ion capacity already reached"
                )
            if n_qubits_in_target_zone >= self.architecture.get_zone_max_ions_transport(
                target_zone
            ):
                raise MoveError(
                    f"Move requires shuttling ion through zone {target_zone},"
                    f" but this zone is at maximum capacity"
                )

            connected_ports = self.macro_arch.get_connected_ports(
                source_zone, target_zone
            )
            source_zone_qubits = self.zone_to_qubits[source_zone]
            move_operations.extend(
                _move_from_zone_position_to_connected_zone_edge(
                    qubit,
                    source_zone_qubits,
                    position_in_zone,
                    connected_ports[0],
                    connected_ports[1],
                    source_zone,
                    target_zone,
                )
            )
            if connected_ports[1] == PortId.p1:
                position_in_zone = VirtualZonePosition.VirtualRight
            else:
                position_in_zone = VirtualZonePosition.VirtualLeft

        self.pytket_circuit.add_custom_gate(move_gate, [new_zone], [qubit])
        self.add_move_barrier()
        old_zone_qubits.remove(qubit)
        if position_in_zone is VirtualZonePosition.VirtualLeft:
            self.zone_to_qubits[new_zone].insert(0, qubit)
        else:
            self.zone_to_qubits[new_zone].append(qubit)
        self.qubit_to_zones[qubit].append(new_zone)
        self.multi_zone_operations[qubit].append(move_operations)

    def move_qubit_precompiled(
        self,
        qubit: int,
        new_zone: int,
        path: list[int],
    ) -> None:
        """Move a qubit from its current zone to new_zone along the given path

        Calculates the needs "PSWAP" and "SHUTTLE" operations to implement move.
        Adds custom gates to underlying Circuit to signify move and prevent optimisation
        through the move.
        Raises error if move is not possible

        :param qubit: qubit to move
        :param new_zone: zone to move it too
         :param path: Path to take for move
        """
        old_zone = self.qubit_to_zones[qubit][-1]
        old_zone_qubits = self.zone_to_qubits[old_zone]

        move_operations = []
        position_in_zone: int | VirtualZonePosition = old_zone_qubits.index(qubit)

        new_zone_limit = self.architecture.get_zone_max_ions_transport(new_zone)
        for source_zone, target_zone in itertools.pairwise(path):
            n_qubits_in_target_zone = len(self.zone_to_qubits[target_zone])
            if target_zone == new_zone and n_qubits_in_target_zone >= new_zone_limit:
                raise MoveError(
                    f"Cannot move ion to zone {target_zone},"
                    f" maximum ion capacity already reached"
                )
            if n_qubits_in_target_zone >= self.architecture.get_zone_max_ions_transport(
                target_zone
            ):
                raise MoveError(
                    f"Move requires shuttling ion through zone {target_zone},"
                    f" but this zone is at maximum capacity"
                )

            connected_ports = self.macro_arch.get_connected_ports(
                source_zone, target_zone
            )
            source_zone_qubits = self.zone_to_qubits[source_zone]
            move_operations.extend(
                _move_from_zone_position_to_connected_zone_edge(
                    qubit,
                    source_zone_qubits,
                    position_in_zone,
                    connected_ports[0],
                    connected_ports[1],
                    source_zone,
                    target_zone,
                )
            )
            if connected_ports[1] == PortId.p1:
                position_in_zone = VirtualZonePosition.VirtualRight
            else:
                position_in_zone = VirtualZonePosition.VirtualLeft

        old_zone_qubits.remove(qubit)
        if position_in_zone is VirtualZonePosition.VirtualLeft:
            self.zone_to_qubits[new_zone].insert(0, qubit)
        else:
            self.zone_to_qubits[new_zone].append(qubit)
        self.qubit_to_zones[qubit].append(new_zone)
        barrier_qubits = [qubit for qubit in range(self.pytket_circuit.n_qubits)]  # noqa: C416
        self.pytket_circuit.add_barrier(barrier_qubits)
        for multi_op in move_operations:
            if isinstance(multi_op, Shuttle):
                self._n_shuttles += 1
                multi_op.append_to_circuit(self)
            if isinstance(multi_op, SwapWithinZone):
                self._n_pswaps += 1
                multi_op.append_to_circuit(self)
        self.pytket_circuit.add_barrier(barrier_qubits)

    def add_routing_ops(self, ops: list[RoutingOp]) -> None:
        all_qubits = [qubit for qubit in range(self.pytket_circuit.n_qubits)]  # noqa: C416
        for op in ops:
            if isinstance(op, RoutingBarrier):
                self.pytket_circuit.add_barrier(all_qubits)
            if isinstance(op, PSwap):
                self._n_pswaps += 1
                SwapWithinZone(op.qubit0, op.qubit1, op.zone_nr).append_to_circuit(self)
            if isinstance(op, ShuttleOp):
                src_zone_qubits = self.zone_to_qubits[op.src_zone]
                trg_zone_qubits = self.zone_to_qubits[op.targ_zone]
                self._n_shuttles += 1
                Shuttle(
                    op.qubits,
                    op.src_zone,
                    op.targ_zone,
                    op.src_port.value,
                    op.targ_port.value,
                ).append_to_circuit(self)
                for qubit in op.qubits:
                    src_zone_qubits.remove(qubit)
                    trg_zone_qubits.append(qubit)

    def add_gate(
        self,
        op_type: OpType,
        args: list[UnitID] | list[int],
        params: list[ParamType] | None = None,
    ) -> "MultiZoneCircuit":
        if op_type == OpType.Barrier:
            raise ValueError(
                "The manual addition of barriers is not currently"
                " allowed within Multi Zone Circuits"
            )
        if params is None:
            self.pytket_circuit.add_gate(op_type, args)
        else:
            self.pytket_circuit.add_gate(op_type, params, args)
        return self

    def CX(self, control: int, target: int, **kwargs: Any) -> "MultiZoneCircuit":
        self.pytket_circuit.CX(control, target, **kwargs)
        return self

    def measure_all(self) -> "MultiZoneCircuit":
        self.pytket_circuit.measure_all()
        return self

    def copy(self) -> "MultiZoneCircuit":
        new_circuit = MultiZoneCircuit(self.architecture, self.initial_zone_to_qubits)
        new_circuit.pytket_circuit = self.pytket_circuit.copy()
        new_circuit.qubit_to_zones = deepcopy(self.qubit_to_zones)
        new_circuit.zone_to_qubits = deepcopy(self.zone_to_qubits)
        new_circuit.multi_zone_operations = deepcopy(self.multi_zone_operations)
        return new_circuit

    def get_n_shuttles(self) -> int:
        """
        Get the number of shuttles used to route the circuit to the architecture
        """
        return self._n_shuttles

    def get_n_pswaps(self) -> int:
        """
        Get the number of pswaps used to route the circuit to the architecture
        """
        return self._n_pswaps

    def validate(self) -> None:
        if self._is_compiled:
            self._validate_compiled()
            return

        current_multiop_index_per_qubit: dict[int, int] = dict.fromkeys(
            self.multi_zone_operations, 0
        )
        for i, cmd in enumerate(self.pytket_circuit):
            op = cmd.op
            if "MOVE_BARRIER" in f"{op}":
                pass
            elif "MOVE" in f"{op}":
                qubit = cmd.args[0].index[0]
                current_multiop_index = current_multiop_index_per_qubit[qubit]
                current_multiop_index_per_qubit[qubit] = current_multiop_index + 1
            else:
                qubits: list[int] = [q.index[0] for q in cmd.args]
                cmd_qubit_zones = [
                    self.qubit_to_zones[q][current_multiop_index_per_qubit[q]]
                    for q in qubits
                ]
                if not all(zone == cmd_qubit_zones[0] for zone in cmd_qubit_zones):
                    qubit_to_zone_message = " ".join(
                        [
                            f"q[{qz[0]}] in zone {qz[1]},"
                            for qz in zip(qubits, cmd_qubit_zones, strict=False)
                        ]
                    )
                    raise AcrossZoneOperationError(
                        f"Operation {i} = {cmd} involved qubits across multiple"
                        f"zones. {qubit_to_zone_message}"
                    )

    def _validate_compiled(self) -> None:  # noqa: PLR0912, PLR0915
        current_placement = deepcopy(self.initial_zone_to_qubits)
        current_qubit_to_zone = get_qubit_to_zone(
            self.pytket_circuit.n_qubits, current_placement
        )
        commands = self.pytket_circuit.get_commands()

        def check_transport_limit(zone_id: int, message: str) -> None:
            if len(
                current_placement[zone_id]
            ) > self.architecture.get_zone_max_ions_transport(zone_id):
                raise ValidationError(message)

        def check_gate_limit(zone_id: int, message: str) -> None:
            if len(
                current_placement[zone_id]
            ) > self.architecture.get_zone_max_ions_gates(zone_id):
                raise ValidationError(message)

        for i, cmd in enumerate(commands):
            op = cmd.op
            optype = op.type
            op_string = f"{op}"
            # check init
            if i < self.architecture.n_zones:
                if "INIT" not in op_string:
                    raise ValidationError(
                        "All zones must be initialized before any other operations"
                    )
                target_zone = int(op.params[0])
                if current_placement[target_zone] != [arg.index[0] for arg in cmd.args]:
                    raise ValidationError(
                        "INIT command does not align with expected initial placement"
                    )
                check_gate_limit(
                    target_zone,
                    f"Initial placement of zone {target_zone}"
                    f" does not respect max qubit limit for gate operation phase",
                )
            elif "MOVE_BARRIER" in op_string:
                pass
            elif "PSWAP" in op_string:
                # check swap
                zone_spec = op.params[0]
                qubit_1 = cmd.args[0].index[0]
                qubit_2 = cmd.args[1].index[0]
                zone = current_qubit_to_zone[qubit_1]
                if zone != current_qubit_to_zone[qubit_2]:
                    raise ValidationError(
                        f"Invalid PSWAP, qubits"
                        f" to swap {[qubit_1, qubit_2]} "
                        f" are not located in the same zone"
                    )
                if zone_spec != zone:
                    raise ValidationError("Faulty zone spec")
                index1 = current_placement[zone].index(qubit_1)
                index2 = current_placement[zone].index(qubit_2)
                if abs(index1 - index2) > 1:
                    raise ValidationError(
                        f"Invalid PSWAP, qubits"
                        f" to swap {[qubit_1, qubit_2]} "
                        f" are not next to each other in zone {zone}"
                    )
                # perform swap
                current_placement[zone][index1] = qubit_2
                current_placement[zone][index2] = qubit_1
            elif "SHUTTLE" in op_string:
                qubits = [arg.index[0] for arg in cmd.args]
                source_zone, target_zone, src_port, trg_port = (
                    int(param) for param in op.params
                )
                origin_zones = [current_qubit_to_zone[qubit] for qubit in qubits]
                if not all(i == j for i, j in itertools.pairwise(origin_zones)):
                    raise ValidationError(
                        "Multi-Shuttle contains qubits not in the same zone"
                    )
                origin_zone = origin_zones[0]
                if not source_zone == origin_zone:
                    raise ValidationError("Qubits not in specified source zone")
                # check zones connected
                if target_zone not in self.macro_arch.zone_connections[origin_zone]:
                    raise ValidationError(
                        f"Invalid SHUTTLE, current zone {origin_zone} "
                        f"and target zone {target_zone} of "
                        f"qubits {qubits} are not connected."
                    )
                connected_ports = self.macro_arch.get_connected_ports(
                    origin_zone, target_zone
                )
                if connected_ports[0].value != src_port:
                    raise ValidationError("Faulty source port spec")
                if connected_ports[1].value != trg_port:
                    raise ValidationError("Faulty target port spec")
                if not all(
                    current_placement[origin_zone].index(q1) + 1
                    == current_placement[origin_zone].index(q2)
                    for q1, q2 in itertools.pairwise(qubits)
                ):
                    raise ValidationError("Invalid SHUTTLE, qubits not contiguous")
                # check connection exists and perform shuttle
                if connected_ports[0] == PortId.p0:
                    if not current_placement[origin_zone].index(qubits[0]) == 0:
                        raise ValidationError(
                            f"Invalid SHUTTLE, qubits {qubits} not at necessary port 0 of zone {origin_zone} with occupancy {current_placement[origin_zone]}. "
                        )
                    for ind in range(len(qubits) - 1, -1, -1):
                        current_placement[origin_zone].pop(ind)
                else:
                    expected_position = len(current_placement[origin_zone]) - 1
                    if (
                        not current_placement[origin_zone].index(qubits[-1])
                        == expected_position
                    ):
                        raise ValidationError(
                            f"Invalid SHUTTLE, qubits{qubits} not at necessary port 1 of zone {origin_zone} with occupancy {current_placement[origin_zone]}. "
                        )
                    for _ in range(len(qubits)):
                        current_placement[origin_zone].pop()

                # ordering in the zones is always from port 0 to port 1
                # so order in the target zone is dependent on if the connected ports
                # if port values are the same the ordering must be flipped
                ordered_qubits = (
                    list(reversed(qubits))
                    if connected_ports[0] == connected_ports[1]
                    else qubits.copy()
                )

                if connected_ports[1] == PortId.p0:
                    ordered_qubits.extend(current_placement[target_zone])
                    current_placement[target_zone] = ordered_qubits
                else:
                    current_placement[target_zone].extend(ordered_qubits)

                for qubit in qubits:
                    current_qubit_to_zone[qubit] = target_zone
                check_transport_limit(
                    target_zone,
                    f"Transport into zone {target_zone} violates max allowed qubits",
                )
            elif len(cmd.args) == 2 and optype not in (OpType.Measure, OpType.Barrier):  # noqa: PLR2004
                qubit_1 = cmd.args[0].index[0]
                qubit_2 = cmd.args[1].index[0]
                current_zone = current_qubit_to_zone[qubit_1]
                if current_zone != current_qubit_to_zone[qubit_2]:
                    raise ValidationError(
                        "Invalid 2 qubit gate. Qubits located in different zones"
                    )
                memory_only = self.architecture.zones[current_zone].memory_only
                if memory_only:
                    raise ValidationError(
                        "Invalid 2 qubit gate. Qubits located in a non-gate zone"
                    )
                check_gate_limit(
                    current_zone,
                    f"Performing gate in zone {current_zone}"
                    f" while it is above limit capacity for performing gates",
                )
            else:
                if optype not in [
                    OpType.Rx,
                    OpType.Ry,
                    OpType.Rz,
                    OpType.Measure,
                    OpType.Barrier,
                ]:
                    raise ValidationError(
                        f"Invalid operation with OpType {optype} detected."
                    )
                if optype != OpType.Barrier:
                    qubit_1 = cmd.args[0].index[0]
                    current_zone = current_qubit_to_zone[qubit_1]
                    memory_only = self.architecture.zones[current_zone].memory_only
                    if memory_only:
                        raise ValidationError(
                            "Invalid 1 qubit gate. Qubit located in a non-gate zone"
                        )
