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
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, assert_never, cast

import numpy
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError, CircuitNotValidError
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
    SimplifyInitial,
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
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from qiskit_aqt_provider.api_client import models, models_generated
from qiskit_aqt_provider.aqt_provider import OFFLINE_SIMULATORS

from .._metadata import __extension_version__
from .aqt_api import (
    AQT_MOCK_DEVICES,
    AqtApi,
    AqtMockApi,
    AqtOfflineApi,
    AqtRemoteApi,
    unwrap,
)
from .aqt_job_data import PytketAqtJob, PytketAqtJobCircuitData

logger = logging.getLogger(__name__)

AQT_PORTAL_URL = "https://arnica.aqt.eu/api/v1"

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"

# Hard-coded for now as there is no API to retrieve these.
# All devices are fully connected.
_AQT_MAX_QUBITS = 20

_GATE_SET = {
    OpType.Rz,
    OpType.PhasedX,
    OpType.XXPhase,
    OpType.Measure,
    OpType.Barrier,
}

AQTResult = tuple[int, list[int]]  # (n_qubits, list of readouts)

AQT_OFFLINE_SIMULATORS = {sim.id: sim for sim in OFFLINE_SIMULATORS}


class AqtAccessError(Exception):
    """Raised when the provided access token does not
     allow access to the specified device.

    If no access token provided, the user will only
     have access to offline simulators
    """


class AQTBackend(Backend):
    """
    Interface to an AQT device or simulator.
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        aqt_workspace_id: str = "default",
        aqt_resource_id: str = "offline_simulator_no_noise",
        access_token: str | None = None,
        label: str = "",
        machine_debug: bool = False,
    ):
        """
        Construct a new AQT backend.

        Requires a valid API key/access token, this can either be provided as a
        parameter or set in config using :py:meth:`pytket.extensions.aqt.set_aqt_config`

        :param      aqt_workspace_id:  the aqt workspace
        :type       aqt_workspace_id:  string
        :param      aqt_resource_id:  the aqt resource id
        :type       aqt_resource_id:  string
        :param      access_token: AQT access token, default None
        :type       access_token: string
        :param      label:        label to apply to submitted jobs
        :type       label:        string
        :param      machine_debug: If true,
         use mock aqt API (for debug/testing purposes)
        :type       label: bool
        """
        super().__init__()
        self._portal_url = AQT_PORTAL_URL

        if machine_debug:
            self._aqt_api: AqtApi = AqtMockApi()
            aqt_workspace_id = AQT_MOCK_DEVICES[0].workspace_id
            aqt_resource_id = AQT_MOCK_DEVICES[0].resource_id
        elif aqt_resource_id in AQT_OFFLINE_SIMULATORS:
            self._aqt_api = AqtOfflineApi(AQT_OFFLINE_SIMULATORS[aqt_resource_id])
        else:
            self._aqt_api = AqtRemoteApi(self._portal_url, access_token)

        # cache of AQT jobs submitted to the server
        # (keys are UUIDs returned from AQT Job submission as str's)
        self._aqt_jobs: dict[str, PytketAqtJob] = dict()  # noqa: C408

        available_devices = self._aqt_api.get_devices()
        matched_devices = [
            device
            for device in available_devices
            if device.workspace_id == aqt_workspace_id
            and device.resource_id == aqt_resource_id
        ]
        if not matched_devices:
            msg = (
                f"The resolved access token does not provide access"
                f" to the resource '{aqt_resource_id}' in workspace"
                f" '{aqt_workspace_id}', use AQTBackend.print_device_table"
                f" to print a list of available resources"
            )
            raise AqtAccessError(msg)
        if len(matched_devices) > 1:
            raise ValueError(
                "More than one AQT device found for given workspace and resource ids"
            )
        self._aqt_device = matched_devices[0]
        self._label = label
        self._backend_info = fully_connected_backendinfo(
            type(self).__name__,
            aqt_resource_id,
            __extension_version__,
            _AQT_MAX_QUBITS,
            _GATE_SET,
            misc={
                "aqt_workspace_id": aqt_workspace_id,
                "aqt_resource_id": aqt_resource_id,
            },
        )
        self._qm = {
            Qubit(i): cast("Qubit", node)
            for i, node in enumerate(self._backend_info.nodes)
        }

    def rebase_pass(self) -> BasePass:
        return _aqt_rebase()

    @property
    def backend_info(self) -> BackendInfo | None:
        return self._backend_info

    @classmethod
    def print_device_table(cls, access_token: str | None = None) -> None:
        """
        Print AQT devices available for the configured access token

        The access token will be resolved by the AQTAccessToken class

        :param      access_token:  optional access token override
        :type       access_token:  string
        """
        aqt_api = AqtRemoteApi(AQT_PORTAL_URL, access_token)
        aqt_api.print_device_table()

    @classmethod
    def available_devices(
        cls, access_token: str | None = None, **kwargs: Any
    ) -> list[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.available_devices`.
        Supported kwargs: none.

        The access token will be resolved by the AQTAccessToken class

        :param      access_token:  optional access token override
        :type       access_token:  string
        """
        aqt_api = AqtRemoteApi(AQT_PORTAL_URL, access_token)
        aqt_devices = aqt_api.get_devices()
        return [
            fully_connected_backendinfo(
                cls.__name__,
                aqt_device.resource_id,
                __extension_version__,
                _AQT_MAX_QUBITS,
                _GATE_SET,
                misc={
                    "aqt_workspace_id": aqt_device.workspace_id,
                    "aqt_resource_id": aqt_device.resource_id,
                },
            )
            for aqt_device in aqt_devices
        ]

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

    def default_compilation_pass(self, optimisation_level: int = 2) -> BasePass:
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
        """
        0: job_id  (jobs can contain multiple circuits)
        1: circuit_index (index of circuit within a job)
        2: measure results permutations as json str
        2: circuit after postprocessing as json str
        """
        return str, int, str, str

    def _get_handle_data(self, handle: ResultHandle) -> tuple[str, int, str, str]:
        self._check_handle_type(handle)
        jobid, circuit_index, measures_str, ppcirc_str = (
            handle[0],
            handle[1],
            handle[2],
            handle[3],
        )
        if not isinstance(jobid, str):
            raise ValueError(f"Invalid type {type(jobid)} for job id, should be `str`")
        if not isinstance(circuit_index, int):
            raise ValueError(
                f"Invalid type {type(circuit_index)} for circuit index, should be `int`"
            )
        if not isinstance(measures_str, str):
            raise ValueError(
                f"Invalid type {type(jobid)} for measure"
                f" permutations string, should be `str`"
            )
        if not isinstance(ppcirc_str, str):
            raise ValueError(
                f"Invalid type {type(jobid)} for post-processed"
                f" circuit string, should be `str`"
            )
        return jobid, circuit_index, measures_str, ppcirc_str

    def _ensure_circuits_have_single_registers(
        self, circuits: Sequence[Circuit]
    ) -> None:
        """This will apply the FlattenRegisters and RenameQubitsPasses if more
        than one qubit and/or bit register is detected

        This can occur if circuits are submitted without compilation.
        This is usually not recommended and can lead to errors,
         so a warning is logged.
        """
        circuit_detected = False
        for circuit in circuits:
            if len(circuit.q_registers) > 1 or len(circuit.c_registers) > 1:
                FlattenRegisters().apply(circuit)
                RenameQubitsPass(self._qm).apply(circuit)
                circuit_detected = True

        if circuit_detected:
            logger.warning(
                "Detected circuits with more than one quantum and/or classical "
                "register. Did you forget to compile? "
                "Flattening registers to avoid errors"
            )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[int | None] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported kwargs:
        - `postprocess`: apply end-of-circuit simplifications and classical
          postprocessing to improve fidelity of results (bool, default False)
        - `simplify_initial`: apply the pytket ``SimplifyInitial`` pass to improve
          fidelity of results assuming all qubits initialized to zero (bool, default
          False)
        """
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(  # noqa: SLF001
            n_shots,
            len(circuits),
            optional=False,
        )
        if isinstance(self._aqt_api, AqtOfflineApi) and not isinstance(n_shots, int):
            raise ValueError(
                "The AQT offline simulators only support a fixed number of shots"
                "per circuit for a batch of circuits, please provide a single"
                "integer value for `n_shots`"
            )
        if valid_check:
            self._ensure_circuits_have_single_registers(circuits)
            self._check_all_circuits(circuits)

        job = PytketAqtJob(
            circuits_data=[
                PytketAqtJobCircuitData(circuit=circ, n_shots=n_shots_list[i])
                for i, circ in enumerate(circuits)
            ]
        )

        if kwargs.get("postprocess", False):
            _perform_circuit_postprocessing(job.circuits_data)

        if kwargs.get("simplify_initial", False):
            _perform_simplify_initial(job.circuits_data)

        _add_aqt_circ_and_measure_data(job.circuits_data)

        handles = []

        job_id = self._aqt_api.post_aqt_job(job, self._aqt_device)

        for i, circ_spec in enumerate(job.circuits_data):
            handle = ResultHandle(
                job_id, i, unwrap(circ_spec.measures), circ_spec.postprocess_json
            )
            handles.append(handle)
            circ_spec.handle = handle

        self._aqt_jobs[job_id] = job

        for handle in handles:
            self._cache[handle] = dict()  # noqa: C408
        return handles

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: dict[str, BackendResult]
    ) -> None:
        if handle in self._cache:
            self._cache[handle].update(result_dict)
        else:
            self._cache[handle] = result_dict

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        jobid, _, measures_str, ppcirc_str = self._get_handle_data(handle)
        if jobid not in self._aqt_jobs:
            raise ValueError("Could not find AQT Job for the given ResultHandle")
        job = self._aqt_jobs[jobid]

        payload = self._aqt_api.get_job_result(jobid)

        if isinstance(payload, models_generated.JobResponseRRQueued):
            return CircuitStatus(StatusEnum.QUEUED, "")

        if isinstance(payload, models_generated.JobResponseRROngoing):
            finished_count = payload.response.finished_count
            num_circuits = len(job.circuits_data)
            msg = (
                f"Circuit belongs to ongoing AQT job,"
                f" {num_circuits - finished_count} of"
                f" {num_circuits} circuits remain for this job"
            )
            return CircuitStatus(StatusEnum.RUNNING, msg)

        if isinstance(payload, models_generated.JobResponseRRFinished):
            # Entire job is complete, so update results of all circuits in the job
            for circuit_index_string, shots in payload.response.result.items():
                circuit_index = int(circuit_index_string)
                circ_handle = unwrap(job.circuits_data[circuit_index].handle)
                circ_measure_permutations = json.loads(measures_str)
                circ_shots = OutcomeArray.from_readouts(
                    numpy.array([[sample.root for sample in shot] for shot in shots])
                ).choose_indices(circ_measure_permutations)
                ppcirc_rep = json.loads(ppcirc_str)
                ppcirc = (
                    Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
                )
                self._update_cache_result(
                    circ_handle,
                    {"result": BackendResult(shots=circ_shots, ppcirc=ppcirc)},
                )
            return CircuitStatus(StatusEnum.COMPLETED, "")

        if isinstance(payload, models_generated.JobResponseRRError):
            return CircuitStatus(StatusEnum.ERROR, payload.response.message)

        if isinstance(payload, models_generated.JobResponseRRCancelled):
            return CircuitStatus(StatusEnum.CANCELLED, "")

        assert_never(payload)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            timeout = cast("float", kwargs.get("timeout"))
            wait = kwargs.get("wait", 1.0)
            # Wait for job to finish; result will then be in the cache.
            end_time = (time.time() + timeout) if (timeout is not None) else None
            while (end_time is None) or (time.time() < end_time):
                circuit_status = self.circuit_status(handle)
                if circuit_status.status is StatusEnum.COMPLETED:
                    return cast("BackendResult", self._cache[handle]["result"])
                if circuit_status.status is StatusEnum.ERROR:
                    raise RuntimeError(circuit_status.message)  # noqa: B904
                time.sleep(cast("float", wait))
            raise RuntimeError(f"Timed out: no results after {timeout} seconds.")  # noqa: B904


def _perform_circuit_postprocessing(
    circ_specs: Sequence[PytketAqtJobCircuitData],
) -> None:
    for circ_spec in circ_specs:
        c0, ppcirc = prepare_circuit(
            circ_spec.circuit, allow_classical=False, xcirc=_xcirc
        )
        circ_spec.circuit = c0
        circ_spec.postprocess_json = json.dumps(ppcirc.to_dict())


def _perform_simplify_initial(circ_specs: Sequence[PytketAqtJobCircuitData]) -> None:
    simp_init_pass = SimplifyInitial(
        allow_classical=False, create_all_qubits=True, xcirc=_xcirc
    )
    for circ_spec in circ_specs:
        simp_init_pass.apply(circ_spec.circuit)


def _add_aqt_circ_and_measure_data(
    circ_specs: Sequence[PytketAqtJobCircuitData],
) -> None:
    """Add the AQT API model representation to the circuit spec of each circuit,
    along with a JSON string describing the measure result permutations."""
    for circ_spec in circ_specs:
        circ_spec.aqt_circuit, circ_spec.measures = _pytket_to_aqt_circuit(
            circ_spec.circuit
        )


def _pytket_to_aqt_circuit(
    pytket_circuit: Circuit,
) -> tuple[models.Circuit, str]:
    """Get the AQT API model representation of a rebased pytket circuit,
    along with a JSON string describing the measure result permutations."""
    ops: list[models.OperationModel] = []
    num_measurements = 0
    measures: list[int | None] = []
    for cmd in pytket_circuit.get_commands():
        op = cmd.op
        optype = op.type
        # https://arnica.aqt.eu/api/v1/docs
        if optype == OpType.Rz:
            ops.append(
                models.Operation.rz(
                    phi=float(op.params[0]),
                    qubit=cmd.args[0].index[0],
                )
            )
        elif optype == OpType.PhasedX:
            ops.append(
                models.Operation.r(
                    theta=_restrict_to_range_zero_to_x(float(op.params[0]), 1),
                    phi=_restrict_to_range_zero_to_x(float(op.params[1]), 2),
                    qubit=cmd.args[0].index[0],
                )
            )
        elif optype == OpType.XXPhase:
            ops.append(
                models.Operation.rxx(
                    theta=float(op.params[0]),
                    qubits=[cmd.args[0].index[0], cmd.args[1].index[0]],
                )
            )
        elif optype == OpType.Measure:
            # predicate has already checked format is correct, so
            # errors are not handled here
            num_measurements += 1
            qb_id = cmd.qubits[0].index[0]
            bit_id = cmd.bits[0].index[0]
            while len(measures) <= bit_id:
                measures.append(None)
            measures[bit_id] = qb_id
        elif optype not in {OpType.noop, OpType.Barrier}:
            message = f"Gate {optype} is not in the allowed AQT gate set"
            raise ValueError(message)
    if num_measurements == 0:
        raise CircuitNotValidError("Circuit must contain at least one measurement")
    if None in measures:
        raise IndexError("Bit index not written to by a measurement.")
    ops.append(models.Operation.measure())
    aqt_circuit = models.Circuit(root=ops)
    return aqt_circuit, json.dumps(measures)


def _aqt_rebase() -> BasePass:
    return AutoRebase({OpType.XXPhase, OpType.Rz, OpType.PhasedX})


_xcirc = Circuit(1).Rx(1, 0)
_xcirc.add_phase(0.5)


def _restrict_to_range_zero_to_x(number: float, x: float) -> float:
    """Use to restrict gate parameters to values accepted by aqt

    Assumes that gate effect is periodic in this parameter with period x
    """
    return float(numpy.fmod(number, x))
