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
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.api_client import models
from qiskit_aqt_provider.api_client.models_generated import (
    JobResponseRRFinished,
    JobUser,
    ResultItem,
    RRFinished,
)
from qiskit_aqt_provider.aqt_job import AQTJob
from qiskit_aqt_provider.aqt_options import AQTOptions
from qiskit_aqt_provider.aqt_resource import AQTResource, OfflineSimulatorResource
from qiskit_aqt_provider.circuit_to_aqt import aqt_to_qiskit_circuit

from .config import AQTAccessToken

if TYPE_CHECKING:
    import httpx
    from qiskit_aqt_provider.aqt_provider import OfflineSimulator

    from .aqt_job_data import PytketAqtJob


@dataclass
class AqtDevice:
    workspace_id: str
    resource_id: str
    description: str
    resource_type: str

    @classmethod
    def from_aqt_resource(cls, aqt_resource: AQTResource) -> AqtDevice:
        return AqtDevice(
            workspace_id=aqt_resource.resource_id.workspace_id,
            resource_id=aqt_resource.resource_id.resource_id,
            resource_type=aqt_resource.resource_id.resource_type,
            description=aqt_resource.resource_id.resource_name,
        )


T = TypeVar("T")


def unwrap(obj: T | None) -> T:
    if obj is None:
        raise ValueError("Value cannot be None")
    return obj


class AqtApi(Protocol):
    def get_devices(self) -> list[AqtDevice]: ...

    def print_device_table(self) -> None: ...

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str: ...

    def get_job_result(self, job_id: str) -> models.JobResponse: ...

    def cancel_job(self, job_id: str) -> None: ...


class AqtRemoteApi(AqtApi):
    """Class implementing AQT's remote API"""

    def __init__(self, base_url: str, access_token: str | None):
        self._base_url = base_url
        self._access_token = AQTAccessToken.resolve(access_token)

    @property
    def _http_client(self) -> httpx.Client:
        """HTTP client for communicating with the AQT cloud service."""
        return models.http_client(base_url=self._base_url, token=self._access_token)

    def get_devices(self) -> list[AqtDevice]:
        aqt_provider = AQTProvider(access_token=self._access_token)
        backend_table = aqt_provider.backends()
        return [
            AqtDevice.from_aqt_resource(aqt_resource)
            for aqt_resource in backend_table.backends
        ]

    def print_device_table(self) -> None:
        aqt_provider = AQTProvider(access_token=self._access_token)
        backend_table = aqt_provider.backends()
        print(backend_table)

    def post_aqt_job(self, job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        aqt_job = _aqt_job_from_pytket_aqt_job(job)
        resp = self._http_client.post(
            f"/submit/{aqt_device.workspace_id}/{aqt_device.resource_id}",
            json=aqt_job.model_dump(),
        )
        resp.raise_for_status()
        return str(models.Response.model_validate(resp.json()).job.job_id)

    def get_job_result(self, job_id: str) -> models.JobResponse:
        resp = self._http_client.get(f"/result/{job_id}")
        resp.raise_for_status()
        return models.Response.model_validate(resp.json())

    def cancel_job(self, job_id: str) -> None:
        resp = self._http_client.delete(f"/jobs/{job_id}")
        resp.raise_for_status()


class AqtOfflineApi(AqtApi):
    """Class implementing AQT's offline Simulator API"""

    def __init__(self, simulator: OfflineSimulator):
        self._aqt_provider = AQTProvider(access_token="offline")
        self._offline_sim = OfflineSimulatorResource(
            self._aqt_provider,
            resource_id=models.Resource(
                workspace_id="default",
                resource_id=simulator.id,
                resource_name=simulator.name,
                resource_type="offline_simulator",
                available_qubits=20,
            ),
            with_noise_model=simulator.noisy,
        )

    def get_devices(self) -> list[AqtDevice]:
        backend_table = self._aqt_provider.backends()
        return [
            AqtDevice.from_aqt_resource(aqt_resource)
            for aqt_resource in backend_table.backends
        ]

    def print_device_table(self) -> None:
        backend_table = self._aqt_provider.backends()
        print(backend_table)

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        circuits = [
            aqt_to_qiskit_circuit(
                unwrap(circuit_spec.aqt_circuit), circuit_spec.circuit.n_qubits
            )
            for circuit_spec in aqt_job.circuits_data
        ]
        options = AQTOptions()
        # Offline API only allows for a fixed n_shots for all circuits
        options.shots = aqt_job.circuits_data[0].n_shots
        job = AQTJob(self._offline_sim, circuits, options)
        return str(self._offline_sim.submit(job))

    def get_job_result(self, job_id: str) -> models.JobResponse:
        return self._offline_sim.result(uuid.UUID(job_id))

    def cancel_job(self, job_id: str) -> None:
        pass


AQT_MOCK_DEVICES: Final = [
    AqtDevice(
        workspace_id="mock_wid",
        resource_id="mock_rid",
        description="Mock Device",
        resource_type="mock",
    ),
]


class AqtMockApi(AqtApi):
    """Mock API for debugging purposes"""

    def __init__(self) -> None:
        self._jobs: dict[str, PytketAqtJob] = dict()  # noqa: C408

    def get_devices(self) -> list[AqtDevice]:
        return AQT_MOCK_DEVICES

    def print_device_table(self) -> None:
        print("Mock device table")

    def post_aqt_job(self, aqt_job: PytketAqtJob, aqt_device: AqtDevice) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = aqt_job
        return job_id

    def get_job_result(self, job_id: str) -> models.JobResponse:
        job = self._jobs[job_id]
        results: dict[str, list[list[ResultItem]]] = dict()  # noqa: C408
        for i, circ_spec in enumerate(job.circuits_data):
            circ_measure_permutations = json.loads(unwrap(circ_spec.measures))
            results[str(i)] = [
                [ResultItem(0) for _ in circ_measure_permutations]
                for _ in range(circ_spec.n_shots)
            ]
        return JobResponseRRFinished(
            job=JobUser(
                job_id=uuid.UUID(job_id),
                job_type="quantum_circuit",
                label="mock-user",
                resource_id=AQT_MOCK_DEVICES[0].resource_id,
                workspace_id=AQT_MOCK_DEVICES[0].workspace_id,
            ),
            response=RRFinished(status="finished", result=results),
        )

    def cancel_job(self, job_id: str) -> None:
        self._jobs.pop(job_id)


def _aqt_job_from_pytket_aqt_job(
    job: PytketAqtJob,
) -> models.SubmitJobRequest:
    """Create AQT SubmitJobRequest from a list of circuits
    and corresponding numbers of shots"""
    return models.SubmitJobRequest(
        job_type="quantum_circuit",
        label="pytket",
        payload=models.QuantumCircuits(
            circuits=[
                models.QuantumCircuit(
                    repetitions=spec.n_shots,
                    quantum_circuit=unwrap(spec.aqt_circuit),
                    number_of_qubits=spec.circuit.n_qubits,
                )
                for spec in job.circuits_data
            ]
        ),
    )
