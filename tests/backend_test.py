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
import os
from collections import Counter

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from hypothesis import given, strategies
from pydantic_core import ValidationError
from pytket.backends import StatusEnum
from pytket.circuit import Circuit, Qubit

from pytket.extensions.aqt import AQTBackend
from pytket.extensions.aqt.backends.aqt import AqtAccessError

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of AQT access token)"


@pytest.fixture(scope="module")
def remote_backend() -> AQTBackend:
    # Use config.set_aqt_config to configure the access token beforehand
    return AQTBackend(
        # These aqt_workspace_id may need to change depending on the access_token used
        aqt_workspace_id="tket-integration",
        aqt_resource_id="simulator_noise",
    )


@pytest.mark.parametrize(
    "backend",
    [
        AQTBackend("default", "offline_simulator_no_noise"),
        AQTBackend("default", "offline_simulator_noise"),
    ],
)
def test_aqt_offline_simulator_backends(backend: AQTBackend) -> None:
    backend = AQTBackend("default", "offline_simulator_no_noise")
    c = Circuit(4, 4)
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.add_barrier([0, 1])
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    n_shots = 10
    res = backend.run_circuit(c, n_shots=n_shots, seed=1, timeout=30)
    shots = res.get_shots()
    counts = res.get_counts()
    assert len(shots) == n_shots
    assert sum(counts.values()) == n_shots


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_remote_sim(remote_backend: AQTBackend) -> None:
    # Run a circuit on the noisy simulator.
    b = remote_backend
    c = Circuit(4, 4)
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.add_barrier([0, 1])
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    res = b.run_circuit(c, n_shots=n_shots, seed=1, timeout=30)
    shots = res.get_shots()
    counts = res.get_counts()
    assert len(shots) == n_shots
    assert sum(counts.values()) == n_shots


def test_bell() -> None:
    # On the noiseless simulator, we should always get Bell states here.
    b = AQTBackend("default", "offline_simulator_no_noise")
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
    assert all(q[0] == q[1] for q in counts)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_resource_not_found_for_access_token() -> None:
    token = "invalid"
    with pytest.raises(AqtAccessError):
        AQTBackend(
            aqt_workspace_id="tket-integration",
            aqt_resource_id="simulator_noise",
            access_token=token,
        )


def test_invalid_request() -> None:
    b = AQTBackend("default", "offline_simulator_no_noise")
    c = Circuit(2, 2).H(0).CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    with pytest.raises(ValidationError) as excinfo:
        b.process_circuits([c], 1000000)
        assert "1000000" in str(excinfo.value)


@pytest.mark.parametrize(
    "b",
    [
        AQTBackend("default", "offline_simulator_no_noise"),
        AQTBackend("default", "offline_simulator_noise"),
    ],
)
def test_handles_offline(b: AQTBackend) -> None:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 5
    res = b.run_circuit(c, n_shots=n_shots, timeout=30)
    shots = res.get_shots()
    assert len(shots) == n_shots
    counts = res.get_counts()
    assert sum(counts.values()) == n_shots
    handles = b.process_circuits([c, c], n_shots=n_shots)
    assert len(handles) == 2
    for handle in handles:
        assert b.circuit_status(handle).status is StatusEnum.COMPLETED


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_handles_remote(remote_backend: AQTBackend) -> None:
    b = remote_backend
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 5
    res = b.run_circuit(c, n_shots=n_shots, timeout=30)
    shots = res.get_shots()
    assert len(shots) == n_shots
    counts = res.get_counts()
    assert sum(counts.values()) == n_shots
    handles = b.process_circuits([c, c], n_shots=n_shots)
    assert len(handles) == 2
    for handle in handles:
        assert b.circuit_status(handle).status in [
            StatusEnum.QUEUED,
            StatusEnum.RUNNING,
            StatusEnum.COMPLETED,
        ]


def test_machine_debug() -> None:
    b = AQTBackend(machine_debug=True)
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
    assert counts == {(0, 0): n_shots}


def test_warning_for_uncompiled_circuit_with_multiple_registers(
    caplog: LogCaptureFixture,
) -> None:
    b = AQTBackend(machine_debug=True)
    c = Circuit(1, 2)
    test_reg = c.add_q_register("test", 1)
    c.Rz(0.5, Qubit(0))
    c.XXPhase(0.5, Qubit(0), test_reg[0])
    c.measure_all()
    n_shots = 10
    counts = b.run_circuit(c, n_shots=n_shots, timeout=30).get_counts()
    assert counts == {(0, 0): n_shots}
    assert "WARNING" in caplog.text


@pytest.mark.parametrize(
    "b",
    [
        AQTBackend(machine_debug=True),
        AQTBackend("default", "offline_simulator_no_noise"),
        AQTBackend("default", "offline_simulator_noise"),
    ],
)
def test_default_pass(b: AQTBackend) -> None:
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c.measure_all()
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)


def test_postprocess() -> None:
    b = AQTBackend("default", "offline_simulator_no_noise")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=100, postprocess=True)
    r = b.get_result(h)
    shots = r.get_shots()
    assert all(shot[0] == shot[1] for shot in shots)


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),
    n_bits=strategies.integers(min_value=1, max_value=10),
)
def test_shots_bits_edgecases(n_shots: int, n_bits: int) -> None:
    aqt_backend = AQTBackend(machine_debug=True)
    c = Circuit(n_bits, n_bits)
    c.measure_all()

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = aqt_backend.process_circuit(c, n_shots)
    res = aqt_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = aqt_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
# This should work without a valid access token, but it will make calls to remote API
def test_retrieve_available_devices() -> None:
    backend_infos = AQTBackend.available_devices()
    assert backend_infos is not None
