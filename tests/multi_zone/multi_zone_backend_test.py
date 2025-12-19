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

import pytest
from pytket.backends import ResultHandle
from pytket.circuit import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import (
    AQTMultiZoneBackend,
    get_aqt_json_syntax_for_compiled_circuit,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit import (
    MultiZoneCircuit,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_config import (
    RoutingConfig,
)
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
)
from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementAlg,
    InitialPlacementSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)


@pytest.fixture()
def backend() -> AQTMultiZoneBackend:
    return AQTMultiZoneBackend(
        architecture=four_zones_in_a_line, access_token="invalid"
    )


def test_not_implemented_functionality_throws(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
        2: [8, 9, 10, 11],
        3: [12, 13, 14, 15],
    }
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 16)
    with pytest.raises(NotImplementedError):
        backend.process_circuits([circuit])  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.process_circuit(circuit)  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.run_circuits([circuit])  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.run_circuit(circuit)  # type: ignore
    with pytest.raises(NotImplementedError):
        backend.circuit_status(ResultHandle())
    with pytest.raises(NotImplementedError):
        backend.get_result(ResultHandle())
    with pytest.raises(NotImplementedError):
        backend.cancel(ResultHandle())


def test_valid_circuit_compiles(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    circuit = backend.compile_manually_routed_multi_zone_circuit(circuit)


def test_circuit_compiles(backend: AQTMultiZoneBackend) -> None:
    circuit = Circuit(8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    backend.compile_and_route_circuit(circuit)


def test_invalid_circuit_does_not_compile(backend: AQTMultiZoneBackend) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    with pytest.raises(Exception):  # noqa: B017
        backend.compile_manually_routed_multi_zone_circuit(circuit)


def test_try_get_aqt_syntax_on_uncompiled_circuit_raises(
    backend: AQTMultiZoneBackend,
) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    with pytest.raises(Exception):  # noqa: B017
        get_aqt_json_syntax_for_compiled_circuit(circuit)


def test_compiled_circuit_has_correct_syntax(backend: AQTMultiZoneBackend) -> None:  # noqa: PLR0915
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.move_qubit(0, 0)
    circuit.move_qubit(3, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    circuit = backend.compile_manually_routed_multi_zone_circuit(circuit)
    aqt_operation_list = get_aqt_json_syntax_for_compiled_circuit(circuit)

    initialized_zones: list[int] = []
    number_initialized_qubits: int = 0
    for i, operation in enumerate(aqt_operation_list):
        if i < circuit.architecture.n_zones:
            assert operation[0] == "INIT"
        else:
            assert operation[0] != "INIT"
        if operation[0] == "INIT":
            initialized_zones.append(operation[1][0])
            number_initialized_qubits += operation[1][1]
        elif operation[0] in ["X", "Y", "Z"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 1
            assert _is_valid_zop(operation[2][0], initialized_zones)
        elif operation[0] in ["MS"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_same_zone(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["SHUTTLE"]:
            assert len(operation) == 3
            assert isinstance(operation[1], int)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_different_zones(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["PSWAP"]:
            assert len(operation) == 2
            assert len(operation[1]) == 2
            assert _zop_addresses_in_same_zone(operation[1][0], operation[1][1])
            assert _is_valid_zop(operation[1][0], initialized_zones)
            assert _is_valid_zop(operation[1][1], initialized_zones)
        else:
            raise Exception(f"Detected invalid operation type: {operation[0]}")
    assert initialized_zones == [zone for zone in range(circuit.architecture.n_zones)]  # noqa: C416
    assert number_initialized_qubits == 8


greedy_routing = RoutingConfig()
if MT_KAHYPAR_INSTALLED:
    from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.graph_partition_gate_selection import (
        PartitionGateSelector,
    )

    graph_routing = RoutingConfig(gate_selector=PartitionGateSelector())
else:
    graph_routing = greedy_routing

legacy_routing = RoutingConfig(use_legacy_greedy_method=True)

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)


@pytest.mark.parametrize(
    "routing_settings",
    [pytest.param(greedy_routing), pytest.param(graph_routing, marks=graph_skipif)],
)
def test_automatically_routed_circuit_has_correct_syntax(  # noqa: PLR0915
    backend: AQTMultiZoneBackend,
    routing_settings: RoutingConfig,
) -> None:
    initial_placement = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    circuit = Circuit(8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    init_pl_settings = InitialPlacementSettings(
        algorithm=InitialPlacementAlg.manual,
        manual_placement=initial_placement,
    )
    compilation_settings = CompilationSettings(
        initial_placement=init_pl_settings, routing=routing_settings
    )
    mz_circuit = backend.compile_and_route_circuit(circuit, compilation_settings)

    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()

    aqt_operation_list = get_aqt_json_syntax_for_compiled_circuit(mz_circuit)

    initialized_zones: list[int] = []
    number_initialized_qubits: int = 0
    aqt_shuttles = 0
    aqt_pswaps = 0
    for i, operation in enumerate(aqt_operation_list):
        if i < mz_circuit.architecture.n_zones:
            assert operation[0] == "INIT"
        else:
            assert operation[0] != "INIT"
        if operation[0] == "INIT":
            initialized_zones.append(operation[1][0])
            number_initialized_qubits += operation[1][1]
        elif operation[0] in ["X", "Y", "Z"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 1
            assert _is_valid_zop(operation[2][0], initialized_zones)
        elif operation[0] in ["MS"]:
            assert len(operation) == 3
            assert isinstance(operation[1], float)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_same_zone(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
        elif operation[0] in ["SHUTTLE"]:
            assert len(operation) == 3
            assert isinstance(operation[1], int)
            assert len(operation[2]) == 2
            assert _zop_addresses_in_different_zones(operation[2][0], operation[2][1])
            assert _is_valid_zop(operation[2][0], initialized_zones)
            assert _is_valid_zop(operation[2][1], initialized_zones)
            aqt_shuttles += 1
        elif operation[0] in ["PSWAP"]:
            assert len(operation) == 2
            assert len(operation[1]) == 2
            assert _zop_addresses_in_same_zone(operation[1][0], operation[1][1])
            assert _is_valid_zop(operation[1][0], initialized_zones)
            assert _is_valid_zop(operation[1][1], initialized_zones)
            aqt_pswaps += 1
        else:
            raise Exception(f"Detected invalid operation type: {operation[0]}")
    assert n_pswaps == aqt_pswaps
    assert n_shuttles == aqt_shuttles
    assert initialized_zones == list(range(mz_circuit.architecture.n_zones))
    assert number_initialized_qubits == 8


def _is_valid_zop(zop: list, zone_list: list[int]) -> bool:
    return all(
        [
            len(zop) == 3,
            zop[0] in zone_list,
            zop[1] > 0,
            zop[2] >= 0 if isinstance(zop[2], int) else all(val >= 0 for val in zop[2]),
            zop[2] < zop[1]
            if isinstance(zop[2], int)
            else all(val < zop[1] for val in zop[2]),
        ]
    )


def _zop_addresses_in_same_zone(zop1: list, zop2: list) -> bool:
    result: bool = zop1[0] == zop2[0]
    return result


def _zop_addresses_in_different_zones(zop1: list, zop2: list) -> bool:
    result: bool = zop1[0] != zop2[0]
    return result
