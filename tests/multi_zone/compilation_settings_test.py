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
from pytket.circuit import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
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
    grid12,
)


@pytest.fixture
def ghz_circuit() -> Circuit:
    circuit = Circuit(16)
    circuit.H(0)
    for i in range(circuit.n_qubits - 1):
        circuit.CX(i, i + 1)
    circuit.measure_all()
    return circuit


manual_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.manual,
    manual_placement={
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7],
        2: [8, 9, 10, 11, 12],
        3: [13, 14, 15],
    },
)

graph_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.graph_partition, zone_free_space=2
)

ordered_placement = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.qubit_order, zone_free_space=2
)

if MT_KAHYPAR_INSTALLED:
    from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.graph_partition_gate_selection import (
        PartitionGateSelector,
    )

    graph_routing = RoutingConfig(gate_selector=PartitionGateSelector())
else:
    graph_routing = RoutingConfig()

greedy_routing = RoutingConfig()

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)


@pytest.mark.parametrize(
    "opt_level, initial_pl_settings, routing_settings",
    [
        pytest.param(0, manual_placement, greedy_routing),
        pytest.param(0, manual_placement, graph_routing, marks=graph_skipif),
        pytest.param(0, ordered_placement, greedy_routing),
        pytest.param(0, ordered_placement, graph_routing, marks=graph_skipif),
        pytest.param(0, graph_placement, greedy_routing, marks=graph_skipif),
        pytest.param(0, graph_placement, graph_routing, marks=graph_skipif),
    ],
)
def test_compilation_settings_linearch(
    opt_level: int,
    initial_pl_settings: InitialPlacementSettings,
    routing_settings: RoutingConfig,
    ghz_circuit: Circuit,
) -> None:
    backend = AQTMultiZoneBackend(
        architecture=four_zones_in_a_line, access_token="invalid"
    )
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=opt_level,
        initial_placement=initial_pl_settings,
        routing=routing_settings,
    )
    compiled = backend.compile_and_route_circuit(ghz_circuit, compilation_settings)
    print("Shuttles: ", compiled.get_n_shuttles())


manual_placement_grid = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.manual,
    manual_placement={
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [11, 12, 13],
        3: [8, 9, 10],
        4: [14, 15],
        5: [6, 7],
    },
)


@pytest.mark.parametrize(
    "opt_level, initial_pl_settings, routing_settings",
    [
        pytest.param(0, manual_placement_grid, greedy_routing),
        pytest.param(0, manual_placement_grid, graph_routing, marks=graph_skipif),
        pytest.param(0, ordered_placement, greedy_routing),
        pytest.param(0, ordered_placement, graph_routing, marks=graph_skipif),
        pytest.param(0, graph_placement, greedy_routing, marks=graph_skipif),
        pytest.param(0, graph_placement, graph_routing, marks=graph_skipif),
    ],
)
def test_compilation_settings_gridarch(
    opt_level: int,
    initial_pl_settings: InitialPlacementSettings,
    routing_settings: RoutingConfig,
    ghz_circuit: Circuit,
) -> None:
    backend = AQTMultiZoneBackend(architecture=grid12, access_token="invalid")
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=opt_level,
        initial_placement=initial_pl_settings,
        routing=routing_settings,
    )
    compiled = backend.compile_and_route_circuit(ghz_circuit, compilation_settings)
    print("Shuttles: ", compiled.get_n_shuttles())
