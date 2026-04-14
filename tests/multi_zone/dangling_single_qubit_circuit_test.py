import pytest

from pytket import Circuit
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
)


def build_test_circuit(n_qubits: int) -> Circuit:
    circ = Circuit(n_qubits + 1, name="QFT")
    for i in range(n_qubits):
        circ.H(i)
        for j in range(i + 1, n_qubits):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(n_qubits // 2):
        circ.SWAP(k, n_qubits - k - 1)
    circ.H(n_qubits)
    circ.S(n_qubits)
    circ.X(n_qubits)
    return circ


test_circ = build_test_circuit(15)

line_backend = AQTMultiZoneBackend(
    architecture=four_zones_in_a_line, access_token="invalid"
)


order_init = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.qubit_order,
    zone_free_space=2,
    max_depth=200,
)

greedy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=RoutingConfig(),
)

if MT_KAHYPAR_INSTALLED:
    from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
        GraphPartitionGateSelector,
        HypergraphPartitionGateSelector,
    )

    graph_compilation_settings = CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(gate_selector=GraphPartitionGateSelector()),
    )

    hypergraph_compilation_settings = CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(gate_selector=HypergraphPartitionGateSelector()),
    )
else:
    graph_compilation_settings = greedy_compilation_settings
    hypergraph_compilation_settings = greedy_compilation_settings


legacy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=RoutingConfig(use_legacy_greedy_method=True),
)

qft_precompiled = line_backend.compile_circuit(test_circ, graph_compilation_settings)

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)


@pytest.mark.parametrize(
    "compilation_settings",
    [
        pytest.param(legacy_compilation_settings),
        pytest.param(greedy_compilation_settings),
        pytest.param(graph_compilation_settings, marks=graph_skipif),
        pytest.param(hypergraph_compilation_settings, marks=graph_skipif),
    ],
)
def test_circuit_with_dangling_single_qubit_gates(
    compilation_settings: CompilationSettings,
) -> None:
    line_backend.route_compiled(
        qft_precompiled,
        compilation_settings,
    )
