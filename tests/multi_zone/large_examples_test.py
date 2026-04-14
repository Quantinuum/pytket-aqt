import os

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
    grid12,
    grid12_mod,
    racetrack,
    racetrack_4_gatezones,
)

graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for testing graph partitioning"
)
skip_if_no_run_long_tests = pytest.mark.skipif(
    not os.getenv("PYTKET_AQT_RUN_LONG_TESTS"),
    reason="Long test is skipped if env var PYTKET_AQT_RUN_LONG_TESTS not set",
)


def quantum_advantage_circuit(Lx: int, Ly: int) -> Circuit:
    N = Lx * Ly
    Ntrot = 30
    dt = 0.2
    hx = 2
    J = 1

    def c(x: int, y: int) -> int:
        if (x - y) % N == Lx or (y - x) % N == Lx:
            return 1
        if x // Lx == y // Lx and x == y - (y % Lx) + (((y % Lx) + 1) % Lx):
            return 1
        if x // Lx == y // Lx and y == x - (x % Lx) + (((x % Lx) + 1) % Lx):
            return 1
        return 0

    coupling_list: list[tuple[int, int]] = []
    coupling_list.extend(
        [(i, j) for i in range(N) for j in range(i + 1, N) if c(i, j) == 1]
    )

    U = Circuit(N)

    for t in range(Ntrot):
        n_trotter_steps = t

        if n_trotter_steps == 0:
            for j in range(N):
                U.Rx(-2 * dt * hx / 2, j)

        for coupling in coupling_list:
            U.ZZPhase(-2 * dt * J, coupling[0], coupling[1])
        for j in range(N):
            U.Rx(-2 * dt * hx, j)

        if n_trotter_steps == Ntrot - 1:
            for j in range(N):
                U.Rx(2 * dt * hx / 2, j)
    return U


## 56 qubit quantum advantage circuit
advantage_circuit_56 = quantum_advantage_circuit(7, 8)

## 30 qubit quantum advantage circuit
advantage_circuit_30 = quantum_advantage_circuit(5, 6)

racetrack_backend = AQTMultiZoneBackend(architecture=racetrack, access_token="invalid")

racetrack_4_gatezones_backend = AQTMultiZoneBackend(
    architecture=racetrack_4_gatezones, access_token="invalid"
)

grid_backend = AQTMultiZoneBackend(architecture=grid12, access_token="invalid")

grid_mod_backend = AQTMultiZoneBackend(architecture=grid12_mod, access_token="invalid")

order_init = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.qubit_order,
    zone_free_space=2,
    max_depth=200,
)
greedy_routing = RoutingConfig()
greedy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=greedy_routing,
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


legacy_routing = RoutingConfig(use_legacy_greedy_method=True)
greedy_routing = RoutingConfig()
legacy_compilation_settings = CompilationSettings(
    pytket_optimisation_level=1,
    initial_placement=order_init,
    routing=legacy_routing,
)


adv_precomp = racetrack_backend.compile_circuit(
    advantage_circuit_56, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings",
    [
        pytest.param(legacy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(greedy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(
            graph_compilation_settings,
            marks=[graph_skipif, skip_if_no_run_long_tests],
        ),
        pytest.param(
            hypergraph_compilation_settings,
            marks=[graph_skipif, skip_if_no_run_long_tests],
        ),
    ],
)
def test_quantum_advantage_racetrack_all_gate_zone(
    compilation_settings: CompilationSettings,
) -> None:
    racetrack_backend.route_compiled(adv_precomp, compilation_settings)


adv_precomp2 = racetrack_4_gatezones_backend.compile_circuit(
    advantage_circuit_56, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings",
    [
        pytest.param(legacy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(greedy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(
            graph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
        pytest.param(
            hypergraph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_racetrack_4_gate_zone(
    compilation_settings: CompilationSettings,
) -> None:
    racetrack_4_gatezones_backend.route_compiled(
        adv_precomp2,
        compilation_settings,
    )


adv_precomp3 = grid_backend.compile_circuit(
    advantage_circuit_30, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings",
    [
        pytest.param(legacy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(greedy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(
            graph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
        pytest.param(
            hypergraph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_grid_12_gate_zone(
    compilation_settings: CompilationSettings,
) -> None:
    grid_backend.route_compiled(
        adv_precomp3,
        compilation_settings,
    )


adv_precomp4 = grid_mod_backend.compile_circuit(
    advantage_circuit_30, graph_compilation_settings
)


@pytest.mark.parametrize(
    "compilation_settings",
    [
        pytest.param(legacy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(greedy_compilation_settings, marks=skip_if_no_run_long_tests),
        pytest.param(
            graph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
        pytest.param(
            hypergraph_compilation_settings,
            marks=[skip_if_no_run_long_tests, graph_skipif],
        ),
    ],
)
def test_quantum_advantage_grid_4_gate_zone(
    compilation_settings: CompilationSettings,
) -> None:
    grid_mod_backend.route_compiled(
        adv_precomp4,
        compilation_settings,
    )
