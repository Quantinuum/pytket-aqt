from pathlib import Path

from pytket import Circuit
from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.multi_zone_architecture.circuit import (
    MultiZoneCircuit,
    write_multi_zone_circuit_movie_html,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
    SiqciArchGateSelector,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.qubit_routing import (
    SingleGateZoneLineArchRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_config import (
    RoutingConfig,
)
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementAlg,
    InitialPlacementSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    siqci_arch,
)


def build_test_circuit(n_qubits: int) -> Circuit:
    nq_build = n_qubits - 1
    circ = Circuit(n_qubits, name="QFT")
    for i in range(nq_build):
        circ.H(i)
        for j in range(i + 1, nq_build):
            circ.CU1(1 / 2 ** (j - i), j, i)
    for k in range(nq_build // 2):
        circ.SWAP(k, nq_build - k - 1)
    circ.H(nq_build)
    circ.S(nq_build)
    circ.X(nq_build)
    return circ


def _routed_non_linear_visualizer_circuit(
    architecture: MultiZoneArchitectureSpec,
) -> MultiZoneCircuit:
    backend = AQTMultiZoneBackend(architecture=architecture, access_token="invalid")
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=InitialPlacementSettings(
            algorithm=InitialPlacementAlg.manual,
            zone_free_space=0,
            manual_placement=[[2], [3], [], [0, 1], [4]],
        ),
        routing=RoutingConfig(
            router=SingleGateZoneLineArchRouter(),
            gate_selector=SiqciArchGateSelector(),
        ),
    )
    qft_circuit = build_test_circuit(5)
    compiled = backend.compile_circuit(qft_circuit, compilation_settings)
    return backend.route_compiled(compiled, compilation_settings)


def test_visualizer_handles_routed_linear_architectures(
    tmp_path: Path,
) -> None:
    circuit = _routed_non_linear_visualizer_circuit(siqci_arch)
    output_path = tmp_path / "movie.html"
    write_multi_zone_circuit_movie_html(
        circuit,
        output_path,
        title=f"linear 5 {circuit.get_n_shuttles()} shuttles, {circuit.get_n_pswaps()} swaps",
    )
