import logging

from numpy import ceil, log2, pi
from pytket.circuit.display import Any

from pytket import Circuit
from pytket.extensions.aqt.backends.aqt_multi_zone import (
    AQTMultiZoneBackend,
    get_aqt_json_syntax_for_compiled_circuit,
)
from pytket.extensions.aqt.logger import configure_logging
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
    GraphPartitionGateSelector,
    HypergraphPartitionGateSelector,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_config import (
    RoutingConfig,
)
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar import (
    MtKahyparConfig,
    configure_mtkahypar,
)
from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementAlg,
    InitialPlacementSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
    PortId,
    PortSpec,
    Zone,
    ZoneConnection,
)

configure_logging(level=logging.DEBUG)


def create_ghz_sequential(num_qubits: int) -> Circuit:
    ghz_circuit = Circuit(num_qubits)
    ghz_circuit.H(0)
    for i in range(ghz_circuit.n_qubits - 1):
        ghz_circuit.CX(i, i + 1)
    return ghz_circuit.measure_all()


def create_ghz_unbalanced(num_qubits: int) -> Circuit:
    ghz_circuit = Circuit(num_qubits)
    for qubit_index in range(ghz_circuit.n_qubits - 1):
        ghz_circuit.CX(0, qubit_index + 1)

    ghz_circuit.Rx(pi / 2, 0)
    for qubit_index in range(1, num_qubits):
        ghz_circuit.Rx(pi / 2, qubit_index)
    return ghz_circuit.measure_all()


def create_ghz_nested(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    circuit.H(0)
    l = int(ceil(log2(num_qubits)))
    for m in range(l, 0, -1):
        for k in range(0, num_qubits, 2**m):
            if k + 2 ** (m - 1) >= num_qubits:
                continue
            circuit.CX(k, k + 2 ** (m - 1))
    return circuit.measure_all()


order_init = InitialPlacementSettings(
    algorithm=InitialPlacementAlg.manual,
    zone_free_space=1,
    manual_placement={0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [10, 11]},
    max_depth=200,
)

compilation_settings: dict[str, CompilationSettings] = {
    "graph": CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(gate_selector=GraphPartitionGateSelector()),
    ),
    "hypergraph": CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(gate_selector=HypergraphPartitionGateSelector()),
    ),
    "greedy": CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(),
    ),
    "legacy": CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=order_init,
        routing=RoutingConfig(use_legacy_greedy_method=True),
    ),
}


def transpile(
    backend: AQTMultiZoneBackend, circuit: Circuit, settings: CompilationSettings
) -> tuple[Circuit, list[list[Any]]]:
    precompiled_circuit = backend.precompile_circuit(circuit, settings)
    routed_circuit = backend.route_precompiled(precompiled_circuit, settings)
    transpiled_circuit = get_aqt_json_syntax_for_compiled_circuit(routed_circuit)

    return (precompiled_circuit, transpiled_circuit)


def print_transpiled(
    transpiled_circuit: list[list[Any]], title="Transpiled Circuit:"
) -> None:
    print("----")
    print(f"{title}:")
    for op in transpiled_circuit:
        print(op)
    print("----")
    print("\n")


def count_ops(transpiled_circuit: list[list[Any]], title="Transpiled Circuit") -> None:
    print("----")
    print(f"Op counts for {title}:")
    shuttle_op_count = 0
    pswap_op_count = 0
    ms_op_count = 0
    sq_count = 0
    for op in transpiled_circuit:
        if "SHUTTLE" in op:
            shuttle_op_count += 1
        if "PSWAP" in op:
            pswap_op_count += 1
        if "MS" in op:
            ms_op_count += 1
        if "X" in op or "Y" in op:
            sq_count += 1
    print(f"SHUTTLE: {shuttle_op_count}")
    print(f"PSWAP: {pswap_op_count}")
    print(f"MS: {ms_op_count}")
    print(f"Single Qubit: {sq_count}")
    print("----")
    print("\n")


# example custom spec
MAX_SUBCRYSTAL_SIZE_B = 10
NUM_ZONES_B = 8
custom_spec = MultiZoneArchitectureSpec(
    n_qubits_max=NUM_ZONES_B * MAX_SUBCRYSTAL_SIZE_B,
    n_zones=NUM_ZONES_B,
    zones=[
        Zone(max_ions_gate_op=mi, memory_only=mem)
        for mi, mem in [
            (MAX_SUBCRYSTAL_SIZE_B, True),
            (MAX_SUBCRYSTAL_SIZE_B, True),
            (MAX_SUBCRYSTAL_SIZE_B, True),
            (MAX_SUBCRYSTAL_SIZE_B, False),
            (MAX_SUBCRYSTAL_SIZE_B, False),
            (MAX_SUBCRYSTAL_SIZE_B, True),
            (MAX_SUBCRYSTAL_SIZE_B, True),
            (MAX_SUBCRYSTAL_SIZE_B, True),
        ]
    ],
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
        )
        for i in range(NUM_ZONES_B - 1)
    ],
)


def main() -> None:
    line_backend = AQTMultiZoneBackend(architecture=custom_spec, access_token="invalid")

    # This can be used to configure the number of threads used and random seed
    # for mt-kahypar. It is not required (then default will be used) and can only
    # be set once
    configure_mtkahypar(MtKahyparConfig(n_threads=1, random_seed=243))
    circuit = create_ghz_sequential(12)
    precompiled_circuit = line_backend.compile_circuit(
        circuit, compilation_settings["graph"]
    )
    for name, comp_settings in compilation_settings.items():
        print("----")
        print(f"Performing {name} routing")
        routed_circuit = line_backend.route_compiled(precompiled_circuit, comp_settings)
        transpiled_circuit = get_aqt_json_syntax_for_compiled_circuit(routed_circuit)
        print_transpiled(transpiled_circuit, "Circuit")
        count_ops(transpiled_circuit, "Sequential GHZ")
        print("----")
        print()


if __name__ == "__main__":
    main()
