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

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.qubit_routing import (
    SiqciArchRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_ops import (
    RoutingBarrier,
    Shuttle,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
    PortId,
    PortSpec,
    Zone,
    ZoneConnection,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    SgzlDynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    linear_8_zones,
    siqci_arch,
)


def _empty_configuration(n_zones: int) -> TrapConfiguration:
    return TrapConfiguration(n_qubits=0, zone_placement=[[] for _ in range(n_zones)])


def _siqci_like_architecture(gate_zone: int) -> MultiZoneArchitectureSpec:
    return MultiZoneArchitectureSpec(
        n_qubits_max=5,
        n_zones=5,
        zones=[
            Zone(
                max_ions_gate_op=2,
                max_ions_transport_op=2,
                memory_only=i != gate_zone,
            )
            for i in range(5)
        ],
        connections=[
            ZoneConnection(
                zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
                zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
            )
            for i in range(4)
        ],
    )


def test_siqci_arch_router_requires_one_or_two_qubits_in_target_gate_zone() -> None:
    dyn_arch = SgzlDynamicArch(siqci_arch, _empty_configuration(siqci_arch.n_zones))

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchRouter target placement must specify one or two qubits "
            r"in the gate zone\."
        ),
    ):
        SiqciArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )


def test_siqci_arch_router_returns_empty_when_target_pair_already_in_gate_zone() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        siqci_arch, TrapConfiguration(2, [[], [], [], [0, 1], []])
    )

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, [[], [], [], [0, 1], []]
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_siqci_arch_router_returns_empty_when_target_singleton_already_in_gate_zone() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        siqci_arch, TrapConfiguration(2, [[], [], [], [0, 1], []])
    )

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, [[], [], [], [0], []]
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_siqci_arch_router_routes_pair_into_gate_zone_without_pswaps_when_unnecessary() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        siqci_arch, TrapConfiguration(4, [[0], [1], [], [2, 3], []])
    )

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, [[], [], [], [0, 1], []]
    )

    assert dyn_arch.trap_configuration.zone_placement == [[], [], [], [0, 1], [2, 3]]
    assert all(isinstance(op, (RoutingBarrier, Shuttle)) for op in result.routing_ops)
    assert sum(isinstance(op, Shuttle) for op in result.routing_ops) == 4
    assert result.cost_estimate == pytest.approx(4.8)


def test_siqci_arch_router_routes_singleton_into_gate_zone_with_cheapest_partner() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        siqci_arch, TrapConfiguration(4, [[0], [1], [], [2, 3], []])
    )

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, [[], [], [], [0], []]
    )

    assert dyn_arch.trap_configuration.zone_placement == [[], [], [], [0, 1], [2, 3]]
    assert all(isinstance(op, (RoutingBarrier, Shuttle)) for op in result.routing_ops)
    assert sum(isinstance(op, Shuttle) for op in result.routing_ops) == 4
    assert result.cost_estimate == pytest.approx(4.8)


def test_siqci_arch_router_fails_for_other_architectures() -> None:
    dyn_arch = SgzlDynamicArch(
        linear_8_zones, _empty_configuration(linear_8_zones.n_zones)
    )

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchRouter can only be used with siqci-like linear "
            r"architectures with one gate zone and capacity-2 zones\."
        ),
    ):
        SiqciArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )


def test_siqci_arch_router_accepts_siqci_like_architecture_with_moved_gate_zone() -> (
    None
):
    moved_gate_zone_arch = _siqci_like_architecture(gate_zone=1)
    dyn_arch = SgzlDynamicArch(
        moved_gate_zone_arch, TrapConfiguration(2, [[], [0, 1], [], [], []])
    )

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, [[], [0, 1], [], [], []]
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_siqci_arch_router_rejects_target_qubits_outside_gate_zone() -> None:
    dyn_arch = SgzlDynamicArch(
        siqci_arch, TrapConfiguration(2, [[], [], [], [0, 1], []])
    )

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchRouter requires target placements to specify qubits only "
            r"in the gate zone\."
        ),
    ):
        SiqciArchRouter().route_source_to_target_config(
            dyn_arch, [[0], [], [], [1], []]
        )
