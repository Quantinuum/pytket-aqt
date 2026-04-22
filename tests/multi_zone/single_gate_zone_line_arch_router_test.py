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
    SingleGateZoneLineArchRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_ops import (
    PSwap,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
    PortId,
    PortSpec,
    Zone,
    ZoneConnection,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
    SgzlDynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    grid12,
    linear_8_zones,
)


def _empty_configuration(n_zones: int) -> TrapConfiguration:
    return TrapConfiguration(n_qubits=0, zone_placement=[[] for _ in range(n_zones)])


def _configuration(zone_placement: list[list[int]]) -> TrapConfiguration:
    return TrapConfiguration(
        n_qubits=sum(len(zone_qubits) for zone_qubits in zone_placement),
        zone_placement=zone_placement,
    )


def _single_gate_zone_line_architecture(
    zone_gate_caps: list[int], gate_zone: int
) -> MultiZoneArchitectureSpec:
    return MultiZoneArchitectureSpec(
        n_qubits_max=sum(zone_gate_caps),
        n_zones=len(zone_gate_caps),
        zones=[
            Zone(
                max_ions_gate_op=capacity,
                max_ions_transport_op=capacity + 1,
                memory_only=zone != gate_zone,
            )
            for zone, capacity in enumerate(zone_gate_caps)
        ],
        connections=[
            ZoneConnection(
                zone_port_spec0=PortSpec(zone_id=zone, port_id=PortId.p1),
                zone_port_spec1=PortSpec(zone_id=zone + 1, port_id=PortId.p0),
            )
            for zone in range(len(zone_gate_caps) - 1)
        ],
    )


def test_single_gate_zone_line_arch_router_returns_empty_result_for_empty_target_placement() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        linear_8_zones, _empty_configuration(linear_8_zones.n_zones)
    )

    result = SingleGateZoneLineArchRouter().route_source_to_target_config(
        dyn_arch, dyn_arch.trap_configuration.zone_placement
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_single_gate_zone_line_arch_router_routes_swap_free_case_into_gate_zone() -> (
    None
):
    arch = _single_gate_zone_line_architecture([2, 2, 2], gate_zone=1)
    dyn_arch = SgzlDynamicArch(arch, _configuration([[0], [1], [2]]))

    result = SingleGateZoneLineArchRouter().route_source_to_target_config(
        dyn_arch,
        [[], [0, 1], []],
    )

    assert result.cost_estimate == 1
    assert not any(isinstance(op, PSwap) for op in result.routing_ops)
    assert dyn_arch.trap_configuration.zone_placement == [[], [0, 1], [2]]


def test_single_gate_zone_line_arch_router_uses_swaps_when_gate_interval_is_too_large() -> (
    None
):
    arch = _single_gate_zone_line_architecture([2, 2, 2], gate_zone=1)
    dyn_arch = SgzlDynamicArch(arch, _configuration([[0, 1], [2, 3], []]))

    result = SingleGateZoneLineArchRouter().route_source_to_target_config(
        dyn_arch,
        [[], [0, 3], []],
    )

    assert result.cost_estimate > 0
    assert any(isinstance(op, PSwap) for op in result.routing_ops)
    assert dyn_arch.trap_configuration.zone_placement[1] == [0, 3]


def test_single_gate_zone_line_arch_router_fails_for_target_placement_outside_gate_zone() -> (
    None
):
    arch = _single_gate_zone_line_architecture([2, 2, 2], gate_zone=1)
    dyn_arch = SgzlDynamicArch(arch, _configuration([[0], [1], [2]]))

    with pytest.raises(
        ValueError,
        match=(
            r"SingleGateZoneLineArchRouter requires target placements to specify "
            r"qubits only in the single gate zone\."
        ),
    ):
        SingleGateZoneLineArchRouter().route_source_to_target_config(
            dyn_arch, [[0], [], []]
        )


def test_single_gate_zone_line_arch_router_fails_for_non_linear_architecture() -> None:
    dyn_arch = DynamicArch(grid12, _empty_configuration(grid12.n_zones))

    with pytest.raises(
        ValueError,
        match=(
            r"This operation requires a SgzlDynamicArch input for a linear "
            r"architecture with exactly one gate zone\."
        ),
    ):
        SingleGateZoneLineArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )


def test_single_gate_zone_line_arch_router_fails_for_multiple_gate_zones() -> None:
    dyn_arch = DynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    with pytest.raises(
        ValueError,
        match=(
            r"This operation requires a SgzlDynamicArch input for a linear "
            r"architecture with exactly one gate zone\."
        ),
    ):
        SingleGateZoneLineArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )
