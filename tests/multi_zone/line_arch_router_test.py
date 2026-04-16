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
    LineArchRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.qubit_routing.line_arch_router import (
    execute_swap_free_segmentation,
    swap_free_routing_segmentation,
    target_zone_interval_qubits,
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
    DynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    grid12,
)


def _empty_configuration(n_zones: int) -> TrapConfiguration:
    return TrapConfiguration(n_qubits=0, zone_placement=[[] for _ in range(n_zones)])


def _configuration(zone_placement: list[list[int]]) -> TrapConfiguration:
    return TrapConfiguration(
        n_qubits=sum(len(zone_qubits) for zone_qubits in zone_placement),
        zone_placement=zone_placement,
    )


def _line_architecture(zone_gate_caps: list[int]) -> MultiZoneArchitectureSpec:
    return MultiZoneArchitectureSpec(
        n_qubits_max=sum(zone_gate_caps),
        n_zones=len(zone_gate_caps),
        zones=[
            Zone(max_ions_gate_op=capacity, max_ions_transport_op=capacity + 1)
            for capacity in zone_gate_caps
        ],
        connections=[
            ZoneConnection(
                zone_port_spec0=PortSpec(zone_id=zone, port_id=PortId.p1),
                zone_port_spec1=PortSpec(zone_id=zone + 1, port_id=PortId.p0),
            )
            for zone in range(len(zone_gate_caps) - 1)
        ],
    )


def test_line_arch_router_returns_dummy_routing_result_for_linear_architecture() -> (
    None
):
    dyn_arch = DynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    result = LineArchRouter().route_source_to_target_config(
        dyn_arch, dyn_arch.trap_configuration.zone_placement
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_line_arch_router_builds_pebble_hole_graph_from_dynamic_arch() -> None:
    dyn_arch = DynamicArch(four_zones_in_a_line, _configuration([[0, 1], [2], [], []]))

    pebble_hole_graph = dyn_arch.pebble_hole_graph()

    assert pebble_hole_graph.placement() == [[0, 1], [2], [], []]


@pytest.mark.parametrize(
    ("zone_placement", "target_placement", "expected_interval_qubits"),
    [
        pytest.param(
            [[0, 1], [2], [3], []],
            [[], [0, 3], [2], []],
            [[], [0, 1, 2, 3], [2], []],
            id="sandwiched-across-multiple-zones",
        ),
        pytest.param(
            [[0, 1], [2, 3], [], []],
            [[], [1, 2], [], []],
            [[], [1, 2], [], []],
            id="no-sandwiched-qubits-across-zone-boundary",
        ),
        pytest.param(
            [[0, 1, 2], [3], [], []],
            [[0, 2], [3], [], []],
            [[0, 1, 2], [3], [], []],
            id="sandwiched-within-single-zone",
        ),
        pytest.param(
            [[0], [1], [2], [3]],
            [[0], [], [2], [3]],
            [[0], [], [2], [3]],
            id="single-qubit-targets-no-sandwiched-qubits",
        ),
    ],
)
def test_line_arch_router_target_zone_interval_qubits_include_sandwiched_qubits(
    zone_placement: list[list[int]],
    target_placement: list[list[int]],
    expected_interval_qubits: list[list[int]],
) -> None:
    dyn_arch = DynamicArch(four_zones_in_a_line, _configuration(zone_placement))

    interval_qubits = target_zone_interval_qubits(
        dyn_arch,
        target_placement=target_placement,
    )

    assert interval_qubits == expected_interval_qubits


@pytest.mark.parametrize(
    (
        "arch",
        "zone_placement",
        "target_placement",
        "expected_block_sizes",
        "expected_zone_placement",
    ),
    [
        pytest.param(
            four_zones_in_a_line,
            [[0, 1], [2], [3], []],
            [[], [0, 1], [2, 3], []],
            [0, 2, 2, 0],
            [[], [0, 1], [2, 3], []],
            id="swap-free-routing-is-possible",
        ),
        pytest.param(
            _line_architecture([2, 2, 2]),
            [[0], [1], [2]],
            [[], [0, 2], []],
            None,
            None,
            id="sandwiched-interval-exceeds-zone-capacity",
        ),
        pytest.param(
            _line_architecture([3, 3]),
            [[0, 1, 2], [3]],
            [[], [0, 1]],
            None,
            None,
            id="not-enough-global-space-to-shift-unspecified-qubits",
        ),
        pytest.param(
            _line_architecture([3, 3]),
            [[0, 1], [2, 3]],
            [[1], [0]],
            None,
            None,
            id="inversion-requires-swap",
        ),
        pytest.param(
            _line_architecture([3, 3, 3, 3, 3]),
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            [[], [0, 1, 2], [], [6, 7, 8], []],
            [0, 3, 3, 3, 1],
            [[], [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
            id="just-enough-space-in-zone3",
        ),
        pytest.param(
            _line_architecture([3, 3, 3, 3, 3]),
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            [[], [0, 1, 2], [], [7, 8, 9], []],
            None,
            None,
            id="not-enough-space-in-zone3",
        ),
    ],
)
def test_swap_free_routing_segmentation_matches_expected_result(
    arch: MultiZoneArchitectureSpec,
    zone_placement: list[list[int]],
    target_placement: list[list[int]],
    expected_block_sizes: list[int] | None,
    expected_zone_placement: list[list[int]] | None,
) -> None:
    dyn_arch = DynamicArch(arch, _configuration(zone_placement))

    segmentation = swap_free_routing_segmentation(
        dyn_arch,
        target_placement=target_placement,
    )

    if expected_block_sizes is None:
        assert segmentation is None
        return

    assert segmentation is not None
    assert segmentation.block_sizes == expected_block_sizes
    assert segmentation.zone_placement == expected_zone_placement


def test_swap_free_routing_segmentation_returns_block_sizes_and_zone_placement() -> (
    None
):
    dyn_arch = DynamicArch(
        _line_architecture([3, 3, 3]),
        _configuration([[0, 1], [2, 3], [4]]),
    )

    segmentation = swap_free_routing_segmentation(
        dyn_arch,
        target_placement=[[], [0, 1, 2], []],
    )

    assert segmentation is not None
    assert segmentation.ordered_zones == [0, 1, 2]
    assert segmentation.block_sizes == [0, 3, 2]
    assert segmentation.zone_placement == [[], [0, 1, 2], [3, 4]]


def test_swap_free_routing_segmentation_returns_none_when_swaps_are_required() -> None:
    dyn_arch = DynamicArch(
        _line_architecture([3, 3]),
        _configuration([[0, 1, 2], [3]]),
    )

    segmentation = swap_free_routing_segmentation(
        dyn_arch,
        target_placement=[[], [0, 1]],
    )

    assert segmentation is None


def test_execute_swap_free_segmentation_returns_shuttles_and_updates_dynamic_arch() -> (
    None
):
    dyn_arch = DynamicArch(
        _line_architecture([3, 3, 3]),
        _configuration([[0, 1], [2, 3], [4]]),
    )
    segmentation = swap_free_routing_segmentation(
        dyn_arch,
        target_placement=[[], [0, 1, 2], []],
    )
    assert segmentation is not None

    result = execute_swap_free_segmentation(dyn_arch, segmentation)

    assert result.cost_estimate == 2
    assert result.routing_ops == [
        RoutingBarrier(),
        Shuttle([3], 1, 2, PortId.p1, PortId.p0),
        Shuttle([0, 1], 0, 1, PortId.p1, PortId.p0),
        RoutingBarrier(),
    ]
    assert dyn_arch.trap_configuration.zone_placement == [[], [0, 1, 2], [3, 4]]


def test_line_arch_router_raises_for_target_placements_that_require_swaps() -> None:
    dyn_arch = DynamicArch(
        _line_architecture([3, 3]),
        _configuration([[0, 1, 2], [3]]),
    )

    with pytest.raises(
        ValueError,
        match=r"LineArchRouter does not yet support target placements that require swaps\.",
    ):
        LineArchRouter().route_source_to_target_config(dyn_arch, [[], [0, 1]])


def test_line_arch_router_fails_for_non_linear_architecture() -> None:
    dyn_arch = DynamicArch(grid12, _empty_configuration(grid12.n_zones))

    with pytest.raises(
        ValueError,
        match=r"LineArchRouter can only be used with linear architectures\.",
    ):
        LineArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )
