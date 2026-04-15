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
    target_zone_interval_qubits,
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


def test_line_arch_router_fails_for_non_linear_architecture() -> None:
    dyn_arch = DynamicArch(grid12, _empty_configuration(grid12.n_zones))

    with pytest.raises(
        ValueError,
        match=r"LineArchRouter can only be used with linear architectures\.",
    ):
        LineArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )
