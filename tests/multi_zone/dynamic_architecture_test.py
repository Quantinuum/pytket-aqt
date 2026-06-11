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
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
    LinearDynamicArch,
    SgzlDynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    grid12,
    linear_8_zones,
)


def _empty_configuration(n_zones: int) -> TrapConfiguration:
    return TrapConfiguration(n_qubits=0, zone_placement=[[] for _ in range(n_zones)])


def test_dynamic_architecture_marks_linear_architecture() -> None:
    dyn_arch = DynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    assert dyn_arch.is_linear_architecture


def test_dynamic_architecture_marks_non_linear_architecture() -> None:
    dyn_arch = DynamicArch(grid12, _empty_configuration(grid12.n_zones))

    assert not dyn_arch.is_linear_architecture


def test_linear_dynamic_architecture_caches_line_structure() -> None:
    dyn_arch = LinearDynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    assert dyn_arch.line_start_zone == 0
    assert dyn_arch.linearly_ordered_zones == (0, 1, 2, 3)
    assert dyn_arch.ordered_zone_positions == {0: 0, 1: 1, 2: 2, 3: 3}
    assert dyn_arch.ordered_qubits() == []


def test_linear_dynamic_architecture_rejects_non_linear_architecture() -> None:
    with pytest.raises(
        ValueError,
        match=r"LinearDynamicArch can only be used with linear architectures\.",
    ):
        LinearDynamicArch(grid12, _empty_configuration(grid12.n_zones))


def test_linear_dynamic_architecture_can_be_promoted_from_dynamic_arch() -> None:
    base_dyn_arch = DynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    dyn_arch = LinearDynamicArch.from_dynamic_arch(base_dyn_arch)

    assert dyn_arch.line_start_zone == 0
    assert dyn_arch.linearly_ordered_zones == (0, 1, 2, 3)


def test_sgzl_dynamic_architecture_caches_single_gate_zone_data() -> None:
    dyn_arch = SgzlDynamicArch(
        linear_8_zones, _empty_configuration(linear_8_zones.n_zones)
    )

    assert dyn_arch.single_gate_zone == 3
    assert dyn_arch.interval_capacities == (18, 6, 24)


def test_sgzl_dynamic_architecture_can_be_promoted_from_linear_dynamic_arch() -> None:
    linear_dyn_arch = LinearDynamicArch(
        linear_8_zones, _empty_configuration(linear_8_zones.n_zones)
    )

    dyn_arch = SgzlDynamicArch.from_linear_dynamic_arch(linear_dyn_arch)

    assert dyn_arch.single_gate_zone == 3


def test_sgzl_dynamic_architecture_rejects_non_single_gate_zone_linear_architecture() -> (
    None
):
    with pytest.raises(
        ValueError,
        match=(
            r"SgzlDynamicArch requires a linear architecture with exactly one gate "
            r"zone\."
        ),
    ):
        SgzlDynamicArch(
            four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
        )
