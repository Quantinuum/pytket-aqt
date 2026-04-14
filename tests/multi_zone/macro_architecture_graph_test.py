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

from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.macro_architecture_graph import (
    MultiZoneArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    gate_zone_type_examples,
    grid12,
    racetrack,
)


def test_macro_architecture_detects_linear_architecture() -> None:
    macro_arch = MultiZoneArch(four_zones_in_a_line)

    assert macro_arch.is_linear_architecture


def test_macro_architecture_rejects_cycle_architecture() -> None:
    macro_arch = MultiZoneArch(racetrack)

    assert not macro_arch.is_linear_architecture


def test_macro_architecture_rejects_branched_architecture() -> None:
    macro_arch = MultiZoneArch(gate_zone_type_examples)

    assert not macro_arch.is_linear_architecture


def test_macro_architecture_rejects_grid_architecture() -> None:
    macro_arch = MultiZoneArch(grid12)

    assert not macro_arch.is_linear_architecture
