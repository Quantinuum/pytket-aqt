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

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
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


def test_dynamic_architecture_marks_linear_architecture() -> None:
    dyn_arch = DynamicArch(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line.n_zones)
    )

    assert dyn_arch.is_linear_architecture


def test_dynamic_architecture_marks_non_linear_architecture() -> None:
    dyn_arch = DynamicArch(grid12, _empty_configuration(grid12.n_zones))

    assert not dyn_arch.is_linear_architecture
