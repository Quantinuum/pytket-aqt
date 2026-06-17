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
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.qubit_routing.general_router import (
    GeneralRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)


def test_general_router_rejects_incomplete_target_placement() -> None:
    dyn_arch = DynamicArch(
        four_zones_in_a_line,
        TrapConfiguration(n_qubits=3, zone_placement=[[0], [1], [2], []]),
    )

    with pytest.raises(ValueError, match="Missing qubits: \\[2\\]"):
        GeneralRouter().route_source_to_target_config(dyn_arch, [[0], [1], [], []])


def test_general_router_rejects_duplicate_target_qubit() -> None:
    dyn_arch = DynamicArch(
        four_zones_in_a_line,
        TrapConfiguration(n_qubits=3, zone_placement=[[0], [1], [2], []]),
    )

    with pytest.raises(ValueError, match="duplicated qubits: \\[1\\]"):
        GeneralRouter().route_source_to_target_config(
            dyn_arch, [[0, 1], [1, 2], [], []]
        )
