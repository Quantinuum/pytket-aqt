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
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    linear_8_zones,
    siqci_arch,
)


def _empty_configuration(n_zones: int) -> TrapConfiguration:
    return TrapConfiguration(n_qubits=0, zone_placement=[[] for _ in range(n_zones)])


def test_siqci_arch_router_returns_dummy_empty_result_for_siqci_arch() -> None:
    dyn_arch = DynamicArch(siqci_arch, _empty_configuration(siqci_arch.n_zones))

    result = SiqciArchRouter().route_source_to_target_config(
        dyn_arch, dyn_arch.trap_configuration.zone_placement
    )

    assert result.cost_estimate == 0
    assert result.routing_ops == []


def test_siqci_arch_router_fails_for_other_architectures() -> None:
    dyn_arch = DynamicArch(linear_8_zones, _empty_configuration(linear_8_zones.n_zones))

    with pytest.raises(
        ValueError,
        match=r"SiqciArchRouter can only be used with the siqci_arch architecture\.",
    ):
        SiqciArchRouter().route_source_to_target_config(
            dyn_arch, dyn_arch.trap_configuration.zone_placement
        )
