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
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
    SiqciArchGateSelector,
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


def _configuration(zone_placement: list[list[int]]) -> TrapConfiguration:
    return TrapConfiguration(
        n_qubits=sum(len(zone_qubits) for zone_qubits in zone_placement),
        zone_placement=zone_placement,
    )


def test_siqci_arch_gate_selector_only_places_gate_qubits() -> None:
    assert SiqciArchGateSelector().only_places_gate_qubits()


def test_siqci_arch_gate_selector_returns_dummy_empty_target_for_siqci_arch() -> None:
    dyn_arch = DynamicArch(siqci_arch, _configuration([[], [], [], [0, 1], []]))

    target_config = SiqciArchGateSelector().next_config(dyn_arch, [])

    assert target_config == [[] for _ in range(siqci_arch.n_zones)]


def test_siqci_arch_gate_selector_requires_gate_zone_to_be_fully_occupied() -> None:
    dyn_arch = DynamicArch(siqci_arch, _configuration([[], [], [], [0], []]))

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchGateSelector requires the gate zone to be fully occupied "
            r"when selecting the next configuration\."
        ),
    ):
        SiqciArchGateSelector().next_config(dyn_arch, [])


def test_siqci_arch_gate_selector_fails_for_other_architectures() -> None:
    dyn_arch = DynamicArch(linear_8_zones, _empty_configuration(linear_8_zones.n_zones))

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchGateSelector can only be used with the siqci_arch "
            r"architecture\."
        ),
    ):
        SiqciArchGateSelector().next_config(dyn_arch, [])
