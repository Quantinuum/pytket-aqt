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
from pytket.circuit import Circuit

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
    SiqciArchGateSelector,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.siqci_arch_gate_selector import (
    handle_2qb_gates_remaining,
    handle_only_single_qubit_gates_remaining,
)
from pytket.extensions.aqt.multi_zone_architecture.depth_list.depth_list import (
    DepthInfo,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
    PhysicalConnection,
    PortId,
    PortSpec,
    Zone,
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


def _configuration(zone_placement: list[list[int]]) -> TrapConfiguration:
    return TrapConfiguration(
        n_qubits=sum(len(zone_qubits) for zone_qubits in zone_placement),
        zone_placement=zone_placement,
    )


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
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=i, port_id=PortId.p1),
                endpoint1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
            )
            for i in range(4)
        ],
    )


def test_siqci_arch_gate_selector_only_places_gate_qubits() -> None:
    assert SiqciArchGateSelector().only_places_gate_qubits()


def test_siqci_arch_gate_selector_returns_dummy_empty_target_for_siqci_arch() -> None:
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[], [], [], [0, 1], []]))

    target_config = SiqciArchGateSelector().next_config(dyn_arch, [])

    assert target_config == [[] for _ in range(siqci_arch.n_zones)]


def test_siqci_arch_gate_selector_requires_gate_zone_to_be_fully_occupied() -> None:
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[], [], [], [0], []]))

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchGateSelector requires the gate zone to be fully occupied "
            r"when selecting the next configuration\."
        ),
    ):
        SiqciArchGateSelector().next_config(dyn_arch, [])


def test_siqci_arch_gate_selector_fails_for_other_architectures() -> None:
    dyn_arch = SgzlDynamicArch(
        linear_8_zones, _empty_configuration(linear_8_zones.n_zones)
    )

    with pytest.raises(
        ValueError,
        match=(
            r"SiqciArchGateSelector can only be used with siqci-like linear "
            r"architectures with one gate zone and capacity-2 zones\."
        ),
    ):
        SiqciArchGateSelector().next_config(dyn_arch, [])


def test_siqci_arch_gate_selector_accepts_siqci_like_architecture_with_moved_gate_zone() -> (
    None
):
    moved_gate_zone_arch = _siqci_like_architecture(gate_zone=1)
    dyn_arch = SgzlDynamicArch(
        moved_gate_zone_arch, _configuration([[0], [1, 2], [3], [4], []])
    )

    target_config = SiqciArchGateSelector().next_config(dyn_arch, [])

    assert target_config == [[] for _ in range(moved_gate_zone_arch.n_zones)]


def test_siqci_arch_gate_selector_prefers_swap_free_pair_from_depth_block_zero() -> (
    None
):
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [], [2, 3], [4]]))

    target_config = handle_2qb_gates_remaining(
        dyn_arch,
        DepthInfo(
            depth_list=[[(0, 1), (3, 4)]],
            depth_blocks=[[{0, 1}, {3, 4}]],
        ),
    )

    assert target_config == [[], [], [], [3, 4], []]


def test_siqci_arch_gate_selector_uses_smallest_interval_pair_when_no_pair_is_swap_free() -> (
    None
):
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [], [2, 3], [4]]))

    target_config = handle_2qb_gates_remaining(
        dyn_arch,
        DepthInfo(
            depth_list=[[(0, 3), (1, 3)]],
            depth_blocks=[[{0, 3}, {1, 3}]],
        ),
    )

    assert target_config == [[], [], [], [1, 3], []]


def test_siqci_arch_gate_selector_keeps_existing_non_gate_pair_for_single_gate_qubit() -> (
    None
):
    dyn_arch = SgzlDynamicArch(
        siqci_arch, _configuration([[0, 1], [2], [], [3, 4], []])
    )
    circ = Circuit(5)
    circ.H(0)

    target_config = handle_only_single_qubit_gates_remaining(
        dyn_arch, circ.get_commands()
    )

    assert target_config == [[], [], [], [0], []]


def test_siqci_arch_gate_selector_uses_adjacent_unpaired_swap_free_pair_for_single_gate_qubit() -> (
    None
):
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [2], [3, 4], []]))
    circ = Circuit(5)
    circ.H(1)

    target_config = handle_only_single_qubit_gates_remaining(
        dyn_arch, circ.get_commands()
    )

    assert target_config == [[], [], [], [1], []]


def test_siqci_arch_gate_selector_falls_back_to_closest_gate_zone_qubit_for_single_gate_qubit() -> (
    None
):
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [], [2, 3], [4]]))
    circ = Circuit(5)
    circ.H(4)

    target_config = handle_only_single_qubit_gates_remaining(
        dyn_arch, circ.get_commands()
    )

    assert target_config == [[], [], [], [4], []]


def test_siqci_arch_gate_selector_chooses_singleton_when_no_pair_is_swap_free() -> None:
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [2], [3, 4], []]))
    circ = Circuit(5)
    circ.H(0)
    circ.Rx(0.5, 4)
    circ.Ry(0.25, 2)

    target_config = handle_only_single_qubit_gates_remaining(
        dyn_arch, circ.get_commands()
    )

    assert target_config == [[], [], [], [2], []]


def test_siqci_arch_gate_selector_prefers_swap_free_pair_over_singleton() -> None:
    dyn_arch = SgzlDynamicArch(siqci_arch, _configuration([[0], [1], [2], [3, 4], []]))
    circ = Circuit(5)
    circ.H(1)
    circ.Rx(0.5, 2)
    circ.Ry(0.25, 4)

    target_config = handle_only_single_qubit_gates_remaining(
        dyn_arch, circ.get_commands()
    )

    assert target_config == [[], [], [], [1, 2], []]
