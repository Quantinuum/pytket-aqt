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

from itertools import combinations

from pytket.circuit import Command

from pytket import OpType

from ...circuit.helpers import ZonePlacement
from ...depth_list.depth_list import DepthInfo, depth_info_from_command_list
from ...trap_architecture.dynamic_architecture import DynamicArch
from ..qubit_routing.line_arch_router import (
    linearly_ordered_zones,
    ordered_qubits,
)
from .gate_selector_protocol import GateSelector


def _single_gate_zone(dyn_arch: DynamicArch) -> int:
    if not dyn_arch.is_linear_architecture:
        raise ValueError(
            "SingleGateZoneLineArchRouter can only be used with linear architectures."
        )
    if len(dyn_arch.gate_zones) != 1:
        raise ValueError(
            "SingleGateZoneLineArchRouter requires an architecture with exactly one gate zone."
        )
    return dyn_arch.gate_zones[0]


def _single_gate_zone_capacities(
    dyn_arch: DynamicArch, gate_zone: int
) -> tuple[int, int, int]:
    ordered_zones = linearly_ordered_zones(dyn_arch)
    gate_zone_position = ordered_zones.index(gate_zone)
    left_capacity = sum(
        int(dyn_arch.zone_max_gate_cap[zone])
        for zone in ordered_zones[:gate_zone_position]
    )
    gate_capacity = int(dyn_arch.zone_max_gate_cap[gate_zone])
    right_capacity = sum(
        int(dyn_arch.zone_max_gate_cap[zone])
        for zone in ordered_zones[gate_zone_position + 1 :]
    )
    return left_capacity, gate_capacity, right_capacity


def _interval_counts(
    current_order: list[int], target_gate_qubits: list[int]
) -> tuple[int, int, int]:
    qubit_positions = {qubit: i for i, qubit in enumerate(current_order)}
    target_positions = sorted(qubit_positions[qubit] for qubit in target_gate_qubits)
    left_count = target_positions[0]
    interval_count = target_positions[-1] - target_positions[0] + 1
    right_count = len(current_order) - target_positions[-1] - 1
    return left_count, interval_count, right_count


def _swap_free_single_gate_zone_possible(
    dyn_arch: DynamicArch, target_gate_qubits: list[int], gate_zone: int
) -> tuple[bool, int]:
    if not target_gate_qubits:
        return True, 0
    left_count, interval_count, right_count = _interval_counts(
        ordered_qubits(dyn_arch), target_gate_qubits
    )
    left_capacity, gate_capacity, right_capacity = _single_gate_zone_capacities(
        dyn_arch, gate_zone
    )
    return (
        interval_count <= gate_capacity
        and left_count <= left_capacity
        and right_count <= right_capacity
    ), interval_count


def handle_2qb_gates_remaining(
    dyn_arch: DynamicArch, depth_info: DepthInfo
) -> ZonePlacement:
    target_placement = [[] for _ in range(dyn_arch.n_zones)]
    gate_zone = _single_gate_zone(dyn_arch)
    gate_capacity = int(dyn_arch.zone_max_gate_cap[gate_zone])
    largest_swap_free_list = []
    smallest_interval_list = []
    current_interval_count = dyn_arch.n_qubits
    swap_free_block_found = False
    for depth in depth_info.depth_blocks:
        swap_free_block_current_depth_found = False
        new_smallest_interval = False
        for block in depth:
            if len(block) > gate_capacity:
                continue
            swap_free, interval_length = _swap_free_single_gate_zone_possible(
                dyn_arch, list(block), gate_zone
            )
            if swap_free:
                swap_free_block_found = True
                swap_free_block_current_depth_found = True
                if len(block) > len(largest_swap_free_list):
                    largest_swap_free_list = list(block)
            if interval_length <= current_interval_count:
                smallest_interval_list = list(block)
                current_interval_count = interval_length
                new_smallest_interval = True
        if swap_free_block_found:
            if not swap_free_block_current_depth_found:
                break
        elif not new_smallest_interval:
            break

    if largest_swap_free_list:
        target_placement[gate_zone] = largest_swap_free_list
    else:
        target_placement[gate_zone] = smallest_interval_list

    return target_placement


def handle_only_single_qubit_gates_remaining(
    dyn_arch: DynamicArch, remaining_commands: list[Command]
) -> ZonePlacement:
    target_placement = [[] for _ in range(dyn_arch.n_zones)]
    gate_zone = _single_gate_zone(dyn_arch)
    gate_capacity = int(dyn_arch.zone_max_gate_cap[gate_zone])
    largest_swap_free_list = []
    smallest_interval_list = []
    current_interval_count = dyn_arch.n_qubits
    qubits_with_gates = []
    for cmd in remaining_commands:
        if cmd.op.type == OpType.Barrier:
            continue
        qubits_with_gates.append(cmd.args[0].index[0])
    swap_free_block_found = False
    for i in range(1, min(len(qubits_with_gates), gate_capacity)):
        swap_free_block_current_depth_found = False
        new_smallest_interval = False
        for block in combinations(qubits_with_gates, i):
            swap_free, interval_length = _swap_free_single_gate_zone_possible(
                dyn_arch, list(block), gate_zone
            )
            if swap_free:
                swap_free_block_found = True
                swap_free_block_current_depth_found = True
                if len(block) > len(largest_swap_free_list):
                    largest_swap_free_list = list(block)
            if interval_length <= current_interval_count:
                smallest_interval_list = list(block)
                current_interval_count = interval_length
                new_smallest_interval = True
        if swap_free_block_found:
            if not swap_free_block_current_depth_found:
                break
        elif not new_smallest_interval:
            break

    if largest_swap_free_list:
        target_placement[gate_zone] = largest_swap_free_list
    else:
        target_placement[gate_zone] = smallest_interval_list

    return target_placement


class SingleGateZoneLineArchGateSelector(GateSelector):
    """Gate selector specialized for linear architectures with a single gate zone.

    Tries to choose configurations that do not require any swap gates if possible

    """

    def only_places_gate_qubits(self) -> bool:
        return True

    def next_config(
        self, dyn_arch: DynamicArch, remaining_commands: list[Command]
    ) -> ZonePlacement:
        current_configuration = dyn_arch.trap_configuration
        n_qubits = current_configuration.n_qubits
        depth_info = depth_info_from_command_list(n_qubits, remaining_commands)
        if depth_info.depth_list:
            placement = handle_2qb_gates_remaining(dyn_arch, depth_info)
        else:
            placement = handle_only_single_qubit_gates_remaining(
                dyn_arch, remaining_commands
            )

        return placement
