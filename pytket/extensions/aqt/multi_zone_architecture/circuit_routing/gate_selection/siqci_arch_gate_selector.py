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
from ...trap_architecture.dynamic_architecture import (
    DynamicArch,
    SgzlDynamicArch,
    require_sgzl_dynamic_arch,
)
from ...trap_architecture.named_architectures import siqci_arch
from .gate_selector_protocol import GateSelector

_PAIRED_ZONE_OCCUPANCY = 2


def _validate_siqci_architecture(dyn_arch: SgzlDynamicArch) -> None:
    if dyn_arch.architecture_spec != siqci_arch:
        raise ValueError(
            "SiqciArchGateSelector can only be used with the siqci_arch architecture."
        )
    gate_zone = dyn_arch.gate_zones[0]
    gate_zone_occupancy = len(dyn_arch.trap_configuration.zone_placement[gate_zone])
    gate_zone_capacity = int(dyn_arch.zone_max_gate_cap[gate_zone])
    if gate_zone_occupancy != gate_zone_capacity:
        raise ValueError(
            "SiqciArchGateSelector requires the gate zone to be fully occupied when selecting the next configuration."
        )


def _swap_free_single_gate_zone_possible(
    dyn_arch: SgzlDynamicArch, target_gate_qubits: list[int]
) -> tuple[bool, int]:
    if not target_gate_qubits:
        raise ValueError(
            "No target qubits provided to _swap_free_single_gate_zone_possible."
        )
    left_count, interval_count, right_count = dyn_arch.interval_counts(
        target_gate_qubits
    )
    return (
        interval_count <= dyn_arch.gate_capacity
        and left_count <= dyn_arch.left_capacity
        and right_count <= dyn_arch.right_capacity
    ), interval_count


def handle_2qb_gates_remaining(dyn_arch: SgzlDynamicArch, depth_info: DepthInfo):
    target_placement = [[] for _ in range(dyn_arch.n_zones)]
    largest_swap_free_list: list[int] = []
    smallest_interval_list: list[int] = []
    current_interval_count = dyn_arch.n_qubits

    # The gate zone can hold exactly two qubits, so only the first depth block
    # can contain viable two-qubit gate candidates for the next gating round.
    # Sorting the candidate blocks gives deterministic behaviour when several
    # pairs have the same routing quality.
    for pair in sorted(
        depth_info.depth_blocks[0], key=lambda block: tuple(sorted(block))
    ):
        candidate_pair = sorted(pair)
        swap_free, interval_length = _swap_free_single_gate_zone_possible(
            dyn_arch, candidate_pair
        )
        if swap_free and len(candidate_pair) > len(largest_swap_free_list):
            largest_swap_free_list = candidate_pair
        if interval_length <= current_interval_count:
            smallest_interval_list = candidate_pair
            current_interval_count = interval_length

    if largest_swap_free_list:
        target_placement[dyn_arch.single_gate_zone] = largest_swap_free_list
    else:
        target_placement[dyn_arch.single_gate_zone] = smallest_interval_list
    return target_placement


def _single_qubit_gate_qubits(remaining_commands: list[Command]) -> list[int]:
    # Preserve the first-seen command order while removing duplicates so that
    # later tie-breaks stay deterministic across repeated runs.
    return list(
        dict.fromkeys(
            cmd.args[0].index[0]
            for cmd in remaining_commands
            if cmd.op.type != OpType.Barrier
        )
    )


def _best_pair_from_candidates(
    dyn_arch: SgzlDynamicArch, candidate_pairs: list[list[int]]
) -> list[int]:
    largest_swap_free_list: list[int] = []
    smallest_interval_list: list[int] = []
    current_interval_count = dyn_arch.n_qubits

    # Normalising and sorting candidate pairs means that when several pairs are
    # equally good, we always choose the same one.
    for pair in sorted((sorted(pair) for pair in candidate_pairs), key=tuple):
        swap_free, interval_length = _swap_free_single_gate_zone_possible(
            dyn_arch, pair
        )
        if swap_free and not largest_swap_free_list:
            largest_swap_free_list = pair
        if interval_length < current_interval_count:
            smallest_interval_list = pair
            current_interval_count = interval_length

    return largest_swap_free_list or smallest_interval_list


def _single_gate_zone_partner_pair(
    dyn_arch: SgzlDynamicArch, target_qubit: int
) -> list[int]:
    ordered_qubits = dyn_arch.ordered_qubits()
    ordered_positions = {qubit: i for i, qubit in enumerate(ordered_qubits)}
    gate_zone_qubits = dyn_arch.trap_configuration.zone_placement[
        dyn_arch.single_gate_zone
    ]
    partner = min(
        (qubit for qubit in gate_zone_qubits if qubit != target_qubit),
        # Use qubit id as a final tie-break so equidistant choices are stable.
        key=lambda qubit: (
            abs(ordered_positions[qubit] - ordered_positions[target_qubit]),
            qubit,
        ),
    )
    return sorted([target_qubit, partner])


def _pair_single_remaining_gate_qubit(
    dyn_arch: SgzlDynamicArch, target_qubit: int
) -> list[int]:
    target_zone, _ = dyn_arch.qubit_to_zone_pos[target_qubit]
    target_zone_qubits = dyn_arch.trap_configuration.zone_placement[target_zone]

    # If the target qubit is already part of a 2-ion pair outside the gate zone,
    # keep that pair together for the next gating round.
    if (
        target_zone != dyn_arch.single_gate_zone
        and len(target_zone_qubits) == _PAIRED_ZONE_OCCUPANCY
    ):
        return sorted(target_zone_qubits)

    adjacent_candidate_pairs: list[list[int]] = []
    target_zone_position = dyn_arch.ordered_zone_positions[target_zone]
    ordered_zones = dyn_arch.linearly_ordered_zones
    for neighbour_position in (target_zone_position - 1, target_zone_position + 1):
        if not 0 <= neighbour_position < len(ordered_zones):
            continue
        neighbour_zone = ordered_zones[neighbour_position]
        neighbour_qubits = dyn_arch.trap_configuration.zone_placement[neighbour_zone]
        if len(neighbour_qubits) == 1:
            adjacent_candidate_pairs.append([target_qubit, neighbour_qubits[0]])

    best_adjacent_pair = _best_pair_from_candidates(dyn_arch, adjacent_candidate_pairs)
    if best_adjacent_pair:
        swap_free, _ = _swap_free_single_gate_zone_possible(
            dyn_arch, best_adjacent_pair
        )
        if swap_free:
            return best_adjacent_pair

    return _single_gate_zone_partner_pair(dyn_arch, target_qubit)


def handle_only_single_qubit_gates_remaining(
    dyn_arch: SgzlDynamicArch, remaining_commands: list[Command]
):
    target_placement = [[] for _ in range(dyn_arch.n_zones)]
    qubits_with_gates = _single_qubit_gate_qubits(remaining_commands)
    if not qubits_with_gates:
        return target_placement

    if len(qubits_with_gates) == 1:
        target_placement[dyn_arch.single_gate_zone] = _pair_single_remaining_gate_qubit(
            dyn_arch, qubits_with_gates[0]
        )
        return target_placement

    target_placement[dyn_arch.single_gate_zone] = _best_pair_from_candidates(
        dyn_arch, [list(pair) for pair in combinations(qubits_with_gates, 2)]
    )
    return target_placement


class SiqciArchGateSelector(GateSelector):
    """Temporary gate selector specialized to the siqci_arch architecture."""

    def only_places_gate_qubits(self) -> bool:
        return True

    def next_config(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        sgzl_dyn_arch = require_sgzl_dynamic_arch(dyn_arch)
        _validate_siqci_architecture(sgzl_dyn_arch)
        current_configuration = sgzl_dyn_arch.trap_configuration
        n_qubits = current_configuration.n_qubits
        depth_info = depth_info_from_command_list(n_qubits, remaining_commands)
        if depth_info.depth_list:
            placement = handle_2qb_gates_remaining(sgzl_dyn_arch, depth_info)
        else:
            placement = handle_only_single_qubit_gates_remaining(
                sgzl_dyn_arch, remaining_commands
            )
        return placement
