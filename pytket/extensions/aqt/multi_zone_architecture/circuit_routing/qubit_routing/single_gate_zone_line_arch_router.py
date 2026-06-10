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

from dataclasses import dataclass

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.dynamic_architecture import (
    DynamicArch,
    SgzlDynamicArch,
    require_sgzl_dynamic_arch,
)
from ..routing_ops import RoutingBarrier, RoutingOp, Shuttle
from .line_arch_router import (
    _current_block_sizes,
    _current_prefix_size,
    _shuttle_across_boundary,
    execute_adjacent_swap_in_workspace,
    ordered_qubits,
    swap_free_routing_segmentation,
)
from .router import Router, RoutingResult

_LEFT_CLASS = 0
_GATE_CLASS = 1
_RIGHT_CLASS = 2
_WORKSPACE_ZONE_MIN_SIZE = 2


@dataclass(frozen=True)
class _PartitionConfig:
    target_qubit_set: frozenset[int]
    left_capacity: int
    gate_capacity: int
    right_capacity: int
    cost_weight: int
    inf_cost: int


@dataclass
class _ClassAssignmentTables:
    dp: list[list[list[int]]]
    predecessor: list[list[list[tuple[int, int, int] | None]]]


@dataclass(frozen=True)
class _PartitionState:
    index: int
    n_left: int
    n_gate: int
    current_cost: int


def _append_routing_ops(
    accumulated_ops: list[RoutingOp], new_ops: list[RoutingOp]
) -> None:
    if not new_ops:
        return
    if (
        accumulated_ops
        and isinstance(accumulated_ops[-1], RoutingBarrier)
        and isinstance(new_ops[0], RoutingBarrier)
    ):
        accumulated_ops.extend(new_ops[1:])
        return
    accumulated_ops.extend(new_ops)


def _validate_target_placement(
    dyn_arch: SgzlDynamicArch, target_placement: ZonePlacement
) -> tuple[int, list[int]]:
    gate_zone = dyn_arch.single_gate_zone
    for zone, zone_qubits in enumerate(target_placement):
        if zone != gate_zone and zone_qubits:
            raise ValueError(
                "SingleGateZoneLineArchRouter requires target placements to specify qubits only in the single gate zone."
            )
    target_gate_qubits = target_placement[gate_zone]
    if len(target_gate_qubits) > int(dyn_arch.zone_max_gate_cap[gate_zone]):
        raise ValueError(
            "SingleGateZoneLineArchRouter target placement exceeds the capacity of the single gate zone."
        )
    return gate_zone, target_gate_qubits


def _swap_free_single_gate_zone_possible(
    dyn_arch: SgzlDynamicArch, target_gate_qubits: list[int]
) -> bool:
    if not target_gate_qubits:
        return True
    left_count, interval_count, right_count = dyn_arch.interval_counts(
        target_gate_qubits
    )
    return (
        interval_count <= dyn_arch.gate_capacity
        and left_count <= dyn_arch.left_capacity
        and right_count <= dyn_arch.right_capacity
    )


def _target_gate_qubits_already_in_gate_zone(
    dyn_arch: SgzlDynamicArch, target_gate_qubits: list[int]
) -> bool:
    if not target_gate_qubits:
        return True
    gate_zone_qubits = set(
        dyn_arch.trap_configuration.zone_placement[dyn_arch.single_gate_zone]
    )
    return set(target_gate_qubits).issubset(gate_zone_qubits)


def _execute_swap_free_single_gate_zone_target(
    dyn_arch: SgzlDynamicArch,
    target_placement: ZonePlacement,
    target_gate_qubits: list[int],
) -> RoutingResult:
    if _target_gate_qubits_already_in_gate_zone(dyn_arch, target_gate_qubits):
        return RoutingResult(cost_estimate=0, routing_ops=[])

    segmentation = swap_free_routing_segmentation(dyn_arch, target_placement)
    if segmentation is None:
        raise ValueError(
            "SingleGateZoneLineArchRouter expected a shuttle-only route to the gate zone but none was found."
        )
    if dyn_arch.trap_configuration.zone_placement == segmentation.zone_placement:
        return RoutingResult(cost_estimate=0, routing_ops=[])

    desired_block_sizes = segmentation.block_sizes
    ops: list[RoutingOp] = []
    total_cost = 0

    def append_shuttle(shuttle: Shuttle, shuttle_cost: int) -> None:
        nonlocal total_cost
        if not ops:
            ops.append(RoutingBarrier())
        ops.extend([shuttle, RoutingBarrier()])
        total_cost += shuttle_cost

    def target_reached() -> bool:
        return _target_gate_qubits_already_in_gate_zone(dyn_arch, target_gate_qubits)

    def sweep_boundaries(move_right: bool) -> bool:
        boundary_indices = (
            range(len(segmentation.ordered_zones) - 2, -1, -1)
            if move_right
            else range(len(segmentation.ordered_zones) - 1)
        )
        progress = False
        for boundary_index in boundary_indices:
            delta = (
                _current_prefix_size(
                    dyn_arch, segmentation.ordered_zones, boundary_index
                )
                - sum(desired_block_sizes[: boundary_index + 1])
                if move_right
                else sum(desired_block_sizes[: boundary_index + 1])
                - _current_prefix_size(
                    dyn_arch, segmentation.ordered_zones, boundary_index
                )
            )
            if delta <= 0:
                continue
            shuttle_result = _shuttle_across_boundary(
                dyn_arch,
                segmentation.ordered_zones,
                boundary_index,
                move_right=move_right,
                max_n_move=delta,
            )
            if shuttle_result is None:
                continue
            shuttle, shuttle_cost = shuttle_result
            append_shuttle(shuttle, shuttle_cost)
            progress = True
            if target_reached():
                break
        return progress

    while (
        _current_block_sizes(dyn_arch, segmentation.ordered_zones)
        != desired_block_sizes
    ):
        if target_reached():
            return RoutingResult(cost_estimate=total_cost, routing_ops=ops)

        progress = sweep_boundaries(move_right=True)
        if target_reached():
            return RoutingResult(cost_estimate=total_cost, routing_ops=ops)
        progress = sweep_boundaries(move_right=False) or progress
        if target_reached():
            return RoutingResult(cost_estimate=total_cost, routing_ops=ops)

        if not progress:
            raise ValueError(
                "SingleGateZoneLineArchRouter could not complete the shuttle-only route into the gate zone."
            )

    return RoutingResult(cost_estimate=total_cost, routing_ops=ops)


def _class_assignment_target_order(
    dyn_arch: SgzlDynamicArch,
    target_gate_qubits: list[int],
) -> list[int]:
    current_order = ordered_qubits(dyn_arch)
    left_capacity, gate_capacity, right_capacity = dyn_arch.interval_capacities
    target_qubit_set = set(target_gate_qubits)
    n_qubits = len(current_order)
    total_capacity = left_capacity + gate_capacity + right_capacity
    if n_qubits > total_capacity:
        raise ValueError(
            "SingleGateZoneLineArchRouter cannot fit all qubits into the final gate-phase capacities."
        )

    config = _PartitionConfig(
        target_qubit_set=frozenset(target_qubit_set),
        left_capacity=left_capacity,
        gate_capacity=gate_capacity,
        right_capacity=right_capacity,
        cost_weight=n_qubits + 1,
        inf_cost=n_qubits**3 * (n_qubits + 1),
    )
    tables = _class_assignment_tables(
        current_order=current_order,
        config=config,
    )
    best_state = _best_class_assignment_state(
        tables=tables,
        n_qubits=n_qubits,
        config=config,
    )
    if best_state is None:
        raise ValueError(
            "SingleGateZoneLineArchRouter could not find a feasible left/gate/right partition."
        )

    class_by_qubit = _reconstruct_class_assignment(
        current_order=current_order,
        predecessor=tables.predecessor,
        best_state=best_state,
    )
    return (
        [qubit for qubit in current_order if class_by_qubit[qubit] == _LEFT_CLASS]
        + [qubit for qubit in current_order if class_by_qubit[qubit] == _GATE_CLASS]
        + [qubit for qubit in current_order if class_by_qubit[qubit] == _RIGHT_CLASS]
    )


def _class_assignment_tables(
    current_order: list[int],
    config: _PartitionConfig,
) -> _ClassAssignmentTables:
    n_qubits = len(current_order)
    dp = [
        [
            [config.inf_cost] * (config.gate_capacity + 1)
            for _ in range(config.left_capacity + 1)
        ]
        for _ in range(n_qubits + 1)
    ]
    predecessor: list[list[list[tuple[int, int, int] | None]]] = [
        [[None] * (config.gate_capacity + 1) for _ in range(config.left_capacity + 1)]
        for _ in range(n_qubits + 1)
    ]
    dp[0][0][0] = 0
    tables = _ClassAssignmentTables(dp=dp, predecessor=predecessor)

    for i, qubit in enumerate(current_order):
        for n_left in range(config.left_capacity + 1):
            for n_gate in range(config.gate_capacity + 1):
                current_cost = dp[i][n_left][n_gate]
                if current_cost >= config.inf_cost:
                    continue
                _update_class_assignment_transitions(
                    tables=tables,
                    qubit=qubit,
                    state=_PartitionState(
                        index=i,
                        n_left=n_left,
                        n_gate=n_gate,
                        current_cost=current_cost,
                    ),
                    config=config,
                )
    return tables


def _update_class_assignment_transitions(
    tables: _ClassAssignmentTables,
    qubit: int,
    state: _PartitionState,
    config: _PartitionConfig,
) -> None:
    n_right = state.index - state.n_left - state.n_gate
    if n_right < 0 or n_right > config.right_capacity:
        return

    if qubit in config.target_qubit_set:
        allowed_classes = [_GATE_CLASS]
    else:
        allowed_classes = [_LEFT_CLASS, _GATE_CLASS, _RIGHT_CLASS]

    for qubit_class in allowed_classes:
        transition = _class_assignment_transition(
            qubit=qubit,
            qubit_class=qubit_class,
            state=state,
            n_right=n_right,
            config=config,
        )
        if transition is None:
            continue
        next_left, next_gate, transition_cost = transition
        next_cost = state.current_cost + transition_cost
        if next_cost < tables.dp[state.index + 1][next_left][next_gate]:
            tables.dp[state.index + 1][next_left][next_gate] = next_cost
            tables.predecessor[state.index + 1][next_left][next_gate] = (
                state.n_left,
                state.n_gate,
                qubit_class,
            )


def _class_assignment_transition(
    qubit: int,
    qubit_class: int,
    state: _PartitionState,
    n_right: int,
    config: _PartitionConfig,
) -> tuple[int, int, int] | None:
    if qubit_class == _LEFT_CLASS:
        if state.n_left >= config.left_capacity:
            return None
        return (
            state.n_left + 1,
            state.n_gate,
            (state.index - state.n_left) * config.cost_weight,
        )

    if qubit_class == _GATE_CLASS:
        if state.n_gate >= config.gate_capacity:
            return None
        return (
            state.n_left,
            state.n_gate + 1,
            (n_right * config.cost_weight)
            + (0 if qubit in config.target_qubit_set else 1),
        )

    if n_right >= config.right_capacity:
        return None
    return state.n_left, state.n_gate, 0


def _best_class_assignment_state(
    tables: _ClassAssignmentTables,
    n_qubits: int,
    config: _PartitionConfig,
) -> tuple[int, int] | None:
    best_state: tuple[int, int] | None = None
    best_cost = config.inf_cost
    for n_left in range(config.left_capacity + 1):
        for n_gate in range(config.gate_capacity + 1):
            n_right = n_qubits - n_left - n_gate
            if (
                0 <= n_right <= config.right_capacity
                and tables.dp[n_qubits][n_left][n_gate] < best_cost
            ):
                best_cost = tables.dp[n_qubits][n_left][n_gate]
                best_state = (n_left, n_gate)
    return best_state


def _reconstruct_class_assignment(
    current_order: list[int],
    predecessor: list[list[list[tuple[int, int, int] | None]]],
    best_state: tuple[int, int],
) -> dict[int, int]:
    class_by_qubit: dict[int, int] = {}
    n_left, n_gate = best_state
    for i in range(len(current_order), 0, -1):
        previous_state = predecessor[i][n_left][n_gate]
        if previous_state is None:
            raise ValueError(
                "SingleGateZoneLineArchRouter is missing a predecessor while reconstructing the class assignment."
            )
        previous_left, previous_gate, qubit_class = previous_state
        class_by_qubit[current_order[i - 1]] = qubit_class
        n_left, n_gate = previous_left, previous_gate
    return class_by_qubit


def _adjacent_swap_sequence(
    current_order: list[int], target_order: list[int]
) -> list[tuple[int, int]]:
    mutable_order = current_order.copy()
    swap_sequence: list[tuple[int, int]] = []
    for target_index, target_qubit in enumerate(target_order):
        current_index = mutable_order.index(target_qubit)
        while current_index > target_index:
            left_qubit = mutable_order[current_index - 1]
            right_qubit = mutable_order[current_index]
            swap_sequence.append((left_qubit, right_qubit))
            mutable_order[current_index - 1], mutable_order[current_index] = (
                mutable_order[current_index],
                mutable_order[current_index - 1],
            )
            current_index -= 1
    return swap_sequence


def _workspace_candidate_zones(
    dyn_arch: SgzlDynamicArch, left_qubit: int, right_qubit: int
) -> list[int]:
    ordered_zones = list(dyn_arch.linearly_ordered_zones)
    ordered_zone_positions = dyn_arch.ordered_zone_positions
    left_zone = int(dyn_arch.qubit_to_zone_pos[left_qubit][0])
    right_zone = int(dyn_arch.qubit_to_zone_pos[right_qubit][0])
    midpoint = (
        ordered_zone_positions[left_zone] + ordered_zone_positions[right_zone]
    ) / 2
    return sorted(
        [
            zone
            for zone in ordered_zones
            if int(dyn_arch.zone_max_gate_cap[zone]) >= _WORKSPACE_ZONE_MIN_SIZE
        ],
        key=lambda zone: abs(ordered_zone_positions[zone] - midpoint),
    )


def _execute_adjacent_swap_sequence(
    dyn_arch: SgzlDynamicArch,
    swap_sequence: list[tuple[int, int]],
    target_gate_qubits: list[int],
) -> tuple[RoutingResult, bool]:
    routing_ops: list[RoutingOp] = []
    total_cost = 0.0
    for left_qubit, right_qubit in swap_sequence:
        if _target_gate_qubits_already_in_gate_zone(dyn_arch, target_gate_qubits):
            return RoutingResult(
                cost_estimate=total_cost, routing_ops=routing_ops
            ), True
        for workspace_zone in _workspace_candidate_zones(
            dyn_arch, left_qubit, right_qubit
        ):
            try:
                swap_result = execute_adjacent_swap_in_workspace(
                    dyn_arch, left_qubit, right_qubit, workspace_zone
                )
            except ValueError:
                continue
            total_cost += swap_result.cost_estimate
            _append_routing_ops(routing_ops, swap_result.routing_ops)
            break
        else:
            raise ValueError(
                "SingleGateZoneLineArchRouter could not realize a planned adjacent swap in any workspace zone."
            )
    return (
        RoutingResult(cost_estimate=total_cost, routing_ops=routing_ops),
        _target_gate_qubits_already_in_gate_zone(dyn_arch, target_gate_qubits),
    )


class SingleGateZoneLineArchRouter(Router):
    """Router specialized for linear architectures with a single gate zone."""

    def route_source_to_target_config(
        self,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> RoutingResult:
        sgzl_dyn_arch = require_sgzl_dynamic_arch(dyn_arch)
        _, target_gate_qubits = _validate_target_placement(
            sgzl_dyn_arch, target_placement
        )
        if not target_gate_qubits:
            return RoutingResult(cost_estimate=0, routing_ops=[])
        if _target_gate_qubits_already_in_gate_zone(sgzl_dyn_arch, target_gate_qubits):
            return RoutingResult(cost_estimate=0, routing_ops=[])

        if _swap_free_single_gate_zone_possible(sgzl_dyn_arch, target_gate_qubits):
            return _execute_swap_free_single_gate_zone_target(
                sgzl_dyn_arch, target_placement, target_gate_qubits
            )

        target_order = _class_assignment_target_order(sgzl_dyn_arch, target_gate_qubits)
        swap_sequence = _adjacent_swap_sequence(
            ordered_qubits(sgzl_dyn_arch), target_order
        )
        swap_result, reached_target_gate_zone = _execute_adjacent_swap_sequence(
            sgzl_dyn_arch, swap_sequence, target_gate_qubits
        )
        if reached_target_gate_zone:
            return swap_result

        final_result = _execute_swap_free_single_gate_zone_target(
            sgzl_dyn_arch, target_placement, target_gate_qubits
        )
        total_cost = swap_result.cost_estimate + final_result.cost_estimate
        routing_ops = swap_result.routing_ops.copy()
        _append_routing_ops(routing_ops, final_result.routing_ops)
        return RoutingResult(cost_estimate=total_cost, routing_ops=routing_ops)
