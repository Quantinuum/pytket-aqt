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
from heapq import heappop, heappush
from itertools import count, pairwise

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.architecture import PortId
from ...trap_architecture.dynamic_architecture import (
    DynamicArch,
    LinearDynamicArch,
    require_linear_dynamic_arch,
)
from ..routing_ops import PSwap, RoutingBarrier, RoutingOp, Shuttle
from .router import Router, RoutingResult

_WORKSPACE_SWAP_SIZE = 2


@dataclass(frozen=True)
class SwapFreeSegmentation:
    ordered_zones: list[int]
    block_sizes: list[int]
    zone_placement: ZonePlacement


@dataclass(frozen=True)
class SwapFreeFailureWitness:
    kind: str
    ordered_zones: list[int]
    left_qubit: int | None = None
    right_qubit: int | None = None
    left_zone: int | None = None
    right_zone: int | None = None
    boundary_index: int | None = None
    min_prefix_size: int | None = None
    max_prefix_size: int | None = None


@dataclass(frozen=True)
class SwapFreeRoutingAnalysis:
    segmentation: SwapFreeSegmentation | None
    failure_witness: SwapFreeFailureWitness | None


@dataclass(frozen=True)
class _LineArchContext:
    n_zones: int
    ordered_zones: tuple[int, ...]
    ordered_zone_positions: dict[int, int]
    zone_gate_capacities: dict[int, int]
    zone_transport_capacities: dict[int, int]
    zone_swap_costs: dict[int, int]
    boundary_ports: tuple[tuple[PortId, PortId], ...]
    boundary_shuttle_costs: tuple[int, ...]


@dataclass(frozen=True)
class _AbstractLineState:
    zone_placement: tuple[tuple[int, ...], ...]


@dataclass
class _SearchState:
    abstract_state: _AbstractLineState
    analysis: SwapFreeRoutingAnalysis
    cost_estimate: float
    depth: int
    parent: "_SearchState | None" = None
    last_swap: tuple[int, int, int] | None = None


def target_zone_interval_qubits(
    dyn_arch: LinearDynamicArch,
    target_placement: ZonePlacement,
) -> list[list[int]]:
    current_qubit_order = ordered_qubits(dyn_arch)
    qubit_to_index = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    target_interval_qubits: list[list[int]] = []

    for zone_qubits in target_placement:
        if not zone_qubits:
            target_interval_qubits.append([])
            continue

        target_indices = sorted(qubit_to_index[qubit] for qubit in zone_qubits)
        interval_qubits = current_qubit_order[
            target_indices[0] : target_indices[-1] + 1
        ]
        target_interval_qubits.append(interval_qubits)

    return target_interval_qubits


def ordered_qubits(dyn_arch: LinearDynamicArch) -> list[int]:
    return dyn_arch.ordered_qubits()


def _line_arch_context(dyn_arch: LinearDynamicArch) -> _LineArchContext:
    ordered_zones = dyn_arch.linearly_ordered_zones
    ordered_zone_positions = dyn_arch.ordered_zone_positions
    boundary_ports = tuple(
        (
            PortId(src_port_value),
            PortId(trg_port_value),
        )
        for left_zone, right_zone in pairwise(ordered_zones)
        for src_port_value, trg_port_value in [
            dyn_arch.connection_ports(left_zone, right_zone)
        ]
    )
    boundary_shuttle_costs = tuple(
        dyn_arch.shuttle_edge_transport_cost(left_zone, right_zone)
        for left_zone, right_zone in pairwise(ordered_zones)
    )
    return _LineArchContext(
        n_zones=dyn_arch.n_zones,
        ordered_zones=ordered_zones,
        ordered_zone_positions=ordered_zone_positions,
        zone_gate_capacities={
            zone: int(dyn_arch.zone_max_gate_cap[zone]) for zone in ordered_zones
        },
        zone_transport_capacities={
            zone: int(dyn_arch.zone_max_transport_cap[zone]) for zone in ordered_zones
        },
        zone_swap_costs={
            zone: int(dyn_arch.zone_swap_costs[zone]) for zone in ordered_zones
        },
        boundary_ports=boundary_ports,
        boundary_shuttle_costs=boundary_shuttle_costs,
    )


def _abstract_state_from_dyn_arch(dyn_arch: DynamicArch) -> _AbstractLineState:
    return _AbstractLineState(
        zone_placement=tuple(
            tuple(zone_qubits)
            for zone_qubits in dyn_arch.trap_configuration.zone_placement
        )
    )


def _ordered_qubits_state(
    state: _AbstractLineState, ordered_zones: tuple[int, ...]
) -> list[int]:
    return [qubit for zone in ordered_zones for qubit in state.zone_placement[zone]]


def analyze_swap_free_routing(
    dyn_arch: LinearDynamicArch,
    target_placement: ZonePlacement,
    fixed_block_sizes: dict[int, int] | None = None,
) -> SwapFreeRoutingAnalysis:
    ordered_zones = linearly_ordered_zones(dyn_arch)
    current_qubit_order = ordered_qubits(dyn_arch)
    zone_capacities = {
        zone: int(dyn_arch.zone_max_gate_cap[zone]) for zone in ordered_zones
    }
    return _analyze_swap_free_routing_from_order(
        current_qubit_order=current_qubit_order,
        ordered_zones=ordered_zones,
        zone_capacities=zone_capacities,
        target_placement=target_placement,
        fixed_block_sizes=fixed_block_sizes,
    )


def _analyze_swap_free_routing_state(
    context: _LineArchContext,
    state: _AbstractLineState,
    target_placement: ZonePlacement,
    fixed_block_sizes: dict[int, int] | None = None,
) -> SwapFreeRoutingAnalysis:
    return _analyze_swap_free_routing_from_order(
        current_qubit_order=_ordered_qubits_state(state, context.ordered_zones),
        ordered_zones=list(context.ordered_zones),
        zone_capacities=context.zone_gate_capacities,
        target_placement=target_placement,
        fixed_block_sizes=fixed_block_sizes,
    )


def _analyze_swap_free_routing_from_order(
    current_qubit_order: list[int],
    ordered_zones: list[int],
    zone_capacities: dict[int, int],
    target_placement: ZonePlacement,
    fixed_block_sizes: dict[int, int] | None = None,
) -> SwapFreeRoutingAnalysis:
    # In the swap-free case, each final zone must receive a contiguous block of the
    # current global qubit order. The DP tracks which prefixes can be assigned to
    # the first k zones while respecting capacities and any explicitly requested zones.
    ordered_zone_positions = {zone: i for i, zone in enumerate(ordered_zones)}
    n_qubits = len(current_qubit_order)
    n_zones = len(ordered_zones)
    fixed_block_sizes = {} if fixed_block_sizes is None else fixed_block_sizes

    required_zone_by_qubit = {
        qubit: zone
        for zone, zone_qubits in enumerate(target_placement)
        for qubit in zone_qubits
    }

    dp = [[False] * (n_qubits + 1) for _ in range(n_zones + 1)]
    predecessor: list[list[int | None]] = [
        [None] * (n_qubits + 1) for _ in range(n_zones + 1)
    ]
    dp[0][0] = True
    predecessor[0][0] = 0

    for zone_pos, zone in enumerate(ordered_zones):
        zone_capacity = zone_capacities[zone]
        fixed_block_size = fixed_block_sizes.get(zone)
        for prefix_length in range(n_qubits + 1):
            if not dp[zone_pos][prefix_length]:
                continue

            min_block_size = 0 if fixed_block_size is None else fixed_block_size
            max_block_size = (
                min(zone_capacity, n_qubits - prefix_length)
                if fixed_block_size is None
                else min(fixed_block_size, n_qubits - prefix_length)
            )

            if min_block_size == 0 and not dp[zone_pos + 1][prefix_length]:
                dp[zone_pos + 1][prefix_length] = True
                predecessor[zone_pos + 1][prefix_length] = prefix_length
            for block_size in range(max(min_block_size, 1), max_block_size + 1):
                qubit = current_qubit_order[prefix_length + block_size - 1]
                required_zone = required_zone_by_qubit.get(qubit)
                if (
                    required_zone is not None
                    and ordered_zone_positions[required_zone] != zone_pos
                ):
                    break
                new_prefix_length = prefix_length + block_size
                if not dp[zone_pos + 1][new_prefix_length]:
                    dp[zone_pos + 1][new_prefix_length] = True
                    predecessor[zone_pos + 1][new_prefix_length] = prefix_length

    if not dp[n_zones][n_qubits]:
        return SwapFreeRoutingAnalysis(
            segmentation=None,
            failure_witness=_first_failure_witness(
                ordered_zones,
                current_qubit_order,
                zone_capacities,
                required_zone_by_qubit,
                fixed_block_sizes,
            ),
        )

    # Reconstruct the chosen block size for each zone by walking the predecessor
    # pointers backwards from the full prefix.
    block_sizes = [0] * n_zones
    prefix_length = n_qubits
    for zone_pos in range(n_zones, 0, -1):
        previous_prefix_length = predecessor[zone_pos][prefix_length]
        if previous_prefix_length is None:
            raise ValueError("Missing predecessor for feasible swap-free segmentation.")
        block_sizes[zone_pos - 1] = prefix_length - previous_prefix_length
        prefix_length = previous_prefix_length

    zone_placement: ZonePlacement = [[] for _ in range(n_zones)]
    order_index = 0
    for zone_pos, zone in enumerate(ordered_zones):
        block_size = block_sizes[zone_pos]
        zone_placement[zone] = current_qubit_order[
            order_index : order_index + block_size
        ]
        order_index += block_size

    return SwapFreeRoutingAnalysis(
        segmentation=SwapFreeSegmentation(
            ordered_zones=ordered_zones,
            block_sizes=block_sizes,
            zone_placement=zone_placement,
        ),
        failure_witness=None,
    )


def swap_free_routing_segmentation(
    dyn_arch: DynamicArch,
    target_placement: ZonePlacement,
    fixed_block_sizes: dict[int, int] | None = None,
) -> SwapFreeSegmentation | None:
    return analyze_swap_free_routing(
        dyn_arch,
        target_placement,
        fixed_block_sizes=fixed_block_sizes,
    ).segmentation


def execute_swap_free_segmentation(
    dyn_arch: DynamicArch,
    segmentation: SwapFreeSegmentation,
) -> RoutingResult:
    if dyn_arch.trap_configuration.zone_placement == segmentation.zone_placement:
        return RoutingResult(cost_estimate=0, routing_ops=[])

    desired_block_sizes = segmentation.block_sizes
    ops: list[RoutingOp] = []
    total_cost = 0

    def append_shuttle(shuttle: Shuttle, shuttle_cost: int) -> None:
        nonlocal total_cost
        # Dependent shuttles within a linear cascade are not freely reorderable:
        # later shuttles often rely on the transport space created by earlier ones.
        # Keep each shuttle in its own move group so the circuit command order
        # matches the occupancy updates used while planning.
        if not ops:
            ops.append(RoutingBarrier())
        ops.extend([shuttle, RoutingBarrier()])
        total_cost += shuttle_cost

    while (
        _current_block_sizes(dyn_arch, segmentation.ordered_zones)
        != desired_block_sizes
    ):
        progress = False

        # First sweep right-to-left, pushing excess qubits to the right. This frees
        # space before we do the complementary left-to-right sweep.
        for boundary_index in range(len(segmentation.ordered_zones) - 2, -1, -1):
            # For a boundary after zone i, the prefix size is the number of qubits
            # currently living in zones [0, ..., i]. If that prefix is larger than
            # the target prefix size, the difference is the number of qubits that
            # must eventually cross this boundary to the right.
            excess = _current_prefix_size(
                dyn_arch, segmentation.ordered_zones, boundary_index
            ) - sum(desired_block_sizes[: boundary_index + 1])
            if excess > 0:
                shuttle_result = _shuttle_across_boundary(
                    dyn_arch,
                    segmentation.ordered_zones,
                    boundary_index,
                    move_right=True,
                    max_n_move=excess,
                )
                if shuttle_result is not None:
                    shuttle, shuttle_cost = shuttle_result
                    append_shuttle(shuttle, shuttle_cost)
                    progress = True

        # Then sweep left-to-right to pull qubits back into prefixes that are still
        # too small after the rightward moves have created space.
        for boundary_index in range(len(segmentation.ordered_zones) - 1):
            # Conversely, if the current prefix is smaller than the target prefix,
            # the deficiency is the number of qubits that still need to cross this
            # boundary from right to left.
            deficiency = sum(
                desired_block_sizes[: boundary_index + 1]
            ) - _current_prefix_size(
                dyn_arch, segmentation.ordered_zones, boundary_index
            )
            if deficiency > 0:
                shuttle_result = _shuttle_across_boundary(
                    dyn_arch,
                    segmentation.ordered_zones,
                    boundary_index,
                    move_right=False,
                    max_n_move=deficiency,
                )
                if shuttle_result is not None:
                    shuttle, shuttle_cost = shuttle_result
                    append_shuttle(shuttle, shuttle_cost)
                    progress = True

        if not progress:
            raise ValueError(
                "Could not execute swap-free segmentation with shuttles only."
            )

    return RoutingResult(cost_estimate=total_cost, routing_ops=ops)


def execute_adjacent_swap_in_workspace(
    dyn_arch: DynamicArch,
    left_qubit: int,
    right_qubit: int,
    workspace_zone: int,
) -> RoutingResult:
    current_qubit_order = ordered_qubits(dyn_arch)
    qubit_positions = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    if qubit_positions[right_qubit] - qubit_positions[left_qubit] != 1:
        raise ValueError("Workspace swaps require adjacent qubits in the global order.")

    target_placement: ZonePlacement = [[] for _ in range(dyn_arch.n_zones)]
    target_placement[workspace_zone] = [left_qubit, right_qubit]
    segmentation = swap_free_routing_segmentation(
        dyn_arch,
        target_placement,
        fixed_block_sizes={workspace_zone: 2},
    )
    if segmentation is None:
        raise ValueError(
            "Could not isolate the adjacent qubit pair in the chosen workspace zone."
        )

    move_result = execute_swap_free_segmentation(dyn_arch, segmentation)
    routing_ops = move_result.routing_ops.copy()
    if not routing_ops or not isinstance(routing_ops[-1], RoutingBarrier):
        routing_ops.append(RoutingBarrier())

    dyn_arch.swap_qubits_in_zone(workspace_zone, left_qubit, right_qubit)
    routing_ops.append(PSwap(workspace_zone, left_qubit, right_qubit))
    routing_ops.append(RoutingBarrier())
    return RoutingResult(
        cost_estimate=move_result.cost_estimate
        + int(dyn_arch.zone_swap_costs[workspace_zone]),
        routing_ops=routing_ops,
    )


def _current_block_sizes_state(
    state: _AbstractLineState, ordered_zones: tuple[int, ...]
) -> list[int]:
    return [len(state.zone_placement[zone]) for zone in ordered_zones]


def _current_prefix_size_state(
    state: _AbstractLineState, ordered_zones: tuple[int, ...], boundary_index: int
) -> int:
    return sum(
        len(state.zone_placement[zone]) for zone in ordered_zones[: boundary_index + 1]
    )


def _shuttle_across_boundary_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    boundary_index: int,
    move_right: bool,
    max_n_move: int,
) -> tuple[_AbstractLineState, int] | None:
    if move_right:
        src_zone = context.ordered_zones[boundary_index]
        trg_zone = context.ordered_zones[boundary_index + 1]
        src_port, trg_port = context.boundary_ports[boundary_index]
    else:
        src_zone = context.ordered_zones[boundary_index + 1]
        trg_zone = context.ordered_zones[boundary_index]
        trg_port, src_port = context.boundary_ports[boundary_index]

    max_batch_size = min(
        max_n_move,
        context.zone_transport_capacities[trg_zone]
        - len(state.zone_placement[trg_zone]),
    )
    if max_batch_size < 1:
        return None

    src_zone_qubits = state.zone_placement[src_zone]
    if not src_zone_qubits:
        return None

    if src_port == PortId.p0:
        qubits = src_zone_qubits[:max_batch_size]
        new_src_zone_qubits = src_zone_qubits[max_batch_size:]
    else:
        qubits = src_zone_qubits[-max_batch_size:]
        new_src_zone_qubits = src_zone_qubits[:-max_batch_size]

    trg_zone_qubits = state.zone_placement[trg_zone]
    new_trg_zone_qubits = (
        qubits + trg_zone_qubits if trg_port == PortId.p0 else trg_zone_qubits + qubits
    )
    zone_placement = list(state.zone_placement)
    zone_placement[src_zone] = new_src_zone_qubits
    zone_placement[trg_zone] = new_trg_zone_qubits
    return (
        _AbstractLineState(zone_placement=tuple(zone_placement)),
        context.boundary_shuttle_costs[boundary_index],
    )


def _execute_swap_free_segmentation_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    segmentation: SwapFreeSegmentation,
) -> tuple[_AbstractLineState, int]:
    if [
        list(zone_qubits) for zone_qubits in state.zone_placement
    ] == segmentation.zone_placement:
        return state, 0

    desired_block_sizes = segmentation.block_sizes
    total_cost = 0
    current_state = state

    while (
        _current_block_sizes_state(current_state, context.ordered_zones)
        != desired_block_sizes
    ):
        progress = False

        for boundary_index in range(len(context.ordered_zones) - 2, -1, -1):
            excess = _current_prefix_size_state(
                current_state, context.ordered_zones, boundary_index
            ) - sum(desired_block_sizes[: boundary_index + 1])
            if excess > 0:
                shuttle_result = _shuttle_across_boundary_state(
                    current_state,
                    context,
                    boundary_index,
                    move_right=True,
                    max_n_move=excess,
                )
                if shuttle_result is not None:
                    current_state, shuttle_cost = shuttle_result
                    total_cost += shuttle_cost
                    progress = True

        for boundary_index in range(len(context.ordered_zones) - 1):
            deficiency = sum(
                desired_block_sizes[: boundary_index + 1]
            ) - _current_prefix_size_state(
                current_state, context.ordered_zones, boundary_index
            )
            if deficiency > 0:
                shuttle_result = _shuttle_across_boundary_state(
                    current_state,
                    context,
                    boundary_index,
                    move_right=False,
                    max_n_move=deficiency,
                )
                if shuttle_result is not None:
                    current_state, shuttle_cost = shuttle_result
                    total_cost += shuttle_cost
                    progress = True

        if not progress:
            raise ValueError(
                "Could not execute swap-free segmentation with shuttles only."
            )

    return current_state, total_cost


def _execute_adjacent_swap_in_workspace_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    left_qubit: int,
    right_qubit: int,
    workspace_zone: int,
) -> tuple[_AbstractLineState, int]:
    current_qubit_order = _ordered_qubits_state(state, context.ordered_zones)
    qubit_positions = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    if qubit_positions[right_qubit] - qubit_positions[left_qubit] != 1:
        raise ValueError("Workspace swaps require adjacent qubits in the global order.")

    target_placement: ZonePlacement = [[] for _ in range(context.n_zones)]
    target_placement[workspace_zone] = [left_qubit, right_qubit]
    segmentation = _analyze_swap_free_routing_state(
        context,
        state,
        target_placement,
        fixed_block_sizes={workspace_zone: _WORKSPACE_SWAP_SIZE},
    ).segmentation
    if segmentation is None:
        raise ValueError(
            "Could not isolate the adjacent qubit pair in the chosen workspace zone."
        )

    isolated_state, move_cost = _execute_swap_free_segmentation_state(
        state, context, segmentation
    )
    workspace_qubits = isolated_state.zone_placement[workspace_zone]
    if workspace_qubits != (left_qubit, right_qubit):
        raise ValueError(
            "Abstract workspace swap did not isolate the requested pair in the expected order."
        )

    zone_placement = list(isolated_state.zone_placement)
    zone_placement[workspace_zone] = (right_qubit, left_qubit)
    return (
        _AbstractLineState(zone_placement=tuple(zone_placement)),
        move_cost + context.zone_swap_costs[workspace_zone],
    )


def _current_block_sizes(dyn_arch: DynamicArch, ordered_zones: list[int]) -> list[int]:
    return [
        len(dyn_arch.trap_configuration.zone_placement[zone]) for zone in ordered_zones
    ]


def _current_prefix_size(
    dyn_arch: DynamicArch, ordered_zones: list[int], boundary_index: int
) -> int:
    # Return the number of qubits currently contained in the prefix of the linear
    # architecture up to and including ordered_zones[boundary_index]. Comparing
    # this with the target prefix size tells us how many qubits still need to
    # cross that boundary, and in which direction.
    return sum(
        len(dyn_arch.trap_configuration.zone_placement[zone])
        for zone in ordered_zones[: boundary_index + 1]
    )


def _shuttle_across_boundary(
    dyn_arch: DynamicArch,
    ordered_zones: list[int],
    boundary_index: int,
    move_right: bool,
    max_n_move: int,
) -> tuple[Shuttle, int] | None:
    if move_right:
        src_zone = ordered_zones[boundary_index]
        trg_zone = ordered_zones[boundary_index + 1]
    else:
        src_zone = ordered_zones[boundary_index + 1]
        trg_zone = ordered_zones[boundary_index]

    max_batch_size = min(
        max_n_move,
        int(dyn_arch.transport_free_space[trg_zone]),
    )
    if max_batch_size < 1:
        return None

    src_zone_qubits = dyn_arch.trap_configuration.zone_placement[src_zone]
    if not src_zone_qubits:
        return None

    src_port_value, trg_port_value = dyn_arch.connection_ports(src_zone, trg_zone)
    src_port = PortId(src_port_value)
    trg_port = PortId(trg_port_value)
    # In a linear architecture, the only qubits that can cross a boundary without
    # additional swaps are the contiguous qubits already sitting at the relevant edge port.
    qubits = (
        src_zone_qubits[:max_batch_size]
        if src_port == PortId.p0
        else src_zone_qubits[-max_batch_size:]
    )
    shuttle_cost = dyn_arch.shuttle_edge_transport_cost(src_zone, trg_zone)
    dyn_arch.move_qubits(qubits, src_zone, trg_zone, trg_port.value)
    return Shuttle(qubits.copy(), src_zone, trg_zone, src_port, trg_port), shuttle_cost


def _candidate_adjacent_pairs(
    dyn_arch: DynamicArch, analysis: SwapFreeRoutingAnalysis
) -> list[tuple[int, int]]:
    return _candidate_adjacent_pairs_from_order(ordered_qubits(dyn_arch), analysis)


def _candidate_adjacent_pairs_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    analysis: SwapFreeRoutingAnalysis,
) -> list[tuple[int, int]]:
    return _candidate_adjacent_pairs_from_order(
        _ordered_qubits_state(state, context.ordered_zones), analysis
    )


def _candidate_adjacent_pairs_from_order(
    current_qubit_order: list[int],
    analysis: SwapFreeRoutingAnalysis,
) -> list[tuple[int, int]]:
    all_pairs = [
        (current_qubit_order[i], current_qubit_order[i + 1])
        for i in range(len(current_qubit_order) - 1)
    ]
    witness = analysis.failure_witness
    if witness is None:
        return all_pairs

    prioritized_pairs: list[tuple[int, int]] = []
    qubit_positions = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    if witness.kind == "inversion":
        if witness.left_qubit is not None:
            left_position = qubit_positions[witness.left_qubit]
            if left_position + 1 < len(current_qubit_order):
                prioritized_pairs.append(
                    (
                        current_qubit_order[left_position],
                        current_qubit_order[left_position + 1],
                    )
                )
        if witness.right_qubit is not None:
            right_position = qubit_positions[witness.right_qubit]
            if right_position > 0:
                prioritized_pairs.append(
                    (
                        current_qubit_order[right_position - 1],
                        current_qubit_order[right_position],
                    )
                )
    elif witness.kind == "boundary":
        boundary_positions = [
            witness.max_prefix_size,
            None if witness.min_prefix_size is None else witness.min_prefix_size - 1,
        ]
        prioritized_pairs.extend(
            [
                (
                    current_qubit_order[boundary_position],
                    current_qubit_order[boundary_position + 1],
                )
                for boundary_position in boundary_positions
                if (
                    boundary_position is not None
                    and 0 <= boundary_position < len(current_qubit_order) - 1
                )
            ]
        )

    return list(dict.fromkeys(prioritized_pairs + all_pairs))


def _workspace_candidate_zones(
    dyn_arch: DynamicArch, left_qubit: int, right_qubit: int
) -> list[int]:
    return _workspace_candidate_zones_from_zone_positions(
        _ordered_zone_positions(dyn_arch),
        {
            qubit: int(zone_pos[0])
            for qubit, zone_pos in enumerate(dyn_arch.qubit_to_zone_pos)
        },
        {
            zone: int(dyn_arch.zone_max_gate_cap[zone])
            for zone in linearly_ordered_zones(dyn_arch)
        },
        left_qubit,
        right_qubit,
    )


def _workspace_candidate_zones_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    left_qubit: int,
    right_qubit: int,
) -> list[int]:
    return _workspace_candidate_zones_from_zone_positions(
        context.ordered_zone_positions,
        _qubit_to_zone_state(state),
        context.zone_gate_capacities,
        left_qubit,
        right_qubit,
    )


def _workspace_candidate_zones_from_zone_positions(
    ordered_zone_positions: dict[int, int],
    qubit_to_zone: dict[int, int],
    zone_gate_capacities: dict[int, int],
    left_qubit: int,
    right_qubit: int,
) -> list[int]:
    left_zone = qubit_to_zone[left_qubit]
    right_zone = qubit_to_zone[right_qubit]
    midpoint = (
        ordered_zone_positions[left_zone] + ordered_zone_positions[right_zone]
    ) / 2
    return sorted(
        [
            zone
            for zone in ordered_zone_positions
            if zone_gate_capacities[zone] >= _WORKSPACE_SWAP_SIZE
        ],
        key=lambda zone: abs(ordered_zone_positions[zone] - midpoint),
    )


def _analysis_score(
    dyn_arch: DynamicArch, analysis: SwapFreeRoutingAnalysis
) -> tuple[int, int]:
    kind_rank = 3
    severity = 0
    if analysis.segmentation is not None:
        kind_rank = 0
    else:
        witness = analysis.failure_witness
        if witness is not None and witness.kind == "inversion":
            kind_rank = 1
            if witness.left_qubit is not None and witness.right_qubit is not None:
                qubit_positions = {
                    qubit: i for i, qubit in enumerate(ordered_qubits(dyn_arch))
                }
                severity = (
                    qubit_positions[witness.right_qubit]
                    - qubit_positions[witness.left_qubit]
                    - 1
                )
        elif witness is not None and witness.kind == "boundary":
            kind_rank = 2
            if (
                witness.min_prefix_size is not None
                and witness.max_prefix_size is not None
            ):
                severity = witness.min_prefix_size - witness.max_prefix_size
    return (kind_rank, severity)


def _search_priority(
    dyn_arch: DynamicArch,
    target_placement: ZonePlacement,
    analysis: SwapFreeRoutingAnalysis,
    cost_estimate: float,
    depth: int,
) -> tuple[int, int, int, int, int, float, int]:
    inversion_count, inversion_span = _target_order_inversion_metrics(
        dyn_arch, target_placement
    )
    zone_distance = _target_zone_distance(dyn_arch, target_placement)
    boundary_gap = 0
    witness = analysis.failure_witness
    if (
        witness is not None
        and witness.kind == "boundary"
        and witness.min_prefix_size is not None
        and witness.max_prefix_size is not None
    ):
        boundary_gap = witness.min_prefix_size - witness.max_prefix_size
    return (
        0 if analysis.segmentation is not None else 1,
        inversion_count,
        inversion_span,
        boundary_gap,
        zone_distance,
        cost_estimate,
        depth,
    )


def _search_priority_state(
    search_state: _SearchState,
    context: _LineArchContext,
    target_placement: ZonePlacement,
) -> tuple[int, int, int, int, int, float, int]:
    inversion_count, inversion_span = _target_order_inversion_metrics_state(
        search_state.abstract_state, context, target_placement
    )
    zone_distance = _target_zone_distance_state(
        search_state.abstract_state, context, target_placement
    )
    boundary_gap = 0
    witness = search_state.analysis.failure_witness
    if (
        witness is not None
        and witness.kind == "boundary"
        and witness.min_prefix_size is not None
        and witness.max_prefix_size is not None
    ):
        boundary_gap = witness.min_prefix_size - witness.max_prefix_size
    return (
        0 if search_state.analysis.segmentation is not None else 1,
        inversion_count,
        inversion_span,
        boundary_gap,
        zone_distance,
        search_state.cost_estimate,
        search_state.depth,
    )


def _target_order_inversion_metrics(
    dyn_arch: DynamicArch, target_placement: ZonePlacement
) -> tuple[int, int]:
    target_zone_positions = _target_zone_positions(dyn_arch, target_placement)
    ordered_target_positions = [
        target_zone_positions[qubit]
        for qubit in ordered_qubits(dyn_arch)
        if qubit in target_zone_positions
    ]

    inversion_count = 0
    inversion_span = 0
    for left_index, left_zone_position in enumerate(ordered_target_positions):
        for right_index in range(left_index + 1, len(ordered_target_positions)):
            right_zone_position = ordered_target_positions[right_index]
            if left_zone_position > right_zone_position:
                inversion_count += 1
                inversion_span += right_index - left_index
    return inversion_count, inversion_span


def _target_order_inversion_metrics_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    target_placement: ZonePlacement,
) -> tuple[int, int]:
    target_zone_positions = _target_zone_positions_from_order(
        context.ordered_zone_positions, target_placement
    )
    ordered_target_positions = [
        target_zone_positions[qubit]
        for qubit in _ordered_qubits_state(state, context.ordered_zones)
        if qubit in target_zone_positions
    ]
    return _ordered_target_inversion_metrics(ordered_target_positions)


def _ordered_target_inversion_metrics(
    ordered_target_positions: list[int],
) -> tuple[int, int]:
    inversion_count = 0
    inversion_span = 0
    for left_index, left_zone_position in enumerate(ordered_target_positions):
        for right_index in range(left_index + 1, len(ordered_target_positions)):
            right_zone_position = ordered_target_positions[right_index]
            if left_zone_position > right_zone_position:
                inversion_count += 1
                inversion_span += right_index - left_index
    return inversion_count, inversion_span


def _target_zone_distance(
    dyn_arch: DynamicArch, target_placement: ZonePlacement
) -> int:
    ordered_zone_positions = _ordered_zone_positions(dyn_arch)
    target_zone_positions = _target_zone_positions(dyn_arch, target_placement)
    return sum(
        abs(
            ordered_zone_positions[int(dyn_arch.qubit_to_zone_pos[qubit][0])]
            - target_zone_positions[qubit]
        )
        for qubit in target_zone_positions
    )


def _target_zone_distance_state(
    state: _AbstractLineState,
    context: _LineArchContext,
    target_placement: ZonePlacement,
) -> int:
    qubit_to_zone = _qubit_to_zone_state(state)
    target_zone_positions = _target_zone_positions_from_order(
        context.ordered_zone_positions, target_placement
    )
    return sum(
        abs(
            context.ordered_zone_positions[qubit_to_zone[qubit]]
            - target_zone_positions[qubit]
        )
        for qubit in target_zone_positions
    )


def _target_zone_positions(
    dyn_arch: DynamicArch, target_placement: ZonePlacement
) -> dict[int, int]:
    return _target_zone_positions_from_order(
        _ordered_zone_positions(dyn_arch), target_placement
    )


def _target_zone_positions_from_order(
    ordered_zone_positions: dict[int, int], target_placement: ZonePlacement
) -> dict[int, int]:
    return {
        qubit: ordered_zone_positions[zone]
        for zone, zone_qubits in enumerate(target_placement)
        for qubit in zone_qubits
    }


def _qubit_to_zone_state(state: _AbstractLineState) -> dict[int, int]:
    return {
        qubit: zone
        for zone, zone_qubits in enumerate(state.zone_placement)
        for qubit in zone_qubits
    }


def _ordered_zone_positions(dyn_arch: DynamicArch) -> dict[int, int]:
    return {zone: i for i, zone in enumerate(linearly_ordered_zones(dyn_arch))}


def _placement_signature(zone_placement: ZonePlacement) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(zone_qubits) for zone_qubits in zone_placement)


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


def _workspace_swap_successors(
    context: _LineArchContext,
    state: _SearchState,
    target_placement: ZonePlacement,
    best_cost_by_signature: dict[_AbstractLineState, float],
    max_candidates: int,
) -> list[tuple[tuple[int, int, int, int, int, float, int], _SearchState]]:
    successors_by_signature: dict[
        _AbstractLineState,
        tuple[tuple[int, int, int, int, int, float, int], _SearchState],
    ] = {}

    for left_qubit, right_qubit in _candidate_adjacent_pairs_state(
        state.abstract_state, context, state.analysis
    ):
        for workspace_zone in _workspace_candidate_zones_state(
            state.abstract_state, context, left_qubit, right_qubit
        ):
            try:
                next_abstract_state, swap_cost = (
                    _execute_adjacent_swap_in_workspace_state(
                        state.abstract_state,
                        context,
                        left_qubit,
                        right_qubit,
                        workspace_zone,
                    )
                )
            except ValueError:
                continue

            next_cost = state.cost_estimate + swap_cost
            if next_cost >= best_cost_by_signature.get(
                next_abstract_state, float("inf")
            ):
                continue

            next_analysis = _analyze_swap_free_routing_state(
                context, next_abstract_state, target_placement
            )
            next_state = _SearchState(
                abstract_state=next_abstract_state,
                analysis=next_analysis,
                cost_estimate=next_cost,
                depth=state.depth + 1,
                parent=state,
                last_swap=(left_qubit, right_qubit, workspace_zone),
            )
            priority = _search_priority_state(next_state, context, target_placement)
            previous_successor = successors_by_signature.get(next_abstract_state)
            if previous_successor is None or (priority, next_cost) < (
                previous_successor[0],
                previous_successor[1].cost_estimate,
            ):
                successors_by_signature[next_abstract_state] = (priority, next_state)

    prioritized_successors = sorted(
        successors_by_signature.values(),
        key=lambda successor: (successor[0], successor[1].cost_estimate),
    )[:max_candidates]
    for _, successor_state in prioritized_successors:
        best_cost_by_signature[successor_state.abstract_state] = (
            successor_state.cost_estimate
        )
    return prioritized_successors


def _search_routing_solution(
    context: _LineArchContext,
    initial_state: _AbstractLineState,
    target_placement: ZonePlacement,
) -> _SearchState | None:
    n_qubits = len(_ordered_qubits_state(initial_state, context.ordered_zones))
    max_depth = max(1, n_qubits**2)
    max_expanded_states = min(10000, max(128, n_qubits**5))
    max_candidates_per_state = min(24, max(6, n_qubits * 2))

    start_analysis = _analyze_swap_free_routing_state(
        context, initial_state, target_placement
    )
    start_state = _SearchState(
        abstract_state=initial_state,
        analysis=start_analysis,
        cost_estimate=0.0,
        depth=0,
    )
    best_cost_by_signature = {initial_state: 0.0}
    frontier: list[
        tuple[tuple[int, int, int, int, int, float, int], int, _SearchState]
    ] = []
    tie_breaker = count()
    heappush(
        frontier,
        (
            _search_priority_state(start_state, context, target_placement),
            next(tie_breaker),
            start_state,
        ),
    )

    expanded_states = 0
    while frontier:
        _, _, state = heappop(frontier)
        if state.cost_estimate > best_cost_by_signature.get(
            state.abstract_state, float("inf")
        ):
            continue

        if state.analysis.segmentation is not None:
            return state

        if state.depth >= max_depth:
            continue

        expanded_states += 1
        if expanded_states > max_expanded_states:
            raise ValueError(
                "LineArchRouter exceeded the bounded workspace-swap search limit without reaching a swap-free segmentation."
            )

        for priority, next_state in _workspace_swap_successors(
            context,
            state,
            target_placement,
            best_cost_by_signature,
            max_candidates_per_state,
        ):
            heappush(frontier, (priority, next(tie_breaker), next_state))

    return None


def _reconstruct_swap_sequence(solution: _SearchState) -> list[tuple[int, int, int]]:
    swap_sequence: list[tuple[int, int, int]] = []
    state = solution
    while state.parent is not None:
        if state.last_swap is None:
            raise ValueError(
                "Search state with parent is missing its workspace-swap action."
            )
        swap_sequence.append(state.last_swap)
        state = state.parent
    swap_sequence.reverse()
    return swap_sequence


def _first_failure_witness(
    ordered_zones: list[int],
    current_qubit_order: list[int],
    zone_capacities: dict[int, int],
    required_zone_by_qubit: dict[int, int],
    fixed_block_sizes: dict[int, int],
) -> SwapFreeFailureWitness:
    ordered_zone_positions = {zone: i for i, zone in enumerate(ordered_zones)}
    previous_specified_qubit: int | None = None
    previous_specified_zone_position: int | None = None
    for qubit in current_qubit_order:
        required_zone = required_zone_by_qubit.get(qubit)
        if required_zone is None:
            continue
        zone_position = ordered_zone_positions[required_zone]
        if (
            previous_specified_zone_position is not None
            and zone_position < previous_specified_zone_position
        ):
            return SwapFreeFailureWitness(
                kind="inversion",
                ordered_zones=ordered_zones,
                left_qubit=previous_specified_qubit,
                right_qubit=qubit,
                left_zone=required_zone_by_qubit[previous_specified_qubit],  # type: ignore[index]
                right_zone=required_zone,
            )
        previous_specified_qubit = qubit
        previous_specified_zone_position = zone_position

    n_qubits = len(current_qubit_order)
    qubit_positions = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    for boundary_index in range(len(ordered_zones) - 1):
        left_zones = ordered_zones[: boundary_index + 1]
        right_zones = ordered_zones[boundary_index + 1 :]

        left_fixed = sum(fixed_block_sizes.get(zone, 0) for zone in left_zones)
        right_fixed = sum(fixed_block_sizes.get(zone, 0) for zone in right_zones)
        left_flexible_capacity = sum(
            zone_capacities[zone]
            for zone in left_zones
            if zone not in fixed_block_sizes
        )
        right_flexible_capacity = sum(
            zone_capacities[zone]
            for zone in right_zones
            if zone not in fixed_block_sizes
        )
        min_prefix_size = max(
            left_fixed, n_qubits - (right_fixed + right_flexible_capacity)
        )
        max_prefix_size = min(
            left_fixed + left_flexible_capacity,
            n_qubits - right_fixed,
        )

        left_target_positions = [
            qubit_positions[qubit]
            for qubit, zone in required_zone_by_qubit.items()
            if ordered_zone_positions[zone] <= boundary_index
        ]
        right_target_positions = [
            qubit_positions[qubit]
            for qubit, zone in required_zone_by_qubit.items()
            if ordered_zone_positions[zone] > boundary_index
        ]
        if left_target_positions:
            min_prefix_size = max(min_prefix_size, max(left_target_positions) + 1)
        if right_target_positions:
            max_prefix_size = min(max_prefix_size, *right_target_positions)

        if min_prefix_size > max_prefix_size:
            return SwapFreeFailureWitness(
                kind="boundary",
                ordered_zones=ordered_zones,
                boundary_index=boundary_index,
                min_prefix_size=min_prefix_size,
                max_prefix_size=max_prefix_size,
            )

    return SwapFreeFailureWitness(kind="unknown", ordered_zones=ordered_zones)


def linearly_ordered_zones(dyn_arch: LinearDynamicArch) -> list[int]:
    return list(dyn_arch.linearly_ordered_zones)


def line_start_zone(dyn_arch: LinearDynamicArch) -> int:
    return dyn_arch.line_start_zone


class LineArchRouter(Router):
    """Router specialized for linear multi-zone architectures."""

    def route_source_to_target_config(
        self, dyn_arch: DynamicArch, target_placement: ZonePlacement
    ) -> RoutingResult:
        linear_dyn_arch = require_linear_dynamic_arch(dyn_arch)
        target_zone_interval_qubits(linear_dyn_arch, target_placement)
        context = _line_arch_context(linear_dyn_arch)
        solution = _search_routing_solution(
            context,
            _abstract_state_from_dyn_arch(linear_dyn_arch),
            target_placement,
        )
        if solution is None:
            raise ValueError(
                "LineArchRouter could not find a legal adjacent workspace swap to repair the target placement."
            )

        total_cost = 0.0
        routing_ops: list[RoutingOp] = []
        for left_qubit, right_qubit, workspace_zone in _reconstruct_swap_sequence(
            solution
        ):
            repair_result = execute_adjacent_swap_in_workspace(
                linear_dyn_arch, left_qubit, right_qubit, workspace_zone
            )
            total_cost += repair_result.cost_estimate
            _append_routing_ops(routing_ops, repair_result.routing_ops)

        final_analysis = analyze_swap_free_routing(linear_dyn_arch, target_placement)
        if final_analysis.segmentation is None:
            raise ValueError(
                "LineArchRouter replay did not reproduce a swap-free state for the final target placement."
            )
        final_result = execute_swap_free_segmentation(
            linear_dyn_arch, final_analysis.segmentation
        )
        total_cost += final_result.cost_estimate
        _append_routing_ops(routing_ops, final_result.routing_ops)
        return RoutingResult(total_cost, routing_ops)
