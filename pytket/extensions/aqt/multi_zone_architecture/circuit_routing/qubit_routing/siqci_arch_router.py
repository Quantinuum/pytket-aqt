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
from itertools import count

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.architecture import PortId
from ...trap_architecture.dynamic_architecture import (
    DynamicArch,
    SgzlDynamicArch,
    require_sgzl_dynamic_arch,
)
from ..routing_ops import PSwap, RoutingBarrier, RoutingOp, Shuttle
from .router import Router, RoutingResult

_PAIRED_ZONE_OCCUPANCY = 2
_INTERNAL_ZONE_MIN_INDEX = 1
_SIQCI_N_ZONES = 5
_SIQCI_N_QUBITS_MAX = 5


@dataclass(frozen=True)
class _SiqciPswapAction:
    """A legal siqci physical swap in a doubly occupied zone."""

    zone: int
    qubit0: int
    qubit1: int
    cost: float


@dataclass(frozen=True)
class _SiqciShuttleAction:
    """A legal siqci shuttle between adjacent zones."""

    qubits: tuple[int, ...]
    src_zone: int
    targ_zone: int
    src_port: PortId
    targ_port: PortId
    cost: float


@dataclass(frozen=True)
class _RouteStep:
    previous_state: tuple[tuple[int, ...], ...]
    action: _SiqciPswapAction | _SiqciShuttleAction


def _validate_siqci_architecture(dyn_arch: SgzlDynamicArch) -> None:
    if (
        dyn_arch.n_zones != _SIQCI_N_ZONES
        or dyn_arch.architecture_spec.n_qubits_max != _SIQCI_N_QUBITS_MAX
        or dyn_arch.gate_capacity != _PAIRED_ZONE_OCCUPANCY
        or any(
            capacity != _PAIRED_ZONE_OCCUPANCY
            for capacity in dyn_arch.zone_max_gate_cap
        )
        or any(
            capacity != _PAIRED_ZONE_OCCUPANCY
            for capacity in dyn_arch.zone_max_transport_cap
        )
    ):
        raise ValueError(
            "SiqciArchRouter can only be used with siqci-like linear architectures with one gate zone and capacity-2 zones."
        )


def _validate_target_placement(
    dyn_arch: SgzlDynamicArch, target_placement: ZonePlacement
) -> list[int]:
    """Return the requested gate-zone occupants after checking the siqci contract.

    The specialized siqci gate selectors may specify either the next two qubits
    that should occupy the single gate zone or a single qubit that must be
    routed there with the cheapest legal partner. All other zones must remain
    unspecified here.
    """
    gate_zone = dyn_arch.single_gate_zone
    for zone, zone_qubits in enumerate(target_placement):
        if zone != gate_zone and zone_qubits:
            raise ValueError(
                "SiqciArchRouter requires target placements to specify qubits only in the gate zone."
            )
    target_gate_qubits = target_placement[gate_zone]
    if len(target_gate_qubits) not in (1, _PAIRED_ZONE_OCCUPANCY):
        raise ValueError(
            "SiqciArchRouter target placement must specify one or two qubits in the gate zone."
        )
    return target_gate_qubits


def _state_from_dyn_arch(dyn_arch: SgzlDynamicArch) -> tuple[tuple[int, ...], ...]:
    """Encode the current machine state as immutable per-zone qubit tuples."""
    return tuple(
        tuple(zone_qubits) for zone_qubits in dyn_arch.trap_configuration.zone_placement
    )


def _edge_qubits(
    zone_qubits: tuple[int, ...], src_port: PortId, n_move: int
) -> tuple[int, ...]:
    """Return the qubits that can leave through the chosen port.

    In a line architecture, only the qubits already sitting at the relevant edge
    of the zone can shuttle across that connection without an intermediate swap.
    """
    if src_port == PortId.p0:
        return zone_qubits[:n_move]
    return zone_qubits[-n_move:]


def _shuttle_state(
    state: tuple[tuple[int, ...], ...],
    action: _SiqciShuttleAction,
) -> tuple[tuple[int, ...], ...]:
    zone_placement = [list(zone_qubits) for zone_qubits in state]
    src_zone_qubits = zone_placement[action.src_zone]
    for qubit in action.qubits:
        src_zone_qubits.remove(qubit)
    if action.targ_port == PortId.p0:
        zone_placement[action.targ_zone] = (
            list(action.qubits) + zone_placement[action.targ_zone]
        )
    else:
        zone_placement[action.targ_zone].extend(action.qubits)
    return tuple(tuple(zone_qubits) for zone_qubits in zone_placement)


def _pswap_state(
    state: tuple[tuple[int, ...], ...], action: _SiqciPswapAction
) -> tuple[tuple[int, ...], ...]:
    zone_placement = [list(zone_qubits) for zone_qubits in state]
    zone_placement[action.zone].reverse()
    return tuple(tuple(zone_qubits) for zone_qubits in zone_placement)


def _is_internal_zone(dyn_arch: SgzlDynamicArch, zone: int) -> bool:
    zone_position = dyn_arch.ordered_zone_positions[zone]
    return (
        _INTERNAL_ZONE_MIN_INDEX
        <= zone_position
        < len(dyn_arch.linearly_ordered_zones) - 1
    )


def _adjacent_zones_empty(
    dyn_arch: SgzlDynamicArch,
    state: tuple[tuple[int, ...], ...],
    zone: int,
) -> bool:
    zone_position = dyn_arch.ordered_zone_positions[zone]
    left_zone = dyn_arch.linearly_ordered_zones[zone_position - 1]
    right_zone = dyn_arch.linearly_ordered_zones[zone_position + 1]
    return not state[left_zone] and not state[right_zone]


def _shuttle_actions(
    dyn_arch: SgzlDynamicArch,
    state: tuple[tuple[int, ...], ...],
    split_penalty: float,
    merge_penalty: float,
) -> list[_SiqciShuttleAction]:
    """List all legal shuttles from the current siqci state.

    Legal shuttles adhere to the following rules:
    - A full 2-ion chain may shuttle only into an adjacent empty zone.
    - A lone ion may shuttle into an adjacent empty or singly occupied zone.
    - Splitting one ion away from a 2-ion chain is only allowed in internal
      zones and only when both adjacent zones are empty.

    The shuttle cost is calculated as the transport edge cost,
     plus possible split or merge penalties.
    """
    actions: list[_SiqciShuttleAction] = []
    ordered_zones = dyn_arch.linearly_ordered_zones
    for zone_position, src_zone in enumerate(ordered_zones):
        src_zone_qubits = state[src_zone]
        src_occupancy = len(src_zone_qubits)
        if src_occupancy == 0:
            continue
        for neighbour_position in (zone_position - 1, zone_position + 1):
            if not 0 <= neighbour_position < len(ordered_zones):
                continue
            targ_zone = ordered_zones[neighbour_position]
            targ_occupancy = len(state[targ_zone])
            src_port_value, targ_port_value = dyn_arch.connection_ports(
                src_zone, targ_zone
            )
            src_port = PortId(src_port_value)
            targ_port = PortId(targ_port_value)
            transport_cost = float(
                dyn_arch.shuttle_edge_transport_cost(src_zone, targ_zone)
            )

            if src_occupancy == _PAIRED_ZONE_OCCUPANCY and targ_occupancy == 0:
                # Moving a complete 2-ion chain is always preferred over splitting
                # because it avoids the extra split penalty and preserves the pair.
                actions.append(
                    _SiqciShuttleAction(
                        qubits=_edge_qubits(
                            src_zone_qubits, src_port, _PAIRED_ZONE_OCCUPANCY
                        ),
                        src_zone=src_zone,
                        targ_zone=targ_zone,
                        src_port=src_port,
                        targ_port=targ_port,
                        cost=transport_cost,
                    )
                )
                if _is_internal_zone(dyn_arch, src_zone) and _adjacent_zones_empty(
                    dyn_arch, state, src_zone
                ):
                    actions.append(
                        _SiqciShuttleAction(
                            qubits=_edge_qubits(src_zone_qubits, src_port, 1),
                            src_zone=src_zone,
                            targ_zone=targ_zone,
                            src_port=src_port,
                            targ_port=targ_port,
                            cost=transport_cost + split_penalty,
                        )
                    )
                continue

            if src_occupancy == 1 and targ_occupancy <= 1:
                # A single ion may move into either an empty zone or a singly
                # occupied zone. Entering a singly occupied zone forms a pair and
                # therefore incurs the configured merge penalty.
                actions.append(
                    _SiqciShuttleAction(
                        qubits=_edge_qubits(src_zone_qubits, src_port, 1),
                        src_zone=src_zone,
                        targ_zone=targ_zone,
                        src_port=src_port,
                        targ_port=targ_port,
                        cost=transport_cost
                        + (merge_penalty if targ_occupancy == 1 else 0.0),
                    )
                )

    return sorted(
        actions,
        key=lambda action: (
            # Keep action generation deterministic so Dijkstra explores equal-cost
            # branches in a stable order across runs.
            action.src_zone,
            action.targ_zone,
            len(action.qubits),
            action.qubits,
        ),
    )


def _pswap_actions(
    dyn_arch: SgzlDynamicArch, state: tuple[tuple[int, ...], ...]
) -> list[_SiqciPswapAction]:
    """List all legal physical swaps in the current siqci state."""
    actions: list[_SiqciPswapAction] = []
    for zone, zone_qubits in enumerate(state):
        if len(zone_qubits) != _PAIRED_ZONE_OCCUPANCY:
            continue
        actions.append(
            _SiqciPswapAction(
                zone=zone,
                qubit0=zone_qubits[0],
                qubit1=zone_qubits[1],
                cost=float(dyn_arch.zone_swap_costs[zone]),
            )
        )
    return actions


def _goal_reached(
    state: tuple[tuple[int, ...], ...], gate_zone: int, target_gate_qubits: list[int]
) -> bool:
    gate_zone_qubits = set(state[gate_zone])
    target_qubits = set(target_gate_qubits)
    if len(target_gate_qubits) == 1:
        return target_qubits.issubset(gate_zone_qubits)
    return gate_zone_qubits == target_qubits


def _append_step_ops(
    ops: list[RoutingOp], action: _SiqciPswapAction | _SiqciShuttleAction
) -> None:
    """Emit one move group per physical action.

    Keeping each action between routing barriers makes the routed circuit follow
    the same sequence that the planner validated.
    """
    if not ops:
        ops.append(RoutingBarrier())
    if isinstance(action, _SiqciPswapAction):
        ops.append(PSwap(action.zone, action.qubit0, action.qubit1))
    else:
        ops.append(
            Shuttle(
                list(action.qubits),
                action.src_zone,
                action.targ_zone,
                action.src_port,
                action.targ_port,
            )
        )
    ops.append(RoutingBarrier())


def _find_route_actions(
    dyn_arch: SgzlDynamicArch,
    start_state: tuple[tuple[int, ...], ...],
    target_gate_qubits: list[int],
    split_penalty: float,
    merge_penalty: float,
) -> list[_SiqciPswapAction | _SiqciShuttleAction]:
    """Find a minimum-cost legal siqci route with Dijkstra search.

    The siqci architecture is small enough that we can search the exact machine
    state space directly instead of relying on heuristics. A state is just the
    ordered qubit occupants of each of the five zones. The outgoing edges are the
    legal shuttles and pswaps produced by the siqci move rules. Because all move
    costs are non-negative, Dijkstra gives the minimum-cost route to the first
    state whose gate-zone occupants match the requested target pair, or contain
    the requested target qubit in the singleton case.
    """
    frontier: list[tuple[float, int, tuple[tuple[int, ...], ...]]] = []
    push_counter = count()
    heappush(frontier, (0.0, next(push_counter), start_state))
    best_cost: dict[tuple[tuple[int, ...], ...], float] = {start_state: 0.0}
    previous_step: dict[tuple[tuple[int, ...], ...], _RouteStep] = {}
    final_state: tuple[tuple[int, ...], ...] | None = None
    gate_zone = dyn_arch.single_gate_zone

    while frontier:
        current_cost, _, state = heappop(frontier)
        if current_cost > best_cost.get(state, float("inf")):
            continue
        if _goal_reached(state, gate_zone, target_gate_qubits):
            # The first goal popped from the priority queue is optimal because the
            # search is over non-negative edge costs.
            final_state = state
            break

        actions: list[_SiqciPswapAction | _SiqciShuttleAction] = _shuttle_actions(
            dyn_arch,
            state,
            split_penalty,
            merge_penalty,
        ) + _pswap_actions(dyn_arch, state)
        for action in actions:
            if isinstance(action, _SiqciShuttleAction):
                next_state = _shuttle_state(state, action)
            else:
                next_state = _pswap_state(state, action)
            next_cost = current_cost + action.cost
            if next_cost >= best_cost.get(next_state, float("inf")):
                continue
            best_cost[next_state] = next_cost
            previous_step[next_state] = _RouteStep(previous_state=state, action=action)
            heappush(frontier, (next_cost, next(push_counter), next_state))

    if final_state is None:
        raise ValueError("SiqciArchRouter could not find a legal routing sequence.")

    action_sequence: list[_SiqciPswapAction | _SiqciShuttleAction] = []
    state = final_state
    while state != start_state:
        route_step = previous_step[state]
        action_sequence.append(route_step.action)
        state = route_step.previous_state
    action_sequence.reverse()
    return action_sequence


def _apply_route_actions(
    dyn_arch: SgzlDynamicArch,
    action_sequence: list[_SiqciPswapAction | _SiqciShuttleAction],
) -> RoutingResult:
    """Replay the planned action sequence on the mutable DynamicArch."""
    routing_ops: list[RoutingOp] = []
    total_cost = 0.0
    for action in action_sequence:
        if isinstance(action, _SiqciShuttleAction):
            dyn_arch.move_qubits(
                list(action.qubits),
                action.src_zone,
                action.targ_zone,
                action.targ_port.value,
            )
        else:
            dyn_arch.swap_qubits_in_zone(action.zone, action.qubit0, action.qubit1)
        _append_step_ops(routing_ops, action)
        total_cost += action.cost
    return RoutingResult(cost_estimate=total_cost, routing_ops=routing_ops)


class SiqciArchRouter(Router):
    """Router specialized to the five-zone `siqci_arch` line architecture.

    This router is much more specialized than the general single-gate-zone line
    router. The architecture is tiny, every zone has gate and transport capacity
    2, and pswaps are allowed in any doubly occupied zone. Because of that, the
    router can plan directly in the exact machine state space.

    The workflow is:
    1. Validate that the target placement specifies either one qubit that must be
       brought into the gate zone or exactly the next two qubits that should
       occupy the single gate zone.
    2. Encode the current zone occupancies as an immutable search state.
    3. Run Dijkstra search over all legal shuttles and pswaps, using the siqci
       transport, split, merge, and swap costs,
    4. Replay the optimal action sequence on the mutable dynamic architecture and
       emit the corresponding routing ops.

    This keeps the implementation compact while still matching the siqci-specific
    movement rules exactly.
    """

    def __init__(self, split_penalty: float = 1.0, merge_penalty: float = 0.8):
        self.split_penalty = split_penalty
        self.merge_penalty = merge_penalty

    def route_source_to_target_config(
        self,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> RoutingResult:
        """Route the current siqci state to the requested next gate-zone target."""
        sgzl_dyn_arch = require_sgzl_dynamic_arch(dyn_arch)
        _validate_siqci_architecture(sgzl_dyn_arch)
        target_gate_qubits = _validate_target_placement(sgzl_dyn_arch, target_placement)
        start_state = _state_from_dyn_arch(sgzl_dyn_arch)
        gate_zone = sgzl_dyn_arch.single_gate_zone
        if _goal_reached(start_state, gate_zone, target_gate_qubits):
            return RoutingResult(cost_estimate=0, routing_ops=[])
        action_sequence = _find_route_actions(
            sgzl_dyn_arch,
            start_state,
            target_gate_qubits,
            self.split_penalty,
            self.merge_penalty,
        )
        return _apply_route_actions(sgzl_dyn_arch, action_sequence)
