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
from ...trap_architecture.named_architectures import siqci_arch
from ..routing_ops import PSwap, RoutingBarrier, RoutingOp, Shuttle
from .router import Router, RoutingResult

_PAIRED_ZONE_OCCUPANCY = 2
_INTERNAL_ZONE_MIN_INDEX = 1


@dataclass(frozen=True)
class _SiqciPswapAction:
    zone: int
    qubit0: int
    qubit1: int
    cost: float


@dataclass(frozen=True)
class _SiqciShuttleAction:
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
    if dyn_arch.architecture_spec != siqci_arch:
        raise ValueError(
            "SiqciArchRouter can only be used with the siqci_arch architecture."
        )


def _validate_target_placement(
    dyn_arch: SgzlDynamicArch, target_placement: ZonePlacement
) -> list[int]:
    gate_zone = dyn_arch.single_gate_zone
    for zone, zone_qubits in enumerate(target_placement):
        if zone != gate_zone and zone_qubits:
            raise ValueError(
                "SiqciArchRouter requires target placements to specify qubits only in the gate zone."
            )
    target_gate_qubits = target_placement[gate_zone]
    if len(target_gate_qubits) not in (0, _PAIRED_ZONE_OCCUPANCY):
        raise ValueError(
            "SiqciArchRouter target placement must specify either zero or two qubits in the gate zone."
        )
    return target_gate_qubits


def _state_from_dyn_arch(dyn_arch: SgzlDynamicArch) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(zone_qubits) for zone_qubits in dyn_arch.trap_configuration.zone_placement
    )


def _edge_qubits(
    zone_qubits: tuple[int, ...], src_port: PortId, n_move: int
) -> tuple[int, ...]:
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
            action.src_zone,
            action.targ_zone,
            len(action.qubits),
            action.qubits,
        ),
    )


def _pswap_actions(
    dyn_arch: SgzlDynamicArch, state: tuple[tuple[int, ...], ...]
) -> list[_SiqciPswapAction]:
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
    if not target_gate_qubits:
        return True
    return set(state[gate_zone]) == set(target_gate_qubits)


def _append_step_ops(
    ops: list[RoutingOp], action: _SiqciPswapAction | _SiqciShuttleAction
) -> None:
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
    """Router specialized to the siqci_arch architecture."""

    def __init__(self, split_penalty: float = 1.0, merge_penalty: float = 0.8):
        self.split_penalty = split_penalty
        self.merge_penalty = merge_penalty

    def route_source_to_target_config(
        self,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> RoutingResult:
        sgzl_dyn_arch = require_sgzl_dynamic_arch(dyn_arch)
        _validate_siqci_architecture(sgzl_dyn_arch)
        target_gate_qubits = _validate_target_placement(sgzl_dyn_arch, target_placement)
        if not target_gate_qubits:
            return RoutingResult(cost_estimate=0, routing_ops=[])

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
