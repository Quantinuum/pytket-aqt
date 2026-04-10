import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...circuit.helpers import ZonePlacement, get_qubit_to_zone
from ...trap_architecture.architecture import PortId
from ...trap_architecture.cost_model import (
    MoveCostResult,
    RoutingCostModel,
    ShuttlePSwapCostModel,
)
from ...trap_architecture.dynamic_architecture import DynamicArch
from ..routing_ops import PSwap, RoutingBarrier, RoutingOp, Shuttle
from .router import Router, RoutingResult

_DEFAULT_COST_MODEL = ShuttlePSwapCostModel()


@dataclass
class MoveGroup:
    qubits: list[int]
    source: int
    target: int
    target_free_space: int


@dataclass
class MoveGroupPath:
    chosen_move_group: MoveGroup
    n_move: int
    qubits: list[int]
    path: list[int]
    cost: int


@dataclass
class GateZoneSpec:
    idx: int
    is_terminal: bool
    no_scheduled_gates: bool
    current_config: list[int]
    target_config: list[int]


@dataclass
class Crystal:
    id: int
    qubits: list[int]


ZoneCrystals = list[list[Crystal]]


class PathClearingRouter(Router):
    """Uses cost model to determine physical operations to add to get to target_placement

    Does not respect the qubit order within the zones of the target placement, the order
    results from the order of moves into the zones, which is determined by trying to minimize
    cost

    TODO: describe algorithm here

    :param cost_model: Cost model used to estimate cost of moves

    """

    def __init__(
        self,
        cost_model: RoutingCostModel = _DEFAULT_COST_MODEL,
    ):
        self._cost_model = cost_model

    def route_source_to_target_config(
        self,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> RoutingResult:
        starting_config = dyn_arch.trap_configuration
        move_ops: list[RoutingOp] = [RoutingBarrier()]
        total_cost = 0
        for zone in dyn_arch.gate_zones:
            connected_zones_port0, connected_zones_port1 = (
                dyn_arch.connected_zones_per_port(zone)
            )
            # handle terminal zones
            if len(connected_zones_port0) == 0:
                self.handle_terminal_gate_zone(zone, 1, dyn_arch, target_placement)
            elif len(connected_zones_port1) == 0:
                self.handle_terminal_gate_zone(zone, 0, dyn_arch, target_placement)

        qubits_to_move = get_needed_movements(
            starting_config.n_qubits, starting_config.zone_placement, target_placement
        )

        def free_space_in_zone_func(zon: int) -> int:
            # use the transport limit - 1 >= gate limit as the base capacity,
            # The -1 ensures that the implementation of a move group doesn't
            # leave the target zone in a blocked state
            return int(dyn_arch.transport_free_space[zon]) - 1

        soft_locked = False  # soft locked means only
        transport_blocked_zone = None
        while qubits_to_move:
            move_groups = get_move_groups(
                qubits_to_move,
                transport_blocked_zone,
                free_space_in_zone_func,
            )
            soft_locked = check_and_handle_soft_locked(soft_locked, move_groups)

            optimal_move_group_result = self.select_move_group(dyn_arch, move_groups)
            chosen_move_group = optimal_move_group_result.chosen_move_group

            # Add ops for optimal move and update dyn_arch internals
            move_ops.extend(
                implement_move_group_result(dyn_arch, optimal_move_group_result)
            )

            total_cost += optimal_move_group_result.cost
            move_ops.append(RoutingBarrier())

            transport_blocked_zone = check_if_transport_blocked(
                soft_locked, chosen_move_group.target, free_space_in_zone_func
            )

            # remove moves that were made
            for q in optimal_move_group_result.qubits:
                qubits_to_move.remove(
                    (q, chosen_move_group.source, chosen_move_group.target)
                )

        return RoutingResult(total_cost, move_ops)

    def handle_terminal_gate_zone(
        self,
        terminal_gate_zone: int,
        out_port: int,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> list[RoutingOp]:
        """
        Strategy:
         - As qubits targeting the zone "move in", remove them from arch and add them onto a separate stack
         - At the beginning, any group of target qubits already at the unconnected port are removed and added to the stack
            - They are already at the target destination
         - Any qubits in the target zone at the beginning will be forced to move out. Even if they could possibly fit.
            - Why: 1.) Makes logic easier 2.) Reduced gate error for following gate round
         - After removing any initially well-placed qubits, the goal becomes to get target qubits to reach the
         unconnected port (without using swaps), at which point they are also removed and added to the stack
            - This requires moving any remaining qubits first and clearing the paths for target qubits to make it to the
            port
         - At the end all target qubits will be in the stack and the zone will be empty. The zone occupants can be replaced by the
         stack contents in the correct order (FIFO at the unconnected port).
        """
        target_config = target_placement[terminal_gate_zone].copy()
        zone_stack = []

        current_config = dyn_arch.trap_configuration.zone_placement[terminal_gate_zone]

        # if out_port = 0 iterate from 1 to 0, else 0 to 1
        q_iter: Iterable[int] = (
            current_config if out_port == 1 else reversed(current_config)
        )

        # handle any target qubits that are already placed at the correct
        # edge of the target zone
        for q in q_iter:
            if q in target_config:
                target_config.remove(q)
                zone_stack.append(q)
            else:
                break
        for q in zone_stack:
            current_config.remove(q)

        # Build Crystal config
        crystal_config = []
        crystal_to_zone_pos = []
        target_crystals = []
        crys_idx = 0
        for zone_id, zone in enumerate(dyn_arch.trap_configuration.zone_placement):
            new_zone_config = []
            previous_in_target = None
            zone_crys_pos = 0
            for q in zone:
                in_target = q in target_config
                if in_target != previous_in_target:
                    new_zone_config.append(Crystal(crys_idx, [q]))
                    crystal_to_zone_pos.append((zone_id, zone_crys_pos))
                    if in_target:
                        target_crystals.append(crys_idx)
                    crys_idx += 1
                    zone_crys_pos += 1
                else:
                    new_zone_config[-1].qubits.append(q)
                previous_in_target = in_target
            crystal_config.append(new_zone_config)

        target_qubits_current_zone_pos = dyn_arch.qubit_to_zone_pos[target_config]

        target_qubit_move_results = []
        for target_c in target_crystals:
            current_zone, current_pos = crystal_to_zone_pos[target_c]
            if current_zone == terminal_gate_zone:
                # handle qubit already in target zone
                step = 1 if out_port == 0 else -1
                qubits_in_way = [current_config[i] for i in range(current_pos)]
                clear_spec = [(current_zone, [q])]

            mg = MoveGroup([target_q], current_zone, terminal_gate_zone, 1)
            target_qubit_move_results.append(self.get_move_path_cost(dyn_arch, mg, 1))

    def select_move_group(
        self, dyn_arch: DynamicArch, move_groups: list[MoveGroup]
    ) -> MoveGroupPath:
        def path_cost_per_qubit_selector(mgp: MoveGroupPath) -> float:
            return mgp.cost / mgp.n_move

        return min(
            self.move_group_selection_generator(dyn_arch, move_groups),
            key=path_cost_per_qubit_selector,
        )

    def move_group_selection_generator(
        self, dyn_arch: DynamicArch, move_groups: list[MoveGroup]
    ) -> Iterator[MoveGroupPath]:
        for move_group in move_groups:
            max_n_move = min(len(move_group.qubits), move_group.target_free_space)
            for n_move in range(max_n_move, 0, -1):
                result = self.get_move_path_cost(dyn_arch, move_group, n_move)
                if result:
                    yield MoveGroupPath(
                        move_group, n_move, result[0], result[1], result[2]
                    )

    def get_move_path_cost(
        self, dyn_arch: DynamicArch, move_group: MoveGroup, n_move: int
    ) -> MoveCostResult | None:
        src = move_group.source
        trg = move_group.target
        # Make sure qubits are ordered the same as in src zone
        move_group.qubits.sort(key=lambda x: dyn_arch.qubit_to_zone_pos[x][1])

        qubits_indx_0 = move_group.qubits[:n_move]
        qubits_indx_1 = move_group.qubits[-n_move:]

        move_result_0 = self._cost_model.move_cost_src_port_0(
            dyn_arch, qubits_indx_0, src, trg
        )
        move_result_1 = self._cost_model.move_cost_src_port_1(
            dyn_arch, qubits_indx_1, src, trg
        )
        if move_result_0 and move_result_1:
            if move_result_0.path_cost <= move_result_1.path_cost:
                return move_result_0
            return move_result_1
        if move_result_0:
            return move_result_0
        if move_result_1:
            return move_result_1
        return None


def implement_move_group_result(
    dyn_arch: DynamicArch,
    mgp: MoveGroupPath,
) -> list[RoutingOp]:
    """Create a list of RoutingOp's that move the qubits (with src zone index, ordered by ascending index)
    from source zone to target port

    Modifies the input current placement to reflect the movement
    """
    ops: list[RoutingOp] = []
    path = mgp.path
    starting_zone = mgp.chosen_move_group.source
    target_zone = mgp.chosen_move_group.target
    all_qubits = mgp.qubits
    # all_qubits should be kept in "logical order", i.e. the order they have
    # in their current zone when viewed from port 0 to port 1. To do this,
    # anytime they are shuttled from a 0 port to a 0 port (or 1 port to 1 port)
    # their order should be reversed

    # For the initial move along the path, the qubits are not necessarily at
    # an edge, but can be anywhere in the starting zone
    # so add swaps, taking this into account and shuttle
    src_port, trg_port = dyn_arch.connection_ports(path[0], path[1])
    current_placement = dyn_arch.trap_configuration.zone_placement
    ops.extend(
        swap_through_zone_and_shuttle_internal_qubits(
            dyn_arch,
            all_qubits,
            (path[0], path[1]),
            (src_port, trg_port),
            current_placement[path[0]],
        )
    )
    # maintain logical ordering
    if src_port == trg_port:
        all_qubits.reverse()

    # For any remaining moves qubits are already necessarily at an edge and
    # the "current_placement" doesn't reflect their temporary position
    current_port = trg_port
    for current_zone, next_zone in itertools.pairwise(path[1:]):
        src_port, trg_port = dyn_arch.connection_ports(current_zone, next_zone)
        if src_port == current_port:
            raise ValueError("Invalid internal shuttle sequence")
        current_zone_occupants = current_placement[current_zone]
        ops.extend(
            swap_through_zone_and_shuttle_edge_qubits(
                all_qubits,
                (current_zone, next_zone),
                (src_port, trg_port),
                current_zone_occupants,
            )
        )
        current_port = trg_port
        # maintain logical ordering
        if src_port == trg_port:
            all_qubits.reverse()

    # Update dyn_arch
    dyn_arch.move_qubits(all_qubits, starting_zone, target_zone, trg_port)
    return ops


def swap_through_zone_and_shuttle_internal_qubits(
    dyn_arch: DynamicArch,
    all_qubits: list[int],
    zones: tuple[int, int],
    ports: tuple[int, int],
    zone_qubits: list[int],
) -> list[RoutingOp]:
    ops: list[RoutingOp] = []
    qubits_index_to_move = [(q, dyn_arch.qubit_to_zone_pos[q, 1]) for q in all_qubits]
    if ports[0] == 0:
        qubits_zone = list(reversed(zone_qubits))
        last_indx = len(zone_qubits) - 1
        # update indices to reflect reversed ordering
        qubit_src_iter: Iterable[tuple[int, Any]] = [
            (q, last_indx - indx) for q, indx in qubits_index_to_move
        ]
    else:
        # reversing makes logic same for moving to port 1 instead of port 0
        qubit_src_iter = reversed(qubits_index_to_move)
        qubits_zone = deepcopy(zone_qubits)

    for qubit, index in qubit_src_iter:
        ops.extend(
            [
                PSwap(zones[0], left_qubit, qubit)
                for left_qubit in qubits_zone[index + 1 :]
            ]
        )
        qubits_zone.pop(index)

    ops.append(
        Shuttle(
            all_qubits.copy(),
            zones[0],
            zones[1],
            PortId(ports[0]),
            PortId(ports[1]),
        )
    )
    return ops


def swap_through_zone_and_shuttle_edge_qubits(
    move_qubits: list[int],
    zones: tuple[int, int],
    ports: tuple[int, int],
    zone_qubits: list[int],
) -> list[RoutingOp]:
    ops: list[RoutingOp] = []

    # 01 -> move qubits
    # abc -> occupants
    if ports[0] == 0:
        move_qubits_iter: Iterable[int] = move_qubits
        current_zone_iter = list(reversed(zone_qubits))
        # [abc01] -> [ab0c1] -> [a0bc1] -> [0abc1] -> [0ab1c] -> [0a1bc] -> [01abc]
    else:
        move_qubits_iter = reversed(move_qubits)
        current_zone_iter = zone_qubits
        # [01abc] -> [0a1bc] -> [0ab1c] -> [0abc1] -> [a0bc1] -> [ab0c1] -> [abc01]
    ops.extend(
        [
            PSwap(zones[0], stay_qubit, move_qubit)
            for move_qubit in move_qubits_iter
            for stay_qubit in current_zone_iter
        ]
    )
    ops.append(
        Shuttle(
            move_qubits.copy(), zones[0], zones[1], PortId(ports[0]), PortId(ports[1])
        )
    )
    return ops


def get_needed_movements(
    n_qubits: int, old_placement: ZonePlacement, new_placement: ZonePlacement
) -> list[tuple[int, int, int]]:
    qubit_to_zone_old = get_qubit_to_zone(n_qubits, old_placement)
    qubit_to_zone_new = get_qubit_to_zone(n_qubits, new_placement)

    return [
        (qubit, int(qubit_to_zone_old[qubit]), int(zone))
        for qubit, zone in enumerate(qubit_to_zone_new)
        if zone != -1
    ]


def get_move_groups(
    qubits_to_move: list[tuple[int, int, int]],
    transport_blocked_zone: int | None,
    free_space_in_zone_func: Callable[[int], int],
) -> list[MoveGroup]:
    grouped = defaultdict(list)
    for qubit, src, trg in qubits_to_move:
        grouped[(src, trg)].append(qubit)
    return (
        [
            MoveGroup(grouped_qbts, src, trg, free_space_in_zone_func(trg))
            for (src, trg), grouped_qbts in grouped.items()
        ]
        if transport_blocked_zone is None
        else [
            # If a zone is transport blocked, only consider moves out of it, in order to unblock it
            # There must be one since the end state is not allowed to be transport blocked
            MoveGroup(grouped_qbts, src, trg, free_space_in_zone_func(trg))
            for (src, trg), grouped_qbts in grouped.items()
            if src == transport_blocked_zone
        ]
    )


def check_and_handle_soft_locked(
    already_soft_locked: bool, move_groups: list[MoveGroup]
) -> bool:
    if (
        already_soft_locked
        or sum(group.target_free_space for group in move_groups) == 0
    ):
        # This can happen if qubits need to swap between full zones or
        # there is a cycle between full zones.
        # To solve, use the full transport capacity for all following rounds
        # The sum will remain zero once this point is reached since any movement
        # from one zone to another will cause free_space calculation to give +1 and -1 for
        # those zones.
        for group in move_groups:
            group.target_free_space += 1
        return True
    return False


def check_if_transport_blocked(
    soft_locked: bool,
    potentially_blocked_zone: int,
    free_space_func: Callable[[int], int],
) -> int | None:
    # When soft locked a move may cause a zone to become transport blocked
    # resulting in its calculated free space taking the value -1
    # This makes sure the next move will be out of this zone, unblocking it
    if soft_locked and free_space_func(potentially_blocked_zone) == -1:
        return potentially_blocked_zone
    return None
