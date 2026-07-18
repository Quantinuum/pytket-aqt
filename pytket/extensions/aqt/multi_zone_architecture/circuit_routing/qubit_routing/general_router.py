import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ...circuit.helpers import ZonePlacement, get_qubit_to_zone
from ...trap_architecture.architecture import PortId
from ...trap_architecture.cost_model import RoutingCostModel, ShuttlePSwapCostModel
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


class GeneralRouter(Router):
    """Uses cost model to determine physical operations to add to get to target_placement

    Does not respect the qubit order within the zones of the target placement, the order
    results from the order of moves into the zones, which is determined by trying to minimize
    cost

    This router tries to perform full moves from a starting zone to a target zone. It supports
    multi-qubit shuttles. It picks and implements moves in order of cost.

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
        validate_complete_target_placement(starting_config.n_qubits, target_placement)
        qubits_to_move = get_needed_movements(
            starting_config.n_qubits, starting_config.zone_placement, target_placement
        )
        pending_move_groups = group_needed_movements(qubits_to_move)

        def free_space_in_zone_func(zon: int) -> int:
            # use the transport limit - 1 >= gate limit as the base capacity,
            # The -1 ensures that the implementation of a move group doesn't
            # leave the target zone in a blocked state
            return int(dyn_arch.transport_free_space[zon]) - 1

        total_cost = 0
        move_ops: list[RoutingOp] = [RoutingBarrier()]
        soft_locked = False  # soft locked means only
        transport_blocked_zone = None
        while pending_move_groups:
            move_groups = get_move_groups(
                pending_move_groups,
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
            remove_moved_qubits_from_group(
                pending_move_groups,
                chosen_move_group.source,
                chosen_move_group.target,
                optimal_move_group_result.qubits,
            )

        return RoutingResult(total_cost, move_ops)

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
            move_group.qubits.sort(
                key=lambda qubit: dyn_arch.qubit_to_zone_pos[qubit, 1]
            )
            max_n_move = min(len(move_group.qubits), move_group.target_free_space)
            for n_move in range(max_n_move, 0, -1):
                result = self.get_move_path_cost(dyn_arch, move_group, n_move)
                if result:
                    yield MoveGroupPath(
                        move_group, n_move, result[0], result[1], result[2]
                    )

    def get_move_path_cost(
        self, dyn_arch: DynamicArch, move_group: MoveGroup, n_move: int
    ) -> tuple[list[int], list[int], int] | None:
        src = move_group.source
        trg = move_group.target

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
                return (
                    qubits_indx_0,
                    move_result_0.optimal_path,
                    move_result_0.path_cost,
                )
            return (
                qubits_indx_1,
                move_result_1.optimal_path,
                move_result_1.path_cost,
            )
        if move_result_0:
            return (
                qubits_indx_0,
                move_result_0.optimal_path,
                move_result_0.path_cost,
            )
        if move_result_1:
            return (
                qubits_indx_1,
                move_result_1.optimal_path,
                move_result_1.path_cost,
            )
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
        qubits_zone = zone_qubits.copy()

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
        (qubit, int(qubit_to_zone_old[qubit]), int(qubit_to_zone_new[qubit]))
        for qubit in range(n_qubits)
        if qubit_to_zone_old[qubit] != qubit_to_zone_new[qubit]
    ]


def group_needed_movements(
    qubits_to_move: list[tuple[int, int, int]],
) -> dict[tuple[int, int], list[int]]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for qubit, src, trg in qubits_to_move:
        grouped[(src, trg)].append(qubit)
    return dict(grouped)


def remove_moved_qubits_from_group(
    pending_move_groups: dict[tuple[int, int], list[int]],
    src: int,
    trg: int,
    moved_qubits: list[int],
) -> None:
    group_key = (src, trg)
    moved_qubits_set = set(moved_qubits)
    remaining_qubits = [
        qubit
        for qubit in pending_move_groups[group_key]
        if qubit not in moved_qubits_set
    ]
    if remaining_qubits:
        pending_move_groups[group_key] = remaining_qubits
    else:
        del pending_move_groups[group_key]


def validate_complete_target_placement(
    n_qubits: int, target_placement: ZonePlacement
) -> None:
    qubit_counts = [0 for _ in range(n_qubits)]
    out_of_range_qubits: list[int] = []
    for zone_qubits in target_placement:
        for qubit in zone_qubits:
            if qubit < 0 or qubit >= n_qubits:
                out_of_range_qubits.append(qubit)
                continue
            qubit_counts[qubit] += 1

    missing_qubits = [qubit for qubit, count in enumerate(qubit_counts) if count == 0]
    duplicated_qubits = [qubit for qubit, count in enumerate(qubit_counts) if count > 1]
    if missing_qubits or duplicated_qubits or out_of_range_qubits:
        raise ValueError(
            "GeneralRouter requires a complete target placement with each qubit "
            f"placed exactly once. Missing qubits: {missing_qubits}; "
            f"duplicated qubits: {duplicated_qubits}; "
            f"out-of-range qubits: {sorted(out_of_range_qubits)}."
        )


def get_move_groups(
    grouped_moves: dict[tuple[int, int], list[int]],
    transport_blocked_zone: int | None,
    free_space_in_zone_func: Callable[[int], int],
) -> list[MoveGroup]:
    return (
        [
            MoveGroup(grouped_qbts.copy(), src, trg, free_space_in_zone_func(trg))
            for (src, trg), grouped_qbts in grouped_moves.items()
        ]
        if transport_blocked_zone is None
        else [
            # If a zone is transport blocked, only consider moves out of it, in order to unblock it
            # There must be one since the end state is not allowed to be transport blocked
            MoveGroup(grouped_qbts.copy(), src, trg, free_space_in_zone_func(trg))
            for (src, trg), grouped_qbts in grouped_moves.items()
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
