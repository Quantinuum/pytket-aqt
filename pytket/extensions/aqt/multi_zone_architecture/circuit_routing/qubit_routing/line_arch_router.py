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
from ...trap_architecture.architecture import PortId
from ...trap_architecture.dynamic_architecture import DynamicArch
from ..routing_ops import RoutingBarrier, RoutingOp, Shuttle
from .router import Router, RoutingResult


@dataclass(frozen=True)
class SwapFreeSegmentation:
    ordered_zones: list[int]
    block_sizes: list[int]
    zone_placement: ZonePlacement


def target_zone_interval_qubits(
    dyn_arch: DynamicArch,
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


def ordered_qubits(dyn_arch: DynamicArch) -> list[int]:
    return [
        qubit
        for zone in linearly_ordered_zones(dyn_arch)
        for qubit in dyn_arch.trap_configuration.zone_placement[zone]
    ]


def swap_free_routing_segmentation(
    dyn_arch: DynamicArch,
    target_placement: ZonePlacement,
) -> SwapFreeSegmentation | None:
    # In the swap-free case, each final zone must receive a contiguous block of the
    # current global qubit order. The DP tracks which prefixes can be assigned to
    # the first k zones while respecting capacities and any explicitly requested zones.
    ordered_zones = linearly_ordered_zones(dyn_arch)
    ordered_zone_positions = {zone: i for i, zone in enumerate(ordered_zones)}
    current_qubit_order = ordered_qubits(dyn_arch)
    n_qubits = len(current_qubit_order)
    n_zones = len(ordered_zones)
    zone_capacities = [int(dyn_arch.zone_max_gate_cap[zone]) for zone in ordered_zones]

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

    for zone_pos, _zone in enumerate(ordered_zones):
        zone_capacity = zone_capacities[zone_pos]
        for prefix_length in range(n_qubits + 1):
            if not dp[zone_pos][prefix_length]:
                continue

            if not dp[zone_pos + 1][prefix_length]:
                dp[zone_pos + 1][prefix_length] = True
                predecessor[zone_pos + 1][prefix_length] = prefix_length
            for block_size in range(
                1, min(zone_capacity, n_qubits - prefix_length) + 1
            ):
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
        return None

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

    zone_placement: ZonePlacement = [[] for _ in range(dyn_arch.n_zones)]
    order_index = 0
    for zone_pos, zone in enumerate(ordered_zones):
        block_size = block_sizes[zone_pos]
        zone_placement[zone] = current_qubit_order[
            order_index : order_index + block_size
        ]
        order_index += block_size

    return SwapFreeSegmentation(
        ordered_zones=ordered_zones,
        block_sizes=block_sizes,
        zone_placement=zone_placement,
    )


def execute_swap_free_segmentation(
    dyn_arch: DynamicArch,
    segmentation: SwapFreeSegmentation,
) -> RoutingResult:
    if dyn_arch.trap_configuration.zone_placement == segmentation.zone_placement:
        return RoutingResult(cost_estimate=0, routing_ops=[])

    desired_block_sizes = segmentation.block_sizes
    ops: list[RoutingOp] = [RoutingBarrier()]
    total_cost = 0

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
                    ops.append(shuttle)
                    total_cost += shuttle_cost
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
                    ops.append(shuttle)
                    total_cost += shuttle_cost
                    progress = True

        if not progress:
            raise ValueError(
                "Could not execute swap-free segmentation with shuttles only."
            )

    ops.append(RoutingBarrier())
    return RoutingResult(cost_estimate=total_cost, routing_ops=ops)


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


def linearly_ordered_zones(dyn_arch: DynamicArch) -> list[int]:
    ordered_zones = [line_start_zone(dyn_arch)]
    previous_zone: int | None = None
    current_zone = ordered_zones[0]

    while True:
        next_zones = [
            zone
            for zone in dyn_arch.connected_zones(current_zone)
            if zone != previous_zone
        ]
        if not next_zones:
            return ordered_zones
        if len(next_zones) > 1:
            raise ValueError("Linear architecture contains a branching zone order.")
        previous_zone, current_zone = current_zone, next_zones[0]
        ordered_zones.append(current_zone)


def line_start_zone(dyn_arch: DynamicArch) -> int:
    for zone in range(dyn_arch.n_zones):
        connected_zones_0, connected_zones_1 = dyn_arch.connected_zones_per_port(zone)
        if not connected_zones_0 and connected_zones_1:
            return zone
    raise ValueError("Could not determine the start zone of linear architecture.")


class LineArchRouter(Router):
    """Router specialized for linear multi-zone architectures."""

    def route_source_to_target_config(
        self, dyn_arch: DynamicArch, target_placement: ZonePlacement
    ) -> RoutingResult:
        if not dyn_arch.is_linear_architecture:
            raise ValueError(
                "LineArchRouter can only be used with linear architectures."
            )
        target_zone_interval_qubits(dyn_arch, target_placement)
        segmentation = swap_free_routing_segmentation(dyn_arch, target_placement)
        if segmentation is None:
            raise ValueError(
                "LineArchRouter does not yet support target placements that require swaps."
            )
        return execute_swap_free_segmentation(dyn_arch, segmentation)
