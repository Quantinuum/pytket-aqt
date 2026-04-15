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
import logging
import math

from pytket.circuit import Command

from ...circuit.helpers import TrapConfiguration, ZonePlacement
from ...depth_list.depth_list import (
    DepthInfo,
    depth_info_from_command_list,
)
from ...graph_algs.graph import GraphData
from ...graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
    MissingMtKahyparInstallError,
)

if MT_KAHYPAR_INSTALLED:
    from ...graph_algs.mt_kahypar import MtKahyparPartitioner
else:
    raise MissingMtKahyparInstallError

from ...trap_architecture.cost_model import RoutingCostModel, ShuttlePSwapCostModel
from ...trap_architecture.dynamic_architecture import DynamicArch
from ..command_filtering import filter_implementable_commands
from .gate_selector_protocol import GateSelector
from .greedy_gate_selection import (
    handle_only_single_qubits_remaining,
    handle_unused_qubits,
)
from .qubit_tracker import QubitTracker

logger = logging.getLogger(__name__)


def log_depth_info(depth_info: DepthInfo) -> None:
    logger.debug("--- Depth List ---")
    for i in range(min(4, len(depth_info.depth_list))):
        msg = f"{i}: gate_pairs {depth_info.depth_list[i]}, blocks {depth_info.depth_blocks[i]} "
        logger.debug(msg)


_DEFAULT_COST_MODEL = ShuttlePSwapCostModel()


class GraphPartitionGateSelector(GateSelector):
    """Uses graph partitioning to determine an optimal new placement of qubits
    in zones

    :param cost_model: Cost model for estimating movement costs
    :param max_depth: Maximum depth used for 2 qubit gate edges
     in model graph
    """

    def __init__(
        self,
        cost_model: RoutingCostModel = _DEFAULT_COST_MODEL,
        max_depth: int = 50,
        only_place_gate_qubits: bool = False,
    ):
        self._cost_model = cost_model
        self._max_depth = max_depth
        self._only_specify_gate_qubits = only_place_gate_qubits

    def next_config(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        """Generates a new qubit placement in zones to implement the next gates

        The returned ZonePlacement
        represents the "optimal" next state to implement the remaining gates in
        the depth list. The ordering of the qubits within the zones is arbitrary. The correct
        ordering will be determined at the qubit routing stage.

        :param dyn_arch: The dynamic architecture containing current configuration of ions in ion trap zones
        :param remaining_commands: The list of gate commands used to determine the next ion placement.
        """
        current_configuration = dyn_arch.trap_configuration
        n_qubits = current_configuration.n_qubits
        depth_info = depth_info_from_command_list(n_qubits, remaining_commands)
        if depth_info.depth_list:
            return self.handle_2qb_gates_remaining(
                dyn_arch, depth_info, remaining_commands
            )
        return self.handle_only_single_qubit_gates_remaining(
            dyn_arch, remaining_commands
        )

    def handle_2qb_gates_remaining(
        self,
        dyn_arch: DynamicArch,
        depth_info: DepthInfo,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        num_zones = dyn_arch.n_zones
        n_qubits = dyn_arch.n_qubits
        shuttle_graph_data = self.get_circuit_shuttle_graph_data(dyn_arch, depth_info)
        partitioner = MtKahyparPartitioner()
        log_depth_info(depth_info)
        vertex_to_part = partitioner.partition_graph(shuttle_graph_data, num_zones)
        new_placement: ZonePlacement = [[] for _ in range(num_zones)]
        part_to_zone = [-1] * num_zones
        for vertex in range(n_qubits, n_qubits + num_zones):
            part_to_zone[vertex_to_part[vertex]] = vertex - n_qubits
        for vertex in range(n_qubits):
            new_placement[part_to_zone[vertex_to_part[vertex]]].append(vertex)

        if self._only_specify_gate_qubits:
            implementable_commands, _ = filter_implementable_commands(
                TrapConfiguration(n_qubits, new_placement),
                dyn_arch.gate_zones,
                remaining_commands,
            )
            return get_placement_of_implementable_qubits(
                new_placement, implementable_commands
            )
        return new_placement

    def handle_only_single_qubit_gates_remaining(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        qubit_tracker = QubitTracker(
            dyn_arch.n_qubits, dyn_arch.trap_configuration.zone_placement
        )
        handle_only_single_qubits_remaining(
            dyn_arch, self._cost_model, remaining_commands, qubit_tracker
        )
        # Now move any unused qubits to vacant spots in new config
        if not self._only_specify_gate_qubits:
            handle_unused_qubits(dyn_arch, self._cost_model, qubit_tracker)
        return qubit_tracker.new_placement()

    def get_circuit_shuttle_graph_data(
        self, dyn_arch: DynamicArch, depth_info: DepthInfo
    ) -> GraphData:
        """Calculate graph data for qubit-zone graph to be partitioned"""
        n_qubits = dyn_arch.n_qubits
        num_zones = dyn_arch.n_zones
        places_per_zone = [
            dyn_arch.zone_max_gate_cap[i]
            + 1  # +1 is for the fixed vertex for each zone itself
            for i in range(num_zones)
        ]
        num_spots = sum(places_per_zone)
        edges: list[tuple[int, int]] = []
        edge_weights: list[int] = []

        depth_list = depth_info.depth_list
        depth_blocks = depth_info.depth_blocks
        cutoff_depth = 1
        for _, blocks in enumerate(
            depth_blocks[1:]
        ):  # at depth 0 all blocks are size 2
            min_block_size = min(len(block) for block in blocks)
            if min_block_size > dyn_arch.largest_gate_zone_max_capacity:
                break
            cutoff_depth += 1

        max_gate_weight = 50000

        # add gate edges
        for depth, pairs in enumerate(depth_list[:cutoff_depth]):
            weight = max_gate_weight - math.floor(
                depth * max_gate_weight * 0.05
            )  # reduce by 5% per depth
            edges.extend(pairs)
            edge_weights.extend([weight] * len(pairs))

        # "Assign" qubits up to cutoff depth to gate zones with high weight
        if dyn_arch.has_memory_zones:
            depth_0_qubits = [q for block in depth_blocks[0] for q in block]
            edge_pair_pairs = [
                (q, zone + n_qubits)
                for q in depth_0_qubits
                for zone in dyn_arch.gate_zones
            ]
            edge_pair_weights = (
                [math.floor(max_gate_weight)]
                * len(depth_0_qubits)
                * len(dyn_arch.gate_zones)
            )
            edges.extend(edge_pair_pairs)
            edge_weights.extend(edge_pair_weights)

        # add shuttling penalty
        max_shuttle_weight = math.floor(max_gate_weight * 0.5)
        for zone, qubits in enumerate(dyn_arch.trap_configuration.zone_placement):
            for qubit in qubits:
                for other_zone in range(num_zones):
                    if other_zone == zone:
                        # if src == trg, penalty for moving to an edge becomes reason to stay
                        penalty = 0
                    else:
                        penalty = math.floor(
                            self.distance_to_closest_port_of_target_zone(
                                dyn_arch, qubit, zone, other_zone
                            )
                            * max_shuttle_weight
                            * 0.1
                        )
                    weight = max_shuttle_weight - penalty
                    if weight < 1:
                        continue
                    edges.append((qubit, other_zone + n_qubits))
                    edge_weights.append(weight)

        num_vertices = num_spots
        vertex_weights = [1 for _ in range(num_vertices)]

        fixed_list = (
            [-1] * n_qubits
            + [zone for zone in range(num_zones)]  # noqa: C416
            + [-1] * (num_vertices - n_qubits - num_zones)
        )

        return GraphData(
            num_vertices,
            vertex_weights,
            edges,
            edge_weights,
            fixed_list,
            places_per_zone,
        )

    def distance_to_closest_port_of_target_zone(
        self,
        dyn_arch: DynamicArch,
        qubit: int,
        src_zone: int,
        trg_zone: int,
    ) -> int:
        """Calculate penalty for shuttling from one zone to another"""
        move_result_0 = self._cost_model.move_cost_src_port_0(
            dyn_arch, [qubit], src_zone, trg_zone
        )
        move_result_1 = self._cost_model.move_cost_src_port_1(
            dyn_arch, [qubit], src_zone, trg_zone
        )
        if move_result_0 and move_result_1:
            return min(
                move_result_0.path_cost,
                move_result_1.path_cost,
            )
        if move_result_0:
            return move_result_0.path_cost
        if move_result_1:
            return move_result_1.path_cost
        raise ValueError("Could note determine path")


def get_placement_of_implementable_qubits(
    target_placement: ZonePlacement,
    implementable_commands: list[Command],
) -> ZonePlacement:
    implementable_qubits: set[int] = set()

    for cmd in implementable_commands:
        implementable_qubits.update(qubit.index[0] for qubit in cmd.args)

    return [
        [qubit for qubit in zone_qubits if qubit in implementable_qubits]
        for zone_qubits in target_placement
    ]
