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

from ...circuit.helpers import ZonePlacement
from ...depth_list.depth_list import (
    DepthInfo,
    depth_info_from_command_list,
)
from ...graph_algs.graph import HypergraphData
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


class HypergraphPartitionGateSelector(GateSelector):
    """Uses hypergraph partitioning to determine an optimal new placement of qubits
    in zones

    :param cost_model: Cost model for estimating movement costs
    :param max_depth: Maximum depth used for 2 qubit gate edges
     in model graph
    """

    def __init__(
        self,
        cost_model: RoutingCostModel = _DEFAULT_COST_MODEL,
        max_depth: int = 50,
    ):
        self._cost_model = cost_model
        self._max_depth = max_depth

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
            return self.handle_2qb_gates_remaining(dyn_arch, depth_info)
        return self.handle_only_single_qubit_gates_remaining(
            dyn_arch, remaining_commands
        )

    def handle_2qb_gates_remaining(
        self,
        dyn_arch: DynamicArch,
        depth_info: DepthInfo,
    ) -> ZonePlacement:
        num_zones = dyn_arch.n_zones
        n_qubits = dyn_arch.n_qubits
        shuttle_graph_data = self.get_circuit_shuttle_hypergraph_data(
            dyn_arch, depth_info
        )
        partitioner = MtKahyparPartitioner()
        log_depth_info(depth_info)
        vertex_to_part = partitioner.partition_hypergraph(shuttle_graph_data, num_zones)
        new_placement: ZonePlacement = [[] for _ in range(num_zones)]
        part_to_zone = [-1] * num_zones
        for vertex in range(n_qubits, n_qubits + num_zones):
            part_to_zone[vertex_to_part[vertex]] = vertex - n_qubits
        for vertex in range(n_qubits):
            new_placement[part_to_zone[vertex_to_part[vertex]]].append(vertex)
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
        handle_unused_qubits(dyn_arch, self._cost_model, qubit_tracker)
        return qubit_tracker.new_placement()

    def get_circuit_shuttle_hypergraph_data(  # noqa: PLR0912
        self, dyn_arch: DynamicArch, depth_info: DepthInfo
    ) -> HypergraphData:
        """Calculate graph data for qubit-zone graph to be partitioned"""
        n_qubits = dyn_arch.n_qubits
        num_zones = dyn_arch.n_zones
        places_per_zone = [
            dyn_arch.zone_max_gate_cap[i]
            + 1  # +1 is for the fixed vertex for each zone itself
            for i in range(num_zones)
        ]
        num_spots = sum(places_per_zone)
        nets: list[list[int]] = []
        net_weights: list[int] = []

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

        # add gate hyperedges
        for depth, blocks in enumerate(depth_blocks[:cutoff_depth]):
            for block in blocks:
                weight = max_gate_weight - math.floor(
                    depth * max_gate_weight * 0.01
                )  # reduce by 5% per depth
                if dyn_arch.has_memory_zones:
                    for zone in dyn_arch.gate_zones:
                        net = [*list(block), zone + n_qubits]
                        nets.append(net)
                    net_weights.extend([weight] * len(dyn_arch.gate_zones))
                else:
                    nets.append(list(block))
                    net_weights.append(weight)

        # add shuttling penalty
        max_shuttle_weight = math.floor(max_gate_weight * 0.8)
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
                            * 0.05
                        )
                    weight = max_shuttle_weight - penalty
                    if weight < 1:
                        continue
                    nets.append([qubit, other_zone + n_qubits])
                    net_weights.append(weight)

        num_vertices = num_spots
        vertex_weights = [1 for _ in range(num_vertices)]

        fixed_list = (
            [-1] * n_qubits
            + [zone for zone in range(num_zones)]  # noqa: C416
            + [-1] * (num_vertices - n_qubits - num_zones)
        )

        return HypergraphData(
            num_vertices,
            vertex_weights,
            nets,
            net_weights,
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
