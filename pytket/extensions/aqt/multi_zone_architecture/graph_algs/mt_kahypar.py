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
from logging import getLogger

import mtkahypar  # type: ignore

from ...multi_zone_architecture.graph_algs.graph import GraphData, HypergraphData

logger = getLogger(__name__)


@dataclass
class MtKahyparConfig:
    n_threads: int = 1
    random_seed: int = 13


MTK: mtkahypar.Initializer | None = None


def configure_mtkahypar(
    config: MtKahyparConfig, warn_configured: bool = True
) -> mtkahypar.Initializer:
    global MTK  # noqa: PLW0603
    if MTK is None:
        mtkahypar.set_seed(config.random_seed)
        MTK = mtkahypar.initialize(config.n_threads)
    elif warn_configured:
        logger.warning(
            "MtKahypar is already configured and can only be configured once"
            ", ignoring new configuration call"
        )
    return MTK


class MtKahyparPartitioner:
    """Class that performs graph partitioning using mt-kahypar"""

    def __init__(self) -> None:
        self.mtk = configure_mtkahypar(MtKahyparConfig(), warn_configured=False)
        self.context = self.mtk.context_from_preset(mtkahypar.PresetType.DEFAULT)
        self.context.logging = False

    def graph_data_to_mtkahypar_graph(self, graph_data: GraphData) -> mtkahypar.Graph:
        return self.mtk.create_graph(
            self.context,
            graph_data.n_vertices,
            len(graph_data.edges),
            graph_data.edges,
            graph_data.vertex_weights,
            graph_data.edge_weights,
        )

    def hypergraph_data_to_mtkahypar_hypergraph(
        self, graph_data: HypergraphData
    ) -> mtkahypar.Hypergraph:
        return self.mtk.create_hypergraph(
            self.context,
            graph_data.n_vertices,
            len(graph_data.nets),
            graph_data.nets,
            graph_data.vertex_weights,
            graph_data.net_weights,
        )

    def graph_data_to_mtkahypar_target_graph(
        self, graph_data: GraphData
    ) -> mtkahypar.TargetGraph:
        return self.mtk.create_target_graph(
            self.context,
            graph_data.n_vertices,
            len(graph_data.edges),
            graph_data.edges,
            graph_data.edge_weights,
        )

    def partition_graph(
        self,
        graph_data: GraphData,
        num_parts: int,
    ) -> list[int]:
        """Partition vertices of graph into num_parts parts

        Returns a list whose i'th element is the part that vertex i is assigned to

        :param graph_data: Graph specification
        :param num_parts: Number of partitions
        """
        avg_part_weight = sum(graph_data.vertex_weights) / num_parts
        self.context.set_partitioning_parameters(
            num_parts,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,
        )
        graph = self.graph_data_to_mtkahypar_graph(graph_data)
        if graph_data.fixed_list:
            graph.add_fixed_vertices(graph_data.fixed_list, num_parts)
        if graph_data.part_max_sizes:
            self.context.set_individual_target_block_weights(graph_data.part_max_sizes)
        part_graph = graph.partition(self.context)
        dbg_msg = f"cut_cost: {part_graph.cut()}"
        logger.debug(dbg_msg)
        vertex_part_id: list[int] = []
        for vertex in range(graph_data.n_vertices):
            vertex_part_id.append(part_graph.block_id(vertex))  # noqa: PERF401
        return vertex_part_id

    def partition_hypergraph(
        self,
        hypergraph_data: HypergraphData,
        num_parts: int,
    ) -> list[int]:
        """Partition vertices of graph into num_parts parts

        Returns a list whose i'th element is the part that vertex i is assigned to

        :param hypergraph_data: Graph specification
        :param num_parts: Number of partitions
        """
        avg_part_weight = sum(hypergraph_data.vertex_weights) / num_parts
        self.context.set_partitioning_parameters(
            num_parts,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,
        )
        graph = self.hypergraph_data_to_mtkahypar_hypergraph(hypergraph_data)
        if hypergraph_data.fixed_list:
            graph.add_fixed_vertices(hypergraph_data.fixed_list, num_parts)
        if hypergraph_data.part_max_sizes:
            self.context.set_individual_target_block_weights(
                hypergraph_data.part_max_sizes
            )
        part_graph = graph.partition(self.context)
        dbg_msg = f"cut_cost: {part_graph.cut()}"
        logger.debug(dbg_msg)
        vertex_part_id: list[int] = []
        for vertex in range(hypergraph_data.n_vertices):
            vertex_part_id.append(part_graph.block_id(vertex))  # noqa: PERF401
        return vertex_part_id

    def map_graph_to_target_graph(
        self, graph_data: GraphData, target_graph_data: GraphData
    ) -> list[int]:
        """Partition vertices of graph onto the nodes of
         another graph minimizing Steiner tree metric

        Returns a list whose i'th element is the target
         graph node that vertex i is assigned to
        """
        avg_part_weight = sum(graph_data.vertex_weights) / target_graph_data.n_vertices
        self.context.set_partitioning_parameters(
            target_graph_data.n_vertices,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,  # This doesn't matter, Steiner tree metric is used
        )
        graph = self.graph_data_to_mtkahypar_graph(graph_data)
        if graph_data.part_max_sizes:
            self.context.set_individual_target_block_weights(graph_data.part_max_sizes)
        target_graph = self.graph_data_to_mtkahypar_target_graph(target_graph_data)
        part_graph = graph.map_onto_graph(target_graph, self.context)
        vertex_part_id: list[int] = []
        for vertex in range(graph_data.n_vertices):
            vertex_part_id.append(part_graph.block_id(vertex))  # noqa: PERF401
        return vertex_part_id
