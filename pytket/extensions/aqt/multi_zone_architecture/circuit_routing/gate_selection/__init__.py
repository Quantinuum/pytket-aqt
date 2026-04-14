from ...graph_algs.mt_kahypar_check import MT_KAHYPAR_INSTALLED
from .greedy_gate_selection import GreedyGateSelector

if MT_KAHYPAR_INSTALLED:
    from .graph_partition_gate_selection import GraphPartitionGateSelector
    from .hypergraph_partition_gate_selection import HypergraphPartitionGateSelector


__all__ = [
    "GreedyGateSelector",
]

if MT_KAHYPAR_INSTALLED:
    __all__.extend(
        [
            "GraphPartitionGateSelector",
            "HypergraphPartitionGateSelector",
        ]
    )
