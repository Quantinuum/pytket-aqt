from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from networkx import (
    Graph,
)

from .architecture import MultiZoneArchitectureSpec, PortId

if TYPE_CHECKING:
    from ..circuit.helpers import TrapConfiguration

Node: TypeAlias = tuple[int, int]

_EMPTY = -1
_MAX_PATH_CANDIDATES = 8
_MAX_MULTI_EVICTION_ZONE_CANDIDATES = 6
_MAX_MULTI_TARGET_NODE_CANDIDATES = 4
_NO_PATH = 10**9


class PebbleMotionError(Exception):
    pass


@dataclass(frozen=True)
class PebbleMove:
    qubit: int
    source: Node
    target: Node


class PebbleHoleGraph:
    def __init__(
        self, spec: MultiZoneArchitectureSpec, start_config: TrapConfiguration
    ):
        self.pebble_graph: Graph = Graph()
        self._zone_capacities: list[int] = []
        self._zone_nodes: list[list[Node]] = []
        placement = start_config.zone_placement
        for zone_id, zone in enumerate(spec.zones):
            transport_max_cap = zone.max_ions_transport_op
            self._zone_capacities.append(transport_max_cap)
            zone_nodes: list[Node] = []
            occupancy = len(placement[zone_id])
            for i in range(transport_max_cap):
                occ = placement[zone_id][i] if i < occupancy else _EMPTY
                node = (zone_id, i)
                self.pebble_graph.add_node(node, occupant=occ)
                zone_nodes.append(node)
            self._zone_nodes.append(zone_nodes)
            for i in range(transport_max_cap - 1):
                self.pebble_graph.add_edge((zone_id, i), (zone_id, i + 1))

        # Add "shuttle" edges between connected zones.
        for connection in spec.port_to_port_connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            node0 = (
                zone0,
                0 if port0 == PortId.p0 else self._zone_capacities[zone0] - 1,
            )
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id
            node1 = (
                zone1,
                0 if port1 == PortId.p0 else self._zone_capacities[zone1] - 1,
            )
            self.pebble_graph.add_edge(
                node0,
                node1,
                transport_cost=connection.shuttle_cost,
                is_shuttle_edge=True,
            )

    def occupant(self, node: Node) -> int:
        return int(self.pebble_graph.nodes[node]["occupant"])

    def is_empty(self, node: Node) -> bool:
        return self.occupant(node) == _EMPTY

    def qubit_node(self, qubit: int) -> Node:
        for node in self.pebble_graph.nodes:
            if self.occupant(node) == qubit:
                return node
        raise PebbleMotionError(f"Qubit {qubit} is not placed on the pebble graph")

    def empty_nodes(self) -> list[Node]:
        return [node for node in self.pebble_graph.nodes if self.is_empty(node)]

    def zone_nodes(self, zone: int) -> list[Node]:
        return self._zone_nodes[zone].copy()

    def qubits_in_zone(self, zone: int) -> list[int]:
        return [
            occupant
            for node in self._zone_nodes[zone]
            if (occupant := self.occupant(node)) != _EMPTY
        ]

    def placement(self) -> list[list[int]]:
        return [self.qubits_in_zone(zone) for zone in range(len(self._zone_nodes))]
