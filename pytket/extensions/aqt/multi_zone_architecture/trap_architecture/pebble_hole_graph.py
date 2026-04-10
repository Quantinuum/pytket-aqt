from networkx import Graph

from ..circuit.helpers import TrapConfiguration
from .architecture import MultiZoneArchitectureSpec, PortId


class PebbleHoleGraph:
    def __init__(
        self, spec: MultiZoneArchitectureSpec, start_config: TrapConfiguration
    ):
        self.pebble_graph: Graph[tuple[int, int]] = Graph()
        self.zone_size: list[int] = []
        placement = start_config.zone_placement
        for zone_id, zone in enumerate(spec.zones):
            transport_max_cap = zone.max_ions_transport_op
            occupancy = len(placement[zone_id])
            self.zone_size.append(occupancy)
            # add nodes
            for i in range(transport_max_cap):
                occ = placement[zone_id][i] if i < occupancy else -1
                self.pebble_graph.add_node((zone_id, i), occupant=occ)
            # add inner zone edges
            for i in range(transport_max_cap - 1):
                self.pebble_graph.add_edge((zone_id, i), (zone_id, i + 1))

        # Add "shuttle" edges between connected zones.
        for connection in spec.connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            node0 = (zone0, 0 if port0 == PortId.p0 else self.zone_size[zone0])
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id
            node1 = (zone1, 0 if port1 == PortId.p0 else self.zone_size[zone1])
            # TODO: update arch spec to include connection shuttle cost and use that as weight
            self.pebble_graph.add_edge(
                node0, node1, transport_cost=1, is_shuttle_edge=True
            )
