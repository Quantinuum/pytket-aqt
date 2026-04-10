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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from networkx import (
    Graph,
    single_source_dijkstra,
)

from .architecture import MultiZoneArchitectureSpec, PortId


@dataclass(frozen=True)
class MacroZoneConfig:
    max_occupancy: int


@dataclass
class MacroZoneData:
    qubits: set[int]
    zone_config: MacroZoneConfig


class MultiZoneArch:
    def __init__(self, spec: MultiZoneArchitectureSpec):
        self.zone_graph: Graph = Graph()
        self.shortest_paths: dict[tuple[int, int], tuple[int, list[int]] | None] = {}
        self.zone_connections: list[list[int]] = [[]] * spec.n_zones
        self.connection_ports: dict[tuple[int, int], tuple[PortId, PortId]] = {}
        self.memory_zones: list[int] = []
        self.gate_zones: list[int] = []

        for zone_id, zone in enumerate(spec.zones):
            zone_data = MacroZoneData(
                qubits=set(),
                zone_config=MacroZoneConfig(max_occupancy=zone.max_ions_gate_op),
            )
            self.zone_graph.add_node(int(zone_id), zone_data=zone_data)
            if zone.memory_only:
                self.memory_zones.append(zone_id)
            else:
                self.gate_zones.append(zone_id)
            self.has_memory_zones = len(self.memory_zones) > 0
            # The zone port graph treats the ports of each zone as separate nodes int the graph

        for connection in spec.connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id

            self.zone_connections[zone0].append(zone1)
            self.zone_connections[zone1].append(zone0)
            if (zone0, zone1) not in self.connection_ports:
                self.connection_ports[(zone0, zone1)] = (port0, port1)
                self.connection_ports[(zone1, zone0)] = (port1, port0)
            else:
                raise ValueError(
                    f"Two connections between zones {zone0} and {zone1}"
                    f" specified, but only 1 connection between two zones is allowed"
                )
            self.zone_graph.add_edge(int(zone0), int(zone1), transport_cost=1)

    def shortest_path(self, zone_1: int, zone_2: int) -> list[int]:
        _, path = self.shortest_path_with_length(zone_1, zone_2)
        return path

    def connected_zones(self, zone) -> Iterable[int]:
        return self.zone_graph.neighbors(zone)

    def connected_zones_per_port(self, zone) -> tuple[list[int], list[int]]:
        neighbors = self.zone_graph.neighbors(zone)
        port_connections = ([], [])
        for neighbor in neighbors:
            port_connection, _ = self.get_connected_ports(zone, neighbor)
            port_connections[port_connection.value].append(neighbor)
        return port_connections

    def shortest_path_with_length(
        self, zone_1: int, zone_2: int
    ) -> tuple[int, list[int]]:
        cached_length_path = self.shortest_paths.get((zone_1, zone_2))
        if cached_length_path:
            return cached_length_path
        length_path = cast(
            "tuple[int, list[int]]",
            single_source_dijkstra(
                self.zone_graph, zone_1, zone_2, weight="transport_cost"
            ),
        )
        self.shortest_paths[(zone_1, zone_2)] = length_path
        self.shortest_paths[(zone_2, zone_1)] = (length_path[0], length_path[1][::-1])
        return length_path

    def get_connected_ports(
        self, source_zone: int, target_zone: int
    ) -> tuple[PortId, PortId]:
        try:
            return self.connection_ports[(source_zone, target_zone)]
        except KeyError:
            raise ValueError(  # noqa: B904
                f"No connection exists between zones {source_zone} and {target_zone}"
            )


def empty_macro_arch_from_architecture(
    architecture: MultiZoneArchitectureSpec,
) -> MultiZoneArch:
    return MultiZoneArch(architecture)
