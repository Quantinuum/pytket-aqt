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

import os
from enum import Enum
from itertools import combinations
from typing import Self

from networkx import DiGraph, Graph, connected_components, single_source_dijkstra
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator


class PortId(Enum):
    """Each Zone has two ports, p0 and p1, that allow for connections to
    other zones"""

    p0 = 0
    p1 = 1


class PortSpec(BaseModel):
    """Uniquely identifies a port within the architecture

    The zone_id identifies the zone and the port_id identifies
    the port. port_id can be either p0 (the shuttling port of position 0)
    or p1 (the shuttling port of the current last position in the zone)

    """

    model_config = ConfigDict(extra="forbid")

    zone_id: int
    port_id: PortId


class ZoneConnection(BaseModel):
    """A connection between two zone ports

    The connection allows shuttling between the zones
    """

    model_config = ConfigDict(extra="forbid")

    zone_port_spec0: PortSpec
    zone_port_spec1: PortSpec
    shuttle_cost: int = 1


class Junction(BaseModel):
    """A shuttling junction that cannot store qubits."""

    model_config = ConfigDict(extra="forbid")

    junction_id: int
    cost: int = 1


class JunctionRef(BaseModel):
    """Identifies a junction endpoint in a physical architecture connection."""

    model_config = ConfigDict(extra="forbid")

    junction_id: int


class PhysicalConnection(BaseModel):
    """A physical shuttling connection between zone ports and/or junctions."""

    model_config = ConfigDict(extra="forbid")

    endpoint0: PortSpec | JunctionRef
    endpoint1: PortSpec | JunctionRef
    shuttle_cost: int = 1


class Operation(BaseModel):
    """Describes an allowed operation and its associated fidelity

    Currently not used
    """

    model_config = ConfigDict(extra="forbid")

    operation_spec: str
    fidelity: float | str


class Zone(BaseModel):
    """Processor Zone within the architecture"""

    model_config = ConfigDict(extra="forbid")

    max_ions_gate_op: int
    max_ions_transport_op: int = -1
    memory_only: bool = False
    swap_cost: int = 1

    @model_validator(mode="after")
    def set_and_validate(self) -> Self:
        if self.max_ions_transport_op == -1:
            self.max_ions_transport_op = self.max_ions_gate_op + 1
        if self.max_ions_gate_op < 1:
            raise ValueError(
                f"'max_ions_gate_op' must be at least 1."
                f" Got max_ions_transport_op={self.max_ions_gate_op},"
            )
        if self.max_ions_transport_op < self.max_ions_gate_op:
            raise ValueError(
                f"'max_ions_transport_op' must be greater or equal to 'max_ions_gate_op'."
                f" Got max_ions_transport_op={self.max_ions_transport_op},"
                f" max_ions_gate_op={self.max_ions_gate_op}"
            )
        return self


class LayoutPosition(BaseModel):
    """Coordinates for visualizing an architecture element."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class VisualizationSpec(BaseModel):
    """Optional architecture layout information for visualizations.

    Coordinates are interpreted as abstract layout units rather than pixels.
    When supplied, the visualizer scales and translates them to fit the
    available SVG area. Position dictionaries are keyed by zone ID or junction
    ID.
    """

    model_config = ConfigDict(extra="forbid")

    zone_positions: dict[int, LayoutPosition] = {}
    junction_positions: dict[int, LayoutPosition] = {}


class MultiZoneArchitectureSpec(BaseModel):
    """Specification of a physical multi-zone trap architecture.

    The architecture is defined by a list of zones, optional junctions, and
    physical shuttling connections between them. Zones are the only places that
    can hold ions. Each zone has two addressable ports, identified by
    `PortSpec(zone_id, port_id)`, and each zone port may participate in at
    most one physical connection. Junctions are transit-only nodes that cannot
    hold ions but may connect to any number of zone ports and/or other
    junctions.

    The `connections` field describes physical edges as `PhysicalConnection`
    objects. Each endpoint is either a zone port (``PortSpec``) or a junction
    (`JunctionRef`), and each connection has a shuttle cost. Junctions also
    have a traversal cost. During validation, the physical graph is expanded
    into cached pairwise `ZoneConnection` objects between every pair of zone
    ports connected by any physical route, using the cheapest route cost when
    multiple routes exist. Downstream routing models consume this expanded
    representation via `port_to_port_connections`.

    The optional `visualization` field can be used to provide a preferred
    physical layout for visualizers. It contains dictionaries of
    `LayoutPosition` objects keyed by zone ID and junction ID. These positions
    are abstract coordinates rather than pixels, and each coordinate denotes
    the center of the corresponding zone or junction. Zone and junction
    positions are interpreted in the same coordinate system, so equal x or y
    values keep those elements aligned when the visualizer scales and
    translates the layout to fit its canvas.
    """

    model_config = ConfigDict(extra="forbid")

    n_qubits_max: int
    n_zones: int
    zones: list[Zone]
    junctions: list[Junction] = []
    connections: list[PhysicalConnection] = []
    visualization: VisualizationSpec | None = None

    _port_to_port_connections: list[ZoneConnection] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def set_and_validate(self) -> Self:
        if self.n_zones != len(self.zones):
            raise ValueError(
                f"Multi-zone architecture defined to have {self.n_zones} zones, but {len(self.zones)} zones were defined."
            )
        self._validate_visualization_spec()
        self._validate_physical_connections()
        self._port_to_port_connections = self._compute_port_to_port_connections()
        return self

    def _validate_visualization_spec(self) -> None:
        if self.visualization is None:
            return
        self._validate_visualization_positions(
            element="zone",
            positions=self.visualization.zone_positions,
            expected_ids=set(range(self.n_zones)),
        )
        self._validate_visualization_positions(
            element="junction",
            positions=self.visualization.junction_positions,
            expected_ids={junction.junction_id for junction in self.junctions},
        )

    @staticmethod
    def _validate_visualization_positions(
        element: str,
        positions: dict[int, LayoutPosition],
        expected_ids: set[int],
    ) -> None:
        if not positions:
            return
        position_ids = set(positions)
        if position_ids != expected_ids:
            raise ValueError(
                f"Visualization spec defines {element} positions for IDs {sorted(position_ids)}, but expected IDs {sorted(expected_ids)}."
            )

    def _validate_port_spec(self, port_spec: PortSpec) -> None:
        if port_spec.zone_id < 0 or port_spec.zone_id >= self.n_zones:
            raise ValueError(
                f"Zone port references zone {port_spec.zone_id}, but architecture has {self.n_zones} zones."
            )

    def _validate_physical_connections(self) -> None:
        junction_ids = [junction.junction_id for junction in self.junctions]
        if len(junction_ids) != len(set(junction_ids)):
            raise ValueError("Junction IDs must be unique.")

        junction_costs = {
            junction.junction_id: junction.cost for junction in self.junctions
        }
        for junction in self.junctions:
            if junction.cost < 0:
                raise ValueError(
                    f"Junction {junction.junction_id} has negative cost {junction.cost}."
                )

        port_degrees: dict[tuple[int, PortId], int] = {}
        for connection in self.connections:
            if connection.shuttle_cost < 0:
                raise ValueError(
                    f"'shuttle_cost' must be non-negative. Got {connection.shuttle_cost}."
                )
            endpoint_keys = [
                self._validate_physical_endpoint(endpoint, junction_costs)
                for endpoint in (connection.endpoint0, connection.endpoint1)
            ]
            if endpoint_keys[0] == endpoint_keys[1]:
                raise ValueError(
                    "A physical connection cannot connect an endpoint to itself."
                )
            for endpoint in (connection.endpoint0, connection.endpoint1):
                if isinstance(endpoint, PortSpec):
                    self._increment_physical_port_degree(port_degrees, endpoint)

    @staticmethod
    def _increment_physical_port_degree(
        port_degrees: dict[tuple[int, PortId], int], port_spec: PortSpec
    ) -> None:
        port_key = (port_spec.zone_id, port_spec.port_id)
        port_degrees[port_key] = port_degrees.get(port_key, 0) + 1
        if port_degrees[port_key] > 1:
            raise ValueError(
                f"Zone {port_spec.zone_id} port {port_spec.port_id.name} has more than one physical connection."
            )

    def _validate_physical_endpoint(
        self, endpoint: PortSpec | JunctionRef, junction_costs: dict[int, int]
    ) -> tuple[str, int, int | None]:
        if isinstance(endpoint, PortSpec):
            self._validate_port_spec(endpoint)
            return "port", endpoint.zone_id, endpoint.port_id.value
        if endpoint.junction_id not in junction_costs:
            raise ValueError(
                f"Physical connection references unknown junction {endpoint.junction_id}."
            )
        return "junction", endpoint.junction_id, None

    @property
    def port_to_port_connections(self) -> list[ZoneConnection]:
        """The cached downstream port-to-port connections."""
        return self._port_to_port_connections

    def _compute_port_to_port_connections(self) -> list[ZoneConnection]:
        """Return the downstream port-to-port connections.

        Physical connections are expanded into the cheapest pairwise zone-port
        connections across each physical connectivity component.
        """
        cheapest_connections: dict[
            tuple[tuple[int, PortId], tuple[int, PortId]], ZoneConnection
        ] = {}

        physical_graph, weighted_physical_graph = self._physical_connectivity_graphs()
        port_nodes = [
            node
            for node, data in physical_graph.nodes(data=True)
            if data["kind"] == "port"
        ]
        for component in connected_components(physical_graph):
            component_ports = [port for port in port_nodes if port in component]
            for port_node0, port_node1 in combinations(component_ports, 2):
                zone0, port0 = self._port_node_to_zone_port(port_node0)
                zone1, port1 = self._port_node_to_zone_port(port_node1)
                if zone0 == zone1:
                    raise ValueError(
                        f"Zone {zone0} has an external path between its two ports."
                    )
                cost, _ = single_source_dijkstra(
                    weighted_physical_graph,
                    port_node0,
                    port_node1,
                    weight="cost",
                )
                connection = ZoneConnection(
                    zone_port_spec0=PortSpec(zone_id=zone0, port_id=port0),
                    zone_port_spec1=PortSpec(zone_id=zone1, port_id=port1),
                    shuttle_cost=int(cost),
                )
                self._keep_cheapest_connection(cheapest_connections, connection)
        return [
            cheapest_connections[key]
            for key in sorted(
                cheapest_connections,
                key=self._connection_key_sort_value,
            )
        ]

    def _physical_connectivity_graphs(self) -> tuple[Graph, DiGraph]:
        graph = Graph()
        weighted_graph = DiGraph()
        for junction in self.junctions:
            node = self._junction_node(junction.junction_id)
            node_data = {
                "kind": "junction",
                "junction_cost": junction.cost,
            }
            graph.add_node(node, **node_data)
            weighted_graph.add_node(node, **node_data)
        for connection in self.connections:
            node0 = self._physical_endpoint_node(connection.endpoint0)
            node1 = self._physical_endpoint_node(connection.endpoint1)
            for node in (node0, node1):
                if node[0] == "port":
                    node_data = {"kind": "port", "junction_cost": 0}
                    graph.add_node(node, **node_data)
                    weighted_graph.add_node(node, **node_data)
            current_edge = graph.get_edge_data(node0, node1)
            if (
                current_edge is None
                or connection.shuttle_cost < current_edge["shuttle_cost"]
            ):
                graph.add_edge(node0, node1, shuttle_cost=connection.shuttle_cost)
            self._add_weighted_physical_edge(
                weighted_graph,
                node0,
                node1,
                connection.shuttle_cost,
            )
            self._add_weighted_physical_edge(
                weighted_graph,
                node1,
                node0,
                connection.shuttle_cost,
            )
        return graph, weighted_graph

    @staticmethod
    def _add_weighted_physical_edge(
        weighted_graph: DiGraph,
        source: tuple[str, int, int | None],
        target: tuple[str, int, int | None],
        shuttle_cost: int,
    ) -> None:
        cost = shuttle_cost + int(weighted_graph.nodes[target]["junction_cost"])
        current_edge = weighted_graph.get_edge_data(source, target)
        if current_edge is None or cost < current_edge["cost"]:
            weighted_graph.add_edge(source, target, cost=cost)

    def _physical_endpoint_node(
        self, endpoint: PortSpec | JunctionRef
    ) -> tuple[str, int, int | None]:
        if isinstance(endpoint, PortSpec):
            return self._port_node(endpoint.zone_id, endpoint.port_id)
        return self._junction_node(endpoint.junction_id)

    @staticmethod
    def _port_node(zone_id: int, port_id: PortId) -> tuple[str, int, int]:
        return ("port", zone_id, port_id.value)

    @staticmethod
    def _junction_node(junction_id: int) -> tuple[str, int, None]:
        return ("junction", junction_id, None)

    @staticmethod
    def _port_node_to_zone_port(
        node: tuple[str, int, int | None],
    ) -> tuple[int, PortId]:
        assert node[0] == "port"
        assert node[2] is not None
        return node[1], PortId(node[2])

    @staticmethod
    def _connection_sort_key(
        connection: ZoneConnection,
    ) -> tuple[tuple[int, PortId], tuple[int, PortId]]:
        zone_port0 = (
            connection.zone_port_spec0.zone_id,
            connection.zone_port_spec0.port_id,
        )
        zone_port1 = (
            connection.zone_port_spec1.zone_id,
            connection.zone_port_spec1.port_id,
        )
        return (
            (zone_port0, zone_port1)
            if (zone_port0[0], zone_port0[1].value)
            <= (zone_port1[0], zone_port1[1].value)
            else (zone_port1, zone_port0)
        )

    @staticmethod
    def _connection_key_sort_value(
        connection_key: tuple[tuple[int, PortId], tuple[int, PortId]],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        zone_port0, zone_port1 = connection_key
        return (
            (zone_port0[0], zone_port0[1].value),
            (zone_port1[0], zone_port1[1].value),
        )

    @classmethod
    def _keep_cheapest_connection(
        cls,
        cheapest_connections: dict[
            tuple[tuple[int, PortId], tuple[int, PortId]], ZoneConnection
        ],
        connection: ZoneConnection,
    ) -> None:
        key = cls._connection_sort_key(connection)
        current_connection = cheapest_connections.get(key)
        if (
            current_connection is None
            or connection.shuttle_cost < current_connection.shuttle_cost
        ):
            cheapest_connections[key] = connection

    def get_zone_max_ions_gates(self, zone_index: int) -> int:
        zone = self.zones[zone_index]
        return zone.max_ions_gate_op

    def get_zone_max_ions_transport(self, zone_index: int) -> int:
        zone = self.zones[zone_index]
        return zone.max_ions_transport_op

    def __str__(self) -> str:
        arch_spec_lines = [
            f"Max number of qubits: {self.n_qubits_max}",
            f"Number of zones: {self.n_zones}",
            "",
        ]
        connections_per_zone_port: list[list[list[tuple[int, PortId]]]] = [
            [[], []] for _ in range(self.n_zones)
        ]
        for connection in self.port_to_port_connections:
            zone_0 = connection.zone_port_spec0.zone_id
            zone_1 = connection.zone_port_spec1.zone_id
            port_0 = connection.zone_port_spec0.port_id
            port_1 = connection.zone_port_spec1.port_id
            connections_per_zone_port[zone_0][port_0.value].append((zone_1, port_1))
            connections_per_zone_port[zone_1][port_1.value].append((zone_0, port_0))

        for zone_id, zone in enumerate(self.zones):
            connections_port_0 = connections_per_zone_port[zone_id][0]
            connections_port_1 = connections_per_zone_port[zone_id][1]
            arch_spec_lines.extend(
                [
                    f"Zone {zone_id}:",
                    f"    Max qubits {zone.max_ions_gate_op}",
                    "    Connections:",
                ]
            )
            arch_spec_lines.append(f"       Port 0: Zone {connections_port_0}")
            arch_spec_lines.append(f"       Port 1: Zone {connections_port_1}")
        return f"{os.linesep}".join(arch_spec_lines)
