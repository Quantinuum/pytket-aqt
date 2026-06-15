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

"""Pre-defined named multi-zone architectures for use in multi-zone circuits"""

from .architecture import (
    Junction,
    JunctionRef,
    LayoutPosition,
    MultiZoneArchitectureSpec,
    Operation,
    PhysicalConnection,
    PortId,
    PortSpec,
    VisualizationSpec,
    Zone,
)

standardOperations = [
    Operation(operation_spec="[X, t, [self, o, p]]", fidelity="0.993"),
    Operation(operation_spec="[MS, t, [[self, o, p], [self, o, p]]]", fidelity="0.983"),
]


def _port(zone_id: int, port_id: PortId) -> PortSpec:
    return PortSpec(zone_id=zone_id, port_id=port_id)


def _junction(junction_id: int) -> JunctionRef:
    return JunctionRef(junction_id=junction_id)


def get_direct_physical_connections(
    zone_ports: list[tuple[int, PortId, int, PortId]],
) -> list[PhysicalConnection]:
    return [
        PhysicalConnection(
            endpoint0=_port(zone0, port0),
            endpoint1=_port(zone1, port1),
        )
        for zone0, port0, zone1, port1 in zone_ports
    ]


def get_linear_connections(n_zones: int) -> list[PhysicalConnection]:
    return get_direct_physical_connections(
        [(i, PortId.p1, i + 1, PortId.p0) for i in range(n_zones - 1)]
    )


def get_junction_connections(
    junction_id: int, zone_ports: list[tuple[int, PortId]]
) -> list[PhysicalConnection]:
    return [
        PhysicalConnection(
            endpoint0=_port(zone_id, port_id),
            endpoint1=_junction(junction_id),
        )
        for zone_id, port_id in zone_ports
    ]


four_zones_in_a_line = MultiZoneArchitectureSpec(
    n_qubits_max=16,
    n_zones=4,
    zones=[
        Zone(max_ions_gate_op=mi, memory_only=mem)
        for mi, mem in [(8, True), (6, False), (6, False), (8, True)]
    ],
    connections=get_linear_connections(4),
)

_SIQCI_GATE_ZONE = 3
siqci_arch = MultiZoneArchitectureSpec(
    n_qubits_max=5,
    n_zones=5,
    zones=[
        Zone(
            max_ions_gate_op=2,
            max_ions_transport_op=2,
            memory_only=i != _SIQCI_GATE_ZONE,
        )
        for i in range(5)
    ],
    connections=get_linear_connections(5),
)

_SIQCI_GATE_ZONE_MIDDLE = 2
siqci_arch_g2 = MultiZoneArchitectureSpec(
    n_qubits_max=5,
    n_zones=5,
    zones=[
        Zone(
            max_ions_gate_op=2,
            max_ions_transport_op=2,
            memory_only=i != _SIQCI_GATE_ZONE_MIDDLE,
        )
        for i in range(5)
    ],
    connections=get_linear_connections(5),
)

_SIQCI_GATE_ZONE_EDGE = 4
siqci_arch_g4 = MultiZoneArchitectureSpec(
    n_qubits_max=5,
    n_zones=5,
    zones=[
        Zone(
            max_ions_gate_op=2,
            max_ions_transport_op=2,
            memory_only=i != _SIQCI_GATE_ZONE_EDGE,
        )
        for i in range(5)
    ],
    connections=get_linear_connections(5),
)

_LINEAR_8_ZONES_GATE_ZONE = 3

linear_8_zones = MultiZoneArchitectureSpec(
    n_qubits_max=30,
    n_zones=8,
    zones=[
        Zone(
            max_ions_gate_op=6,
            max_ions_transport_op=7,
            memory_only=i != _LINEAR_8_ZONES_GATE_ZONE,
        )
        for i in range(8)
    ],
    connections=get_linear_connections(8),
)

_LINEAR_9_ZONES_GATE_ZONE = 4

linear_9_zones = MultiZoneArchitectureSpec(
    n_qubits_max=30,
    n_zones=9,
    zones=[
        Zone(
            max_ions_gate_op=6,
            max_ions_transport_op=7,
            memory_only=i != _LINEAR_9_ZONES_GATE_ZONE,
        )
        for i in range(9)
    ],
    connections=get_linear_connections(9),
)


racetrack_max_ions = 6
racetrack = MultiZoneArchitectureSpec(
    n_qubits_max=84,
    n_zones=28,
    zones=[Zone(max_ions_gate_op=racetrack_max_ions) for _ in range(28)],
    connections=get_direct_physical_connections(
        [(i % 28, PortId.p1, (i + 1) % 28, PortId.p0) for i in range(28)]
    ),
)

racetrack_4_gatezones = MultiZoneArchitectureSpec(
    n_qubits_max=84,
    n_zones=28,
    zones=[
        (
            Zone(max_ions_gate_op=racetrack_max_ions)
            if i in [0, 1, 2, 3]
            else Zone(max_ions_gate_op=racetrack_max_ions, memory_only=True)
        )
        for i in range(28)
    ],
    connections=get_direct_physical_connections(
        [(i % 28, PortId.p1, (i + 1) % 28, PortId.p0) for i in range(28)]
    ),
)


"""
grid12:

|- 0 -|- 1 -|
2     3     4
|- 5 -|- 6 -|
7     8     9
|- 10-|- 11-|

for horizontal zones port 0 is left, port 1 is right
for vertical zones port 0 is up, port 1 is down
"""
grid_zone_max_ion = 8
grid12_junction_groups = [
    [(0, PortId.p0), (2, PortId.p0)],
    [(0, PortId.p1), (1, PortId.p0), (3, PortId.p0)],
    [(1, PortId.p1), (4, PortId.p0)],
    [(2, PortId.p1), (5, PortId.p0), (7, PortId.p0)],
    [(3, PortId.p1), (6, PortId.p0), (5, PortId.p1), (8, PortId.p0)],
    [(6, PortId.p1), (4, PortId.p1), (9, PortId.p0)],
    [(7, PortId.p1), (10, PortId.p0)],
    [(8, PortId.p1), (10, PortId.p1), (11, PortId.p0)],
    [(9, PortId.p1), (11, PortId.p1)],
]
grid12_junctions = [
    Junction(junction_id=junction_id)
    for junction_id in range(len(grid12_junction_groups))
]
grid12_connections = [
    connection
    for junction_id, zone_ports in enumerate(grid12_junction_groups)
    for connection in get_junction_connections(junction_id, zone_ports)
]
grid12_visualization = VisualizationSpec(
    zone_positions={
        0: LayoutPosition(x=0.5, y=0.0),
        1: LayoutPosition(x=1.5, y=0.0),
        2: LayoutPosition(x=0.0, y=0.5),
        3: LayoutPosition(x=1.0, y=0.5),
        4: LayoutPosition(x=2.0, y=0.5),
        5: LayoutPosition(x=0.5, y=1.0),
        6: LayoutPosition(x=1.5, y=1.0),
        7: LayoutPosition(x=0.0, y=1.5),
        8: LayoutPosition(x=1.0, y=1.5),
        9: LayoutPosition(x=2.0, y=1.5),
        10: LayoutPosition(x=0.5, y=2.0),
        11: LayoutPosition(x=1.5, y=2.0),
    },
    junction_positions={
        0: LayoutPosition(x=0.0, y=0.0),
        1: LayoutPosition(x=1.0, y=0.0),
        2: LayoutPosition(x=2.0, y=0.0),
        3: LayoutPosition(x=0.0, y=1.0),
        4: LayoutPosition(x=1.0, y=1.0),
        5: LayoutPosition(x=2.0, y=1.0),
        6: LayoutPosition(x=0.0, y=2.0),
        7: LayoutPosition(x=1.0, y=2.0),
        8: LayoutPosition(x=2.0, y=2.0),
    },
)

grid12 = MultiZoneArchitectureSpec(
    n_qubits_max=32,
    n_zones=12,
    zones=[Zone(max_ions_gate_op=grid_zone_max_ion) for _ in range(12)],
    junctions=grid12_junctions,
    connections=grid12_connections,
    visualization=grid12_visualization,
)

grid_zone_max_ion = 8
grid12_mod = MultiZoneArchitectureSpec(
    n_qubits_max=32,
    n_zones=12,
    zones=[
        (
            Zone(max_ions_gate_op=grid_zone_max_ion)
            if i in [2, 4, 7, 9]
            else Zone(max_ions_gate_op=grid_zone_max_ion, memory_only=True)
        )
        for i in range(12)
    ],
    junctions=grid12_junctions,
    connections=grid12_connections,
    visualization=grid12_visualization,
)


"""
gate_zone_type_examples:

Has gate zones in terminal (i.e. only one connected zone) positions (0, 10)
 and with two (4), three (6), and 6 (11) connected zones

* = gate zone

0* -- 1 -|- 2 -|- 3 -- 4* -- 5
         6*    7
         |- 8 -|- 9 -- 10*
               11*
           12 -|- 13
               14

for horizontal zones port 0 is left, port 1 is right
for vertical zones port 0 is up, port 1 is down
"""
n_zones = 15
zone_max = 4
gate_zone_type_junction_groups = [
    [(0, PortId.p1), (1, PortId.p0)],
    [(1, PortId.p1), (2, PortId.p0), (6, PortId.p0)],
    [(2, PortId.p1), (3, PortId.p0), (7, PortId.p0)],
    [(3, PortId.p1), (4, PortId.p0)],
    [(4, PortId.p1), (5, PortId.p0)],
    [(6, PortId.p1), (8, PortId.p0)],
    [(8, PortId.p1), (9, PortId.p0), (7, PortId.p1), (11, PortId.p0)],
    [(9, PortId.p1), (10, PortId.p0)],
    [(12, PortId.p1), (13, PortId.p0), (11, PortId.p1), (14, PortId.p0)],
]
gate_zone_type_junctions = [
    Junction(junction_id=junction_id)
    for junction_id in range(len(gate_zone_type_junction_groups))
]
gate_zone_type_junction_connections = [
    connection
    for junction_id, zone_ports in enumerate(gate_zone_type_junction_groups)
    for connection in get_junction_connections(junction_id, zone_ports)
]
gate_zone_type_visualization = VisualizationSpec(
    zone_positions={
        0: LayoutPosition(x=0.0, y=0.0),
        1: LayoutPosition(x=1.0, y=0.0),
        2: LayoutPosition(x=2.0, y=0.0),
        3: LayoutPosition(x=3.0, y=0.0),
        4: LayoutPosition(x=4.0, y=0.0),
        5: LayoutPosition(x=5.0, y=0.0),
        6: LayoutPosition(x=1.5, y=0.5),
        7: LayoutPosition(x=2.5, y=0.5),
        8: LayoutPosition(x=2.0, y=1.0),
        9: LayoutPosition(x=3.0, y=1.0),
        10: LayoutPosition(x=4.0, y=1.0),
        11: LayoutPosition(x=2.5, y=1.5),
        12: LayoutPosition(x=2.0, y=2.0),
        13: LayoutPosition(x=3.0, y=2.0),
        14: LayoutPosition(x=2.5, y=2.5),
    },
    junction_positions={
        0: LayoutPosition(x=0.5, y=0.0),
        1: LayoutPosition(x=1.5, y=0.0),
        2: LayoutPosition(x=2.5, y=0.0),
        3: LayoutPosition(x=3.5, y=0.0),
        4: LayoutPosition(x=4.5, y=0.0),
        5: LayoutPosition(x=1.5, y=1.0),
        6: LayoutPosition(x=2.5, y=1.0),
        7: LayoutPosition(x=3.5, y=1.0),
        8: LayoutPosition(x=2.5, y=2.0),
    },
)
gate_zone_type_examples = MultiZoneArchitectureSpec(
    n_qubits_max=30,
    n_zones=n_zones,
    zones=[
        (
            Zone(max_ions_gate_op=zone_max)
            if i in [0, 4, 6, 10, 11]
            else Zone(max_ions_gate_op=zone_max, memory_only=True)
        )
        for i in range(n_zones)
    ],
    junctions=gate_zone_type_junctions,
    connections=[
        *gate_zone_type_junction_connections,
    ],
    visualization=gate_zone_type_visualization,
)
