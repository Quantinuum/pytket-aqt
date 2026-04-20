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

from itertools import combinations

from .architecture import (
    MultiZoneArchitectureSpec,
    Operation,
    PortId,
    PortSpec,
    Zone,
    ZoneConnection,
)

standardOperations = [
    Operation(operation_spec="[X, t, [self, o, p]]", fidelity="0.993"),
    Operation(operation_spec="[MS, t, [[self, o, p], [self, o, p]]]", fidelity="0.983"),
]


four_zones_in_a_line = MultiZoneArchitectureSpec(
    n_qubits_max=16,
    n_zones=4,
    zones=[
        Zone(max_ions_gate_op=mi, memory_only=mem)
        for mi, mem in [(8, True), (6, False), (6, False), (8, True)]
    ],
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
        )
        for i in range(3)
    ],
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
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
        )
        for i in range(7)
    ],
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
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
        )
        for i in range(8)
    ],
)


racetrack_max_ions = 6
racetrack = MultiZoneArchitectureSpec(
    n_qubits_max=84,
    n_zones=28,
    zones=[Zone(max_ions_gate_op=racetrack_max_ions) for _ in range(28)],
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i % 28, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=(i + 1) % 28, port_id=PortId.p0),
        )
        for i in range(28)
    ],
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
    connections=[
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=i % 28, port_id=PortId.p1),
            zone_port_spec1=PortSpec(zone_id=(i + 1) % 28, port_id=PortId.p0),
        )
        for i in range(28)
    ],
)


def get_all_to_all_port_connections(
    zone_ports: list[tuple[int, PortId]],
) -> list[ZoneConnection]:
    """Return a list of ZoneConnections connecting
    all the zone ports in the given list"""
    return [
        ZoneConnection(
            zone_port_spec0=PortSpec(zone_id=zone_port0[0], port_id=zone_port0[1]),
            zone_port_spec1=PortSpec(zone_id=zone_port1[0], port_id=zone_port1[1]),
        )
        for zone_port0, zone_port1 in combinations(zone_ports, 2)
    ]


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
grid12 = MultiZoneArchitectureSpec(
    n_qubits_max=32,
    n_zones=12,
    zones=[Zone(max_ions_gate_op=grid_zone_max_ion) for _ in range(12)],
    connections=get_all_to_all_port_connections([(0, PortId.p0), (2, PortId.p0)])
    + get_all_to_all_port_connections([(0, PortId.p1), (1, PortId.p0), (3, PortId.p0)])
    + get_all_to_all_port_connections([(1, PortId.p1), (4, PortId.p0)])
    + get_all_to_all_port_connections([(2, PortId.p1), (5, PortId.p0), (7, PortId.p0)])
    + get_all_to_all_port_connections(
        [(3, PortId.p1), (6, PortId.p0), (5, PortId.p1), (8, PortId.p0)]
    )
    + get_all_to_all_port_connections([(6, PortId.p1), (4, PortId.p1), (9, PortId.p0)])
    + get_all_to_all_port_connections([(7, PortId.p1), (10, PortId.p0)])
    + get_all_to_all_port_connections(
        [(8, PortId.p1), (10, PortId.p1), (11, PortId.p0)]
    )
    + get_all_to_all_port_connections([(9, PortId.p1), (11, PortId.p1)]),
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
    connections=get_all_to_all_port_connections([(0, PortId.p0), (2, PortId.p0)])
    + get_all_to_all_port_connections([(0, PortId.p1), (1, PortId.p0), (3, PortId.p0)])
    + get_all_to_all_port_connections([(1, PortId.p1), (4, PortId.p0)])
    + get_all_to_all_port_connections([(2, PortId.p1), (5, PortId.p0), (7, PortId.p0)])
    + get_all_to_all_port_connections(
        [(3, PortId.p1), (6, PortId.p0), (5, PortId.p1), (8, PortId.p0)]
    )
    + get_all_to_all_port_connections([(6, PortId.p1), (4, PortId.p1), (9, PortId.p0)])
    + get_all_to_all_port_connections([(7, PortId.p1), (10, PortId.p0)])
    + get_all_to_all_port_connections(
        [(8, PortId.p1), (10, PortId.p1), (11, PortId.p0)]
    )
    + get_all_to_all_port_connections([(9, PortId.p1), (11, PortId.p1)]),
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
    connections=[
        # horizontal connections
        *[
            ZoneConnection(
                zone_port_spec0=PortSpec(zone_id=i, port_id=PortId.p1),
                zone_port_spec1=PortSpec(zone_id=i + 1, port_id=PortId.p0),
            )
            for i in [0, 1, 2, 3, 4, 8, 9, 12]
        ],
        # vertical connections
        *[
            ZoneConnection(
                zone_port_spec0=PortSpec(zone_id=i, port_id=pi),
                zone_port_spec1=PortSpec(zone_id=j, port_id=pj),
            )
            for i, pi, j, pj in [
                (1, PortId.p1, 6, PortId.p0),
                (2, PortId.p0, 6, PortId.p0),
                (8, PortId.p0, 6, PortId.p1),
                (3, PortId.p1, 7, PortId.p0),
                (4, PortId.p0, 7, PortId.p0),
                (8, PortId.p1, 7, PortId.p1),
                (9, PortId.p0, 7, PortId.p1),
                (11, PortId.p0, 7, PortId.p1),
                (8, PortId.p1, 11, PortId.p0),
                (9, PortId.p0, 11, PortId.p0),
                (12, PortId.p1, 11, PortId.p1),
                (13, PortId.p0, 11, PortId.p1),
                (14, PortId.p0, 11, PortId.p1),
                (12, PortId.p1, 14, PortId.p0),
                (13, PortId.p0, 14, PortId.p0),
            ]
        ],
    ],
)
