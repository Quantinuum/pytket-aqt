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
from typing import Self

from pydantic import BaseModel, model_validator


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

    zone_id: int
    port_id: PortId


class ZoneConnection(BaseModel):
    """A connection between two zone ports

    The connection allows shuttling between the zones
    """

    zone_port_spec0: PortSpec
    zone_port_spec1: PortSpec


class Operation(BaseModel):
    """Describes an allowed operation and its associated fidelity

    Currently not used
    """

    operation_spec: str
    fidelity: float | str


class Zone(BaseModel):
    """Processor Zone within the architecture"""

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
        if self.max_ions_transport_op <= self.max_ions_gate_op:
            raise ValueError(
                f"'max_ions_transport_op' must be greater than 'max_ions_gate_op'."
                f" Got max_ions_transport_op={self.max_ions_transport_op},"
                f" max_ions_gate_op={self.max_ions_gate_op}"
            )
        return self


class MultiZoneArchitectureSpec(BaseModel):
    """Class that determines the entire Multi-Zone Architecture"""

    n_qubits_max: int
    n_zones: int
    zones: list[Zone]
    connections: list[ZoneConnection]

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
        for connection in self.connections:
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
