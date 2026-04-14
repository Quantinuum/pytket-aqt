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

from collections.abc import Generator
from copy import deepcopy

import numpy as np
from networkx import bfs_layers

from ..circuit.helpers import TrapConfiguration, get_qubit_to_zone_pos
from .architecture import MultiZoneArchitectureSpec
from .architecture_portgraph import MultiZonePortGraph
from .macro_architecture_graph import MultiZoneArch


class DynamicArch:
    """Dynamic Architecture class

    This class combines both static and dynamic architectures information
    and can be considered a mutable snapshot of the current state
    of the trap architecture including placement of qubits in zones

    """

    def __init__(
        self, arch: MultiZoneArchitectureSpec, configuration: TrapConfiguration
    ):
        # static  (doesn't change with qubit movement)
        self._arch = arch
        self._macro_arch = MultiZoneArch(arch)
        self.zone_max_gate_cap = np.array(
            [arch.get_zone_max_ions_gates(zone) for zone in range(arch.n_zones)]
        )
        self.zone_max_transport_cap = np.array(
            [arch.get_zone_max_ions_transport(zone) for zone in range(arch.n_zones)]
        )
        self.zone_swap_costs = np.array([zone.swap_cost for zone in arch.zones])

        # dynamic (changes with qubit movement
        self._current_config = deepcopy(configuration)
        self.qubit_to_zone_pos = get_qubit_to_zone_pos(
            configuration.n_qubits, configuration.zone_placement
        )
        self._port_graph = MultiZonePortGraph(arch, configuration)
        self.zone_occupancy = np.array(
            [len(zone) for zone in self._current_config.zone_placement], dtype=np.int64
        )
        self.transport_free_space = self.zone_max_transport_cap - self.zone_occupancy
        self._n_gate_zone_spots = int(
            sum(
                self.zone_max_gate_cap[gate_zone]
                for gate_zone in self._macro_arch.gate_zones
            )
        )
        self._largest_gate_zone_max_capacity = int(
            max(
                self.zone_max_gate_cap[gate_zone]
                for gate_zone in self._macro_arch.gate_zones
            )
        )

    def shuttle_only_shortest_path_and_path_capacity(
        self, src_zone: int, trg_zone: int
    ) -> tuple[int, list[int], int]:
        length, shortest_path = self._macro_arch.shortest_path_with_length(
            src_zone, trg_zone
        )
        max_transport = self.transport_free_space[shortest_path[1:]].min()
        return length, shortest_path, max_transport

    def shortest_port_path_length(
        self, src_zone: int, src_port: int, trg_zone: int, n_move: int
    ) -> tuple[list[int], int, int] | tuple[None, None, None]:
        return self._port_graph.shortest_port_path_length(
            src_zone, src_port, trg_zone, n_move
        )

    def connection_ports(self, zone1: int, zone2: int) -> tuple[int, int]:
        port1, port2 = self._macro_arch.get_connected_ports(zone1, zone2)
        return port1.value, port2.value

    @property
    def n_zones(self) -> int:
        return self._arch.n_zones

    @property
    def n_qubits(self) -> int:
        return self._current_config.n_qubits

    @property
    def has_memory_zones(self) -> bool:
        return self._macro_arch.has_memory_zones

    @property
    def gate_zones(self) -> list[int]:
        return self._macro_arch.gate_zones

    @property
    def trap_configuration(self) -> TrapConfiguration:
        return self._current_config

    @property
    def n_gate_zone_spots(self) -> int:
        return self._n_gate_zone_spots

    @property
    def largest_gate_zone_max_capacity(self) -> int:
        return self._largest_gate_zone_max_capacity

    def move_qubits(
        self, qubits: list[int], src_zone: int, trg_zone: int, trg_port: int
    ) -> None:
        # update config
        for qubit in qubits:
            self._current_config.zone_placement[src_zone].remove(qubit)
        if trg_port == 0:
            self._current_config.zone_placement[trg_zone] = (
                qubits + self._current_config.zone_placement[trg_zone]
            )
        else:
            self._current_config.zone_placement[trg_zone].extend(qubits)
        # update port graph weights
        for zone in [src_zone, trg_zone]:
            self._port_graph.update_zone_occupancy_weight(
                zone, len(self._current_config.zone_placement[zone])
            )
        # update qubit_to_zone_pos
        self.qubit_to_zone_pos = get_qubit_to_zone_pos(
            self._current_config.n_qubits, self._current_config.zone_placement
        )
        # update zone_occupancy
        n_move = len(qubits)
        self.zone_occupancy[src_zone] -= n_move
        self.zone_occupancy[trg_zone] += n_move
        # update transport_free_space
        self.transport_free_space = self.zone_max_transport_cap - self.zone_occupancy

    def is_gate_zone(self, zone: int) -> bool:
        return not self._arch.zones[zone].memory_only

    def macro_graph_bfs_layers(
        self, starting_zone: int
    ) -> Generator[list[int], None, None]:
        return bfs_layers(self._macro_arch.zone_graph, starting_zone)
