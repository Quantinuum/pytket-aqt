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

from collections.abc import Generator, Iterable
from copy import deepcopy

import numpy as np
from networkx import bfs_layers

from ..circuit.helpers import TrapConfiguration, get_qubit_to_zone_pos
from .architecture import MultiZoneArchitectureSpec
from .architecture_portgraph import MultiZonePortGraph
from .macro_architecture_graph import MultiZoneArch
from .pebble_hole_graph import PebbleHoleGraph

_ALLOWED_SWAP_SIZE = 2


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
        self._port_graph = MultiZonePortGraph(arch, configuration)
        self._current_config = deepcopy(configuration)
        self.qubit_to_zone_pos = get_qubit_to_zone_pos(
            configuration.n_qubits, configuration.zone_placement
        )
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

    def closest_target_zone_port_path_lengths(
        self,
        src_zone: int,
        src_port: int,
        n_move: int = 1,
        cutoff: int | None = None,
    ) -> list[int | None]:
        return self._port_graph.closest_target_zone_port_path_lengths(
            src_zone, src_port, n_move, cutoff
        )

    def connection_ports(self, zone1: int, zone2: int) -> tuple[int, int]:
        port1, port2 = self._macro_arch.get_connected_ports(zone1, zone2)
        return port1.value, port2.value

    def shuttle_edge_transport_cost(self, zone1: int, zone2: int) -> int:
        port1, port2 = self.connection_ports(zone1, zone2)
        return self._port_graph.shuttle_edge_transport_cost(zone1, port1, zone2, port2)

    def connected_zones(self, zone: int) -> Iterable[int]:
        return self._macro_arch.connected_zones(zone)

    def connected_zones_per_port(self, zone: int) -> tuple[list[int], list[int]]:
        return self._macro_arch.connected_zones_per_port(zone)

    @property
    def n_zones(self) -> int:
        return self._arch.n_zones

    @property
    def n_qubits(self) -> int:
        return self._current_config.n_qubits

    @property
    def architecture_spec(self) -> MultiZoneArchitectureSpec:
        return self._arch

    @property
    def has_memory_zones(self) -> bool:
        return self._macro_arch.has_memory_zones

    @property
    def is_linear_architecture(self) -> bool:
        return self._macro_arch.is_linear_architecture

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

    def swap_qubits_in_zone(self, zone: int, qubit0: int, qubit1: int) -> None:
        zone_qubits = self._current_config.zone_placement[zone]
        if len(zone_qubits) != _ALLOWED_SWAP_SIZE or set(zone_qubits) != {
            qubit0,
            qubit1,
        }:
            raise ValueError(
                "Zone swaps are only supported when the zone contains exactly the two qubits to be swapped."
            )
        zone_qubits.reverse()
        self.qubit_to_zone_pos = get_qubit_to_zone_pos(
            self._current_config.n_qubits, self._current_config.zone_placement
        )

    def copy_dynamic_state_from(self, other: "DynamicArch") -> None:
        self._current_config = deepcopy(other.trap_configuration)
        self._port_graph = MultiZonePortGraph(self._arch, self._current_config)
        self.qubit_to_zone_pos = get_qubit_to_zone_pos(
            self._current_config.n_qubits, self._current_config.zone_placement
        )
        self.zone_occupancy = np.array(
            [len(zone) for zone in self._current_config.zone_placement], dtype=np.int64
        )
        self.transport_free_space = self.zone_max_transport_cap - self.zone_occupancy

    def is_gate_zone(self, zone: int) -> bool:
        return not self._arch.zones[zone].memory_only

    def macro_graph_bfs_layers(
        self, starting_zone: int
    ) -> Generator[list[int], None, None]:
        return bfs_layers(self._macro_arch.zone_graph, starting_zone)

    def pebble_hole_graph(self) -> PebbleHoleGraph:
        return PebbleHoleGraph(self._arch, self._current_config)


class LinearDynamicArch(DynamicArch):
    """Dynamic architecture specialization for linear macro architectures."""

    def __init__(
        self, arch: MultiZoneArchitectureSpec, configuration: TrapConfiguration
    ):
        super().__init__(arch, configuration)
        if not self.is_linear_architecture:
            raise ValueError(
                "LinearDynamicArch can only be used with linear architectures."
            )
        self._line_start_zone = self._compute_line_start_zone()
        self._linearly_ordered_zones = self._compute_linearly_ordered_zones()
        self._ordered_zone_positions = {
            zone: i for i, zone in enumerate(self._linearly_ordered_zones)
        }

    @classmethod
    def from_dynamic_arch(cls, dyn_arch: DynamicArch) -> "LinearDynamicArch":
        return cls(dyn_arch.architecture_spec, dyn_arch.trap_configuration)

    def _compute_line_start_zone(self) -> int:
        for zone in range(self.n_zones):
            connected_zones_0, connected_zones_1 = self.connected_zones_per_port(zone)
            if not connected_zones_0 and connected_zones_1:
                return zone
        raise ValueError("Could not determine the start zone of linear architecture.")

    def _compute_linearly_ordered_zones(self) -> tuple[int, ...]:
        ordered_zones = [self._line_start_zone]
        previous_zone: int | None = None
        current_zone = self._line_start_zone

        while True:
            next_zones = [
                zone
                for zone in self.connected_zones(current_zone)
                if zone != previous_zone
            ]
            if not next_zones:
                return tuple(ordered_zones)
            if len(next_zones) > 1:
                raise ValueError("Linear architecture contains a branching zone order.")
            previous_zone, current_zone = current_zone, next_zones[0]
            ordered_zones.append(current_zone)

    @property
    def line_start_zone(self) -> int:
        return self._line_start_zone

    @property
    def linearly_ordered_zones(self) -> tuple[int, ...]:
        return self._linearly_ordered_zones

    @property
    def ordered_zone_positions(self) -> dict[int, int]:
        return self._ordered_zone_positions.copy()

    def ordered_qubits(self) -> list[int]:
        return [
            qubit
            for zone in self._linearly_ordered_zones
            for qubit in self.trap_configuration.zone_placement[zone]
        ]


def require_linear_dynamic_arch(dyn_arch: DynamicArch) -> LinearDynamicArch:
    if not isinstance(dyn_arch, LinearDynamicArch):
        raise ValueError(
            "This operation requires a LinearDynamicArch input for a linear architecture."
        )
    return dyn_arch


class SgzlDynamicArch(LinearDynamicArch):
    """Linear dynamic architecture specialization with exactly one gate zone."""

    def __init__(
        self, arch: MultiZoneArchitectureSpec, configuration: TrapConfiguration
    ):
        super().__init__(arch, configuration)
        if len(self.gate_zones) != 1:
            raise ValueError(
                "SgzlDynamicArch requires a linear architecture with exactly one gate zone."
            )
        self._single_gate_zone = self.gate_zones[0]
        gate_zone_position = self._ordered_zone_positions[self._single_gate_zone]
        self._interval_capacities = (
            sum(
                int(self.zone_max_gate_cap[zone])
                for zone in self._linearly_ordered_zones[:gate_zone_position]
            ),
            int(self.zone_max_gate_cap[self._single_gate_zone]),
            sum(
                int(self.zone_max_gate_cap[zone])
                for zone in self._linearly_ordered_zones[gate_zone_position + 1 :]
            ),
        )

    @classmethod
    def from_linear_dynamic_arch(cls, dyn_arch: LinearDynamicArch) -> "SgzlDynamicArch":
        return cls(dyn_arch.architecture_spec, dyn_arch.trap_configuration)

    @classmethod
    def from_dynamic_arch(cls, dyn_arch: DynamicArch) -> "SgzlDynamicArch":
        return cls(dyn_arch.architecture_spec, dyn_arch.trap_configuration)

    @property
    def single_gate_zone(self) -> int:
        return self._single_gate_zone

    @property
    def interval_capacities(self) -> tuple[int, int, int]:
        return self._interval_capacities

    @property
    def left_capacity(self) -> int:
        return self._interval_capacities[0]

    @property
    def gate_capacity(self) -> int:
        return self._interval_capacities[1]

    @property
    def right_capacity(self) -> int:
        return self._interval_capacities[2]

    def interval_counts(self, target_gate_qubits: list[int]) -> tuple[int, int, int]:
        qubit_positions = {qubit: i for i, qubit in enumerate(self.ordered_qubits())}
        target_positions = sorted(
            qubit_positions[qubit] for qubit in target_gate_qubits
        )
        left_count = target_positions[0]
        interval_count = target_positions[-1] - target_positions[0] + 1
        right_count = len(qubit_positions) - target_positions[-1] - 1
        return left_count, interval_count, right_count


def require_sgzl_dynamic_arch(dyn_arch: DynamicArch) -> SgzlDynamicArch:
    if not isinstance(dyn_arch, SgzlDynamicArch):
        raise ValueError(
            "This operation requires a SgzlDynamicArch input for a linear architecture with exactly one gate zone."
        )
    return dyn_arch
