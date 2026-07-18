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
from collections.abc import Callable
from typing import cast

from networkx import (
    Graph,
    single_source_dijkstra,
)
from networkx.exception import NetworkXNoPath

from ..circuit.helpers import TrapConfiguration
from .architecture import MultiZoneArchitectureSpec


def zone_port_to_port_id(zone: int, port: int) -> int:
    return 2 * zone + port


def port_id_to_zone_port(port_id: int) -> tuple[int, int]:
    return port_id // 2, port_id % 2


def port_path_to_zone_path(port_path: list[int]) -> list[int]:
    """Translate a path through zone ports to a path through zones

    A port path starts with the port of a zone (s_p), moves through the two ports
    of n >= 0 intermediary zones (i_p0, i_p1) and stops at a target port of a zone (t_p)

    s_p -> 0_p0 -> 0_p1 -> ... -> n_p0 -> n_p1 -> t_p
    """
    # add source zone
    result = [port_id_to_zone_port(port_path[0])[0]]
    for port_id in port_path[1:]:
        zone, _ = port_id_to_zone_port(port_id)
        if result[-1] != zone:
            result.append(zone)
    return result


def shortest_path_and_its_length(
    g: Graph,
    source: int,
    target: int,
    weight: str | Callable[[int, int, dict[str, int]], int | None],
) -> tuple[int, list[int]]:
    length, path = single_source_dijkstra(g, source, target, weight=weight)
    assert isinstance(length, int)
    assert isinstance(path, list)
    return length, path


class MultiZonePortGraph:
    def __init__(
        self, spec: MultiZoneArchitectureSpec, start_config: TrapConfiguration
    ):
        self.swap_costs = [zone.swap_cost for zone in spec.zones]

        self.port_graph: Graph[int] = Graph()
        self.n_zones = spec.n_zones
        placement = start_config.zone_placement

        # Add "capacity" edges between the two ports of a single zone.
        # weight = occupancy = 0 for now but later will be dynamically set to current occupancy
        for zone_id, zone in enumerate(spec.zones):
            zone_port0_id = zone_port_to_port_id(zone_id, 0)
            zone_port1_id = zone_port_to_port_id(zone_id, 1)
            occupancy = len(placement[zone_id])
            transport_max_cap = zone.max_ions_transport_op
            transport_capacity = transport_max_cap - occupancy
            self.port_graph.add_node(zone_port0_id)
            self.port_graph.add_node(zone_port1_id)
            self.port_graph.add_edge(
                zone_port0_id,
                zone_port1_id,
                max_cap_transport=transport_max_cap,
                transport_capacity=transport_capacity,
                occupancy=occupancy,
                transport_cost=occupancy * zone.swap_cost,
                is_shuttle_edge=False,
            )

        # Add "shuttle" edges between connected zones.
        for connection in spec.port_to_port_connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            portid0 = zone_port_to_port_id(zone0, port0.value)
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id
            portid1 = zone_port_to_port_id(zone1, port1.value)
            # TODO: update arch spec to include connection shuttle cost and use that as weight
            self.port_graph.add_edge(
                portid0,
                portid1,
                transport_cost=connection.shuttle_cost,
                is_shuttle_edge=True,
            )

        self._path_cache: dict[
            tuple[int, int, int, int],
            tuple[list[int], int, int] | tuple[None, None, None],
        ] = {}
        self._target_zone_path_lengths_cache: dict[
            tuple[int, int, int, int | None], list[int | None]
        ] = {}

    def update_zone_occupancy_weight(self, zone: int, zone_occupancy: int) -> None:
        edge_dict = self.port_graph.edges[
            zone_port_to_port_id(zone, 0), zone_port_to_port_id(zone, 1)
        ]
        edge_dict["occupancy"] = zone_occupancy
        # transport capacity (from port to port of a zone) is set to the amount of free space in the zone
        edge_dict["transport_capacity"] = (
            edge_dict["max_cap_transport"] - zone_occupancy
        )
        edge_dict["transport_cost"] = zone_occupancy * self.swap_costs[zone]
        self._path_cache.clear()
        self._target_zone_path_lengths_cache.clear()

    def shuttle_edge_transport_cost(
        self, src_zone: int, src_port: int, trg_zone: int, trg_port: int
    ) -> int:
        edge_dict = self.port_graph.edges[
            zone_port_to_port_id(src_zone, src_port),
            zone_port_to_port_id(trg_zone, trg_port),
        ]
        return int(edge_dict["transport_cost"])

    def shortest_port_to_port_path_length(
        self,
        start_zone: int,
        start_port: int,
        targ_zone: int,
        targ_port: int,
        n_move: int = 1,
    ) -> tuple[list[int], int, int] | tuple[None, None, None]:
        """Return the shortest path length for going from starting (zone, port) target (zone, port)"""
        raise NotImplementedError

    def shortest_port_path_length(
        self, start_zone: int, start_port: int, targ_zone: int, n_move: int = 1
    ) -> tuple[list[int], int, int] | tuple[None, None, None]:
        """Return the shortest path length for going from starting (zone, port) "closest" port of a target zone

        This algorithm assumes that transport of at least 1 qubit is possible between start and target
        zones, i.e. that none of the zones in between are transport blocked. Only the
        start zone may be transport blocked.

        :param start_zone: The zone we are moving out of
        :param start_port: The port of the start zone we are starting at
        :param targ_zone: The zone we want to move to
        :param n_move: The number of qubits we want to move simultaneously

        :returns: None if there is no path that can move the desired number of qubits. Otherwise,
        the return value is a tuple. The first value is the shortest zone path
        from the starting (zone, port) to the closest port of the target zone. The second
        is the calculated port path length. The third is the corresponding closest port of the
        target zone.
        """
        cache_result = self._path_cache.get(
            (start_zone, start_port, targ_zone, n_move), None
        )
        if cache_result:
            return cache_result
        result = self.uncached_shortest_port_path_length(
            start_zone, start_port, targ_zone, n_move
        )
        self._path_cache[(start_zone, start_port, targ_zone, n_move)] = result
        return result

    def closest_target_zone_port_path_lengths(
        self,
        start_zone: int,
        start_port: int,
        n_move: int = 1,
        cutoff: int | None = None,
    ) -> list[int | None]:
        """Return shortest path lengths from a source port to each target zone.

        For each target zone, the returned entry is the distance to the cheaper
        of the target zone's two ports. A value of ``None`` means no target port
        was reachable within the optional cutoff.
        """
        cache_key = (start_zone, start_port, n_move, cutoff)
        cache_result = self._target_zone_path_lengths_cache.get(cache_key)
        if cache_result is not None:
            return cache_result

        port_id_start = zone_port_to_port_id(start_zone, start_port)

        def move_weight(u: int, v: int, d: dict[str, int]) -> int | None:
            if d["is_shuttle_edge"]:
                return d["transport_cost"]
            return (
                d["transport_cost"] * n_move
                if d["transport_capacity"] >= n_move
                else None
            )

        raw_lengths, _ = single_source_dijkstra(
            self.port_graph,
            port_id_start,
            cutoff=cutoff,
            weight="transport_cost" if n_move == 1 else move_weight,
        )
        lengths = cast("dict[int, int]", raw_lengths)
        result: list[int | None] = []
        for target_zone in range(self.n_zones):
            target_port_0 = zone_port_to_port_id(target_zone, 0)
            target_port_1 = zone_port_to_port_id(target_zone, 1)
            length_0 = lengths.get(target_port_0)
            length_1 = lengths.get(target_port_1)
            if length_0 is None:
                result.append(length_1)
            elif length_1 is None:
                result.append(length_0)
            else:
                result.append(min(length_0, length_1))

        self._target_zone_path_lengths_cache[cache_key] = result
        return result

    def uncached_shortest_port_path_length(
        self, start_zone: int, start_port: int, targ_zone: int, n_move: int = 1
    ) -> tuple[list[int], int, int] | tuple[None, None, None]:
        """Return the shortest path length for going from starting (zone, port) "closest" port of a target zone

        This algorithm assumes that transport of at least 1 qubit is possible between start and target
        zones, i.e. that none of the zones in between are transport blocked. Only the
        start zone may be transport blocked.

        :param start_zone: The zone we are moving out of
        :param start_port: The port of the start zone we are starting at
        :param targ_zone: The zone we want to move to
        :param n_move: The number of qubits we want to move simultaneously

        :returns: None if there is no path that can move the desired number of qubits. Otherwise,
        the return value is a tuple. The first value is the shortest zone path
        from the starting (zone, port) to the closest port of the target zone. The second
        is the calculated port path length. The third is the corresponding closest port of the
        target zone.
        """

        port_id_start = zone_port_to_port_id(start_zone, start_port)
        port_idt0 = zone_port_to_port_id(targ_zone, 0)
        port_idt1 = zone_port_to_port_id(targ_zone, 1)

        def move_weight(u: int, v: int, d: dict[str, int]) -> int | None:
            if d["is_shuttle_edge"]:
                return d["transport_cost"]
            return (
                d["transport_cost"] * n_move
                if d["transport_capacity"] >= n_move
                else None
            )

        if n_move == 1:
            length_s0t0, path_s0t0 = shortest_path_and_its_length(
                self.port_graph, port_id_start, port_idt0, weight="transport_cost"
            )

            length_s0t1, path_s0t1 = shortest_path_and_its_length(
                self.port_graph, port_id_start, port_idt1, weight="transport_cost"
            )
            assert isinstance(length_s0t1, int)
            return (
                (port_path_to_zone_path(path_s0t0), length_s0t0, 0)
                if length_s0t0 <= length_s0t1
                else (port_path_to_zone_path(path_s0t1), length_s0t1, 1)
            )
        path_exists0 = True
        path_exists1 = True
        try:
            length_s0t0, path_s0t0 = shortest_path_and_its_length(
                self.port_graph, port_id_start, port_idt0, weight=move_weight
            )
        except NetworkXNoPath:
            length_s0t0, path_s0t0 = 0, []
            path_exists0 = False
        try:
            length_s0t1, path_s0t1 = shortest_path_and_its_length(
                self.port_graph, port_id_start, port_idt1, weight=move_weight
            )
        except NetworkXNoPath:
            length_s0t1, path_s0t1 = 0, []
            path_exists1 = False
        match (path_exists0, path_exists1):
            case (True, True):
                return (
                    (port_path_to_zone_path(path_s0t0), length_s0t0, 0)
                    if length_s0t0 <= length_s0t1
                    else (port_path_to_zone_path(path_s0t1), length_s0t1, 1)
                )
            case (True, False):
                return port_path_to_zone_path(path_s0t0), length_s0t0, 0
            case (False, True):
                return port_path_to_zone_path(path_s0t1), length_s0t1, 1
            case _:
                return None, None, None
