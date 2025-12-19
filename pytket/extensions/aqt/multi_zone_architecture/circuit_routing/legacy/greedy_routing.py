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
from copy import deepcopy

from networkx import bfs_layers
from pytket.circuit import Circuit, Command, OpType, Qubit

from ...circuit.helpers import (
    ZonePlacement,
    ZoneRoutingError,
    get_qubit_to_zone,
)
from ...circuit.multizone_circuit import (
    MultiZoneCircuit,
)
from ...trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
)


class QubitTracker:
    """Tracks which qubits are in which zones for the entire architecture"""

    def __init__(self, n_qubits: int, initial_placement: ZonePlacement) -> None:
        self._current_placement = deepcopy(initial_placement)
        self._current_qubit_to_zone = get_qubit_to_zone(n_qubits, initial_placement)

    def current_zone(self, qubit: int) -> int:
        return int(self._current_qubit_to_zone[qubit])

    def zone_occupants(self, zone: int) -> list[int]:
        return self._current_placement[zone]

    def move_qubit(self, qubit: int, starting_zone: int, target_zone: int) -> None:
        self._current_placement[starting_zone].remove(qubit)
        self._current_placement[target_zone].append(qubit)
        self._current_qubit_to_zone[qubit] = target_zone


class GreedyCircuitRouter:
    """Uses a simple greedy algorithm to add shuttles and swaps to a circuit

    Any time an unimplementable gate is detected, the qubits involved are
    immediately moved to a gate zone based on some simple move heuristics

    :param circuit: The circuit to be routed
    :param arch: The architecture to route to
    :param initial_placement: The initial placement of ions in the ion trap zones
    """

    def __init__(
        self,
        circuit: Circuit,
        arch: MultiZoneArchitectureSpec,
        initial_placement: ZonePlacement,
    ):
        self._circuit = circuit
        self._arch = arch
        self._initial_placement = initial_placement

    def get_routed_circuit(self) -> MultiZoneCircuit:  # noqa: PLR0912
        """Returns the routed MultiZoneCircuit"""
        n_qubits = self._circuit.n_qubits
        mz_circuit = MultiZoneCircuit(
            self._arch, self._initial_placement, n_qubits, self._circuit.n_bits
        )

        qubit_tracker = QubitTracker(n_qubits, self._initial_placement)
        waiting_one_qubit_gates: dict[int, list[Command]] = {}

        for cmd in self._circuit.get_commands():
            n_args = len(cmd.args)
            if n_args == 1 or cmd.op.type == OpType.Measure:
                qubit0 = cmd.args[0].index[0]
                if (
                    qubit_tracker.current_zone(qubit0)
                    in mz_circuit.macro_arch.gate_zones
                ):
                    mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
                elif qubit0 in waiting_one_qubit_gates:
                    waiting_one_qubit_gates[qubit0].append(cmd)
                else:
                    waiting_one_qubit_gates[qubit0] = [cmd]
            elif n_args == 2:  # noqa: PLR2004
                qubit0 = cmd.args[0].index[0]
                qubit1 = cmd.args[1].index[0]
                if isinstance(cmd.args[0], Qubit) and isinstance(cmd.args[1], Qubit):
                    _make_necessary_moves_2q(
                        qubit0,
                        qubit1,
                        mz_circuit,
                        qubit_tracker,
                    )
                #  apply waiting one-qubit gates first
                for qubit in (qubit0, qubit1):
                    while waiting_one_qubit_gates.get(qubit):
                        waiting_cmd = waiting_one_qubit_gates[qubit].pop(0)
                        mz_circuit.add_gate(
                            waiting_cmd.op.type, waiting_cmd.args, waiting_cmd.op.params
                        )
                    if qubit in waiting_one_qubit_gates:
                        waiting_one_qubit_gates.pop(qubit)
                mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
            else:
                raise ZoneRoutingError("Circuit must be rebased to the AQT gate set")
        for i, waiting_gates in waiting_one_qubit_gates.items():
            if waiting_gates:
                _make_necessary_moves_1q(i, mz_circuit, qubit_tracker)
                if (
                    qubit_tracker.current_zone(i)
                    not in mz_circuit.macro_arch.gate_zones
                ):
                    raise Exception("This is a problem")
                for cmd in waiting_gates:
                    mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)
        return mz_circuit


def _move_qubit(
    qubit_to_move: int,
    starting_zone: int,
    target_zone: int,
    mz_circ: MultiZoneCircuit,
    qubit_tracker: QubitTracker,
) -> None:
    shortest_path = mz_circ.macro_arch.shortest_path(starting_zone, target_zone)
    mz_circ.move_qubit_precompiled(qubit_to_move, target_zone, shortest_path)
    qubit_tracker.move_qubit(qubit_to_move, starting_zone, target_zone)


def _find_target_zone(
    starting_zone: int,
    potential_swap_zones: list[int],
    mz_circ: MultiZoneCircuit,
    qubit_tracker: QubitTracker,
) -> int:
    # find the closest zone to starting_zone with at least 2 free spots
    # using breadth-first-search
    # potential_swap_zone does not require the free spot check
    targ_zone = -1
    for layer in bfs_layers(mz_circ.macro_arch.zone_graph, starting_zone):
        for zone in layer:
            if zone == starting_zone:
                continue
            zone_occupancy = len(qubit_tracker.zone_occupants(zone))
            max_zone_occupancy = mz_circ.architecture.get_zone_max_ions_gates(zone)
            zone_free_space = max_zone_occupancy - zone_occupancy
            if zone_free_space > 0 or zone in potential_swap_zones:
                targ_zone = zone
                break
        if targ_zone != -1:
            break
    if targ_zone == -1:
        raise Exception("No viable zone found")
    return targ_zone


def _make_necessary_moves_1q(
    qubit0: int,
    mz_circ: MultiZoneCircuit,
    qubit_tracker: QubitTracker,
) -> None:
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move one qubit into a gate zone

    :param qubit0: qubit in gate
    :param mz_circ: the MultiZoneCircuit
    :param qubit_tracker: QubitTracker object for tracking/updating current
     placement of qubits
    """

    zone0 = qubit_tracker.current_zone(qubit0)
    is_gate_zone0 = not mz_circ.architecture.zones[zone0].memory_only
    if is_gate_zone0:
        return
    best_gate_zone, best_gate_zone_free_space = _find_best_gate_zone_to_move_to(
        [zone0], mz_circ, qubit_tracker
    )
    match best_gate_zone_free_space:
        case 0:
            # find first qubit in gate_zone that isn't involved (we know that
            # any qubit currently in a gate zone no longer requires the application
            # of gates, otherwise it would
            # have been done already, so no need to check for that)
            qubit_to_remove = qubit_tracker.zone_occupants(best_gate_zone)[0]
            potential_swap_zones = [zone0]
            target_zone = _find_target_zone(
                best_gate_zone, potential_swap_zones, mz_circ, qubit_tracker
            )
            _move_qubit(
                qubit_to_remove, best_gate_zone, target_zone, mz_circ, qubit_tracker
            )
            _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
        case a if a > 0:
            _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)


def _gate_zone_metric(
    gate_zone: int,
    zones: list[int],
    gate_zone_free_space: int,
    mz_circ: MultiZoneCircuit,
) -> int:
    """Calculate metric estimating cost of moving one qubit from each of a
     list of zones to a specific gate zone

    Prefers gate zones with low total distance and high free space
    """
    distances = [
        len(mz_circ.macro_arch.shortest_path(gate_zone, zone)) for zone in zones
    ]
    return sum(distances) - gate_zone_free_space * len(distances)


def _find_best_gate_zone_to_move_to(
    zones: list[int], mz_circ: MultiZoneCircuit, qubit_tracker: QubitTracker
) -> tuple[int, int]:
    """Determines which gate zone to move to

    Assumes one qubit needs to move from each zone in the zones list to the gate zone
    """
    min_metric = (
        2 * mz_circ.architecture.n_zones
    )  # this is strictly larger than the largest possible
    best_gate_zone, best_gate_zone_free_space = (-1, -1)
    for gate_zone in mz_circ.macro_arch.gate_zones:
        free_space = mz_circ.architecture.get_zone_max_ions_gates(gate_zone) - len(
            qubit_tracker.zone_occupants(gate_zone)
        )
        metric = _gate_zone_metric(gate_zone, zones, free_space, mz_circ)
        if metric < min_metric:
            min_metric = metric
            best_gate_zone, best_gate_zone_free_space = (gate_zone, free_space)
            continue
    return best_gate_zone, best_gate_zone_free_space


def _make_necessary_moves_2q(  # noqa: PLR0912, PLR0915
    qubit0: int,
    qubit1: int,
    mz_circ: MultiZoneCircuit,
    qubit_tracker: QubitTracker,
) -> None:
    """
    This routine performs the necessary operations within a multi-zone circuit
     to move two qubits into the same (gate) zone

    :param qubit0: first qubit in gate
    :param qubit1: second qubit in gate
    :param mz_circ: the MultiZoneCircuit
    :param qubit_tracker: QubitTracker object for tracking/updating
     current placement of qubits
    """

    zone0 = qubit_tracker.current_zone(qubit0)
    zone1 = qubit_tracker.current_zone(qubit1)
    is_gate_zone0 = not mz_circ.architecture.zones[zone0].memory_only
    is_gate_zone1 = not mz_circ.architecture.zones[zone1].memory_only
    if zone0 == zone1 and is_gate_zone0:
        return
    free_space_zone_0 = mz_circ.architecture.get_zone_max_ions_gates(zone0) - len(
        qubit_tracker.zone_occupants(zone0)
    )
    free_space_zone_1 = mz_circ.architecture.get_zone_max_ions_gates(zone1) - len(
        qubit_tracker.zone_occupants(zone1)
    )
    if is_gate_zone1 and is_gate_zone0:
        match (free_space_zone_0, free_space_zone_1):
            case (0, 0):
                # find first qubit in zone1 that isn't qubit1
                uninvolved_qubit = next(
                    qubit
                    for qubit in qubit_tracker.zone_occupants(zone1)
                    if qubit != qubit1
                )
                # find the closest zone to zone1 with at least 2 free spots
                target_zone = _find_target_zone(zone1, [zone0], mz_circ, qubit_tracker)
                _move_qubit(
                    uninvolved_qubit, zone1, target_zone, mz_circ, qubit_tracker
                )
                # send qubit0 to zone1
                _move_qubit(qubit0, zone0, zone1, mz_circ, qubit_tracker)
            case (free0, free1) if free0 >= free1:
                _move_qubit(qubit1, zone1, zone0, mz_circ, qubit_tracker)
            case (_, _):
                _move_qubit(qubit0, zone0, zone1, mz_circ, qubit_tracker)
    elif is_gate_zone0 or is_gate_zone1:
        gate_zone_qubit, gate_zone, free_space_gate_zone, mem_zone_qubit, mem_zone = (
            (qubit0, zone0, free_space_zone_0, qubit1, zone1)
            if is_gate_zone0
            else (qubit1, zone1, free_space_zone_1, qubit0, zone0)
        )
        match free_space_gate_zone:
            case 0:
                # find first qubit in gate_zone that isn't involved
                uninvolved_qubit = next(
                    qubit
                    for qubit in qubit_tracker.zone_occupants(gate_zone)
                    if qubit != gate_zone_qubit
                )
                target_zone = _find_target_zone(
                    gate_zone, [mem_zone], mz_circ, qubit_tracker
                )
                _move_qubit(
                    uninvolved_qubit, gate_zone, target_zone, mz_circ, qubit_tracker
                )
                _move_qubit(mem_zone_qubit, mem_zone, gate_zone, mz_circ, qubit_tracker)
            case a if a > 0:
                _move_qubit(mem_zone_qubit, mem_zone, gate_zone, mz_circ, qubit_tracker)
    else:
        best_gate_zone, best_gate_zone_free_space = _find_best_gate_zone_to_move_to(
            [zone0, zone1], mz_circ, qubit_tracker
        )

        match best_gate_zone_free_space:
            case 0:
                moved0 = False
                moved1 = False
                # find first qubit in gate_zone that isn't involved
                qubits_to_remove = qubit_tracker.zone_occupants(best_gate_zone)[0:2]
                potential_swap_zones = [zone0, zone1]
                target_zone = _find_target_zone(
                    best_gate_zone, potential_swap_zones, mz_circ, qubit_tracker
                )
                _move_qubit(
                    qubits_to_remove[0],
                    best_gate_zone,
                    target_zone,
                    mz_circ,
                    qubit_tracker,
                )
                # If this is a swap we need to perform the entire swap, otherwise the
                # current target zone could be too full
                if target_zone == zone0:
                    _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
                    moved0 = True
                    potential_swap_zones.remove(zone0)
                elif target_zone == zone1:
                    _move_qubit(qubit1, zone1, best_gate_zone, mz_circ, qubit_tracker)
                    moved1 = True
                    potential_swap_zones.remove(zone1)

                target_zone = _find_target_zone(
                    best_gate_zone, potential_swap_zones, mz_circ, qubit_tracker
                )
                _move_qubit(
                    qubits_to_remove[1],
                    best_gate_zone,
                    target_zone,
                    mz_circ,
                    qubit_tracker,
                )
                if not moved0:
                    _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
                if not moved1:
                    _move_qubit(qubit1, zone1, best_gate_zone, mz_circ, qubit_tracker)

            case 1:
                moved0 = False
                moved1 = False
                qubit_to_remove = qubit_tracker.zone_occupants(best_gate_zone)[0]
                potential_swap_zones = [zone0, zone1]
                target_zone = _find_target_zone(
                    best_gate_zone, potential_swap_zones, mz_circ, qubit_tracker
                )
                _move_qubit(
                    qubit_to_remove, best_gate_zone, target_zone, mz_circ, qubit_tracker
                )
                # If this is a swap we need to perform the entire swap, otherwise the
                # current target zone could be too full
                if target_zone == zone0:
                    _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
                    moved0 = True
                    potential_swap_zones.remove(zone0)
                elif target_zone == zone1:
                    _move_qubit(qubit1, zone1, best_gate_zone, mz_circ, qubit_tracker)
                    moved1 = True
                    potential_swap_zones.remove(zone1)
                if not moved0:
                    _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
                if not moved1:
                    _move_qubit(qubit1, zone1, best_gate_zone, mz_circ, qubit_tracker)
            case a if a > 1:
                _move_qubit(qubit0, zone0, best_gate_zone, mz_circ, qubit_tracker)
                _move_qubit(qubit1, zone1, best_gate_zone, mz_circ, qubit_tracker)
