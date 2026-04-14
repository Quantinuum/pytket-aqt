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

from pytket.circuit import Command, OpType

from ...circuit.helpers import ZonePlacement
from ...depth_list.depth_list import depth_info_from_command_list
from ...trap_architecture.cost_model import (
    RoutingCostModel,
    ShuttlePSwapCostModel,
    unwrap_move_cost_result,
)
from ...trap_architecture.dynamic_architecture import DynamicArch
from .gate_selector_protocol import GateSelector
from .qubit_tracker import QubitTracker

_DEFAULT_COST_MODEL = ShuttlePSwapCostModel()


class GreedyGateSelector(GateSelector):
    """Uses a simple greedy algorithm for gate selection

    The routed circuit can be directly run on the given Architecture

    :param cost_model: Cost model for estimating movement costs
    """

    def __init__(
        self,
        cost_model: RoutingCostModel = _DEFAULT_COST_MODEL,
    ):
        self._cost_model = cost_model

    def next_config(
        self,
        dyn_arch: DynamicArch,
        remaining_commands: list[Command],
    ) -> ZonePlacement:
        """Generates a new TrapConfiguration to implement the next gates

        The returned TrapConfiguration
        represents the "optimal" next state to implement the remaining gates in
        the depth list. The ordering of the qubits within the zones is arbitrary. The correct
        ordering will be determined at the qubit routing stage.

        Cycle through commands and assign qubits to new zones in order to implement them on
        a "first come, first served" basis

        Rules:

        - One qubit gates do not necessitate a move (until all 2 qubit gates are accounted for).
          If they cannot be performed in the current config, they can wait until that qubit
           requires a 2 qubit gate (or at the end of the circuit if no 2 qubit gates remain)
        - Two qubit gates will cause one or more moves to place the participating qubits into
           the same memory zone. Once the placement of a qubit is changed in order to implement
           a two qubit gate, it is "locked" in place. This means at some point, no more legal
           moves will be possible. The algorithm will terminate at that point and the current
           configuration will be emitted as the next config.

           Necessary moves may require the displacement of other qubits, due to max
           qubit requirements (using the "gate operation" maximum). This displacement is
           taken into account "lazily". Any qubits that have been displaced and were not involved
           in gates before termination will be placed in the "nearest" zones with available
           capacity

        :param dyn_arch: The dynamic architecture containing the current configuration of ions in ion trap zones
        :param remaining_commands: The list of gate commands used to determine the next ion placement.
        """
        n_qubits = dyn_arch.n_qubits
        qubit_tracker = QubitTracker(
            n_qubits, dyn_arch.trap_configuration.zone_placement
        )

        two_qubit_gate_depth_info = depth_info_from_command_list(
            n_qubits, remaining_commands
        )

        if two_qubit_gate_depth_info.depth_list:
            self.handle_depth_list(
                dyn_arch, two_qubit_gate_depth_info.depth_list, qubit_tracker
            )
        else:
            handle_only_single_qubits_remaining(
                dyn_arch,
                self._cost_model,
                remaining_commands,
                qubit_tracker,
            )
        # Now move any unused qubits to vacant spots in new config
        handle_unused_qubits(dyn_arch, self._cost_model, qubit_tracker)
        return qubit_tracker.new_placement()

    def handle_depth_list(
        self,
        dyn_arch: DynamicArch,
        depth_list: list[list[tuple[int, int]]],
        qubit_tracker: QubitTracker,
    ) -> None:
        n_qubits = dyn_arch.n_qubits
        # locked qubits have already been assigned a zone in the new config and should therefore not move again
        locked_qubits = []
        # must wait qubits have had a 2 qubit gate that could not be implemented in the new config
        # thus any following 2 qubit gate involving them will not be implementable in the new config
        must_wait_qubits: list[int] = []
        for depth in depth_list:
            for pair in depth:
                # If all qubits must wait or no more spots are available in gate zones, we can stop
                max_gate_zone_free_space = max(
                    [
                        dyn_arch.zone_max_gate_cap[gz]
                        - qubit_tracker.n_zone_new_occupants(gz)
                        for gz in dyn_arch.gate_zones
                    ]
                )
                if len(must_wait_qubits) == n_qubits or max_gate_zone_free_space == 0:
                    break

                qubit0 = pair[0]
                qubit1 = pair[1]
                # handle case that one or both qubits must wait
                if handle_must_wait(qubit0, qubit1, must_wait_qubits):
                    continue

                # starting here, neither qubit must wait

                is_locked_0 = qubit0 in locked_qubits
                is_locked_1 = qubit1 in locked_qubits

                # handle both locked
                if is_locked_0 and is_locked_1:
                    zone0 = qubit_tracker.current_zone(qubit0)
                    zone1 = qubit_tracker.current_zone(qubit1)
                    if zone0 != zone1:
                        # if not in same zone, gate cannot be implemented
                        # so new gates involving these qubits must wait
                        must_wait_qubits.extend([qubit0, qubit1])
                    continue

                # handle only one locked
                if is_locked_0 or is_locked_1:
                    locked_q, other_q = (
                        (qubit0, qubit1) if is_locked_0 else (qubit1, qubit0)
                    )
                    implementable = handle_one_locked_qubit(
                        dyn_arch, locked_q, other_q, qubit_tracker
                    )
                    if implementable:
                        locked_qubits.append(other_q)
                    else:
                        must_wait_qubits.append(other_q)
                    continue

                # starting here, neither qubit is locked

                zone0 = qubit_tracker.current_zone(qubit0)
                zone1 = qubit_tracker.current_zone(qubit1)
                is_gate_zone0 = dyn_arch.is_gate_zone(zone0)
                is_gate_zone1 = dyn_arch.is_gate_zone(zone1)

                free_space_zone_0 = dyn_arch.zone_max_gate_cap[
                    zone0
                ] - qubit_tracker.n_zone_new_occupants(zone0)
                free_space_zone_1 = dyn_arch.zone_max_gate_cap[
                    zone1
                ] - qubit_tracker.n_zone_new_occupants(zone1)

                # Need to find a gate zone with 2 spots available to put the qubits, since
                # neither has been placed in the new config yet
                # Starting from and including their current zones, find the closest gate zone
                # that has two spots available

                num_spots_needed = 2

                if is_gate_zone0 and free_space_zone_0 >= num_spots_needed:
                    qubit_tracker.lock_qubit(qubit0, zone0, zone0)
                    qubit_tracker.lock_qubit(qubit1, zone1, zone0)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                if is_gate_zone1 and free_space_zone_1 >= num_spots_needed:
                    qubit_tracker.lock_qubit(qubit0, zone0, zone1)
                    qubit_tracker.lock_qubit(qubit1, zone1, zone1)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                # try to find the closest gate zone with two available spots
                closest_zone = find_best_gate_zone_to_move_to(
                    dyn_arch,
                    self._cost_model,
                    [(qubit0, zone0), (qubit1, zone1)],
                    qubit_tracker,
                )
                if closest_zone is not None:
                    qubit_tracker.lock_qubit(qubit0, zone0, closest_zone)
                    qubit_tracker.lock_qubit(qubit1, zone1, closest_zone)
                    locked_qubits.extend([qubit0, qubit1])
                    continue

                must_wait_qubits.extend([qubit0, qubit1])
                # If made it here:
                # No longer any gate zones left with two spots available
                # A gate with one locked qubit in a zone with one space
                # left could still be implemented, so continue loop


def handle_only_single_qubits_remaining(
    dyn_arch: DynamicArch,
    cost_model: RoutingCostModel,
    remaining_commands: list[Command],
    qubit_tracker: QubitTracker,
) -> None:
    # locked qubits have already been assigned a zone in the new config and should therefore not move again
    locked_qubits = []
    for cmd in remaining_commands:
        if cmd.op.type in [OpType.Barrier]:
            continue
        qubit0 = cmd.args[0].index[0]
        is_locked_0 = qubit0 in locked_qubits
        if is_locked_0:
            continue
        zone0 = qubit_tracker.current_zone(qubit0)
        is_gate_zone0 = dyn_arch.is_gate_zone(zone0)
        free_space_zone_0 = dyn_arch.zone_max_gate_cap[
            zone0
        ] - qubit_tracker.n_zone_new_occupants(zone0)
        if is_gate_zone0 and free_space_zone_0 >= 1:
            qubit_tracker.lock_qubit(qubit0, zone0, zone0)
            locked_qubits.append(qubit0)
            continue

        # try to find the closest gate zone with two available spots
        closest_zone = find_best_gate_zone_to_move_to(
            dyn_arch,
            cost_model,
            [(qubit0, zone0)],
            qubit_tracker,
        )
        if closest_zone is not None:
            qubit_tracker.lock_qubit(qubit0, zone0, closest_zone)
            locked_qubits.extend([qubit0])
            continue
        # if closest_zone is None, there are no more gate zone spots available
        break


def handle_must_wait(qubit0: int, qubit1: int, must_wait_qubits: list[int]) -> bool:
    must_wait_0 = qubit0 in must_wait_qubits
    must_wait_1 = qubit1 in must_wait_qubits
    if must_wait_0 or must_wait_1:
        if must_wait_0 and not must_wait_1:
            must_wait_qubits.append(qubit1)
        if must_wait_1 and not must_wait_0:
            must_wait_qubits.append(qubit0)
        return True
    return False


def handle_one_locked_qubit(
    dyn_arch: DynamicArch,
    locked_qubit: int,
    other_qubit: int,
    qubit_tracker: QubitTracker,
) -> bool:
    """Handle the case were one qubit is locked and one isn't

    Assumes other qubit is not in "must wait" qubits

    A qubit being locked implies it is in a gate zone. Move other qubit to its zone if
    space is available (return True). If not, must gate can't be implemented, so add other_qubit to must
    wait (return False).
    """
    lq_zone = qubit_tracker.current_zone(locked_qubit)
    free_space_lq_zone = dyn_arch.zone_max_gate_cap[
        lq_zone
    ] - qubit_tracker.n_zone_new_occupants(lq_zone)
    if free_space_lq_zone >= 1:
        oq_zone = qubit_tracker.current_zone(other_qubit)
        qubit_tracker.lock_qubit(other_qubit, oq_zone, lq_zone)
        return True
    return False


def find_best_gate_zone_to_move_to(
    dyn_arch: DynamicArch,
    cost_model: RoutingCostModel,
    qubit_zones: list[tuple[int, int]],
    qubit_tracker: QubitTracker,
) -> int | None:
    """Determines which gate zone to move to

    Assumes the specified qubits need to move from their respective zones to the gate zone
    The gate zone must have at least len(zones) places available

    Return gate zone id or None if no gate zone available with two spots
    """
    min_metric = (
        dyn_arch.n_zones * 100
    )  # this is strictly larger than the largest possible
    best_gate_zone = -1
    for gate_zone in dyn_arch.gate_zones:
        free_space = dyn_arch.zone_max_gate_cap[
            gate_zone
        ] - qubit_tracker.n_zone_new_occupants(gate_zone)
        if free_space >= len(qubit_zones):
            metric = gate_zone_metric(dyn_arch, cost_model, gate_zone, qubit_zones)
            if metric < min_metric:
                min_metric = metric
                best_gate_zone = gate_zone
                continue
    if best_gate_zone == -1:
        return None
    return best_gate_zone


def gate_zone_metric(
    dyn_arch: DynamicArch,
    cost_model: RoutingCostModel,
    gate_zone: int,
    qubit_zones: list[tuple[int, int]],
) -> int:
    """Calculate metric estimating cost of moving one qubit from each of a
     list of zones to a specific gate zone

    Prefers gate zones with low total distance
    """
    return sum(
        unwrap_move_cost_result(
            cost_model.move_cost(dyn_arch, [qubit], zone, gate_zone)
        ).path_cost
        for qubit, zone in qubit_zones
    )


def move_qubits_to_closest_available_spots(
    dyn_arch: DynamicArch,
    cost_model: RoutingCostModel,
    zone_qubits: list[int],
    starting_zone: int,
    qubit_tracker: QubitTracker,
) -> None:
    """Determines which a zone to move to

    Finds closest zone with a free spot available

    Assumes one qubit needs to move from each zone in the zones list to the gate zone
    The gate zone must have at least two places available

    Return gate zone id or None if no gate zone available with two spots
    """
    qubits = deepcopy(zone_qubits)
    # just use the position of the first qubit as approximation instead of getting
    # "closest zones" for each qubit separately
    for zone in cost_model.closest_zones(dyn_arch, qubits[0], starting_zone):
        zone_occupancy = qubit_tracker.n_zone_new_occupants(zone)
        max_zone_occupancy = dyn_arch.zone_max_gate_cap[zone]
        zone_free_space = max_zone_occupancy - zone_occupancy
        while zone_free_space > 0 and qubits:
            qubit_tracker.lock_qubit(qubits.pop(), starting_zone, zone)
            zone_free_space -= 1
        if not qubits:
            break


def handle_unused_qubits(
    dyn_arch: DynamicArch,
    cost_model: RoutingCostModel,
    qubit_tracker: QubitTracker,
) -> None:
    # Take care of qubits that don't actually need to move (free space available in their old zone)
    for zone in range(dyn_arch.n_zones):
        zone_leftovers = qubit_tracker.old_zone_occupants(zone).copy()
        free_space_zone = dyn_arch.zone_max_gate_cap[
            zone
        ] - qubit_tracker.n_zone_new_occupants(zone)
        for qubit in zone_leftovers:
            if free_space_zone == 0:
                break
            qubit_tracker.lock_qubit(qubit, zone, zone)
            free_space_zone -= 1
    # Take care of qubits that need to move to a new zone (no more free space in their old zone)
    for zone in range(dyn_arch.n_zones):
        zone_leftovers = qubit_tracker.old_zone_occupants(zone)
        if zone_leftovers:
            move_qubits_to_closest_available_spots(
                dyn_arch, cost_model, zone_leftovers, zone, qubit_tracker
            )
