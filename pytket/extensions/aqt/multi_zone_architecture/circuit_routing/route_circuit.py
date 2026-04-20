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
import logging
from copy import deepcopy

from pytket.circuit import Circuit

from ..circuit.helpers import TrapConfiguration, ZonePlacement
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..trap_architecture.architecture import MultiZoneArchitectureSpec
from ..trap_architecture.dynamic_architecture import DynamicArch
from .command_filtering import filter_implementable_commands
from .gate_selection.greedy_gate_selection import GreedyGateSelector
from .routing_config import RoutingConfig

logger = logging.getLogger(__name__)


def route_circuit(
    circuit: Circuit,
    arch: MultiZoneArchitectureSpec,
    initial_placement: ZonePlacement,
    routing_config: RoutingConfig,
) -> MultiZoneCircuit:
    """
    Route a Circuit to a given MultiZoneArchitecture by adding
     physical operations where needed

    The Circuit provided cannot have more qubits than allowed by
     the architecture.

    :param circuit: A pytket Circuit to be routed
    :param arch: MultiZoneArchitecture to route into
    :param initial_placement: The initial mapping of architecture
     zones to lists of qubits
    :param routing_config: Configuration to control routing options
    """

    dynamic_arch = DynamicArch(
        arch, TrapConfiguration(circuit.n_qubits, initial_placement)
    )

    mz_circuit = MultiZoneCircuit(
        arch, initial_placement, circuit.n_qubits, circuit.n_bits
    )

    gate_selector = routing_config.gate_selector
    router = routing_config.router

    commands = circuit.get_commands().copy()

    # Add implementable gates from initial config
    implementable, commands = filter_implementable_commands(
        dynamic_arch.trap_configuration, dynamic_arch.gate_zones, commands
    )

    [mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params) for cmd in implementable]

    routing_step = 0
    while commands:
        target_config = gate_selector.next_config(dynamic_arch, commands)
        # Add operations needed move from the source to target configuration

        old = [set(zone_q) for zone_q in dynamic_arch.trap_configuration.zone_placement]
        new = [set(zone_q) for zone_q in target_config]
        if old == new or all(len(zone) == 0 for zone in new):
            if isinstance(gate_selector, GreedyGateSelector):
                raise Exception(
                    f"Gate selector did not produce new configuration. Routing step: {routing_step}"
                )
            logger.warning(
                "Chosen gate selector did not produce new configuration. Using greedy gate selection for this round"
            )
            fallback_selector = GreedyGateSelector(
                only_place_gate_qubits=gate_selector.only_places_gate_qubits()
            )
            target_config = fallback_selector.next_config(dynamic_arch, commands)
            if old == [set(zone_q) for zone_q in target_config]:
                raise Exception(
                    f"Fallback gate selector did not produce new configuration. Routing step: {routing_step}"
                )

        old_placement = deepcopy(dynamic_arch.trap_configuration.zone_placement)
        routing_result = router.route_source_to_target_config(
            dynamic_arch, target_config
        )
        log_movement(
            routing_step,
            old_placement,
            dynamic_arch.trap_configuration.zone_placement,
        )
        # Add routing operations to circuit
        mz_circuit.add_routing_ops(routing_result.routing_ops)
        # Add implementable gates from new config
        implementable, commands = filter_implementable_commands(
            dynamic_arch.trap_configuration, dynamic_arch.gate_zones, commands
        )
        for cmd in implementable:
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)

        # increment routing step
        routing_step += 1
    return mz_circuit


def log_movement(
    routing_step: int, old_placement: ZonePlacement, new_placement: ZonePlacement
) -> None:
    title_line = f"--- Configuration change {routing_step} ---"
    logger.debug(title_line)
    for zone, old_occupants in enumerate(old_placement):
        changes_str = ", ".join(
            [f"+{i}" for i in set(new_placement[zone]).difference(old_occupants)]
            + [f"-{i}" for i in set(old_occupants).difference(new_placement[zone])]
        )
        logging_string = (
            f"Z{zone}: {old_occupants} -> {new_placement[zone]} -- ({changes_str})"
        )
        logger.debug(logging_string)
