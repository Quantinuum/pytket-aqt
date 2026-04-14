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

from pytket.circuit import Circuit, Command

from pytket import OpType

from ..circuit.helpers import TrapConfiguration, ZonePlacement, get_qubit_to_zone
from ..circuit.multizone_circuit import MultiZoneCircuit
from ..trap_architecture.architecture import MultiZoneArchitectureSpec
from ..trap_architecture.dynamic_architecture import DynamicArch
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
    implementable, commands = filter_implementable_commands(dynamic_arch, commands)

    [mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params) for cmd in implementable]

    routing_step = 0
    while commands:
        target_config = gate_selector.next_config(dynamic_arch, commands)
        # Add operations needed move from the source to target configuration

        old = [set(zone_q) for zone_q in dynamic_arch.trap_configuration.zone_placement]
        new = [set(zone_q) for zone_q in target_config]
        if old == new:
            if isinstance(gate_selector, GreedyGateSelector):
                raise Exception(
                    f"Gate selector did not produce new configuration. Routing step: {routing_step}"
                )
            logger.warning(
                "Chosen gate selector did not produce new configuration. Using greedy gate selection for this round"
            )
            target_config = GreedyGateSelector().next_config(dynamic_arch, commands)

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
        implementable, commands = filter_implementable_commands(dynamic_arch, commands)
        for cmd in implementable:
            mz_circuit.add_gate(cmd.op.type, cmd.args, cmd.op.params)

        # increment routing step
        routing_step += 1
    return mz_circuit


def filter_implementable_commands(
    dynamic_arch: DynamicArch,
    commands: list[Command],
) -> tuple[list[Command], list[Command]]:
    """Split gates into currently implementable and those that require a new config"""
    leftovers: list[Command] = []
    implementable: list[Command] = []
    # stragglers are qubits with pending 2 qubit gates that cannot
    # be performed in the current config
    # they have to wait for the next iteration
    current_config = dynamic_arch.trap_configuration
    gate_zones = dynamic_arch.gate_zones
    stragglers: set[int] = set()
    qubit_to_zone_old = get_qubit_to_zone(
        current_config.n_qubits, current_config.zone_placement
    )
    last_cmd_index = 0
    for i, cmd in enumerate(commands):
        if cmd.op.type in [OpType.Barrier]:
            implementable.append(cmd)
        last_cmd_index = i
        n_args = len(cmd.args)
        qubit0 = cmd.args[0].index[0]
        zone0 = qubit_to_zone_old[qubit0]
        if n_args == 1:
            if qubit0 in stragglers or zone0 not in gate_zones:
                leftovers.append(cmd)
            else:
                implementable.append(cmd)
        elif n_args == 2:  # noqa: PLR2004
            qubit1 = cmd.args[1].index[0]
            if qubit0 in stragglers:
                stragglers.add(qubit1)
                leftovers.append(cmd)
                continue
            if qubit1 in stragglers:
                stragglers.add(qubit0)
                leftovers.append(cmd)
                continue
            if zone0 == qubit_to_zone_old[qubit1] and zone0 in gate_zones:
                implementable.append(cmd)
            else:
                leftovers.append(cmd)
                stragglers.update([qubit0, qubit1])
        if len(stragglers) >= current_config.n_qubits:
            # at this point no more gates can be performed in this config
            break
    return implementable, leftovers + commands[last_cmd_index + 1 :]


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
