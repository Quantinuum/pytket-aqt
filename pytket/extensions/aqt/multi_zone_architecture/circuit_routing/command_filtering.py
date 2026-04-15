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

from pytket.circuit import Command, OpType

from ..circuit.helpers import TrapConfiguration, get_qubit_to_zone


def filter_implementable_commands(
    trap_configuration: TrapConfiguration,
    gate_zones: list[int],
    commands: list[Command],
) -> tuple[list[Command], list[Command]]:
    """Split gates into those implementable with the current trap configuration and those that require a new config."""
    leftovers: list[Command] = []
    implementable: list[Command] = []
    stragglers: set[int] = set()
    qubit_to_zone = get_qubit_to_zone(
        trap_configuration.n_qubits, trap_configuration.zone_placement
    )
    last_cmd_index = 0
    for i, cmd in enumerate(commands):
        if cmd.op.type == OpType.Barrier:
            implementable.append(cmd)
            continue

        last_cmd_index = i
        n_args = len(cmd.args)
        qubit0 = cmd.args[0].index[0]
        zone0 = qubit_to_zone[qubit0]
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
            if zone0 == qubit_to_zone[qubit1] and zone0 in gate_zones:
                implementable.append(cmd)
            else:
                leftovers.append(cmd)
                stragglers.update([qubit0, qubit1])
        if len(stragglers) >= trap_configuration.n_qubits:
            break
    return implementable, leftovers + commands[last_cmd_index + 1 :]
