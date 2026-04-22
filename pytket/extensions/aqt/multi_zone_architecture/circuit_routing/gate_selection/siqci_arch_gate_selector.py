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

from pytket.circuit import Command

from ...circuit.helpers import ZonePlacement
from ...depth_list.depth_list import DepthInfo, depth_info_from_command_list
from ...trap_architecture.dynamic_architecture import DynamicArch
from ...trap_architecture.named_architectures import siqci_arch
from .gate_selector_protocol import GateSelector


def _validate_siqci_architecture(dyn_arch: DynamicArch) -> None:
    if dyn_arch.architecture_spec != siqci_arch:
        raise ValueError(
            "SiqciArchGateSelector can only be used with the siqci_arch architecture."
        )
    gate_zone = dyn_arch.gate_zones[0]
    gate_zone_occupancy = len(dyn_arch.trap_configuration.zone_placement[gate_zone])
    gate_zone_capacity = int(dyn_arch.zone_max_gate_cap[gate_zone])
    if gate_zone_occupancy != gate_zone_capacity:
        raise ValueError(
            "SiqciArchGateSelector requires the gate zone to be fully occupied when selecting the next configuration."
        )


def handle_2qb_gates_remaining(dyn_arch: DynamicArch, depth_info: DepthInfo):
    return [[] for _ in range(dyn_arch.n_zones)]


def handle_only_single_qubit_gates_remaining(
    dyn_arch: DynamicArch, remaining_commands: list[Command]
):
    return [[] for _ in range(dyn_arch.n_zones)]


class SiqciArchGateSelector(GateSelector):
    """Temporary gate selector specialized to the siqci_arch architecture."""

    def only_places_gate_qubits(self) -> bool:
        return True

    def next_config(
        self, dyn_arch: DynamicArch, remaining_commands: list[Command]
    ) -> ZonePlacement:
        _validate_siqci_architecture(dyn_arch)
        current_configuration = dyn_arch.trap_configuration
        n_qubits = current_configuration.n_qubits
        depth_info = depth_info_from_command_list(n_qubits, remaining_commands)
        if depth_info.depth_list:
            placement = handle_2qb_gates_remaining(dyn_arch, depth_info)
        else:
            placement = handle_only_single_qubit_gates_remaining(
                dyn_arch, remaining_commands
            )
        return placement
