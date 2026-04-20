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
from typing import Protocol

from pytket.circuit import Command

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.dynamic_architecture import DynamicArch


class GateSelector(Protocol):
    """A class protocol for calculating the optimal placement of ions in zones to implement upcoming gates"""

    def next_config(
        self, dyn_arch: DynamicArch, remaining_commands: list[Command]
    ) -> ZonePlacement: ...

    """Returns the optimal placement of qubits in zones (no ordering within zones)"""

    def only_places_gate_qubits(self) -> bool: ...
