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

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.dynamic_architecture import DynamicArch
from .router import Router, RoutingResult


def target_zone_interval_qubits(
    dyn_arch: DynamicArch,
    target_placement: ZonePlacement,
) -> list[list[int]]:
    current_qubit_order = [
        qubit
        for zone in linearly_ordered_zones(dyn_arch)
        for qubit in dyn_arch.trap_configuration.zone_placement[zone]
    ]
    qubit_to_index = {qubit: i for i, qubit in enumerate(current_qubit_order)}
    target_interval_qubits: list[list[int]] = []

    for zone_qubits in target_placement:
        if not zone_qubits:
            target_interval_qubits.append([])
            continue

        target_indices = sorted(qubit_to_index[qubit] for qubit in zone_qubits)
        interval_qubits = current_qubit_order[
            target_indices[0] : target_indices[-1] + 1
        ]
        target_interval_qubits.append(interval_qubits)

    return target_interval_qubits


def linearly_ordered_zones(dyn_arch: DynamicArch) -> list[int]:
    ordered_zones = [line_start_zone(dyn_arch)]
    previous_zone: int | None = None
    current_zone = ordered_zones[0]

    while True:
        next_zones = [
            zone
            for zone in dyn_arch.connected_zones(current_zone)
            if zone != previous_zone
        ]
        if not next_zones:
            return ordered_zones
        if len(next_zones) > 1:
            raise ValueError("Linear architecture contains a branching zone order.")
        previous_zone, current_zone = current_zone, next_zones[0]
        ordered_zones.append(current_zone)


def line_start_zone(dyn_arch: DynamicArch) -> int:
    for zone in range(dyn_arch.n_zones):
        connected_zones_0, connected_zones_1 = dyn_arch.connected_zones_per_port(zone)
        if not connected_zones_0 and connected_zones_1:
            return zone
    raise ValueError("Could not determine the start zone of linear architecture.")


class LineArchRouter(Router):
    """Router specialized for linear multi-zone architectures."""

    def route_source_to_target_config(
        self, dyn_arch: DynamicArch, target_placement: ZonePlacement
    ) -> RoutingResult:
        if not dyn_arch.is_linear_architecture:
            raise ValueError(
                "LineArchRouter can only be used with linear architectures."
            )
        target_zone_interval_qubits(dyn_arch, target_placement)
        return RoutingResult(cost_estimate=0, routing_ops=[])
