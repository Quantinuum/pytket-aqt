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
from ...trap_architecture.named_architectures import siqci_arch
from .router import Router, RoutingResult


def _validate_siqci_architecture(dyn_arch: DynamicArch) -> None:
    if dyn_arch.architecture_spec != siqci_arch:
        raise ValueError(
            "SiqciArchRouter can only be used with the siqci_arch architecture."
        )


class SiqciArchRouter(Router):
    """Temporary router specialized to the siqci_arch architecture."""

    def route_source_to_target_config(
        self, dyn_arch: DynamicArch, target_placement: ZonePlacement
    ) -> RoutingResult:
        _validate_siqci_architecture(dyn_arch)
        return RoutingResult(cost_estimate=0, routing_ops=[])
