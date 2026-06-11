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
from dataclasses import dataclass
from typing import Protocol

from ...circuit.helpers import ZonePlacement
from ...trap_architecture.dynamic_architecture import DynamicArch
from ..routing_ops import RoutingOp


@dataclass
class RoutingResult:
    cost_estimate: float
    routing_ops: list[RoutingOp]


class Router(Protocol):
    def route_source_to_target_config(
        self,
        dyn_arch: DynamicArch,
        target_placement: ZonePlacement,
    ) -> RoutingResult: ...
