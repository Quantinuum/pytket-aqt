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
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from pytket.backends import ResultHandle
from pytket.circuit import Circuit
from qiskit_aqt_provider.api_client import models


@dataclass
class PytketAqtJobCircuitData:
    circuit: Circuit
    n_shots: int
    postprocess_json: str = field(default=json.dumps(None))
    aqt_circuit: models.Circuit | None = None
    measures: str | None = None
    handle: ResultHandle | None = None


@dataclass
class PytketAqtJob:
    circuits_data: Sequence[PytketAqtJobCircuitData]
