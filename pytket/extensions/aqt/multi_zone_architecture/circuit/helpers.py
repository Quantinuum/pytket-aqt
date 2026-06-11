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
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ZonePlacement = list[list[int]]


def get_qubit_to_zone(
    n_qubits: int, placement: ZonePlacement
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    qubit_to_zone = -np.ones(n_qubits, dtype=np.int64)
    for zone, qubits in enumerate(placement):
        qubit_to_zone[qubits] = zone
    return qubit_to_zone


def get_qubit_to_zone_pos(
    n_qubits: int, placement: ZonePlacement
) -> np.ndarray[tuple[int, int], np.dtype[np.uint64]]:
    qubit_to_zone = np.zeros((n_qubits, 2), dtype=np.uint64)
    for zone, qubits in enumerate(placement):
        if qubits:
            qubit_to_zone[qubits] = [(zone, pos) for pos in range(len(qubits))]
    return qubit_to_zone


class ZoneRoutingError(Exception):
    pass


@dataclass
class TrapConfiguration:
    n_qubits: int
    zone_placement: ZonePlacement
