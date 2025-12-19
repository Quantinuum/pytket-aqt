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

import pytest
from pytket.circuit import Circuit, OpType

from pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit import (
    AcrossZoneOperationError,
    MoveError,
    MultiZoneCircuit,
    QubitPlacementError,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)


@pytest.fixture()
def initial_placement() -> dict[int, list[int]]:
    return {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}


@pytest.fixture()
def fix_circuit(initial_placement: dict[int, list[int]]) -> MultiZoneCircuit:
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.move_qubit(3, 1)
    circuit.move_qubit(0, 1)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    return circuit


@pytest.fixture()
def circuit_precompile() -> Circuit:
    circuit = Circuit(8)
    circuit.CX(0, 1).CX(2, 3).CX(4, 5).CX(6, 7)
    circuit.CX(1, 2).CX(3, 4).CX(5, 6).CX(7, 0)
    circuit.measure_all()
    return circuit


def test_circuit_has_correct_init_gates_at_beginning(
    fix_circuit: MultiZoneCircuit, initial_placement: dict[int, list[int]]
) -> None:
    circuit_placement = {}
    for gate in fix_circuit.pytket_circuit:
        op = gate.op
        if "INIT" not in op.__str__():
            break
        gate_zone = int(op.params[0])
        gate_qubits = [q.index[0] for q in gate.args]
        circuit_placement[gate_zone] = gate_qubits
    for zone in range(fix_circuit.architecture.n_zones):
        if zone not in initial_placement:
            initial_placement[zone] = []
    assert circuit_placement == initial_placement


def test_circuit_contains_correct_number_of_moves_shuttles_swaps(
    fix_circuit: MultiZoneCircuit,
) -> None:
    move_barriers, moves, shuttles, swaps = 0, 0, 0, 0
    for gate in fix_circuit.pytket_circuit:
        op = gate.op
        if "MOVE_BARRIER" in op.__str__():
            move_barriers += 1
            continue
        if "MOVE" in op.__str__():
            moves += 1
            continue
        if "SHUTTLE" in op.__str__():
            shuttles += 1
            continue
        if "PSWAP" in op.__str__():
            swaps += 1
            continue
    assert (move_barriers, moves, shuttles, swaps) == (2, 2, 0, 0)


def test_redundant_move_raises_move_error(fix_circuit: MultiZoneCircuit) -> None:
    with pytest.raises(MoveError):
        fix_circuit.move_qubit(2, 0)


def test_move_on_missing_qubit_raises_placement_error(
    fix_circuit: MultiZoneCircuit,
) -> None:
    with pytest.raises(QubitPlacementError):
        fix_circuit.move_qubit(9, 1)


def test_add_barrier_throws_value_error(fix_circuit: MultiZoneCircuit) -> None:
    with pytest.raises(ValueError):
        fix_circuit.add_gate(OpType.Barrier, [0])


def test_validation_of_circuit_with_operation_across_zones_throws(
    initial_placement: dict[int, list[int]],
) -> None:
    circuit = MultiZoneCircuit(four_zones_in_a_line, initial_placement, 8)
    circuit.CX(0, 4)
    circuit.measure_all()
    with pytest.raises(AcrossZoneOperationError):
        circuit.validate()


def test_validation_of_valid_circuit_does_not_throw(
    fix_circuit: MultiZoneCircuit,
) -> None:
    fix_circuit.validate()
