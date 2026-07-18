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

from pytket import Circuit
from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.command_filtering import (
    filter_implementable_commands,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.greedy_gate_selection import (
    GreedyGateSelector,
    gate_zone_metric,
)
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.cost_model import (
    ShuttlePSwapCostModel,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)

if MT_KAHYPAR_INSTALLED:
    from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.graph_partition_gate_selection import (
        GraphPartitionGateSelector,
        get_placement_of_implementable_qubits,
    )


graph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED, reason="mtkahypar required for graph partition tests"
)


def _make_dynamic_arch(zone_placement: list[list[int]]) -> DynamicArch:
    n_qubits = sum(len(zone_qubits) for zone_qubits in zone_placement)
    return DynamicArch(
        four_zones_in_a_line, TrapConfiguration(n_qubits, zone_placement)
    )


def _flatten(placement: list[list[int]]) -> set[int]:
    return {qubit for zone_qubits in placement for qubit in zone_qubits}


def test_greedy_gate_selector_only_places_gate_qubits_for_single_qubit_round() -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [], []])
    circ = Circuit(2)
    circ.X(0)

    placement = GreedyGateSelector(only_place_gate_qubits=True).next_config(
        dyn_arch, circ.get_commands()
    )

    assert _flatten(placement) == {0}


def test_optimized_greedy_gate_zone_metric_matches_generic_cost_model() -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [2, 3], []])
    qubit_zones = [(0, 0), (3, 2)]
    gate_zone = 1

    class DelegatingPSwapCostModel:
        def __init__(self) -> None:
            self._delegate = ShuttlePSwapCostModel()

        def move_cost(self, *args, **kwargs):
            return self._delegate.move_cost(*args, **kwargs)

        def move_cost_src_port_0(self, *args, **kwargs):
            return self._delegate.move_cost_src_port_0(*args, **kwargs)

        def move_cost_src_port_1(self, *args, **kwargs):
            return self._delegate.move_cost_src_port_1(*args, **kwargs)

        def closest_zones(self, *args, **kwargs):
            return self._delegate.closest_zones(*args, **kwargs)

    assert gate_zone_metric(
        dyn_arch, ShuttlePSwapCostModel(), gate_zone, qubit_zones
    ) == gate_zone_metric(dyn_arch, DelegatingPSwapCostModel(), gate_zone, qubit_zones)


@graph_skipif
def test_get_placement_of_implementable_qubits_removes_non_implementable_qubits() -> (
    None
):
    circ = Circuit(4)
    circ.CX(0, 1)
    circ.X(2)
    circ.CX(2, 3)
    circ.X(3)
    implementable_commands, remaining_commands = filter_implementable_commands(
        TrapConfiguration(4, [[], [0, 1, 2], [3], []]),
        gate_zones=[1, 2],
        commands=circ.get_commands(),
    )

    placement = get_placement_of_implementable_qubits(
        target_placement=[[], [0, 1, 2], [3], []],
        implementable_commands=implementable_commands,
    )

    assert [cmd.op.type.name for cmd in implementable_commands] == ["CX", "X"]
    assert [cmd.op.type.name for cmd in remaining_commands] == ["CX", "X"]
    assert placement == [[], [0, 1, 2], [], []]


@graph_skipif
def test_graph_partition_gate_selector_only_places_gate_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [2, 3], []])
    circ = Circuit(4)
    circ.CX(0, 1)
    circ.X(2)
    circ.CX(2, 3)
    circ.X(3)

    def mock_partition_graph(
        _self: object, _graph_data: object, _num_parts: int
    ) -> list[int]:
        """
        This is an n_qubit + n_zone sized list
        The first n_qubit = 4 values are the placement of the qubits in zones
        The lase n_zone = 4 values represent a permutation of the zones

        This places qubits in the following way:
           0, 1, 2 -> zone 1
           3 -> zone 2

        The permutation of the zones is the identity,
         so it doesn't change the placement
        """
        return [1, 1, 1, 2, 0, 1, 2, 3]

    monkeypatch.setattr(
        "pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.graph_partition_gate_selection.MtKahyparPartitioner.partition_graph",
        mock_partition_graph,
    )

    placement = GraphPartitionGateSelector(only_place_gate_qubits=True).next_config(
        dyn_arch, circ.get_commands()
    )

    assert placement == [[], [0, 1, 2], [], []]
