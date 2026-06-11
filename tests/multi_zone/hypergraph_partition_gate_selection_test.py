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
from pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar_check import (
    MT_KAHYPAR_INSTALLED,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.dynamic_architecture import (
    DynamicArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
)

if MT_KAHYPAR_INSTALLED:
    from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.hypergraph_partition_gate_selection import (
        HypergraphPartitionGateSelector,
    )


hypergraph_skipif = pytest.mark.skipif(
    not MT_KAHYPAR_INSTALLED,
    reason="mtkahypar required for hypergraph partition tests",
)


def _make_dynamic_arch(zone_placement: list[list[int]]) -> DynamicArch:
    n_qubits = sum(len(zone_qubits) for zone_qubits in zone_placement)
    return DynamicArch(
        four_zones_in_a_line, TrapConfiguration(n_qubits, zone_placement)
    )


def _flatten(placement: list[list[int]]) -> set[int]:
    return {qubit for zone_qubits in placement for qubit in zone_qubits}


@hypergraph_skipif
def test_hypergraph_partition_gate_selector_only_places_gate_qubits_for_single_qubit_round() -> (
    None
):
    dyn_arch = _make_dynamic_arch([[0, 1], [], [], []])
    circ = Circuit(2)
    circ.X(0)

    placement = HypergraphPartitionGateSelector(
        only_place_gate_qubits=True
    ).next_config(dyn_arch, circ.get_commands())

    assert _flatten(placement) == {0}


@hypergraph_skipif
def test_hypergraph_partition_gate_selector_only_places_gate_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [2, 3], []])
    circ = Circuit(4)
    circ.CX(0, 1)
    circ.X(2)
    circ.CX(2, 3)
    circ.X(3)

    def mock_partition_hypergraph(
        _self: object, _graph_data: object, _num_parts: int
    ) -> list[int]:
        return [1, 1, 1, 2, 0, 1, 2, 3]

    monkeypatch.setattr(
        "pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection.hypergraph_partition_gate_selection.MtKahyparPartitioner.partition_hypergraph",
        mock_partition_hypergraph,
    )

    placement = HypergraphPartitionGateSelector(
        only_place_gate_qubits=True
    ).next_config(dyn_arch, circ.get_commands())

    assert placement == [[], [0, 1, 2], [], []]
