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
from pytket.extensions.aqt.multi_zone_architecture.depth_list.depth_list import (
    DepthInfo,
    depth_info_from_command_list,
    depth_info_from_command_list_until_block_size_exceeds,
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
    from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.cost_model import (
        ShuttlePSwapCostModel,
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


def test_bounded_depth_info_stops_when_blocks_exceed_gate_capacity() -> None:
    circ = Circuit(4)
    circ.CX(0, 1)
    circ.CX(2, 3)
    circ.CX(1, 2)
    circ.CX(0, 3)

    full_depth_info = depth_info_from_command_list(4, circ.get_commands())
    bounded_depth_info = depth_info_from_command_list_until_block_size_exceeds(
        4, circ.get_commands(), max_block_size=2
    )

    assert len(full_depth_info.depth_blocks) == 2
    assert bounded_depth_info.depth_list == full_depth_info.depth_list[:1]
    assert bounded_depth_info.depth_blocks == full_depth_info.depth_blocks[:1]


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


@hypergraph_skipif
def test_optimized_shuttling_hyperedges_match_generic_cost_model() -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [2, 3], []])
    max_shuttle_weight = 40000

    fast_nets: list[list[int]] = []
    fast_weights: list[int] = []
    HypergraphPartitionGateSelector().add_shuttling_penalty_hyperedges(
        dyn_arch, fast_nets, fast_weights, max_shuttle_weight
    )

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

    generic_nets: list[list[int]] = []
    generic_weights: list[int] = []
    HypergraphPartitionGateSelector(
        cost_model=DelegatingPSwapCostModel()
    ).add_shuttling_penalty_hyperedges(
        dyn_arch, generic_nets, generic_weights, max_shuttle_weight
    )

    assert fast_nets == generic_nets
    assert fast_weights == generic_weights


@hypergraph_skipif
def test_hypergraph_gate_hyperedges_skip_nonpositive_weights() -> None:
    dyn_arch = _make_dynamic_arch([[0, 1], [], [], []])
    depth_info = DepthInfo(
        depth_list=[],
        depth_blocks=[[{0, 1}] for _ in range(105)],
    )

    hypergraph_data = (
        HypergraphPartitionGateSelector().get_circuit_shuttle_hypergraph_data(
            dyn_arch, depth_info
        )
    )

    assert all(weight > 0 for weight in hypergraph_data.net_weights)
