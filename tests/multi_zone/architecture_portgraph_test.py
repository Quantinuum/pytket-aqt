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

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    MultiZoneArchitectureSpec,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture_portgraph import (
    MultiZonePortGraph,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    gate_zone_type_examples,
    grid12,
    racetrack,
)


def _empty_configuration(spec: MultiZoneArchitectureSpec) -> TrapConfiguration:
    return TrapConfiguration(
        n_qubits=0, zone_placement=[[] for _ in range(spec.n_zones)]
    )


def test_portgraph_detects_linear_architecture() -> None:
    port_graph = MultiZonePortGraph(
        four_zones_in_a_line, _empty_configuration(four_zones_in_a_line)
    )

    assert port_graph.is_linear_architecture()


def test_portgraph_rejects_cycle_architecture() -> None:
    port_graph = MultiZonePortGraph(racetrack, _empty_configuration(racetrack))

    assert not port_graph.is_linear_architecture()


def test_portgraph_rejects_branched_architecture() -> None:
    port_graph = MultiZonePortGraph(
        gate_zone_type_examples, _empty_configuration(gate_zone_type_examples)
    )

    assert not port_graph.is_linear_architecture()


def test_portgraph_rejects_grid_architecture() -> None:
    port_graph = MultiZonePortGraph(grid12, _empty_configuration(grid12))

    assert not port_graph.is_linear_architecture()
