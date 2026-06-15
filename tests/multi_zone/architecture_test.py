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
from pydantic import ValidationError

from pytket.extensions.aqt.multi_zone_architecture.circuit.helpers import (
    TrapConfiguration,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture import (
    Junction,
    JunctionRef,
    LayoutPosition,
    MultiZoneArchitectureSpec,
    PhysicalConnection,
    PortId,
    PortSpec,
    VisualizationSpec,
    Zone,
    ZoneConnection,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.architecture_portgraph import (
    MultiZonePortGraph,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.macro_architecture_graph import (
    MultiZoneArch,
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    gate_zone_type_examples,
    grid12,
    grid12_mod,
)


def test_architecture_str() -> None:
    four_zones_in_a_line_str = four_zones_in_a_line.__str__()
    assert isinstance(four_zones_in_a_line_str, str)


def _port(zone_id: int, port_id: PortId) -> PortSpec:
    return PortSpec(zone_id=zone_id, port_id=port_id)


def _junction(junction_id: int) -> JunctionRef:
    return JunctionRef(junction_id=junction_id)


def _basic_architecture(
    *,
    n_zones: int = 3,
    junctions: list[Junction] | None = None,
    connections: list[PhysicalConnection] | None = None,
) -> MultiZoneArchitectureSpec:
    return MultiZoneArchitectureSpec(
        n_qubits_max=6,
        n_zones=n_zones,
        zones=[Zone(max_ions_gate_op=2) for _ in range(n_zones)],
        junctions=junctions or [],
        connections=connections or [],
    )


def test_junction_connections_expand_to_cheapest_pairwise_connections() -> None:
    architecture = _basic_architecture(
        junctions=[
            Junction(junction_id=0, cost=7),
            Junction(junction_id=1, cost=11),
            Junction(junction_id=2, cost=13),
            Junction(junction_id=3, cost=1),
        ],
        connections=[
            PhysicalConnection(
                endpoint0=_port(0, PortId.p0),
                endpoint1=_junction(0),
                shuttle_cost=2,
            ),
            PhysicalConnection(
                endpoint0=_junction(0), endpoint1=_junction(1), shuttle_cost=3
            ),
            PhysicalConnection(
                endpoint0=_junction(1), endpoint1=_junction(2), shuttle_cost=50
            ),
            PhysicalConnection(
                endpoint0=_junction(0), endpoint1=_junction(3), shuttle_cost=4
            ),
            PhysicalConnection(
                endpoint0=_junction(3), endpoint1=_junction(2), shuttle_cost=5
            ),
            PhysicalConnection(
                endpoint0=_junction(2),
                endpoint1=_port(1, PortId.p0),
                shuttle_cost=6,
            ),
            PhysicalConnection(
                endpoint0=_junction(1),
                endpoint1=_port(2, PortId.p0),
                shuttle_cost=8,
            ),
        ],
    )

    connection_costs = {
        frozenset(
            (
                (
                    connection.zone_port_spec0.zone_id,
                    connection.zone_port_spec0.port_id,
                ),
                (
                    connection.zone_port_spec1.zone_id,
                    connection.zone_port_spec1.port_id,
                ),
            )
        ): connection.shuttle_cost
        for connection in architecture.port_to_port_connections
    }

    assert (
        connection_costs[frozenset(((0, PortId.p0), (1, PortId.p0)))]
        == 2 + 7 + 4 + 1 + 5 + 13 + 6
    )
    assert (
        connection_costs[frozenset(((0, PortId.p0), (2, PortId.p0)))]
        == 2 + 7 + 3 + 11 + 8
    )
    assert (
        connection_costs[frozenset(((1, PortId.p0), (2, PortId.p0)))]
        == 6 + 13 + 5 + 1 + 4 + 7 + 3 + 11 + 8
    )


def test_zone_port_can_have_only_one_physical_connection() -> None:
    with pytest.raises(ValueError, match="more than one physical connection"):
        _basic_architecture(
            junctions=[Junction(junction_id=0), Junction(junction_id=1)],
            connections=[
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_junction(0),
                ),
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_junction(1),
                ),
            ],
        )


def test_zone_port_direct_connection_counts_against_physical_degree() -> None:
    with pytest.raises(ValueError, match="more than one physical connection"):
        _basic_architecture(
            n_zones=2,
            connections=[
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_port(1, PortId.p0),
                ),
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_junction(0),
                ),
            ],
            junctions=[Junction(junction_id=0)],
        )


def test_junction_connections_reject_external_path_between_ports_of_same_zone() -> None:
    with pytest.raises(ValueError, match="external path between its two ports"):
        _basic_architecture(
            n_zones=1,
            junctions=[Junction(junction_id=0)],
            connections=[
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_junction(0),
                ),
                PhysicalConnection(
                    endpoint0=_junction(0),
                    endpoint1=_port(0, PortId.p1),
                ),
            ],
        )


def test_macro_architecture_uses_cheapest_connection_between_same_zones() -> None:
    architecture = _basic_architecture(
        n_zones=2,
        connections=[
            PhysicalConnection(
                endpoint0=_port(0, PortId.p0),
                endpoint1=_port(1, PortId.p0),
                shuttle_cost=10,
            ),
            PhysicalConnection(
                endpoint0=_port(0, PortId.p1),
                endpoint1=_port(1, PortId.p1),
                shuttle_cost=3,
            ),
        ],
    )

    macro_architecture = MultiZoneArch(architecture)

    assert macro_architecture.shortest_path_with_length(0, 1)[0] == 3
    assert macro_architecture.get_connected_ports(0, 1) == (PortId.p1, PortId.p1)


def test_port_graph_uses_expanded_junction_connection_cost() -> None:
    architecture = _basic_architecture(
        n_zones=2,
        junctions=[Junction(junction_id=0, cost=5)],
        connections=[
            PhysicalConnection(
                endpoint0=_port(0, PortId.p0),
                endpoint1=_junction(0),
                shuttle_cost=2,
            ),
            PhysicalConnection(
                endpoint0=_junction(0),
                endpoint1=_port(1, PortId.p0),
                shuttle_cost=3,
            ),
        ],
    )
    port_graph = MultiZonePortGraph(
        architecture,
        TrapConfiguration(n_qubits=0, zone_placement=[[], []]),
    )

    assert port_graph.shuttle_edge_transport_cost(0, 0, 1, 0) == 10


def test_port_to_port_connections_are_cached_on_architecture() -> None:
    architecture = _basic_architecture(
        n_zones=2,
        connections=[
            PhysicalConnection(
                endpoint0=_port(0, PortId.p0),
                endpoint1=_port(1, PortId.p0),
            )
        ],
    )

    assert len(architecture.port_to_port_connections) == 1
    assert (
        architecture.port_to_port_connections is architecture.port_to_port_connections
    )


def test_old_zone_connection_api_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _basic_architecture(
            n_zones=2,
            connections=[
                ZoneConnection(
                    zone_port_spec0=_port(0, PortId.p0),
                    zone_port_spec1=_port(1, PortId.p0),
                )
            ],  # type: ignore[list-item]
        )


def test_old_junction_connections_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MultiZoneArchitectureSpec(
            n_qubits_max=2,
            n_zones=2,
            zones=[Zone(max_ions_gate_op=2) for _ in range(2)],
            junctions=[Junction(junction_id=0)],
            connections=[],
            junction_connections=[
                PhysicalConnection(
                    endpoint0=_port(0, PortId.p0),
                    endpoint1=_junction(0),
                )
            ],
        )


def test_visualization_spec_rejects_missing_or_unknown_positions() -> None:
    with pytest.raises(ValueError, match="zone positions"):
        MultiZoneArchitectureSpec(
            n_qubits_max=2,
            n_zones=2,
            zones=[Zone(max_ions_gate_op=2) for _ in range(2)],
            visualization=VisualizationSpec(
                zone_positions={0: LayoutPosition(x=0.0, y=0.0)}
            ),
        )

    with pytest.raises(ValueError, match="zone positions"):
        MultiZoneArchitectureSpec(
            n_qubits_max=2,
            n_zones=2,
            zones=[Zone(max_ions_gate_op=2) for _ in range(2)],
            visualization=VisualizationSpec(
                zone_positions={
                    0: LayoutPosition(x=0.0, y=0.0),
                    2: LayoutPosition(x=1.0, y=0.0),
                }
            ),
        )

    with pytest.raises(ValueError, match="junction positions"):
        MultiZoneArchitectureSpec(
            n_qubits_max=2,
            n_zones=1,
            zones=[Zone(max_ions_gate_op=2)],
            junctions=[Junction(junction_id=0), Junction(junction_id=1)],
            visualization=VisualizationSpec(
                junction_positions={0: LayoutPosition(x=0.0, y=0.0)}
            ),
        )


@pytest.mark.parametrize("architecture", [grid12, grid12_mod, gate_zone_type_examples])
def test_named_non_linear_architectures_define_visualization_positions(
    architecture: MultiZoneArchitectureSpec,
) -> None:
    assert architecture.visualization is not None
    assert len(architecture.visualization.zone_positions) == architecture.n_zones
    assert len(architecture.visualization.junction_positions) == len(
        architecture.junctions
    )
