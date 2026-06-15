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

from pathlib import Path

import pytest
from pytket.circuit import OpType

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.multi_zone_architecture.circuit import (
    MultiZoneCircuit,
    build_multi_zone_circuit_movie,
    build_multi_zone_circuit_movie_frames,
    generate_multi_zone_circuit_movie_html,
    write_multi_zone_circuit_movie_html,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit_visualizer import (
    _best_non_linear_zone_orientations,
    _enforce_minimum_zone_border_distance,
    _junction_layout,
    _raw_port_positions,
    _slot_centers,
    _SlotLayout,
    _zone_layout,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.gate_selection import (
    GreedyGateSelector,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.qubit_routing import (
    GeneralRouter,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_config import (
    RoutingConfig,
)
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.routing_ops import (
    PSwap,
    RoutingBarrier,
    Shuttle,
)
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import (
    CompilationSettings,
)
from pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings import (
    InitialPlacementAlg,
    InitialPlacementSettings,
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
)
from pytket.extensions.aqt.multi_zone_architecture.trap_architecture.named_architectures import (
    four_zones_in_a_line,
    gate_zone_type_examples,
    grid12,
    linear_8_zones,
)
from tests.multi_zone.dangling_single_qubit_circuit_test import build_test_circuit


def _compiled_visualizer_circuit() -> MultiZoneCircuit:
    circuit = MultiZoneCircuit(four_zones_in_a_line, {0: [0, 1], 1: [2]}, 3)
    circuit.add_routing_ops(
        [
            RoutingBarrier(),
            Shuttle([1], 0, 1, PortId.p1, PortId.p0),
            RoutingBarrier(),
            PSwap(1, 1, 2),
            RoutingBarrier(),
        ]
    )
    circuit.add_gate(OpType.XXPhase, [1, 2], [0.5])
    circuit.is_compiled = True
    circuit.validate()
    return circuit


def _compiled_visualizer_multi_gate_circuit() -> MultiZoneCircuit:
    circuit = MultiZoneCircuit(four_zones_in_a_line, {1: [0, 1], 2: [2]}, 3)
    circuit.add_gate(OpType.Rx, [0], [0.25])
    circuit.add_gate(OpType.XXPhase, [0, 1], [0.5])
    circuit.add_gate(OpType.Ry, [2], [0.75])
    circuit.add_routing_ops(
        [
            RoutingBarrier(),
            Shuttle([1], 1, 2, PortId.p1, PortId.p0),
            RoutingBarrier(),
        ]
    )
    circuit.add_gate(OpType.Rz, [0], [0.125])
    circuit.add_gate(OpType.XXPhase, [1, 2], [0.25])
    circuit.is_compiled = True
    circuit.validate()
    return circuit


def _linear_8_zone_visualizer_circuit() -> MultiZoneCircuit:
    circuit = MultiZoneCircuit(
        linear_8_zones,
        {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [9, 10, 11],
            4: [12, 13, 14],
            5: [15, 16, 17],
            6: [18, 19, 20],
            7: [21, 22, 23],
        },
        24,
    )
    circuit.is_compiled = True
    circuit.validate()
    return circuit


def _routed_non_linear_visualizer_circuit(
    architecture: MultiZoneArchitectureSpec,
) -> MultiZoneCircuit:
    backend = AQTMultiZoneBackend(architecture=architecture, access_token="invalid")
    compilation_settings = CompilationSettings(
        pytket_optimisation_level=1,
        initial_placement=InitialPlacementSettings(
            algorithm=InitialPlacementAlg.qubit_order,
            zone_free_space=2,
            max_depth=200,
        ),
        routing=RoutingConfig(
            router=GeneralRouter(),
            gate_selector=GreedyGateSelector(),
        ),
    )
    qft_circuit = build_test_circuit(30)
    compiled = backend.compile_circuit(qft_circuit, compilation_settings)
    return backend.route_compiled(compiled, compilation_settings)


def _branching_orientation_test_circuit() -> MultiZoneCircuit:
    architecture = MultiZoneArchitectureSpec(
        n_qubits_max=0,
        n_zones=3,
        zones=[Zone(max_ions_gate_op=2) for _ in range(3)],
        connections=[
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p0),
                endpoint1=PortSpec(zone_id=1, port_id=PortId.p1),
            ),
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p1),
                endpoint1=PortSpec(zone_id=2, port_id=PortId.p0),
            ),
        ],
    )
    return MultiZoneCircuit(architecture, {}, 0)


def _branched_port_layout_test_circuit() -> MultiZoneCircuit:
    architecture = MultiZoneArchitectureSpec(
        n_qubits_max=0,
        n_zones=4,
        zones=[Zone(max_ions_gate_op=2) for _ in range(4)],
        junctions=[Junction(junction_id=0)],
        connections=[
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p0),
                endpoint1=PortSpec(zone_id=1, port_id=PortId.p1),
            ),
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p1),
                endpoint1=JunctionRef(junction_id=0),
            ),
            PhysicalConnection(
                endpoint0=JunctionRef(junction_id=0),
                endpoint1=PortSpec(zone_id=2, port_id=PortId.p0),
            ),
            PhysicalConnection(
                endpoint0=JunctionRef(junction_id=0),
                endpoint1=PortSpec(zone_id=3, port_id=PortId.p0),
            ),
        ],
    )
    return MultiZoneCircuit(architecture, {}, 0)


def test_build_multi_zone_circuit_movie_frames_tracks_shuttles_swaps_and_gates() -> (
    None
):
    circuit = _compiled_visualizer_circuit()

    frames = build_multi_zone_circuit_movie_frames(circuit)

    assert [frame.kind for frame in frames] == ["initial", "shuttle", "pswap", "gate"]
    assert frames[0].kind == "initial"
    assert frames[0].zone_placement == [[0, 1], [2], [], []]
    shuttle_frame = next(frame for frame in frames if frame.kind == "shuttle")
    assert shuttle_frame.zone_placement == [[0], [1, 2], [], []]
    assert shuttle_frame.command_text == "SHUTTLE(0→1) 1;"
    assert shuttle_frame.highlight_qubits == [1]
    assert shuttle_frame.shuttle == {
        "source_zone": 0,
        "target_zone": 1,
        "source_port": PortId.p1.value,
        "target_port": PortId.p0.value,
        "path": [
            {"kind": "port", "zone": 0, "port": PortId.p1.value},
            {"kind": "port", "zone": 1, "port": PortId.p0.value},
        ],
        "qubits": [1],
    }
    pswap_frame = next(frame for frame in frames if frame.kind == "pswap")
    assert pswap_frame.zone_placement == [[0], [2, 1], [], []]
    assert pswap_frame.command_text == "PSWAP 1↔2;"
    gate_frame = next(frame for frame in frames if frame.kind == "gate")
    assert gate_frame.command_text == "QOPS 1-2;"
    assert gate_frame.highlight_qubits == [1, 2]
    assert gate_frame.zone_placement == [[0], [2, 1], [], []]


def test_build_multi_zone_circuit_movie_frames_condenses_contiguous_gate_blocks() -> (
    None
):
    circuit = _compiled_visualizer_multi_gate_circuit()

    frames = build_multi_zone_circuit_movie_frames(circuit)

    assert [frame.kind for frame in frames] == [
        "initial",
        "gate",
        "new_target",
        "shuttle",
        "gate",
    ]
    assert frames[0].upcoming_qubits == [0, 1, 2]
    assert frames[0].upcoming_gate_zone_by_qubit == {0: 1, 1: 1, 2: 2}
    assert frames[1].command_text == "QOPS 0-2;"
    assert frames[1].highlight_qubits == [0, 1, 2]
    assert frames[2].command_text == "New target"
    assert frames[2].zone_placement == frames[1].zone_placement
    assert frames[2].upcoming_gate_zone_by_qubit == {1: 2, 2: 2, 0: 1}
    assert frames[3].kind == "shuttle"
    assert frames[4].command_text == "QOPS 0-2;"
    assert frames[4].highlight_qubits == [0, 1, 2]


def test_build_multi_zone_circuit_movie_frames_tracks_upcoming_gate_zones() -> None:
    circuit = _compiled_visualizer_multi_gate_circuit()

    frames = build_multi_zone_circuit_movie_frames(circuit, condense_quantum_ops=False)

    shuttle_frame = next(frame for frame in frames if frame.kind == "shuttle")

    assert shuttle_frame.upcoming_qubits == [0, 1, 2]
    assert shuttle_frame.upcoming_gate_zone_by_qubit == {1: 2, 2: 2, 0: 1}


def test_build_multi_zone_circuit_movie_frames_can_keep_individual_gate_frames() -> (
    None
):
    circuit = _compiled_visualizer_multi_gate_circuit()

    frames = build_multi_zone_circuit_movie_frames(circuit, condense_quantum_ops=False)

    assert [frame.kind for frame in frames] == [
        "initial",
        "gate",
        "gate",
        "gate",
        "new_target",
        "shuttle",
        "gate",
        "gate",
    ]
    assert frames[1].command_text.startswith("Rx(")
    assert frames[2].command_text.startswith("Ry(")
    assert frames[3].command_text.startswith("XXPhase(")
    assert frames[4].command_text == "New target"
    assert frames[4].upcoming_gate_zone_by_qubit == {1: 2, 2: 2, 0: 1}
    assert frames[5].command_text == "SHUTTLE(1→2) 1;"
    assert frames[6].command_text.startswith("Rz(")
    assert frames[7].command_text.startswith("XXPhase(")


def test_generate_multi_zone_circuit_movie_html_contains_embedded_movie_data() -> None:
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(circuit, title="Visualizer Test")

    assert "<!DOCTYPE html>" in html
    assert "Visualizer Test" in html
    assert "movieData" in html
    assert "Initial placement" in html
    assert "QOPS 1-2;" in html
    assert "SHUTTLE(0\\u21921) 1;" in html
    assert "PSWAP 1\\u21942;" in html
    assert '"n_qubits": 3' in html
    assert "textContent = String(qubit);" in html
    assert 'return "#111111";' in html
    assert 'const operationsList = document.getElementById("operations-list");' in html
    assert "const currentOperationRow = 10;" in html
    assert "const visibleOperationRows = 16;" in html
    assert "function upcomingQubitColor(targetGateZone)" in html
    assert "let isPlaying = false;" in html
    assert '<button id="play-pause" type="button">Play</button>' in html
    assert (
        '<input id="frame-duration" type="number" min="1" step="0.1" value=300.0>'
        in html
    )
    assert (
        'const frameDurationInput = document.getElementById("frame-duration");' in html
    )
    assert "const parsedDuration = Number(frameDurationInput.value);" in html
    assert "return parsedDuration;" in html
    assert "return movieData.frame_duration_ms;" in html
    assert 'frameDurationInput.addEventListener("change", () => {' in html
    assert "if (frameIndex >= movieData.frames.length - 1)" in html
    assert 'playPause.textContent = "Restart";' in html
    assert "if (!isPlaying && frameIndex >= movieData.frames.length - 1)" in html
    assert "frameIndex = 0;" in html
    assert "renderFrame(frameIndex, false);" in html
    assert "renderFrame(frameIndex);\n  </script>" in html


def test_build_multi_zone_circuit_movie_assigns_distinct_gate_zone_colors() -> None:
    movie = build_multi_zone_circuit_movie(_compiled_visualizer_multi_gate_circuit())

    gate_zones = [zone for zone in movie.zones if zone["is_gate_zone"]]

    assert len(gate_zones) == 2
    assert gate_zones[0]["gate_color"] != gate_zones[1]["gate_color"]
    assert gate_zones[0]["gate_qubit_color"] != gate_zones[1]["gate_qubit_color"]
    assert gate_zones[0]["gate_color"] == "#f6d2a4"
    assert gate_zones[1]["gate_color"] == "#f3e49d"


def test_generate_multi_zone_circuit_movie_html_can_disable_quantum_op_condensing() -> (
    None
):
    circuit = _compiled_visualizer_multi_gate_circuit()

    html = generate_multi_zone_circuit_movie_html(
        circuit,
        title="Uncondensed Visualizer Test",
        condense_quantum_ops=False,
    )

    assert "QOPS 0-2;" not in html
    assert "Rx(0.25) q[0];" in html
    assert "XXPhase(0.5) q[0], q[1];" in html
    assert "Ry(0.75) q[2];" in html


def test_generate_multi_zone_circuit_movie_html_contains_operation_stream_hooks() -> (
    None
):
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(circuit, title="Visualizer Test")

    assert ".operations-panel {" in html
    assert ".operation-row.gate {" in html
    assert ".operation-row.movement {" in html
    assert ".operation-row.current {" in html
    assert "function renderOperationStream(index)" in html
    assert 'row.className = "operation-row";' in html
    assert 'row.classList.add("gate");' in html
    assert 'row.classList.add("movement");' in html
    assert 'row.classList.add("current");' in html
    assert "row.textContent = frameData.command_text;" in html


def test_generate_multi_zone_circuit_movie_html_contains_visual_styling_and_motion_hooks() -> (  # noqa: PLR0915
    None
):
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(circuit, title="Visualizer Test")

    assert "transport-slot-mark" in html
    assert 'frame.kind === "gate" ? frame.highlight_qubits : []' in html
    assert "function edgeAnchor(zone, port)" in html
    assert "function animateShuttle(frame, previousFrame, positions)" in html
    assert "function animatePswap(frame, previousFrame, positions)" in html
    assert "function physicalNodePosition(node)" in html
    assert "function pointAlongPolyline(points, progress)" in html
    assert (
        "const route = frame.shuttle.path.map((node) => physicalNodePosition(node));"
        in html
    )
    assert "const duration = Math.max(120, currentFrameDuration() * 0.99);" in html
    assert "const directionX = dx / length;" in html
    assert "const directionY = dy / length;" in html
    assert "const chainSpacing = 12;" in html
    assert "pointAlongPolyline(path.points, progress)" in html
    assert "x: point.x + chainOffsetX" in html
    assert "y: point.y + chainOffsetY" in html
    assert "const angle = progress * Math.PI;" in html
    assert (
        "const x = path.centerX + (path.relX * cosAngle) - (path.relY * sinAngle);"
        in html
    )
    assert (
        "const y = path.centerY + (path.relX * sinAngle) + (path.relY * cosAngle);"
        in html
    )
    assert "x: port === 0 ? zone.x : zone.x + zone.width" in html
    assert "function translateZone(zone, dx, dy)" in html
    assert "function applyZoneOrientation(zone)" in html
    assert "zone.orientation = (zone.orientation + 1) % 4;" in html
    assert "if (zone.orientation === 1)" in html
    assert "if (zone.orientation === 2)" in html
    assert "if (zone.orientation === 3)" in html
    assert "y: port === 0 ? zone.y + zone.height : zone.y" in html
    assert "x: port === 0 ? zone.x + zone.width : zone.x" in html
    assert "y: port === 0 ? zone.y : zone.y + zone.height" in html
    assert "function toggleZoneOrientation(zoneId)" in html
    assert "function updateZoneGraphics(zoneId)" in html
    assert 'rotateIcon.textContent = "↻";' in html
    assert (
        'box.addEventListener("pointerdown", (event) => startZoneDrag(zone.id, event));'
        in html
    )
    assert (
        'label.addEventListener("pointerdown", (event) => startZoneDrag(zone.id, event));'
        in html
    )
    assert 'rotateHandle.addEventListener("pointerdown", (event) => {' in html
    assert "function startJunctionDrag(junctionId, event)" in html
    assert "function updateJunctionGraphics(junctionId)" in html
    assert 'class: "junction-node"' in html
    assert "junction-label" not in html
    assert 'svg.addEventListener("pointermove", handleDrag);' in html
    assert 'svg.addEventListener("pointerup", stopDrag);' in html
    assert 'fill: zone.is_gate_zone ? zone.gate_color : "var(--zone-fill)"' in html
    assert (
        'stroke: zone.is_gate_zone ? zone.gate_stroke_color : "var(--zone-stroke)"'
        in html
    )
    assert "fill: var(--zone-fill);" not in html
    assert "color: #b91c1c;" in html
    assert "label.textContent = `Z${zone.id}`;" in html
    assert '} else if (animate && frame.kind === "pswap") {' in html
    assert (
        "Movie of a routed multi-zone circuit with gate highlights and physical motion."
        not in html
    )
    assert "gate cap" not in html
    assert "transport cap" not in html
    assert " | memory" not in html
    assert " | gate" not in html
    assert "Zone ${zone.id}" not in html
    assert "color: var(--highlight);" not in html


def test_write_multi_zone_circuit_movie_html_writes_file(tmp_path: Path) -> None:
    circuit = _compiled_visualizer_circuit()
    output_path = tmp_path / "movie.html"

    written_path = write_multi_zone_circuit_movie_html(
        circuit,
        output_path,
        title="Written Movie",
    )
    movie = build_multi_zone_circuit_movie(circuit, title="Written Movie")

    assert written_path == output_path
    assert output_path.exists()
    assert f'"title": "{movie.title}"' in output_path.read_text(encoding="utf-8")


def test_build_multi_zone_circuit_movie_edges_include_port_anchors() -> None:
    circuit = _compiled_visualizer_circuit()

    movie = build_multi_zone_circuit_movie(circuit)

    assert {
        (
            edge["source"]["zone"],
            edge["target"]["zone"],
            edge["source"]["port"],
            edge["target"]["port"],
        )
        for edge in movie.edges
    } == {
        (0, 1, PortId.p1.value, PortId.p0.value),
        (1, 2, PortId.p1.value, PortId.p0.value),
        (2, 3, PortId.p1.value, PortId.p0.value),
    }
    assert all(zone["orientation"] == 0 for zone in movie.zones)
    assert all(zone["base_width"] == zone["width"] for zone in movie.zones)
    assert all(zone["base_height"] == zone["height"] for zone in movie.zones)


def test_visualizer_uses_architecture_visualization_positions() -> None:
    architecture = MultiZoneArchitectureSpec(
        n_qubits_max=0,
        n_zones=2,
        zones=[Zone(max_ions_gate_op=2) for _ in range(2)],
        junctions=[Junction(junction_id=0)],
        connections=[
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p1),
                endpoint1=JunctionRef(junction_id=0),
            ),
            PhysicalConnection(
                endpoint0=JunctionRef(junction_id=0),
                endpoint1=PortSpec(zone_id=1, port_id=PortId.p0),
            ),
        ],
        visualization=VisualizationSpec(
            zone_positions={
                0: LayoutPosition(x=0.0, y=0.0),
                1: LayoutPosition(x=10.0, y=0.0),
            },
            junction_positions={0: LayoutPosition(x=0.0, y=0.0)},
        ),
    )
    circuit = MultiZoneCircuit(architecture, {}, 0)

    zones = _zone_layout(circuit)
    junctions = _junction_layout(circuit, zones)

    assert zones[0]["center_x"] == junctions[0]["x"]
    assert zones[0]["center_y"] == junctions[0]["y"]


def test_visualizer_preserves_visualization_position_aspect_ratio() -> None:
    architecture = MultiZoneArchitectureSpec(
        n_qubits_max=0,
        n_zones=2,
        zones=[Zone(max_ions_gate_op=2) for _ in range(2)],
        junctions=[Junction(junction_id=0)],
        connections=[
            PhysicalConnection(
                endpoint0=PortSpec(zone_id=0, port_id=PortId.p1),
                endpoint1=JunctionRef(junction_id=0),
            ),
            PhysicalConnection(
                endpoint0=JunctionRef(junction_id=0),
                endpoint1=PortSpec(zone_id=1, port_id=PortId.p0),
            ),
        ],
        visualization=VisualizationSpec(
            zone_positions={
                0: LayoutPosition(x=0.0, y=0.0),
                1: LayoutPosition(x=10.0, y=0.0),
            },
            junction_positions={0: LayoutPosition(x=0.0, y=5.0)},
        ),
    )
    circuit = MultiZoneCircuit(architecture, {}, 0)

    zones = _zone_layout(circuit)
    junctions = _junction_layout(circuit, zones)

    rendered_dx = abs(zones[1]["center_x"] - zones[0]["center_x"])
    rendered_dy = abs(junctions[0]["y"] - zones[0]["center_y"])
    assert rendered_dx == pytest.approx(2 * rendered_dy)


def test_generate_multi_zone_circuit_movie_html_uses_custom_frame_duration() -> None:
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(
        circuit, title="Custom Duration", frame_duration_ms=123.5
    )

    assert '"frame_duration_ms": 123.5' in html
    assert (
        '<input id="frame-duration" type="number" min="1" step="0.1" value=123.5>'
        in html
    )


def test_zone_layout_wraps_linear_architecture_without_overlap() -> None:
    circuit = _linear_8_zone_visualizer_circuit()

    zones = _zone_layout(circuit)

    for zone_index, zone in enumerate(zones):
        for other_zone in zones[zone_index + 1 :]:
            overlaps_horizontally = (
                zone["x"] < other_zone["x"] + other_zone["width"]
                and other_zone["x"] < zone["x"] + zone["width"]
            )
            overlaps_vertically = (
                zone["y"] < other_zone["y"] + other_zone["height"]
                and other_zone["y"] < zone["y"] + zone["height"]
            )
            assert not (overlaps_horizontally and overlaps_vertically)
    assert len({zone["y"] for zone in zones}) > 1
    assert all(zone["height"] == 52 for zone in zones)


@pytest.mark.parametrize(
    ("centers", "expected_orientation"),
    [
        pytest.param(
            {0: (0.0, 0.0), 1: (0.0, 10.0), 2: (0.0, -10.0)},
            1,
            id="vertical-facing",
        ),
        pytest.param(
            {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (-10.0, 0.0)},
            2,
            id="horizontal-reversed-facing",
        ),
    ],
)
def test_best_non_linear_zone_orientations_faces_connected_ports_towards_neighbors(
    centers: dict[int, tuple[float, float]],
    expected_orientation: int,
) -> None:
    circuit = _branching_orientation_test_circuit()

    orientations = _best_non_linear_zone_orientations(circuit, centers)

    assert orientations[0] == expected_orientation


def test_raw_port_positions_keep_connected_ports_closer_than_unconnected_ports() -> (
    None
):
    circuit = _branched_port_layout_test_circuit()

    port_positions = _raw_port_positions(circuit)

    zone0_p0 = port_positions[(0, 0)]
    zone0_p1 = port_positions[(0, 1)]
    zone1_p1 = port_positions[(1, 1)]
    zone2_p0 = port_positions[(2, 0)]
    zone3_p0 = port_positions[(3, 0)]

    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    assert distance(zone0_p0, zone1_p1) < distance(zone0_p1, zone1_p1)
    assert distance(zone0_p1, zone2_p0) < distance(zone0_p0, zone2_p0)
    assert distance(zone0_p1, zone3_p0) < distance(zone0_p0, zone3_p0)


def test_enforce_minimum_zone_border_distance_separates_overlapping_zones() -> None:
    centers = {
        0: (100.0, 100.0),
        1: (120.0, 110.0),
    }
    box_sizes = {
        0: (160.0, 52.0),
        1: (160.0, 52.0),
    }

    adjusted = _enforce_minimum_zone_border_distance(
        centers, box_sizes, min_border_gap=26.0
    )

    dx = abs(adjusted[1][0] - adjusted[0][0])
    dy = abs(adjusted[1][1] - adjusted[0][1])
    required_dx = ((box_sizes[0][0] + box_sizes[1][0]) / 2) + 26.0
    required_dy = ((box_sizes[0][1] + box_sizes[1][1]) / 2) + 26.0
    assert dx >= required_dx or dy >= required_dy


def test_slot_centers_use_single_horizontal_row() -> None:
    slots = _slot_centers(
        _SlotLayout(
            x=100,
            y=50,
            width=160,
            height=84,
            gate_capacity=5,
            transport_capacity=7,
        )
    )

    assert len({slot["y"] for slot in slots}) == 1
    assert [slot["x"] for slot in slots] == sorted(slot["x"] for slot in slots)
    assert [slot["transport_only"] for slot in slots] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    ("architecture", "title"),
    [
        pytest.param(grid12, "Grid12 QFT Movie", id="grid12"),
        pytest.param(
            gate_zone_type_examples,
            "Gate Zone Types QFT Movie",
            id="gate-zone-type-examples",
        ),
    ],
)
def test_visualizer_handles_routed_non_linear_architectures(
    architecture: MultiZoneArchitectureSpec,
    title: str,
    tmp_path: Path,
) -> None:
    circuit = _routed_non_linear_visualizer_circuit(architecture)
    output_path = tmp_path / "movie.html"
    write_multi_zone_circuit_movie_html(
        circuit,
        output_path,
        title=title,
    )
    movie = build_multi_zone_circuit_movie(circuit, title="Written Movie")

    assert circuit.macro_arch.is_linear_architecture is False
    assert len(movie.zones) == architecture.n_zones
    assert len(movie.junctions) == len(architecture.junctions)
    assert len(movie.edges) == len(architecture.connections)
    assert len(movie.frames) > 1
    assert any(frame.kind == "gate" for frame in movie.frames)
    assert any(frame.kind in {"shuttle", "pswap"} for frame in movie.frames)
