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
    MultiZoneArchitectureSpec,
    PortId,
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
    qft_circuit = build_test_circuit(15)
    compiled = backend.compile_circuit(qft_circuit, compilation_settings)
    return backend.route_compiled(compiled, compilation_settings)


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
    assert shuttle_frame.highlight_qubits == [1]
    assert shuttle_frame.shuttle == {
        "source_zone": 0,
        "target_zone": 1,
        "source_port": PortId.p1.value,
        "target_port": PortId.p0.value,
        "qubits": [1],
    }
    pswap_frame = next(frame for frame in frames if frame.kind == "pswap")
    assert pswap_frame.zone_placement == [[0], [2, 1], [], []]
    gate_frame = next(frame for frame in frames if frame.kind == "gate")
    assert gate_frame.highlight_qubits == [1, 2]
    assert gate_frame.zone_placement == [[0], [2, 1], [], []]


def test_generate_multi_zone_circuit_movie_html_contains_embedded_movie_data() -> None:
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(circuit, title="Visualizer Test")

    assert "<!DOCTYPE html>" in html
    assert "Visualizer Test" in html
    assert "movieData" in html
    assert "Initial placement" in html
    assert '"n_qubits": 3' in html
    assert "textContent = String(qubit);" in html
    assert 'return "#3a86ff";' in html
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
    assert "renderFrame(frameIndex);\n  </script>" in html


def test_generate_multi_zone_circuit_movie_html_contains_visual_styling_and_motion_hooks() -> (
    None
):
    circuit = _compiled_visualizer_circuit()

    html = generate_multi_zone_circuit_movie_html(circuit, title="Visualizer Test")

    assert "transport-slot-mark" in html
    assert 'frame.kind === "gate" ? frame.highlight_qubits : []' in html
    assert "function edgeAnchor(zone, port)" in html
    assert "function animateShuttle(frame, previousFrame, positions)" in html
    assert (
        "const sourceAnchor = edgeAnchor(sourceZone, frame.shuttle.source_port);"
        in html
    )
    assert (
        "const targetAnchor = edgeAnchor(targetZone, frame.shuttle.target_port);"
        in html
    )
    assert "const duration = Math.max(120, currentFrameDuration() * 0.99);" in html
    assert "const directionX = dx / length;" in html
    assert "const directionY = dy / length;" in html
    assert "const chainSpacing = 12;" in html
    assert "sourceAnchor.x - chainOffsetX" in html
    assert "sourceAnchor.y - chainOffsetY" in html
    assert "targetAnchor.x + chainOffsetX" in html
    assert "targetAnchor.y + chainOffsetY" in html
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
    assert 'svg.addEventListener("pointermove", handleZoneDrag);' in html
    assert 'svg.addEventListener("pointerup", stopZoneDrag);' in html
    assert "--gate-zone-fill: #f3c2c2;" in html
    assert "--gate-zone-stroke: #ba5c5c;" in html
    assert "label.textContent = `Z${zone.id}`;" in html
    assert (
        "Movie of a routed multi-zone circuit with gate highlights and physical motion."
        not in html
    )
    assert "gate cap" not in html
    assert "transport cap" not in html
    assert " | memory" not in html
    assert " | gate" not in html
    assert "Zone ${zone.id}" not in html


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
            edge["source"],
            edge["target"],
            edge["source_port"],
            edge["target_port"],
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
    assert len(movie.edges) == len(architecture.connections)
    assert len(movie.frames) > 1
    assert any(frame.kind == "gate" for frame in movie.frames)
    assert any(frame.kind in {"shuttle", "pswap"} for frame in movie.frames)
