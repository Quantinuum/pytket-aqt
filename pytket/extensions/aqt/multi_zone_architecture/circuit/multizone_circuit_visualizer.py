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

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from networkx import Graph, spring_layout
from pytket.circuit import OpType

from ..trap_architecture.architecture import PortId
from ..trap_architecture.architecture_portgraph import (
    port_id_to_zone_port,
    zone_port_to_port_id,
)
from .multizone_circuit import MultiZoneCircuit, ValidationError

_SVG_WIDTH = 1280
_SVG_HEIGHT = 760
_LAYOUT_MARGIN_X = 64
_LAYOUT_MARGIN_Y = 88
_ZONE_GAP_X = 24
_ZONE_GAP_Y = 116
_ZONE_MIN_WIDTH = 132
_ZONE_MAX_WIDTH = 188
_ZONE_HEIGHT = 52
_SLOT_GAP_X = 24
_SLOT_MARGIN_X = 20
_NON_LINEAR_ZONE_BORDER_GAP = _ZONE_HEIGHT / 2
_ORIENTATION_HORIZONTAL = 0
_ORIENTATION_VERTICAL_P0_BOTTOM = 1
_ORIENTATION_HORIZONTAL_REVERSED = 2
_ORIENTATION_VERTICAL_P0_TOP = 3
_PORT_GRAPH_INTERNAL_WEIGHT = 4.0
_PORT_GRAPH_CONNECTION_WEIGHT = 1.0
_QUANTUM_GATE_TYPES = {
    OpType.CX,
    OpType.CY,
    OpType.CZ,
    OpType.H,
    OpType.Measure,
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.S,
    OpType.Sdg,
    OpType.SWAP,
    OpType.T,
    OpType.Tdg,
    OpType.XXPhase,
    OpType.Y,
    OpType.Z,
    OpType.ZZPhase,
}


@dataclass
class MultiZoneCircuitMovieFrame:
    command_index: int | None
    command_text: str
    kind: str
    zone_placement: list[list[int]]
    highlight_qubits: list[int]
    upcoming_qubits: list[int]
    upcoming_gate_zone_by_qubit: dict[int, int]
    shuttle: dict[str, Any] | None = None


@dataclass(frozen=True)
class MultiZoneCircuitMovie:
    title: str
    n_qubits: int
    zones: list[dict[str, Any]]
    edges: list[dict[str, int]]
    frames: list[MultiZoneCircuitMovieFrame]


@dataclass(frozen=True)
class _SlotLayout:
    x: float
    y: float
    width: float
    height: float
    gate_capacity: int
    transport_capacity: int


def _gate_zone_color_map(circuit: MultiZoneCircuit) -> dict[int, dict[str, str]]:
    gate_zone_palette = [
        {"fill": "#f6d2a4", "stroke": "#cf8b2d", "qubit": "#d99a32"},
        {"fill": "#f3e49d", "stroke": "#b99a26", "qubit": "#d5b227"},
        {"fill": "#d9e9c3", "stroke": "#7ba348", "qubit": "#8ab14d"},
        {"fill": "#c6e4df", "stroke": "#4c9a90", "qubit": "#51aaa0"},
        {"fill": "#d7d5f5", "stroke": "#746db6", "qubit": "#7d74d4"},
        {"fill": "#f1d7bf", "stroke": "#b87f4a", "qubit": "#c9894d"},
    ]
    gate_zones = sorted(circuit.macro_arch.gate_zones)
    return {
        zone: gate_zone_palette[index % len(gate_zone_palette)]
        for index, zone in enumerate(gate_zones)
    }


def build_multi_zone_circuit_movie(
    circuit: MultiZoneCircuit,
    *,
    title: str | None = None,
    highlight_upcoming_qubits: bool = True,
    condense_quantum_ops: bool = True,
) -> MultiZoneCircuitMovie:
    if not circuit.is_compiled:
        raise ValueError(
            "Multi-zone circuit movie generation requires a compiled routed circuit."
        )
    zones = _zone_layout(circuit)
    gate_zone_colors = _gate_zone_color_map(circuit)
    for zone in zones:
        if zone["is_gate_zone"]:
            zone["gate_color"] = gate_zone_colors[zone["id"]]["fill"]
            zone["gate_stroke_color"] = gate_zone_colors[zone["id"]]["stroke"]
            zone["gate_qubit_color"] = gate_zone_colors[zone["id"]]["qubit"]
    edges = [
        {
            "source": int(source_zone),
            "target": int(target_zone),
            "source_port": circuit.macro_arch.get_connected_ports(
                int(source_zone), int(target_zone)
            )[0].value,
            "target_port": circuit.macro_arch.get_connected_ports(
                int(source_zone), int(target_zone)
            )[1].value,
        }
        for source_zone, target_zone in circuit.macro_arch.zone_graph.edges()
    ]
    frames = build_multi_zone_circuit_movie_frames(
        circuit,
        highlight_upcoming_qubits=highlight_upcoming_qubits,
        condense_quantum_ops=condense_quantum_ops,
    )
    return MultiZoneCircuitMovie(
        title="Multi-Zone Circuit Movie" if title is None else title,
        n_qubits=circuit.pytket_circuit.n_qubits,
        zones=zones,
        edges=edges,
        frames=frames,
    )


def build_multi_zone_circuit_movie_frames(
    circuit: MultiZoneCircuit,
    *,
    highlight_upcoming_qubits: bool = True,
    condense_quantum_ops: bool = True,
) -> list[MultiZoneCircuitMovieFrame]:
    frames = _build_raw_multi_zone_circuit_movie_frames(circuit)
    if highlight_upcoming_qubits:
        frames = _set_upcoming_qubits(frames)
    frames = _insert_new_target_frames(frames)
    if condense_quantum_ops:
        frames = _condense_quantum_gate_frames(frames)
    return frames


def _build_raw_multi_zone_circuit_movie_frames(
    circuit: MultiZoneCircuit,
) -> list[MultiZoneCircuitMovieFrame]:
    if not circuit.is_compiled:
        raise ValueError(
            "Multi-zone circuit movie generation requires a compiled routed circuit."
        )

    current_placement = deepcopy(circuit.initial_zone_to_qubits)
    frames = [
        MultiZoneCircuitMovieFrame(
            command_index=None,
            command_text="Initial placement",
            kind="initial",
            zone_placement=deepcopy(current_placement),
            highlight_qubits=[],
            upcoming_qubits=[],
            upcoming_gate_zone_by_qubit={},
            shuttle=None,
        )
    ]

    for command_index, cmd in enumerate(circuit.pytket_circuit.get_commands()):
        op = cmd.op
        op_string = f"{op}"
        if command_index < circuit.architecture.n_zones and "INIT" not in op_string:
            raise ValidationError(
                "All zones must be initialized before movie playback begins."
            )

        if "INIT" in op_string:
            continue
        if "MOVE_BARRIER" in op_string or op.type == OpType.Barrier:
            continue
        if "PSWAP" in op_string:
            _apply_pswap(current_placement, cmd)
            frame = MultiZoneCircuitMovieFrame(
                command_index=command_index,
                command_text=_format_pswap_command_text(cmd),
                kind="pswap",
                zone_placement=deepcopy(current_placement),
                highlight_qubits=[arg.index[0] for arg in cmd.args],
                upcoming_qubits=[],
                upcoming_gate_zone_by_qubit={},
            )
        elif "SHUTTLE" in op_string:
            source_zone, target_zone, source_port, target_port = (
                int(param) for param in cmd.op.params
            )
            _apply_shuttle(circuit, current_placement, cmd)
            frame = MultiZoneCircuitMovieFrame(
                command_index=command_index,
                command_text=_format_shuttle_command_text(cmd),
                kind="shuttle",
                zone_placement=deepcopy(current_placement),
                highlight_qubits=[arg.index[0] for arg in cmd.args],
                upcoming_qubits=[],
                upcoming_gate_zone_by_qubit={},
                shuttle={
                    "source_zone": source_zone,
                    "target_zone": target_zone,
                    "source_port": source_port,
                    "target_port": target_port,
                    "qubits": [arg.index[0] for arg in cmd.args],
                },
            )
        elif _is_quantum_gate(op.type, op_string):
            frame = MultiZoneCircuitMovieFrame(
                command_index=command_index,
                command_text=str(cmd),
                kind="gate",
                zone_placement=deepcopy(current_placement),
                highlight_qubits=[arg.index[0] for arg in cmd.args],
                upcoming_qubits=[],
                upcoming_gate_zone_by_qubit={},
                shuttle=None,
            )
        else:
            continue
        frames.append(frame)

    return frames


def _set_upcoming_qubits(
    frames: list[MultiZoneCircuitMovieFrame],
) -> list[MultiZoneCircuitMovieFrame]:
    involved_qubits: set[int] = set()
    upcoming_gate_zone_by_qubit: dict[int, int] = {}
    reset = False
    for frame in reversed(frames):
        if frame.kind == "gate":
            if reset:
                involved_qubits.clear()
                upcoming_gate_zone_by_qubit.clear()
                reset = False
            frame.upcoming_qubits = list(involved_qubits)
            frame.upcoming_gate_zone_by_qubit = upcoming_gate_zone_by_qubit.copy()
            qubit_to_zone = {
                qubit: zone
                for zone, zone_qubits in enumerate(frame.zone_placement)
                for qubit in zone_qubits
            }
            involved_qubits.update(frame.highlight_qubits)
            for qubit in frame.highlight_qubits:
                upcoming_gate_zone_by_qubit[qubit] = qubit_to_zone[qubit]
            continue
        frame.upcoming_qubits = list(involved_qubits)
        frame.upcoming_gate_zone_by_qubit = upcoming_gate_zone_by_qubit.copy()
        if frame.kind in ["pswap", "shuttle"]:
            reset = True
    return frames


def _insert_new_target_frames(
    frames: list[MultiZoneCircuitMovieFrame],
) -> list[MultiZoneCircuitMovieFrame]:
    augmented_frames: list[MultiZoneCircuitMovieFrame] = []
    for index, frame in enumerate(frames):
        augmented_frames.append(frame)
        if frame.kind != "gate":
            continue

        next_frame = frames[index + 1] if index + 1 < len(frames) else None
        if next_frame is None or next_frame.kind == "gate":
            continue

        if not next_frame.upcoming_qubits:
            continue

        augmented_frames.append(
            MultiZoneCircuitMovieFrame(
                command_index=None,
                command_text="New target",
                kind="new_target",
                zone_placement=deepcopy(frame.zone_placement),
                highlight_qubits=[],
                upcoming_qubits=next_frame.upcoming_qubits.copy(),
                upcoming_gate_zone_by_qubit=next_frame.upcoming_gate_zone_by_qubit.copy(),
                shuttle=None,
            )
        )
    return augmented_frames


def _condense_quantum_gate_frames(
    frames: list[MultiZoneCircuitMovieFrame],
) -> list[MultiZoneCircuitMovieFrame]:
    condensed_frames: list[MultiZoneCircuitMovieFrame] = []
    gate_block: list[MultiZoneCircuitMovieFrame] = []

    def flush_gate_block() -> None:
        if not gate_block:
            return
        if len(gate_block) == 1:
            condensed_frames.append(
                MultiZoneCircuitMovieFrame(
                    command_index=gate_block[0].command_index,
                    command_text=f"QOPS {_format_qubit_span_text(gate_block[0].highlight_qubits)};",
                    kind="gate",
                    zone_placement=deepcopy(gate_block[0].zone_placement),
                    highlight_qubits=sorted(gate_block[0].highlight_qubits),
                    upcoming_qubits=[],
                    upcoming_gate_zone_by_qubit={},
                    shuttle=None,
                )
            )
            gate_block.clear()
            return
        involved_qubits = sorted(
            {qubit for frame in gate_block for qubit in frame.highlight_qubits}
        )
        condensed_frames.append(
            MultiZoneCircuitMovieFrame(
                command_index=gate_block[0].command_index,
                command_text=f"QOPS {_format_qubit_span_text(involved_qubits)};",
                kind="gate",
                zone_placement=deepcopy(gate_block[-1].zone_placement),
                highlight_qubits=involved_qubits,
                upcoming_qubits=[],
                upcoming_gate_zone_by_qubit={},
                shuttle=None,
            )
        )
        gate_block.clear()

    for frame in frames:
        if frame.kind == "gate":
            gate_block.append(frame)
            continue
        flush_gate_block()
        condensed_frames.append(frame)
    flush_gate_block()
    return condensed_frames


def _format_qubit_span_text(qubits: list[int]) -> str:
    if not qubits:
        return ""
    sorted_qubits = sorted(qubits)
    spans: list[str] = []
    span_start = sorted_qubits[0]
    span_end = sorted_qubits[0]

    for qubit in sorted_qubits[1:]:
        if qubit == span_end + 1:
            span_end = qubit
            continue
        spans.append(
            f"{span_start}-{span_end}" if span_start != span_end else f"{span_start}"
        )
        span_start = qubit
        span_end = qubit

    spans.append(
        f"{span_start}-{span_end}" if span_start != span_end else f"{span_start}"
    )
    return " ".join(spans)


def _format_pswap_command_text(cmd: Any) -> str:
    qubit_0 = cmd.args[0].index[0]
    qubit_1 = cmd.args[1].index[0]
    return f"PSWAP {qubit_0}↔{qubit_1};"


def _format_shuttle_command_text(cmd: Any) -> str:
    source_zone = int(cmd.op.params[0])
    target_zone = int(cmd.op.params[1])
    qubit_text = " ".join(str(arg.index[0]) for arg in cmd.args)
    return f"SHUTTLE({source_zone}→{target_zone}) {qubit_text};"


def generate_multi_zone_circuit_movie_html(
    circuit: MultiZoneCircuit,
    *,
    title: str | None = None,
    frame_duration_ms: float = 300.0,
    highlight_upcoming_qubits: bool = True,
    condense_quantum_ops: bool = True,
) -> str:
    movie = build_multi_zone_circuit_movie(
        circuit,
        title=title,
        highlight_upcoming_qubits=highlight_upcoming_qubits,
        condense_quantum_ops=condense_quantum_ops,
    )
    movie_dict = {
        "title": movie.title,
        "n_qubits": movie.n_qubits,
        "zones": movie.zones,
        "edges": movie.edges,
        "frames": [asdict(frame) for frame in movie.frames],
        "svg_width": _SVG_WIDTH,
        "svg_height": _SVG_HEIGHT,
        "frame_duration_ms": frame_duration_ms,
    }
    movie_json = json.dumps(movie_dict)
    default_frame_duration_ms = json.dumps(frame_duration_ms)
    page_title = escape(movie.title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{
      --bg: #fbf8f2;
      --fg: #1f2933;
      --muted: #5f6c7b;
      --zone-fill: #ececec;
      --zone-stroke: #111111;
      --edge: #b9c1c9;
      --slot: #d0d0d0;
      --highlight: #d62828;
      --control: #f0ece2;
      --control-stroke: #d1c8b8;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top, #fffefb 0%, var(--bg) 70%);
      color: var(--fg);
    }}
    .page {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border: 1px solid var(--control-stroke);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.75);
      backdrop-filter: blur(8px);
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--control-stroke);
      border-radius: 999px;
      background: var(--control);
      color: var(--fg);
      padding: 9px 14px;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{
      background: #faf6ee;
    }}
    input[type="number"] {{
      border: 1px solid var(--control-stroke);
      border-radius: 999px;
      background: var(--control);
      color: var(--fg);
      padding: 9px 14px;
      font: inherit;
    }}
    input[type="range"] {{
      flex: 1 1 320px;
    }}
    .frame-label {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .movie-layout {{
      display: flex;
      gap: 18px;
      align-items: flex-start;
    }}
    .operations-panel {{
      width: 340px;
      border: 1px solid var(--control-stroke);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.72);
      padding: 14px 16px;
      box-sizing: border-box;
    }}
    .operations-list {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .operation-row {{
      min-height: 26px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 20px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border-top: 1px solid transparent;
      border-bottom: 1px solid transparent;
    }}
    .operation-row.movement {{
      color: #2563eb;
    }}
    .operation-row.gate {{
      color: #b91c1c;
    }}
    .operation-row.current {{
      background: rgba(246, 214, 74, 0.45);
      border-top-color: #111827;
      border-bottom-color: #111827;
    }}
    .operation-row.empty {{
      color: transparent;
    }}
    .canvas {{
      flex: 1 1 auto;
      border: 1px solid var(--control-stroke);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.72);
      overflow: hidden;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .edge {{
      stroke: var(--edge);
      stroke-width: 6;
      stroke-linecap: round;
    }}
    .zone-box {{
      stroke-width: 2.2;
      rx: 20;
      ry: 20;
      cursor: grab;
    }}
    .zone-box.dragging {{
      cursor: grabbing;
    }}
    .zone-label {{
      font-size: 16px;
      font-weight: 700;
      text-anchor: middle;
      fill: var(--fg);
      cursor: grab;
    }}
    .zone-rotate-handle {{
      fill: rgba(255, 255, 255, 0.92);
      stroke-width: 1.6;
      cursor: pointer;
    }}
    .zone-rotate-icon {{
      font-size: 11px;
      font-weight: 700;
      text-anchor: middle;
      dominant-baseline: middle;
      fill: var(--muted);
      pointer-events: none;
    }}
    .slot {{
      fill: var(--slot);
      opacity: 0.65;
    }}
    .transport-slot-mark {{
      stroke: rgba(95, 108, 123, 0.45);
      stroke-width: 1.8;
      stroke-linecap: round;
    }}
    .qubit {{
      transition: opacity 0.2s ease;
    }}
    .qubit circle {{
      transition: fill 0.25s ease, stroke 0.25s ease;
    }}
    .qubit text {{
      font-size: 11px;
      font-weight: 700;
      text-anchor: middle;
      dominant-baseline: middle;
      pointer-events: none;
      fill: white;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{page_title}</h1>
    <div class="controls">
      <button id="play-pause" type="button">Play</button>
      <button id="prev" type="button">Prev</button>
      <button id="next" type="button">Next</button>
      <label for="frame-duration" class="frame-label">ms / frame</label>
      <input id="frame-duration" type="number" min="1" step="0.1" value={default_frame_duration_ms}>
      <input id="timeline" type="range" min="0" max="0" value="0">
      <div id="frame-label" class="frame-label"></div>
    </div>
    <div class="movie-layout">
      <div class="operations-panel">
        <div id="operations-list" class="operations-list"></div>
      </div>
      <div class="canvas">
        <svg id="movie" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" aria-label="Multi-zone circuit movie"></svg>
      </div>
    </div>
  </div>
  <script>
    const movieData = {movie_json};
    const svg = document.getElementById("movie");
    const operationsList = document.getElementById("operations-list");
    const timeline = document.getElementById("timeline");
    const playPause = document.getElementById("play-pause");
    const prevButton = document.getElementById("prev");
    const nextButton = document.getElementById("next");
    const frameDurationInput = document.getElementById("frame-duration");
    const frameLabel = document.getElementById("frame-label");
    const ns = "http://www.w3.org/2000/svg";
    const slotGap = 24;
    const slotMargin = 20;
    const qubitRadius = 11;
    const activeQubitRadius = 13;
    const qubitStroke = "rgba(15, 23, 42, 0.32)";
    const activeQubitStroke = "#7f1d1d";
    const currentOperationRow = 10;
    const visibleOperationRows = 16;
    let frameIndex = 0;
    let isPlaying = false;
    let timer = null;
    let dragState = null;
    let shuttleAnimationRequest = null;
    let shuttleAnimationToken = 0;

    timeline.max = String(movieData.frames.length - 1);
    for (let rowIndex = 0; rowIndex < visibleOperationRows; rowIndex += 1) {{
      operationsList.appendChild(document.createElement("div"));
    }}

    function createSvg(tag, attrs) {{
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      return node;
    }}

    function translate(x, y) {{
      return `translate(${{x}} ${{y}})`;
    }}

    function qubitColor(qubit) {{
      return "#111111";
    }}

    function upcomingQubitColor(targetGateZone) {{
      if (targetGateZone === null || targetGateZone === undefined) {{
        return "#eace09";
      }}
      const zone = zoneMap.get(Number(targetGateZone));
      return zone?.gate_qubit_color ?? "#eace09";
    }}

    function pointAlongShuttle(start, sourceAnchor, targetAnchor, end, progress) {{
      if (progress <= 0.28) {{
        const local = progress / 0.28;
        return {{
          x: start.x + ((sourceAnchor.x - start.x) * local),
          y: start.y + ((sourceAnchor.y - start.y) * local),
        }};
      }}
      if (progress <= 0.72) {{
        const local = (progress - 0.28) / 0.44;
        return {{
          x: sourceAnchor.x + ((targetAnchor.x - sourceAnchor.x) * local),
          y: sourceAnchor.y + ((targetAnchor.y - sourceAnchor.y) * local),
        }};
      }}
      const local = (progress - 0.72) / 0.28;
      return {{
        x: targetAnchor.x + ((end.x - targetAnchor.x) * local),
        y: targetAnchor.y + ((end.y - targetAnchor.y) * local),
      }};
    }}

    function edgeAnchor(zone, port) {{
      if (zone.orientation === 1) {{
        return {{
          x: zone.center_x,
          y: port === 0 ? zone.y + zone.height : zone.y,
        }};
      }}
      if (zone.orientation === 2) {{
        return {{
          x: port === 0 ? zone.x + zone.width : zone.x,
          y: zone.center_y,
        }};
      }}
      if (zone.orientation === 3) {{
        return {{
          x: zone.center_x,
          y: port === 0 ? zone.y : zone.y + zone.height,
        }};
      }}
      return {{
        x: port === 0 ? zone.x : zone.x + zone.width,
        y: zone.center_y,
      }};
    }}

    const staticLayer = createSvg("g", {{}});
    const qubitLayer = createSvg("g", {{}});
    svg.appendChild(staticLayer);
    svg.appendChild(qubitLayer);

    const zoneMap = new Map(movieData.zones.map((zone) => [zone.id, zone]));
    const zoneVisuals = new Map();
    const edgeVisuals = [];

    function translateZone(zone, dx, dy) {{
      zone.x += dx;
      zone.y += dy;
      zone.center_x += dx;
      zone.center_y += dy;
      zone.slots.forEach((slot) => {{
        slot.x += dx;
        slot.y += dy;
      }});
    }}

    function layoutZoneSlots(zone) {{
      if (zone.slots.length === 0) {{
        return;
      }}
      if (zone.orientation === 1 || zone.orientation === 3) {{
        const slotX = zone.center_x;
        const usableHeight = zone.height - (2 * slotMargin);
        const gap = Math.min(slotGap, usableHeight / Math.max(zone.slots.length - 1, 1));
        const totalSlotsHeight = gap * Math.max(zone.slots.length - 1, 0);
        const descending = zone.orientation === 1;
        const startY = descending
          ? zone.center_y + (totalSlotsHeight / 2)
          : zone.center_y - (totalSlotsHeight / 2);
        zone.slots.forEach((slot, slotIndex) => {{
          slot.x = slotX;
          slot.y = descending ? startY - (slotIndex * gap) : startY + (slotIndex * gap);
        }});
        return;
      }}
      const slotY = zone.center_y + 2;
      const usableWidth = zone.width - (2 * slotMargin);
      const gap = Math.min(slotGap, usableWidth / Math.max(zone.slots.length - 1, 1));
      const totalSlotsWidth = gap * Math.max(zone.slots.length - 1, 0);
      const leftToRight = zone.orientation === 0;
      const startX = leftToRight
        ? zone.center_x - (totalSlotsWidth / 2)
        : zone.center_x + (totalSlotsWidth / 2);
      zone.slots.forEach((slot, slotIndex) => {{
        slot.x = leftToRight ? startX + (slotIndex * gap) : startX - (slotIndex * gap);
        slot.y = slotY;
      }});
    }}

    function applyZoneOrientation(zone) {{
      const isVertical = zone.orientation === 1 || zone.orientation === 3;
      const width = isVertical ? zone.base_height : zone.base_width;
      const height = isVertical ? zone.base_width : zone.base_height;
      zone.width = width;
      zone.height = height;
      zone.x = zone.center_x - (width / 2);
      zone.y = zone.center_y - (height / 2);
      layoutZoneSlots(zone);
    }}

    function updateEdgePositions() {{
      edgeVisuals.forEach((edgeVisual) => {{
        const source = zoneMap.get(edgeVisual.edge.source);
        const target = zoneMap.get(edgeVisual.edge.target);
        const sourceAnchor = edgeAnchor(source, edgeVisual.edge.source_port);
        const targetAnchor = edgeAnchor(target, edgeVisual.edge.target_port);
        edgeVisual.line.setAttribute("x1", String(sourceAnchor.x));
        edgeVisual.line.setAttribute("y1", String(sourceAnchor.y));
        edgeVisual.line.setAttribute("x2", String(targetAnchor.x));
        edgeVisual.line.setAttribute("y2", String(targetAnchor.y));
      }});
    }}

    function updateZoneGraphics(zoneId) {{
      const zone = zoneMap.get(zoneId);
      const visuals = zoneVisuals.get(zoneId);
      visuals.box.setAttribute("x", String(zone.x));
      visuals.box.setAttribute("y", String(zone.y));
      visuals.box.setAttribute("width", String(zone.width));
      visuals.box.setAttribute("height", String(zone.height));
      visuals.label.setAttribute("x", String(zone.center_x));
      visuals.label.setAttribute("y", String(zone.y - 10));
      visuals.rotateHandle.setAttribute("cx", String(zone.x + zone.width - 12));
      visuals.rotateHandle.setAttribute("cy", String(zone.y + 12));
      visuals.rotateIcon.setAttribute("x", String(zone.x + zone.width - 12));
      visuals.rotateIcon.setAttribute("y", String(zone.y + 13));
      zone.slots.forEach((slot, slotIndex) => {{
        const slotVisual = visuals.slots[slotIndex];
        slotVisual.circle.setAttribute("cx", String(slot.x));
        slotVisual.circle.setAttribute("cy", String(slot.y));
        if (slotVisual.crossLines !== null) {{
          slotVisual.crossLines[0].setAttribute("x1", String(slot.x - 4));
          slotVisual.crossLines[0].setAttribute("y1", String(slot.y));
          slotVisual.crossLines[0].setAttribute("x2", String(slot.x + 4));
          slotVisual.crossLines[0].setAttribute("y2", String(slot.y));
          slotVisual.crossLines[1].setAttribute("x1", String(slot.x));
          slotVisual.crossLines[1].setAttribute("y1", String(slot.y - 4));
          slotVisual.crossLines[1].setAttribute("x2", String(slot.x));
          slotVisual.crossLines[1].setAttribute("y2", String(slot.y + 4));
        }}
      }});
      updateEdgePositions();
      renderFrame(frameIndex, false);
    }}

    function svgPointFromClient(clientX, clientY) {{
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }}

    function startZoneDrag(zoneId, event) {{
      event.preventDefault();
      const point = svgPointFromClient(event.clientX, event.clientY);
      dragState = {{
        zoneId,
        lastX: point.x,
        lastY: point.y,
      }};
      const visuals = zoneVisuals.get(zoneId);
      visuals.box.classList.add("dragging");
    }}

    function handleZoneDrag(event) {{
      if (dragState === null) {{
        return;
      }}
      const point = svgPointFromClient(event.clientX, event.clientY);
      const dx = point.x - dragState.lastX;
      const dy = point.y - dragState.lastY;
      dragState.lastX = point.x;
      dragState.lastY = point.y;
      const zone = zoneMap.get(dragState.zoneId);
      translateZone(zone, dx, dy);
      updateZoneGraphics(dragState.zoneId);
    }}

    function stopZoneDrag() {{
      if (dragState === null) {{
        return;
      }}
      zoneVisuals.get(dragState.zoneId).box.classList.remove("dragging");
      dragState = null;
    }}

    function toggleZoneOrientation(zoneId) {{
      const zone = zoneMap.get(zoneId);
      zone.orientation = (zone.orientation + 1) % 4;
      applyZoneOrientation(zone);
      updateZoneGraphics(zoneId);
    }}

    movieData.edges.forEach((edge) => {{
      const source = zoneMap.get(edge.source);
      const target = zoneMap.get(edge.target);
      const sourceAnchor = edgeAnchor(source, edge.source_port);
      const targetAnchor = edgeAnchor(target, edge.target_port);
      const line = createSvg("line", {{
        x1: sourceAnchor.x,
        y1: sourceAnchor.y,
        x2: targetAnchor.x,
        y2: targetAnchor.y,
        class: "edge",
      }});
      edgeVisuals.push({{ edge, line }});
      staticLayer.appendChild(line);
    }});

    movieData.zones.forEach((zone) => {{
      applyZoneOrientation(zone);
      const box = createSvg("rect", {{
        x: zone.x,
        y: zone.y,
        width: zone.width,
        height: zone.height,
        class: `zone-box${{zone.is_gate_zone ? " gate-zone" : ""}}`,
        fill: zone.is_gate_zone ? zone.gate_color : "var(--zone-fill)",
        stroke: zone.is_gate_zone ? zone.gate_stroke_color : "var(--zone-stroke)",
      }});
      staticLayer.appendChild(box);

      const slotVisuals = [];
      zone.slots.forEach((slot) => {{
        const circle = createSvg("circle", {{
          cx: slot.x,
          cy: slot.y,
          r: 8,
          class: "slot",
        }});
        staticLayer.appendChild(circle);
        let crossLines = null;
        if (slot.transport_only) {{
          const horizontal = createSvg("line", {{
            x1: slot.x - 4,
            y1: slot.y,
            x2: slot.x + 4,
            y2: slot.y,
            class: "transport-slot-mark",
          }});
          const vertical = createSvg("line", {{
            x1: slot.x,
            y1: slot.y - 4,
            x2: slot.x,
            y2: slot.y + 4,
            class: "transport-slot-mark",
          }});
          staticLayer.appendChild(horizontal);
          staticLayer.appendChild(vertical);
          crossLines = [horizontal, vertical];
        }}
        slotVisuals.push({{ circle, crossLines }});
      }});

      const rotateHandle = createSvg("circle", {{
        cx: zone.x + zone.width - 12,
        cy: zone.y + 12,
        r: 8,
        class: `zone-rotate-handle${{zone.is_gate_zone ? " gate-zone" : ""}}`,
        stroke: zone.is_gate_zone ? zone.gate_stroke_color : "var(--zone-stroke)",
      }});
      const rotateIcon = createSvg("text", {{
        x: zone.x + zone.width - 12,
        y: zone.y + 13,
        class: "zone-rotate-icon",
      }});
      rotateIcon.textContent = "↻";
      staticLayer.appendChild(rotateHandle);
      staticLayer.appendChild(rotateIcon);

      const label = createSvg("text", {{
        x: zone.center_x,
        y: zone.y - 10,
        class: "zone-label",
      }});
      label.textContent = `Z${{zone.id}}`;
      staticLayer.appendChild(label);

      zoneVisuals.set(zone.id, {{
        box,
        label,
        rotateHandle,
        rotateIcon,
        slots: slotVisuals,
      }});
      box.addEventListener("pointerdown", (event) => startZoneDrag(zone.id, event));
      label.addEventListener("pointerdown", (event) => startZoneDrag(zone.id, event));
      rotateHandle.addEventListener("pointerdown", (event) => {{
        event.preventDefault();
        event.stopPropagation();
        toggleZoneOrientation(zone.id);
      }});
    }});

    svg.addEventListener("pointermove", handleZoneDrag);
    svg.addEventListener("pointerup", stopZoneDrag);
    svg.addEventListener("pointerleave", stopZoneDrag);

    const qubitElements = new Map();
    for (let qubit = 0; qubit < movieData.n_qubits; qubit += 1) {{
      const group = createSvg("g", {{ class: "qubit", id: `qubit-${{qubit}}` }});
      const circle = createSvg("circle", {{
        cx: 0,
        cy: 0,
        r: qubitRadius,
        fill: qubitColor(qubit),
        stroke: qubitStroke,
        "stroke-width": 2.2,
      }});
      const text = createSvg("text", {{ x: 0, y: 0 }});
      text.textContent = String(qubit);
      group.appendChild(circle);
      group.appendChild(text);
      qubitLayer.appendChild(group);
      qubitElements.set(qubit, {{ group, circle, text }});
    }}

    function qubitTransforms(frame) {{
      const positions = new Map();
      frame.zone_placement.forEach((zoneQubits, zoneId) => {{
        const zone = zoneMap.get(zoneId);
        zoneQubits.forEach((qubit, slotIndex) => {{
          const slot = zone.slots[Math.min(slotIndex, zone.slots.length - 1)];
          positions.set(qubit, {{ x: slot.x, y: slot.y }});
        }});
      }});
      return positions;
    }}

    function setQubitAppearance(qubitElement, qubit, active, upcomingGateZone) {{
      if (active){{
        qubitElement.circle.setAttribute("fill", "#d62828");
      }} else if (upcomingGateZone !== null && upcomingGateZone !== undefined) {{
        qubitElement.circle.setAttribute("fill", upcomingQubitColor(upcomingGateZone));
      }} else {{
        qubitElement.circle.setAttribute("fill", qubitColor(qubit));
      }}
      qubitElement.circle.setAttribute(
        "stroke",
        active ? activeQubitStroke : qubitStroke
      );
      qubitElement.circle.setAttribute(
        "r",
        String(active ? activeQubitRadius : qubitRadius)
      );
    }}

    function stopQubitAnimations() {{
      shuttleAnimationToken += 1;
      if (shuttleAnimationRequest !== null) {{
        cancelAnimationFrame(shuttleAnimationRequest);
        shuttleAnimationRequest = null;
      }}
    }}

    function animateShuttle(frame, previousFrame, positions) {{
      if (previousFrame === null || frame.shuttle === null) {{
        return;
      }}
      const previousPositions = qubitTransforms(previousFrame);
      const sourceZone = zoneMap.get(frame.shuttle.source_zone);
      const targetZone = zoneMap.get(frame.shuttle.target_zone);
      const sourceAnchor = edgeAnchor(sourceZone, frame.shuttle.source_port);
      const targetAnchor = edgeAnchor(targetZone, frame.shuttle.target_port);
      const duration = Math.max(120, currentFrameDuration() * 0.99);
      const dx = targetAnchor.x - sourceAnchor.x;
      const dy = targetAnchor.y - sourceAnchor.y;
      const length = Math.hypot(dx, dy) || 1;
      const directionX = dx / length;
      const directionY = dy / length;
      const chainSpacing = 12;
      const token = shuttleAnimationToken;
      const shuttleQubitPositions = new Map();
      frame.shuttle.qubits.forEach((qubit, shuttleIndex) => {{
        const qubitElement = qubitElements.get(qubit);
        const start = previousPositions.get(qubit);
        const end = positions.get(qubit);
        if (!qubitElement || !start || !end) {{
          return;
        }}
        const centeredIndex = shuttleIndex - ((frame.shuttle.qubits.length - 1) / 2);
        const chainOffsetX = centeredIndex * chainSpacing * directionX;
        const chainOffsetY = centeredIndex * chainSpacing * directionY;
        shuttleQubitPositions.set(qubit, {{
          qubitElement,
          start,
          end,
          sourceAnchor: {{
            x: sourceAnchor.x - chainOffsetX,
            y: sourceAnchor.y - chainOffsetY,
          }},
          targetAnchor: {{
            x: targetAnchor.x + chainOffsetX,
            y: targetAnchor.y + chainOffsetY,
          }},
        }});
      }});

      const startedAt = performance.now();
      const step = (now) => {{
        if (token !== shuttleAnimationToken) {{
          return;
        }}
        const progress = Math.min((now - startedAt) / duration, 1);
        shuttleQubitPositions.forEach((path) => {{
          const point = pointAlongShuttle(
            path.start,
            path.sourceAnchor,
            path.targetAnchor,
            path.end,
            progress
          );
          path.qubitElement.group.setAttribute("transform", translate(point.x, point.y));
        }});
        if (progress < 1) {{
          shuttleAnimationRequest = requestAnimationFrame(step);
        }} else {{
          shuttleAnimationRequest = null;
        }}
      }};
      shuttleAnimationRequest = requestAnimationFrame(step);
    }}

    function animatePswap(frame, previousFrame, positions) {{
      if (previousFrame === null || frame.highlight_qubits.length !== 2) {{
        return;
      }}
      const previousPositions = qubitTransforms(previousFrame);
      const duration = Math.max(120, currentFrameDuration() * 0.99);
      const token = shuttleAnimationToken;
      const swapPaths = frame.highlight_qubits.map((qubit) => {{
        const qubitElement = qubitElements.get(qubit);
        const start = previousPositions.get(qubit);
        const end = positions.get(qubit);
        if (!qubitElement || !start || !end) {{
          return null;
        }}
        const centerX = (start.x + end.x) / 2;
        const centerY = (start.y + end.y) / 2;
        return {{
          qubitElement,
          end,
          centerX,
          centerY,
          relX: start.x - centerX,
          relY: start.y - centerY,
        }};
      }}).filter((path) => path !== null);
      if (swapPaths.length !== 2) {{
        return;
      }}

      const startedAt = performance.now();
      const step = (now) => {{
        if (token !== shuttleAnimationToken) {{
          return;
        }}
        const progress = Math.min((now - startedAt) / duration, 1);
        const angle = progress * Math.PI;
        const cosAngle = Math.cos(angle);
        const sinAngle = Math.sin(angle);
        swapPaths.forEach((path) => {{
          const x = path.centerX + (path.relX * cosAngle) - (path.relY * sinAngle);
          const y = path.centerY + (path.relX * sinAngle) + (path.relY * cosAngle);
          path.qubitElement.group.setAttribute("transform", translate(x, y));
        }});
        if (progress < 1) {{
          shuttleAnimationRequest = requestAnimationFrame(step);
        }} else {{
          swapPaths.forEach((path) => {{
            path.qubitElement.group.setAttribute(
              "transform",
              translate(path.end.x, path.end.y)
            );
          }});
          shuttleAnimationRequest = null;
        }}
      }};
      shuttleAnimationRequest = requestAnimationFrame(step);
    }}

    function renderFrame(index, animate = true) {{
      stopQubitAnimations();
      const frame = movieData.frames[index];
      const previousFrame = index > 0 ? movieData.frames[index - 1] : null;
      const activeQubits = new Set(
        frame.kind === "gate" ? frame.highlight_qubits : []
      );
      const upcomingGateZones = frame.upcoming_gate_zone_by_qubit ?? {{}};
      const positions = qubitTransforms(frame);
      qubitElements.forEach((qubitElement, qubit) => {{
        const pos = positions.get(qubit);
        if (!pos) {{
          qubitElement.group.style.opacity = "0";
          return;
        }}
        qubitElement.group.style.opacity = "1";
        qubitElement.group.setAttribute("transform", translate(pos.x, pos.y));
        setQubitAppearance(
          qubitElement,
          qubit,
          activeQubits.has(qubit),
          upcomingGateZones[String(qubit)] ?? null
        );
      }});
      if (animate && frame.kind === "shuttle") {{
        animateShuttle(frame, previousFrame, positions);
      }} else if (animate && frame.kind === "pswap") {{
        animatePswap(frame, previousFrame, positions);
      }}
      renderOperationStream(index);
      timeline.value = String(index);
      frameLabel.textContent = `Frame ${{index + 1}} / ${{movieData.frames.length}}`;
    }}

    function renderOperationStream(index) {{
      const rows = operationsList.children;
      for (let rowIndex = 0; rowIndex < visibleOperationRows; rowIndex += 1) {{
        const frameOffset = currentOperationRow - rowIndex;
        const frameAtRow = index + frameOffset;
        const row = rows[rowIndex];
        row.className = "operation-row";
        const frameData = movieData.frames[frameAtRow];
        if (frameData && (frameData.kind === "shuttle" || frameData.kind === "pswap")) {{
          row.classList.add("movement");
        }} else if (frameData && frameData.kind === "gate") {{
          row.classList.add("gate");
        }}
        if (rowIndex === currentOperationRow) {{
          row.classList.add("current");
        }}
        if (frameAtRow < 0 || frameAtRow >= movieData.frames.length) {{
          row.textContent = "";
          row.classList.add("empty");
          continue;
        }}
        row.textContent = frameData.command_text;
      }}
    }}

    function stopTimer() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
    }}

    function currentFrameDuration() {{
      const parsedDuration = Number(frameDurationInput.value);
      if (Number.isFinite(parsedDuration) && parsedDuration > 0) {{
        return parsedDuration;
      }}
      return movieData.frame_duration_ms;
    }}

    function startTimer() {{
      stopTimer();
      timer = setInterval(() => {{
        if (frameIndex >= movieData.frames.length - 1) {{
          stopTimer();
          isPlaying = false;
          playPause.textContent = "Restart";
          return;
        }}
        frameIndex += 1;
        renderFrame(frameIndex);
      }}, currentFrameDuration());
    }}

    playPause.addEventListener("click", () => {{
      if (!isPlaying && frameIndex >= movieData.frames.length - 1) {{
        frameIndex = 0;
        renderFrame(frameIndex, false);
      }}
      isPlaying = !isPlaying;
      playPause.textContent = isPlaying ? "Pause" : "Play";
      if (isPlaying) {{
        startTimer();
      }} else {{
        stopTimer();
      }}
    }});

    prevButton.addEventListener("click", () => {{
      frameIndex = (frameIndex - 1 + movieData.frames.length) % movieData.frames.length;
      renderFrame(frameIndex);
    }});

    nextButton.addEventListener("click", () => {{
      frameIndex = (frameIndex + 1) % movieData.frames.length;
      renderFrame(frameIndex);
    }});

    timeline.addEventListener("input", (event) => {{
      frameIndex = Number(event.target.value);
      renderFrame(frameIndex);
    }});

    frameDurationInput.addEventListener("change", () => {{
      if (isPlaying) {{
        startTimer();
      }}
    }});

    renderFrame(frameIndex);
  </script>
</body>
</html>
"""


def write_multi_zone_circuit_movie_html(  # noqa PLR0913
    circuit: MultiZoneCircuit,
    output_path: str | Path,
    *,
    title: str | None = None,
    frame_duration_ms: float = 300.0,
    highlight_upcoming_qubits: bool = True,
    condense_quantum_ops: bool = True,
) -> Path:
    path = Path(output_path)
    path.write_text(
        generate_multi_zone_circuit_movie_html(
            circuit,
            title=title,
            frame_duration_ms=frame_duration_ms,
            highlight_upcoming_qubits=highlight_upcoming_qubits,
            condense_quantum_ops=condense_quantum_ops,
        ),
        encoding="utf-8",
    )
    return path


def _zone_layout(circuit: MultiZoneCircuit) -> list[dict[str, Any]]:
    zone_width, zone_height = _zone_dimensions(circuit)
    if circuit.macro_arch.is_linear_architecture:
        return _wrapped_linear_zone_layout(
            circuit,
            zone_width=zone_width,
            zone_height=zone_height,
        )

    raw_port_positions = _raw_port_positions(circuit)
    raw_positions = _raw_zone_positions(circuit, raw_port_positions)
    xs = [position[0] for position in raw_positions.values()]
    ys = [position[1] for position in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    centers: dict[int, tuple[float, float]] = {}
    for zone in range(circuit.architecture.n_zones):
        x_raw, y_raw = raw_positions[zone]
        center_x = (
            _LAYOUT_MARGIN_X
            + (zone_width / 2)
            + ((x_raw - min_x) / span_x)
            * (_SVG_WIDTH - zone_width - (2 * _LAYOUT_MARGIN_X))
        )
        center_y = (
            _LAYOUT_MARGIN_Y
            + (zone_height / 2)
            + ((y_raw - min_y) / span_y)
            * (_SVG_HEIGHT - zone_height - (2 * _LAYOUT_MARGIN_Y))
        )
        centers[zone] = (center_x, center_y)

    orientations = _best_non_linear_zone_orientations(
        circuit, centers, raw_port_positions
    )
    box_sizes = {
        zone: _oriented_zone_dimensions(zone_width, zone_height, orientations[zone])
        for zone in range(circuit.architecture.n_zones)
    }
    centers = _enforce_minimum_zone_border_distance(
        centers,
        box_sizes,
        min_border_gap=_NON_LINEAR_ZONE_BORDER_GAP,
    )

    zones: list[dict[str, Any]] = []
    for zone in range(circuit.architecture.n_zones):
        center_x, center_y = centers[zone]
        x = center_x - (zone_width / 2)
        y = center_y - (zone_height / 2)
        zone_slots = _slot_centers(
            _SlotLayout(
                x=x,
                y=y,
                width=zone_width,
                height=zone_height,
                gate_capacity=circuit.architecture.get_zone_max_ions_gates(zone),
                transport_capacity=circuit.architecture.get_zone_max_ions_transport(
                    zone
                ),
            )
        )
        zones.append(
            {
                "id": zone,
                "x": round(x, 2),
                "y": round(y, 2),
                "width": zone_width,
                "height": zone_height,
                "base_width": zone_width,
                "base_height": zone_height,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "orientation": orientations[zone],
                "gate_capacity": circuit.architecture.get_zone_max_ions_gates(zone),
                "transport_capacity": circuit.architecture.get_zone_max_ions_transport(
                    zone
                ),
                "is_gate_zone": not circuit.architecture.zones[zone].memory_only,
                "slots": zone_slots,
            }
        )
    return zones


def _best_non_linear_zone_orientations(
    circuit: MultiZoneCircuit,
    centers: dict[int, tuple[float, float]],
    raw_port_positions: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> dict[int, int]:
    return {
        zone: max(
            range(4),
            key=lambda orientation: _score_zone_orientation(
                circuit, zone, centers, orientation, raw_port_positions
            ),
        )
        for zone in range(circuit.architecture.n_zones)
    }


def _score_zone_orientation(
    circuit: MultiZoneCircuit,
    zone: int,
    centers: dict[int, tuple[float, float]],
    orientation: int,
    raw_port_positions: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> float:
    zone_x, zone_y = centers[zone]
    score = 0.0
    for neighbor in circuit.macro_arch.connected_zones(zone):
        neighbor_x, neighbor_y = centers[int(neighbor)]
        dx = neighbor_x - zone_x
        dy = neighbor_y - zone_y
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        port, _ = circuit.macro_arch.get_connected_ports(zone, int(neighbor))
        port_dx, port_dy = _port_direction(orientation, port)
        score += ((dx / length) * port_dx) + ((dy / length) * port_dy)
    if raw_port_positions is not None:
        axis_score = _orientation_axis_alignment(raw_port_positions, zone, orientation)
        score += 0.6 * axis_score
    return score


def _port_direction(orientation: int, port: PortId) -> tuple[float, float]:
    if orientation == _ORIENTATION_VERTICAL_P0_BOTTOM:
        return (0.0, 1.0) if port == PortId.p0 else (0.0, -1.0)
    if orientation == _ORIENTATION_HORIZONTAL_REVERSED:
        return (1.0, 0.0) if port == PortId.p0 else (-1.0, 0.0)
    if orientation == _ORIENTATION_VERTICAL_P0_TOP:
        return (0.0, -1.0) if port == PortId.p0 else (0.0, 1.0)
    return (-1.0, 0.0) if port == PortId.p0 else (1.0, 0.0)


def _orientation_axis_alignment(
    raw_port_positions: dict[tuple[int, int], tuple[float, float]],
    zone: int,
    orientation: int,
) -> float:
    p0_x, p0_y = raw_port_positions[(zone, 0)]
    p1_x, p1_y = raw_port_positions[(zone, 1)]
    dx = p1_x - p0_x
    dy = p1_y - p0_y
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0
    axis_dx, axis_dy = _orientation_axis(orientation)
    return ((dx / length) * axis_dx) + ((dy / length) * axis_dy)


def _orientation_axis(orientation: int) -> tuple[float, float]:
    if orientation == _ORIENTATION_VERTICAL_P0_BOTTOM:
        return (0.0, -1.0)
    if orientation == _ORIENTATION_HORIZONTAL_REVERSED:
        return (-1.0, 0.0)
    if orientation == _ORIENTATION_VERTICAL_P0_TOP:
        return (0.0, 1.0)
    return (1.0, 0.0)


def _oriented_zone_dimensions(
    base_width: float, base_height: float, orientation: int
) -> tuple[float, float]:
    if orientation in (_ORIENTATION_VERTICAL_P0_BOTTOM, _ORIENTATION_VERTICAL_P0_TOP):
        return base_height, base_width
    return base_width, base_height


def _enforce_minimum_zone_border_distance(
    centers: dict[int, tuple[float, float]],
    box_sizes: dict[int, tuple[float, float]],
    min_border_gap: float,
    max_iterations: int = 80,
) -> dict[int, tuple[float, float]]:
    adjusted = {zone: [x, y] for zone, (x, y) in centers.items()}
    for _ in range(max_iterations):
        moved = False
        zones = sorted(adjusted)
        for index, zone_a in enumerate(zones):
            ax, ay = adjusted[zone_a]
            width_a, height_a = box_sizes[zone_a]
            for zone_b in zones[index + 1 :]:
                bx, by = adjusted[zone_b]
                width_b, height_b = box_sizes[zone_b]
                required_dx = (width_a + width_b) / 2 + min_border_gap
                required_dy = (height_a + height_b) / 2 + min_border_gap
                overlap_x = required_dx - abs(bx - ax)
                overlap_y = required_dy - abs(by - ay)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                if overlap_x <= overlap_y:
                    direction = 1.0 if bx >= ax else -1.0
                    if bx == ax:
                        direction = 1.0 if zone_b > zone_a else -1.0
                    shift = overlap_x / 2
                    adjusted[zone_a][0] -= direction * shift
                    adjusted[zone_b][0] += direction * shift
                else:
                    direction = 1.0 if by >= ay else -1.0
                    if by == ay:
                        direction = 1.0 if zone_b > zone_a else -1.0
                    shift = overlap_y / 2
                    adjusted[zone_a][1] -= direction * shift
                    adjusted[zone_b][1] += direction * shift
        if not moved:
            break
    return {zone: (position[0], position[1]) for zone, position in adjusted.items()}


def _wrapped_linear_zone_layout(
    circuit: MultiZoneCircuit,
    *,
    zone_width: float,
    zone_height: float,
) -> list[dict[str, Any]]:
    n_zones = circuit.architecture.n_zones
    available_width = _SVG_WIDTH - (2 * _LAYOUT_MARGIN_X)
    zones_per_row = max(
        1, int((available_width + _ZONE_GAP_X) // (zone_width + _ZONE_GAP_X))
    )
    ordered_zones = sorted(circuit.macro_arch.zone_graph.nodes())
    zones: list[dict[str, Any]] = []
    for zone_index, zone in enumerate(ordered_zones):
        row_index = zone_index // zones_per_row
        col_index = zone_index % zones_per_row
        zones_in_row = min(zones_per_row, n_zones - row_index * zones_per_row)
        row_width = zones_in_row * zone_width + (max(zones_in_row - 1, 0) * _ZONE_GAP_X)
        x_start = (_SVG_WIDTH - row_width) / 2
        x = x_start + col_index * (zone_width + _ZONE_GAP_X)
        y = _LAYOUT_MARGIN_Y + row_index * _ZONE_GAP_Y
        zone_slots = _slot_centers(
            _SlotLayout(
                x=x,
                y=y,
                width=zone_width,
                height=zone_height,
                gate_capacity=circuit.architecture.get_zone_max_ions_gates(int(zone)),
                transport_capacity=circuit.architecture.get_zone_max_ions_transport(
                    int(zone)
                ),
            )
        )
        zones.append(
            {
                "id": int(zone),
                "x": round(x, 2),
                "y": round(y, 2),
                "width": zone_width,
                "height": zone_height,
                "base_width": zone_width,
                "base_height": zone_height,
                "center_x": round(x + (zone_width / 2), 2),
                "center_y": round(y + (zone_height / 2), 2),
                "orientation": 0,
                "gate_capacity": circuit.architecture.get_zone_max_ions_gates(
                    int(zone)
                ),
                "transport_capacity": circuit.architecture.get_zone_max_ions_transport(
                    int(zone)
                ),
                "is_gate_zone": not circuit.architecture.zones[int(zone)].memory_only,
                "slots": zone_slots,
            }
        )
    return zones


def _raw_zone_positions(
    circuit: MultiZoneCircuit,
    raw_port_positions: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> dict[int, tuple[float, float]]:
    if circuit.macro_arch.is_linear_architecture:
        return {
            zone: (float(index), 0.0)
            for index, zone in enumerate(sorted(circuit.macro_arch.zone_graph.nodes()))
        }
    port_positions = (
        _raw_port_positions(circuit)
        if raw_port_positions is None
        else raw_port_positions
    )
    return {
        zone: (
            (port_positions[(zone, 0)][0] + port_positions[(zone, 1)][0]) / 2.0,
            (port_positions[(zone, 0)][1] + port_positions[(zone, 1)][1]) / 2.0,
        )
        for zone in range(circuit.architecture.n_zones)
    }


def _raw_port_positions(
    circuit: MultiZoneCircuit,
) -> dict[tuple[int, int], tuple[float, float]]:
    if circuit.macro_arch.is_linear_architecture:
        return {
            (int(zone), port): (float(index) * 2.0 + float(port), 0.0)
            for index, zone in enumerate(sorted(circuit.macro_arch.zone_graph.nodes()))
            for port in (0, 1)
        }

    port_graph: Graph[int] = Graph()
    for zone in range(circuit.architecture.n_zones):
        port_graph.add_edge(
            zone_port_to_port_id(zone, 0),
            zone_port_to_port_id(zone, 1),
            weight=_PORT_GRAPH_INTERNAL_WEIGHT,
        )
    for connection in circuit.architecture.connections:
        zone0 = connection.zone_port_spec0.zone_id
        port0 = connection.zone_port_spec0.port_id.value
        zone1 = connection.zone_port_spec1.zone_id
        port1 = connection.zone_port_spec1.port_id.value
        port_graph.add_edge(
            zone_port_to_port_id(zone0, port0),
            zone_port_to_port_id(zone1, port1),
            weight=_PORT_GRAPH_CONNECTION_WEIGHT,
        )

    positions = spring_layout(port_graph, seed=11, weight="weight")
    return {
        port_id_to_zone_port(int(port_id)): (float(position[0]), float(position[1]))
        for port_id, position in positions.items()
    }


def _zone_dimensions(circuit: MultiZoneCircuit) -> tuple[float, float]:
    max_transport_capacity = max(
        circuit.architecture.get_zone_max_ions_transport(zone)
        for zone in range(circuit.architecture.n_zones)
    )
    ideal_width = min(
        _ZONE_MAX_WIDTH,
        max(
            _ZONE_MIN_WIDTH,
            (2 * _SLOT_MARGIN_X) + ((max_transport_capacity - 1) * _SLOT_GAP_X) + 24,
        ),
    )
    available_width = _SVG_WIDTH - (2 * _LAYOUT_MARGIN_X)
    max_zones_per_row = max(
        1, int((available_width + _ZONE_GAP_X) // (_ZONE_MIN_WIDTH + _ZONE_GAP_X))
    )
    zones_per_row = min(circuit.architecture.n_zones, max_zones_per_row)
    fitted_width = (
        available_width - ((zones_per_row - 1) * _ZONE_GAP_X)
    ) / zones_per_row
    return min(ideal_width, fitted_width), _ZONE_HEIGHT


def _slot_centers(layout: _SlotLayout) -> list[dict[str, float]]:
    if layout.transport_capacity < 1:
        return []
    slot_y = layout.y + (layout.height / 2) + 2
    usable_width = layout.width - (2 * _SLOT_MARGIN_X)
    slot_gap = min(_SLOT_GAP_X, usable_width / max(layout.transport_capacity - 1, 1))
    total_slots_width = slot_gap * max(layout.transport_capacity - 1, 0)
    x_start = layout.x + (layout.width / 2) - (total_slots_width / 2)
    positions: list[dict[str, float]] = []
    for slot in range(layout.transport_capacity):
        slot_x = x_start + (slot * slot_gap)
        positions.append(
            {
                "x": round(slot_x, 2),
                "y": round(slot_y, 2),
                "transport_only": slot >= layout.gate_capacity,
            }
        )
    return positions


def _apply_pswap(current_placement: list[list[int]], cmd: Any) -> None:
    zone = int(cmd.op.params[0])
    qubit_0 = cmd.args[0].index[0]
    qubit_1 = cmd.args[1].index[0]
    index_0 = current_placement[zone].index(qubit_0)
    index_1 = current_placement[zone].index(qubit_1)
    current_placement[zone][index_0] = qubit_1
    current_placement[zone][index_1] = qubit_0


def _apply_shuttle(
    circuit: MultiZoneCircuit,
    current_placement: list[list[int]],
    cmd: Any,
) -> None:
    qubits = [arg.index[0] for arg in cmd.args]
    source_zone, target_zone, _, _ = (int(param) for param in cmd.op.params)
    connected_ports = circuit.macro_arch.get_connected_ports(source_zone, target_zone)

    if connected_ports[0] == PortId.p0:
        for index in range(len(qubits) - 1, -1, -1):
            current_placement[source_zone].pop(index)
    else:
        for _ in range(len(qubits)):
            current_placement[source_zone].pop()

    ordered_qubits = (
        list(reversed(qubits)) if connected_ports[0] == connected_ports[1] else qubits
    )
    if connected_ports[1] == PortId.p0:
        current_placement[target_zone] = ordered_qubits + current_placement[target_zone]
    else:
        current_placement[target_zone].extend(ordered_qubits)


def _is_quantum_gate(op_type: OpType, op_string: str) -> bool:
    if op_type in _QUANTUM_GATE_TYPES:
        return True
    return (
        "INIT" not in op_string
        and "SHUTTLE" not in op_string
        and "PSWAP" not in op_string
        and "MOVE_BARRIER" not in op_string
    )
