"""Fail-closed Godot replay-to-source safety CI.

The product contract deliberately does not infer a game-specific hazard model.
The game supplies a deterministic replay adapter which writes one numeric risk
observation per action. FlashPatch only issues a PASS after it has replayed the
original project, made exactly one declared source edit in an isolated copy,
and replayed the same trace successfully.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import signal
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .core import analyze
from .godot import GodotRendererReplayRunner, GodotReplayRunner
from .renderer_artifact import (
    RendererArtifactError,
    open_renderer_artifact,
    renderer_rgb_sha256,
    renderer_visual_change_ratio,
)


SCHEMA = "flashpatch-godot-safety-ci-v1"
_EXPORTED_PARAMETER = re.compile(
    r"^(?P<prefix>\s*@export\s+var\s+(?P<name>[A-Za-z_]\w*)\s*(?::[^=]+)?=\s*).+$"
)
_ROOT_NODE = re.compile(r'^\[node\s+name="(?P<name>[^"]+)"')
_GODOT_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_MAIN_SCENE = re.compile(r'^run/main_scene="(?P<path>[^"]+)"$')
_SCRIPT_RESOURCE = re.compile(
    r'^\[ext_resource\b(?=[^\]]*\btype="Script")(?=[^\]]*\bpath="res://(?P<path>[^"]+)")(?=[^\]]*\bid="(?P<id>[^"]+)")[^\]]*\]'
)
_SHADER_RESOURCE = re.compile(
    r'^\[ext_resource\b(?=[^\]]*\btype="Shader")(?=[^\]]*\bpath="(?P<path>[^"]+)")(?=[^\]]*\bid="(?P<id>[^"]+)")[^\]]*\]\s*$'
)
_SHADER_MATERIAL_RESOURCE = re.compile(
    r'^\[sub_resource\b(?=[^\]]*\btype="ShaderMaterial")(?=[^\]]*\bid="(?P<id>[^"]+)")[^\]]*\]\s*$'
)
_SECTION_HEADER = re.compile(r"^\[[A-Za-z_]+(?:\s|\])")
_EXT_RESOURCE_BINDING = re.compile(
    r'^\s*shader\s*=\s*ExtResource\("(?P<id>[^"]+)"\)\s*$'
)
_MATERIAL_BINDING = re.compile(
    r'^\s*material\s*=\s*SubResource\("(?P<id>[^"]+)"\)\s*$'
)
_SHADER_PARAMETER_ASSIGNMENT = re.compile(
    r"^(?P<prefix>[ \t]*shader_parameter/(?P<name>[A-Za-z_]\w*)[ \t]*=[ \t]*)"
    r"(?P<value>.*?)(?P<suffix>[ \t]*)$"
)
_NUMERIC_LITERAL = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)
_UNIFORM_DECLARATION = re.compile(
    r"^\s*uniform\s+(?P<type>[A-Za-z_]\w*)\s+(?P<name>[A-Za-z_]\w*)(?P<rest>.*?)\s*;\s*$"
)
_EXACT_FLOAT_UNIFORM = re.compile(
    r"^\s*uniform\s+float\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*=\s*[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)?\s*;\s*$"
)
_NODE_NAME = re.compile(r'\bname="(?P<name>[^"]+)"')
_NODE_PARENT = re.compile(r'\bparent="(?P<parent>[^"]+)"')
_VISUAL_PARAMETER_KINDS = frozenset({
    "affected_area", "brightness", "contrast", "duration", "duty_cycle",
    "frequency", "intensity", "interval", "modulation_amplitude", "red_channel_intensity",
})
_RENDERER_ANALYSIS_CACHE: dict[
    tuple[str, str, str], dict[str, object]
] = {}
_RENDERER_ANALYSIS_CACHE_MAX_ENTRIES = 8
RENDERER_ANALYSIS_TIMEOUT_SECONDS = 300.0


class ContractError(ValueError):
    """The project cannot support a trustworthy FlashPatch conclusion."""


@dataclass(frozen=True)
class PatchCandidate:
    source: Path
    parameter: str
    replacement: object
    parameter_kind: str | None = None
    runtime_binding: str = "scene"
    runtime_resource: Path | None = None
    source_kind: str = "gdscript_export"


@dataclass(frozen=True)
class SafetyContract:
    contract_path: Path
    project: Path
    trace: Path
    scene: Path
    signal_kind: str
    signal_field: str
    timing_field: str
    state_field: str
    threshold: float
    candidates: tuple[PatchCandidate, ...]


@dataclass(frozen=True)
class RiskMeasurement:
    maximum: float
    timing_key: str
    state_key: str
    details: dict[str, object]


def _analyze_with_deadline(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    timeout_seconds: float,
) -> object:
    """Bound the synchronous detector without silently discarding its result."""

    if timeout_seconds <= 0:
        raise ContractError("renderer detector deadline must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise ContractError(
            "renderer detector deadline is unavailable outside the main thread"
        )
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise ContractError(
            "renderer detector deadline is unavailable on this execution platform"
        )
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise ContractError(
            "renderer detector deadline cannot replace an active process deadline"
        )
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_exceeded(signum: int, frame: object) -> None:
        del signum, frame
        raise ContractError(
            f"renderer detector exceeded {timeout_seconds:g}-second deadline"
        )

    signal.signal(signal.SIGALRM, deadline_exceeded)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return analyze(frames, timestamps)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _renderer_analysis_summary(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    renderer_rgb_raw_sha256: str,
    timestamps_sha256: str,
    timeout_seconds: float = RENDERER_ANALYSIS_TIMEOUT_SECONDS,
) -> tuple[dict[str, object], bool]:
    """Analyze every frame once per exact RGB/timestamp stream in this process."""

    key = (
        "flashpatch-core-analyze-defaults-v1",
        renderer_rgb_raw_sha256,
        timestamps_sha256,
    )
    cached = _RENDERER_ANALYSIS_CACHE.get(key)
    if cached is not None:
        return dict(cached), True
    result = _analyze_with_deadline(
        frames,
        timestamps,
        timeout_seconds=timeout_seconds,
    )
    summary: dict[str, object] = {
        "max_flash_count": result.max_flash_count,
        "max_affected_fraction": result.max_affected_fraction,
        "hazardous": result.hazardous,
        "hazard_kinds": sorted({window.kind for window in result.windows}),
        "hazard_frame_indices": np.flatnonzero(
            np.any(result.hazard_mask, axis=(1, 2))
        ).tolist(),
    }
    if len(_RENDERER_ANALYSIS_CACHE) >= _RENDERER_ANALYSIS_CACHE_MAX_ENTRIES:
        _RENDERER_ANALYSIS_CACHE.pop(next(iter(_RENDERER_ANALYSIS_CACHE)))
    _RENDERER_ANALYSIS_CACHE[key] = dict(summary)
    return summary, False


def _semantic_invariants(replay: dict[str, object]) -> dict[str, object]:
    raw = replay.get("semantic_invariants")
    if not isinstance(raw, dict):
        raise ContractError("replay must provide semantic_invariants for renderer evidence")
    required = ("terminal_completion", "terminal_state", "player_world_digest", "score")
    if any(key not in raw for key in required):
        raise ContractError("semantic_invariants omits a required renderer-preservation field")
    if raw["terminal_completion"] is not True:
        raise ContractError("semantic_invariants.terminal_completion must be true")
    if not isinstance(raw["terminal_state"], str) or not raw["terminal_state"]:
        raise ContractError("semantic_invariants.terminal_state must be a non-empty string")
    if not isinstance(raw["player_world_digest"], str) or not raw["player_world_digest"]:
        raise ContractError("semantic_invariants.player_world_digest must be a non-empty string")
    if raw["score"] != "score_not_applicable" and not isinstance(raw["score"], (int, float, str)):
        raise ContractError("semantic_invariants.score must be scalar or score_not_applicable")
    return dict(raw)


def _runtime_attribution(
    replay: dict[str, object],
    measurement: RiskMeasurement,
    candidate: PatchCandidate,
    project: Path,
    expected_node: str | None,
    expected_resource: str,
    expected_source_line: int,
) -> dict[str, object]:
    raw = replay.get("runtime_events")
    if not isinstance(raw, list) or not raw:
        raise ContractError("renderer replay must provide non-empty runtime_events")
    hazard_frames = measurement.details.get("hazard_frame_indices")
    if not isinstance(hazard_frames, list) or not hazard_frames:
        raise ContractError("renderer risk must identify hazard frame indices for attribution")
    # ``analyze`` reports a transition at the index of its source image.  A
    # runtime render-property event can therefore join either endpoint of the
    # observed transition.  This is a fixed, one-frame temporal alignment, not
    # a tolerance window: it prevents a later unrelated event being promoted.
    hazard_event_frames = {
        frame_index + offset
        for frame_index in hazard_frames
        if isinstance(frame_index, int) and not isinstance(frame_index, bool)
        for offset in (0, 1)
    }
    expected_script = candidate.source.relative_to(project).as_posix()
    matched: list[dict[str, object]] = []
    for event in raw:
        if not isinstance(event, dict):
            continue
        if (
            event.get("frame_index") in hazard_event_frames
            and event.get("property") == candidate.parameter
            and event.get("script_path") == f"res://{expected_script}"
            and event.get("resource_path") == expected_resource
            and event.get("event_kind") == "render_property"
            and isinstance(event.get("node_path"), str)
            and (
                event["node_path"] == expected_node
                if expected_node is not None
                else event["node_path"].startswith("/root/") and event["node_path"] != "/root"
            )
            and isinstance(event.get("source_line"), int)
            and event["source_line"] == expected_source_line
            and isinstance(event.get("factual_value"), (int, float))
            and not isinstance(event.get("factual_value"), bool)
        ):
            matched.append(event)
    if not matched:
        candidate_events = [
            event for event in raw
            if isinstance(event, dict)
            and event.get("property") == candidate.parameter
            and event.get("script_path") == f"res://{expected_script}"
            and event.get("resource_path") == expected_resource
            and event.get("event_kind") == "render_property"
            and isinstance(event.get("node_path"), str)
            and (
                event["node_path"] == expected_node
                if expected_node is not None
                else event["node_path"].startswith("/root/") and event["node_path"] != "/root"
            )
            and event.get("source_line") == expected_source_line
        ]
        observed_frames = sorted({event.get("frame_index") for event in candidate_events if isinstance(event.get("frame_index"), int)})
        raise ContractError(
            "no observed runtime contributor joined a hazard frame "
            f"(hazard_event_frames={sorted(hazard_event_frames)}, observed_candidate_frames={observed_frames})"
        )
    identities = {(event["node_path"], event["script_path"], event["source_line"], event["property"]) for event in matched}
    if len(identities) != 1:
        raise ContractError("multiple runtime contributors tie for the declared source parameter")
    factual_values = {float(event["factual_value"]) for event in matched}
    if len(factual_values) != 1:
        raise ContractError("runtime contributor reports inconsistent factual property values")
    node_path, script_path, source_line, parameter = next(iter(identities))
    return {
        "node": node_path,
        "script_path": script_path,
        "source_line": source_line,
        "parameter": parameter,
        "hazard_event_count": len(matched),
        "factual_value": factual_values.pop(),
    }


def _candidate_runtime_application(
    replay: dict[str, object],
    candidate: PatchCandidate,
    project: Path,
    factual_runtime: dict[str, object],
    expected_resource: str,
) -> int:
    """Require the counterfactual replay to report the patched runtime value."""
    raw = replay.get("runtime_events")
    if not isinstance(raw, list) or not raw:
        raise ContractError("candidate renderer replay must provide runtime_events")
    expected_script = f"res://{candidate.source.relative_to(project).as_posix()}"
    expected_node = factual_runtime.get("node")
    expected_line = factual_runtime.get("source_line")
    replacement = float(candidate.replacement)
    matched = [
        event
        for event in raw
        if isinstance(event, dict)
        and event.get("node_path") == expected_node
        and event.get("script_path") == expected_script
        and event.get("resource_path") == expected_resource
        and event.get("source_line") == expected_line
        and event.get("property") == candidate.parameter
        and event.get("event_kind") == "render_property"
    ]
    if not matched:
        raise ContractError("candidate replay omitted the factual runtime contributor")
    values = [event.get("factual_value") for event in matched]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) != replacement
        for value in values
    ):
        raise ContractError("candidate runtime contributor does not report the replacement value")
    return len(matched)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty path string")
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_contract(project: Path | str, trace_or_contract: Path | str) -> SafetyContract:
    project_path = Path(project).resolve()
    supplied = Path(trace_or_contract).resolve()
    is_contract = False
    if supplied.is_file():
        try:
            probe = json.loads(supplied.read_text(encoding="utf-8"))
            is_contract = isinstance(probe, dict) and probe.get("schema") == SCHEMA
        except json.JSONDecodeError:
            is_contract = False
    contract_path = supplied if is_contract else project_path / "flashpatch.json"
    if not is_contract and not contract_path.is_file():
        raise ContractError(
            "a direct trace requires project/flashpatch.json; pass a FlashPatch contract instead"
        )
    if not project_path.is_dir() or not (project_path / "project.godot").is_file():
        raise ContractError("project must be a Godot project directory containing project.godot")
    if not contract_path.is_file():
        raise ContractError(f"contract file is missing: {contract_path}")
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"contract is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ContractError(f"contract schema must be {SCHEMA!r}")
    base = contract_path.parent
    declared_project = raw.get("project")
    if declared_project is not None and _resolve(base, declared_project, "project") != project_path:
        raise ContractError("contract project does not match the CLI project argument")
    trace = _resolve(base, raw.get("trace"), "trace")
    if not is_contract and supplied != trace:
        raise ContractError("direct trace does not match the trace declared in project/flashpatch.json")
    scene = _resolve(project_path, raw.get("scene"), "scene")
    if not trace.is_file():
        raise ContractError(f"trace file is missing: {trace}")
    if not scene.is_file() or project_path not in scene.parents:
        raise ContractError("scene must be an existing file inside the project")
    signal = raw.get("risk_signal")
    if not isinstance(signal, dict):
        raise ContractError("risk_signal must declare a replay output field and threshold")
    kind = signal.get("kind")
    field = signal.get("field")
    threshold = signal.get("threshold")
    timing_field = raw.get("timing_field")
    state_field = raw.get("state_field")
    if kind not in {"replay_observations_v1", "frame_npz_v1"}:
        raise ContractError("risk_signal.kind must be replay_observations_v1 or frame_npz_v1")
    if not isinstance(field, str) or not field:
        raise ContractError("risk_signal.field must be a non-empty string")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ContractError("risk_signal.threshold must be a finite number")
    if float(threshold) <= 0:
        raise ContractError("risk_signal.threshold must be greater than zero")
    if not isinstance(timing_field, str) or not timing_field:
        raise ContractError("timing_field must name replay output action-frame sequence")
    if not isinstance(state_field, str) or not state_field:
        raise ContractError("state_field must name deterministic gameplay-state fingerprint")
    raw_candidates = raw.get("patch_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ContractError("patch_candidates must contain at least one declared source edit")
    candidates: list[PatchCandidate] = []
    seen: set[tuple[Path, str, str]] = set()
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise ContractError(f"patch_candidates[{index}] must be an object")
        source = _resolve(project_path, item.get("source"), f"patch_candidates[{index}].source")
        parameter = item.get("parameter")
        if not source.is_file() or project_path not in source.parents:
            raise ContractError(f"patch_candidates[{index}].source must be a file inside the project")
        if not isinstance(parameter, str) or not _GODOT_IDENTIFIER.fullmatch(parameter):
            raise ContractError(f"patch_candidates[{index}].parameter must be a Godot identifier")
        if "replacement" not in item:
            raise ContractError(f"patch_candidates[{index}].replacement is required")
        replacement = item["replacement"]
        if isinstance(replacement, bool):
            pass
        elif isinstance(replacement, (int, float)) and not isinstance(replacement, bool) and math.isfinite(float(replacement)):
            pass
        else:
            raise ContractError(f"patch_candidates[{index}].replacement must be a finite scalar or boolean")
        parameter_kind = item.get("parameter_kind")
        if kind == "frame_npz_v1":
            if isinstance(replacement, bool):
                raise ContractError(f"patch_candidates[{index}].replacement must be a numeric scalar for renderer evidence")
            if not isinstance(parameter_kind, str) or parameter_kind not in _VISUAL_PARAMETER_KINDS:
                supported = ", ".join(sorted(_VISUAL_PARAMETER_KINDS))
                raise ContractError(f"patch_candidates[{index}].parameter_kind must be one of: {supported}")
        elif parameter_kind is not None and (
            not isinstance(parameter_kind, str) or parameter_kind not in _VISUAL_PARAMETER_KINDS
        ):
            raise ContractError(f"patch_candidates[{index}].parameter_kind is not a supported visual parameter kind")
        source_kind = item.get("source_kind", "gdscript_export")
        if source_kind not in {"gdscript_export", "tscn_shader_parameter"}:
            raise ContractError(
                f"patch_candidates[{index}].source_kind must be gdscript_export or tscn_shader_parameter"
            )
        if source_kind == "tscn_shader_parameter":
            if isinstance(replacement, bool) or not isinstance(replacement, (int, float)):
                raise ContractError(
                    f"patch_candidates[{index}].replacement must be numeric for tscn_shader_parameter"
                )
            _validate_tscn_shader_parameter_binding(
                source, parameter, project_path, scene
            )
        runtime_binding = item.get("runtime_binding", "scene")
        if runtime_binding not in {"scene", "dynamic"}:
            raise ContractError(f"patch_candidates[{index}].runtime_binding must be scene or dynamic")
        runtime_resource: Path | None = None
        if runtime_binding == "dynamic":
            runtime_resource = _resolve(project_path, item.get("runtime_resource"), f"patch_candidates[{index}].runtime_resource")
            if not runtime_resource.is_file() or project_path not in runtime_resource.parents:
                raise ContractError(f"patch_candidates[{index}].runtime_resource must be a file inside the project")
        elif item.get("runtime_resource") is not None:
            raise ContractError(f"patch_candidates[{index}].runtime_resource is only valid for dynamic binding")
        identity = (source, parameter, json.dumps(replacement, sort_keys=True))
        if identity in seen:
            raise ContractError(f"duplicate patch candidate: {source.name}:{parameter}:{replacement}")
        seen.add(identity)
        candidates.append(
            PatchCandidate(
                source,
                parameter,
                replacement,
                parameter_kind,
                runtime_binding,
                runtime_resource,
                source_kind,
            )
        )
    _require_main_scene(project_path, scene)
    _validate_trace(trace)
    return SafetyContract(contract_path, project_path, trace, scene, kind, field, timing_field, state_field, float(threshold), tuple(candidates))


def _validate_trace(path: Path) -> None:
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"trace is not valid JSON: {exc.msg}") from exc
    if not isinstance(trace, dict) or isinstance(trace.get("fixed_fps"), bool) or not isinstance(trace.get("fixed_fps"), int) or trace["fixed_fps"] <= 0:
        raise ContractError("trace must contain positive integer fixed_fps")
    actions = trace.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ContractError("trace must contain at least one action")
    frames: list[int] = []
    for action in actions:
        if not isinstance(action, dict) or isinstance(action.get("frame"), bool) or not isinstance(action.get("frame"), int) or action["frame"] < 0:
            raise ContractError("every trace action must contain non-negative integer frame")
        frames.append(action["frame"])
    if frames != sorted(set(frames)):
        raise ContractError("trace action frames must be unique and ascending")


def _observations(replay: dict[str, object], field: str, expected_count: int) -> list[float]:
    raw = replay.get(field)
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ContractError(f"replay output field {field!r} must be a list matching trace length")
    values: list[float] = []
    for value in raw:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError(f"replay output field {field!r} must contain finite numbers")
        values.append(float(value))
    return values


def _measure_risk(
    replay: dict[str, object],
    replay_path: Path,
    contract: SafetyContract,
    expected_actions: int,
) -> RiskMeasurement:
    expected_frames = [action["frame"] for action in json.loads(contract.trace.read_text(encoding="utf-8"))["actions"]]
    timeline = replay.get(contract.timing_field)
    if timeline != expected_frames:
        raise ContractError("replay action-frame sequence does not exactly match declared trace")
    timing_key = f"sha256:{hashlib.sha256(json.dumps(timeline, separators=(',', ':')).encode()).hexdigest()}"
    state = replay.get(contract.state_field)
    if not isinstance(state, str) or not state:
        raise ContractError("replay gameplay-state fingerprint must be a non-empty string")
    state_key = f"sha256:{hashlib.sha256(state.encode('utf-8')).hexdigest()}"
    if contract.signal_kind == "replay_observations_v1":
        values = _observations(replay, contract.signal_field, expected_actions)
        return RiskMeasurement(
            max(values, default=0.0),
            timing_key,
            state_key,
            {"observation_count": len(values), "gameplay_state_sha256": state_key},
        )
    raw_path = replay.get(contract.signal_field)
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"replay output field {contract.signal_field!r} must name an NPZ frame artifact")
    frame_reference = Path(raw_path)
    if frame_reference.is_absolute() or ".." in frame_reference.parts:
        raise ContractError("frame artifact path must be relative to its replay output directory")
    frame_path = (replay_path.parent / frame_reference).resolve()
    if replay_path.parent.resolve() not in frame_path.parents:
        raise ContractError("frame artifact must stay inside its replay output directory")
    if not frame_path.is_file():
        raise ContractError(f"frame artifact is missing: {frame_path}")
    capture = replay.get("renderer_capture")
    if not isinstance(capture, dict):
        raise ContractError("renderer replay must provide renderer_capture metadata")
    if capture.get("trace_sha256") != _sha256(contract.trace):
        raise ContractError("renderer_capture trace hash does not match declared trace")
    if not isinstance(capture.get("godot_version"), str) or not capture["godot_version"]:
        raise ContractError("renderer_capture must identify the Godot version")
    if capture.get("renderer_configuration") != {"display_driver": "x11", "rendering_driver": "opengl3"}:
        raise ContractError("renderer_capture configuration is not the required X11/OpenGL renderer")
    try:
        with open_renderer_artifact(frame_path) as artifact:
            frames = artifact.frames
            timestamps = artifact.timestamps
            renderer_rgb_raw_sha256 = renderer_rgb_sha256(frames)
            timestamps_sha256 = hashlib.sha256(
                timestamps.astype("<f8", copy=False).tobytes()
            ).hexdigest()
            analysis_summary, detector_cache_hit = _renderer_analysis_summary(
                frames,
                timestamps,
                renderer_rgb_raw_sha256=renderer_rgb_raw_sha256,
                timestamps_sha256=timestamps_sha256,
            )
            frame_count = len(frames)
    except RendererArtifactError as exc:
        raise ContractError(
            f"frame artifact must contain valid frames and timestamps: {exc}"
        ) from exc
    # ``max_flash_count`` is intentionally zero for a regular-pattern
    # violation: it counts temporal flashes, not stripe-pair violations.  A
    # renderer safety decision must nevertheless retain a non-zero residual
    # score for every detected hazard kind, otherwise a pattern can pass a
    # scalar threshold as a false SAFE.
    hazardous = bool(analysis_summary["hazardous"])
    residual_risk = (
        max(float(analysis_summary["max_flash_count"]), 1.0)
        if hazardous
        else 0.0
    )
    return RiskMeasurement(
        residual_risk,
        timing_key,
        state_key,
        {
            "frame_artifact": str(frame_path),
            "frame_artifact_sha256": _sha256(frame_path),
            "renderer_rgb_raw_sha256": renderer_rgb_raw_sha256,
            "timestamps_sha256": timestamps_sha256,
            "frame_count": frame_count,
            "max_affected_fraction": analysis_summary["max_affected_fraction"],
            "hazardous": hazardous,
            "hazard_kinds": analysis_summary["hazard_kinds"],
            "hazard_frame_indices": analysis_summary["hazard_frame_indices"],
            "detector_cache_hit": detector_cache_hit,
            "gameplay_state_sha256": state_key,
            "renderer_capture": capture,
            "runtime_script_sha256": replay.get("runtime_script_sha256"),
            "runtime_source_line": replay.get("runtime_source_line"),
            "state_stream_sha256": replay.get("state_stream_sha256"),
            "state_stream_artifact": replay.get("state_stream_artifact"),
            "state_stream_tick_domain": replay.get("state_stream_tick_domain"),
            "state_stream_record_count": replay.get("state_stream_record_count"),
            "final_state_sha256": replay.get("final_state_sha256"),
            "final_state_raw_sha256": replay.get("final_state_raw_sha256"),
            "final_state_artifact": replay.get("final_state_artifact"),
            "tick_domain": replay.get("tick_domain"),
            "action_acknowledgements": replay.get("action_acknowledgements"),
            "action_acknowledgement_evidence": replay.get(
                "action_acknowledgement_evidence"
            ),
        },
    )


def _visual_change_ratio(factual: Path, candidate: Path) -> float:
    try:
        with open_renderer_artifact(factual) as factual_artifact:
            with open_renderer_artifact(candidate) as candidate_artifact:
                return renderer_visual_change_ratio(
                    factual_artifact.frames,
                    candidate_artifact.frames,
                )
    except RendererArtifactError as exc:
        raise ContractError(f"renderer frame artifacts cannot be compared: {exc}") from exc


def _root_node(scene: Path) -> str:
    for line in scene.read_text(encoding="utf-8").splitlines():
        matched = _ROOT_NODE.match(line)
        if matched:
            return f"/root/{matched.group('name')}"
    raise ContractError("scene has no readable root node")


def _require_main_scene(project: Path, scene: Path) -> None:
    for line in (project / "project.godot").read_text(encoding="utf-8").splitlines():
        match = _MAIN_SCENE.match(line)
        if match:
            raw_path = match.group("path")
            relative = raw_path.removeprefix("res://")
            candidate = (project / relative).resolve()
            if project.resolve() not in candidate.parents:
                raise ContractError("project.godot run/main_scene must stay inside project")
            main_scene = candidate
            if main_scene != scene:
                raise ContractError("contract scene must equal project.godot run/main_scene")
            return
    raise ContractError("project.godot must declare run/main_scene")


def _scene_binds_source(scene: Path, source: Path, project: Path) -> str:
    relative = source.relative_to(project).as_posix()
    lines = scene.read_text(encoding="utf-8").splitlines()
    resource_id = next(
        (match.group("id") for line in lines if (match := _SCRIPT_RESOURCE.match(line)) and match.group("path") == relative),
        None,
    )
    if resource_id is None:
        raise ContractError(
            f"scene {scene.name} does not bind declared source {relative}; attribution is inconclusive"
        )
    runtime_node: str | None = None
    for line in lines:
        if line.startswith("[node "):
            name = _NODE_NAME.search(line)
            parent = _NODE_PARENT.search(line)
            if name is None:
                runtime_node = None
                continue
            if parent is None:
                runtime_node = f"/root/{name.group('name')}"
            else:
                parent_path = parent.group("parent")
                if parent_path == ".":
                    runtime_node = f"/root/{_root_node(scene).split('/')[-1]}/{name.group('name')}"
                else:
                    runtime_node = f"/root/{_root_node(scene).split('/')[-1]}/{parent_path}/{name.group('name')}"
        if runtime_node is not None and line.strip() == f'script = ExtResource("{resource_id}")':
            return runtime_node
    raise ContractError(
        f"scene {scene.name} does not attach declared source {relative} to a runtime node; attribution is inconclusive"
    )


def _line_body(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _tscn_sections(lines: list[str]) -> list[tuple[int, int, str]]:
    starts = [
        index
        for index, line in enumerate(lines)
        if _SECTION_HEADER.match(_line_body(line)[0])
    ]
    return [
        (start, starts[position + 1] if position + 1 < len(starts) else len(lines), _line_body(lines[start])[0])
        for position, start in enumerate(starts)
    ]


def _validate_tscn_shader_parameter_binding(
    source: Path,
    parameter: str,
    project: Path,
    scene: Path,
) -> None:
    """Validate one conservative, same-scene ShaderMaterial parameter binding."""

    if source != scene or source.suffix != ".tscn":
        raise ContractError(
            "tscn_shader_parameter source must be the declared same-scene .tscn file"
        )
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    sections = _tscn_sections(lines)
    assignments: list[tuple[int, re.Match[str]]] = []
    for line_number, line in enumerate(lines, start=1):
        match = _SHADER_PARAMETER_ASSIGNMENT.fullmatch(_line_body(line)[0])
        if match is None or match.group("name") != parameter:
            continue
        if _NUMERIC_LITERAL.fullmatch(match.group("value")) is None:
            raise ContractError(
                f"shader_parameter/{parameter} assignment must contain a numeric literal only"
            )
        assignments.append((line_number, match))
    if len(assignments) != 1:
        raise ContractError(
            f"candidate {source.name}:{parameter} must match exactly one shader_parameter assignment"
        )
    assignment_index = assignments[0][0] - 1
    containing = [
        section for section in sections if section[0] < assignment_index < section[1]
    ]
    if len(containing) != 1:
        raise ContractError("shader_parameter assignment must belong to one ShaderMaterial subresource")
    start, end, header = containing[0]
    material_match = _SHADER_MATERIAL_RESOURCE.fullmatch(header)
    if material_match is None:
        raise ContractError("shader_parameter assignment must belong to a ShaderMaterial subresource")
    material_id = material_match.group("id")
    matching_material_sections = [
        candidate
        for _, _, candidate in sections
        if (match := _SHADER_MATERIAL_RESOURCE.fullmatch(candidate)) is not None
        and match.group("id") == material_id
    ]
    if len(matching_material_sections) != 1:
        raise ContractError("ShaderMaterial subresource ID must identify exactly one subresource")
    shader_bindings = [
        match.group("id")
        for line in lines[start + 1 : end]
        if (match := _EXT_RESOURCE_BINDING.fullmatch(_line_body(line)[0])) is not None
    ]
    if len(shader_bindings) != 1:
        raise ContractError("ShaderMaterial must bind exactly one Shader ext_resource")
    shader_id = shader_bindings[0]
    shader_resources = [
        match
        for line in lines
        if (match := _SHADER_RESOURCE.fullmatch(_line_body(line)[0])) is not None
        and match.group("id") == shader_id
    ]
    if len(shader_resources) != 1:
        raise ContractError("ShaderMaterial must bind one exact project-local Shader ext_resource")
    shader_reference = shader_resources[0].group("path")
    if not shader_reference.startswith("res://") or not shader_reference.endswith(".gdshader"):
        raise ContractError("Shader ext_resource must be a project-local .gdshader path")
    shader_path = (project / shader_reference.removeprefix("res://")).resolve()
    if project.resolve() not in shader_path.parents or not shader_path.is_file():
        raise ContractError("Shader ext_resource must resolve to a project-local .gdshader file")
    material_nodes = [
        section_start
        for section_start, section_end, section_header in sections
        if section_header.startswith("[node ")
        for line in lines[section_start + 1 : section_end]
        if (match := _MATERIAL_BINDING.fullmatch(_line_body(line)[0])) is not None
        and match.group("id") == material_id
    ]
    if len(material_nodes) != 1:
        raise ContractError(
            "ShaderMaterial subresource must bind material on exactly one node"
        )
    declarations = [
        (_UNIFORM_DECLARATION.fullmatch(line), _EXACT_FLOAT_UNIFORM.fullmatch(line))
        for line in shader_path.read_text(encoding="utf-8").splitlines()
        if (declaration := _UNIFORM_DECLARATION.fullmatch(line)) is not None
        and declaration.group("name") == parameter
    ]
    if (
        len(declarations) != 1
        or declarations[0][1] is None
        or declarations[0][1].group("name") != parameter
    ):
        raise ContractError(
            f"shader must contain exactly one exact uniform float declaration for {parameter}"
        )


def _patch_tscn_shader_parameter(
    source: Path, parameter: str, replacement: object
) -> tuple[str, int]:
    """Replace only one numeric shader_parameter token, preserving all other bytes."""

    if (
        isinstance(replacement, bool)
        or not isinstance(replacement, (int, float))
        or not math.isfinite(float(replacement))
    ):
        raise ContractError("tscn shader parameter replacement must be a finite numeric scalar")
    original_bytes = source.read_bytes()
    try:
        original_text = original_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("tscn shader parameter source must be UTF-8") from exc
    original = original_text.splitlines(keepends=True)
    matches: list[tuple[int, re.Match[str], str, str]] = []
    for index, line in enumerate(original):
        body, ending = _line_body(line)
        match = _SHADER_PARAMETER_ASSIGNMENT.fullmatch(body)
        if match is None or match.group("name") != parameter:
            continue
        if _NUMERIC_LITERAL.fullmatch(match.group("value")) is None:
            raise ContractError(
                f"shader_parameter/{parameter} assignment must contain a numeric literal only"
            )
        matches.append((index, match, body, ending))
    if len(matches) != 1:
        raise ContractError(
            f"candidate {source.name}:{parameter} must match exactly one shader_parameter assignment"
        )
    index, match, body, ending = matches[0]
    replacement_text = str(replacement)
    updated = list(original)
    updated[index] = (
        body[: match.start("value")]
        + replacement_text
        + body[match.end("value") :]
        + ending
    )
    updated_bytes = "".join(updated).encode("utf-8")
    source.write_bytes(updated_bytes)
    diff = "".join(
        difflib.unified_diff(
            original,
            updated,
            fromfile=f"a/{source.name}",
            tofile=f"b/{source.name}",
        )
    )
    return diff, index + 1


def _patch_export(source: Path, parameter: str, replacement: object) -> tuple[str, int]:
    original = source.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = list(original)
    matched_at: list[int] = []
    replacement_text = str(replacement).lower() if isinstance(replacement, bool) else str(replacement)
    for index, line in enumerate(updated):
        matched = _EXPORTED_PARAMETER.match(line)
        if matched and matched.group("name") == parameter:
            updated[index] = matched.group("prefix") + replacement_text + "\n"
            matched_at.append(index)
    if len(matched_at) != 1:
        raise ContractError(
            f"candidate {source.name}:{parameter} must match exactly one @export var assignment"
        )
    source.write_text("".join(updated), encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(original, updated, fromfile=f"a/{source.name}", tofile=f"b/{source.name}")
    )
    return diff, matched_at[0] + 1


def _source_tree_sha256(project: Path) -> str:
    """Hash every sealed project file with an unambiguous relative path."""
    root = Path(project).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"sealed project contains a non-file entry: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _receipt_base(contract: SafetyContract) -> dict[str, object]:
    renderer = contract.signal_kind == "frame_npz_v1"
    return {
        "schema": "flashpatch-renderer-engine-receipt-v1" if renderer else "flashpatch-safety-ci-receipt-v1",
        "contract_schema": SCHEMA,
        "project": str(contract.project),
        "contract": str(contract.contract_path),
        "trace": str(contract.trace),
        "input_sha256": {
            "project.godot": _sha256(contract.project / "project.godot"),
            "trace": _sha256(contract.trace),
            "source_snapshot": _source_tree_sha256(contract.project),
        },
        "risk_signal": {
            "kind": contract.signal_kind,
            "field": contract.signal_field,
            "threshold": contract.threshold,
            "measurement": "renderer-captured pixel detector" if renderer else "developer-declared deterministic replay signal",
            "limitation": "Rendered-pixel evidence is not a clinical safety certification." if renderer else "This release does not capture or certify rendered pixels or clinical safety.",
        },
    }


def _snapshot_contract(contract: SafetyContract, workspace: Path) -> tuple[SafetyContract, SafetyContract, Path]:
    """Create a sealed input tree before executing untrusted Godot code."""
    run_root = Path(tempfile.mkdtemp(prefix="flashpatch-run-", dir=workspace))
    sealed_project = run_root / "sealed-project"
    shutil.copytree(contract.project, sealed_project)

    def seal_path(path: Path, label: str) -> Path:
        try:
            return sealed_project / path.relative_to(contract.project)
        except ValueError:
            destination = run_root / "sealed-inputs" / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            return destination

    sealed = replace(
        contract,
        contract_path=seal_path(contract.contract_path, "contract.json"),
        project=sealed_project,
        trace=seal_path(contract.trace, "trace.json"),
        scene=seal_path(contract.scene, "scene.tscn"),
        candidates=tuple(
            PatchCandidate(
                seal_path(candidate.source, f"candidate-{index}.gd"),
                candidate.parameter,
                candidate.replacement,
                candidate.parameter_kind,
                candidate.runtime_binding,
                seal_path(candidate.runtime_resource, f"candidate-resource-{index}.tscn") if candidate.runtime_resource is not None else None,
                candidate.source_kind,
            )
            for index, candidate in enumerate(contract.candidates)
        ),
    )
    factual_project = run_root / "factual-project"
    shutil.copytree(sealed_project, factual_project)

    def factual_path(path: Path, label: str) -> Path:
        try:
            return factual_project / path.relative_to(sealed_project)
        except ValueError:
            destination = run_root / "factual-inputs" / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            return destination

    factual = replace(
        sealed,
        project=factual_project,
        trace=factual_path(sealed.trace, "trace.json"),
        scene=factual_path(sealed.scene, "scene.tscn"),
        candidates=tuple(
            PatchCandidate(
                factual_path(candidate.source, f"candidate-{index}.gd"),
                candidate.parameter,
                candidate.replacement,
                candidate.parameter_kind,
                candidate.runtime_binding,
                factual_path(candidate.runtime_resource, f"candidate-resource-{index}.tscn") if candidate.runtime_resource is not None else None,
                candidate.source_kind,
            )
            for index, candidate in enumerate(sealed.candidates)
        ),
    )
    return sealed, factual, run_root


def compile_project(
    project: Path | str,
    trace_or_contract: Path | str,
    *,
    workspace: Path | str,
    runner_factory: Callable[[Path], GodotReplayRunner] = GodotReplayRunner,
    checkpoint_path: Path | str | None = None,
) -> dict[str, object]:
    """Run the fail-closed contract and return a durable receipt payload."""
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    contract: SafetyContract | None = None
    receipt: dict[str, object] | None = None

    def checkpoint(phase: str) -> None:
        if receipt is None or checkpoint_path is None:
            return
        payload = dict(receipt)
        payload["execution_checkpoint"] = phase
        write_receipt(payload, checkpoint_path)

    try:
        contract = load_contract(project, trace_or_contract)
        sealed, factual_contract, run_root = _snapshot_contract(contract, workspace_path)
        receipt = _receipt_base(sealed)
        receipt["sealed_input_root"] = str(sealed.project)
        checkpoint("SEALED")
        action_count = len(json.loads(sealed.trace.read_text(encoding="utf-8"))["actions"])
        factual_path = run_root / "factual-replay.json"
        factual_runner = (
            GodotRendererReplayRunner(factual_contract.project)
            if sealed.signal_kind == "frame_npz_v1" and runner_factory is GodotReplayRunner
            else runner_factory(factual_contract.project)
        )
        factual = factual_runner.replay(factual_contract.trace, factual_path)
        factual_measurement = _measure_risk(factual, factual_path, sealed, action_count)
        factual_invariants = _semantic_invariants(factual) if sealed.signal_kind == "frame_npz_v1" else None
        factual_max = factual_measurement.maximum
        receipt["factual_replay"] = {
            "artifact": str(factual_path),
            "sha256": _sha256(factual_path),
            "max_risk": factual_max,
            **factual_measurement.details,
        }
        if factual_invariants is not None:
            receipt["factual_replay"]["semantic_invariants"] = factual_invariants
        checkpoint("FACTUAL_REPLAYED")
        if factual_max < sealed.threshold and not factual_measurement.details.get("hazardous", False):
            receipt.update({"verdict": "SAFE", "reason": "no_hazard_in_declared_trace", "candidates": []})
            return receipt
        outcomes: list[dict[str, object]] = []
        for index, candidate in enumerate(sealed.candidates):
            factual_candidate = factual_contract.candidates[index]
            candidate_root = run_root / "counterfactual" / f"{index:02d}"
            copy_root = candidate_root / "project"
            shutil.copytree(sealed.project, copy_root)
            copied_source = copy_root / candidate.source.relative_to(sealed.project)
            try:
                if candidate.source_kind != "gdscript_export":
                    raise ContractError(
                        "tscn_shader_parameter requires the native ShaderMaterial runtime observer "
                        "and candidate-tree receipt path"
                    )
                diff, source_line = _patch_export(copied_source, candidate.parameter, candidate.replacement)
                expected_node = (
                    _scene_binds_source(sealed.scene, candidate.source, sealed.project)
                    if candidate.runtime_binding == "scene"
                    else None
                )
                runtime_resource = (
                    candidate.runtime_resource
                    if candidate.runtime_resource is not None
                    else sealed.scene
                )
                diff_path = candidate_root / "patch.diff"
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                diff_path.write_text(diff, encoding="utf-8")
                replay_path = candidate_root / "replay.json"
                candidate_runner = (
                    GodotRendererReplayRunner(copy_root)
                    if sealed.signal_kind == "frame_npz_v1" and runner_factory is GodotReplayRunner
                    else runner_factory(copy_root)
                )
                replay = candidate_runner.replay(sealed.trace, replay_path)
                measurement = _measure_risk(replay, replay_path, sealed, action_count)
                maximum = measurement.maximum
                invariants = _semantic_invariants(replay) if sealed.signal_kind == "frame_npz_v1" else None
                runtime = (
                    _runtime_attribution(
                        factual,
                        factual_measurement,
                        factual_candidate,
                        factual_contract.project,
                        expected_node,
                        f"res://{runtime_resource.relative_to(sealed.project).as_posix()}",
                        source_line,
                    )
                    if sealed.signal_kind == "frame_npz_v1"
                    else {"node": expected_node, "source_line": source_line, "parameter": candidate.parameter}
                )
                candidate_runtime_event_count = (
                    _candidate_runtime_application(
                        replay,
                        factual_candidate,
                        factual_contract.project,
                        runtime,
                        f"res://{runtime_resource.relative_to(sealed.project).as_posix()}",
                    )
                    if sealed.signal_kind == "frame_npz_v1"
                    else None
                )
                visual_change_ratio = (
                    _visual_change_ratio(
                        Path(str(factual_measurement.details["frame_artifact"])),
                        Path(str(measurement.details["frame_artifact"])),
                    )
                    if sealed.signal_kind == "frame_npz_v1"
                    else None
                )
                patch_magnitude = (
                    abs(float(runtime["factual_value"]) - float(candidate.replacement))
                    if sealed.signal_kind == "frame_npz_v1"
                    else None
                )
                outcomes.append({
                    "source": str(candidate.source), "source_sha256": _sha256(candidate.source),
                    "source_line": runtime["source_line"],
                    "node": runtime["node"], "parameter": candidate.parameter,
                    "replacement": candidate.replacement, "parameter_kind": candidate.parameter_kind,
                    "runtime_binding": candidate.runtime_binding, "artifact": str(replay_path),
                    "artifact_sha256": _sha256(replay_path),
                    "diff": str(diff_path), "diff_sha256": _sha256(diff_path), "max_risk": maximum,
                    "causal_contribution": factual_max - maximum,
                    "hazard_removed": (
                        maximum < sealed.threshold
                        and not measurement.details.get("hazardous", False)
                    ),
                    "timing_preserved": measurement.timing_key == factual_measurement.timing_key,
                    "gameplay_state_preserved": measurement.state_key == factual_measurement.state_key,
                    "semantic_invariants_preserved": invariants == factual_invariants if invariants is not None else True,
                    "runtime_attribution": runtime if sealed.signal_kind == "frame_npz_v1" else None,
                    "candidate_runtime_event_count": candidate_runtime_event_count,
                    "changed_source_assignments": 1,
                    "patch_magnitude": patch_magnitude,
                    "visual_change_ratio": visual_change_ratio,
                    **measurement.details,
                    "status": "EVALUATED",
                })
            except (ContractError, RuntimeError, OSError) as exc:
                outcomes.append({"source": str(candidate.source), "parameter": candidate.parameter, "status": "INCONCLUSIVE", "reason": str(exc)})
            receipt["candidates"] = outcomes
            checkpoint(f"CANDIDATE_{index:02d}_RECORDED")
        valid = [outcome for outcome in outcomes if outcome.get("status") == "EVALUATED"]
        successful = [outcome for outcome in valid if outcome["hazard_removed"] and outcome["timing_preserved"] and outcome["gameplay_state_preserved"] and outcome["semantic_invariants_preserved"] and float(outcome["causal_contribution"]) > 0.0]
        receipt["candidates"] = outcomes
        if not valid:
            receipt.update({"verdict": "INCONCLUSIVE", "reason": "no_candidate_completed_counterfactual_replay"})
            return receipt
        invalid_preservation = [
            outcome
            for outcome in valid
            if outcome["hazard_removed"]
            and (
                not outcome["timing_preserved"]
                or not outcome["gameplay_state_preserved"]
                or not outcome["semantic_invariants_preserved"]
            )
        ]
        if invalid_preservation:
            receipt.update({"verdict": "FAIL", "reason": "patch_broke_declared_gameplay_invariants"})
            return receipt
        if not successful:
            if len(sealed.candidates) > 1:
                receipt.update({
                    "verdict": "INCONCLUSIVE",
                    "reason": "multiple_parameters_required_no_single_patch_authorized",
                })
            else:
                receipt.update({"verdict": "FAIL", "reason": "hazard_persists_after_all_declared_candidates"})
            return receipt
        successful.sort(
            key=lambda item: (
                float(item["patch_magnitude"]) if item["patch_magnitude"] is not None else 0.0,
                -float(item["causal_contribution"]),
                str(item["parameter"]),
                str(item["replacement"]),
            )
        )
        best = successful[0]
        ties = [
            item
            for item in successful
            if item["patch_magnitude"] == best["patch_magnitude"]
            and float(item["causal_contribution"]) == float(best["causal_contribution"])
        ]
        if len(ties) != 1:
            receipt.update({"verdict": "INCONCLUSIVE", "reason": "multiple_equally_effective_patch_candidates"})
            return receipt
        receipt.update({"verdict": "PASS", "reason": "same_trace_counterfactual_removed_declared_risk", "attribution": best, "patch_minimality": "one declared exported parameter with minimum declared scalar delta"})
        return receipt
    except (ContractError, FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        if receipt is not None:
            receipt.update({"verdict": "INCONCLUSIVE", "reason": str(exc)})
            return receipt
        schema = "flashpatch-renderer-engine-receipt-v1" if contract is not None and contract.signal_kind == "frame_npz_v1" else "flashpatch-safety-ci-receipt-v1"
        return {"schema": schema, "verdict": "INCONCLUSIVE", "reason": str(exc)}


def write_receipt(receipt: dict[str, object], output: Path | str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(receipt)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_sha256"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
