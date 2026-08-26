from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
from dataclasses import replace
from typing import Callable

import numpy as np
import pytest

from flashpatch.safety_ci import (
    ContractError,
    RiskMeasurement,
    _RENDERER_ANALYSIS_CACHE,
    _candidate_runtime_application,
    _measure_risk,
    _patch_tscn_shader_parameter,
    _renderer_analysis_summary,
    _require_main_scene,
    _runtime_attribution,
    _scene_binds_source,
    _source_tree_sha256,
    compile_project,
    load_contract,
    write_receipt,
)


ROOT = Path(__file__).parents[1]
SMOKE_PROJECT = ROOT / "examples" / "godot" / "interaction-burst"
SMOKE_CONTRACT = SMOKE_PROJECT / "flashpatch.contract.json"


def test_renderer_analysis_cache_reuses_only_exact_rgb_and_timestamp_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RENDERER_ANALYSIS_CACHE.clear()
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    timestamps = np.arange(3, dtype=np.float64) / 60.0
    calls = 0

    class Result:
        max_flash_count = 0.0
        max_affected_fraction = 0.0
        hazardous = False
        windows: tuple[object, ...] = ()
        hazard_mask = np.zeros((3, 2, 2), dtype=bool)

    def counted_analyze(observed_frames: np.ndarray, observed_timestamps: np.ndarray) -> Result:
        nonlocal calls
        calls += 1
        assert np.array_equal(observed_frames, frames)
        assert np.array_equal(observed_timestamps, timestamps)
        return Result()

    monkeypatch.setattr("flashpatch.safety_ci.analyze", counted_analyze)
    first, first_hit = _renderer_analysis_summary(
        frames,
        timestamps,
        renderer_rgb_raw_sha256="rgb-a",
        timestamps_sha256="time-a",
    )
    second, second_hit = _renderer_analysis_summary(
        frames,
        timestamps,
        renderer_rgb_raw_sha256="rgb-a",
        timestamps_sha256="time-a",
    )

    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert calls == 1

    _renderer_analysis_summary(
        frames,
        timestamps,
        renderer_rgb_raw_sha256="rgb-b",
        timestamps_sha256="time-a",
    )
    assert calls == 2


def test_renderer_analysis_deadline_fails_closed_before_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RENDERER_ANALYSIS_CACHE.clear()
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    timestamps = np.arange(3, dtype=np.float64) / 60.0

    def blocked_analyze(
        observed_frames: np.ndarray,
        observed_timestamps: np.ndarray,
    ) -> object:
        assert np.array_equal(observed_frames, frames)
        assert np.array_equal(observed_timestamps, timestamps)
        time.sleep(1.0)
        raise AssertionError("deadline did not interrupt the detector")

    monkeypatch.setattr("flashpatch.safety_ci.analyze", blocked_analyze)

    with pytest.raises(
        ContractError,
        match="renderer detector exceeded 0.05-second deadline",
    ):
        _renderer_analysis_summary(
            frames,
            timestamps,
            renderer_rgb_raw_sha256="deadline-rgb",
            timestamps_sha256="deadline-time",
            timeout_seconds=0.05,
        )

    assert _RENDERER_ANALYSIS_CACHE == {}


def _renderer_capture() -> dict[str, object]:
    return {
        "trace_sha256": f"sha256:{hashlib.sha256((SMOKE_PROJECT / 'trace.json').read_bytes()).hexdigest()}",
        "godot_version": "test-godot",
        "renderer_configuration": {"display_driver": "x11", "rendering_driver": "opengl3"},
    }


def _shader_parameter_project(
    tmp_path: Path,
    *,
    scene_text: str | None = None,
    shader_text: str = "shader_type canvas_item;\nuniform float flash_intensity;\n",
    source_kind: str | None = "tscn_shader_parameter",
) -> tuple[Path, Path, Path]:
    project = tmp_path / "shader-project"
    shader = project / "effects" / "flash.gdshader"
    shader.parent.mkdir(parents=True, exist_ok=True)
    scene = project / "main.tscn"
    default_scene = (
        '[gd_scene load_steps=3 format=3]\r\n'
        '\r\n'
        '[ext_resource type="Shader" path="res://effects/flash.gdshader" id="1_shader"]\r\n'
        '\r\n'
        '[sub_resource type="ShaderMaterial" id="ShaderMaterial_flash"]\r\n'
        'shader = ExtResource("1_shader")\r\n'
        'shader_parameter/flash_intensity = 1.250\r\n'
        '\r\n'
        '[node name="Main" type="Node2D"]\r\n'
        'material = SubResource("ShaderMaterial_flash")\r\n'
    )
    scene.write_bytes((default_scene if scene_text is None else scene_text).encode("utf-8"))
    shader.write_text(shader_text, encoding="utf-8")
    (project / "project.godot").write_text(
        'run/main_scene="res://main.tscn"\n', encoding="utf-8"
    )
    (project / "trace.json").write_text(
        json.dumps({"fixed_fps": 60, "actions": [{"frame": 0, "event": "demo"}]}),
        encoding="utf-8",
    )
    candidate: dict[str, object] = {
        "source": "main.tscn",
        "parameter": "flash_intensity",
        "replacement": 0.125,
    }
    if source_kind is not None:
        candidate["source_kind"] = source_kind
    contract_path = project / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema": "flashpatch-godot-safety-ci-v1",
                "trace": "trace.json",
                "scene": "main.tscn",
                "timing_field": "action_frames",
                "state_field": "gameplay_state",
                "risk_signal": {
                    "kind": "replay_observations_v1",
                    "field": "risk",
                    "threshold": 1.0,
                },
                "patch_candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    return project, contract_path, scene


def test_patch_candidate_source_kind_defaults_to_gdscript_export() -> None:
    candidate = load_contract(SMOKE_PROJECT, SMOKE_CONTRACT).candidates[0]

    assert candidate.source_kind == "gdscript_export"


def test_tscn_shader_parameter_static_binding_and_patch_are_byte_exact(
    tmp_path: Path,
) -> None:
    project, contract_path, scene = _shader_parameter_project(tmp_path)
    original = scene.read_bytes()

    candidate = load_contract(project, contract_path).candidates[0]
    diff, source_line = _patch_tscn_shader_parameter(
        scene, candidate.parameter, candidate.replacement
    )

    assert candidate.source_kind == "tscn_shader_parameter"
    assert candidate.source == scene
    assert source_line == 7
    assert scene.read_bytes() == original.replace(b"1.250", b"0.125")
    assert "-shader_parameter/flash_intensity = 1.250\r\n" in diff
    assert "+shader_parameter/flash_intensity = 0.125\r\n" in diff


@pytest.mark.parametrize(
    ("scene_transform", "error"),
    [
        (
            lambda scene: scene.replace(
                'material = SubResource("ShaderMaterial_flash")',
                'material = SubResource("ShaderMaterial_other")',
            ),
            "exactly one node",
        ),
        (
            lambda scene: scene.replace("1.250", "1.250 * 0.5"),
            "numeric literal",
        ),
        (
            lambda scene: scene
            + '\r\n[node name="AlsoBound" type="Node2D" parent="."]\r\n'
            + 'material = SubResource("ShaderMaterial_flash")\r\n',
            "exactly one node",
        ),
        (
            lambda scene: scene.replace(
                "shader_parameter/flash_intensity = 1.250",
                "shader_parameter/flash_intensity = 1.250\r\n"
                "shader_parameter/flash_intensity = 0.750",
            ),
            "exactly one shader_parameter",
        ),
    ],
)
def test_tscn_shader_parameter_rejects_ambiguous_binding_or_value(
    tmp_path: Path,
    scene_transform: Callable[[str], str],
    error: str,
) -> None:
    base = _shader_parameter_project(tmp_path)[2].read_text(encoding="utf-8")
    transformed = scene_transform(base)
    project, contract_path, _ = _shader_parameter_project(
        tmp_path, scene_text=transformed
    )

    with pytest.raises(ContractError, match=error):
        load_contract(project, contract_path)


def test_tscn_shader_parameter_requires_exact_float_uniform(tmp_path: Path) -> None:
    project, contract_path, _ = _shader_parameter_project(
        tmp_path,
        shader_text="shader_type canvas_item;\nuniform vec4 flash_intensity;\n",
    )

    with pytest.raises(ContractError, match="exact uniform float declaration"):
        load_contract(project, contract_path)


def test_upstream_godot_scene_syntax_binds_root_probe_and_relative_main_scene(tmp_path: Path) -> None:
    project = tmp_path / "pong"
    project.mkdir()
    scene = project / "pong.tscn"
    (project / "project.godot").write_text('run/main_scene="pong.tscn"\n', encoding="utf-8")
    (project / "flashpatch_probe.gd").write_text("extends Node2D\n", encoding="utf-8")
    scene.write_text(
        '[ext_resource type="Script" path="res://flashpatch_probe.gd" id="9"]\n\n'
        '[node name="Pong" type="Node2D"]\nscript = ExtResource("9")\n',
        encoding="utf-8",
    )
    _require_main_scene(project, scene)
    assert _scene_binds_source(scene, project / "flashpatch_probe.gd", project) == "/root/Pong"


def test_dynamic_runtime_binding_requires_an_observed_non_root_contributor(tmp_path: Path) -> None:
    project = tmp_path / "dynamic"
    project.mkdir()
    scene = project / "main.tscn"
    scene.write_text("[node name=\"Game\" type=\"Node2D\"]\n", encoding="utf-8")
    (project / "project.godot").write_text('run/main_scene="main.tscn"\n', encoding="utf-8")
    source = project / "effect.gd"
    source.write_text("@export var flash_intensity: float = 1.0\n", encoding="utf-8")
    trace = project / "trace.json"
    trace.write_text(json.dumps({"fixed_fps": 60, "actions": [{"frame": 0, "event": "demo"}]}), encoding="utf-8")
    contract_path = project / "contract.json"
    contract_path.write_text(json.dumps({
        "schema": "flashpatch-godot-safety-ci-v1",
        "trace": "trace.json",
        "scene": "main.tscn",
        "timing_field": "action_frames",
        "state_field": "gameplay_state",
        "risk_signal": {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0},
        "patch_candidates": [{
            "source": "effect.gd",
            "parameter": "flash_intensity",
            "parameter_kind": "intensity",
            "replacement": 0.0,
            "runtime_binding": "dynamic",
            "runtime_resource": "main.tscn",
        }],
    }), encoding="utf-8")
    contract = load_contract(project, contract_path)
    candidate = contract.candidates[0]
    assert candidate.runtime_binding == "dynamic"
    assert candidate.runtime_resource == scene
    replay = {
        "runtime_events": [{
            "frame_index": 1,
            "node_path": "/root/Game/RoutShockwave",
            "script_path": "res://effect.gd",
            "resource_path": "res://main.tscn",
            "source_line": 1,
            "property": "flash_intensity",
            "factual_value": 1.0,
            "event_kind": "render_property",
        }],
    }
    measurement = RiskMeasurement(5.0, "timing", "state", {"hazard_frame_indices": [1]})
    runtime = _runtime_attribution(replay, measurement, candidate, project, None, "res://main.tscn", 1)
    assert runtime["node"] == "/root/Game/RoutShockwave"
    # A detector index denotes the first image of a hazardous transition; the
    # render-property observation at the following captured image is still
    # that same transition, while a later event is not.
    replay["runtime_events"][0]["frame_index"] = 2
    measurement = RiskMeasurement(5.0, "timing", "state", {"hazard_frame_indices": [1]})
    assert _runtime_attribution(replay, measurement, candidate, project, None, "res://main.tscn", 1)["hazard_event_count"] == 1
    replay["runtime_events"][0]["frame_index"] = 3
    with pytest.raises(ContractError, match="no observed runtime contributor"):
        _runtime_attribution(replay, measurement, candidate, project, None, "res://main.tscn", 1)
    replay["runtime_events"][0]["frame_index"] = 1
    replay["runtime_events"][0]["node_path"] = "/root"
    with pytest.raises(ContractError, match="no observed runtime contributor"):
        _runtime_attribution(replay, measurement, candidate, project, None, "res://main.tscn", 1)


def test_counterfactual_runtime_event_must_report_replacement_value(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "effect.gd"
    source.write_text("@export var flash_intensity: float = 1.0\n", encoding="utf-8")
    candidate = replace(
        load_contract(SMOKE_PROJECT, SMOKE_CONTRACT).candidates[0],
        source=source,
        parameter="flash_intensity",
        replacement=0.0,
    )
    runtime = {"node": "/root/Game/Effect", "source_line": 1}
    event = {
        "node_path": "/root/Game/Effect",
        "script_path": "res://effect.gd",
        "resource_path": "res://main.tscn",
        "source_line": 1,
        "property": "flash_intensity",
        "factual_value": 0.0,
        "event_kind": "render_property",
    }

    assert _candidate_runtime_application(
        {"runtime_events": [event]}, candidate, project, runtime, "res://main.tscn"
    ) == 1
    event["factual_value"] = 1.0
    with pytest.raises(ContractError, match="replacement value"):
        _candidate_runtime_application(
            {"runtime_events": [event]}, candidate, project, runtime, "res://main.tscn"
        )
def test_contract_closes_declared_replay_to_single_source_patch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    receipt = compile_project(SMOKE_PROJECT, SMOKE_CONTRACT, workspace=tmp_path / "work", checkpoint_path=checkpoint)

    assert receipt["verdict"] == "PASS"
    assert receipt["risk_signal"]["measurement"] == "developer-declared deterministic replay signal"
    assert receipt["risk_signal"]["kind"] == "replay_observations_v1"
    assert receipt["risk_signal"]["limitation"]
    attribution = receipt["attribution"]
    assert attribution["parameter"] == "burst_intensity"
    assert attribution["hazard_removed"] is True
    assert attribution["timing_preserved"] is True
    assert Path(attribution["diff"]).is_file()
    assert attribution["diff_sha256"].startswith("sha256:")
    assert attribution["artifact_sha256"].startswith("sha256:")
    assert receipt["factual_replay"]["sha256"].startswith("sha256:")
    saved_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved_checkpoint["execution_checkpoint"] == "CANDIDATE_00_RECORDED"
    assert saved_checkpoint["factual_replay"]["sha256"].startswith("sha256:")


def test_missing_contract_is_inconclusive_and_receipt_is_hash_bound(tmp_path: Path) -> None:
    receipt = compile_project(SMOKE_PROJECT, tmp_path / "missing.json", workspace=tmp_path / "work")
    output = tmp_path / "receipt.json"
    write_receipt(receipt, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["verdict"] == "INCONCLUSIVE"
    assert "requires project/flashpatch.json" in saved["reason"]
    assert saved["receipt_sha256"].startswith("sha256:")


def test_direct_trace_with_project_contract_can_return_safe(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    trace = project / "safe-trace.json"
    trace.write_text(
        json.dumps({"fixed_fps": 60, "actions": [{"frame": 0, "charge": True}, {"frame": 1}]}),
        encoding="utf-8",
    )
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["trace"] = "safe-trace.json"
    (project / "flashpatch.json").write_text(json.dumps(contract), encoding="utf-8")

    receipt = compile_project(project, trace, workspace=tmp_path / "work")

    assert receipt["verdict"] == "SAFE"
    assert receipt["reason"] == "no_hazard_in_declared_trace"


def test_unbound_source_candidate_is_inconclusive(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    (project / "other.gd").write_text("@export var burst_intensity: float = 1.0\n", encoding="utf-8")
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["patch_candidates"][0]["source"] = "other.gd"
    contract_path = project / "unbound-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work")

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["reason"] == "no_candidate_completed_counterfactual_replay"
    assert "does not bind declared source" in receipt["candidates"][0]["reason"]


def test_frame_npz_signal_uses_pixel_detector_and_binds_artifact(tmp_path: Path) -> None:
    frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
    frames[1::2] = 255
    artifact = tmp_path / "frames.npz"
    np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
    contract = replace(
        load_contract(SMOKE_PROJECT, SMOKE_CONTRACT),
        signal_kind="frame_npz_v1",
        signal_field="frames_npz",
    )

    measurement = _measure_risk(
        {"frames_npz": artifact.name, "action_frames": [0, 1, 2, 3], "gameplay_state": "stable", "renderer_capture": _renderer_capture()},
        tmp_path / "replay.json",
        contract,
        4,
    )

    assert measurement.maximum > 0.0
    assert measurement.details["hazardous"] is True
    assert measurement.details["frame_artifact_sha256"].startswith("sha256:")


def test_frame_npz_regular_pattern_cannot_collapse_to_safe_scalar(tmp_path: Path) -> None:
    frames = np.full((3, 16, 32, 3), 96, dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        for stripe in range(12):
            value = 240 if (stripe + frame_index) % 2 else 8
            frame[:, stripe * 2 : stripe * 2 + 2] = value
    artifact = tmp_path / "regular-pattern.npz"
    np.savez_compressed(
        artifact,
        frames=frames,
        timestamps=np.arange(len(frames), dtype=np.float64) / 10.0,
    )
    contract = replace(
        load_contract(SMOKE_PROJECT, SMOKE_CONTRACT),
        signal_kind="frame_npz_v1",
        signal_field="frames_npz",
    )

    measurement = _measure_risk(
        {
            "frames_npz": artifact.name,
            "action_frames": [0, 1, 2, 3],
            "gameplay_state": "stable",
            "renderer_capture": _renderer_capture(),
        },
        tmp_path / "replay.json",
        contract,
        4,
    )

    assert measurement.maximum == 1.0
    assert measurement.details["hazardous"] is True
    assert measurement.details["hazard_kinds"] == ["regular_pattern"]


def test_frame_npz_contract_closes_to_source_patch(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {
        "kind": "frame_npz_v1",
        "field": "frames_npz",
        "threshold": 1.0,
    }
    contract["patch_candidates"][0]["parameter_kind"] = "intensity"
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class FrameRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            patched = "burst_intensity: float = 0.0" in (self.project / "main.gd").read_text()
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not patched:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            output.write_text(json.dumps({
                "status": "REPLAYED",
                "frames_npz": artifact.name,
                "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3],
                "gameplay_state": "stable",
                "semantic_invariants": {
                    "terminal_completion": True,
                    "terminal_state": "stable",
                    "player_world_digest": "stable",
                    "score": "score_not_applicable",
                },
                "runtime_events": [
                    {
                        "frame_index": frame,
                        "timestamp_us": frame + 1,
                        "node_path": "/root/InteractionBurst",
                        "resource_path": "res://main.tscn",
                        "script_path": "res://main.gd",
                        "source_line": 3,
                        "property": "burst_intensity",
                        "factual_value": 0.0 if patched else 1.0,
                        "event_kind": "render_property",
                    }
                    for frame in range(8)
                ],
            }), encoding="utf-8")
            return json.loads(output.read_text())

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=FrameRunner)

    assert receipt["verdict"] == "PASS"
    assert receipt["schema"] == "flashpatch-renderer-engine-receipt-v1"
    assert receipt["factual_replay"]["hazardous"] is True
    assert receipt["attribution"]["hazard_removed"] is True
    assert receipt["attribution"]["frame_artifact_sha256"].startswith("sha256:")
    assert receipt["attribution"]["runtime_attribution"]["node"] == "/root/InteractionBurst"


def test_tscn_shader_parameter_cannot_claim_renderer_pass_without_native_observer(
    tmp_path: Path,
) -> None:
    project, contract_path, _ = _shader_parameter_project(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["risk_signal"] = {
        "kind": "frame_npz_v1",
        "field": "frames_npz",
        "threshold": 1.0,
    }
    contract["patch_candidates"][0]["parameter_kind"] = "intensity"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class ShaderLikeRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(
                artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0
            )
            renderer_capture = _renderer_capture()
            renderer_capture["trace_sha256"] = f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}"
            payload: dict[str, object] = {
                "status": "REPLAYED",
                "frames_npz": artifact.name,
                "renderer_capture": renderer_capture,
                "action_frames": [0],
                "gameplay_state": "stable",
                "semantic_invariants": {
                    "terminal_completion": True,
                    "terminal_state": "stable",
                    "player_world_digest": "stable",
                    "score": "score_not_applicable",
                },
                "runtime_events": [],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(
        project, contract_path, workspace=tmp_path / "work", runner_factory=ShaderLikeRunner
    )

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["reason"] == "no_candidate_completed_counterfactual_replay"
    assert "native ShaderMaterial runtime observer" in receipt["candidates"][0]["reason"]


def test_frame_npz_contract_rejects_missing_runtime_attribution(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0}
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class NoAttributionRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3], "gameplay_state": "stable",
                "semantic_invariants": {"terminal_completion": True, "terminal_state": "stable", "player_world_digest": "stable", "score": "score_not_applicable"},
                "runtime_events": [],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=NoAttributionRunner)

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["reason"] == "no_candidate_completed_counterfactual_replay"
    assert "runtime_events" in receipt["candidates"][0]["reason"]


def test_frame_npz_contract_rejects_hazard_removal_when_gameplay_changes(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0}
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class StateBreakingRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            patched = "burst_intensity: float = 0.0" in (self.project / "main.gd").read_text()
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not patched:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            state = "changed" if patched else "stable"
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3], "gameplay_state": state,
                "semantic_invariants": {"terminal_completion": True, "terminal_state": state, "player_world_digest": state, "score": "score_not_applicable"},
                "runtime_events": [
                    {"frame_index": frame, "timestamp_us": frame + 1, "node_path": "/root/InteractionBurst", "resource_path": "res://main.tscn", "script_path": "res://main.gd", "source_line": 3, "property": "burst_intensity", "factual_value": 0.0 if patched else 1.0, "event_kind": "render_property"}
                    for frame in range(8)
                ],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=StateBreakingRunner)

    assert receipt["verdict"] == "FAIL"
    assert receipt["reason"] == "patch_broke_declared_gameplay_invariants"


def test_frame_npz_contract_rejects_wrong_runtime_node(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0}
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class WrongNodeRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            patched = "burst_intensity: float = 0.0" in (self.project / "main.gd").read_text()
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not patched:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3], "gameplay_state": "stable",
                "semantic_invariants": {"terminal_completion": True, "terminal_state": "stable", "player_world_digest": "stable", "score": "score_not_applicable"},
                "runtime_events": [
                    {"frame_index": frame, "timestamp_us": frame + 1, "node_path": "/root/WrongNode", "resource_path": "res://main.tscn", "script_path": "res://main.gd", "source_line": 3, "property": "burst_intensity", "factual_value": 0.0 if patched else 1.0, "event_kind": "render_property"}
                    for frame in range(8)
                ],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=WrongNodeRunner)

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert "runtime contributor" in receipt["candidates"][0]["reason"]


def test_frame_npz_contract_returns_safe_for_actual_frame_shape(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0}
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class SafeFrameRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), timestamps=np.arange(4, dtype=np.float64) / 10.0)
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3], "gameplay_state": "stable",
                "semantic_invariants": {"terminal_completion": True, "terminal_state": "stable", "player_world_digest": "stable", "score": "score_not_applicable"},
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=SafeFrameRunner)

    assert receipt["verdict"] == "SAFE"


def test_frame_npz_contract_rejects_malformed_capture(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads(SMOKE_CONTRACT.read_text(encoding="utf-8"))
    contract["risk_signal"] = {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0}
    contract_path = project / "frame-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class MalformedFrameRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=np.zeros((2, 8, 8, 3), dtype=np.uint8), timestamps=np.array([0.0, 0.0]))
            payload = {"status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(), "action_frames": [0, 1, 2, 3], "gameplay_state": "stable"}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=MalformedFrameRunner)

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert "timestamps" in receipt["reason"] or "three frames" in receipt["reason"]


def test_factual_replay_cannot_mutate_counterfactual_snapshot(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)

    class MutatingRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            source = self.project / "main.gd"
            patched = "burst_intensity: float = 0.0" in source.read_text()
            payload = {"status": "REPLAYED", "observations": [0.0 if patched else 1.0] * 4, "action_frames": [0, 1, 2, 3], "gameplay_state": "stable"}
            if "factual-project" in str(self.project):
                source.write_text(source.read_text().replace("1.0", "0.0", 1), encoding="utf-8")
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, SMOKE_CONTRACT, workspace=tmp_path / "work", runner_factory=MutatingRunner)

    assert receipt["verdict"] == "PASS"
    assert "burst_intensity: float = 1.0" in (project / "main.gd").read_text()
    assert receipt["input_sha256"]["source_snapshot"].startswith("sha256:")


def test_source_snapshot_hash_binds_relative_paths_and_contents(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.godot").write_text("[application]\n", encoding="utf-8")
    (project / "main.gd").write_text("extends Node\n", encoding="utf-8")

    first = _source_tree_sha256(project)
    (project / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    second = _source_tree_sha256(project)
    (project / "other.gd").write_text("extends Node2D\n", encoding="utf-8")
    third = _source_tree_sha256(project)

    assert first.startswith("sha256:")
    assert first != second
    assert second != third


def test_mismatched_action_frame_sequence_is_inconclusive(tmp_path: Path) -> None:
    class MismatchedTimingRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            payload = {"status": "REPLAYED", "observations": [1.0] * 4, "action_frames": [0, 1, 3, 2], "gameplay_state": "stable"}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(SMOKE_PROJECT, SMOKE_CONTRACT, workspace=tmp_path / "work", runner_factory=MismatchedTimingRunner)

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert "action-frame sequence" in receipt["reason"]


def test_persistent_risk_after_valid_candidate_is_fail(tmp_path: Path) -> None:
    class PersistentRiskRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            payload = {"status": "REPLAYED", "observations": [1.0] * 4, "action_frames": [0, 1, 2, 3], "gameplay_state": "stable"}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(SMOKE_PROJECT, SMOKE_CONTRACT, workspace=tmp_path / "work", runner_factory=PersistentRiskRunner)

    assert receipt["verdict"] == "FAIL"
    assert receipt["reason"] == "hazard_persists_after_all_declared_candidates"


def test_renderer_selects_minimum_declared_single_parameter_delta(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads((SMOKE_PROJECT / "flashpatch.renderer.contract.json").read_text(encoding="utf-8"))
    contract["patch_candidates"] = [
        {"source": "main.gd", "parameter": "burst_intensity", "parameter_kind": "intensity", "replacement": 0.5},
        {"source": "main.gd", "parameter": "burst_intensity", "parameter_kind": "intensity", "replacement": 0.0},
    ]
    contract_path = project / "renderer-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class MinimumDeltaRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            source = (self.project / "main.gd").read_text(encoding="utf-8")
            patched = "burst_intensity: float = 1.0" not in source
            runtime_value = 1.0
            if "burst_intensity: float = 0.5" in source:
                runtime_value = 0.5
            elif "burst_intensity: float = 0.0" in source:
                runtime_value = 0.0
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not patched:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name, "renderer_capture": _renderer_capture(),
                "action_frames": [0, 1, 2, 3], "gameplay_state": "stable",
                "semantic_invariants": {"terminal_completion": True, "terminal_state": "stable", "player_world_digest": "stable", "score": "score_not_applicable"},
                "runtime_events": [
                    {"frame_index": frame, "timestamp_us": frame + 1, "node_path": "/root/InteractionBurst", "resource_path": "res://main.tscn", "script_path": "res://main.gd", "source_line": 3, "property": "burst_intensity", "factual_value": runtime_value, "event_kind": "render_property"}
                    for frame in range(8)
                ],
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=MinimumDeltaRunner)

    assert receipt["verdict"] == "PASS"
    assert receipt["attribution"]["replacement"] == 0.5
    assert receipt["attribution"]["changed_source_assignments"] == 1
    assert receipt["attribution"]["patch_magnitude"] == 0.5
    assert receipt["attribution"]["visual_change_ratio"] > 0.0


def test_renderer_contract_rejects_missing_visual_parameter_taxonomy(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract = json.loads((SMOKE_PROJECT / "flashpatch.renderer.contract.json").read_text(encoding="utf-8"))
    del contract["patch_candidates"][0]["parameter_kind"]
    contract_path = project / "renderer-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ContractError, match="parameter_kind"):
        load_contract(project, contract_path)


def test_renderer_capture_trace_metadata_mismatch_is_inconclusive(tmp_path: Path) -> None:
    frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
    frames[1::2] = 255
    artifact = tmp_path / "frames.npz"
    np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
    contract = replace(load_contract(SMOKE_PROJECT, SMOKE_CONTRACT), signal_kind="frame_npz_v1", signal_field="frames_npz")
    capture = _renderer_capture()
    capture["trace_sha256"] = "sha256:not-the-declared-trace"

    with pytest.raises(ContractError, match="trace hash"):
        _measure_risk(
            {"frames_npz": artifact.name, "action_frames": [0, 1, 2, 3], "gameplay_state": "stable", "renderer_capture": capture},
            tmp_path / "replay.json",
            contract,
            4,
        )


def test_renderer_capture_rejects_reversed_timestamps(tmp_path: Path) -> None:
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    artifact = tmp_path / "frames.npz"
    np.savez_compressed(artifact, frames=frames, timestamps=np.array([0.2, 0.1, 0.3]))
    contract = replace(load_contract(SMOKE_PROJECT, SMOKE_CONTRACT), signal_kind="frame_npz_v1", signal_field="frames_npz")
    with pytest.raises(ValueError, match="timestamps"):
        _measure_risk({"frames_npz": artifact.name, "action_frames": [0, 1, 2, 3], "gameplay_state": "stable", "renderer_capture": _renderer_capture()}, tmp_path / "replay.json", contract, 4)


@pytest.mark.parametrize("field,value", [("resource_path", "res://other.tscn"), ("source_line", 99), ("property", "unused_intensity")])
def test_renderer_rejects_runtime_provenance_mismatch(tmp_path: Path, field: str, value: object) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    contract_path = project / "renderer-contract.json"
    contract_path.write_text((SMOKE_PROJECT / "flashpatch.renderer.contract.json").read_text(), encoding="utf-8")

    class MismatchRunner:
        def __init__(self, replay_project: Path) -> None:
            self.project = replay_project
        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            patched = "burst_intensity: float = 0.0" in (self.project / "main.gd").read_text()
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not patched:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            event = {"frame_index": 1, "timestamp_us": 1, "node_path": "/root/InteractionBurst", "resource_path": "res://main.tscn", "script_path": "res://main.gd", "source_line": 3, "property": "burst_intensity", "factual_value": 1.0, "event_kind": "render_property"}
            event[field] = value
            payload = {"status":"REPLAYED","frames_npz":artifact.name,"renderer_capture":_renderer_capture(),"action_frames":[0,1,2,3],"gameplay_state":"stable","semantic_invariants":{"terminal_completion":True,"terminal_state":"stable","player_world_digest":"stable","score":"score_not_applicable"},"runtime_events":[event]}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=MismatchRunner)
    assert receipt["verdict"] == "INCONCLUSIVE"


def test_renderer_never_combines_two_parameters_to_create_a_pass(tmp_path: Path) -> None:
    project = tmp_path / "interaction-burst"
    shutil.copytree(SMOKE_PROJECT, project)
    source = project / "main.gd"
    source.write_text(source.read_text(encoding="utf-8") + "@export var secondary_intensity: float = 1.0\n", encoding="utf-8")
    contract = json.loads((SMOKE_PROJECT / "flashpatch.renderer.contract.json").read_text())
    contract["patch_candidates"].append({"source":"main.gd","parameter":"secondary_intensity","parameter_kind":"intensity","replacement":0.0})
    contract_path = project / "renderer-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    class RequiresBothRunner:
        def __init__(self, replay_project: Path) -> None: self.project = replay_project
        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            text = (self.project / "main.gd").read_text()
            both = "burst_intensity: float = 0.0" in text and "secondary_intensity: float = 0.0" in text
            frames = np.zeros((8, 8, 8, 3), dtype=np.uint8)
            if not both:
                frames[1::2] = 255
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
            burst_value = 0.0 if "burst_intensity: float = 0.0" in text else 1.0
            secondary_value = 0.0 if "secondary_intensity: float = 0.0" in text else 1.0
            payload = {"status":"REPLAYED","frames_npz":artifact.name,"renderer_capture":_renderer_capture(),"action_frames":[0,1,2,3],"gameplay_state":"stable","semantic_invariants":{"terminal_completion":True,"terminal_state":"stable","player_world_digest":"stable","score":"score_not_applicable"},"runtime_events":[{"frame_index":1,"timestamp_us":1,"node_path":"/root/InteractionBurst","resource_path":"res://main.tscn","script_path":"res://main.gd","source_line":3,"property":"burst_intensity","factual_value":burst_value,"event_kind":"render_property"},{"frame_index":1,"timestamp_us":1,"node_path":"/root/InteractionBurst","resource_path":"res://main.tscn","script_path":"res://main.gd","source_line":8,"property":"secondary_intensity","factual_value":secondary_value,"event_kind":"render_property"}]}
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    receipt = compile_project(project, contract_path, workspace=tmp_path / "work", runner_factory=RequiresBothRunner)
    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["reason"] == "multiple_parameters_required_no_single_patch_authorized"
