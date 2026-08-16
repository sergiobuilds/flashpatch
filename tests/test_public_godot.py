from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from flashpatch.public_godot import (
    GODOT_DEMO_REVISION,
    SPARTA_REVISION,
    NativeMainOptionButtonPopupPreflightRunner,
    NativeTscnTokenPatch,
    NativeMainUiRectPreflightRunner,
    SpartaRendererRunner,
    classify_capture_only_qualification,
    classify_native_main_capture_qualification,
    execute_controlled_pong,
    execute_controlled_sparta,
    execute_native_main_shader_counterfactual,
    execute_native_main_ui_rect_preflight,
    execute_native_main_option_button_popup_preflight,
    execute_repeated_capture_only_qualification,
    materialize_capture_only_qualification,
    materialize_controlled_native_main_shader_qualification,
    materialize_controlled_pong,
    materialize_controlled_sparta,
    materialize_native_main_capture_qualification,
    materialize_native_main_ui_rect_preflight,
    materialize_native_main_option_button_popup_preflight,
)
from flashpatch.public_godot import _verify_native_main_candidate_tree


def _upstream(tmp_path: Path) -> Path:
    root = tmp_path / "upstream"
    root.mkdir()
    (root / "project.godot").write_text('config_version=5\nrun/main_scene="pong.tscn"\n', encoding="utf-8")
    (root / "pong.tscn").write_text(
        '[ext_resource type="Script" path="res://logic/ceiling_floor.gd" id="8"]\n\n[node name="Pong" type="Node2D"]\n',
        encoding="utf-8",
    )
    return root


def _shader_tscn_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "shader-tscn-upstream"
    root.mkdir()
    (root / "project.godot").write_text(
        'config_version=5\nrun/main_scene="main.tscn"\n', encoding="utf-8"
    )
    (root / "main.tscn").write_bytes(
        b"[gd_scene format=3]\r\n\r\n"
        b"[sub_resource type=\"ShaderMaterial\" id=\"ShaderMaterial_flash\"]\r\n"
        b"shader_parameter/flash_intensity = 1.250\r\n\r\n"
        b"[node name=\"Main\" type=\"Node2D\"]\r\n"
        b"material = SubResource(\"ShaderMaterial_flash\")\r\n"
    )
    (root / "logic.gd").write_text("extends Node\n", encoding="utf-8")
    return root


def _bound_shader_tscn_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "bound-shader-tscn-upstream"
    (root / "effects").mkdir(parents=True)
    (root / "project.godot").write_text(
        'config_version=5\nrun/main_scene="main.tscn"\n', encoding="utf-8"
    )
    (root / "main.tscn").write_text(
        '[gd_scene load_steps=3 format=3]\n'
        '\n'
        '[ext_resource type="Shader" path="res://effects/flash.gdshader" id="1"]\n'
        '\n'
        '[sub_resource type="ShaderMaterial" id="ShaderMaterial_flash"]\n'
        'shader = ExtResource("1")\n'
        'shader_parameter/flash_intensity = 1.250\n'
        '\n'
        '[node name="Main" type="Node2D"]\n'
        '\n'
        '[node name="Flash" type="ColorRect" parent="."]\n'
        'material = SubResource("ShaderMaterial_flash")\n',
        encoding="utf-8",
    )
    (root / "effects" / "flash.gdshader").write_text(
        "shader_type canvas_item;\nuniform float flash_intensity = 1.0;\n",
        encoding="utf-8",
    )
    return root


def _native_shader_counterfactual_qualification(tmp_path: Path):
    observation = {
        "kind": "shader_parameter",
        "node_path": "/root/Main/Flash",
        "property": "flash_intensity",
        "source_path": "res://main.tscn",
        "resource_path": "res://main.tscn",
        "source_line": 7,
        "shader_path": "res://effects/flash.gdshader",
    }
    return materialize_native_main_capture_qualification(
        _bound_shader_tscn_upstream(tmp_path),
        tmp_path / "native-shader-qualification",
        capture_frames=4,
        actions=[{"frame": 1, "action": "flash", "pressed": True}],
        runtime_observations=[observation],
        scenario_readiness={
            "required_node_paths": ["/root/Main/Flash"],
            "required_group_minimums": {},
            "required_visible": [{"node_path": "/root/Main/Flash", "visible": True}],
        },
    )


def _controlled_native_shader_counterfactual_qualification(tmp_path: Path):
    return materialize_controlled_native_main_shader_qualification(
        tmp_path / "controlled-native-shader-qualification"
    )


def _install_native_shader_counterfactual_fakes(
    monkeypatch: pytest.MonkeyPatch,
    qualification,
    *,
    meaningful_oracle: bool = False,
    attack: str | None = None,
) -> None:
    def fake_analyze(frames: np.ndarray, timestamps: np.ndarray) -> SimpleNamespace:
        del timestamps
        hazardous = bool(frames[0, 0, 0, 0])
        hazard_mask = np.zeros(frames.shape[:3], dtype=np.bool_)
        if hazardous:
            hazard_mask[1] = True
        return SimpleNamespace(
            hazardous=hazardous,
            max_flash_count=1.0 if hazardous else 0.0,
            hazard_mask=hazard_mask,
        )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        candidate = self.project.name == "candidate-project"
        hazardous = not candidate or attack == "residual_hazard"
        declared_trace = json.loads(trace.read_text(encoding="utf-8"))
        frames = np.full(
            (int(declared_trace["capture_frames"]), 4, 4, 3),
            255 if hazardous else 0,
            dtype=np.uint8,
        )
        payload = _native_main_replay(qualification, output, frames=frames)
        payload["action_frames"] = [item["frame"] for item in declared_trace["actions"]]
        source = (self.project / "main.tscn").read_text(encoding="utf-8")
        matched = re.search(r"shader_parameter/flash_intensity\s*=\s*([^\s]+)", source)
        assert matched is not None
        observed_value = float(matched.group(1))
        for event in payload["runtime_events"]:
            event["factual_value"] = observed_value
        if meaningful_oracle:
            payload["gameplay_state"] = "state-stream-sha256:stable"
            payload["semantic_invariants"] = {
                "terminal_completion": True,
                "terminal_state": "completed",
                "player_world_digest": "world-sha256:stable",
                "score": "score_not_applicable",
            }
        if candidate and attack == "state_mismatch":
            payload["state_observations"][-1]["values"]["player_x"] = 65
        if candidate and attack == "wrong_identity":
            payload["runtime_events"][0]["shader_path"] = "res://effects/other.gdshader"
        if candidate and attack == "wrong_value":
            payload["runtime_events"][0]["factual_value"] = 0.5
        if candidate and attack == "tree_mutation":
            shader = self.project / "effects" / "flash.gdshader"
            shader.write_text(shader.read_text(encoding="utf-8") + "// changed during replay\n", encoding="utf-8")
        output.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("flashpatch.public_godot.analyze", fake_analyze)
    def fake_init(
        self,
        project: Path,
        godot_binary: Path | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        del godot_binary, timeout_seconds
        self.project = Path(project).resolve()

    monkeypatch.setattr(
        "flashpatch.public_godot.GodotNativeMainRendererReplayRunner.__init__",
        fake_init,
    )
    monkeypatch.setattr(
        "flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay",
        replay,
    )


def _option_button_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "option-button-upstream"
    root.mkdir()
    (root / "project.godot").write_text(
        'config_version=5\nrun/main_scene="main.tscn"\n',
        encoding="utf-8",
    )
    (root / "main.tscn").write_text(
        "[gd_scene format=3]\n\n"
        '[node name="Pong" type="Control"]\n\n'
        '[node name="Options" type="OptionButton" parent="."]\n',
        encoding="utf-8",
    )
    return root


def _popup_metadata_result(
    *,
    option_button_path: str = "/root/Pong/Options",
    items: list[dict[str, object]] | None = None,
    activation: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": "flashpatch-native-main-option-button-popup-preflight-v2",
        "status": "PROBED",
        "execution_mode": "metadata_only_native_main_option_button_popup_preflight",
        "qualification_only": True,
        "scoreable": False,
        "renderer_png_capture": False,
        "detector_executed": False,
        "original_main_scene": "res://main.tscn",
        "current_scene_node_path": "/root/Pong",
        "wrapper_ancestor_paths": [],
        "option_button": {
            "node_path": option_button_path,
            "rect": [120.0, 80.0, 64.0, 24.0],
            "visible": True,
            "enabled": True,
            "viewport": [320.0, 180.0],
        },
        "activation": activation or {
            "kind": "input_event_left_click",
            "button_center": [152.0, 92.0],
        },
        "items": items if items is not None else [{
            "index": 0,
            "text": "Fast",
            "enabled": True,
        }],
    }


def _godot3_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "godot3-upstream"
    root.mkdir()
    (root / "project.godot").write_text('config_version=4\nrun/main_scene="main.tscn"\n', encoding="utf-8")
    (root / "main.tscn").write_text('[gd_scene format=2]\n\n[node name="Main" type="Node"]\n', encoding="utf-8")
    return root


def _pin_clean_repository(project: Path, repository: str) -> str:
    def git(*arguments: str) -> None:
        subprocess.run(["git", "-C", str(project), *arguments], check=True, capture_output=True, text=True)

    git("init")
    git("add", ".")
    git("-c", "user.name=FlashPatch Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    git("remote", "add", "origin", repository)
    return subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _native_main_replay(
    qualification,
    output: Path,
    *,
    runtime_scene: str | None = None,
    frames: np.ndarray | None = None,
    timestamps_us: list[int] | None = None,
    actual_timestamps_us: list[int] | None = None,
    scoreable: bool = False,
    native_equivalence: str = "NOT_ESTABLISHED",
) -> dict[str, object]:
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    capture_frames = int(trace["capture_frames"])
    warmup_frames = int(trace["warmup_frames"])
    fixed_fps = int(trace["fixed_fps"])
    presentation = timestamps_us if timestamps_us is not None else [
        int((warmup_frames + index) * 1_000_000 / fixed_fps)
        for index in range(capture_frames)
    ]
    actual = actual_timestamps_us if actual_timestamps_us is not None else [
        (index + 1) * 10_000 for index in range(capture_frames)
    ]
    pixels = frames if frames is not None else np.zeros((capture_frames, 4, 4, 3), dtype=np.uint8)
    artifact = output.with_name("renderer-frames.npz")
    np.savez_compressed(
        artifact,
        frames=pixels,
        timestamps=np.asarray(presentation, dtype=np.float64) / 1_000_000.0,
    )
    scene = runtime_scene or qualification.original_main_scene
    acknowledgements = [
        {
            "frame": action["frame"],
            "action": action["action"],
            "pressed": action["pressed"],
            "status": "APPLIED",
        }
        for index, action in enumerate(trace["actions"])
    ]
    pointer_acknowledgements = [
        {
            "frame": event["frame"],
            "kind": event["kind"],
            "x": event["x"],
            "y": event["y"],
            "status": "APPLIED",
        }
        for event in trace["pointer_events"]
    ]
    key_acknowledgements = [
        {"frame": event["frame"], "key": event["key"], "status": "APPLIED"}
        for event in trace["key_events"]
    ]
    runtime_events = [
        (
            {
                "frame_index": frame_index,
                "node_path": observation["node_path"],
                "source_path": observation["source_path"],
                "resource_path": observation["resource_path"],
                "source_line": observation["source_line"],
                "shader_path": observation["shader_path"],
                "property": observation["property"],
                "factual_value": 1.0,
                "event_kind": "shader_parameter",
            }
            if observation.get("kind") == "shader_parameter"
            else {
                "frame_index": frame_index,
                "node_path": observation["node_path"],
                "script_path": observation["script_path"],
                "resource_path": observation["resource_path"],
                "source_line": observation["source_line"],
                "property": observation["property"],
                "factual_value": 1.0,
                "event_kind": "render_property",
            }
        )
        for frame_index in range(capture_frames)
        for observation in trace["runtime_observations"]
    ]
    ui_selection_observations = [
        {
            "frame_index": frame_index,
            "node_path": path,
            "selected_index": 0,
            "selected_text": "Disabled",
            "popup_visible": False,
            "popup_focused_index": -1,
        }
        for frame_index in range(capture_frames)
        for path in trace["ui_selection_observations"]
    ]
    ui_selection_signal_events = []
    declared_readiness = trace["scenario_readiness"]
    readiness = (
        {"declared": False, "satisfied": False, "reason": "scenario_readiness_not_declared"}
        if declared_readiness is None
        else {
            "declared": True,
            "satisfied": True,
            "missing_node_paths": [],
            "observed_group_counts": dict(declared_readiness["required_group_minimums"]),
            "insufficient_groups": {},
            "visibility_mismatches": [],
            "selection_mismatches": [],
        }
    )
    replay = {
        "status": "REPLAYED",
        "qualification_only": True,
        "scoreable": scoreable,
        "native_equivalence": native_equivalence,
        "execution_mode": "instrumented_native_main_scene_capture",
        "action_acknowledgements": acknowledgements,
        "pointer_acknowledgements": pointer_acknowledgements,
        "key_acknowledgements": key_acknowledgements,
        "runtime_events": runtime_events,
        "ui_selection_observations": ui_selection_observations,
        "ui_selection_signal_events": ui_selection_signal_events,
        "scenario_readiness": readiness,
        "native_main": {
            "current_scene_exists": True,
            "expected_scene_file_path": qualification.original_main_scene,
            "runtime_scene_file_path": scene,
            "scene_file_path_match": scene == qualification.original_main_scene,
            "current_scene_node_path": "/root/Pong",
            "wrapper_ancestor_paths": [],
            "no_wrapper_ancestor": True,
            "verification_process_frame": 1,
        },
        "renderer_capture": {
            "timestamps_us": presentation,
            "actual_capture_timestamps_us": actual,
            "warmup_frames": warmup_frames,
            "capture_trace_frame_indices": list(
                range(warmup_frames, warmup_frames + capture_frames)
            ),
        },
        "frames_npz": artifact.name,
    }
    descriptor = trace.get("state_observation")
    if descriptor is not None:
        state_observations = []
        for frame_index, timestamp in enumerate(presentation):
            transitioned = frame_index >= 1
            state_observations.append({
                "frame_index": frame_index,
                "presentation_timestamp_us": timestamp,
                "values": {
                    "phase": 1 if transitioned else 0,
                    "player_x": 64 if transitioned else 0,
                    "world_epoch": 3 if transitioned else 0,
                    "completed": transitioned,
                    "score": 7 if transitioned else 0,
                },
            })
        replay["state_observation_descriptor"] = descriptor
        replay["state_observations"] = state_observations
    output.write_text(json.dumps(replay), encoding="utf-8")
    return replay


def _sparta_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "sparta"
    (root / "tools" / "demo").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scenes").mkdir()
    (root / "project.godot").write_text('run/main_scene="res://scenes/MainMenu.tscn"\n', encoding="utf-8")
    (root / "tools" / "demo" / "DemoInputRecorder.tscn").write_text("[node name=\"DemoInputRecorder\" type=\"Node\"]\n", encoding="utf-8")
    (root / "tools" / "demo" / "DemoInputRecorder.gd").write_text(
        "extends Node\n"
        "\t_cam = _battle.get_node(\"Camera2D\")\n\t_apply_camera(0)\n"
        "\tif _frame_ticks.has(tick) and not _captured.has(tick):\n\t\t_captured[tick] = true\n\t\t_capture_frame(tick)\n"
        "func _capture_frame(tick: int) -> void:\n"
        "\tvar img := get_viewport().get_texture().get_image()\n"
        "\tvar path := \"frame_%06d.png\" % tick\n"
        "\tvar err: int = img.save_png(path)\n"
        "\tif err != OK:\n\t\tpass\n"
        "\telse:\n\t\tprint(\"[demo-input] captured frame at tick %d -> %s (%dx%d)\" % [tick, path, img.get_width(), img.get_height()])\n"
        "\n\n## Set the camera to the track's framing for\n",
        encoding="utf-8",
    )
    (root / "scripts" / "RoutShockwave.gd").write_text(
        "class_name RoutShockwave\nextends TransientEffect\n"
        "\t\tdraw_circle(Vector2.ZERO, disc_r, Color(_color, fade * 0.1))\n"
        "\tdraw_arc(Vector2.ZERO, r, 0.0, TAU, 32, Color(_color, fade * 0.6), 2.0)\n",
        encoding="utf-8",
    )
    (root / "scenes" / "Battle.tscn").write_text("[node name=\"Battle\" type=\"Node2D\"]\n", encoding="utf-8")
    return root


def test_controlled_pong_is_a_separate_mutation_with_one_exported_patch(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    result = materialize_controlled_pong(upstream, tmp_path / "controlled")

    assert (upstream / "flashpatch_probe.gd").exists() is False
    assert result.source["source_revision"] == GODOT_DEMO_REVISION
    script = result.mutation_script.read_text(encoding="utf-8")
    assert script.count("@export var") == 1
    assert "@export var flash_intensity: float = 1.0" in script
    assert '"engine_process_frame": Engine.get_process_frames()' in script
    assert '"ball_speed": float(ball.get("_speed"))' in script
    assert '"actual_capture_timestamp_us": _actual_capture_timestamps_us.back()' in script
    scene = (result.project / "pong.tscn").read_text(encoding="utf-8")
    assert 'path="res://flashpatch_probe.gd"' in scene
    assert 'script = ExtResource("9")' in scene
    contract = json.loads(result.contract.read_text(encoding="utf-8"))
    assert contract["patch_candidates"] == [{
        "source": "flashpatch_probe.gd",
        "parameter": "flash_intensity",
        "parameter_kind": "intensity",
        "replacement": 0.0,
    }]
    assert json.loads(result.trace.read_text(encoding="utf-8"))["actions"] == [
        {"frame": 0, "action": "left_move_down", "pressed": True},
        {"frame": 12, "action": "left_move_down", "pressed": False},
    ]


def test_controlled_pong_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    destination = tmp_path / "controlled"
    materialize_controlled_pong(upstream, destination)
    with pytest.raises(FileExistsError):
        materialize_controlled_pong(upstream, destination)


def test_capture_only_qualification_preserves_upstream_and_has_no_patch_contract(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    before = (upstream / "project.godot").read_bytes()
    result = materialize_capture_only_qualification(
        upstream, tmp_path / "qualification", capture_frames=8,
        actions=[{"frame": 0, "action": "left_move_down", "pressed": True}],
    )

    assert (upstream / "project.godot").read_bytes() == before
    assert result.original_main_scene == "res://pong.tscn"
    assert not (upstream / ".flashpatch").exists()
    trace = json.loads(result.trace.read_text(encoding="utf-8"))
    assert trace["capture_frames"] == 8
    assert trace["original_main_scene"] == "res://pong.tscn"
    assert result.visual_candidates == ()
    assert "renderer_hazard_requires_runtime_attribution" not in (result.project / ".flashpatch" / "qualification.gd").read_text(encoding="utf-8")
    assert 'run/main_scene="res://.flashpatch/qualification.tscn"' in (result.project / "project.godot").read_text(encoding="utf-8")


def test_capture_only_probe_does_not_reference_native_main_locals() -> None:
    from flashpatch.public_godot import QUALIFICATION_PROBE_SCRIPT

    assert "current_scene != current_scene" not in QUALIFICATION_PROBE_SCRIPT
    assert "expected_scene" not in QUALIFICATION_PROBE_SCRIPT


def test_declared_scene_transition_requires_complete_one_way_ledger() -> None:
    from flashpatch.public_godot import _verify_declared_scene_transition

    trace = {
        "original_main_scene": "res://menu.tscn",
        "capture_frames": 4,
        "scene_transition": {
            "from_scene": "res://menu.tscn",
            "to_scene": "res://game.tscn",
            "earliest_frame": 1,
            "latest_frame": 2,
        },
    }
    replay = {
        "scene_observations": [
            {"frame_index": 0, "scene_file_path": "res://menu.tscn", "current_scene_node_path": "/root/Menu", "current_scene_instance_id": 1, "wrapper_ancestor_paths": []},
            {"frame_index": 1, "scene_file_path": "res://game.tscn", "current_scene_node_path": "/root/Game", "current_scene_instance_id": 2, "wrapper_ancestor_paths": []},
            {"frame_index": 2, "scene_file_path": "res://game.tscn", "current_scene_node_path": "/root/Game", "current_scene_instance_id": 2, "wrapper_ancestor_paths": []},
            {"frame_index": 3, "scene_file_path": "res://game.tscn", "current_scene_node_path": "/root/Game", "current_scene_instance_id": 2, "wrapper_ancestor_paths": []},
        ],
        "scene_transition_acknowledgement": {"from_scene": "res://menu.tscn", "to_scene": "res://game.tscn", "observed_frame": 1, "status": "APPLIED"},
    }

    _verify_declared_scene_transition(trace, replay)

    replay["scene_observations"][3]["scene_file_path"] = "res://menu.tscn"
    with pytest.raises(RuntimeError, match="did not persist"):
        _verify_declared_scene_transition(trace, replay)


def test_native_main_materializer_binds_declared_transition_target_bytes(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    (upstream / "game.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    qualification = materialize_native_main_capture_qualification(
        upstream, tmp_path / "native-transition", capture_frames=8,
        scene_transition={
            "from_scene": "res://pong.tscn", "to_scene": "res://game.tscn",
            "earliest_frame": 1, "latest_frame": 3,
        },
    )

    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))

    assert trace["scene_transition"]["to_scene"] == "res://game.tscn"
    assert qualification.native_main["source_scenes"]["to_scene"] is not None


def test_native_main_materializer_binds_declared_ui_rect_paths(tmp_path: Path) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-ui", capture_frames=8,
        ui_rect_observations=["/root/Pong/StartButton"],
    )
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    probe = (qualification.project / ".flashpatch" / "native_main_capture.gd").read_text(encoding="utf-8")

    assert trace["ui_rect_observations"] == ["/root/Pong/StartButton"]
    assert "func _record_ui_rect_observations" in probe


def test_ui_rect_preflight_is_metadata_only_and_preserves_native_main_bytes(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    source_scene = (upstream / "pong.tscn").read_bytes()
    preflight = materialize_native_main_ui_rect_preflight(
        upstream,
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )

    trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    script = (preflight.project / ".flashpatch" / "ui_rect_preflight.gd").read_text(encoding="utf-8")
    config = (preflight.project / "project.godot").read_text(encoding="utf-8")
    assert trace == {
        "schema": "flashpatch-native-main-ui-rect-preflight-trace-v1",
        "original_main_scene": "res://pong.tscn",
        "control_paths": ["/root/Pong/StartButton"],
    }
    assert (preflight.project / "pong.tscn").read_bytes() == source_scene
    assert 'run/main_scene="pong.tscn"' in config
    assert 'FlashPatchUiRectPreflight="*res://.flashpatch/ui_rect_preflight.gd"' in config
    assert "get_global_rect()" in script
    assert "as BaseButton" in script
    assert "save_png" not in script
    assert "RenderingServer.frame_post_draw" not in script
    assert "--renderer-capture" not in NativeMainUiRectPreflightRunner.__dict__.get("replay").__code__.co_consts


def test_ui_rect_preflight_runner_sanitizes_headless_import_then_launches_native_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "receipt.json"
    runner = object.__new__(NativeMainUiRectPreflightRunner)
    runner.project = project
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    commands: list[tuple[list[str], object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs.get("env")))
        if "--headless" not in command:
            output.write_text('{"status":"PROBED"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.public_godot.subprocess.run", fake_run)
    result = runner.replay(trace, output)

    assert result == {"status": "PROBED"}
    assert len(commands) == 2
    assert "--headless" in commands[0][0]
    assert "--import" in commands[0][0]
    assert isinstance(commands[0][1], dict)
    assert commands[0][1].get("DISPLAY") is None
    assert commands[0][1].get("WAYLAND_DISPLAY") is None
    assert "--headless" not in commands[1][0]
    assert "--display-driver" in commands[1][0]
    assert "x11" in commands[1][0]


@pytest.mark.parametrize(
    "control_paths",
    [[], ["Pong/StartButton"], ["/root/Pong/../StartButton"], ["/root/Pong:property"], ["/root/Pong/Button", "/root/Pong/Button"]],
)
def test_ui_rect_preflight_rejects_undeclared_or_unsafe_control_paths(
    tmp_path: Path,
    control_paths: list[str],
) -> None:
    with pytest.raises(ValueError, match="Control path|requires declared"):
        materialize_native_main_ui_rect_preflight(
            _upstream(tmp_path),
            tmp_path / "ui-rect",
            control_paths=control_paths,
        )


def test_ui_rect_preflight_rejects_source_drift_before_process_launch(tmp_path: Path) -> None:
    preflight = materialize_native_main_ui_rect_preflight(
        _upstream(tmp_path),
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )
    (preflight.project / "pong.tscn").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source bytes changed"):
        execute_native_main_ui_rect_preflight(preflight, tmp_path / "receipt.json")


def test_ui_rect_preflight_rejects_injected_probe_drift_before_process_launch(tmp_path: Path) -> None:
    preflight = materialize_native_main_ui_rect_preflight(
        _upstream(tmp_path),
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )
    (preflight.project / ".flashpatch" / "ui_rect_preflight.gd").write_text(
        "extends Node\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="source bytes changed"):
        execute_native_main_ui_rect_preflight(preflight, tmp_path / "receipt.json")


def test_ui_rect_preflight_rejects_declared_path_trace_drift_before_process_launch(tmp_path: Path) -> None:
    preflight = materialize_native_main_ui_rect_preflight(
        _upstream(tmp_path),
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )
    trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    trace["control_paths"] = ["/root/Pong/OtherButton"]
    preflight.trace.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source bytes changed"):
        execute_native_main_ui_rect_preflight(preflight, tmp_path / "receipt.json")


def test_ui_rect_preflight_validates_and_binds_metadata_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = materialize_native_main_ui_rect_preflight(
        _upstream(tmp_path),
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return {
            "schema": "flashpatch-native-main-ui-rect-preflight-v1",
            "status": "PROBED",
            "execution_mode": "metadata_only_native_main_ui_rect_preflight",
            "qualification_only": True,
            "scoreable": False,
            "renderer_png_capture": False,
            "detector_executed": False,
            "original_main_scene": "res://pong.tscn",
            "current_scene_node_path": "/root/Pong",
            "wrapper_ancestor_paths": [],
            "observations": [{
                "node_path": "/root/Pong/StartButton",
                "rect": [120.0, 80.0, 64.0, 24.0],
                "visible": True,
                "enabled": True,
                "viewport": [320.0, 180.0],
            }],
        }

    monkeypatch.setattr("flashpatch.public_godot.NativeMainUiRectPreflightRunner.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("flashpatch.public_godot.NativeMainUiRectPreflightRunner.replay", replay)
    receipt = execute_native_main_ui_rect_preflight(preflight, tmp_path / "receipt.json")

    assert receipt["renderer_png_capture"] is False
    assert receipt["detector_executed"] is False
    assert receipt["scoreable"] is False
    assert receipt["observations"][0]["enabled"] is True
    assert receipt["trace_sha256"].startswith("sha256:")
    assert receipt["source_binding"] == preflight.source_binding
    persisted = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert persisted == receipt


def test_ui_rect_preflight_fails_closed_for_non_factual_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = materialize_native_main_ui_rect_preflight(
        _upstream(tmp_path),
        tmp_path / "ui-rect",
        control_paths=["/root/Pong/StartButton"],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return {
            "schema": "flashpatch-native-main-ui-rect-preflight-v1",
            "status": "PROBED",
            "execution_mode": "metadata_only_native_main_ui_rect_preflight",
            "qualification_only": True,
            "scoreable": False,
            "renderer_png_capture": False,
            "detector_executed": False,
            "original_main_scene": "res://pong.tscn",
            "current_scene_node_path": "/root/Pong",
            "wrapper_ancestor_paths": [],
            "observations": [{
                "node_path": "/root/Pong/StartButton",
                "rect": [120.0, 80.0, 0.0, 24.0],
                "visible": True,
                "enabled": True,
                "viewport": [320.0, 180.0],
            }],
        }

    monkeypatch.setattr("flashpatch.public_godot.NativeMainUiRectPreflightRunner.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("flashpatch.public_godot.NativeMainUiRectPreflightRunner.replay", replay)
    with pytest.raises(RuntimeError, match="observation is invalid"):
        execute_native_main_ui_rect_preflight(preflight, tmp_path / "receipt.json")


def test_option_button_popup_preflight_is_autoload_only_and_preserves_main_scene(tmp_path: Path) -> None:
    upstream = _option_button_upstream(tmp_path)
    source_scene = (upstream / "main.tscn").read_bytes()
    preflight = materialize_native_main_option_button_popup_preflight(
        upstream,
        tmp_path / "option-popup",
        option_button_path="/root/Pong/Options",
    )

    trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    script = (preflight.project / ".flashpatch" / "option_button_popup_preflight.gd").read_text(encoding="utf-8")
    config = (preflight.project / "project.godot").read_text(encoding="utf-8")

    assert trace == {
        "schema": "flashpatch-native-main-option-button-popup-preflight-trace-v1",
        "original_main_scene": "res://main.tscn",
        "option_button_path": "/root/Pong/Options",
        "activation": {"kind": "input_event_left_click"},
    }
    assert (preflight.project / "main.tscn").read_bytes() == source_scene
    assert 'run/main_scene="main.tscn"' in config
    assert 'FlashPatchOptionButtonPopupPreflight="*res://.flashpatch/option_button_popup_preflight.gd"' in config
    assert script.count("await get_tree().process_frame") == 2
    assert "as OptionButton" in script
    assert "InputEventMouseButton.new()" in script
    assert "option_button.show_popup()" in script
    assert "save_png" not in script
    assert "analyze(" not in script
    assert "scoreable\": false" in script


def test_option_button_popup_preflight_runner_persists_bounded_stdout_on_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "option-button-popup-preflight.json"
    runner = object.__new__(NativeMainOptionButtonPopupPreflightRunner)
    runner.project = project
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    stdout = "x" * (NativeMainOptionButtonPopupPreflightRunner.MAX_STDOUT_LOG_BYTES + 1)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0 if "--import" in command else 2, stdout)

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.public_godot.subprocess.run", fake_run)

    log_path = output.with_suffix(".stdout.log")
    with pytest.raises(RuntimeError, match=f"stdout log: {log_path}"):
        runner.replay(trace, output)

    assert log_path.is_file()
    assert log_path.stat().st_size <= NativeMainOptionButtonPopupPreflightRunner.MAX_STDOUT_LOG_BYTES
    assert log_path.read_bytes().endswith(b"\n[FlashPatch: stdout truncated]\n")
    assert not output.exists()


def test_option_button_popup_preflight_runner_persists_stdout_when_metadata_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "option-button-popup-preflight.json"
    runner = object.__new__(NativeMainOptionButtonPopupPreflightRunner)
    runner.project = project
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "probe ran but produced no metadata\n")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.public_godot.subprocess.run", fake_run)

    log_path = output.with_suffix(".stdout.log")
    with pytest.raises(RuntimeError, match=f"stdout log: {log_path}"):
        runner.replay(trace, output)

    assert log_path.read_text(encoding="utf-8") == "probe ran but produced no metadata\n"
    assert not output.exists()


def test_option_button_popup_preflight_runner_persists_partial_stdout_on_runtime_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "option-button-popup-preflight.json"
    runner = object.__new__(NativeMainOptionButtonPopupPreflightRunner)
    runner.project = project
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    runtime_error = (
        b"SCRIPT ERROR: Invalid call. Nonexistent function 'is_visible_in_tree' "
        b"in base 'PopupMenu'.\n"
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--import" in command:
            return subprocess.CompletedProcess(command, 0, "imported\n")
        raise subprocess.TimeoutExpired(command, 7, output=runtime_error)

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.public_godot.subprocess.run", fake_run)

    log_path = output.with_suffix(".stdout.log")
    with pytest.raises(
        RuntimeError,
        match=rf"timed out after 7 seconds; see stdout log: {log_path}",
    ):
        runner.replay(trace, output)

    assert log_path.read_bytes() == runtime_error
    assert not output.exists()


@pytest.mark.parametrize("option_button_path", ["", "/root", "/root/Pong/../Options", "/root/Pong:property"])
def test_option_button_popup_preflight_rejects_unsafe_or_undeclared_path(
    tmp_path: Path,
    option_button_path: str,
) -> None:
    with pytest.raises(ValueError, match="OptionButton path|declared /root"):
        materialize_native_main_option_button_popup_preflight(
            _option_button_upstream(tmp_path),
            tmp_path / "option-popup",
            option_button_path=option_button_path,
        )


def test_option_button_popup_probe_rejects_missing_or_non_optionbutton_at_runtime() -> None:
    from flashpatch.public_godot import NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4

    assert "get_node_or_null(NodePath(declared_path)) as OptionButton" in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4
    assert "declared path is missing or not an OptionButton" in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4


def test_option_button_popup_probe_uses_godot4_window_geometry_not_canvasitem_api() -> None:
    from flashpatch.public_godot import NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4

    assert "popup.visible" in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4
    assert "popup.get_global_rect()" not in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4
    assert "popup.is_visible_in_tree()" not in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4
    assert "popup.get_item_rect(" not in NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4


@pytest.mark.parametrize("drift_target", ["project", "main", "injected", "trace"])
def test_option_button_popup_preflight_rejects_any_preexecution_source_drift(
    tmp_path: Path,
    drift_target: str,
) -> None:
    preflight = materialize_native_main_option_button_popup_preflight(
        _option_button_upstream(tmp_path),
        tmp_path / "option-popup",
        option_button_path="/root/Pong/Options",
    )
    targets = {
        "project": preflight.project / "project.godot",
        "main": preflight.project / "main.tscn",
        "injected": preflight.project / ".flashpatch" / "option_button_popup_preflight.gd",
        "trace": preflight.trace,
    }
    target = targets[drift_target]
    if drift_target == "trace":
        trace = json.loads(target.read_text(encoding="utf-8"))
        trace["activation"] = {"kind": "explicit_popup_call"}
        target.write_text(json.dumps(trace), encoding="utf-8")
    else:
        target.write_bytes(target.read_bytes() + b"drift\n")

    with pytest.raises(RuntimeError, match="source bytes changed"):
        execute_native_main_option_button_popup_preflight(preflight, tmp_path / "receipt.json")


def test_option_button_popup_preflight_validates_exact_non_scoreable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = materialize_native_main_option_button_popup_preflight(
        _option_button_upstream(tmp_path),
        tmp_path / "option-popup",
        option_button_path="/root/Pong/Options",
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _popup_metadata_result()

    monkeypatch.setattr("flashpatch.public_godot.NativeMainOptionButtonPopupPreflightRunner.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("flashpatch.public_godot.NativeMainOptionButtonPopupPreflightRunner.replay", replay)
    receipt = execute_native_main_option_button_popup_preflight(preflight, tmp_path / "receipt.json")

    assert receipt["scoreable"] is False
    assert receipt["renderer_png_capture"] is False
    assert receipt["detector_executed"] is False
    assert receipt["activation"]["button_center"] == [152.0, 92.0]
    assert receipt["items"] == _popup_metadata_result()["items"]
    assert receipt["source_binding"] == preflight.source_binding
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{
            "index": 0,
            "text": "Fast",
            "rect": [120.0, 104.0, 64.0, 20.0],
            "visible": False,
            "enabled": True,
            "viewport": [320.0, 180.0],
        }],
    ],
)
def test_option_button_popup_preflight_fails_closed_for_absent_or_inactive_popup_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    items: list[dict[str, object]],
) -> None:
    preflight = materialize_native_main_option_button_popup_preflight(
        _option_button_upstream(tmp_path),
        tmp_path / "option-popup",
        option_button_path="/root/Pong/Options",
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _popup_metadata_result(items=items)

    monkeypatch.setattr("flashpatch.public_godot.NativeMainOptionButtonPopupPreflightRunner.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("flashpatch.public_godot.NativeMainOptionButtonPopupPreflightRunner.replay", replay)
    with pytest.raises(RuntimeError, match="popup items are absent|popup item is invalid"):
        execute_native_main_option_button_popup_preflight(preflight, tmp_path / "receipt.json")


def test_option_button_popup_preflight_allows_only_trace_recorded_explicit_popup_call(tmp_path: Path) -> None:
    upstream = _option_button_upstream(tmp_path)
    preflight = materialize_native_main_option_button_popup_preflight(
        upstream,
        tmp_path / "option-popup",
        option_button_path="/root/Pong/Options",
        activation={"kind": "explicit_popup_call"},
    )

    trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    assert trace["activation"] == {"kind": "explicit_popup_call"}
    with pytest.raises(ValueError, match="activation kind"):
        materialize_native_main_option_button_popup_preflight(
            upstream,
            tmp_path / "invalid-popup",
            option_button_path="/root/Pong/Options",
            activation={"kind": "undocumented_popup_call"},
        )


def test_native_main_materializer_rejects_missing_transition_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="transition target"):
        materialize_native_main_capture_qualification(
            _upstream(tmp_path), tmp_path / "native-transition", capture_frames=8,
            scene_transition={
                "from_scene": "res://pong.tscn", "to_scene": "res://missing.tscn",
                "earliest_frame": 1, "latest_frame": 3,
            },
        )


def test_native_main_transition_target_drift_is_rejected_before_replay(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    (upstream / "game.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    qualification = materialize_native_main_capture_qualification(
        upstream, tmp_path / "native-transition", capture_frames=8,
        scene_transition={
            "from_scene": "res://pong.tscn", "to_scene": "res://game.tscn",
            "earliest_frame": 1, "latest_frame": 3,
        },
    )
    (qualification.project / "game.tscn").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source tree bytes changed|transition target"):
        from flashpatch.public_godot import _require_unchanged_native_main
        _require_unchanged_native_main(qualification)


def test_capture_only_qualification_materializes_godot3_probe(tmp_path: Path) -> None:
    result = materialize_capture_only_qualification(_godot3_upstream(tmp_path), tmp_path / "qualification", capture_frames=8)

    probe = (result.project / ".flashpatch" / "qualification.gd").read_text(encoding="utf-8")
    scene = (result.project / ".flashpatch" / "qualification.tscn").read_text(encoding="utf-8")
    assert "VisualServer" in probe
    assert "OS.get_ticks_usec" in probe
    assert "format=2" in scene


def test_native_main_qualification_keeps_configured_main_scene_bytes_and_adds_only_autoload(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    qualification = materialize_native_main_capture_qualification(
        upstream,
        tmp_path / "native-main",
        capture_frames=8,
        actions=[
            {"frame": 1, "action": "left", "pressed": True},
            {"frame": 2, "action": "unknown", "pressed": False},
        ],
    )

    assert qualification.original_main_scene == "res://pong.tscn"
    assert qualification.native_main["run_main_scene_unchanged"] is True
    assert qualification.native_main["main_scene_bytes_match"] is True
    assert qualification.native_main["native_equivalence"] == "NOT_ESTABLISHED"
    config = (qualification.project / "project.godot").read_text(encoding="utf-8")
    assert 'run/main_scene="pong.tscn"' in config
    assert 'FlashPatchNativeMainCapture="*res://.flashpatch/native_main_capture.gd"' in config
    assert (qualification.project / "pong.tscn").read_bytes() == (upstream / "pong.tscn").read_bytes()
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    assert trace["actions"] == [
        {"frame": 1, "action": "left", "pressed": True},
        {"frame": 2, "action": "unknown", "pressed": False},
    ]
    assert trace["warmup_frames"] == 0
    collector = (qualification.project / ".flashpatch" / "native_main_capture.gd").read_text(encoding="utf-8")
    assert "await get_tree().process_frame" in collector
    assert "get_tree().current_scene" in collector
    assert "current_scene.scene_file_path" in collector
    assert "no_wrapper_ancestor" in collector
    assert "InputMap.has_action(name)" in collector
    assert '"scene_observations": _scene_observations' in collector
    assert "func _record_scene_observation" in collector
    assert "func _scene_transition_acknowledgement" in collector
    assert '"status": "MISSING_INPUT_MAP_ACTION"' in collector
    assert "await get_tree().process_frame" in collector
    assert "RenderingServer.force_draw()" in collector
    assert "actual_capture_timestamps_us" in collector


@pytest.mark.parametrize("warmup_frames", [-1, True, 1.5])
def test_native_main_materializer_rejects_invalid_warmup_frames(
    tmp_path: Path, warmup_frames: object
) -> None:
    with pytest.raises(ValueError, match="warmup_frames"):
        materialize_native_main_capture_qualification(
            _upstream(tmp_path),
            tmp_path / "native-main",
            capture_frames=4,
            warmup_frames=warmup_frames,  # type: ignore[arg-type]
        )


def test_native_main_warmup_applies_absolute_pointer_and_action_before_rgb_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path),
        tmp_path / "native-main",
        capture_frames=4,
        warmup_frames=3,
        actions=[{"frame": 1, "action": "left", "pressed": True}],
        pointer_events=[{"frame": 2, "kind": "left_click", "x": 0.5, "y": 0.5}],
    )
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    collector = (qualification.project / ".flashpatch" / "native_main_capture.gd").read_text(encoding="utf-8")

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert trace["warmup_frames"] == 3
    assert trace["actions"][0]["frame"] == 1
    assert trace["pointer_events"][0]["frame"] == 2
    assert receipt["action_acknowledgements"][0]["status"] == "APPLIED"
    assert receipt["pointer_acknowledgements"][0]["status"] == "APPLIED"
    assert receipt["frame_count"] == 4
    assert "for tick in range(_warmup_frames + int(_trace[\"capture_frames\"]))" in collector
    assert "if tick < _warmup_frames:" in collector


def test_native_main_warmup_binds_absolute_presentation_timestamp_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=4, warmup_frames=3,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["warmup_frames"] == 3
    assert receipt["capture_trace_frame_indices"] == [3, 4, 5, 6]
    assert receipt["presentation_timestamps_us"] == [50_000, 66_666, 83_333, 100_000]


def test_native_main_zero_warmup_preserves_capture_zero_presentation_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=4,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["warmup_frames"] == 0
    assert receipt["capture_trace_frame_indices"] == [0, 1, 2, 3]
    assert receipt["presentation_timestamps_us"] == [0, 16_666, 33_333, 50_000]


def test_native_main_classifier_rejects_warmup_timestamp_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=4, warmup_frames=3,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(
            qualification, output, timestamps_us=[0, 16_666, 33_333, 50_000],
        )

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="absolute fixed media time"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_classifier_emits_only_non_scoreable_non_equivalent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path),
        tmp_path / "native-main",
        capture_frames=8,
        actions=[
            {"frame": 1, "action": "left", "pressed": True},
            {"frame": 2, "action": "unknown", "pressed": False},
        ],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["schema"] == "flashpatch-godot-native-main-capture-v1"
    assert receipt["execution_mode"] == "instrumented_native_main_scene_capture"
    assert receipt["qualification_only"] is True
    assert receipt["scoreable"] is False
    assert receipt["native_equivalence"] == "NOT_ESTABLISHED"
    assert receipt["native_main"]["current_scene_node_path"] == "/root/Pong"
    assert [item["status"] for item in receipt["action_acknowledgements"]] == ["APPLIED", "APPLIED"]
    persisted = json.loads(Path(str(receipt["receipt"])).read_text(encoding="utf-8"))
    assert persisted["scoreable"] is False
    assert persisted["native_equivalence"] == "NOT_ESTABLISHED"


def test_native_main_classifier_rejects_unapplied_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path),
        tmp_path / "native-main",
        capture_frames=8,
        actions=[{"frame": 1, "action": "left", "pressed": True}],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["action_acknowledgements"][0]["status"] = "MISSING_INPUT_MAP_ACTION"
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="action was not applied"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_trace_binds_launch_pointer_and_scenario_readiness(tmp_path: Path) -> None:
    readiness = {
        "required_node_paths": ["/root/Pong"],
        "required_group_minimums": {"player_actor": 1},
        "required_visible": [{"node_path": "/root/Pong", "visible": True}],
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path),
        tmp_path / "native-main",
        capture_frames=8,
        launch_arguments=["--facility-regression"],
        pointer_events=[{"frame": 1, "kind": "left_click", "x": 0.5, "y": 0.5}],
        key_events=[{"frame": 2, "key": "down"}, {"frame": 3, "key": "enter"}],
        scenario_readiness=readiness,
    )

    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    assert trace["launch_arguments"] == ["--facility-regression"]
    assert trace["pointer_events"] == [{"frame": 1, "kind": "left_click", "x": 0.5, "y": 0.5}]
    assert trace["key_events"] == [{"frame": 2, "key": "down"}, {"frame": 3, "key": "enter"}]
    assert trace["scenario_readiness"] == readiness
    collector = (qualification.project / ".flashpatch" / "native_main_capture.gd").read_text(encoding="utf-8")
    assert "InputEventMouseButton.new()" in collector
    assert "InputEventKey.new()" in collector
    assert "Input.parse_input_event(press)" in collector
    assert "func _scenario_readiness()" in collector


def test_native_main_classifier_accepts_godot_serialized_pointer_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        pointer_events=[{"frame": 1, "kind": "left_click", "x": 0.5, "y": 17.0 / 18.0}],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["pointer_acknowledgements"][0]["y"] = float("0.944444444444444")
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["pointer_acknowledgements"][0]["status"] == "APPLIED"


def test_native_main_classifier_rejects_materially_different_pointer_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        pointer_events=[{"frame": 1, "kind": "left_click", "x": 0.5, "y": 17.0 / 18.0}],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["pointer_acknowledgements"][0]["y"] = 0.9
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="pointer acknowledgement"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_runtime_observations_are_frame_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {
        "node_path": "/root/Pong",
        "property": "flash_intensity",
        "script_path": "res://logic/pong.gd",
        "resource_path": "res://pong.tscn",
        "source_line": 3,
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        runtime_observations=[observation],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert len(receipt["runtime_events"]) == 8
    assert receipt["runtime_events"][0]["node_path"] == "/root/Pong"
    assert receipt["runtime_events"][0]["factual_value"] == 1.0


def test_native_main_shader_runtime_observations_are_frame_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {
        "kind": "shader_parameter",
        "node_path": "/root/Pong/Flash",
        "property": "flash_intensity",
        "source_path": "res://pong.tscn",
        "resource_path": "res://pong.tscn",
        "source_line": 7,
        "shader_path": "res://effects/flash.gdshader",
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        runtime_observations=[observation],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert len(receipt["runtime_events"]) == 8
    assert receipt["runtime_events"][0] == {
        "frame_index": 0,
        "node_path": "/root/Pong/Flash",
        "source_path": "res://pong.tscn",
        "resource_path": "res://pong.tscn",
        "source_line": 7,
        "shader_path": "res://effects/flash.gdshader",
        "property": "flash_intensity",
        "factual_value": 1.0,
        "event_kind": "shader_parameter",
    }


@pytest.mark.parametrize("observation", [
    {
        "kind": "shader_parameter", "node_path": "/root/Pong/Flash", "property": "flash_intensity",
        "source_path": "res://pong.gd", "resource_path": "res://pong.tscn", "source_line": 7,
        "shader_path": "res://effects/flash.gdshader",
    },
    {
        "kind": "shader_parameter", "node_path": "/root/Pong/Flash", "property": "flash_intensity",
        "source_path": "res://pong.tscn", "resource_path": "res://pong.tscn", "source_line": 7,
        "shader_path": "res://effects/flash.gdshader", "script_path": "res://logic/pong.gd",
    },
])
def test_native_main_materializer_rejects_invalid_shader_runtime_observation(tmp_path: Path, observation: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="shader runtime observation"):
        materialize_native_main_capture_qualification(
            _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
            runtime_observations=[observation],
        )


def test_native_main_classifier_rejects_shader_runtime_event_with_render_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observation = {
        "kind": "shader_parameter", "node_path": "/root/Pong/Flash", "property": "flash_intensity",
        "source_path": "res://pong.tscn", "resource_path": "res://pong.tscn", "source_line": 7,
        "shader_path": "res://effects/flash.gdshader",
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        runtime_observations=[observation],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["runtime_events"][0]["script_path"] = "res://logic/pong.gd"
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="shader runtime observation"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


@pytest.mark.parametrize(
    ("launch_arguments", "pointer_events", "scenario_readiness"),
    [
        (["--output"], [], None),
        ([], [{"frame": 1, "kind": "right_click", "x": 0.5, "y": 0.5}], None),
        ([], [], {"required_node_paths": ["relative"], "required_group_minimums": {"guard_actor": 1}, "required_visible": []}),
    ],
)
def test_native_main_materializer_rejects_unsafe_launch_pointer_or_readiness(
    tmp_path: Path,
    launch_arguments: list[str],
    pointer_events: list[dict[str, object]],
    scenario_readiness: dict[str, object] | None,
) -> None:
    with pytest.raises(ValueError):
        materialize_native_main_capture_qualification(
            _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
            launch_arguments=launch_arguments,
            pointer_events=pointer_events,
            scenario_readiness=scenario_readiness,
        )


@pytest.mark.parametrize("key_events", [[{"frame": 1, "key": "left"}], [{"frame": 8, "key": "enter"}]])
def test_native_main_materializer_rejects_invalid_key_events(tmp_path: Path, key_events: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="key event"):
        materialize_native_main_capture_qualification(
            _upstream(tmp_path), tmp_path / "native-main", capture_frames=8, key_events=key_events,
        )


def test_native_main_classifier_rejects_unapplied_key_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        key_events=[{"frame": 1, "key": "enter"}],
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["key_acknowledgements"][0]["status"] = "INVALID_KEY_EVENT"
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="key acknowledgement"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_classifier_requires_declared_readiness_before_safe_scenario_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = {
        "required_node_paths": ["/root/Pong"],
        "required_group_minimums": {"player_actor": 1},
        "required_visible": [{"node_path": "/root/Pong", "visible": True}],
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        pointer_events=[{"frame": 1, "kind": "left_click", "x": 0.5, "y": 0.5}],
        scenario_readiness=readiness,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["decision"] == "SAFE_SCENARIO_READY"
    assert receipt["scenario_readiness"]["satisfied"] is True


def test_native_main_ui_only_readiness_allows_no_gameplay_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = {
        "required_node_paths": ["/root/Pong"],
        "required_group_minimums": {},
        "required_visible": [{"node_path": "/root/Pong", "visible": True}],
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        scenario_readiness=readiness,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output)

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    receipt = classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")

    assert receipt["decision"] == "SAFE_SCENARIO_READY"
    assert receipt["scenario_readiness"]["observed_group_counts"] == {}


def test_native_main_classifier_requires_declared_option_selection_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = {
        "required_node_paths": ["/root/Pong"],
        "required_group_minimums": {},
        "required_visible": [{"node_path": "/root/Pong", "visible": True}],
        "required_option_selection": [{
            "node_path": "/root/Pong",
            "selected_index": 11,
            "selected_text": "FX: OldFilm",
        }],
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
        ui_selection_observations=["/root/Pong"], scenario_readiness=readiness,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["ui_selection_signal_events"] = []
        return payload

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="selection signal"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_classifier_rejects_post_selection_state_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = {
        "required_node_paths": ["/root/Pong"],
        "required_group_minimums": {},
        "required_visible": [{"node_path": "/root/Pong", "visible": True}],
        "required_option_selection": [{
            "node_path": "/root/Pong",
            "selected_index": 11,
            "selected_text": "FX: OldFilm",
        }],
    }
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path),
        tmp_path / "native-main",
        capture_frames=8,
        ui_selection_observations=["/root/Pong"],
        scenario_readiness=readiness,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        payload["scenario_readiness"]["selection_mismatches"] = [{
            "node_path": "/root/Pong",
            "expected_index": 11,
            "observed_index": 10,
        }]
        return payload

    monkeypatch.setattr(
        "flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay
    )
    with pytest.raises(RuntimeError, match="scenario readiness"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


@pytest.mark.parametrize("mutation", ["run_main_scene", "main_scene_bytes"])
def test_native_main_classifier_rejects_changed_main_scene(
    tmp_path: Path, mutation: str
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
    )
    if mutation == "run_main_scene":
        config = qualification.project / "project.godot"
        config.write_text(config.read_text(encoding="utf-8").replace('run/main_scene="pong.tscn"', 'run/main_scene="other.tscn"'), encoding="utf-8")
    else:
        (qualification.project / "pong.tscn").write_text("[node name=\"Changed\" type=\"Node\"]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_main_classifier_rejects_wrong_runtime_current_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(qualification, output, runtime_scene="res://wrapper.tscn")

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="current_scene or wrapper topology"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


@pytest.mark.parametrize("failure", ["missing_frame", "missing_timestamps"])
def test_native_main_classifier_rejects_missing_frames_or_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        if failure == "missing_frame":
            return _native_main_replay(
                qualification, output, frames=np.zeros((7, 4, 4, 3), dtype=np.uint8),
            )
        return _native_main_replay(qualification, output, timestamps_us=[])

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="timestamps|frames"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


@pytest.mark.parametrize(
    ("scoreable", "native_equivalence"),
    [(True, "NOT_ESTABLISHED"), (False, "ESTABLISHED")],
)
def test_native_main_classifier_rejects_scoreable_or_equivalent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoreable: bool,
    native_equivalence: str,
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _upstream(tmp_path), tmp_path / "native-main", capture_frames=8,
    )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        return _native_main_replay(
            qualification,
            output,
            scoreable=scoreable,
            native_equivalence=native_equivalence,
        )

    monkeypatch.setattr("flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay)
    with pytest.raises(RuntimeError, match="scoreability or equivalence"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_capture_only_qualification_records_machine_checked_clean_pinned_source(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    repository = "https://github.com/example/open-rts"
    revision = _pin_clean_repository(upstream, repository)

    qualification = materialize_capture_only_qualification(
        upstream, tmp_path / "qualification", repository=repository, revision=revision,
    )

    assert qualification.source_provenance is not None
    assert qualification.source_provenance["status"] == "VERIFIED_CLEAN_PINNED_SOURCE"
    assert qualification.source_provenance["repository"] == repository
    assert qualification.source_provenance["revision"] == revision
    assert qualification.source_provenance["source_tree_sha256"].startswith("sha256:")
    assert len(qualification.source_provenance["source_tree_sha256"]) == 71


def test_capture_only_qualification_rejects_dirty_or_mismatched_pinned_source(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    repository = "https://github.com/example/open-rts"
    revision = _pin_clean_repository(upstream, repository)
    (upstream / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkout is dirty"):
        materialize_capture_only_qualification(
            upstream, tmp_path / "dirty", repository=repository, revision=revision,
        )
    (upstream / "untracked.txt").unlink()
    with pytest.raises(ValueError, match="repository does not match"):
        materialize_capture_only_qualification(
            upstream, tmp_path / "wrong-remote", repository="https://github.com/example/other", revision=revision,
        )
    with pytest.raises(ValueError, match="revision does not match"):
        materialize_capture_only_qualification(
            upstream, tmp_path / "wrong-revision", repository=repository, revision="0" * 40,
        )


def test_clean_pinned_source_hash_does_not_follow_tracked_symlink_target(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    outside = tmp_path / "outside.gd"
    outside.write_text("first external content\n", encoding="utf-8")
    (upstream / "external.gd").symlink_to(outside)
    repository = "https://github.com/example/open-rts"
    revision = _pin_clean_repository(upstream, repository)

    first = materialize_capture_only_qualification(
        upstream, tmp_path / "first", repository=repository, revision=revision,
    ).source_provenance
    outside.write_text("changed external content\n", encoding="utf-8")
    second = materialize_capture_only_qualification(
        upstream, tmp_path / "second", repository=repository, revision=revision,
    ).source_provenance

    assert first is not None
    assert second is not None
    assert first["source_tree_sha256"] == second["source_tree_sha256"]


def test_capture_only_qualification_inventories_only_project_local_godot4_visual_exports(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    effects = upstream / "effects"
    effects.mkdir()
    (effects / "flash.gd").write_text(
        "@export var flash_intensity: float = 1.0\n"
        "@export var blink_frequency = 3.0\n"
        "@export var player_speed: float = 2.0\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.gd"
    outside.write_text("@export var flash_intensity: float = 1.0\n", encoding="utf-8")
    (effects / "escaped.gd").symlink_to(outside)

    qualification = materialize_capture_only_qualification(upstream, tmp_path / "qualification")

    assert qualification.visual_candidates == (
        {"source": "effects/flash.gd", "line": 1, "parameter": "flash_intensity", "declared_type": "float"},
        {"source": "effects/flash.gd", "line": 2, "parameter": "blink_frequency", "declared_type": None},
    )


def test_capture_only_qualification_inventories_godot3_export_type(tmp_path: Path) -> None:
    upstream = _godot3_upstream(tmp_path)
    (upstream / "effect.gd").write_text(
        "export(float) var flash_intensity = 1.0\n"
        "export var blink_frequency = 3.0\n",
        encoding="utf-8",
    )

    qualification = materialize_capture_only_qualification(upstream, tmp_path / "qualification")

    assert qualification.visual_candidates == (
        {"source": "effect.gd", "line": 1, "parameter": "flash_intensity", "declared_type": "float"},
        {"source": "effect.gd", "line": 2, "parameter": "blink_frequency", "declared_type": None},
    )


def test_capture_only_qualification_never_promotes_hazard_to_patch_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = materialize_capture_only_qualification(_upstream(tmp_path), tmp_path / "qualification", capture_frames=8)

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
        frames[1::2] = 255
        artifact = output.with_name("renderer-frames.npz")
        np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
        output.write_text("{}", encoding="utf-8")
        return {"frames_npz": artifact.name}

    monkeypatch.setattr("flashpatch.public_godot.GodotRendererReplayRunner.replay", replay)
    receipt = classify_capture_only_qualification(qualification, tmp_path / "replay.json")

    assert receipt["decision"] == "HAZARDOUS_PATCH_INELIGIBLE"
    assert receipt["qualification_only"] is True
    assert receipt["hazard_frame_indices"]
    assert json.loads(Path(str(receipt["receipt"])).read_text(encoding="utf-8"))["decision"] == "HAZARDOUS_PATCH_INELIGIBLE"


def test_capture_only_hazard_with_visual_export_remains_attribution_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _upstream(tmp_path)
    (upstream / "effect.gd").write_text("@export var flash_intensity: float = 1.0\n", encoding="utf-8")
    qualification = materialize_capture_only_qualification(upstream, tmp_path / "qualification", capture_frames=8)

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
        frames[1::2] = 255
        artifact = output.with_name("renderer-frames.npz")
        np.savez_compressed(artifact, frames=frames, timestamps=np.arange(8, dtype=np.float64) / 10.0)
        output.write_text("{}", encoding="utf-8")
        return {"frames_npz": artifact.name}

    monkeypatch.setattr("flashpatch.public_godot.GodotRendererReplayRunner.replay", replay)
    receipt = classify_capture_only_qualification(qualification, tmp_path / "replay.json")

    assert receipt["decision"] == "HAZARDOUS_ATTRIBUTION_PENDING"
    assert receipt["scoreable"] is False
    assert receipt["visual_candidates"] == [
        {"source": "effect.gd", "line": 1, "parameter": "flash_intensity", "declared_type": "float"}
    ]


def test_capture_only_safe_result_stays_safe_even_with_visual_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _upstream(tmp_path)
    (upstream / "effect.gd").write_text("@export var flash_intensity: float = 1.0\n", encoding="utf-8")
    qualification = materialize_capture_only_qualification(upstream, tmp_path / "qualification", capture_frames=8)

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        artifact = output.with_name("renderer-frames.npz")
        np.savez_compressed(artifact, frames=np.zeros((8, 4, 4, 3), dtype=np.uint8), timestamps=np.arange(8, dtype=np.float64) / 10.0)
        output.write_text("{}", encoding="utf-8")
        return {"frames_npz": artifact.name}

    monkeypatch.setattr("flashpatch.public_godot.GodotRendererReplayRunner.replay", replay)
    receipt = classify_capture_only_qualification(qualification, tmp_path / "replay.json")

    assert receipt["decision"] == "SAFE"
    assert receipt["scoreable"] is False


def test_capture_only_qualification_requires_three_identical_process_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _upstream(tmp_path)

    def classify(qualification, output, *, godot_binary=None):
        output.write_text("{}", encoding="utf-8")
        receipt = output.with_name("qualification-receipt.json")
        receipt.write_text("{}", encoding="utf-8")
        return {
            "decision": "SAFE", "frame_artifact_sha256": "a" * 64,
            "trace_sha256": "sha256:" + "b" * 64, "receipt": str(receipt),
        }

    monkeypatch.setattr("flashpatch.public_godot.classify_capture_only_qualification", classify)
    result = execute_repeated_capture_only_qualification(upstream, tmp_path / "repeat")

    assert result["status"] == "PROCESS_REPRODUCIBLE"
    assert result["scoreable"] is False
    assert len(result["runs"]) == 3
    assert json.loads(Path(str(result["receipt"])).read_text(encoding="utf-8"))["status"] == "PROCESS_REPRODUCIBLE"


def test_controlled_sparta_keeps_upstream_clean_and_declares_dynamic_binding(tmp_path: Path) -> None:
    upstream = _sparta_upstream(tmp_path)
    result = materialize_controlled_sparta(upstream, tmp_path / "controlled-sparta")

    assert "@export var flashpatch_intensity" not in (upstream / "scripts" / "RoutShockwave.gd").read_text(encoding="utf-8")
    assert result.source["source_revision"] == SPARTA_REVISION
    script = result.mutation_script.read_text(encoding="utf-8")
    assert "@export var flashpatch_intensity: float = 1.0" in script
    recorder = (result.project / "tools" / "demo" / "DemoInputRecorder.gd").read_text(encoding="utf-8")
    assert "RoutShockwave.spawn(_battle, Vector2(800.0, 480.0), 1400.0" in recorder
    assert "func _flashpatch_record_capture" in recorder
    assert '"source_line": _flashpatch_exported_line(source_text, "flashpatch_intensity")' in recorder
    assert '"script_path": script_path' in recorder
    assert '"resource_path": resource_path' in recorder
    assert '"actual_capture_timestamp_us": actual_us' in recorder
    assert '"presentation_timestamp_us": int(tick * 1000000 / fixed_fps)' in recorder
    assert '"viewport_use_hdr_2d": get_viewport().use_hdr_2d' in recorder
    assert '"display_server": DisplayServer.get_name()' in recorder
    assert '"rendering_method": RenderingServer.get_current_rendering_method()' in recorder
    assert '"rendering_driver": RenderingServer.get_current_rendering_driver_name()' in recorder
    assert '"hdr_output_enabled": DisplayServer.window_is_hdr_output_enabled()' in recorder
    assert 'OS.get_environment("FLASHPATCH_ACTION_EVENTS")' in recorder
    assert '"observation": "runtime_node_present_at_capture"' in recorder
    assert "img.convert(Image.FORMAT_RGB8)" in recorder
    assert "flashpatch_source_pixel_format := img.get_format()" in recorder
    assert '"source_line": 4' not in recorder
    assert "FileAccess.READ_WRITE if FileAccess.file_exists(path) else FileAccess.WRITE_READ" in recorder
    contract = json.loads(result.contract.read_text(encoding="utf-8"))
    assert contract["patch_candidates"][0]["runtime_binding"] == "dynamic"
    assert contract["patch_candidates"][0]["runtime_resource"] == "scenes/Battle.tscn"


def test_sparta_runtime_evidence_helpers_fail_closed_for_absent_or_malformed_data(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"frame_index": 0}\nnot-json\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed"):
        SpartaRendererRunner._read_jsonl(missing, "capture metadata")
    with pytest.raises(RuntimeError, match="malformed"):
        SpartaRendererRunner._read_jsonl(malformed, "runtime event")


def test_sparta_action_acknowledgement_requires_matching_runtime_observation() -> None:
    trace_actions = [{"frame": 0, "action": "controlled_shockwave"}]
    runtime_events = [
        {
            "frame_index": 0,
            "node_path": "/root/DemoInputRecorder/Battle/RoutShockwave",
            "property": "flashpatch_intensity",
            "factual_value": 1.0,
        }
    ]
    action_events = [
        {
            "frame": 0,
            "action": "controlled_shockwave",
            "status": "APPLIED",
            "observation": "runtime_node_present_at_capture",
            "node_path": "/root/DemoInputRecorder/Battle/RoutShockwave",
            "property": "flashpatch_intensity",
            "factual_value": 1.0,
        }
    ]

    assert SpartaRendererRunner._action_acknowledgements(
        trace_actions, action_events, runtime_events
    ) == [{"frame": 0, "status": "APPLIED"}]

    action_events[0]["node_path"] = "/root/wrong"
    with pytest.raises(RuntimeError, match="lacks runtime evidence"):
        SpartaRendererRunner._action_acknowledgements(
            trace_actions, action_events, runtime_events
        )


def test_sparta_state_preservation_hashes_actual_contiguous_artifacts(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "hash_stream.jsonl").write_text(
        "".join(
            json.dumps({"tick": tick, "cheap": f"hash-{tick}"}) + "\n"
            for tick in range(5)
        ),
        encoding="utf-8",
    )
    (state / "state_00003.json").write_text(
        json.dumps({"units": [{"id": 1}], "tick": 3}, indent=2) + "\n",
        encoding="utf-8",
    )

    evidence = SpartaRendererRunner._state_preservation_evidence(state, 4)

    assert evidence["state_stream_tick_domain"] == [0, 4]
    assert evidence["state_stream_record_count"] == 5
    assert evidence["state_stream_artifact"] == str(state / "hash_stream.jsonl")
    assert evidence["final_state_artifact"] == str(state / "state_00003.json")
    assert evidence["state_stream_sha256"]
    assert evidence["final_state_sha256"]
    assert evidence["final_state_raw_sha256"]

    (state / "hash_stream.jsonl").write_text(
        '{"tick":0}\n{"tick":2}\n{"tick":3}\n{"tick":4}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not contiguous"):
        SpartaRendererRunner._state_preservation_evidence(state, 4)


def test_sparta_source_line_is_measured_from_the_executed_source(tmp_path: Path) -> None:
    source = tmp_path / "RoutShockwave.gd"
    source.write_text(
        "class_name RoutShockwave\nextends Node2D\n\n# inserted comment\n"
        "@export var flashpatch_intensity: float = 1.0\n",
        encoding="utf-8",
    )

    assert SpartaRendererRunner._exported_parameter_line(
        source, "flashpatch_intensity"
    ) == 5

    source.write_text(
        source.read_text(encoding="utf-8")
        + "@export var flashpatch_intensity: float = 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unique exported parameter line"):
        SpartaRendererRunner._exported_parameter_line(
            source, "flashpatch_intensity"
        )


def test_sparta_renderer_adapter_fails_closed_when_preparation_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"actions": [{"frame": 0}]}), encoding="utf-8")
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\n", encoding="utf-8")
    godot.chmod(0o755)

    runner = SpartaRendererRunner(project, godot_binary=godot, timeout_seconds=1)
    monkeypatch.setattr(
        runner,
        "_run_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("timed out")),
    )
    with pytest.raises(RuntimeError, match="preparation timed out"):
        runner.replay(trace, tmp_path / "out" / "replay.json")
    assert (project / ".flashpatch-runtime-trace.json").read_bytes() == trace.read_bytes()


def test_controlled_sparta_persists_a_labelled_engine_receipt(tmp_path: Path) -> None:
    upstream = _sparta_upstream(tmp_path)
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\n", encoding="utf-8")
    godot.chmod(0o755)

    class SafeRenderer:
        def __init__(self, project: Path) -> None:
            self.project = project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(4, dtype=np.float64) / 60.0)
            payload = {
                "status": "REPLAYED",
                "frames_npz": artifact.name,
                "renderer_capture": {
                    "trace_sha256": "sha256:" + __import__("hashlib").sha256(trace.read_bytes()).hexdigest(),
                    "godot_version": "fake",
                    "renderer_configuration": {"display_driver": "x11", "rendering_driver": "opengl3"},
                },
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

    run = execute_controlled_sparta(
        upstream,
        tmp_path / "run",
        godot_binary=godot,
        runner_factory=SafeRenderer,
    )
    saved = json.loads(run.receipt_path.read_text(encoding="utf-8"))
    assert run.receipt["verdict"] == "SAFE"
    assert saved["controlled_mutation"] is True
    assert saved["upstream"]["upstream_defect"] is False
    assert saved["upstream"]["source_revision"] == SPARTA_REVISION


def test_controlled_run_persists_engine_receipt_and_controlled_label(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)

    class SafeRenderer:
        def __init__(self, project: Path) -> None:
            self.project = project

        def replay(self, trace: Path, output: Path) -> dict[str, object]:
            frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
            artifact = output.with_name("frames.npz")
            np.savez_compressed(artifact, frames=frames, timestamps=np.arange(4, dtype=np.float64) / 60.0)
            payload = {
                "status": "REPLAYED", "frames_npz": artifact.name,
                "renderer_capture": {"trace_sha256": "sha256:" + __import__("hashlib").sha256(trace.read_bytes()).hexdigest(), "godot_version": "fake", "renderer_configuration": {"display_driver": "x11", "rendering_driver": "opengl3"}},
                "action_frames": [0, 12], "gameplay_state": "stable",
                "semantic_invariants": {"terminal_completion": True, "terminal_state": "stable", "player_world_digest": "stable", "score": "score_not_applicable"},
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            return payload

    run = execute_controlled_pong(upstream, tmp_path / "run", runner_factory=SafeRenderer)
    saved = json.loads(run.receipt_path.read_text(encoding="utf-8"))
    manifest_path = run.receipt_path.parent / saved["artifact_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert run.receipt["verdict"] == "SAFE"
    assert saved["controlled_mutation"] is True
    assert saved["upstream"]["upstream_defect"] is False
    assert saved["upstream"]["source_revision"] == GODOT_DEMO_REVISION
    assert saved["artifact_manifest"]["sha256"] == __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["controlled_mutation"] is True
    assert all(len(item["sha256"]) == 64 and item["bytes"] >= 0 for item in manifest["artifacts"])


def test_native_main_candidate_tree_binds_factual_and_one_declared_tscn_token(tmp_path: Path) -> None:
    qualification = materialize_native_main_capture_qualification(
        _shader_tscn_upstream(tmp_path), tmp_path / "baseline", capture_frames=4
    )
    factual = tmp_path / "factual"
    candidate = tmp_path / "candidate"
    shutil.copytree(qualification.project, factual)
    shutil.copytree(qualification.project, candidate)
    factual_receipt = _verify_native_main_candidate_tree(qualification, factual)
    assert factual_receipt["mode"] == "FACTUAL_BYTE_EXACT"
    assert factual_receipt["changed_file_count"] == 0
    target = candidate / "main.tscn"
    target.write_bytes(target.read_bytes().replace(b"1.250", b"0.125"))
    receipt = _verify_native_main_candidate_tree(
        qualification,
        candidate,
        patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 4),
    )
    assert receipt["mode"] == "CANDIDATE_ONE_TSCN_NUMERIC_TOKEN"
    assert receipt["source"]["change_kind"] == "ONE_FINITE_NUMERIC_TOKEN"
    assert "1.250" not in json.dumps(receipt)
    assert "0.125" not in json.dumps(receipt)


def test_native_main_candidate_tree_rejects_additional_source_mutation(tmp_path: Path) -> None:
    qualification = materialize_native_main_capture_qualification(
        _shader_tscn_upstream(tmp_path), tmp_path / "baseline", capture_frames=4
    )
    candidate = tmp_path / "candidate"
    shutil.copytree(qualification.project, candidate)
    target = candidate / "main.tscn"
    target.write_bytes(target.read_bytes().replace(b"1.250", b"0.125").replace(b"Node2D", b"Node"))
    with pytest.raises(RuntimeError, match="outside the declared numeric token"):
        _verify_native_main_candidate_tree(
            qualification, candidate, patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 4)
        )


def test_native_main_candidate_tree_rejects_flashpatch_mutation_and_extra_file(tmp_path: Path) -> None:
    qualification = materialize_native_main_capture_qualification(
        _shader_tscn_upstream(tmp_path), tmp_path / "baseline", capture_frames=4
    )
    mutated = tmp_path / "flashpatch-mutated"
    shutil.copytree(qualification.project, mutated)
    (mutated / ".flashpatch" / "native-main-trace.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="factual project tree bytes changed"):
        _verify_native_main_candidate_tree(qualification, mutated)
    extra = tmp_path / "extra"
    shutil.copytree(qualification.project, extra)
    (extra / "surprise.gd").write_text("extends Node\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or extra files"):
        _verify_native_main_candidate_tree(qualification, extra)


def test_native_main_candidate_tree_rejects_symlink(tmp_path: Path) -> None:
    qualification = materialize_native_main_capture_qualification(
        _shader_tscn_upstream(tmp_path), tmp_path / "baseline", capture_frames=4
    )
    candidate = tmp_path / "candidate"
    shutil.copytree(qualification.project, candidate)
    (candidate / "logic.gd").unlink()
    os.symlink("main.tscn", candidate / "logic.gd")
    with pytest.raises(RuntimeError, match="contains symlink"):
        _verify_native_main_candidate_tree(qualification, candidate)


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        (b"not_a_number", "not one exact numeric assignment"),
        (b"0.125\r\nshader_parameter/flash_intensity = 0.250", "missing, redundant, or at the wrong source line"),
    ],
)
def test_native_main_candidate_tree_rejects_non_numeric_or_redundant_token(
    tmp_path: Path, replacement: bytes, error: str
) -> None:
    qualification = materialize_native_main_capture_qualification(
        _shader_tscn_upstream(tmp_path), tmp_path / "baseline", capture_frames=4
    )
    candidate = tmp_path / "candidate"
    shutil.copytree(qualification.project, candidate)
    target = candidate / "main.tscn"
    target.write_bytes(target.read_bytes().replace(b"1.250", replacement))
    with pytest.raises(RuntimeError, match=error):
        _verify_native_main_candidate_tree(
            qualification, candidate, patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 4)
        )


def test_native_shader_counterfactual_is_inconclusive_without_preservation_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _native_shader_counterfactual_qualification(tmp_path)
    _install_native_shader_counterfactual_fakes(monkeypatch, qualification)

    receipt = execute_native_main_shader_counterfactual(
        qualification,
        tmp_path / "counterfactual-run",
        patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 7),
        replacement=0.0,
    )

    assert receipt["decision"] == "INCONCLUSIVE"
    assert receipt["reason"] == "native_state_or_semantic_preservation_oracle_missing"
    assert receipt["qualification_only"] is True
    assert receipt["scoreable"] is False
    assert receipt["native_equivalence"] == "NOT_ESTABLISHED"
    assert receipt["renderer_comparison"]["hazard_removed"] is True
    assert receipt["preservation"]["oracle_available"] is False
    assert receipt["preservation"]["gameplay_state_equal"] is False
    assert receipt["preservation"]["semantic_invariants_equal"] is False
    assert receipt["preservation"]["typed_full_state_stream_equal"] is False
    assert receipt["preservation"]["typed_full_state_stream"] is None
    assert receipt["factual"]["tree_binding_before"] == receipt["factual"]["tree_binding_after"]
    assert receipt["candidate"]["tree_binding_before"] == receipt["candidate"]["tree_binding_after"]
    assert receipt["factual"]["trace_sha256"] == receipt["candidate"]["trace_sha256"]


def test_generic_native_semantic_invariants_cannot_authorize_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _native_shader_counterfactual_qualification(tmp_path)
    _install_native_shader_counterfactual_fakes(
        monkeypatch, qualification, meaningful_oracle=True
    )

    receipt = execute_native_main_shader_counterfactual(
        qualification,
        tmp_path / "counterfactual-run",
        patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 7),
        replacement=0.0,
    )

    assert receipt["decision"] == "INCONCLUSIVE"
    assert receipt["preservation"]["oracle_available"] is False


def test_controlled_native_materializer_seals_fixture_provider_and_trace(tmp_path: Path) -> None:
    qualification = _controlled_native_shader_counterfactual_qualification(tmp_path)
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    fixture = qualification.native_main["controlled_fixture"]

    assert qualification.native_main["controlled_mutation"] is True
    assert qualification.native_main["upstream_defect"] is False
    assert qualification.native_main["scoreable"] is False
    assert qualification.native_main["native_equivalence"] == "NOT_ESTABLISHED"
    assert fixture["schema"] == "flashpatch-controlled-native-shader-fixture-v1"
    assert fixture["provider_script"] == "res://controlled_state_provider.gd"
    assert fixture["provider_script_sha256"] == trace["state_observation"]["provider_script_sha256"]
    assert fixture["trace_sha256"] == "sha256:" + __import__("hashlib").sha256(
        qualification.trace.read_bytes()
    ).hexdigest()
    assert not (qualification.project / ".git").exists()


@pytest.mark.parametrize(("fixed_fps", "capture_frames"), [(30, 12), (60, 13)])
def test_controlled_native_materializer_rejects_trace_variants(
    tmp_path: Path, fixed_fps: int, capture_frames: int
) -> None:
    with pytest.raises(ValueError, match="frozen 60 fps, 12-frame trace"):
        materialize_controlled_native_main_shader_qualification(
            tmp_path / "controlled-native-shader-qualification",
            fixed_fps=fixed_fps,
            capture_frames=capture_frames,
        )


def test_controlled_native_trace_rejects_malformed_oracle_descriptor(tmp_path: Path) -> None:
    qualification = _controlled_native_shader_counterfactual_qualification(tmp_path)
    trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    trace["state_observation"]["node_path"] = "/root/Attacker"
    qualification.trace.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(RuntimeError, match="trace contract is invalid"):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


@pytest.mark.parametrize(
    ("attack", "error"),
    [
        ("missing_frame", "not frame complete"),
        ("malformed_type", "typed values are invalid"),
        ("missing_transition", "required action transition"),
        ("missing_terminal", "did not reach its terminal state"),
    ],
)
def test_controlled_native_classifier_rejects_malformed_state_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    error: str,
) -> None:
    qualification = _controlled_native_shader_counterfactual_qualification(tmp_path)

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        payload = _native_main_replay(qualification, output)
        if attack == "missing_frame":
            payload["state_observations"].pop()
        elif attack == "malformed_type":
            payload["state_observations"][0]["values"]["completed"] = 0
        elif attack == "missing_transition":
            payload["state_observations"][1]["values"]["phase"] = 0
        elif attack == "missing_terminal":
            payload["state_observations"][-1]["values"]["completed"] = False
        output.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        "flashpatch.public_godot.GodotNativeMainRendererReplayRunner.replay", replay
    )
    with pytest.raises(RuntimeError, match=error):
        classify_native_main_capture_qualification(qualification, tmp_path / "replay.json")


def test_native_shader_counterfactual_passes_only_with_equal_meaningful_preservation_oracles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _controlled_native_shader_counterfactual_qualification(tmp_path)
    _install_native_shader_counterfactual_fakes(monkeypatch, qualification)

    receipt = execute_native_main_shader_counterfactual(
        qualification,
        tmp_path / "counterfactual-run",
        patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 8),
        replacement=0.0,
    )

    assert receipt["decision"] == "PASS"
    assert receipt["scoreable"] is False
    assert receipt["preservation"]["oracle_available"] is True
    assert receipt["preservation"]["gameplay_state_equal"] is True
    assert receipt["preservation"]["semantic_invariants_equal"] is True
    assert receipt["preservation"]["typed_full_state_stream_equal"] is True
    assert len(receipt["preservation"]["typed_full_state_stream"]) == 12
    assert receipt["preservation"]["terminal_completion"] is True
    assert receipt["preservation"]["terminal_state"] == {"completed": True, "phase": 1}
    assert receipt["preservation"]["score"] == 7
    assert receipt["factual"]["hazard_frame_indices"] == [1]
    assert receipt["candidate"]["hazard_frame_indices"] == []
    assert receipt["factual"]["renderer_rgb_sha256"] != receipt["candidate"]["renderer_rgb_sha256"]
    serialized = json.dumps(receipt)
    assert "1.250" not in serialized
    assert '"replacement": 0.0' not in serialized


def test_native_shader_counterfactual_rejects_mismatched_preservation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _controlled_native_shader_counterfactual_qualification(tmp_path)
    _install_native_shader_counterfactual_fakes(
        monkeypatch,
        qualification,
        meaningful_oracle=True,
        attack="state_mismatch",
    )

    with pytest.raises(RuntimeError, match="gameplay state or semantic invariants changed"):
        execute_native_main_shader_counterfactual(
            qualification,
            tmp_path / "counterfactual-run",
            patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 8),
            replacement=0.0,
        )


@pytest.mark.parametrize(
    ("attack", "error"),
    [
        ("wrong_identity", "runtime observations do not match declared contributors"),
        ("wrong_value", "runtime value or identity does not match source"),
        ("residual_hazard", "candidate retains renderer hazard"),
        ("tree_mutation", "candidate must change exactly one declared source file"),
    ],
)
def test_native_shader_counterfactual_fails_closed_on_runtime_renderer_or_tree_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    error: str,
) -> None:
    qualification = _native_shader_counterfactual_qualification(tmp_path)
    _install_native_shader_counterfactual_fakes(
        monkeypatch,
        qualification,
        meaningful_oracle=True,
        attack=attack,
    )

    with pytest.raises(RuntimeError, match=error):
        execute_native_main_shader_counterfactual(
            qualification,
            tmp_path / "counterfactual-run",
            patch=NativeTscnTokenPatch("res://main.tscn", "flash_intensity", 7),
            replacement=0.0,
        )
