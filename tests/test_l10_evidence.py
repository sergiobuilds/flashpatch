from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import flashpatch.l10_evidence as l10_evidence

from flashpatch.l10_evidence import (
    L10EvidenceError,
    _execution_log,
    _unity_execution_log,
    _verify_unity_capture_outcomes,
    promote_godot_controlled_evidence,
    promote_unreal_controlled_evidence,
    _verify_godot_git_source,
    _verify_unity_git_source,
)
from flashpatch.l10_unity import unity_adapter_fingerprints
from flashpatch.l10_unity_runner import UNITY_VULKAN_LOADER_SHA256


VERSION = "5.6.0-43139311+UE5"


def test_unreal_promotion_rejects_unbound_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "Content/Maps").mkdir(parents=True)
    (source / "Content/Maps/Blank.umap").write_bytes(b"unbound map")
    (source / "LICENCE").write_text("MIT\n")

    with pytest.raises(L10EvidenceError, match="owned Git checkout"):
        promote_unreal_controlled_evidence(
            tmp_path / "runtime",
            source,
            tmp_path / "evidence",
        )


def _capture(root: Path, *, mode: str = "factual") -> str:
    root.mkdir()
    hashes = []
    for index in range(18):
        path = root / f"{index:04d}.png"
        assert cv2.imwrite(str(path), np.full((2, 3, 3), index, dtype=np.uint8))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    png_set = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    marker = {
        "schema": "flashpatch-l10-unreal-execution-marker-v1",
        "engine": "Unreal",
        "engine_version": VERSION,
        "frame_count": 18,
        "mode": mode,
        "png_set_sha256": png_set,
    }
    (root / "execution-marker.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    return png_set


def _log(path: Path, capture: Path) -> None:
    marker = json.loads((capture / "execution-marker.json").read_text())
    path.write_text(
        f'LogCsvProfiler: Display: Metadata set : engineversion="{VERSION}"\n'
        "LogCsvProfiler: Display: Metadata set : commandline=\"-ExecutePythonScript=/project/FlashPatchL10Capture.py\"\n"
        "LogInit: Display: Engine is initialized.\n"
        "FLASHPATCH_L10_COMPLETE " + json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n"
    )


def test_packs_engine_log_and_independent_execution_marker(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _capture(capture)
    log = tmp_path / "engine.log"
    _log(log, capture)
    result = _execution_log(
        log,
        capture,
        tmp_path / "output",
        "run.log.gz",
        lane="factual",
        frames_sha256="a" * 64,
        command_sha256="b" * 64,
    )
    assert len(result["sha256"]) == 64


def test_rejects_engine_log_without_real_engine_semantics(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _capture(capture)
    log = tmp_path / "engine.log"
    log.write_text("FORGED-NOT-ENGINE\n")
    with pytest.raises(L10EvidenceError, match="lacks required"):
        _execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="factual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
        )


def test_rejects_marker_for_another_lane_or_png_set(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _capture(capture, mode="counterfactual")
    log = tmp_path / "engine.log"
    _log(log, capture)
    with pytest.raises(L10EvidenceError, match="does not bind"):
        _execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="factual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
        )


def _unity_v2_runtime(
    root: Path,
    *,
    graphics_device_type: str = "Vulkan",
    graphics_device_vendor: str = "NVIDIA",
    graphics_device_id: int = 0x2C05,
    graphics_device_name: str = "NVIDIA GeForce RTX 5070 Ti",
    static_rgb: bool = False,
    static_rendered_value: bool = False,
    lane: str = "factual",
) -> tuple[Path, Path, str, str]:
    capture = root / "capture"
    capture.mkdir()
    png_hashes = []
    for index in range(121):
        value = 0 if static_rgb else index
        path = capture / f"{index:04d}.png"
        assert cv2.imwrite(str(path), np.full((2, 3, 3), value, dtype=np.uint8))
        png_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    png_set = hashlib.sha256("\n".join(png_hashes).encode("ascii")).hexdigest()
    events = {
        "events": [
            {
                "frame": index,
                "rendered_value": 1.0 if static_rendered_value else float(index),
            }
            for index in range(121)
        ]
    }
    (capture / "runtime-events.json").write_text(
        json.dumps(events, indent=2, sort_keys=True) + "\n"
    )
    (capture / "state-stream.json").write_text(
        json.dumps(
            [{"frame": index, "terminal_state": "captured"} for index in range(121)],
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    runtime_events_sha256 = hashlib.sha256(
        (capture / "runtime-events.json").read_bytes()
    ).hexdigest()
    state_stream_sha256 = hashlib.sha256(
        (capture / "state-stream.json").read_bytes()
    ).hexdigest()
    project_manifest_sha256 = "c" * 64
    scene_sha256 = "d" * 64
    marker = {
        "adapter_sha256": unity_adapter_fingerprints()[
            "Assets/Editor/FlashPatchL10Capture.cs"
        ],
        "engine": "Unity",
        "engine_version": "2022.3.8f1",
        "frame_count": 121,
        "graphics_device_id": graphics_device_id,
        "graphics_device_name": graphics_device_name,
        "graphics_device_type": graphics_device_type,
        "graphics_device_vendor": graphics_device_vendor,
        "mode": lane,
        "png_set_sha256": png_set,
        "project_manifest_sha256": project_manifest_sha256,
        "replay_profile": "deterministic-step-v2",
        "runtime_events_sha256": runtime_events_sha256,
        "scene_sha256": scene_sha256,
        "schema": "flashpatch-l10-unity-execution-marker-v2",
        "state_stream_sha256": state_stream_sha256,
    }
    marker_raw = json.dumps(marker, indent=2, sort_keys=True) + "\n"
    (capture / "execution-marker.json").write_text(marker_raw)
    log = root / "unity.log"
    log.write_text(
        f"FLASHPATCH_L10_VULKAN_LOADER_VERIFIED {UNITY_VULKAN_LOADER_SHA256}\n"
        "Unity Editor version:    2022.3.8f1\n"
        "-executeMethod\nFlashPatchL10Capture.Run\n"
        "FLASHPATCH_L10_COMPLETE "
        + json.dumps(marker, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    return capture, log, project_manifest_sha256, scene_sha256


def test_unity_v2_log_accepts_dynamic_nvidia_vulkan_capture(tmp_path: Path) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(tmp_path)
    result = _unity_execution_log(
        log,
        capture,
        tmp_path / "output",
        "run.log.gz",
        lane="factual",
        frames_sha256="a" * 64,
        command_sha256="b" * 64,
        project_manifest_sha256=manifest_sha256,
        scene_sha256=scene_sha256,
    )
    assert len(result["sha256"]) == 64


def test_unity_v2_log_accepts_stable_counterfactual_cause_with_dynamic_rgb(
    tmp_path: Path,
) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(
        tmp_path,
        static_rendered_value=True,
        lane="counterfactual",
    )
    result = _unity_execution_log(
        log,
        capture,
        tmp_path / "output",
        "run.log.gz",
        lane="counterfactual",
        frames_sha256="a" * 64,
        command_sha256="b" * 64,
        project_manifest_sha256=manifest_sha256,
        scene_sha256=scene_sha256,
    )
    assert len(result["sha256"]) == 64


def test_unity_v2_log_rejects_dynamic_counterfactual_cause(
    tmp_path: Path,
) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(
        tmp_path,
        lane="counterfactual",
    )
    with pytest.raises(L10EvidenceError, match="dynamic trace"):
        _unity_execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="counterfactual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
            project_manifest_sha256=manifest_sha256,
            scene_sha256=scene_sha256,
        )


def test_unity_v2_log_rejects_static_factual_cause(
    tmp_path: Path,
) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(
        tmp_path,
        static_rendered_value=True,
    )
    with pytest.raises(L10EvidenceError, match="dynamic trace"):
        _unity_execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="factual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
            project_manifest_sha256=manifest_sha256,
            scene_sha256=scene_sha256,
        )


def test_unity_v2_log_rejects_opengl_fallback(tmp_path: Path) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(
        tmp_path,
        graphics_device_type="OpenGLCore",
        graphics_device_vendor="Mesa",
        graphics_device_id=0,
        graphics_device_name="llvmpipe",
    )
    with pytest.raises(L10EvidenceError, match="unapproved graphics device"):
        _unity_execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="factual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
            project_manifest_sha256=manifest_sha256,
            scene_sha256=scene_sha256,
        )


def test_unity_v2_log_rejects_static_rgb_with_dynamic_runtime(tmp_path: Path) -> None:
    capture, log, manifest_sha256, scene_sha256 = _unity_v2_runtime(
        tmp_path,
        static_rgb=True,
    )
    with pytest.raises(L10EvidenceError, match="dynamic trace"):
        _unity_execution_log(
            log,
            capture,
            tmp_path / "output",
            "run.log.gz",
            lane="factual",
            frames_sha256="a" * 64,
            command_sha256="b" * 64,
            project_manifest_sha256=manifest_sha256,
            scene_sha256=scene_sha256,
        )


def _unity_outcome(root: Path, name: str, result: str) -> None:
    pack = root / f"{name}-pack"
    pack.mkdir(parents=True)
    frame_count = 16
    frames = np.zeros((frame_count, 2, 3, 3), dtype=np.uint8)
    if result == "HAZARDOUS":
        frames[1::2] = 255
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    frames_path = pack / "frames.npz"
    np.savez_compressed(frames_path, frames=frames, timestamps=timestamps)
    timestamps_path = pack / "timestamps.json"
    timestamps_path.write_text(json.dumps(timestamps.tolist(), indent=2) + "\n")
    from flashpatch.core import analyze
    observed = analyze(frames, timestamps)
    observed_result = "HAZARDOUS" if observed.hazardous else "SAFE"
    if result == "HAZARDOUS" and observed_result != "HAZARDOUS":
        raise AssertionError("test fixture did not produce a hazard")
    hazard_frames = sorted({
        index
        for window in observed.windows
        for index, timestamp in enumerate(timestamps)
        if window.start <= timestamp <= window.end
    })
    detector = {
        "result": observed_result,
        "hazard_frames": hazard_frames,
        "frames_rgb_sha256": hashlib.sha256(frames.tobytes(order="C")).hexdigest(),
        "timestamps_sha256": hashlib.sha256(timestamps.tobytes(order="C")).hexdigest(),
    }
    detector_path = pack / "detector.json"
    detector_path.write_text(json.dumps(detector, indent=2, sort_keys=True) + "\n")
    manifest = {
        "result": detector["result"],
        "hazard_frames": detector["hazard_frames"],
        "detector": {
            "path": detector_path.name,
            "sha256": hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        },
        "frames": {"path": frames_path.name, "sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest()},
        "timestamps": {"path": timestamps_path.name, "sha256": hashlib.sha256(timestamps_path.read_bytes()).hexdigest()},
    }
    (pack / "capture-pack.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_unity_outcomes_fail_closed_before_safe_safe_promotion(tmp_path: Path) -> None:
    for ordinal in (0, 1, 2, 3):
        for lane in ("factual", "counterfactual"):
            name = f"unity-{lane}-baseline" if ordinal == 0 else f"unity-repeat-{ordinal}-{lane}"
            _unity_outcome(tmp_path, name, "SAFE")
    with pytest.raises(L10EvidenceError, match="do not establish hazard removal"):
        _verify_unity_capture_outcomes(tmp_path)
    result = _verify_unity_capture_outcomes(tmp_path, require_pass=False)
    assert result == {
        "hazard_removed": False,
        "repeat_reproducible": True,
        "factual_result": "SAFE",
        "counterfactual_result": "SAFE",
    }


def test_unity_outcomes_require_three_exact_repeats(tmp_path: Path) -> None:
    for ordinal in (0, 1, 2, 3):
        for lane, result in (
            ("factual", "HAZARDOUS"),
            ("counterfactual", "SAFE"),
        ):
            name = f"unity-{lane}-baseline" if ordinal == 0 else f"unity-repeat-{ordinal}-{lane}"
            _unity_outcome(tmp_path, name, result)
    _verify_unity_capture_outcomes(tmp_path)
    detector = tmp_path / "unity-repeat-3-factual-pack/detector.json"
    value = json.loads(detector.read_text())
    value["frames_rgb_sha256"] = "d" * 64
    detector.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    manifest = tmp_path / "unity-repeat-3-factual-pack/capture-pack.json"
    manifest_value = json.loads(manifest.read_text())
    manifest_value["detector"]["sha256"] = hashlib.sha256(detector.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(manifest_value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10EvidenceError, match="binding is invalid"):
        _verify_unity_capture_outcomes(tmp_path)


def test_rejects_synthetic_unity_source_without_git_identity(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source"
    source.mkdir()
    with pytest.raises(L10EvidenceError, match="owned Git checkout"):
        _verify_unity_git_source(source)


def test_rejects_synthetic_godot_source_without_git_identity(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source"
    source.mkdir()
    with pytest.raises(L10EvidenceError, match="owned Git checkout"):
        _verify_godot_git_source(source)


def test_godot_promotion_requires_live_renderer_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        l10_evidence,
        "_verify_godot_git_source",
        lambda source: {
            "repository": "https://github.com/sergiobuilds/flashpatch",
            "revision": "9e5fc3bcc922984a05a9edf32c664b85eb76dad3",
            "tree_sha256": "a7981ea1f18849f89d9cccda249152c91aa3a8e43c90732482a980f370e5af46",
        },
    )
    with pytest.raises(L10EvidenceError, match="actual renderer display"):
        promote_godot_controlled_evidence(
            tmp_path / "source",
            tmp_path / "godot",
            tmp_path / "evidence",
        )


def test_godot_source_rejects_hidden_index_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)

    def git_output(_source: Path, *args: str) -> str:
        command = tuple(args)
        if command == ("remote", "get-url", "origin"):
            return "https://github.com/sergiobuilds/flashpatch.git\n"
        if command == ("rev-parse", "HEAD"):
            return "9e5fc3bcc922984a05a9edf32c664b85eb76dad3\n"
        if command == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if command == ("ls-files", "-v"):
            return "h LICENSE\n"
        raise AssertionError(f"unexpected Git command: {command}")

    monkeypatch.setattr(l10_evidence, "_git_output", git_output)
    with pytest.raises(L10EvidenceError, match="hidden file flags"):
        _verify_godot_git_source(source)
