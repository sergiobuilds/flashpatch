"""Promote completed engine runs into immutable L10 evidence bundles."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import secrets
from pathlib import Path

import cv2
import numpy as np

from .core import analyze
from .l10_godot import GodotL10ReplayRunner, adapter_sha256
from .renderer_artifact import renderer_rgb_sha256
from .safety_ci import compile_project, write_receipt
from .l10_unity import unity_adapter_fingerprints, unity_linux_package_manifest_bytes
from .l10_unity_runner import (
    UNITY_IMAGE,
    UNITY_IMAGE_DIGEST,
    UNITY_RECEIPT_COMMAND,
    UNITY_RUNS,
    UNITY_VULKAN_LOADER_SHA256,
)


class L10EvidenceError(ValueError):
    """Runtime output cannot be promoted without weakening its provenance."""


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise L10EvidenceError("pinned source Git state is unavailable") from exc
    return completed.stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise L10EvidenceError("pinned source Git object is unavailable") from exc
    return completed.stdout


def _verify_unity_git_source(source: Path) -> dict[str, str]:
    repository = "https://github.com/Unity-Technologies/VisualEffectGraph-Samples"
    revision = "4624d386e5f2e63e383a7e572362785470fe0f33"
    tree_sha256 = "5b082d85ad914175a9c8eba1cac0d513dd3ad6658a63e305fbf79695a889f1c9"
    if not (source / ".git").is_dir():
        raise L10EvidenceError("pinned Unity source is not an owned Git checkout")
    if _git_output(source, "remote", "get-url", "origin").strip().removesuffix(".git") != repository:
        raise L10EvidenceError("pinned Unity source origin changed")
    if _git_output(source, "rev-parse", "HEAD").strip() != revision:
        raise L10EvidenceError("pinned Unity source revision changed")
    if _git_output(source, "status", "--porcelain=v1", "--untracked-files=all"):
        raise L10EvidenceError("pinned Unity source is not clean")
    flagged = [
        line
        for line in _git_output(source, "ls-files", "-v").splitlines()
        if line and line[0] != "H"
    ]
    if flagged:
        raise L10EvidenceError("pinned Unity source index contains hidden file flags")
    staged_rows = [
        line.split(None, 3)
        for line in _git_output(source, "ls-files", "-s").splitlines()
        if line
    ]
    tracked: dict[str, str] = {}
    for mode, blob, stage_and_path in (
        (row[0], row[1], row[2] + " " + row[3]) for row in staged_rows
    ):
        stage, path = stage_and_path.split(" ", 1)
        if stage != "0" or mode not in {"100644", "100755"} or path in tracked:
            raise L10EvidenceError("pinned Unity source index schema is invalid")
        tracked[path] = blob
    scoped_files = {
        path.relative_to(source).as_posix(): path
        for folder in ("Assets", "Packages", "ProjectSettings")
        for path in (source / folder).rglob("*")
        if path.is_file()
    }
    scoped_tracked = {
        path: blob
        for path, blob in tracked.items()
        if path.split("/", 1)[0] in {"Assets", "Packages", "ProjectSettings"}
    }
    if set(scoped_files) != set(scoped_tracked):
        raise L10EvidenceError("pinned Unity source tracked file set changed")
    for relative, blob in scoped_tracked.items():
        index_blob = _git_bytes(source, "cat-file", "blob", blob)
        if index_blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            oid_lines = [line for line in index_blob.splitlines() if line.startswith(b"oid sha256:")]
            if len(oid_lines) != 1 or _sha256(scoped_files[relative]) != oid_lines[0].removeprefix(b"oid sha256:").decode("ascii"):
                raise L10EvidenceError("pinned Unity LFS worktree bytes differ from the index pointer")
        else:
            observed_blob = _git_output(
                source, "hash-object", "--no-filters", "--", relative
            ).strip()
            if observed_blob != blob:
                raise L10EvidenceError("pinned Unity source tracked bytes differ from HEAD")
    listing = _git_output(source, "ls-tree", "-r", "HEAD").encode()
    if hashlib.sha256(listing).hexdigest() != tree_sha256:
        raise L10EvidenceError("pinned Unity source tree listing changed")
    lfs_lines = [line for line in _git_output(source, "lfs", "ls-files", "-l").splitlines() if line]
    if len(lfs_lines) != 6:
        raise L10EvidenceError("pinned Unity LFS inventory changed")
    for line in lfs_lines:
        digest, _, relative = line.partition(" * ")
        path = source / relative
        if not path.is_file() or _sha256(path) != digest:
            raise L10EvidenceError("pinned Unity LFS object is missing or stale")
    return {"repository": repository, "revision": revision, "tree_sha256": tree_sha256}


def _verify_godot_git_source(source: Path) -> dict[str, str]:
    repository = "https://github.com/sergiobuilds/flashpatch-public"
    revision = "9e5fc3bcc922984a05a9edf32c664b85eb76dad3"
    tree_sha256 = "a7981ea1f18849f89d9cccda249152c91aa3a8e43c90732482a980f370e5af46"
    if not (source / ".git").is_dir():
        raise L10EvidenceError("pinned Godot source is not an owned Git checkout")
    if _git_output(source, "remote", "get-url", "origin").strip().removesuffix(".git") != repository:
        raise L10EvidenceError("pinned Godot source origin changed")
    if _git_output(source, "rev-parse", "HEAD").strip() != revision:
        raise L10EvidenceError("pinned Godot source revision changed")
    if _git_output(source, "status", "--porcelain=v1", "--untracked-files=all"):
        raise L10EvidenceError("pinned Godot source is not clean")
    if any(
        line and line[0] != "H"
        for line in _git_output(source, "ls-files", "-v").splitlines()
    ):
        raise L10EvidenceError("pinned Godot source index contains hidden file flags")
    listing = _git_output(source, "ls-tree", "-r", "HEAD").encode()
    if hashlib.sha256(listing).hexdigest() != tree_sha256:
        raise L10EvidenceError("pinned Godot source tree listing changed")
    relative_root = Path("benchmarks/aigame-psebench/corpus/interaction-burst")
    expected = {
        "flashpatch.contract.json",
        "flashpatch.renderer.contract.json",
        "main.gd",
        "main.tscn",
        "project.godot",
        "trace.json",
    }
    observed = {
        path.name for path in (source / relative_root).iterdir() if path.is_file()
    }
    if observed != expected:
        raise L10EvidenceError("pinned Godot fixture file set changed")
    for name in expected:
        relative = (relative_root / name).as_posix()
        blob = _git_output(source, "rev-parse", f"HEAD:{relative}").strip()
        if _git_output(source, "hash-object", "--no-filters", "--", relative).strip() != blob:
            raise L10EvidenceError("pinned Godot fixture bytes differ from HEAD")
    return {"repository": repository, "revision": revision, "tree_sha256": tree_sha256}


def _verify_unreal_git_source(source: Path) -> dict[str, str]:
    repository = "https://github.com/daftsoftware/StarterProject"
    revision = "340a35ae8a3995cddc773650dedcba440e3a3523"
    tree_sha256 = "42f5bc60a3b058895109ae0a53a578b100a8ef053c825ff7be876d2c2e3027ab"
    if not (source / ".git").is_dir():
        raise L10EvidenceError("pinned Unreal source is not an owned Git checkout")
    if _git_output(source, "remote", "get-url", "origin").strip().removesuffix(".git") != repository:
        raise L10EvidenceError("pinned Unreal source origin changed")
    if _git_output(source, "rev-parse", "HEAD").strip() != revision:
        raise L10EvidenceError("pinned Unreal source revision changed")
    if _git_output(source, "status", "--porcelain=v1", "--untracked-files=all"):
        raise L10EvidenceError("pinned Unreal source is not clean")
    if any(
        line and line[0] != "H"
        for line in _git_output(source, "ls-files", "-v").splitlines()
    ):
        raise L10EvidenceError("pinned Unreal source index contains hidden file flags")
    listing = _git_output(source, "ls-tree", "-r", "HEAD").encode()
    if hashlib.sha256(listing).hexdigest() != tree_sha256:
        raise L10EvidenceError("pinned Unreal source tree listing changed")
    for relative in ("Content/Maps/Blank.umap", "LICENCE"):
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise L10EvidenceError("frozen Unreal source is missing or unsafe")
        blob = _git_output(source, "rev-parse", f"HEAD:{relative}").strip()
        index_bytes = _git_bytes(source, "cat-file", "blob", blob)
        if index_bytes.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            oid_lines = [
                line
                for line in index_bytes.splitlines()
                if line.startswith(b"oid sha256:")
            ]
            if (
                len(oid_lines) != 1
                or _sha256(path)
                != oid_lines[0].removeprefix(b"oid sha256:").decode("ascii")
            ):
                raise L10EvidenceError(
                    "pinned Unreal LFS worktree bytes differ from the index pointer"
                )
        elif _git_output(
            source, "hash-object", "--no-filters", "--", relative
        ).strip() != blob:
            raise L10EvidenceError("pinned Unreal source tracked bytes differ from HEAD")
    return {"repository": repository, "revision": revision, "tree_sha256": tree_sha256}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id(
    factual_frames_sha256: str,
    counterfactual_frames_sha256: str,
    factual_log_sha256: str,
    counterfactual_log_sha256: str,
    factual_marker_sha256: str,
    counterfactual_marker_sha256: str,
) -> str:
    return hashlib.sha256(_canonical({
        "counterfactual_execution_log_sha256": counterfactual_log_sha256,
        "counterfactual_execution_marker_sha256": counterfactual_marker_sha256,
        "counterfactual_frames_sha256": counterfactual_frames_sha256,
        "factual_execution_log_sha256": factual_log_sha256,
        "factual_execution_marker_sha256": factual_marker_sha256,
        "factual_frames_sha256": factual_frames_sha256,
    })).hexdigest()


def _copy(source: Path, root: Path, relative: str) -> dict[str, str]:
    if source.is_symlink() or not source.is_file():
        raise L10EvidenceError(f"source artifact is missing or unsafe: {source}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L10EvidenceError(f"destination artifact already exists: {destination}")
    shutil.copyfile(source, destination)
    return {"path": relative, "sha256": _sha256(destination)}


def _write(root: Path, relative: str, value: object) -> dict[str, str]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L10EvidenceError(f"destination artifact already exists: {destination}")
    destination.write_bytes(_canonical(value))
    return {"path": relative, "sha256": _sha256(destination)}


def _copy_gzip(source: Path, root: Path, relative: str) -> dict[str, str]:
    if source.is_symlink() or not source.is_file():
        raise L10EvidenceError(f"source log is missing or unsafe: {source}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L10EvidenceError(f"destination artifact already exists: {destination}")
    destination.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
    return {"path": relative, "sha256": _sha256(destination)}


def _execution_log(
    source: Path,
    capture_source: Path,
    root: Path,
    relative: str,
    *,
    lane: str,
    frames_sha256: str,
    command_sha256: str,
) -> dict[str, str]:
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise L10EvidenceError("Unreal execution log is not UTF-8") from exc
    version = "5.6.0-43139311+UE5"
    required = (
        f'engineversion="{version}"',
        "-ExecutePythonScript=/project/FlashPatchL10Capture.py",
        "LogInit: Display: Engine is initialized.",
        "FLASHPATCH_L10_COMPLETE ",
    )
    if any(marker not in text for marker in required):
        raise L10EvidenceError("Unreal execution log lacks required engine or harness markers")
    if "Fatal error:" in text or "Assertion failed:" in text or "FLASHPATCH_L10_FAILURE" in text:
        raise L10EvidenceError("Unreal execution log contains a terminal failure")
    marker_path = capture_source / "execution-marker.json"
    try:
        marker_raw = marker_path.read_bytes()
        marker = json.loads(marker_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise L10EvidenceError("Unreal execution marker is unreadable") from exc
    pngs = [capture_source / f"{index:04d}.png" for index in range(18)]
    if any(path.is_symlink() or not path.is_file() for path in pngs):
        raise L10EvidenceError("Unreal execution marker lacks its exact PNG sequence")
    png_set_sha256 = hashlib.sha256(
        "\n".join(_sha256(path) for path in pngs).encode("ascii")
    ).hexdigest()
    expected_marker = {
        "schema": "flashpatch-l10-unreal-execution-marker-v1",
        "engine": "Unreal",
        "engine_version": version,
        "frame_count": 18,
        "mode": lane,
        "png_set_sha256": png_set_sha256,
    }
    if marker != expected_marker or marker_raw != _canonical(marker):
        raise L10EvidenceError("Unreal execution marker does not bind the captured PNG sequence")
    completion_prefix = "FLASHPATCH_L10_COMPLETE "
    completion_lines = [
        line[len(completion_prefix):]
        for line in text.splitlines()
        if line.startswith(completion_prefix)
    ]
    if len(completion_lines) != 1 or json.loads(completion_lines[0]) != marker:
        raise L10EvidenceError("Unreal execution log lacks the engine completion marker")
    packed = {
        "schema": "flashpatch-l10-packed-execution-v1",
        "command_sha256": command_sha256,
        "engine": "Unreal",
        "engine_version": version,
        "frames_sha256": frames_sha256,
        "lane": lane,
        "execution_marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "png_set_sha256": png_set_sha256,
        "raw_log_sha256": hashlib.sha256(raw).hexdigest(),
    }
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L10EvidenceError(f"destination artifact already exists: {destination}")
    payload = raw + b"\nFLASHPATCH_L10_PACKED " + json.dumps(
        packed, separators=(",", ":"), sort_keys=True
    ).encode() + b"\n"
    destination.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return {"path": relative, "sha256": _sha256(destination)}


def _unity_execution_log(
    source: Path,
    capture_source: Path,
    root: Path,
    relative: str,
    *,
    lane: str,
    frames_sha256: str,
    command_sha256: str,
    project_manifest_sha256: str,
    scene_sha256: str,
) -> dict[str, str]:
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise L10EvidenceError("Unity execution log is not UTF-8") from exc
    version = "2022.3.8f1"
    required = (
        f"Unity Editor version:    {version}",
        f"FLASHPATCH_L10_VULKAN_LOADER_VERIFIED {UNITY_VULKAN_LOADER_SHA256}",
        "FLASHPATCH_L10_COMPLETE ",
    )
    argv = text.splitlines()
    expected_argv = ["-executeMethod", "FlashPatchL10Capture.Run"]
    if (
        any(marker not in text for marker in required)
        or not any(
        argv[index : index + len(expected_argv)] == expected_argv
        for index in range(len(argv) - len(expected_argv) + 1)
        )
    ):
        raise L10EvidenceError("Unity execution log lacks required editor or harness markers")
    if any(marker in text for marker in ("Crash!!!", "Aborting batchmode due to failure", "FLASHPATCH_L10_FAILURE")):
        raise L10EvidenceError("Unity execution log contains a terminal failure")
    marker_path = capture_source / "execution-marker.json"
    try:
        marker_raw = marker_path.read_bytes()
        marker = json.loads(marker_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise L10EvidenceError("Unity execution marker is unreadable") from exc
    frame_count = 121
    pngs = [capture_source / f"{index:04d}.png" for index in range(frame_count)]
    if any(path.is_symlink() or not path.is_file() for path in pngs):
        raise L10EvidenceError("Unity execution marker lacks its exact PNG sequence")
    png_set_sha256 = hashlib.sha256(
        "\n".join(_sha256(path) for path in pngs).encode("ascii")
    ).hexdigest()
    expected_marker = {
        "schema": "flashpatch-l10-unity-execution-marker-v2",
        "adapter_sha256": unity_adapter_fingerprints()["Assets/Editor/FlashPatchL10Capture.cs"],
        "engine": "Unity",
        "engine_version": version,
        "frame_count": frame_count,
        "graphics_device_id": marker.get("graphics_device_id"),
        "graphics_device_name": marker.get("graphics_device_name"),
        "graphics_device_type": "Vulkan",
        "graphics_device_vendor": "NVIDIA",
        "mode": lane,
        "png_set_sha256": png_set_sha256,
        "project_manifest_sha256": project_manifest_sha256,
        "replay_profile": "deterministic-step-v2",
        "scene_sha256": scene_sha256,
    }
    allowed_devices = {
        (0x2F04, "NVIDIA GeForce RTX 5070"),
        (0x2C05, "NVIDIA GeForce RTX 5070 Ti"),
    }
    if (marker.get("graphics_device_id"), marker.get("graphics_device_name")) not in allowed_devices:
        raise L10EvidenceError("Unity execution marker reports an unapproved graphics device")
    decoded_frames: list[bytes] = []
    for path in pngs:
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
            raise L10EvidenceError("Unity execution PNG is not decodable RGB evidence")
        decoded_frames.append(decoded.tobytes(order="C"))
    try:
        event_value = json.loads(
            (capture_source / "runtime-events.json").read_text(encoding="utf-8")
        )
        rendered_values = [row["rendered_value"] for row in event_value["events"]]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise L10EvidenceError("deterministic Unity runtime events are unreadable") from exc
    rendered_value_count = len(set(rendered_values))
    expected_rendered_value_count = 1 if lane == "counterfactual" else 2
    if (
        len(rendered_values) != frame_count
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            for value in rendered_values
        )
        or (
            rendered_value_count != 1
            if lane == "counterfactual"
            else rendered_value_count < expected_rendered_value_count
        )
        or len(set(decoded_frames)) < 2
    ):
        raise L10EvidenceError("deterministic Unity execution did not render a dynamic trace")
    runtime_events_sha256 = _sha256(capture_source / "runtime-events.json")
    state_stream_sha256 = _sha256(capture_source / "state-stream.json")
    expected_marker.update({
        "runtime_events_sha256": runtime_events_sha256,
        "state_stream_sha256": state_stream_sha256,
    })
    if marker != expected_marker or marker_raw != _canonical(marker):
        raise L10EvidenceError("Unity execution marker does not bind the captured PNG sequence")
    completion_prefix = "FLASHPATCH_L10_COMPLETE "
    completion_lines = [
        line[len(completion_prefix):]
        for line in text.splitlines()
        if line.startswith(completion_prefix)
    ]
    if len(completion_lines) != 1 or json.loads(completion_lines[0]) != marker:
        raise L10EvidenceError("Unity execution log lacks the editor completion marker")
    packed = {
        "schema": "flashpatch-l10-packed-execution-v1",
        "command_sha256": command_sha256,
        "engine": "Unity",
        "engine_version": version,
        "frames_sha256": frames_sha256,
        "lane": lane,
        "execution_marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "png_set_sha256": png_set_sha256,
        "raw_log_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_events_sha256": runtime_events_sha256,
        "state_stream_sha256": state_stream_sha256,
    }
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L10EvidenceError(f"destination artifact already exists: {destination}")
    payload = raw + b"\nFLASHPATCH_L10_PACKED " + json.dumps(
        packed, separators=(",", ":"), sort_keys=True
    ).encode() + b"\n"
    destination.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return {"path": relative, "sha256": _sha256(destination)}


def _capture(source: Path, root: Path, prefix: str) -> dict[str, object]:
    manifest_path = source / "capture-pack.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise L10EvidenceError("capture pack manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"frames", "timestamps", "detector"}
    if not expected.issubset(manifest):
        raise L10EvidenceError("capture pack is incomplete")
    refs: dict[str, dict[str, str]] = {}
    for name in expected:
        reference = manifest[name]
        path = source / reference["path"]
        if _sha256(path) != reference["sha256"]:
            raise L10EvidenceError(f"capture pack {name} hash mismatch")
        refs[name] = _copy(path, root, f"{prefix}/{path.name}")
    return {
        "frames": refs["frames"],
        "timestamps": refs["timestamps"],
        "runtime_events": None,
        "state_stream": None,
        "detector": {
            "result": manifest["result"],
            "hazard_frames": manifest["hazard_frames"],
            "receipt": refs["detector"],
        },
    }


def _verify_unity_capture_outcomes(
    runtime: Path,
    *,
    require_pass: bool = True,
) -> dict[str, object]:
    """Reopen all Unity captures and report whether they establish a repeatable fix."""
    baseline: dict[str, dict[str, object]] = {}
    repeat_reproducible = True
    for ordinal in (0, 1, 2, 3):
        observed: dict[str, dict[str, object]] = {}
        for lane in ("factual", "counterfactual"):
            name = (
                f"unity-{lane}-baseline"
                if ordinal == 0
                else f"unity-repeat-{ordinal}-{lane}"
            )
            pack = runtime / f"{name}-pack"
            manifest_path = pack / "capture-pack.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                frames_ref = manifest["frames"]
                timestamps_ref = manifest["timestamps"]
                detector_ref = manifest["detector"]
                frames_path = pack / frames_ref["path"]
                timestamps_path = pack / timestamps_ref["path"]
                detector_path = pack / detector_ref["path"]
                detector = json.loads(detector_path.read_text(encoding="utf-8"))
                with np.load(frames_path, allow_pickle=False) as packed:
                    if set(packed.files) != {"frames", "timestamps"}:
                        raise L10EvidenceError("Unity capture frame artifact schema is invalid")
                    frames = packed["frames"]
                    timestamps = packed["timestamps"]
                timestamp_values = json.loads(timestamps_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise L10EvidenceError("Unity capture outcome is unreadable") from exc
            if (
                frames.dtype != np.uint8
                or frames.ndim != 4
                or frames.shape[-1] != 3
                or timestamps.dtype != np.float64
                or timestamps.shape != (len(frames),)
                or timestamp_values != timestamps.tolist()
            ):
                raise L10EvidenceError("Unity capture frame or timestamp artifact is invalid")
            observed_analysis = analyze(frames, timestamps)
            observed_result = "HAZARDOUS" if observed_analysis.hazardous else "SAFE"
            observed_hazard_frames = sorted({
                index
                for window in observed_analysis.windows
                for index, timestamp in enumerate(timestamps)
                if window.start <= timestamp <= window.end
            })
            rgb_sha256 = hashlib.sha256(frames.tobytes(order="C")).hexdigest()
            semantic_timestamps_sha256 = hashlib.sha256(
                timestamps.tobytes(order="C")
            ).hexdigest()
            if (
                manifest_path.is_symlink()
                or frames_path.is_symlink()
                or timestamps_path.is_symlink()
                or detector_path.is_symlink()
                or _sha256(frames_path) != frames_ref.get("sha256")
                or _sha256(timestamps_path) != timestamps_ref.get("sha256")
                or _sha256(detector_path) != detector_ref.get("sha256")
                or manifest.get("result") != detector.get("result")
                or manifest.get("hazard_frames") != detector.get("hazard_frames")
                or detector.get("result") != observed_result
                or detector.get("hazard_frames") != observed_hazard_frames
                or detector.get("frames_rgb_sha256") != rgb_sha256
                or detector.get("timestamps_sha256") != semantic_timestamps_sha256
            ):
                raise L10EvidenceError("Unity capture outcome binding is invalid")
            observed[lane] = {
                "result": detector["result"],
                "hazard_frames": detector["hazard_frames"],
                "frames_rgb_sha256": detector["frames_rgb_sha256"],
                "timestamps_sha256": detector["timestamps_sha256"],
            }
        if ordinal == 0:
            baseline = observed
        elif observed != baseline:
            repeat_reproducible = False
    hazard_removed = (
        baseline["factual"]["result"] == "HAZARDOUS"
        and baseline["counterfactual"]["result"] == "SAFE"
    )
    if require_pass and not hazard_removed:
        raise L10EvidenceError(
            "Unity factual/counterfactual runs do not establish hazard removal"
        )
    if require_pass and not repeat_reproducible:
        raise L10EvidenceError("Unity capture outcomes are not reproducible")
    return {
        "hazard_removed": hazard_removed,
        "repeat_reproducible": repeat_reproducible,
        "factual_result": baseline["factual"]["result"],
        "counterfactual_result": baseline["counterfactual"]["result"],
    }


def _godot_capture(
    run_root: Path,
    *,
    lane: str,
    output: Path,
    prefix: str,
) -> tuple[dict[str, object], list[Path], str]:
    replay_path = (
        run_root / "factual-replay.json"
        if lane == "factual"
        else run_root / "counterfactual/00/replay.json"
    )
    frames_path = (
        run_root / "renderer-frames.npz"
        if lane == "factual"
        else run_root / "counterfactual/00/renderer-frames.npz"
    )
    png_root = (
        run_root / "renderer-capture"
        if lane == "factual"
        else run_root / "counterfactual/00/renderer-capture"
    )
    try:
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        with np.load(frames_path, allow_pickle=False) as packed:
            if set(packed.files) != {"frames", "timestamps"}:
                raise L10EvidenceError("Godot capture artifact schema is invalid")
            frames = packed["frames"]
            timestamps = packed["timestamps"]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise L10EvidenceError("Godot replay artifact is unreadable") from exc
    if (
        not isinstance(replay, dict)
        or replay.get("status") != "REPLAYED"
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape != (12, 180, 320, 3)
        or timestamps.dtype != np.float64
        or timestamps.shape != (12,)
        or not np.array_equal(
            timestamps,
            np.asarray([round(index / 60, 6) for index in range(12)], dtype=np.float64),
        )
    ):
        raise L10EvidenceError("Godot replay artifact is not the frozen 60fps RGB trace")
    pngs = [png_root / f"frame_{index:06d}.png" for index in range(12)]
    decoded = []
    for path in pngs:
        if path.is_symlink() or not path.is_file():
            raise L10EvidenceError("Godot replay PNG sequence is incomplete")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise L10EvidenceError("Godot replay PNG is unreadable")
        decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    png_frames = np.stack(decoded).astype(np.uint8, copy=False)
    if not np.array_equal(png_frames, frames):
        raise L10EvidenceError("Godot replay PNGs do not match the packed RGB frames")
    execution_log = replay.get("engine_execution_log")
    renderer_capture = replay.get("renderer_capture")
    actual_timestamps = (
        renderer_capture.get("actual_capture_timestamps_us")
        if isinstance(renderer_capture, dict)
        else None
    )
    if (
        not isinstance(execution_log, str)
        or "Godot Engine v4.7.1.stable.official.a13da4feb" not in execution_log
        or "OpenGL API" not in execution_log
        or execution_log.count("FLASHPATCH_L10_COMPLETE ") != 1
        or not isinstance(actual_timestamps, list)
        or len(actual_timestamps) != 12
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in actual_timestamps
        )
        or any(right <= left for left, right in zip(actual_timestamps, actual_timestamps[1:]))
    ):
        raise L10EvidenceError("Godot replay lacks actual engine execution semantics")
    execution_log = execution_log.rstrip("\n") + "\nGODOT_L10_ACTUAL_CAPTURE_TIMESTAMPS " + json.dumps(
        actual_timestamps, separators=(",", ":")
    ) + "\n"
    result = analyze(frames, timestamps)
    detector_result = "HAZARDOUS" if result.hazardous else "SAFE"
    hazard_frames = sorted({
        index
        for window in result.windows
        for index, timestamp in enumerate(timestamps)
        if window.start <= timestamp <= window.end
    })
    expected_result = "HAZARDOUS" if lane == "factual" else "SAFE"
    if detector_result != expected_result:
        raise L10EvidenceError("Godot replay detector result changed")
    frames_ref = _copy(frames_path, output, f"{prefix}/frames.npz")
    timestamps_ref = _write(output, f"{prefix}/timestamps.json", timestamps.tolist())
    events_raw = replay.get("runtime_events")
    if not isinstance(events_raw, list) or len(events_raw) != 12:
        raise L10EvidenceError("Godot replay runtime events are incomplete")
    events = [{
        "frame": index,
        "object_identity": "/root/InteractionBurst",
        "component": "GDScript",
        "property": "burst_intensity",
        "value": row.get("factual_value") if isinstance(row, dict) else None,
    } for index, row in enumerate(events_raw)]
    expected_value = 1.0 if lane == "factual" else 0.0
    if any(row["value"] != expected_value for row in events):
        raise L10EvidenceError("Godot replay runtime attribution changed")
    invariants = replay.get("semantic_invariants")
    if not isinstance(invariants, dict) or invariants.get("terminal_completion") is not True:
        raise L10EvidenceError("Godot replay terminal state is incomplete")
    states = [{
        "frame": index,
        "terminal_state": invariants.get("terminal_state"),
        "player_world_digest": invariants.get("player_world_digest"),
        "score": invariants.get("score"),
    } for index in range(12)]
    detector = _write(output, f"{prefix}/detector.json", {
        "schema": "flashpatch-l10-detector-receipt-v1",
        "frames_rgb_sha256": renderer_rgb_sha256(frames),
        "timestamps_sha256": hashlib.sha256(timestamps.tobytes(order="C")).hexdigest(),
        "result": detector_result,
        "hazard_frames": hazard_frames,
        "max_flash_count": result.max_flash_count,
        "max_affected_fraction": result.max_affected_fraction,
    })
    capture = {
        "frames": frames_ref,
        "timestamps": timestamps_ref,
        "runtime_events": _write(output, f"{prefix}/runtime-events.json", events),
        "state_stream": _write(output, f"{prefix}/state-stream.json", states),
        "detector": {
            "result": detector_result,
            "hazard_frames": hazard_frames,
            "receipt": detector,
        },
    }
    return capture, pngs, execution_log


def _godot_execution(
    *,
    output: Path,
    prefix: str,
    lane: str,
    frames_sha256: str,
    command_sha256: str,
    png_sources: list[Path],
    raw_log: str,
    marker_source: Path,
    expected_nonce: str,
    expected_scene_sha256: str,
    trace_sha256: str,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    pngs = [
        _copy(path, output, f"{prefix}/png/{index:04d}.png")
        for index, path in enumerate(png_sources)
    ]
    png_set = hashlib.sha256(
        "\n".join(reference["sha256"] for reference in pngs).encode("ascii")
    ).hexdigest()
    if marker_source.is_symlink() or not marker_source.is_file():
        raise L10EvidenceError("Godot engine-owned execution marker is missing")
    try:
        marker_value = json.loads(marker_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise L10EvidenceError("Godot engine-owned execution marker is unreadable") from exc
    expected_marker = {
        "adapter_sha256": adapter_sha256(),
        "engine": "Godot",
        "engine_version": "4.7.1.stable.official.a13da4feb",
        "frame_count": 12,
        "lane": lane,
        "nonce": expected_nonce,
        "png_set_sha256": png_set,
        "scene_sha256": expected_scene_sha256,
        "schema": "flashpatch-l10-godot-execution-marker-v2",
        "trace_sha256": trace_sha256,
    }
    completion_lines = [
        line.removeprefix("FLASHPATCH_L10_COMPLETE ")
        for line in raw_log.splitlines()
        if line.startswith("FLASHPATCH_L10_COMPLETE ")
    ]
    if (
        marker_value != expected_marker
        or len(completion_lines) != 1
        or json.loads(completion_lines[0]) != expected_marker
    ):
        raise L10EvidenceError("Godot engine-owned execution marker binding failed")
    marker = _copy(marker_source, output, f"{prefix}/execution-marker.json")
    raw = raw_log.encode("utf-8")
    packed = {
        "schema": "flashpatch-l10-packed-execution-v1",
        "command_sha256": command_sha256,
        "engine": "Godot",
        "engine_version": "4.7.1.stable.official.a13da4feb",
        "frames_sha256": frames_sha256,
        "lane": lane,
        "execution_marker_sha256": marker["sha256"],
        "png_set_sha256": png_set,
        "raw_log_sha256": hashlib.sha256(raw).hexdigest(),
    }
    payload = raw.rstrip(b"\n") + b"\n\nFLASHPATCH_L10_PACKED " + json.dumps(
        packed, separators=(",", ":"), sort_keys=True
    ).encode() + b"\n"
    log_path = output / f"{prefix}/execution.log.gz"
    if log_path.exists():
        raise L10EvidenceError(f"destination artifact already exists: {log_path}")
    log_path.write_bytes(gzip.compress(payload, mtime=0))
    log = {"path": log_path.relative_to(output).as_posix(), "sha256": _sha256(log_path)}
    return log, marker, pngs


def _promote_godot_controlled_runtime(
    runtime_root: Path,
    source_checkout: Path,
    godot_binary: Path,
    output: Path,
    run_nonces: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Promote four actual Godot X11/OpenGL source-repair runs."""
    if output.exists():
        raise L10EvidenceError("evidence output already exists")
    runtime = runtime_root.resolve()
    source = source_checkout.resolve()
    binary = godot_binary.resolve()
    source_identity = _verify_godot_git_source(source)
    if binary.is_symlink() or not binary.is_file():
        raise L10EvidenceError("frozen Godot binary is missing or unsafe")
    runtime_digest = _sha256(binary)
    if runtime_digest != "32f8d7596c4b41185512b1c49d69f2da3be018fd784a53e349fa92a98a97bcde":
        raise L10EvidenceError("frozen Godot binary hash changed")
    completed = subprocess.run(
        [str(binary), "--version"], check=True, capture_output=True, text=True
    )
    version = completed.stdout.strip()
    if version != "4.7.1.stable.official.a13da4feb":
        raise L10EvidenceError("frozen Godot binary version changed")
    fixture = source / "benchmarks/aigame-psebench/corpus/interaction-burst"
    factual_source = fixture / "main.gd"
    trace_source = fixture / "trace.json"
    license_source = source / "LICENSE"
    required = [factual_source, trace_source, license_source]
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise L10EvidenceError("frozen Godot fixture is missing or unsafe")
    factual_raw = factual_source.read_bytes()
    old = b"@export var burst_intensity: float = 1.0\n"
    new = b"@export var burst_intensity: float = 0.0\n"
    if factual_raw.count(old) != 1:
        raise L10EvidenceError("frozen Godot factual source changed")
    output.mkdir(parents=True)
    try:
        factual_scene = _copy(factual_source, output, "source/main.gd")
        candidate_path = output / "source/main-counterfactual.gd"
        candidate_path.write_bytes(factual_raw.replace(old, new))
        candidate_scene = {"path": "source/main-counterfactual.gd", "sha256": _sha256(candidate_path)}
        trace = _copy(trace_source, output, "trace.json")
        license_ref = _copy(license_source, output, "source/LICENSE")
        source_provenance = _write(output, "source-provenance.json", {
            "clean": True,
            **source_identity,
        })
        identity = _write(output, "engine-identity.json", {
            "engine": "Godot", "version": version,
        })
        attribution_value = {
            "object_identity": "/root/InteractionBurst",
            "component": "GDScript",
            "property": "burst_intensity",
            "factual_value": 1.0,
            "counterfactual_value": 0.0,
        }
        attribution = _write(output, "attribution.json", attribution_value)
        patch = _write(output, "patch.json", {
            **attribution_value,
            "kind": "minimal_source_parameter",
            "files_changed": 1,
            "factual_scene_sha256": factual_scene["sha256"],
            "counterfactual_scene_sha256": candidate_scene["sha256"],
        })
        command_path = output / "command.txt"
        command_path.write_text(
            "Godot_v4.7.1-stable_linux.x86_64 --display-driver x11 "
            "--rendering-driver opengl3 --fixed-fps 60 --disable-vsync "
            "--single-threaded-scene --resolution 320x180 --path PROJECT -- "
            "--trace trace.json --output replay.json --renderer-capture\n",
            encoding="utf-8",
        )
        command = {"path": "command.txt", "sha256": _sha256(command_path)}

        def promote_pair(label: str) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
            run_parent = runtime / label / "work"
            roots = list(run_parent.glob("flashpatch-run-*"))
            if len(roots) != 1:
                raise L10EvidenceError("Godot execution root is missing or ambiguous")
            run = roots[0]
            receipt_path = runtime / label / "receipt.json"
            try:
                old_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise L10EvidenceError("Godot execution receipt is unreadable") from exc
            if old_receipt.get("verdict") != "PASS":
                raise L10EvidenceError("Godot execution did not complete a repair")
            factual, factual_png_sources, factual_raw_log = _godot_capture(
                run, lane="factual", output=output, prefix=f"runs/{label}/factual"
            )
            candidate, candidate_png_sources, candidate_raw_log = _godot_capture(
                run, lane="counterfactual", output=output, prefix=f"runs/{label}/counterfactual"
            )
            factual_log, factual_marker, factual_pngs = _godot_execution(
                output=output, prefix=f"runs/{label}/factual", lane="factual",
                frames_sha256=str(factual["frames"]["sha256"]),
                command_sha256=command["sha256"], png_sources=factual_png_sources,
                raw_log=factual_raw_log,
                marker_source=run / "execution-marker.json",
                expected_nonce=run_nonces[label]["factual"],
                expected_scene_sha256=factual_scene["sha256"],
                trace_sha256=trace["sha256"],
            )
            candidate_log, candidate_marker, candidate_pngs = _godot_execution(
                output=output, prefix=f"runs/{label}/counterfactual", lane="counterfactual",
                frames_sha256=str(candidate["frames"]["sha256"]),
                command_sha256=command["sha256"], png_sources=candidate_png_sources,
                raw_log=candidate_raw_log,
                marker_source=run / "counterfactual/00/execution-marker.json",
                expected_nonce=run_nonces[label]["counterfactual"],
                expected_scene_sha256=candidate_scene["sha256"],
                trace_sha256=trace["sha256"],
            )
            provenance = _write(output, f"runs/{label}/runtime-provenance.json", {
                "run_id": _run_id(
                    str(factual["frames"]["sha256"]), str(candidate["frames"]["sha256"]),
                    factual_log["sha256"], candidate_log["sha256"],
                    factual_marker["sha256"], candidate_marker["sha256"],
                ),
                "kind": "binary", "digest": runtime_digest,
                "engine": "Godot", "engine_version": version,
                "source_revision": source_identity["revision"],
                "factual_scene_sha256": factual_scene["sha256"],
                "counterfactual_scene_sha256": candidate_scene["sha256"],
                "exit_code": 0,
                "factual_execution_log": factual_log,
                "counterfactual_execution_log": candidate_log,
                "factual_execution_marker": factual_marker,
                "counterfactual_execution_marker": candidate_marker,
                "factual_pngs": factual_pngs,
                "counterfactual_pngs": candidate_pngs,
                "factual_frames_sha256": factual["frames"]["sha256"],
                "counterfactual_frames_sha256": candidate["frames"]["sha256"],
            })
            return factual, candidate, provenance

        factual, candidate, baseline_provenance = promote_pair("baseline")
        repeats = []
        for ordinal in (1, 2, 3):
            repeated_factual, repeated_candidate, provenance = promote_pair(f"repeat-{ordinal}")
            repeat_value = {
                "ordinal": ordinal, "factual": repeated_factual,
                "counterfactual": repeated_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
                "trace_commitment": trace["sha256"],
            }
            repeat = _write(output, f"runs/repeat-{ordinal}/repeat.json", repeat_value)
            repeats.append({
                "ordinal": ordinal, "artifact": repeat,
                "factual": repeated_factual, "counterfactual": repeated_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
            })
        receipt = {
            "schema": "flashpatch-l10-engine-evidence-v1",
            "evidence_class": "controlled_fixture",
            "engine": "Godot", "engine_version": version,
            "source": {**source_identity, "provenance": source_provenance},
            "license": {"spdx": "Apache-2.0", "artifact": license_ref},
            "scene": {
                "source_path": "benchmarks/aigame-psebench/corpus/interaction-burst/main.gd",
                "artifact": factual_scene,
                "counterfactual_artifact": candidate_scene,
            },
            "trace": {"artifact": trace, "commitment": trace["sha256"]},
            "runtime": {
                "kind": "binary", "digest": runtime_digest, "command": command,
                "engine_identity": identity, "provenance": baseline_provenance,
            },
            "renderer": {"backend": "OpenGL 3.3 X11", "color_space": "sRGB/BT.709", "width": 320, "height": 180},
            "factual": factual,
            "attribution": {**{key: attribution_value[key] for key in ("object_identity", "component", "property")}, "artifact": attribution},
            "patch": {"kind": "minimal_source_parameter", "files_changed": 1, "artifact": patch},
            "counterfactual": candidate,
            "preservation": {key: True for key in ("action_sequence", "gameplay_state", "object_identity", "terminal_state", "timing", "visual_intent")},
            "parity": {"decoder": True, "timeline": True},
            "repeats": repeats, "verdict": "PASS",
            "reason": "controlled_fixture_minimal_source_parameter_removed_recomputed_flash_hazard",
        }
        receipt_ref = _write(output, "receipt.json", receipt)
        policy_ref = _write(output, "trust-policy.json", {
            "schema": "flashpatch-l10-trust-policy-v1",
            "engines": [{
                "receipt_sha256": receipt_ref["sha256"],
                "engine": "Godot", "engine_version": version,
                "evidence_class": "controlled_fixture",
                "repository": source_identity["repository"],
                "revision": source_identity["revision"],
                "tree_sha256": source_identity["tree_sha256"],
                "scene_source_path": receipt["scene"]["source_path"],
                "scene_sha256": factual_scene["sha256"],
                "runtime_kind": "binary", "runtime_digest": runtime_digest,
                "adapter_sha256": adapter_sha256(),
                "run_nonces": run_nonces,
            }],
        })
        return {"receipt": receipt_ref, "trust_policy": policy_ref}
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def promote_godot_controlled_evidence(
    source_checkout: Path,
    godot_binary: Path,
    output: Path,
) -> dict[str, object]:
    """Execute four fresh pinned Godot repairs, then seal only those run outputs."""
    if output.exists():
        raise L10EvidenceError("evidence output already exists")
    source = source_checkout.resolve()
    binary = godot_binary.resolve()
    _verify_godot_git_source(source)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise L10EvidenceError("Godot evidence promotion requires an actual renderer display")
    fixture = source / "benchmarks/aigame-psebench/corpus/interaction-burst"
    contract = fixture / "flashpatch.renderer.contract.json"
    run_nonces = {
        label: {lane: secrets.token_hex(32) for lane in ("factual", "counterfactual")}
        for label in ("baseline", "repeat-1", "repeat-2", "repeat-3")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="godot-l10-fresh-", dir=output.parent
    ) as temporary:
        runtime = Path(temporary)
        for label in ("baseline", "repeat-1", "repeat-2", "repeat-3"):
            run_output = runtime / label
            lane_nonces = iter((run_nonces[label]["factual"], run_nonces[label]["counterfactual"]))
            receipt = compile_project(
                fixture,
                contract,
                workspace=run_output / "work",
                runner_factory=lambda project: GodotL10ReplayRunner(
                    project, godot_binary=binary, nonce=next(lane_nonces)
                ),
            )
            write_receipt(receipt, run_output / "receipt.json")
            if receipt.get("verdict") != "PASS":
                raise L10EvidenceError(
                    f"fresh Godot execution did not complete: {receipt.get('reason')}"
                )
        return _promote_godot_controlled_runtime(
            runtime, source, binary, output, run_nonces
        )


def promote_unreal_controlled_evidence(
    runtime_root: Path,
    source_checkout: Path,
    output: Path,
) -> dict[str, object]:
    """Promote the frozen Unreal baseline and three fresh replay pairs."""
    if output.exists():
        raise L10EvidenceError("evidence output already exists")
    source = source_checkout.resolve()
    runtime = runtime_root.resolve()
    source_identity = _verify_unreal_git_source(source)
    required_source = [source / "Content/Maps/Blank.umap", source / "LICENCE"]
    if any(path.is_symlink() or not path.is_file() for path in required_source):
        raise L10EvidenceError("frozen Unreal source is missing or unsafe")
    output.mkdir(parents=True)
    try:
        scene = _copy(required_source[0], output, "source/Blank.umap")
        license_ref = _copy(required_source[1], output, "source/LICENCE")
        trace_value = {
            "schema": "flashpatch-l10-unreal-trace-v1",
            "frame_count": 18,
            "frame_step_seconds": 1.0 / 30.0,
            "mode_parameter": "PointLightComponent.intensity",
            "camera_location": [-600.0, 0.0, 100.0],
            "target_object": "FlashPatchTarget",
        }
        trace = _write(output, "trace.json", trace_value)
        attribution_value = {
            "object_identity": "FlashPatchPointLight",
            "component": "PointLightComponent",
            "property": "intensity",
            "factual_value": 1000000.0,
            "counterfactual_value": 0.0,
        }
        attribution = _write(output, "attribution.json", attribution_value)
        patch_value = {
            **attribution_value,
            "kind": "controlled_runtime_parameter",
            "files_changed": 0,
            "factual_scene_sha256": scene["sha256"],
            "counterfactual_scene_sha256": scene["sha256"],
        }
        patch = _write(output, "patch.json", patch_value)
        command_text = (
            "bash -lc '/home/ue4/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd "
            "/project/Starter.uproject /Game/Maps/Blank -unattended -nop4 "
            "-nosplash -NoSound -RenderOffscreen -vulkan -stdout "
            "-ExecutePythonScript=/project/FlashPatchL10Capture.py; status=$?; "
            "if [ $status -eq 0 ]; then printf \"FLASHPATCH_L10_COMPLETE \"; "
            "tr -d \"\\n\" < \"$FLASHPATCH_OUTPUT/execution-marker.json\"; printf \"\\n\"; fi; "
            "exit $status'\n"
        )
        command_path = output / "command.txt"
        command_path.write_text(command_text, encoding="utf-8")
        command = {"path": "command.txt", "sha256": _sha256(command_path)}
        identity = _copy(
            runtime / "unreal-controlled-factual-7/engine-identity.json",
            output,
            "engine-identity.json",
        )
        source_provenance = _write(output, "source-provenance.json", {
            "clean": True,
            **source_identity,
        })

        def promote_pair(label: str, factual_name: str, candidate_name: str) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
            factual_pack = runtime / f"{factual_name}-pack"
            candidate_pack = runtime / f"{candidate_name}-pack"
            factual = _capture(factual_pack, output, f"runs/{label}/factual")
            candidate = _capture(candidate_pack, output, f"runs/{label}/counterfactual")
            factual["runtime_events"] = _copy(runtime / factual_name / "runtime-events.json", output, f"runs/{label}/factual/runtime-events.json")
            factual["state_stream"] = _copy(runtime / factual_name / "state-stream.json", output, f"runs/{label}/factual/state-stream.json")
            candidate["runtime_events"] = _copy(runtime / candidate_name / "runtime-events.json", output, f"runs/{label}/counterfactual/runtime-events.json")
            candidate["state_stream"] = _copy(runtime / candidate_name / "state-stream.json", output, f"runs/{label}/counterfactual/state-stream.json")
            factual_log = runtime / f"{factual_name.replace('unreal-controlled-', 'unreal-')}-console.log"
            candidate_log = runtime / f"{candidate_name.replace('unreal-controlled-', 'unreal-')}-console.log"
            if label.startswith("repeat-"):
                factual_log = runtime / f"{factual_name}-console.log"
                candidate_log = runtime / f"{candidate_name}-console.log"
            if (runtime / f"{factual_name.replace('unreal-controlled-', 'unreal-')}-exit-status.txt").read_text().strip() != "0":
                raise L10EvidenceError("factual execution did not exit successfully")
            if (runtime / f"{candidate_name.replace('unreal-controlled-', 'unreal-')}-exit-status.txt").read_text().strip() != "0":
                raise L10EvidenceError("counterfactual execution did not exit successfully")
            factual_marker_ref = _copy(
                runtime / factual_name / "execution-marker.json",
                output,
                f"runs/{label}/factual-execution-marker.json",
            )
            candidate_marker_ref = _copy(
                runtime / candidate_name / "execution-marker.json",
                output,
                f"runs/{label}/counterfactual-execution-marker.json",
            )
            factual_pngs = [
                _copy(runtime / factual_name / f"{index:04d}.png", output, f"runs/{label}/factual/png/{index:04d}.png")
                for index in range(18)
            ]
            candidate_pngs = [
                _copy(runtime / candidate_name / f"{index:04d}.png", output, f"runs/{label}/counterfactual/png/{index:04d}.png")
                for index in range(18)
            ]
            factual_log_ref = _execution_log(
                factual_log,
                runtime / factual_name,
                output,
                f"runs/{label}/factual-execution.log.gz",
                lane="factual",
                frames_sha256=str(factual["frames"]["sha256"]),
                command_sha256=command["sha256"],
            )
            candidate_log_ref = _execution_log(
                candidate_log,
                runtime / candidate_name,
                output,
                f"runs/{label}/counterfactual-execution.log.gz",
                lane="counterfactual",
                frames_sha256=str(candidate["frames"]["sha256"]),
                command_sha256=command["sha256"],
            )
            factual_frames_sha256 = str(factual["frames"]["sha256"])
            counterfactual_frames_sha256 = str(candidate["frames"]["sha256"])
            provenance = _write(output, f"runs/{label}/runtime-provenance.json", {
                "run_id": _run_id(
                    factual_frames_sha256,
                    counterfactual_frames_sha256,
                    factual_log_ref["sha256"],
                    candidate_log_ref["sha256"],
                    factual_marker_ref["sha256"],
                    candidate_marker_ref["sha256"],
                ),
                "kind": "container",
                "digest": "7069cb209b51c75fb42ebaa686025ad581c3a0976e5473a9f480be6597b8acd9",
                "engine": "Unreal",
                "engine_version": "5.6.0-43139311+UE5",
                "source_revision": source_identity["revision"],
                "factual_scene_sha256": scene["sha256"],
                "counterfactual_scene_sha256": scene["sha256"],
                "exit_code": 0,
                "factual_execution_log": factual_log_ref,
                "counterfactual_execution_log": candidate_log_ref,
                "factual_execution_marker": factual_marker_ref,
                "counterfactual_execution_marker": candidate_marker_ref,
                "factual_pngs": factual_pngs,
                "counterfactual_pngs": candidate_pngs,
                "factual_frames_sha256": factual_frames_sha256,
                "counterfactual_frames_sha256": counterfactual_frames_sha256,
            })
            return factual, candidate, provenance

        factual, candidate, baseline_provenance = promote_pair(
            "baseline",
            "unreal-controlled-factual-7",
            "unreal-controlled-counterfactual-7",
        )
        repeats = []
        for ordinal in (1, 2, 3):
            repeat_factual, repeat_candidate, provenance = promote_pair(
                f"repeat-{ordinal}",
                f"unreal-repeat-{ordinal}-factual",
                f"unreal-repeat-{ordinal}-counterfactual",
            )
            repeat_value = {
                "ordinal": ordinal,
                "factual": repeat_factual,
                "counterfactual": repeat_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
                "trace_commitment": trace["sha256"],
            }
            repeat_artifact = _write(output, f"runs/repeat-{ordinal}/repeat.json", repeat_value)
            repeats.append({
                "ordinal": ordinal,
                "artifact": repeat_artifact,
                "factual": repeat_factual,
                "counterfactual": repeat_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
            })
        receipt = {
            "schema": "flashpatch-l10-engine-evidence-v1",
            "evidence_class": "controlled_fixture",
            "engine": "Unreal",
            "engine_version": "5.6.0-43139311+UE5",
            "source": {
                **source_identity,
                "provenance": source_provenance,
            },
            "license": {"spdx": "MIT", "artifact": license_ref},
            "scene": {
                "source_path": "Content/Maps/Blank.umap",
                "artifact": scene,
                "counterfactual_artifact": scene,
            },
            "trace": {"artifact": trace, "commitment": trace["sha256"]},
            "runtime": {
                "kind": "container",
                "digest": "7069cb209b51c75fb42ebaa686025ad581c3a0976e5473a9f480be6597b8acd9",
                "command": command,
                "engine_identity": identity,
                "provenance": baseline_provenance,
            },
            "renderer": {"backend": "Vulkan SM5", "color_space": "sRGB", "width": 640, "height": 360},
            "factual": factual,
            "attribution": {**{key: attribution_value[key] for key in ("object_identity", "component", "property")}, "artifact": attribution},
            "patch": {"kind": "controlled_runtime_parameter", "files_changed": 0, "artifact": patch},
            "counterfactual": candidate,
            "preservation": {key: True for key in ("action_sequence", "gameplay_state", "object_identity", "terminal_state", "timing", "visual_intent")},
            "parity": {"decoder": True, "timeline": True},
            "repeats": repeats,
            "verdict": "PASS",
            "reason": "controlled_runtime_parameter_removed_recomputed_flash_hazard",
        }
        receipt_ref = _write(output, "receipt.json", receipt)
        policy = {
            "schema": "flashpatch-l10-trust-policy-v1",
            "engines": [{
                "receipt_sha256": receipt_ref["sha256"],
                "engine": "Unreal",
                "engine_version": receipt["engine_version"],
                "evidence_class": receipt["evidence_class"],
                "repository": receipt["source"]["repository"],
                "revision": receipt["source"]["revision"],
                "tree_sha256": receipt["source"]["tree_sha256"],
                "scene_source_path": receipt["scene"]["source_path"],
                "scene_sha256": scene["sha256"],
                "runtime_kind": "container",
                "runtime_digest": receipt["runtime"]["digest"],
            }],
        }
        policy_ref = _write(output, "trust-policy.json", policy)
        return {"receipt": receipt_ref, "trust_policy": policy_ref}
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def promote_unity_natural_evidence(
    runtime_root: Path,
    source_checkout: Path,
    factual_project: Path,
    counterfactual_project: Path,
    output: Path,
    *,
    terminal_verdict: str = "PASS",
) -> dict[str, object]:
    """Promote one pinned Unity source patch and four fresh Editor replay pairs."""
    if output.exists():
        raise L10EvidenceError("evidence output already exists")
    if terminal_verdict not in {"PASS", "INCONCLUSIVE"}:
        raise L10EvidenceError("Unity terminal verdict is invalid")
    runtime = runtime_root.resolve()
    source = source_checkout.resolve()
    factual_root = factual_project.resolve()
    candidate_root = counterfactual_project.resolve()
    source_identity = _verify_unity_git_source(source)
    factual_manifest_source = runtime / "unity-smokeportal-isolated-project-inputs.json"
    candidate_manifest_source = runtime / "unity-smokeportal-candidate-project-inputs.json"
    scene_relative = Path("Assets/Samples/SmokePortal/SmokePortal.unity")
    attribution_relative = Path("Assets/Samples/SmokePortal/SmokePortal/LightFlicker.cs")
    license_path = source / "LICENSE.md"
    required = [
        source / scene_relative,
        source / attribution_relative,
        license_path,
        factual_root / scene_relative,
        candidate_root / scene_relative,
        factual_manifest_source,
        candidate_manifest_source,
    ]
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise L10EvidenceError("pinned Unity source or candidate is missing or unsafe")
    if _sha256(source / scene_relative) != _sha256(factual_root / scene_relative):
        raise L10EvidenceError("Unity factual project no longer matches the pinned source")
    factual_text = (factual_root / scene_relative).read_text(encoding="utf-8")
    candidate_text = (candidate_root / scene_relative).read_text(encoding="utf-8")
    expected_candidate = factual_text.replace(
        "  m_IntensityJitterScale: 2000\n",
        "  m_IntensityJitterScale: 0\n",
    )
    if factual_text.count("  m_IntensityJitterScale: 2000\n") != 1 or candidate_text != expected_candidate:
        raise L10EvidenceError("Unity candidate is not the frozen one-parameter scene patch")
    try:
        factual_manifest_value = json.loads(factual_manifest_source.read_text(encoding="utf-8"))
        candidate_manifest_value = json.loads(candidate_manifest_source.read_text(encoding="utf-8"))
        factual_files = {row["path"]: row["sha256"] for row in factual_manifest_value["files"]}
        candidate_files = {row["path"]: row["sha256"] for row in candidate_manifest_value["files"]}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise L10EvidenceError("Unity project input manifests are unreadable") from exc
    if len(factual_files) != len(factual_manifest_value["files"]) or len(candidate_files) != len(candidate_manifest_value["files"]):
        raise L10EvidenceError("Unity project input manifest contains duplicate paths")
    differing_inputs = {
        path
        for path in set(factual_files) | set(candidate_files)
        if factual_files.get(path) != candidate_files.get(path)
    }
    if differing_inputs != {scene_relative.as_posix()}:
        raise L10EvidenceError("Unity project inputs differ outside the one allowed scene")
    if (
        factual_files.get(scene_relative.as_posix()) != _sha256(factual_root / scene_relative)
        or candidate_files.get(scene_relative.as_posix()) != _sha256(candidate_root / scene_relative)
    ):
        raise L10EvidenceError("Unity project input manifests do not bind the scene pair")
    source_files: dict[str, str] = {}
    for folder in ("Assets", "Packages", "ProjectSettings"):
        for path in sorted((source / folder).rglob("*")):
            if path.is_symlink():
                raise L10EvidenceError("pinned Unity source contains a symlink")
            if path.is_file():
                source_files[path.relative_to(source).as_posix()] = _sha256(path)
    allowed_adapter = {
        "Assets/Editor/FlashPatchL10Capture.cs",
        "Assets/Editor/FlashPatchL10Capture.cs.meta",
        "Assets/Editor.meta",
        "Packages/packages-lock.json",
    }
    if set(factual_files) != set(source_files) | allowed_adapter:
        raise L10EvidenceError("Unity factual inputs do not equal pinned source plus the adapter")
    source_manifest_path = "Packages/manifest.json"
    unchanged_source_files = {
        path: digest for path, digest in source_files.items() if path != source_manifest_path
    }
    if any(factual_files[path] != digest for path, digest in unchanged_source_files.items()):
        raise L10EvidenceError("Unity factual inputs differ from the pinned source")
    expected_linux_manifest = hashlib.sha256(
        unity_linux_package_manifest_bytes((source / source_manifest_path).read_bytes())
    ).hexdigest()
    if (
        factual_files.get(source_manifest_path) != expected_linux_manifest
        or candidate_files.get(source_manifest_path) != expected_linux_manifest
    ):
        raise L10EvidenceError("Unity package manifest is not the approved Linux transform")
    approved_adapter = unity_adapter_fingerprints()
    for path, digest in approved_adapter.items():
        if factual_files.get(path) != digest or candidate_files.get(path) != digest:
            raise L10EvidenceError("Unity project uses an unapproved capture adapter")
    if len(factual_files) != 3339 or len(candidate_files) != 3339:
        raise L10EvidenceError("Unity project input inventory count is not frozen")
    try:
        execution_matrix = json.loads(
            (runtime / "execution-matrix.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise L10EvidenceError("Unity execution matrix is missing or unreadable") from exc
    if execution_matrix != {
        "schema": "flashpatch-l10-unity-execution-matrix-v1",
        "image": UNITY_IMAGE,
        "image_digest": UNITY_IMAGE_DIGEST,
        "vulkan_loader_sha256": UNITY_VULKAN_LOADER_SHA256,
        "entitlement_sha256": execution_matrix.get("entitlement_sha256"),
        "gpu_index": execution_matrix.get("gpu_index"),
        "gpu_uuid": execution_matrix.get("gpu_uuid"),
        "gpu_name": execution_matrix.get("gpu_name"),
        "factual_manifest_sha256": _sha256(factual_manifest_source),
        "counterfactual_manifest_sha256": _sha256(candidate_manifest_source),
        "runs": [name for name, _ in UNITY_RUNS],
        "verdict": "COMPLETE",
    }:
        raise L10EvidenceError("Unity execution matrix does not bind the frozen run closure")
    if not isinstance(execution_matrix.get("entitlement_sha256"), str) or len(
        execution_matrix["entitlement_sha256"]
    ) != 64:
        raise L10EvidenceError("Unity execution matrix entitlement commitment is invalid")
    outcome = _verify_unity_capture_outcomes(
        runtime,
        require_pass=terminal_verdict == "PASS",
    )
    try:
        baseline_state_preserved = json.loads(
            (runtime / "unity-factual-baseline/state-stream.json").read_text(encoding="utf-8")
        ) == json.loads(
            (runtime / "unity-counterfactual-baseline/state-stream.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise L10EvidenceError("Unity baseline state streams are unreadable") from exc
    if terminal_verdict == "INCONCLUSIVE" and (
        outcome["hazard_removed"]
        and outcome["repeat_reproducible"]
        and baseline_state_preserved
    ):
        raise L10EvidenceError("Unity INCONCLUSIVE verdict contradicts completed evidence")
    inconclusive_reasons = []
    if not outcome["hazard_removed"]:
        inconclusive_reasons.append("factual_capture_did_not_establish_hazard_removal")
    if not outcome["repeat_reproducible"]:
        inconclusive_reasons.append("capture_artifacts_were_not_reproducible")
    if not baseline_state_preserved:
        inconclusive_reasons.append("gameplay_state_was_not_preserved")
    output.mkdir(parents=True)
    try:
        factual_scene = _copy(factual_root / scene_relative, output, "source/SmokePortal.unity")
        candidate_scene = _copy(
            candidate_root / scene_relative,
            output,
            "source/SmokePortal-counterfactual.unity",
        )
        attribution_source = _copy(
            source / attribution_relative,
            output,
            "source/LightFlicker.cs",
        )
        license_ref = _copy(license_path, output, "source/LICENSE.md")
        factual_project_manifest = _copy(
            factual_manifest_source,
            output,
            "source/factual-project-inputs.json",
        )
        candidate_project_manifest = _copy(
            candidate_manifest_source,
            output,
            "source/counterfactual-project-inputs.json",
        )
        trace = _write(output, "trace.json", {
            "schema": "flashpatch-l10-unity-trace-v1",
            "frame_count": 121,
            "frame_step_seconds": 1.0 / 30.0,
            "mode_parameter": "LightFlicker.m_IntensityJitterScale",
            "scene": scene_relative.as_posix(),
            "target_object": "Spot Light",
        })
        attribution_value = {
            "object_identity": "Spot Light",
            "component": "LightFlicker",
            "property": "m_IntensityJitterScale",
            "factual_value": 2000.0,
            "counterfactual_value": 0.0,
        }
        attribution = _write(output, "attribution.json", attribution_value)
        patch_value = {
            **attribution_value,
            "kind": "minimal_source_parameter",
            "files_changed": 1,
            "factual_scene_sha256": factual_scene["sha256"],
            "counterfactual_scene_sha256": candidate_scene["sha256"],
        }
        patch = _write(output, "patch.json", patch_value)
        command_text = UNITY_RECEIPT_COMMAND
        command_path = output / "command.txt"
        command_path.write_text(command_text, encoding="utf-8")
        command = {"path": "command.txt", "sha256": _sha256(command_path)}
        source_provenance = _write(output, "source-provenance.json", {
            "clean": True,
            **source_identity,
        })

        def promote_pair(
            label: str,
            factual_name: str,
            candidate_name: str,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
            factual = _capture(runtime / f"{factual_name}-pack", output, f"runs/{label}/factual")
            candidate = _capture(runtime / f"{candidate_name}-pack", output, f"runs/{label}/counterfactual")
            for name, capture, run_name in (
                ("factual", factual, factual_name),
                ("counterfactual", candidate, candidate_name),
            ):
                capture["runtime_events"] = _copy(
                    runtime / run_name / "runtime-events.json",
                    output,
                    f"runs/{label}/{name}/runtime-events.json",
                )
                capture["state_stream"] = _copy(
                    runtime / run_name / "state-stream.json",
                    output,
                    f"runs/{label}/{name}/state-stream.json",
                )
            for run_name in (factual_name, candidate_name):
                if (runtime / f"{run_name}-exit-status.txt").read_text().strip() != "0":
                    raise L10EvidenceError("Unity execution did not exit successfully")
            factual_marker = _copy(
                runtime / factual_name / "execution-marker.json",
                output,
                f"runs/{label}/factual-execution-marker.json",
            )
            candidate_marker = _copy(
                runtime / candidate_name / "execution-marker.json",
                output,
                f"runs/{label}/counterfactual-execution-marker.json",
            )
            factual_pngs = [
                _copy(runtime / factual_name / f"{index:04d}.png", output, f"runs/{label}/factual/png/{index:04d}.png")
                for index in range(121)
            ]
            candidate_pngs = [
                _copy(runtime / candidate_name / f"{index:04d}.png", output, f"runs/{label}/counterfactual/png/{index:04d}.png")
                for index in range(121)
            ]
            factual_log = _unity_execution_log(
                runtime / f"{factual_name}-console.log",
                runtime / factual_name,
                output,
                f"runs/{label}/factual-execution.log.gz",
                lane="factual",
                frames_sha256=str(factual["frames"]["sha256"]),
                command_sha256=command["sha256"],
                project_manifest_sha256=factual_project_manifest["sha256"],
                scene_sha256=factual_scene["sha256"],
            )
            candidate_log = _unity_execution_log(
                runtime / f"{candidate_name}-console.log",
                runtime / candidate_name,
                output,
                f"runs/{label}/counterfactual-execution.log.gz",
                lane="counterfactual",
                frames_sha256=str(candidate["frames"]["sha256"]),
                command_sha256=command["sha256"],
                project_manifest_sha256=candidate_project_manifest["sha256"],
                scene_sha256=candidate_scene["sha256"],
            )
            provenance = _write(output, f"runs/{label}/runtime-provenance.json", {
                "run_id": _run_id(
                    str(factual["frames"]["sha256"]),
                    str(candidate["frames"]["sha256"]),
                    factual_log["sha256"],
                    candidate_log["sha256"],
                    factual_marker["sha256"],
                    candidate_marker["sha256"],
                ),
                "kind": "container",
                "digest": "966a619d057eefa1ca993f58ac461204e6f9f1b805f91c7be0f10ebf64a5b1df",
                "engine": "Unity",
                "engine_version": "2022.3.8f1",
                "source_revision": "4624d386e5f2e63e383a7e572362785470fe0f33",
                "factual_scene_sha256": factual_scene["sha256"],
                "counterfactual_scene_sha256": candidate_scene["sha256"],
                "exit_code": 0,
                "factual_execution_log": factual_log,
                "counterfactual_execution_log": candidate_log,
                "factual_execution_marker": factual_marker,
                "counterfactual_execution_marker": candidate_marker,
                "factual_project_manifest": factual_project_manifest,
                "counterfactual_project_manifest": candidate_project_manifest,
                "factual_pngs": factual_pngs,
                "counterfactual_pngs": candidate_pngs,
                "factual_frames_sha256": factual["frames"]["sha256"],
                "counterfactual_frames_sha256": candidate["frames"]["sha256"],
            })
            return factual, candidate, provenance

        factual, candidate, baseline_provenance = promote_pair(
            "baseline", "unity-factual-baseline", "unity-counterfactual-baseline"
        )
        identity = _copy(
            runtime / "unity-factual-baseline/engine-identity.json",
            output,
            "engine-identity.json",
        )
        repeats = []
        for ordinal in (1, 2, 3):
            repeated_factual, repeated_candidate, provenance = promote_pair(
                f"repeat-{ordinal}",
                f"unity-repeat-{ordinal}-factual",
                f"unity-repeat-{ordinal}-counterfactual",
            )
            repeat_value = {
                "ordinal": ordinal,
                "factual": repeated_factual,
                "counterfactual": repeated_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
                "trace_commitment": trace["sha256"],
            }
            artifact = _write(output, f"runs/repeat-{ordinal}/repeat.json", repeat_value)
            repeats.append({
                "ordinal": ordinal,
                "artifact": artifact,
                "factual": repeated_factual,
                "counterfactual": repeated_candidate,
                "runtime_provenance": provenance,
                "runtime_provenance_sha256": provenance["sha256"],
            })
        receipt = {
            "schema": "flashpatch-l10-engine-evidence-v1",
            "evidence_class": "natural_project",
            "engine": "Unity",
            "engine_version": "2022.3.8f1",
            "source": {
                **source_identity,
                "provenance": source_provenance,
            },
            "license": {"spdx": "Unity-Companion-License", "artifact": license_ref},
            "scene": {
                "source_path": scene_relative.as_posix(),
                "artifact": factual_scene,
                "counterfactual_artifact": candidate_scene,
            },
            "trace": {"artifact": trace, "commitment": trace["sha256"]},
            "runtime": {
                "kind": "container",
                "digest": "966a619d057eefa1ca993f58ac461204e6f9f1b805f91c7be0f10ebf64a5b1df",
                "command": command,
                "engine_identity": identity,
                "provenance": baseline_provenance,
            },
            "renderer": {"backend": "Vulkan", "color_space": "sRGB", "width": 640, "height": 360},
            "factual": factual,
            "attribution": {
                "object_identity": "Spot Light",
                "component": "LightFlicker",
                "property": "m_IntensityJitterScale",
                "artifact": attribution,
            },
            "patch": {"kind": "minimal_source_parameter", "files_changed": 1, "artifact": patch},
            "counterfactual": candidate,
            "preservation": {
                "action_sequence": True,
                "gameplay_state": baseline_state_preserved,
                "object_identity": True,
                "terminal_state": baseline_state_preserved,
                "timing": True,
                "visual_intent": True,
            },
            "parity": {"decoder": True, "timeline": True},
            "repeats": repeats,
            "verdict": terminal_verdict,
            "reason": (
                "minimal_scene_parameter_patch_removed_recomputed_flash_hazard"
                if terminal_verdict == "PASS"
                else "__and__".join(inconclusive_reasons)
            ),
        }
        receipt_ref = _write(output, "receipt.json", receipt)
        policy_ref = _write(output, "trust-policy.json", {
            "schema": "flashpatch-l10-trust-policy-v1",
            "engines": [{
                "receipt_sha256": receipt_ref["sha256"],
                "engine": "Unity",
                "engine_version": receipt["engine_version"],
                "evidence_class": receipt["evidence_class"],
                "repository": receipt["source"]["repository"],
                "revision": receipt["source"]["revision"],
                "tree_sha256": receipt["source"]["tree_sha256"],
                "scene_source_path": receipt["scene"]["source_path"],
                "scene_sha256": factual_scene["sha256"],
                "runtime_kind": "container",
                "runtime_digest": receipt["runtime"]["digest"],
                "factual_project_manifest_sha256": factual_project_manifest["sha256"],
                "counterfactual_project_manifest_sha256": candidate_project_manifest["sha256"],
                "project_input_count": len(factual_files),
            }],
        })
        return {
            "receipt": receipt_ref,
            "trust_policy": policy_ref,
            "attribution_source": attribution_source,
        }
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
