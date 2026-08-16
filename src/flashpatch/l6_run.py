from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .l6_authority import L6_PREFLIGHT_PINS, L6PreflightPins


PREFLIGHT_SCHEMA = "flashpatch-l6-preflight-v1"
PREFLIGHT_CHECKPOINT_SCHEMA = "flashpatch-l6-preflight-checkpoint-v1"
EXECUTION_SCHEMA = "flashpatch-l6-sparta-execution-v1"
EXECUTION_CHECKPOINT_SCHEMA = "flashpatch-l6-sparta-execution-checkpoint-v1"
L6_REPEAT_COUNT = 3


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _run_command(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_observation(upstream: Path, *arguments: str) -> dict[str, Any]:
    return _run_command(("git", "-C", str(upstream), *arguments))


def _normalize_repository_url(value: str) -> str:
    """Compare the canonical HTTPS repository identity, not a clone suffix."""

    return value.rstrip("/").removesuffix(".git")


def _absolute_lexical(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _symlink_components(path: Path) -> list[str]:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    symlinks: list[str] = []
    for part in absolute.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                symlinks.append(str(current))
        except OSError:
            continue
    return symlinks


def _project_symlinks(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    symlinks: list[str] = []
    try:
        for directory, names, files in os.walk(root, followlinks=False):
            relative_directory = Path(directory).relative_to(root)
            if relative_directory == Path(".") and ".git" in names:
                names.remove(".git")
            for name in sorted((*names, *files)):
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    symlinks.append(candidate.relative_to(root).as_posix())
    except OSError as exc:
        symlinks.append(f"<scan-error:{exc}>")
    return sorted(symlinks)


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    *,
    expected: Any,
    observed: Any,
    passed: bool,
) -> None:
    checks[name] = {
        "expected": expected,
        "observed": observed,
        "status": "PASS" if passed else "INCONCLUSIVE",
    }


def _create_fresh_run_root(run_root: Path) -> Path:
    run_root = _absolute_lexical(run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir()
    return run_root


def run_preflight(
    upstream: Path | str,
    godot: Path | str,
    run_root: Path | str,
    *,
    pins: L6PreflightPins = L6_PREFLIGHT_PINS,
) -> dict[str, Any]:
    """Measure every L6 input pin and seal an immutable preflight receipt.

    The Godot executable is invoked only for ``--version`` and only after its
    path, file type, executable bit, symlink status, and digest have passed.
    No import or replay command is reachable from this preflight function.
    """

    supplied_upstream = Path(upstream)
    supplied_godot = Path(godot)
    root = _create_fresh_run_root(Path(run_root))
    checks: dict[str, dict[str, Any]] = {}

    upstream_lexical = _absolute_lexical(supplied_upstream)
    godot_lexical = _absolute_lexical(supplied_godot)
    _check(
        checks,
        "upstream_path",
        expected=pins.local_checkout,
        observed=str(upstream_lexical),
        passed=supplied_upstream.is_absolute()
        and str(supplied_upstream) == pins.local_checkout,
    )

    top_level = _git_observation(upstream_lexical, "rev-parse", "--show-toplevel")
    _check(
        checks,
        "git_top_level",
        expected=pins.local_checkout,
        observed=top_level,
        passed=top_level["returncode"] == 0
        and top_level["stdout"] == pins.local_checkout,
    )

    origin = _git_observation(upstream_lexical, "config", "--get", "remote.origin.url")
    _check(
        checks,
        "git_origin",
        expected=pins.repository,
        observed=origin,
        passed=(
            origin["returncode"] == 0
            and _normalize_repository_url(origin["stdout"])
            == _normalize_repository_url(pins.repository)
        ),
    )

    revision = _git_observation(upstream_lexical, "rev-parse", "HEAD")
    _check(
        checks,
        "git_revision",
        expected=pins.revision,
        observed=revision,
        passed=revision["returncode"] == 0 and revision["stdout"] == pins.revision,
    )

    status_result = _git_observation(
        upstream_lexical, "status", "--porcelain=v1", "--untracked-files=all"
    )
    _check(
        checks,
        "git_clean_tree",
        expected={"returncode": 0, "stdout": ""},
        observed=status_result,
        passed=status_result["returncode"] == 0 and status_result["stdout"] == "",
    )

    license_hash = _sha256_file(upstream_lexical / "LICENSE")
    _check(
        checks,
        "license_sha256",
        expected=pins.license_sha256,
        observed=license_hash,
        passed=license_hash == pins.license_sha256,
    )

    required_inputs = {
        relative: (upstream_lexical / relative).is_file()
        for relative in pins.required_inputs
    }
    _check(
        checks,
        "required_project_inputs",
        expected={relative: True for relative in pins.required_inputs},
        observed=required_inputs,
        passed=all(required_inputs.values()),
    )
    entry_relative = pins.entry_scene.removeprefix("res://")
    entry_exists = (upstream_lexical / entry_relative).is_file()
    _check(
        checks,
        "entry_scene",
        expected={"resource_path": pins.entry_scene, "exists": True},
        observed={"resource_path": pins.entry_scene, "exists": entry_exists},
        passed=pins.entry_scene.startswith("res://") and entry_exists,
    )

    symlink_observation = {
        "upstream_path_components": _symlink_components(supplied_upstream),
        "project_tree": _project_symlinks(upstream_lexical),
        "godot_path_components": _symlink_components(supplied_godot),
    }
    _check(
        checks,
        "symlink_free_inputs",
        expected={
            "upstream_path_components": [],
            "project_tree": [],
            "godot_path_components": [],
        },
        observed=symlink_observation,
        passed=not any(symlink_observation.values()),
    )

    try:
        canonical_godot = str(godot_lexical.resolve(strict=True))
    except OSError:
        canonical_godot = None
    godot_path_ok = (
        supplied_godot.is_absolute()
        and str(supplied_godot) == pins.godot_binary
        and canonical_godot == pins.godot_binary
    )
    _check(
        checks,
        "godot_canonical_path",
        expected=pins.godot_binary,
        observed={"supplied": str(godot_lexical), "resolved": canonical_godot},
        passed=godot_path_ok,
    )

    try:
        godot_mode = godot_lexical.stat().st_mode
        godot_regular = stat.S_ISREG(godot_mode)
        godot_executable = bool(godot_mode & 0o111) and os.access(godot_lexical, os.X_OK)
    except OSError:
        godot_regular = False
        godot_executable = False
    _check(
        checks,
        "godot_regular_executable",
        expected={"regular": True, "executable": True},
        observed={"regular": godot_regular, "executable": godot_executable},
        passed=godot_regular and godot_executable,
    )

    godot_hash = _sha256_file(godot_lexical)
    _check(
        checks,
        "godot_sha256",
        expected=pins.godot_binary_sha256,
        observed=godot_hash,
        passed=godot_hash == pins.godot_binary_sha256,
    )

    may_measure_version = all(
        checks[name]["status"] == "PASS"
        for name in (
            "symlink_free_inputs",
            "godot_canonical_path",
            "godot_regular_executable",
            "godot_sha256",
        )
    )
    if may_measure_version:
        version = _run_command((str(godot_lexical), "--version"))
    else:
        version = {
            "returncode": None,
            "stdout": "",
            "stderr": "measurement blocked by failed Godot trust gate",
        }
    _check(
        checks,
        "godot_version",
        expected=pins.godot_version,
        observed=version,
        passed=version["returncode"] == 0 and version["stdout"] == pins.godot_version,
    )

    passed = all(check["status"] == "PASS" for check in checks.values())
    verdict = "PASS" if passed else "INCONCLUSIVE"
    expected_pin_payload = {
        "repository": pins.repository,
        "revision": pins.revision,
        "license_sha256": pins.license_sha256,
        "local_checkout": pins.local_checkout,
        "entry_scene": pins.entry_scene,
        "required_inputs": list(pins.required_inputs),
        "godot_binary": pins.godot_binary,
        "godot_binary_sha256": pins.godot_binary_sha256,
        "godot_version": pins.godot_version,
        "xvfb_screen": pins.xvfb_screen,
        "rendering_driver": pins.rendering_driver,
        "fixed_fps": pins.fixed_fps,
        "capture_ticks": pins.capture_ticks,
        "timeout_seconds": pins.timeout_seconds,
    }
    receipt: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "execution_id": root.name,
        "preflight_input_sha256": _sha256_bytes(_canonical_json(expected_pin_payload)),
        "verdict": verdict,
        "preflight_verdict": verdict,
        "upstream_product_verdict": "INCONCLUSIVE",
        "controlled_mutation": False,
        "upstream_defect": False,
        "replay_allowed": passed,
        "phase": "PREFLIGHT_SEALED",
        "pins": expected_pin_payload,
        "checks": checks,
        "replay_contract": {
            "entry_scene": pins.entry_scene,
            "display": ["xvfb-run", "-a", "-s", f"-screen 0 {pins.xvfb_screen}"],
            "godot_arguments": [
                "--rendering-driver",
                pins.rendering_driver,
                "--fixed-fps",
                str(pins.fixed_fps),
            ],
            "capture_ticks": pins.capture_ticks,
            "timeout_seconds": pins.timeout_seconds,
        },
    }
    receipt_bytes = _canonical_json(receipt)
    (root / "preflight.json").write_bytes(receipt_bytes)
    checkpoint = {
        "schema": PREFLIGHT_CHECKPOINT_SCHEMA,
        "phase": "PREFLIGHT_SEALED",
        "preflight_verdict": verdict,
        "preflight_receipt_sha256": _sha256_bytes(receipt_bytes),
    }
    (root / "preflight-checkpoint.json").write_bytes(_canonical_json(checkpoint))
    return receipt


def _owned_file(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() and root.resolve() in resolved.parents else None


def _load_owned_json(root: Path, value: object) -> tuple[Path | None, dict[str, Any] | None]:
    path = _owned_file(root, value)
    if path is None:
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, None
    return path, payload if isinstance(payload, dict) else None


def _evidence_gap(gaps: list[dict[str, str]], gate: str, reason: str) -> None:
    item = {"gate": gate, "reason": reason}
    if item not in gaps:
        gaps.append(item)


def _renderer_color_space_gap(
    label: str,
    capture: dict[str, object],
) -> str | None:
    """Return a precise fail-closed reason for renderer color provenance.

    Godot's ``Image.FORMAT_RGB8`` identifies channel layout and bit depth. A
    runtime observation that ``Viewport.use_hdr_2d`` is false additionally
    rules out the documented HDR-linear 2D texture path, but neither value is
    transfer-function or color-primary metadata. They therefore remain
    insufficient to assert the combined sRGB/BT.709 contract.
    """

    provenance = capture.get("color_space_provenance")
    observations = (
        provenance.get("runtime_observations")
        if isinstance(provenance, dict)
        else None
    )
    if (
        capture.get("color_space") == "sRGB"
        and isinstance(provenance, dict)
        and provenance.get("status") == "ENGINE_CONTRACT_DERIVED"
        and provenance.get("profile")
        == {
            "encoding": "sRGB",
            "color_primaries": "BT.709",
            "white_point": "D65",
        }
        and provenance.get("engine_contract")
        == {
            "godot_revision": "5b4e0cb0f",
            "renderer_documentation": "https://docs.godotengine.org/en/4.7/engine_details/architecture/internal_rendering_architecture.html",
            "compatibility_claim": "OpenGL uses Compatibility; Compatibility colors are stored in sRGB with no HDR support",
            "scope": "renderer frame encoding only; no physical display, ICC, or Xvfb colorimetry claim",
        }
        and capture.get("godot_version") == L6_PREFLIGHT_PINS.godot_version
        and isinstance(observations, dict)
        and observations
        == {
            "viewport_use_hdr_2d": False,
            "viewport_use_hdr_2d_api": "get_viewport().use_hdr_2d",
            "image_format": "Image.FORMAT_RGB8",
            "image_format_api": "Image.get_format()",
            "display_server": "X11",
            "display_server_api": "DisplayServer.get_name()",
            "rendering_method": "gl_compatibility",
            "rendering_method_api": "RenderingServer.get_current_rendering_method()",
            "rendering_driver": "opengl3",
            "rendering_driver_api": "RenderingServer.get_current_rendering_driver_name()",
            "hdr_output_supported": False,
            "hdr_output_requested": False,
            "hdr_output_enabled": False,
            "hdr_output_api": "DisplayServer.window_is_hdr_output_supported/requested/enabled()",
        }
    ):
        return None
    if (
        capture.get("color_space") == "UNSPECIFIED"
        and isinstance(provenance, dict)
        and provenance.get("status") == "INSUFFICIENT"
        and isinstance(observations, dict)
        and observations.get("viewport_use_hdr_2d") is False
        and observations.get("viewport_use_hdr_2d_api")
        == "get_viewport().use_hdr_2d"
        and observations.get("image_format") == "Image.FORMAT_RGB8"
        and observations.get("image_format_api") == "Image.get_format()"
    ):
        return f"{label}_color_space_unproven"
    if capture.get("color_space") == "sRGB/BT.709":
        return f"{label}_color_space_direct_provenance_missing"
    if isinstance(provenance, dict) and provenance.get("status") == "ENGINE_CONTRACT_DERIVED":
        return f"{label}_color_space_engine_contract_invalid"
    return f"{label}_color_space_missing_or_invalid"


def _renderer_evidence(
    run_root: Path,
    label: str,
    evidence: object,
    gaps: list[dict[str, str]],
) -> dict[str, object] | None:
    """Validate renderer-owned RGB evidence without manufacturing a fallback."""

    if not isinstance(evidence, dict):
        _evidence_gap(gaps, "renderer_capture", f"{label}_evidence_missing")
        return None
    frame_path = _owned_file(run_root, evidence.get("frame_artifact"))
    if frame_path is None:
        _evidence_gap(gaps, "renderer_capture", f"{label}_frame_artifact_missing_or_external")
        return None
    capture = evidence.get("renderer_capture")
    if not isinstance(capture, dict):
        _evidence_gap(gaps, "renderer_capture", f"{label}_capture_metadata_missing")
        return None

    actual = capture.get("actual_capture_timestamps_us")
    if (
        not isinstance(actual, list)
        or len(actual) != L6_PREFLIGHT_PINS.capture_ticks
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in actual)
        or any(right <= left for left, right in zip(actual, actual[1:]))
    ):
        _evidence_gap(gaps, "renderer_capture", f"{label}_actual_timestamps_missing_or_invalid")
    color_gap = _renderer_color_space_gap(label, capture)
    if color_gap is not None:
        _evidence_gap(gaps, "renderer_capture", color_gap)
    if capture.get("capture_kind") != "godot_viewport_rgb_png_sequence":
        _evidence_gap(gaps, "renderer_capture", f"{label}_viewport_capture_provenance_missing")

    try:
        import numpy as np

        from .renderer_artifact import RendererArtifactError, open_renderer_artifact

        with open_renderer_artifact(frame_path) as artifact:
            expected_shape = (
                L6_PREFLIGHT_PINS.capture_ticks,
                720,
                1280,
                3,
            )
            if artifact.frames.shape != expected_shape or artifact.frames.dtype != np.uint8:
                _evidence_gap(gaps, "renderer_capture", f"{label}_rgb_shape_or_dtype_invalid")
            expected_us = np.asarray(
                [
                    (index * 1_000_000) // L6_PREFLIGHT_PINS.fixed_fps
                    for index in range(L6_PREFLIGHT_PINS.capture_ticks)
                ],
                dtype=np.int64,
            )
            declared_us = capture.get("presentation_timestamps_us")
            measured_timestamps_sha256 = _sha256_bytes(
                artifact.timestamps.astype("<f8", copy=False).tobytes()
            )
            if evidence.get("timestamps_sha256") != measured_timestamps_sha256:
                _evidence_gap(
                    gaps,
                    "renderer_capture",
                    f"{label}_presentation_timestamp_hash_mismatch",
                )
            expected_seconds = np.arange(
                L6_PREFLIGHT_PINS.capture_ticks, dtype=np.float64
            ) / float(L6_PREFLIGHT_PINS.fixed_fps)
            if declared_us != expected_us.tolist() or not np.array_equal(
                artifact.timestamps, expected_seconds
            ):
                _evidence_gap(gaps, "renderer_capture", f"{label}_presentation_timestamps_invalid")
    except (OSError, ValueError, RendererArtifactError):
        _evidence_gap(gaps, "renderer_capture", f"{label}_frame_artifact_invalid")
        return None

    return {
        "frame_artifact_sha256": _sha256_file(frame_path),
        "presentation_timestamps_sha256": measured_timestamps_sha256,
    }


def _runtime_events(
    run_root: Path,
    label: str,
    evidence: object,
    gaps: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, dict):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_evidence_missing")
        return []
    _, replay = _load_owned_json(run_root, evidence.get("artifact"))
    if replay is None:
        _evidence_gap(gaps, "runtime_attribution", f"{label}_replay_missing_or_invalid")
        return []
    raw = replay.get("runtime_events")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, dict) for item in raw):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_events_missing")
        return []
    events = [dict(item) for item in raw]
    required = {
        "normalized_node_identity",
        "spawned_ordinal",
        "script_path",
        "script_path_observation",
        "script_sha256",
        "resource_path",
        "resource_path_observation",
        "resource_provenance",
        "source_line",
        "source_line_observation",
        "property",
        "factual_value",
        "actual_capture_timestamp_us",
    }
    if any(not required.issubset(event) for event in events):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_provenance_fields_missing")
        return events
    if any(
        event.get("script_path_observation") != "node.get_script().resource_path"
        or event.get("resource_provenance") not in {"node_owner_scene", "packed_scene_state"}
        or event.get("script_path") != "res://scripts/RoutShockwave.gd"
        or event.get("resource_path") != "res://scenes/Battle.tscn"
        or event.get("resource_path_observation") != "battle.scene_file_path"
        or event.get("property") != "flashpatch_intensity"
        or event.get("source_line_observation")
        != "FileAccess.get_file_as_string(node.get_script().resource_path)"
        or isinstance(event.get("source_line"), bool)
        or not isinstance(event.get("source_line"), int)
        or int(event.get("source_line", 0)) < 1
        or not isinstance(event.get("spawned_ordinal"), int)
        or isinstance(event.get("spawned_ordinal"), bool)
        or not isinstance(event.get("normalized_node_identity"), str)
        or not event.get("normalized_node_identity")
        for event in events
    ):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_provenance_mismatch")
    source_hash = evidence.get("runtime_script_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_script_hash_missing")
    elif any(event.get("script_sha256") != source_hash for event in events):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_script_hash_mismatch")
    runtime_source_line = evidence.get("runtime_source_line")
    if (
        isinstance(runtime_source_line, bool)
        or not isinstance(runtime_source_line, int)
        or runtime_source_line < 1
        or any(event.get("source_line") != runtime_source_line for event in events)
    ):
        _evidence_gap(gaps, "runtime_attribution", f"{label}_runtime_source_line_mismatch")
    return events


def _single_patch_evidence(
    run_root: Path,
    candidate: object,
    gaps: list[dict[str, str]],
) -> dict[str, object] | None:
    if not isinstance(candidate, dict):
        _evidence_gap(gaps, "patch_validation", "candidate_missing")
        return None
    if (
        candidate.get("parameter") != "flashpatch_intensity"
        or isinstance(candidate.get("source_line"), bool)
        or not isinstance(candidate.get("source_line"), int)
        or int(candidate.get("source_line", 0)) < 1
        or candidate.get("replacement") != 0.0
        or candidate.get("changed_source_assignments") != 1
    ):
        _evidence_gap(gaps, "patch_validation", "single_allowlisted_assignment_not_proven")
    diff_path = _owned_file(run_root, candidate.get("diff"))
    if diff_path is None:
        _evidence_gap(gaps, "patch_validation", "source_diff_missing_or_external")
        return None
    try:
        changed = [
            line
            for line in diff_path.read_text(encoding="utf-8").splitlines()
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith(("+++", "---"))
        ]
    except OSError:
        changed = []
    if len(changed) != 2 or not all("flashpatch_intensity" in line for line in changed):
        _evidence_gap(gaps, "patch_validation", "diff_is_not_exactly_one_parameter_assignment")
    return {"diff_sha256": _sha256_file(diff_path)}


def _preservation_artifacts(
    run_root: Path,
    label: str,
    evidence: dict[str, object],
    gaps: list[dict[str, str]],
) -> dict[str, object] | None:
    stream = _owned_file(run_root, evidence.get("state_stream_artifact"))
    final_state = _owned_file(run_root, evidence.get("final_state_artifact"))
    if stream is None or final_state is None:
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_state_artifact_missing_or_external",
        )
        return None
    if _sha256_file(stream) != evidence.get("state_stream_sha256"):
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_state_stream_hash_mismatch",
        )
    if _sha256_file(final_state) != evidence.get("final_state_raw_sha256"):
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_final_state_raw_hash_mismatch",
        )
    try:
        records = [
            json.loads(line)
            for line in stream.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ticks = [record.get("tick") for record in records]
        final_payload = json.loads(final_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, AttributeError):
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_state_artifact_malformed",
        )
        return None
    if (
        not records
        or not all(isinstance(record, dict) for record in records)
        or any(isinstance(tick, bool) or not isinstance(tick, int) for tick in ticks)
        or ticks != list(range(len(ticks)))
        or ticks[: L6_PREFLIGHT_PINS.capture_ticks]
        != list(range(L6_PREFLIGHT_PINS.capture_ticks))
        or evidence.get("state_stream_tick_domain") != [ticks[0], ticks[-1]]
        or evidence.get("state_stream_record_count") != len(records)
    ):
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_state_stream_tick_evidence_invalid",
        )
    final_tick = L6_PREFLIGHT_PINS.capture_ticks - 1
    if not isinstance(final_payload, dict) or final_payload.get("tick") != final_tick:
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_final_state_tick_invalid",
        )
    elif _sha256_bytes(_canonical_json(final_payload)) != evidence.get(
        "final_state_sha256"
    ):
        _evidence_gap(
            gaps,
            "gameplay_preservation",
            f"{label}_final_state_canonical_hash_mismatch",
        )
    return {
        "state_stream_artifact_sha256": _sha256_file(stream),
        "final_state_artifact_sha256": _sha256_file(final_state),
    }


def _bound_preservation_record(
    evidence: dict[str, object],
    artifacts: dict[str, object] | None,
) -> dict[str, object]:
    """Carry measured preservation fields without leaking transient paths."""

    return {
        "action_acknowledgements": evidence["action_acknowledgements"],
        "action_acknowledgements_sha256": _sha256_bytes(
            _canonical_json(evidence["action_acknowledgements"])
        ),
        "tick_domain": evidence["tick_domain"],
        "tick_domain_sha256": _sha256_bytes(
            _canonical_json(evidence["tick_domain"])
        ),
        "presentation_timestamps_sha256": evidence["timestamps_sha256"],
        "state_stream_sha256": evidence["state_stream_sha256"],
        "state_stream_tick_domain": evidence["state_stream_tick_domain"],
        "state_stream_record_count": evidence["state_stream_record_count"],
        "final_state_sha256": evidence["final_state_sha256"],
        "final_state_raw_sha256": evidence["final_state_raw_sha256"],
        "state_stream_artifact_sha256": artifacts.get(
            "state_stream_artifact_sha256"
        )
        if artifacts is not None
        else None,
        "final_state_artifact_sha256": artifacts.get(
            "final_state_artifact_sha256"
        )
        if artifacts is not None
        else None,
    }


def _preservation_evidence(
    run_root: Path,
    factual: object,
    candidate: object,
    gaps: list[dict[str, str]],
) -> dict[str, object] | None:
    if not isinstance(factual, dict) or not isinstance(candidate, dict):
        _evidence_gap(gaps, "gameplay_preservation", "factual_or_candidate_evidence_missing")
        return None
    required = (
        "state_stream_sha256",
        "final_state_sha256",
        "tick_domain",
        "timestamps_sha256",
        "action_acknowledgements",
        "state_stream_artifact",
        "state_stream_tick_domain",
        "state_stream_record_count",
        "final_state_artifact",
        "final_state_raw_sha256",
    )
    if any(key not in factual or key not in candidate for key in required):
        _evidence_gap(gaps, "gameplay_preservation", "explicit_preservation_fields_missing")
        return None
    expected_actions = [{"frame": 0, "status": "APPLIED"}]
    if factual["action_acknowledgements"] != expected_actions:
        _evidence_gap(gaps, "gameplay_preservation", "factual_action_not_applied")
    if candidate["action_acknowledgements"] != expected_actions:
        _evidence_gap(gaps, "gameplay_preservation", "candidate_action_not_applied")
    if factual["tick_domain"] != [0, 160] or candidate["tick_domain"] != [0, 160]:
        _evidence_gap(gaps, "gameplay_preservation", "tick_domain_mismatch")
    factual_artifacts = _preservation_artifacts(
        run_root, "factual", factual, gaps
    )
    candidate_artifacts = _preservation_artifacts(
        run_root, "candidate", candidate, gaps
    )
    comparable = (
        "state_stream_sha256",
        "final_state_sha256",
        "final_state_raw_sha256",
        "tick_domain",
        "timestamps_sha256",
        "state_stream_tick_domain",
        "state_stream_record_count",
    )
    for key in comparable:
        if factual[key] != candidate[key]:
            _evidence_gap(gaps, "gameplay_preservation", f"{key}_mismatch")
    factual_record = _bound_preservation_record(factual, factual_artifacts)
    candidate_record = _bound_preservation_record(candidate, candidate_artifacts)
    bound_records = {
        "factual": factual_record,
        "candidate": candidate_record,
    }
    return {
        "state_stream_sha256": factual["state_stream_sha256"],
        "final_state_sha256": factual["final_state_sha256"],
        "final_state_raw_sha256": factual["final_state_raw_sha256"],
        "tick_domain": factual["tick_domain"],
        "presentation_timestamps_sha256": factual["timestamps_sha256"],
        "action_acknowledgements": factual.get("action_acknowledgements"),
        **bound_records,
        "preservation_evidence_sha256": _sha256_bytes(
            _canonical_json(bound_records)
        ),
    }


def _validate_positive_engine_receipt(
    run_root: Path,
    receipt_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    gaps: list[dict[str, str]] = []
    _, receipt = _load_owned_json(run_root, str(receipt_path))
    if receipt is None:
        return ([{"gate": "engine", "reason": "engine_receipt_missing_or_invalid"}], {})
    if receipt.get("verdict") != "PASS":
        _evidence_gap(
            gaps,
            "engine",
            f"engine_verdict_{str(receipt.get('verdict', 'MISSING')).lower()}:"
            f"{str(receipt.get('reason', 'reason_missing'))}",
        )
        return gaps, {"engine_receipt_sha256": _sha256_file(receipt_path)}
    if receipt.get("controlled_mutation") is not True:
        _evidence_gap(gaps, "engine", "controlled_mutation_label_missing")
    upstream = receipt.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("upstream_defect") is not False:
        _evidence_gap(gaps, "engine", "upstream_defect_boundary_missing")

    factual = receipt.get("factual_replay")
    candidates = receipt.get("candidates")
    candidate = receipt.get("attribution")
    if not isinstance(candidates, list) or len(candidates) != 1 or candidate != candidates[0]:
        _evidence_gap(gaps, "patch_validation", "exactly_one_selected_candidate_not_proven")
    factual_renderer = _renderer_evidence(run_root, "factual", factual, gaps)
    candidate_renderer = _renderer_evidence(run_root, "candidate", candidate, gaps)
    factual_events = _runtime_events(run_root, "factual", factual, gaps)
    candidate_events = _runtime_events(run_root, "candidate", candidate, gaps)
    patch = _single_patch_evidence(run_root, candidate, gaps)
    preservation = _preservation_evidence(run_root, factual, candidate, gaps)

    if isinstance(factual, dict):
        if not factual.get("hazardous") or not isinstance(factual.get("max_risk"), (int, float)) or float(factual["max_risk"]) < 1.0:
            _evidence_gap(gaps, "risk", "factual_renderer_hazard_not_proven")
    if isinstance(candidate, dict):
        if candidate.get("hazardous") is not False or candidate.get("max_risk") != 0.0:
            _evidence_gap(gaps, "risk", "candidate_zero_residual_not_proven")
    if factual_events and candidate_events:
        factual_identity = {
            (event.get("normalized_node_identity"), event.get("spawned_ordinal"))
            for event in factual_events
        }
        candidate_identity = {
            (event.get("normalized_node_identity"), event.get("spawned_ordinal"))
            for event in candidate_events
        }
        if factual_identity != candidate_identity:
            _evidence_gap(gaps, "runtime_attribution", "candidate_runtime_identity_mismatch")
        if any(event.get("factual_value") != 1.0 for event in factual_events):
            _evidence_gap(gaps, "runtime_attribution", "factual_runtime_value_mismatch")
        if any(event.get("factual_value") != 0.0 for event in candidate_events):
            _evidence_gap(gaps, "runtime_attribution", "candidate_runtime_value_mismatch")

    repeatable_events = [
        {
            key: value
            for key, value in event.items()
            if key != "actual_capture_timestamp_us"
        }
        for event in factual_events + candidate_events
    ]
    summary = {
        "engine_receipt_sha256": _sha256_file(receipt_path),
        "source_snapshot_sha256": receipt.get("input_sha256", {}).get("source_snapshot")
        if isinstance(receipt.get("input_sha256"), dict)
        else None,
        "factual_renderer": factual_renderer,
        "candidate_renderer": candidate_renderer,
        "runtime_events_sha256": _sha256_bytes(_canonical_json(repeatable_events))
        if factual_events and candidate_events
        else None,
        "patch": patch,
        "preservation": preservation,
    }
    return gaps, summary


def _write_execution_receipt(root: Path, receipt: dict[str, Any]) -> None:
    receipt_bytes = _canonical_json(receipt)
    (root / "execution-receipt.json").write_bytes(receipt_bytes)
    checkpoint = {
        "schema": EXECUTION_CHECKPOINT_SCHEMA,
        "phase": receipt["phase"],
        "verdict": receipt["verdict"],
        "execution_receipt_sha256": _sha256_bytes(receipt_bytes),
    }
    (root / "execution-checkpoint.json").write_bytes(_canonical_json(checkpoint))


def run_positive(
    upstream: Path | str,
    godot: Path | str,
    run_root: Path | str,
    *,
    runs: int = L6_REPEAT_COUNT,
    pins: L6PreflightPins = L6_PREFLIGHT_PINS,
) -> dict[str, Any]:
    """Attempt the real L6 chain and issue PASS only for complete runtime proof."""

    if isinstance(runs, bool) or runs != L6_REPEAT_COUNT:
        raise ValueError(f"L6 positive execution requires exactly {L6_REPEAT_COUNT} runs")
    root = _create_fresh_run_root(Path(run_root))
    preflight = run_preflight(upstream, godot, root / "preflight", pins=pins)
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "execution_id": root.name,
        "input_id": preflight["preflight_input_sha256"],
        "verdict": "INCONCLUSIVE",
        "phase": "PREFLIGHT_SEALED",
        "controlled_mutation": False,
        "upstream_defect": False,
        "upstream_product_verdict": "INCONCLUSIVE",
        "runs_requested": runs,
        "runs_completed": 0,
        "preflight": {
            "path": "preflight/preflight.json",
            "sha256": _sha256_file(root / "preflight" / "preflight.json"),
            "verdict": preflight["preflight_verdict"],
        },
        "attempts": [],
        "evidence_gaps": [],
    }
    if preflight["preflight_verdict"] != "PASS" or preflight["replay_allowed"] is not True:
        receipt["phase"] = "PREFLIGHT_INCONCLUSIVE"
        receipt["evidence_gaps"] = [{"gate": "preflight", "reason": "pinned_input_mismatch"}]
        _write_execution_receipt(root, receipt)
        return receipt

    from .public_godot import execute_controlled_sparta

    summaries: list[dict[str, object]] = []
    for index in range(runs):
        attempt_root = root / f"run-{index + 1:03d}"
        try:
            run = execute_controlled_sparta(
                Path(upstream),
                attempt_root,
                godot_binary=Path(godot),
            )
        except Exception as exc:
            gaps = [{
                "gate": "engine",
                "reason": f"runtime_error:{type(exc).__name__}:{exc}",
            }]
            receipt["attempts"].append({
                "run": index + 1,
                "path": attempt_root.relative_to(root).as_posix(),
                "verdict": "INCONCLUSIVE",
                "evidence_gaps": gaps,
            })
            receipt["evidence_gaps"] = gaps
            receipt["phase"] = "RUNTIME_INCONCLUSIVE"
            _write_execution_receipt(root, receipt)
            return receipt

        gaps, summary = _validate_positive_engine_receipt(
            attempt_root,
            run.receipt_path,
        )
        receipt["runs_completed"] = index + 1
        receipt["controlled_mutation"] = True
        receipt["attempts"].append({
            "run": index + 1,
            "path": attempt_root.relative_to(root).as_posix(),
            "engine_receipt": run.receipt_path.relative_to(root).as_posix(),
            "engine_receipt_sha256": _sha256_file(run.receipt_path),
            "verdict": "PASS" if not gaps else "INCONCLUSIVE",
            "evidence_gaps": gaps,
            "deterministic_evidence": summary,
        })
        if gaps:
            receipt["evidence_gaps"] = gaps
            receipt["phase"] = "RUNTIME_EVIDENCE_INCONCLUSIVE"
            _write_execution_receipt(root, receipt)
            return receipt
        summaries.append(summary)

    stable_summaries = [
        {key: value for key, value in summary.items() if key != "engine_receipt_sha256"}
        for summary in summaries
    ]
    if any(summary != stable_summaries[0] for summary in stable_summaries[1:]):
        receipt["evidence_gaps"] = [{"gate": "repeat", "reason": "deterministic_evidence_mismatch"}]
        receipt["phase"] = "REPEAT_INCONCLUSIVE"
        _write_execution_receipt(root, receipt)
        return receipt

    receipt["verdict"] = "PASS"
    receipt["phase"] = "POSITIVE_REPEATS_VERIFIED"
    receipt["deterministic_evidence"] = stable_summaries[0]
    _write_execution_receipt(root, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m flashpatch.l6_run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--runs", type=int)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.preflight:
            receipt = run_preflight(
                arguments.upstream,
                arguments.godot,
                arguments.run_root,
            )
            verdict_field = "preflight_verdict"
        else:
            receipt = run_positive(
                arguments.upstream,
                arguments.godot,
                arguments.run_root,
                runs=arguments.runs,
            )
            verdict_field = "verdict"
    except FileExistsError:
        print("L6 run root already exists", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json(receipt))
    return 0 if receipt[verdict_field] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
