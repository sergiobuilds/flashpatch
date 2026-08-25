from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import cv2
import pytest

from flashpatch.l10_receipt import L10ReceiptError, verify_l10_bundle, verify_l10_receipt
from flashpatch.l10_unity import unity_adapter_fingerprints
from flashpatch.l10_unity_runner import (
    UNITY_RECEIPT_COMMAND,
    UNITY_VULKAN_LOADER_SHA256,
)


def _policy(*receipts: Path) -> tuple[Path, str]:
    entries = []
    for receipt in receipts:
        value = json.loads(receipt.read_text())
        scene = receipt.parent / value["scene"]["artifact"]["path"]
        entry = {
            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "engine": value["engine"],
            "engine_version": value["engine_version"],
            "evidence_class": value["evidence_class"],
            "repository": value["source"]["repository"],
            "revision": value["source"]["revision"],
            "tree_sha256": value["source"]["tree_sha256"],
            "scene_source_path": value["scene"]["source_path"],
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "runtime_kind": value["runtime"]["kind"],
            "runtime_digest": value["runtime"]["digest"],
        }
        if value["engine"] == "Unity":
            provenance = json.loads(
                (receipt.parent / value["runtime"]["provenance"]["path"]).read_text()
            )
            factual_manifest = receipt.parent / provenance["factual_project_manifest"]["path"]
            candidate_manifest = receipt.parent / provenance["counterfactual_project_manifest"]["path"]
            entry.update({
                "factual_project_manifest_sha256": hashlib.sha256(factual_manifest.read_bytes()).hexdigest(),
                "counterfactual_project_manifest_sha256": hashlib.sha256(candidate_manifest.read_bytes()).hexdigest(),
                "project_input_count": len(json.loads(factual_manifest.read_text())["files"]),
            })
        entries.append(entry)
    policy = {
        "schema": "flashpatch-l10-trust-policy-v1",
        "engines": entries,
    }
    path = receipts[0].parent / "trust-policy.json"
    raw = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _verify(receipt: Path) -> dict[str, object]:
    policy, digest = _policy(receipt)
    return verify_l10_receipt(receipt, policy, digest)


def _write(root: Path, name: str, value: bytes) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"path": name, "sha256": hashlib.sha256(value).hexdigest()}


def _run_id(
    factual_frames: dict[str, str],
    counterfactual_frames: dict[str, str],
    factual_log: dict[str, str],
    counterfactual_log: dict[str, str],
    factual_marker: dict[str, str],
    counterfactual_marker: dict[str, str],
) -> str:
    raw = (json.dumps({
        "counterfactual_execution_log_sha256": counterfactual_log["sha256"],
        "counterfactual_execution_marker_sha256": counterfactual_marker["sha256"],
        "counterfactual_frames_sha256": counterfactual_frames["sha256"],
        "factual_execution_log_sha256": factual_log["sha256"],
        "factual_execution_marker_sha256": factual_marker["sha256"],
        "factual_frames_sha256": factual_frames["sha256"],
    }, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _unity_execution(
    root: Path,
    prefix: str,
    capture: dict[str, object],
    lane: str,
    command: dict[str, str],
    project_manifest: dict[str, str],
    scene_sha256: str,
    *,
    deterministic: bool = False,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    with np.load(root / capture["frames"]["path"], allow_pickle=False) as payload:
        frames = payload["frames"]
    pngs = []
    png_hashes = []
    for index, frame in enumerate(frames):
        success, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        assert success
        reference = _write(root, f"{prefix}/png/{index:04d}.png", encoded.tobytes())
        pngs.append(reference)
        png_hashes.append(reference["sha256"])
    png_set = hashlib.sha256("\n".join(png_hashes).encode("ascii")).hexdigest()
    marker_value = {
        "schema": "flashpatch-l10-unity-execution-marker-v1",
        "engine": "Unity",
        "engine_version": "2022.3.8f1",
        "frame_count": len(frames),
        "mode": lane,
        "png_set_sha256": png_set,
        "project_manifest_sha256": project_manifest["sha256"],
        "scene_sha256": scene_sha256,
    }
    if deterministic:
        marker_value.update({
            "adapter_sha256": unity_adapter_fingerprints()[
                "Assets/Editor/FlashPatchL10Capture.cs"
            ],
            "graphics_device_id": 0x2C05,
            "graphics_device_name": "NVIDIA GeForce RTX 5070 Ti",
            "graphics_device_type": "Vulkan",
            "graphics_device_vendor": "NVIDIA",
            "replay_profile": "deterministic-step-v2",
            "runtime_events_sha256": capture["runtime_events"]["sha256"],
            "schema": "flashpatch-l10-unity-execution-marker-v2",
            "state_stream_sha256": capture["state_stream"]["sha256"],
        })
    marker_raw = (json.dumps(marker_value, indent=2, sort_keys=True) + "\n").encode()
    marker = _write(root, f"{prefix}/execution-marker.json", marker_raw)
    completion = json.dumps(marker_value, separators=(",", ":"), sort_keys=True)
    raw_log = (
        f"FlashPatch synthetic run: {prefix}\n"
        + (
            f"FLASHPATCH_L10_VULKAN_LOADER_VERIFIED {UNITY_VULKAN_LOADER_SHA256}\n"
            if deterministic
            else ""
        )
        +
        "Unity Editor version:    2022.3.8f1\n"
        "Vulkan vendor=[NVIDIA] id=[10de]\n"
        "Vulkan renderer=[NVIDIA GeForce RTX 5070 Ti] id=[2c05]\n"
        "-executeMethod\nFlashPatchL10Capture.Run\n"
        f"FLASHPATCH_L10_COMPLETE {completion}\n"
    ).encode()
    packed = {
        "schema": "flashpatch-l10-packed-execution-v1",
        "command_sha256": command["sha256"],
        "engine": "Unity",
        "engine_version": "2022.3.8f1",
        "frames_sha256": capture["frames"]["sha256"],
        "lane": lane,
        "execution_marker_sha256": marker["sha256"],
        "png_set_sha256": png_set,
        "raw_log_sha256": hashlib.sha256(raw_log).hexdigest(),
    }
    if deterministic:
        packed.update({
            "runtime_events_sha256": capture["runtime_events"]["sha256"],
            "state_stream_sha256": capture["state_stream"]["sha256"],
        })
    log_raw = raw_log + b"\nFLASHPATCH_L10_PACKED " + json.dumps(
        packed, separators=(",", ":"), sort_keys=True
    ).encode() + b"\n"
    log = _write(root, f"{prefix}/execution.log.gz", gzip.compress(log_raw, mtime=0))
    return log, marker, pngs


def _receipt(
    root: Path,
    *,
    inconclusive: bool = False,
    deterministic: bool = False,
    downgraded_marker_prefix: str | None = None,
    state_preserved: bool = True,
) -> Path:
    trace = _write(root, "trace.json", b'{"frames":[0,1,2]}\n')
    identity = _write(root, "engine.json", b'{"engine":"Unity","version":"2022.3.8f1"}\n')
    command = _write(
        root,
        "command.txt",
        UNITY_RECEIPT_COMMAND.encode() if deterministic else b"Unity -batchmode\n",
    )
    license_ref = _write(root, "LICENSE", b"sample license\n")
    attribution_payload = {
        "component": "Light",
        "counterfactual_value": 0,
        "factual_value": 1,
        "object_identity": "Portal",
        "property": "intensity",
    }
    attribution = _write(root, "attribution.json", (json.dumps(attribution_payload, indent=2, sort_keys=True) + "\n").encode())
    patch_payload = {
        **attribution_payload,
        "files_changed": 1,
        "kind": "minimal_source_parameter",
    }
    timestamp_values = np.arange(18, dtype=np.float64) / 30.0
    timestamps = _write(root, "timestamps.json", (json.dumps(timestamp_values.tolist(), indent=2) + "\n").encode())

    def capture(prefix: str, values: list[int], result: str, hazards: list[int]) -> dict[str, object]:
        frames = np.stack([np.full((4, 5, 3), value, dtype=np.uint8) for value in values])
        frame_path = root / f"{prefix}-frames.npz"
        np.savez_compressed(frame_path, frames=frames, timestamps=timestamp_values)
        frame_ref = {"path": frame_path.name, "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest()}
        rgb_hash = hashlib.sha256(frames.tobytes()).hexdigest()
        detector_payload = {
            "frames_rgb_sha256": rgb_hash,
            "hazard_frames": hazards,
            "result": result,
            "timestamps_sha256": hashlib.sha256(timestamp_values.tobytes()).hexdigest(),
        }
        detector_ref = _write(
            root,
            f"{prefix}-detector.json",
            (json.dumps(detector_payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        events = [{
            "frame": index,
            "object_identity": "Portal",
            "component": "Light",
            "property": "intensity",
            "value": (
                1
                if prefix.endswith("factual") or prefix == "factual"
                else 0
            ),
            **({
                "rendered_value": (
                    float(index + 1)
                    if prefix.endswith("factual") or prefix == "factual"
                    else 1.0
                )
            } if deterministic else {}),
        } for index in range(len(values))]
        states = [{"frame": index, "terminal_state": "captured"} for index in range(len(values))]
        return {
            "frames": frame_ref,
            "timestamps": timestamps,
            "runtime_events": _write(root, f"{prefix}-events.json", (json.dumps(events, indent=2, sort_keys=True) + "\n").encode()),
            "state_stream": _write(root, f"{prefix}-state.json", (json.dumps(states, indent=2, sort_keys=True) + "\n").encode()),
            "detector": {"result": result, "hazard_frames": hazards, "receipt": detector_ref},
        }

    factual_values = [31] * 18 if inconclusive else [0 if index % 2 == 0 else 255 for index in range(18)]
    factual_hazards = [] if inconclusive else list(range(1, 18))
    candidate_values = [32 + index % 2 for index in range(18)] if deterministic else [32] * 18
    factual_result = "SAFE" if inconclusive else "HAZARDOUS"
    factual_capture = capture("factual", factual_values, factual_result, factual_hazards)
    candidate_capture = capture("candidate", candidate_values, "SAFE", [])
    if not state_preserved:
        factual_state_path = root / factual_capture["state_stream"]["path"]
        factual_states = json.loads(factual_state_path.read_text())
        factual_states[0]["terminal_state"] = "diverged"
        factual_state_path.write_text(
            json.dumps(factual_states, indent=2, sort_keys=True) + "\n"
        )
        factual_capture["state_stream"]["sha256"] = hashlib.sha256(
            factual_state_path.read_bytes()
        ).hexdigest()
    tree_hash = "b" * 64
    source_provenance = _write(root, "source-provenance.json", (json.dumps({
        "clean": True,
        "repository": "https://github.com/example/game",
        "revision": "a" * 40,
        "tree_sha256": tree_hash,
    }, indent=2, sort_keys=True) + "\n").encode())
    scene_artifact = _write(
        root,
        "Scene.unity",
        b"scene prefix\n  m_IntensityJitterScale: 2000\nscene suffix\n",
    )
    counterfactual_scene_artifact = _write(
        root,
        "Scene-candidate.unity",
        b"scene prefix\n  m_IntensityJitterScale: 0\nscene suffix\n",
    )
    shared_input_hash = hashlib.sha256(b"shared project input\n").hexdigest()
    adapter_row = [{
        "path": "Assets/Editor/FlashPatchL10Capture.cs",
        "sha256": (
            unity_adapter_fingerprints()["Assets/Editor/FlashPatchL10Capture.cs"]
            if deterministic
            else _UNITY_LEGACY_ADAPTER_SHA256_V1
        ),
    }]
    factual_project_manifest = _write(
        root,
        "factual-project-inputs.json",
        (json.dumps({
            "files": [
                {"path": "Assets/Scene.unity", "sha256": scene_artifact["sha256"]},
                {"path": "Packages/manifest.json", "sha256": shared_input_hash},
                *adapter_row,
            ],
            "schema": "flashpatch-l10-unity-project-inputs-v1",
        }, indent=2, sort_keys=True) + "\n").encode(),
    )
    candidate_project_manifest = _write(
        root,
        "counterfactual-project-inputs.json",
        (json.dumps({
            "files": [
                {"path": "Assets/Scene.unity", "sha256": counterfactual_scene_artifact["sha256"]},
                {"path": "Packages/manifest.json", "sha256": shared_input_hash},
                *adapter_row,
            ],
            "schema": "flashpatch-l10-unity-project-inputs-v1",
        }, indent=2, sort_keys=True) + "\n").encode(),
    )
    patch_payload.update({
        "factual_scene_sha256": scene_artifact["sha256"],
        "counterfactual_scene_sha256": counterfactual_scene_artifact["sha256"],
    })
    patch = _write(root, "change.json", (json.dumps(patch_payload, indent=2, sort_keys=True) + "\n").encode())
    baseline_factual_log, baseline_factual_marker, baseline_factual_pngs = _unity_execution(
        root, "baseline/factual", factual_capture, "factual", command,
        factual_project_manifest, scene_artifact["sha256"],
        deterministic=deterministic and downgraded_marker_prefix != "baseline/factual",
    )
    baseline_candidate_log, baseline_candidate_marker, baseline_candidate_pngs = _unity_execution(
        root, "baseline/counterfactual", candidate_capture, "counterfactual", command,
        candidate_project_manifest, counterfactual_scene_artifact["sha256"],
        deterministic=deterministic and downgraded_marker_prefix != "baseline/counterfactual",
    )
    runtime_payload = {
        "digest": "d" * 64,
        "engine": "Unity",
        "engine_version": "2022.3.8f1",
        "factual_execution_log": baseline_factual_log,
        "counterfactual_execution_log": baseline_candidate_log,
        "factual_execution_marker": baseline_factual_marker,
        "counterfactual_execution_marker": baseline_candidate_marker,
        "factual_pngs": baseline_factual_pngs,
        "counterfactual_pngs": baseline_candidate_pngs,
        "factual_project_manifest": factual_project_manifest,
        "counterfactual_project_manifest": candidate_project_manifest,
        "factual_frames_sha256": factual_capture["frames"]["sha256"],
        "counterfactual_frames_sha256": candidate_capture["frames"]["sha256"],
        "exit_code": 0,
        "kind": "container",
        "run_id": _run_id(
            factual_capture["frames"],
            candidate_capture["frames"],
            baseline_factual_log,
            baseline_candidate_log,
            baseline_factual_marker,
            baseline_candidate_marker,
        ),
        "factual_scene_sha256": scene_artifact["sha256"],
        "counterfactual_scene_sha256": counterfactual_scene_artifact["sha256"],
        "source_revision": "a" * 40,
    }
    runtime_provenance = _write(root, "runtime-provenance.json", (json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n").encode())
    repeats = []
    for ordinal in (1, 2, 3):
        repeated_factual_values = [31 + ordinal] * 18 if inconclusive else factual_values
        repeated_candidate_values = [32 + ordinal] * 18 if inconclusive else candidate_values
        repeated_factual = capture(f"repeat-{ordinal}-factual", repeated_factual_values, factual_result, factual_hazards)
        repeated_candidate = capture(f"repeat-{ordinal}-candidate", repeated_candidate_values, "SAFE", [])
        repeat_factual_log, repeat_factual_marker, repeat_factual_pngs = _unity_execution(
            root, f"repeats/{ordinal}/factual", repeated_factual, "factual", command,
            factual_project_manifest, scene_artifact["sha256"],
            deterministic=(
                deterministic
                and downgraded_marker_prefix != f"repeats/{ordinal}/factual"
            ),
        )
        repeat_candidate_log, repeat_candidate_marker, repeat_candidate_pngs = _unity_execution(
            root, f"repeats/{ordinal}/counterfactual", repeated_candidate, "counterfactual", command,
            candidate_project_manifest, counterfactual_scene_artifact["sha256"],
            deterministic=(
                deterministic
                and downgraded_marker_prefix != f"repeats/{ordinal}/counterfactual"
            ),
        )
        repeat_runtime_payload = {
            **runtime_payload,
            "run_id": _run_id(
                repeated_factual["frames"],
                repeated_candidate["frames"],
                repeat_factual_log,
                repeat_candidate_log,
                repeat_factual_marker,
                repeat_candidate_marker,
            ),
            "factual_execution_log": repeat_factual_log,
            "counterfactual_execution_log": repeat_candidate_log,
            "factual_execution_marker": repeat_factual_marker,
            "counterfactual_execution_marker": repeat_candidate_marker,
            "factual_pngs": repeat_factual_pngs,
            "counterfactual_pngs": repeat_candidate_pngs,
            "factual_project_manifest": factual_project_manifest,
            "counterfactual_project_manifest": candidate_project_manifest,
            "factual_frames_sha256": repeated_factual["frames"]["sha256"],
            "counterfactual_frames_sha256": repeated_candidate["frames"]["sha256"],
        }
        repeat_runtime = _write(
            root,
            f"repeats/{ordinal}-runtime-provenance.json",
            (json.dumps(repeat_runtime_payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        payload = {
            "counterfactual": repeated_candidate,
            "factual": repeated_factual,
            "ordinal": ordinal,
            "runtime_provenance": repeat_runtime,
            "runtime_provenance_sha256": repeat_runtime["sha256"],
            "trace_commitment": trace["sha256"],
        }
        artifact = _write(
            root,
            f"repeats/{ordinal}.json",
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        repeats.append({
            "ordinal": ordinal,
            "artifact": artifact,
            "factual": repeated_factual,
            "counterfactual": repeated_candidate,
            "runtime_provenance": repeat_runtime,
            "runtime_provenance_sha256": repeat_runtime["sha256"],
        })
    value = {
        "schema": "flashpatch-l10-engine-evidence-v1",
        "evidence_class": "natural_project",
        "engine": "Unity",
        "engine_version": "2022.3.8f1",
        "source": {
            "repository": "https://github.com/example/game",
            "revision": "a" * 40,
            "tree_sha256": tree_hash,
            "provenance": source_provenance,
        },
        "license": {"spdx": "MIT", "artifact": license_ref},
        "scene": {
            "source_path": "Assets/Scene.unity",
            "artifact": scene_artifact,
            "counterfactual_artifact": counterfactual_scene_artifact,
        },
        "trace": {"artifact": trace, "commitment": trace["sha256"]},
        "runtime": {
            "kind": "container",
            "digest": "d" * 64,
            "command": command,
            "engine_identity": identity,
            "provenance": runtime_provenance,
        },
        "renderer": {"backend": "Vulkan", "color_space": "sRGB", "width": 5, "height": 4},
        "factual": factual_capture,
        "attribution": {
            "object_identity": "Portal",
            "component": "Light",
            "property": "intensity",
            "artifact": attribution,
        },
        "patch": {"kind": "minimal_source_parameter", "files_changed": 1, "artifact": patch},
        "counterfactual": candidate_capture,
        "preservation": {
            "action_sequence": True,
            "gameplay_state": state_preserved,
            "object_identity": True,
            "terminal_state": state_preserved,
            "timing": True,
            "visual_intent": True,
        },
        "parity": {"decoder": True, "timeline": True},
        "repeats": repeats,
        "verdict": "INCONCLUSIVE" if inconclusive or not state_preserved else "PASS",
        "reason": (
            "factual_capture_was_safe_and_rgb_artifact_hashes_were_not_reproducible"
            if inconclusive
            else (
                "gameplay_state_was_not_preserved"
                if not state_preserved
                else "same_trace_counterfactual_removed_declared_risk"
            )
        ),
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_verifies_complete_engine_receipt(tmp_path: Path) -> None:
    result = _verify(_receipt(tmp_path))
    assert result["verified"] is True
    assert result["artifact_graph_verified"] is True
    assert result["execution_origin_authenticated"] is False
    assert result["repeats"] == 3


def test_verifies_deterministic_unity_v2_receipt(tmp_path: Path) -> None:
    result = _verify(_receipt(tmp_path, deterministic=True))
    assert result["verified"] is True
    assert result["repeat_reproducible"] is True


@pytest.mark.parametrize(
    "prefix",
    ("baseline/factual", "repeats/2/counterfactual"),
)
def test_rejects_deterministic_unity_v2_marker_schema_downgrade(
    tmp_path: Path,
    prefix: str,
) -> None:
    receipt = _receipt(
        tmp_path,
        deterministic=True,
        downgraded_marker_prefix=prefix,
    )
    with pytest.raises(L10ReceiptError, match="marker schema does not match the adapter"):
        _verify(receipt)


def test_rejects_deterministic_unity_v2_llvmpipe_marker(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, deterministic=True)
    value = json.loads(receipt.read_text())
    provenance_path = tmp_path / value["runtime"]["provenance"]["path"]
    provenance = json.loads(provenance_path.read_text())
    marker_ref = provenance["factual_execution_marker"]
    marker_path = tmp_path / marker_ref["path"]
    marker = json.loads(marker_path.read_text())
    marker.update({
        "graphics_device_id": 0,
        "graphics_device_name": "llvmpipe",
        "graphics_device_type": "OpenGLCore",
        "graphics_device_vendor": "Mesa",
    })
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    marker_ref["sha256"] = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    value["runtime"]["provenance"]["sha256"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="graphics device is not approved"):
        _verify(receipt)


def test_rejects_deterministic_unity_v2_without_rendered_values(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path, deterministic=True)
    value = json.loads(receipt.read_text())
    events_ref = value["factual"]["runtime_events"]
    events_path = tmp_path / events_ref["path"]
    events = json.loads(events_path.read_text())
    for row in events:
        row.pop("rendered_value")
    events_path.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n")
    events_ref["sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="execution marker binding"):
        _verify(receipt)


def test_rejects_deterministic_unity_v2_with_dynamic_counterfactual_cause(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path, deterministic=True)
    value = json.loads(receipt.read_text())
    events_ref = value["counterfactual"]["runtime_events"]
    events_path = tmp_path / events_ref["path"]
    events = json.loads(events_path.read_text())
    for index, row in enumerate(events):
        row["rendered_value"] = float(index + 1)
    events_path.write_text(json.dumps(events, indent=2, sort_keys=True) + "\n")
    events_ref["sha256"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="execution marker binding"):
        _verify(receipt)


def test_reopens_inconclusive_engine_receipt_without_promoting_it(tmp_path: Path) -> None:
    result = _verify(_receipt(tmp_path, inconclusive=True))
    assert result["verified"] is False
    assert result["artifact_graph_verified"] is True
    assert result["repeat_reproducible"] is False
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["scoreable"] is False
    assert result["external_claim_authorized"] is False
    assert result["gate_results"] == {
        "actual_final_frame_capture": True,
        "allowlisted_single_property_change": True,
        "editor_entitlement_bound": False,
        "editor_execution": True,
        "execution_origin_authenticated": False,
        "gameplay_state_preservation": True,
        "hazard_reduction_or_removal": False,
        "independent_repeated_executions": True,
        "repeat_capture_reproducibility": False,
        "same_trace": True,
        "source_revision_and_license": True,
        "strict_timestamp_integrity": True,
    }
    assert result["unmet_gates"] == [
        "EDITOR_ENTITLEMENT_NOT_BOUND",
        "EXECUTION_ORIGIN_NOT_AUTHENTICATED",
        "HAZARD_REDUCTION_NOT_ESTABLISHED",
        "REPEAT_CAPTURE_NOT_REPRODUCIBLE",
    ]


def test_rejects_inconclusive_receipt_with_false_preservation_claim(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, inconclusive=True)
    value = json.loads(receipt.read_text())
    value["preservation"]["gameplay_state"] = False
    value["preservation"]["terminal_state"] = False
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="does not match gameplay state"):
        _verify(receipt)


@pytest.mark.parametrize(
    "false_flag",
    (None, "action_sequence", "object_identity", "timing", "visual_intent"),
)
def test_rejects_inconclusive_downgrade_of_completed_evidence(
    tmp_path: Path,
    false_flag: str | None,
) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["verdict"] = "INCONCLUSIVE"
    value["reason"] = "downgraded"
    if false_flag is not None:
        value["preservation"][false_flag] = False
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="preservation claim|contradicts completed"):
        _verify(receipt)


def test_accepts_inconclusive_when_hazard_removed_but_state_not_preserved(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path, state_preserved=False)
    result = _verify(receipt)
    assert result["verified"] is False
    assert result["artifact_graph_verified"] is True
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gate_results"]["gameplay_state_preservation"] is False
    assert "GAMEPLAY_STATE_NOT_PRESERVED" in result["unmet_gates"]


def test_rejects_tampered_artifact(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "factual-frames.npz").write_bytes(b"tampered")
    with pytest.raises(L10ReceiptError, match="hash mismatch"):
        _verify(receipt)


def test_rejects_symlinked_artifact_parent(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = tmp_path / "repeats"
    moved = outside / "repeats"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(L10ReceiptError, match="symlink"):
        _verify(receipt)


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    raw = receipt.read_text()
    receipt.write_text(raw.replace('"schema":', '"schema":"counterfeit", "schema":', 1))
    with pytest.raises(L10ReceiptError, match="duplicate"):
        _verify(receipt)


def test_rejects_fake_engine_identity(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "engine.json").write_text('{"engine":"Unreal","version":"5.6.0"}\n')
    value = json.loads(receipt.read_text())
    value["runtime"]["engine_identity"]["sha256"] = hashlib.sha256((tmp_path / "engine.json").read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="identity"):
        _verify(receipt)


def test_rejects_stale_or_missing_repeat(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["repeats"] = value["repeats"][:2]
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="three"):
        _verify(receipt)


def test_rejects_symlinked_receipt_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    receipt = _receipt(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(L10ReceiptError, match="symlink"):
        _verify(alias / receipt.name)


def test_bundle_requires_all_three_distinct_engines(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    unity = _receipt(receipts)
    ref = {"path": str(unity.relative_to(tmp_path)), "sha256": hashlib.sha256(unity.read_bytes()).hexdigest()}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema": "flashpatch-l10-engine-neutral-bundle-v1",
        "receipts": [ref, ref, ref],
    }, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="duplicate engine"):
        policy, digest = _policy(unity)
        verify_l10_bundle(bundle, policy, digest)


def test_controlled_passes_do_not_establish_engine_neutral_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classes = {
        "Godot": "controlled_fixture",
        "Unity": "natural_project",
        "Unreal": "controlled_fixture",
    }
    references = []
    for engine, evidence_class in classes.items():
        path = tmp_path / f"{engine.lower()}.json"
        path.write_text(
            json.dumps(
                {"evidence_class": evidence_class},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        references.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "flashpatch-l10-engine-neutral-bundle-v1",
                "receipts": references,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    def verified_receipt(path: Path, *_: object) -> dict[str, object]:
        engine = path.stem.title()
        return {
            "verified": True,
            "engine": engine,
            "evidence_class": classes[engine],
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(
        "flashpatch.l10_receipt.verify_l10_receipt",
        verified_receipt,
    )
    result = verify_l10_bundle(bundle, tmp_path / "unused-policy.json", "a" * 64)

    assert result["verified"] is False
    assert result["engine_neutral"] is False
    assert result["verdict"] == "INCONCLUSIVE"


def test_repository_engine_bundle_reopens_fail_closed() -> None:
    evidence = Path(__file__).parents[1] / "evidence/l10"
    policy = evidence / "engine-neutral-inconclusive-v1-trust-policy.json"
    result = verify_l10_bundle(
        evidence / "engine-neutral-inconclusive-v1-bundle.json",
        policy,
        hashlib.sha256(policy.read_bytes()).hexdigest(),
    )

    assert {
        engine: (row["evidence_class"], row["verdict"], row["verified"])
        for engine, row in result["engines"].items()
    } == {
        "Godot": ("controlled_fixture", "PASS", True),
        "Unity": ("natural_project", "INCONCLUSIVE", False),
        "Unreal": ("controlled_fixture", "PASS", True),
    }
    assert result["evidence_classes"] == {
        "controlled_fixture": ["Godot", "Unreal"],
        "natural_project": ["Unity"],
    }
    assert result["verified"] is False
    assert result["engine_neutral"] is False
    assert result["external_claim_authorized"] is False
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["engines"]["Unity"]["unmet_gates"] == [
        "EDITOR_ENTITLEMENT_NOT_BOUND",
        "EXECUTION_ORIGIN_NOT_AUTHENTICATED",
        "GAMEPLAY_STATE_NOT_PRESERVED",
        "HAZARD_REDUCTION_NOT_ESTABLISHED",
        "REPEAT_CAPTURE_NOT_REPRODUCIBLE",
    ]


def test_repository_unity_deadline_assessment_reopens_exact_verification() -> None:
    evidence = Path(__file__).parents[1] / "evidence/l10"
    assessment_path = (
        evidence
        / "unity-natural-2022.3.8f1-inconclusive-v1/deadline-assessment.json"
    )
    assessment_raw = assessment_path.read_bytes()
    assessment = json.loads(assessment_raw)
    assert assessment_raw == (
        json.dumps(assessment, indent=2, sort_keys=True) + "\n"
    ).encode()
    assert assessment["schema"] == "flashpatch-l10-unity-deadline-assessment-v1"

    policy = evidence / "engine-neutral-inconclusive-v1-trust-policy.json"
    receipt = evidence / assessment["anchors"]["unity_receipt"]["path"]
    verification = verify_l10_receipt(
        receipt,
        policy,
        hashlib.sha256(policy.read_bytes()).hexdigest(),
    )
    assert assessment["verification"] == verification
    assert assessment["verdict"] == verification["verdict"] == "INCONCLUSIVE"
    assert assessment["external_claim_authorized"] is False
    assert assessment["unmet_gates"] == verification["unmet_gates"]

    for name, reference in assessment["anchors"].items():
        path = evidence / reference["path"]
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    assert {
        row["path"]: row["sha256"]
        for row in assessment["l10_receipt_inventory"]
    } == {
        row["path"]: row["sha256"]
        for row in json.loads(
            (evidence / "engine-neutral-inconclusive-v1-bundle.json").read_text()
        )["receipts"]
    }


def test_rejects_identical_factual_and_counterfactual_capture(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["counterfactual"]["frames"] = value["factual"]["frames"]
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="identical"):
        _verify(receipt)


def test_rejects_out_of_range_hazard_frame(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["factual"]["detector"]["hazard_frames"] = [99]
    detector = tmp_path / value["factual"]["detector"]["receipt"]["path"]
    detector_value = json.loads(detector.read_text())
    detector_value["hazard_frames"] = [99]
    detector.write_text(json.dumps(detector_value, indent=2, sort_keys=True) + "\n")
    value["factual"]["detector"]["receipt"]["sha256"] = hashlib.sha256(detector.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="out of range"):
        _verify(receipt)


def test_rejects_forged_detector_claim_despite_matching_receipt(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["factual"]["detector"]["result"] = "SAFE"
    value["factual"]["detector"]["hazard_frames"] = []
    detector = tmp_path / value["factual"]["detector"]["receipt"]["path"]
    detector_value = json.loads(detector.read_text())
    detector_value["result"] = "SAFE"
    detector_value["hazard_frames"] = []
    detector.write_text(json.dumps(detector_value, indent=2, sort_keys=True) + "\n")
    value["factual"]["detector"]["receipt"]["sha256"] = hashlib.sha256(detector.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="fresh analysis"):
        _verify(receipt)


def test_rejects_repeat_reusing_baseline_capture(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    value["repeats"][0]["factual"] = value["factual"]
    repeat = tmp_path / value["repeats"][0]["artifact"]["path"]
    repeat_value = json.loads(repeat.read_text())
    repeat_value["factual"] = value["factual"]
    repeat.write_text(json.dumps(repeat_value, indent=2, sort_keys=True) + "\n")
    value["repeats"][0]["artifact"]["sha256"] = hashlib.sha256(repeat.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="reuses"):
        _verify(receipt)


def test_rejects_run_id_not_derived_from_bound_artifacts(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    provenance = tmp_path / value["runtime"]["provenance"]["path"]
    provenance_value = json.loads(provenance.read_text())
    provenance_value["run_id"] = "f" * 64
    provenance.write_text(json.dumps(provenance_value, indent=2, sort_keys=True) + "\n")
    value["runtime"]["provenance"]["sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="not derived"):
        _verify(receipt)


def test_rejects_run_provenance_bound_to_other_frames(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    provenance = tmp_path / value["runtime"]["provenance"]["path"]
    provenance_value = json.loads(provenance.read_text())
    provenance_value["factual_frames_sha256"] = "e" * 64
    provenance.write_text(json.dumps(provenance_value, indent=2, sort_keys=True) + "\n")
    value["runtime"]["provenance"]["sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="successful engine execution"):
        _verify(receipt)


def test_rejects_unity_scene_pair_that_is_not_exact_one_parameter_patch(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    candidate = tmp_path / value["scene"]["counterfactual_artifact"]["path"]
    candidate.write_bytes(b"unrelated candidate scene\n")
    value["scene"]["counterfactual_artifact"]["sha256"] = hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="exact one-parameter"):
        _verify(receipt)


def test_rejects_unity_project_manifests_differing_outside_scene(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    provenance_path = tmp_path / value["runtime"]["provenance"]["path"]
    provenance = json.loads(provenance_path.read_text())
    manifest_ref = provenance["counterfactual_project_manifest"]
    manifest_path = tmp_path / manifest_ref["path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][1]["sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_ref["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    value["runtime"]["provenance"]["sha256"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="marker artifact binding"):
        _verify(receipt)


def test_rejects_receipt_not_matching_externally_hashed_policy(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    policy, digest = _policy(receipt)
    value = json.loads(receipt.read_text())
    value["source"]["repository"] = "https://github.com/attacker/forged-game"
    provenance = tmp_path / value["source"]["provenance"]["path"]
    provenance_value = json.loads(provenance.read_text())
    provenance_value["repository"] = value["source"]["repository"]
    provenance.write_text(json.dumps(provenance_value, indent=2, sort_keys=True) + "\n")
    value["source"]["provenance"]["sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="externally anchored"):
        verify_l10_receipt(receipt, policy, digest)


def test_rejects_resealed_event_and_state_refs_against_original_policy(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path, deterministic=True, inconclusive=True)
    policy, digest = _policy(receipt)
    value = json.loads(receipt.read_text())
    for field, payload in (
        ("runtime_events", [{"frame": 0, "rendered_value": 1000000.0}]),
        ("state_stream", [{"frame": 0, "terminal_state": "forged"}]),
    ):
        reference = value["factual"][field]
        artifact = tmp_path / reference["path"]
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reference["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="externally anchored"):
        verify_l10_receipt(receipt, policy, digest)


def test_rejects_resealed_state_even_with_regenerated_policy(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path, deterministic=True, inconclusive=True)
    value = json.loads(receipt.read_text())
    reference = value["factual"]["state_stream"]
    artifact = tmp_path / reference["path"]
    states = json.loads(artifact.read_text())
    states[0]["terminal_state"] = "forged-after-execution"
    artifact.write_text(json.dumps(states, indent=2, sort_keys=True) + "\n")
    reference["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="execution marker binding"):
        _verify(receipt)


def test_rejects_tampered_trust_policy(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    policy, digest = _policy(receipt)
    policy.write_bytes(policy.read_bytes() + b" ")
    with pytest.raises(L10ReceiptError, match="hash mismatch"):
        verify_l10_receipt(receipt, policy, digest)


def test_rejects_hardlinked_receipt(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "receipt-hardlink.json").hardlink_to(receipt)
    with pytest.raises(L10ReceiptError, match="exclusively owned"):
        _verify(receipt)


def test_bundle_rejects_when_no_engine_has_natural_project_evidence(tmp_path: Path) -> None:
    receipts = []
    receipt_paths = []
    for index, engine in enumerate(("Godot", "Unity", "Unreal")):
        root = tmp_path / engine
        root.mkdir()
        path = _receipt(root)
        value = json.loads(path.read_text())
        value["evidence_class"] = "controlled_fixture"
        value["engine"] = engine
        identity = root / value["runtime"]["engine_identity"]["path"]
        identity.write_text(json.dumps({"engine": engine, "version": value["engine_version"]}, separators=(",", ":")) + "\n")
        value["runtime"]["engine_identity"]["sha256"] = hashlib.sha256(identity.read_bytes()).hexdigest()
        provenance = root / value["runtime"]["provenance"]["path"]
        provenance_value = json.loads(provenance.read_text())
        provenance_value["engine"] = engine
        provenance.write_text(json.dumps(provenance_value, indent=2, sort_keys=True) + "\n")
        value["runtime"]["provenance"]["sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
        value["patch"]["kind"] = "controlled_runtime_parameter"
        value["patch"]["files_changed"] = 0
        patch_path = root / value["patch"]["artifact"]["path"]
        patch_value = json.loads(patch_path.read_text())
        patch_value["kind"] = value["patch"]["kind"]
        patch_value["files_changed"] = value["patch"]["files_changed"]
        patch_path.write_text(json.dumps(patch_value, indent=2, sort_keys=True) + "\n")
        value["patch"]["artifact"]["sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        for repeat in value["repeats"]:
            repeat_provenance_path = root / repeat["runtime_provenance"]["path"]
            repeat_provenance_value = json.loads(repeat_provenance_path.read_text())
            repeat_provenance_value["engine"] = engine
            repeat_provenance_path.write_text(
                json.dumps(repeat_provenance_value, indent=2, sort_keys=True) + "\n"
            )
            repeat["runtime_provenance"]["sha256"] = hashlib.sha256(
                repeat_provenance_path.read_bytes()
            ).hexdigest()
            repeat["runtime_provenance_sha256"] = repeat["runtime_provenance"]["sha256"]
            repeat_path = root / repeat["artifact"]["path"]
            repeat_value = json.loads(repeat_path.read_text())
            repeat_value["runtime_provenance"] = repeat["runtime_provenance"]
            repeat_value["runtime_provenance_sha256"] = repeat["runtime_provenance_sha256"]
            repeat_path.write_text(json.dumps(repeat_value, indent=2, sort_keys=True) + "\n")
            repeat["artifact"]["sha256"] = hashlib.sha256(repeat_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        receipt_paths.append(path)
        receipts.append({"path": str(path.relative_to(tmp_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"schema": "flashpatch-l10-engine-neutral-bundle-v1", "receipts": receipts}, indent=2, sort_keys=True) + "\n")
    with pytest.raises(L10ReceiptError, match="at least one natural-project"):
        policy, digest = _policy(*receipt_paths)
        verify_l10_bundle(bundle, policy, digest)
_UNITY_LEGACY_ADAPTER_SHA256_V1 = (
    "1b1b131793a1a802c1897dd23523324e21034a383ae881bc3719308c6fe85c03"
)
