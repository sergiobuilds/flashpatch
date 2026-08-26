from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .safety_ci import compile_project, load_contract


SCHEMA = "flashpatch-godot-submission-demo-v1"
VERIFY_SCHEMA = "flashpatch-godot-submission-demo-verification-v1"


class SubmissionDemoError(ValueError):
    """A submission-demo artifact failed its public verification contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_engine_receipt(value: object, output: Path) -> object:
    if isinstance(value, dict):
        return {key: _relative_engine_receipt(item, output) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_engine_receipt(item, output) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(output.resolve()).as_posix()
            except ValueError:
                return f"external-input/{candidate.name}"
    return value


def _load_frames(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise SubmissionDemoError("renderer artifact must contain uint8 RGB frames")
    return frames


def _write_rgb(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise SubmissionDemoError(f"could not write image: {path.name}")


def _comparison(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    height, width = before.shape[:2]
    header = 38
    canvas = np.zeros((height + header, width * 2, 3), dtype=np.uint8)
    canvas[header:, :width] = before
    canvas[header:, width:] = after
    cv2.putText(canvas, "BEFORE: HAZARD", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, "AFTER: PASS", (width + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (width, 0), (width, height + header), (0, 180, 255), 2)
    return canvas


def _minimal_single_line_diff(diff: str, source: str, source_line: int) -> str:
    deleted = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if len(deleted) != 1 or len(added) != 1:
        raise SubmissionDemoError("submission demo patch must contain exactly one changed source line")
    return "\n".join([
        f"--- a/{source}",
        f"+++ b/{source}",
        f"@@ -{source_line} +{source_line} @@",
        deleted[0],
        added[0],
        "",
    ])


def run_submission_godot_demo(
    project: Path | str,
    contract_path: Path | str,
    output: Path | str,
) -> dict[str, object]:
    project_path = Path(project).resolve()
    contract_file = Path(contract_path).resolve()
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    contract = load_contract(project_path, contract_file)
    if len(contract.candidates) != 1:
        raise SubmissionDemoError("submission demo requires exactly one declared patch candidate")
    declared = contract.candidates[0]
    source_hash_before = _sha256_file(declared.source)
    engine = compile_project(project_path, contract_file, workspace=output_path / "workspace")
    source_hash_after = _sha256_file(declared.source)

    engine_path = output_path / "engine-receipt.json"
    _write_json(engine_path, _relative_engine_receipt(engine, output_path))
    if engine.get("verdict") != "PASS":
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "verdict": engine.get("verdict", "INCONCLUSIVE"),
            "reason": engine.get("reason", "engine_demo_did_not_pass"),
            "source_isolation": {
                "original_unchanged": source_hash_before == source_hash_after,
                "source_sha256_before": source_hash_before,
                "source_sha256_after": source_hash_after,
            },
            "artifacts": {"engine-receipt.json": _sha256_file(engine_path)},
        }
        receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
        _write_json(output_path / "receipt.json", receipt)
        return receipt

    factual = engine.get("factual_replay")
    attribution = engine.get("attribution")
    if not isinstance(factual, dict) or not isinstance(attribution, dict):
        raise SubmissionDemoError("PASS engine receipt omitted factual or attribution evidence")
    factual_frames = _load_frames(Path(str(factual["frame_artifact"])))
    candidate_frames = _load_frames(Path(str(attribution["frame_artifact"])))
    hazard_indices = factual.get("hazard_frame_indices")
    if not isinstance(hazard_indices, list) or not hazard_indices:
        raise SubmissionDemoError("PASS engine receipt omitted hazard frame indices")
    valid_indices = [
        index for index in hazard_indices
        if isinstance(index, int) and 0 <= index < min(len(factual_frames), len(candidate_frames))
    ]
    if not valid_indices:
        raise SubmissionDemoError("PASS engine receipt has no usable hazard frame index")
    frame_index = max(valid_indices, key=lambda index: float(np.mean(factual_frames[index])))

    before_path = output_path / "before.png"
    after_path = output_path / "after.png"
    comparison_path = output_path / "comparison.png"
    diff_path = output_path / "patch.diff"
    _write_rgb(before_path, factual_frames[frame_index])
    _write_rgb(after_path, candidate_frames[frame_index])
    _write_rgb(comparison_path, _comparison(factual_frames[frame_index], candidate_frames[frame_index]))
    original_diff = Path(str(attribution["diff"]))
    source_line = attribution.get("source_line")
    if not isinstance(source_line, int) or source_line < 1:
        raise SubmissionDemoError("PASS engine receipt omitted a valid source line")
    source_relative = declared.source.relative_to(project_path).as_posix()
    diff_path.write_text(
        _minimal_single_line_diff(
            original_diff.read_text(encoding="utf-8"),
            source_relative,
            source_line,
        ),
        encoding="utf-8",
    )

    factual_capture = factual.get("renderer_capture")
    candidate_capture = attribution.get("renderer_capture")
    runtime = attribution.get("runtime_attribution")
    if not isinstance(factual_capture, dict) or not isinstance(candidate_capture, dict) or not isinstance(runtime, dict):
        raise SubmissionDemoError("PASS engine receipt omitted renderer or runtime attribution")
    artifacts = {
        name: _sha256_file(output_path / name)
        for name in ("before.png", "after.png", "comparison.png", "patch.diff", "engine-receipt.json")
    }
    receipt = {
        "schema": SCHEMA,
        "verdict": "PASS",
        "reason": engine.get("reason"),
        "engine": {
            "name": "Godot",
            "version": factual_capture.get("godot_version"),
            "renderer_configuration": factual_capture.get("renderer_configuration"),
            "actual_renderer_frames": True,
        },
        "input": {
            "project": project_path.name,
            "contract": contract_file.name,
            "project_file_sha256": engine["input_sha256"]["project.godot"],
            "source_snapshot_sha256": engine["input_sha256"]["source_snapshot"],
            "trace_sha256": engine["input_sha256"]["trace"],
        },
        "hazard": {
            "before": {
                "hazardous": factual.get("hazardous"),
                "max_risk": factual.get("max_risk"),
                "kinds": factual.get("hazard_kinds"),
            },
            "after": {
                "hazardous": attribution.get("hazardous"),
                "max_risk": attribution.get("max_risk"),
                "kinds": attribution.get("hazard_kinds"),
            },
            "threshold": engine["risk_signal"]["threshold"],
            "comparison_frame_index": frame_index,
        },
        "patch": {
            "source": source_relative,
            "source_line": source_line,
            "node": attribution.get("node"),
            "parameter": attribution.get("parameter"),
            "original_value": runtime.get("factual_value"),
            "replacement": attribution.get("replacement"),
            "changed_source_assignments": attribution.get("changed_source_assignments"),
            "causal_contribution": attribution.get("causal_contribution"),
            "diff": "patch.diff",
        },
        "replay": {
            "same_trace": factual_capture.get("trace_sha256") == candidate_capture.get("trace_sha256"),
            "trace_sha256": factual_capture.get("trace_sha256"),
            "timing_preserved": attribution.get("timing_preserved"),
            "gameplay_state_preserved": attribution.get("gameplay_state_preserved"),
            "semantic_invariants_preserved": attribution.get("semantic_invariants_preserved"),
            "gameplay_state_sha256": attribution.get("gameplay_state_sha256"),
        },
        "source_isolation": {
            "original_unchanged": source_hash_before == source_hash_after,
            "source_sha256_before": source_hash_before,
            "source_sha256_after": source_hash_after,
            "patched_only_in_workspace_copy": True,
        },
        "artifacts": artifacts,
        "engine_receipt_payload_sha256": _sha256_bytes(_canonical(engine)),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
    _write_json(output_path / "receipt.json", receipt)
    return receipt


def _artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        raise SubmissionDemoError(f"artifact path is missing or escapes receipt root: {relative}")
    return candidate


def verify_submission_godot_demo(receipt_path: Path | str) -> dict[str, object]:
    path = Path(receipt_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise SubmissionDemoError("submission demo receipt schema is invalid")
    claimed = payload.get("receipt_sha256")
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if claimed != _sha256_bytes(_canonical(unhashed)):
        raise SubmissionDemoError("submission demo receipt hash does not match")
    if payload.get("verdict") != "PASS":
        raise SubmissionDemoError("submission demo receipt is not PASS")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "before.png", "after.png", "comparison.png", "patch.diff", "engine-receipt.json"
    }:
        raise SubmissionDemoError("submission demo artifact manifest is incomplete")
    for relative, expected_hash in artifacts.items():
        artifact = _artifact_path(path.parent, relative)
        if _sha256_file(artifact) != expected_hash:
            raise SubmissionDemoError(f"submission demo artifact hash mismatch: {relative}")
    hazard = payload.get("hazard")
    replay = payload.get("replay")
    patch = payload.get("patch")
    isolation = payload.get("source_isolation")
    if not all(isinstance(item, dict) for item in (hazard, replay, patch, isolation)):
        raise SubmissionDemoError("submission demo receipt omitted a required evidence section")
    before = hazard.get("before")
    after = hazard.get("after")
    threshold = hazard.get("threshold")
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(threshold, (int, float)):
        raise SubmissionDemoError("submission demo hazard evidence is invalid")
    if before.get("hazardous") is not True or after.get("hazardous") is not False:
        raise SubmissionDemoError("submission demo does not prove hazard removal")
    if not isinstance(before.get("max_risk"), (int, float)) or float(before["max_risk"]) < float(threshold):
        raise SubmissionDemoError("submission demo factual risk is below its threshold")
    if not isinstance(after.get("max_risk"), (int, float)) or float(after["max_risk"]) >= float(threshold):
        raise SubmissionDemoError("submission demo patched risk is not below its threshold")
    if patch.get("changed_source_assignments") != 1:
        raise SubmissionDemoError("submission demo patch is not a single source assignment")
    if not all(replay.get(key) is True for key in (
        "same_trace", "timing_preserved", "gameplay_state_preserved", "semantic_invariants_preserved"
    )):
        raise SubmissionDemoError("submission demo replay preservation evidence is incomplete")
    if isolation.get("original_unchanged") is not True or isolation.get("patched_only_in_workspace_copy") is not True:
        raise SubmissionDemoError("submission demo did not preserve the original project")
    if isolation.get("source_sha256_before") != isolation.get("source_sha256_after"):
        raise SubmissionDemoError("submission demo original source hash changed")
    for image_name in ("before.png", "after.png", "comparison.png"):
        if cv2.imread(str(path.parent / image_name), cv2.IMREAD_COLOR) is None:
            raise SubmissionDemoError(f"submission demo image cannot be decoded: {image_name}")
    return {
        "schema": VERIFY_SCHEMA,
        "verified": True,
        "verdict": "PASS",
        "receipt_sha256": claimed,
        "artifact_count": len(artifacts),
        "parameter": patch.get("parameter"),
        "hazard_before": before.get("max_risk"),
        "hazard_after": after.get("max_risk"),
        "gameplay_state_preserved": replay.get("gameplay_state_preserved"),
    }


def format_submission_godot_demo(receipt: dict[str, Any], output: Path | str) -> str:
    if receipt.get("verdict") != "PASS":
        return f"FlashPatch Godot demo\nRESULT {receipt.get('verdict')}\nREASON {receipt.get('reason')}"
    hazard = receipt["hazard"]
    patch = receipt["patch"]
    replay = receipt["replay"]
    return "\n".join([
        "FlashPatch Godot renderer demo",
        f"INPUT     actual Godot frames, risk={hazard['before']['max_risk']}",
        f"LOCALIZE  {patch['node']} -> {patch['source']}:{patch['source_line']} -> {patch['parameter']}",
        f"PATCH     {patch['original_value']} -> {patch['replacement']} ({patch['changed_source_assignments']} source assignment)",
        f"REPLAY    same_trace={str(replay['same_trace']).lower()} gameplay_state_preserved={str(replay['gameplay_state_preserved']).lower()}",
        f"RESULT    {receipt['verdict']} risk={hazard['before']['max_risk']} -> {hazard['after']['max_risk']}",
        f"ARTIFACTS {Path(output).resolve()}",
        "VERIFY    flashpatch verify-godot-demo <artifacts>/receipt.json",
    ])
