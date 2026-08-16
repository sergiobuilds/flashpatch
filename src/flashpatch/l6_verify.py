from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .l6_authority import L6_PREFLIGHT_PINS
from .l6_run import (
    EXECUTION_CHECKPOINT_SCHEMA,
    EXECUTION_SCHEMA,
    L6_REPEAT_COUNT,
    PREFLIGHT_CHECKPOINT_SCHEMA,
    PREFLIGHT_SCHEMA,
    _validate_positive_engine_receipt,
)


VERIFICATION_SCHEMA = "flashpatch-l6-verification-v1"
ADVERSARIAL_MANIFEST_SCHEMA = "flashpatch-l6-adversarial-manifest-v1"
ADVERSARIAL_RESULT_SCHEMA = "flashpatch-l6-adversarial-result-v1"
CANONICAL_ADVERSARIAL_MANIFEST = Path(
    "artifacts/l6/adversarial/adversarial-manifest.json"
)

_EXPECTED_ADVERSARIAL = {
    "N1": ("renderer_capture", "timestamps_invalid"),
    "N2": ("runtime_attribution", "provenance_mismatch"),
    "N3": ("runtime_attribution", "contributor_not_observed"),
    "N4": ("gameplay_preservation", "invariant_mismatch"),
    "N5": ("patch_search", "no_factual_hazard"),
    "N6": ("patch_validation", "counterfactual_not_causal"),
}
_EXPECTED_CHECKS = {
    "upstream_path",
    "git_top_level",
    "git_origin",
    "git_revision",
    "git_clean_tree",
    "license_sha256",
    "required_project_inputs",
    "entry_scene",
    "symlink_free_inputs",
    "godot_canonical_path",
    "godot_regular_executable",
    "godot_sha256",
    "godot_version",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerificationFailure(Exception):
    """One stable fail-closed L6 verifier decision."""

    def __init__(self, gate: str, reason: str) -> None:
        super().__init__(f"FAIL_CLOSED:{gate}:{reason}")
        self.gate = gate
        self.reason = reason

    @property
    def diagnostic(self) -> str:
        return f"FAIL_CLOSED:{self.gate}:{self.reason}"


def _fail(gate: str, reason: str) -> NoReturn:
    raise VerificationFailure(gate, reason)


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _owned_file(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    resolved_root = root.resolve()
    lexical = candidate if candidate.is_absolute() else resolved_root / candidate
    try:
        lexical_relative = lexical.relative_to(resolved_root)
        current = resolved_root
        for part in lexical_relative.parts:
            current /= part
            if current.is_symlink():
                return None
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or relative == Path("."):
        return None
    return resolved


def _owned_directory(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    resolved_root = root.resolve()
    lexical = candidate if candidate.is_absolute() else resolved_root / candidate
    try:
        lexical_relative = lexical.relative_to(resolved_root)
        current = resolved_root
        for part in lexical_relative.parts:
            current /= part
            if current.is_symlink():
                return None
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved.is_dir() or relative == Path("."):
        return None
    return resolved


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_owned_json(root: Path, value: object) -> tuple[Path | None, dict[str, Any] | None]:
    path = _owned_file(root, value)
    if path is None:
        return None, None
    return path, _read_json_file(path)


def _require_case_root(case_root: Path | str) -> Path:
    supplied = Path(case_root)
    if supplied.is_symlink():
        _fail("receipt_integrity", "case_root_invalid")
    try:
        root = supplied.resolve(strict=True)
    except OSError:
        _fail("receipt_integrity", "case_root_missing")
    if not root.is_dir() or root.is_symlink():
        _fail("receipt_integrity", "case_root_invalid")
    return root


def _presentation_timestamps() -> list[int]:
    return [
        (index * 1_000_000) // L6_PREFLIGHT_PINS.fixed_fps
        for index in range(L6_PREFLIGHT_PINS.capture_ticks)
    ]


def _require_replay_views(
    engine_root: Path,
    engine: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    factual = engine.get("factual_replay")
    candidates = engine.get("candidates")
    attribution = engine.get("attribution")
    if (
        not isinstance(factual, dict)
        or not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], dict)
        or attribution != candidates[0]
    ):
        _fail("patch_validation", "counterfactual_not_causal")
    candidate = candidates[0]

    expected_presentation = _presentation_timestamps()
    for replay in (factual, candidate):
        capture = replay.get("renderer_capture")
        if not isinstance(capture, dict):
            _fail("renderer_capture", "timestamps_invalid")
        actual = capture.get("actual_capture_timestamps_us")
        presentation = capture.get("presentation_timestamps_us")
        if (
            not isinstance(actual, list)
            or len(actual) != L6_PREFLIGHT_PINS.capture_ticks
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in actual
            )
            or any(right <= left for left, right in zip(actual, actual[1:]))
            or presentation != expected_presentation
            or not _is_hex64(replay.get("timestamps_sha256"))
        ):
            _fail("renderer_capture", "timestamps_invalid")

    event_lists: list[list[dict[str, Any]]] = []
    for replay in (factual, candidate):
        _, payload = _read_owned_json(engine_root, replay.get("artifact"))
        raw_events = payload.get("runtime_events") if isinstance(payload, dict) else None
        if (
            not isinstance(raw_events, list)
            or not raw_events
            or not all(isinstance(item, dict) for item in raw_events)
        ):
            _fail("runtime_attribution", "contributor_not_observed")
        event_lists.append([dict(item) for item in raw_events])

    required = {
        "actual_capture_timestamp_us",
        "factual_value",
        "frame_index",
        "node_path",
        "normalized_node_identity",
        "property",
        "resource_path",
        "resource_path_observation",
        "resource_provenance",
        "script_path",
        "script_path_observation",
        "script_sha256",
        "source_line",
        "source_line_observation",
        "spawned_ordinal",
    }
    for replay, events in zip((factual, candidate), event_lists, strict=True):
        actual = replay["renderer_capture"]["actual_capture_timestamps_us"]
        runtime_hash = replay.get("runtime_script_sha256")
        if (
            len(events) != L6_PREFLIGHT_PINS.capture_ticks
            or not _is_hex64(runtime_hash)
            or any(not required.issubset(event) for event in events)
            or any(
                event.get("frame_index") != index
                or event.get("actual_capture_timestamp_us") != actual[index]
                or not isinstance(event.get("node_path"), str)
                or event.get("node_path") == "/root"
                or "/Battle/" not in event.get("node_path", "")
                or event.get("normalized_node_identity")
                != "res://scenes/Battle.tscn::res://scripts/RoutShockwave.gd::0"
                or event.get("spawned_ordinal") != 0
                or event.get("script_path") != "res://scripts/RoutShockwave.gd"
                or event.get("script_path_observation")
                != "node.get_script().resource_path"
                or event.get("script_sha256") != runtime_hash
                or event.get("resource_path") != "res://scenes/Battle.tscn"
                or event.get("resource_path_observation") != "battle.scene_file_path"
                or event.get("resource_provenance")
                not in {"node_owner_scene", "packed_scene_state"}
                or event.get("source_line") != 4
                or event.get("source_line_observation")
                != "FileAccess.get_file_as_string(node.get_script().resource_path)"
                or event.get("property") != "flashpatch_intensity"
                for index, event in enumerate(events)
            )
            or replay.get("runtime_source_line") != 4
        ):
            _fail("runtime_attribution", "provenance_mismatch")

    factual_identity = [
        (event["frame_index"], event["normalized_node_identity"], event["spawned_ordinal"])
        for event in event_lists[0]
    ]
    candidate_identity = [
        (event["frame_index"], event["normalized_node_identity"], event["spawned_ordinal"])
        for event in event_lists[1]
    ]
    if factual_identity != candidate_identity:
        _fail("runtime_attribution", "provenance_mismatch")

    return factual, candidate, event_lists[0], event_lists[1]


def _verify_preservation_claims(
    factual: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    expected_actions = [{"frame": 0, "status": "APPLIED"}]
    required_hashes = (
        "state_stream_sha256",
        "final_state_sha256",
        "final_state_raw_sha256",
        "timestamps_sha256",
    )
    if (
        factual.get("action_acknowledgements") != expected_actions
        or candidate.get("action_acknowledgements") != expected_actions
        or factual.get("tick_domain") != [0, 160]
        or candidate.get("tick_domain") != [0, 160]
        or factual.get("state_stream_tick_domain") != [0, 161]
        or candidate.get("state_stream_tick_domain") != [0, 161]
        or factual.get("state_stream_record_count") != 162
        or candidate.get("state_stream_record_count") != 162
        or any(not _is_hex64(factual.get(key)) for key in required_hashes)
        or any(not _is_hex64(candidate.get(key)) for key in required_hashes)
        or any(factual.get(key) != candidate.get(key) for key in required_hashes)
        or candidate.get("semantic_invariants_preserved") is not True
        or candidate.get("timing_preserved") is not True
    ):
        _fail("gameplay_preservation", "invariant_mismatch")


def _verify_patch_claims(
    engine_root: Path,
    factual: dict[str, Any],
    candidate: dict[str, Any],
    factual_events: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
) -> None:
    factual_risk = factual.get("max_risk")
    if (
        factual.get("hazardous") is not True
        or isinstance(factual_risk, bool)
        or not isinstance(factual_risk, (int, float))
        or float(factual_risk) < 1.0
        or not isinstance(factual.get("hazard_frame_indices"), list)
        or not factual.get("hazard_frame_indices")
        or not isinstance(factual.get("hazard_kinds"), list)
        or not factual.get("hazard_kinds")
    ):
        _fail("patch_search", "no_factual_hazard")

    if (
        candidate.get("parameter") != "flashpatch_intensity"
        or candidate.get("source_line") != 4
        or candidate.get("replacement") != 0.0
        or candidate.get("changed_source_assignments") != 1
        or candidate.get("hazardous") is not False
        or candidate.get("max_risk") != 0.0
        or candidate.get("hazard_frame_indices") != []
        or candidate.get("hazard_kinds") != []
        or candidate.get("status") != "EVALUATED"
        or any(event.get("factual_value") != 1.0 for event in factual_events)
        or any(event.get("factual_value") != 0.0 for event in candidate_events)
    ):
        _fail("patch_validation", "counterfactual_not_causal")

    diff_path = _owned_file(engine_root, candidate.get("diff"))
    if diff_path is None:
        _fail("patch_validation", "counterfactual_not_causal")
    try:
        changed = [
            line
            for line in diff_path.read_text(encoding="utf-8").splitlines()
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith(("+++", "---"))
        ]
    except (OSError, UnicodeError):
        _fail("patch_validation", "counterfactual_not_causal")
    if (
        len(changed) != 2
        or "flashpatch_intensity: float = 1.0" not in changed[0]
        or "flashpatch_intensity: float = 0.0" not in changed[1]
    ):
        _fail("patch_validation", "counterfactual_not_causal")


def _verify_owned_artifacts_and_inputs(
    engine_root: Path,
    engine: dict[str, Any],
) -> None:
    factual = engine["factual_replay"]
    candidate = engine["candidates"][0]
    for replay in (factual, candidate):
        for field in (
            "artifact",
            "frame_artifact",
            "state_stream_artifact",
            "final_state_artifact",
        ):
            if _owned_file(engine_root, replay.get(field)) is None:
                _fail("receipt_integrity", "receipt_owned_artifact_invalid")
    if _owned_file(engine_root, candidate.get("diff")) is None:
        _fail("patch_validation", "counterfactual_not_causal")

    sealed = _owned_directory(engine_root, engine.get("sealed_input_root"))
    project = _owned_directory(engine_root, engine.get("project"))
    trace = _owned_file(engine_root, engine.get("trace"))
    contract = _owned_file(engine_root, engine.get("contract"))
    if sealed is None or project != sealed or trace is None or contract is None:
        _fail("receipt_integrity", "sealed_input_binding_invalid")
    if any(path.is_symlink() for path in sealed.rglob("*")):
        _fail("receipt_integrity", "sealed_input_binding_invalid")
    project_file = _owned_file(sealed, "project.godot")
    inputs = engine.get("input_sha256")
    if project_file is None or not isinstance(inputs, dict):
        _fail("receipt_integrity", "sealed_input_binding_invalid")

    from .safety_ci import _source_tree_sha256

    if (
        inputs.get("project.godot") != f"sha256:{_sha256_file(project_file)}"
        or inputs.get("trace") != f"sha256:{_sha256_file(trace)}"
        or inputs.get("source_snapshot") != _source_tree_sha256(sealed)
        or factual.get("renderer_capture", {}).get("trace_sha256")
        != inputs.get("trace")
        or candidate.get("renderer_capture", {}).get("trace_sha256")
        != inputs.get("trace")
    ):
        _fail("receipt_integrity", "sealed_input_hash_mismatch")


def _verify_engine_contract_gates(
    engine_root: Path,
    engine: dict[str, Any],
) -> None:
    upstream = engine.get("upstream")
    if (
        engine.get("schema") != "flashpatch-renderer-engine-receipt-v1"
        or engine.get("contract_schema") != "flashpatch-godot-safety-ci-v1"
        or engine.get("reason")
        != "same_trace_counterfactual_removed_declared_risk"
        or
        engine.get("controlled_mutation") is not True
        or not isinstance(upstream, dict)
        or upstream.get("classification")
        != "external_dynamic_effect_controlled_mutation"
        or upstream.get("license") != "MIT"
        or upstream.get("project_path") != "."
        or upstream.get("repository_url") != L6_PREFLIGHT_PINS.repository
        or upstream.get("source_revision") != L6_PREFLIGHT_PINS.revision
        or upstream.get("upstream_defect") is not False
    ):
        _fail("receipt_integrity", "claim_boundary_invalid")
    factual, candidate, factual_events, candidate_events = _require_replay_views(
        engine_root, engine
    )
    _verify_preservation_claims(factual, candidate)
    _verify_patch_claims(
        engine_root, factual, candidate, factual_events, candidate_events
    )


def _mapped_engine_failure(gaps: list[dict[str, str]]) -> NoReturn:
    for gap in gaps:
        gate = gap.get("gate")
        reason = str(gap.get("reason", ""))
        if gate == "renderer_capture" and "timestamp" in reason:
            _fail("renderer_capture", "timestamps_invalid")
        if gate == "runtime_attribution" and "events_missing" in reason:
            _fail("runtime_attribution", "contributor_not_observed")
        if gate == "runtime_attribution":
            _fail("runtime_attribution", "provenance_mismatch")
        if gate == "gameplay_preservation":
            _fail("gameplay_preservation", "invariant_mismatch")
        if gate == "patch_validation":
            _fail("patch_validation", "counterfactual_not_causal")
        if gate == "risk" and "factual" in reason:
            _fail("patch_search", "no_factual_hazard")
        if gate == "risk":
            _fail("patch_validation", "counterfactual_not_causal")
    _fail("receipt_integrity", "engine_evidence_invalid")


def _verify_preflight(root: Path, execution: dict[str, Any]) -> None:
    preflight_ref = execution.get("preflight")
    if not isinstance(preflight_ref, dict):
        _fail("receipt_integrity", "preflight_binding_invalid")
    path = _owned_file(root, preflight_ref.get("path"))
    if path is None:
        _fail("receipt_integrity", "preflight_binding_invalid")
    payload = _read_json_file(path)
    if payload is None or _sha256_file(path) != preflight_ref.get("sha256"):
        _fail("receipt_integrity", "preflight_binding_invalid")
    checkpoint_path = _owned_file(root, "preflight/preflight-checkpoint.json")
    checkpoint = (
        _read_json_file(checkpoint_path) if checkpoint_path is not None else None
    )
    pins = asdict(L6_PREFLIGHT_PINS)
    pins["required_inputs"] = list(pins["required_inputs"])
    checks = payload.get("checks")
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("preflight_verdict") != "PASS"
        or payload.get("verdict") != "PASS"
        or payload.get("upstream_product_verdict") != "INCONCLUSIVE"
        or payload.get("controlled_mutation") is not False
        or payload.get("upstream_defect") is not False
        or payload.get("replay_allowed") is not True
        or payload.get("pins") != pins
        or payload.get("preflight_input_sha256") != execution.get("input_id")
        or preflight_ref.get("verdict") != "PASS"
        or checkpoint
        != {
            "schema": PREFLIGHT_CHECKPOINT_SCHEMA,
            "phase": "PREFLIGHT_SEALED",
            "preflight_verdict": "PASS",
            "preflight_receipt_sha256": _sha256_file(path),
        }
        or not isinstance(checks, dict)
        or set(checks) != _EXPECTED_CHECKS
        or any(
            not isinstance(value, dict) or value.get("status") != "PASS"
            for value in checks.values()
        )
    ):
        _fail("receipt_integrity", "preflight_binding_invalid")


def verify_case(case_root: Path | str) -> dict[str, Any]:
    """Re-open every receipt-owned artifact and verify one complete L6 run."""

    root = _require_case_root(case_root)
    receipt_path = _owned_file(root, "execution-receipt.json")
    checkpoint_path = _owned_file(root, "execution-checkpoint.json")
    if receipt_path is None or checkpoint_path is None:
        _fail("receipt_integrity", "execution_receipt_missing_or_invalid")
    execution = _read_json_file(receipt_path)
    checkpoint = _read_json_file(checkpoint_path)
    if execution is None or checkpoint is None:
        _fail("receipt_integrity", "execution_receipt_missing_or_invalid")

    receipt_sha256 = _sha256_file(receipt_path)
    if (
        receipt_path.read_bytes() != _canonical_json(execution)
        or checkpoint_path.read_bytes() != _canonical_json(checkpoint)
        or
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("verdict") != "PASS"
        or execution.get("phase") != "POSITIVE_REPEATS_VERIFIED"
        or execution.get("controlled_mutation") is not True
        or execution.get("upstream_defect") is not False
        or execution.get("upstream_product_verdict") != "INCONCLUSIVE"
        or execution.get("runs_requested") != L6_REPEAT_COUNT
        or execution.get("runs_completed") != L6_REPEAT_COUNT
        or execution.get("evidence_gaps") != []
        or not isinstance(execution.get("input_id"), str)
        or not execution.get("input_id")
        or not isinstance(execution.get("execution_id"), str)
        or not execution.get("execution_id")
        or checkpoint
        != {
            "schema": EXECUTION_CHECKPOINT_SCHEMA,
            "phase": "POSITIVE_REPEATS_VERIFIED",
            "verdict": "PASS",
            "execution_receipt_sha256": receipt_sha256,
        }
    ):
        _fail("receipt_integrity", "execution_receipt_missing_or_invalid")

    _verify_preflight(root, execution)
    attempts = execution.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != L6_REPEAT_COUNT:
        _fail("receipt_integrity", "repeat_evidence_invalid")

    summaries: list[dict[str, object]] = []
    for index, attempt in enumerate(attempts, start=1):
        expected_path = f"run-{index:03d}"
        if (
            not isinstance(attempt, dict)
            or attempt.get("run") != index
            or attempt.get("path") != expected_path
            or attempt.get("verdict") != "PASS"
            or attempt.get("evidence_gaps") != []
            or attempt.get("engine_receipt")
            != f"{expected_path}/engine-receipt.json"
        ):
            _fail("receipt_integrity", "repeat_evidence_invalid")
        engine_root = _owned_directory(root, expected_path)
        engine_path = _owned_file(root, attempt.get("engine_receipt"))
        if engine_root is None or engine_path is None:
            _fail("receipt_integrity", "engine_receipt_missing_or_invalid")
        engine = _read_json_file(engine_path)
        if engine is None or engine.get("verdict") != "PASS":
            _fail("receipt_integrity", "engine_receipt_missing_or_invalid")

        # The category gates intentionally run before the outer digest check.
        # A tamper that destroys one semantic proof therefore receives its
        # stable N1-N6 diagnostic rather than a less useful generic hash error.
        _verify_engine_contract_gates(engine_root, engine)
        receipt_claim = engine.get("receipt_sha256")
        unhashed = dict(engine)
        unhashed.pop("receipt_sha256", None)
        expected_receipt_claim = "sha256:" + _sha256_bytes(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if receipt_claim != expected_receipt_claim:
            _fail("receipt_integrity", "engine_self_hash_mismatch")
        if _sha256_file(engine_path) != attempt.get("engine_receipt_sha256"):
            _fail("receipt_integrity", "engine_receipt_hash_mismatch")
        _verify_owned_artifacts_and_inputs(engine_root, engine)
        gaps, summary = _validate_positive_engine_receipt(engine_root, engine_path)
        if gaps:
            _mapped_engine_failure(gaps)
        if summary != attempt.get("deterministic_evidence"):
            _fail("receipt_integrity", "attempt_summary_mismatch")
        summaries.append(summary)

    stable = [
        {key: value for key, value in summary.items() if key != "engine_receipt_sha256"}
        for summary in summaries
    ]
    if (
        any(summary != stable[0] for summary in stable[1:])
        or execution.get("deterministic_evidence") != stable[0]
    ):
        _fail("receipt_integrity", "repeat_evidence_invalid")

    return {
        "case_sha256": receipt_sha256,
        "controlled_mutation": True,
        "execution_id": execution["execution_id"],
        "input_id": execution["input_id"],
        "phase": "POSITIVE_REPEATS_VERIFIED",
        "schema": VERIFICATION_SCHEMA,
        "upstream_defect": False,
        "verdict": "PASS",
        "verified_runs": L6_REPEAT_COUNT,
    }


def _tree_sha256(root: Path) -> str:
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            _fail("adversarial_aggregate", "fixture_symlink_rejected")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
    if not records:
        _fail("adversarial_aggregate", "fixture_empty")
    return _sha256_bytes(_canonical_json(records))


def _adversarial_result(gate: str, reason: str) -> dict[str, str]:
    return {
        "gate": gate,
        "reason": reason,
        "schema": ADVERSARIAL_RESULT_SCHEMA,
        "verdict": "FAIL_CLOSED",
    }


def _require_canonical_manifest_path(path: Path | str) -> Path:
    supplied = Path(path)
    expected = (Path.cwd() / CANONICAL_ADVERSARIAL_MANIFEST).resolve()
    actual = (supplied if supplied.is_absolute() else Path.cwd() / supplied).resolve()
    if actual != expected:
        _fail("adversarial_aggregate", "noncanonical_manifest_path")
    return actual


def verify_aggregate(manifest_path: Path | str) -> None:
    """Re-run exactly the six canonical independent L6 negative fixtures."""

    path = _require_canonical_manifest_path(manifest_path)
    payload = _read_json_file(path)
    if payload is None:
        _fail("adversarial_aggregate", "manifest_missing_or_invalid")
    try:
        raw = path.read_bytes()
    except OSError:
        _fail("adversarial_aggregate", "manifest_missing_or_invalid")
    if raw != _canonical_json(payload):
        _fail("adversarial_aggregate", "manifest_not_canonical")
    cases = payload.get("cases")
    if (
        payload.get("schema") != ADVERSARIAL_MANIFEST_SCHEMA
        or list(payload) != ["cases", "schema", "source_positive_execution_receipt_sha256"]
        or not _is_hex64(payload.get("source_positive_execution_receipt_sha256"))
        or not isinstance(cases, list)
        or [case.get("id") if isinstance(case, dict) else None for case in cases]
        != list(_EXPECTED_ADVERSARIAL)
    ):
        _fail("adversarial_aggregate", "manifest_contract_invalid")

    parent = path.parent
    for record in cases:
        case_id = record["id"]
        gate, reason = _EXPECTED_ADVERSARIAL[case_id]
        diagnostic = f"FAIL_CLOSED:{gate}:{reason}\n"
        result = _adversarial_result(gate, reason)
        expected_record_keys = {
            "expected",
            "fixture",
            "id",
            "pass_receipt",
            "raw_input_sha256",
        }
        if set(record) != expected_record_keys:
            _fail("adversarial_aggregate", "manifest_contract_invalid")
        fixture = _owned_directory(parent, record.get("fixture"))
        pass_receipt = Path(record.get("pass_receipt", ""))
        pass_path = pass_receipt if pass_receipt.is_absolute() else parent / pass_receipt
        try:
            pass_path.resolve().relative_to(parent.resolve())
        except (OSError, ValueError):
            _fail("adversarial_aggregate", "pass_receipt_path_invalid")
        expected = record.get("expected")
        if (
            fixture is None
            or record.get("fixture") != f"cases/{case_id}"
            or record.get("pass_receipt")
            != f"cases/{case_id}/verification-pass.json"
            or not _is_hex64(record.get("raw_input_sha256"))
            or _tree_sha256(fixture) != record.get("raw_input_sha256")
            or pass_path.exists()
            or expected
            != {
                "exit": 2,
                "result_json": result,
                "result_json_sha256": _sha256_bytes(_canonical_json(result)),
                "stderr_sha256": _sha256_bytes(diagnostic.encode("utf-8")),
                "stdout_sha256": _sha256_bytes(b""),
            }
        ):
            _fail("adversarial_aggregate", "case_record_invalid")
        try:
            verify_case(fixture)
        except VerificationFailure as exc:
            if exc.gate != gate or exc.reason != reason:
                _fail("adversarial_aggregate", "unexpected_case_diagnostic")
        else:
            _fail("adversarial_aggregate", "negative_case_passed")
        if pass_path.exists():
            _fail("adversarial_aggregate", "pass_receipt_created")


def materialize_adversarial_manifest(
    positive_case: Path | str,
    manifest_path: Path | str = CANONICAL_ADVERSARIAL_MANIFEST,
) -> Path:
    """Create six compact, immutable, run-local N1-N6 fixtures.

    The generated tree lives under ignored ``artifacts/``. It copies only the
    receipt documents and runtime-event JSON needed to reach each semantic
    gate; renderer/state binaries remain owned by the already verified positive
    case and are never accepted as external fixture evidence.
    """

    positive_root = _require_case_root(positive_case)
    positive_result = verify_case(positive_root)
    destination = _require_canonical_manifest_path(manifest_path)
    if destination.exists() or destination.parent.exists():
        raise FileExistsError(destination.parent)
    cases_parent = destination.parent / "cases"
    cases_parent.mkdir(parents=True)

    if _read_json_file(positive_root / "execution-receipt.json") is None:
        _fail("adversarial_aggregate", "positive_receipt_invalid")
    source_preflight = (positive_root / "preflight" / "preflight.json").read_bytes()
    source_preflight_checkpoint = (
        positive_root / "preflight" / "preflight-checkpoint.json"
    ).read_bytes()
    source_checkpoint = (positive_root / "execution-checkpoint.json").read_bytes()

    records: list[dict[str, Any]] = []
    for case_id, (gate, reason) in _EXPECTED_ADVERSARIAL.items():
        case_root = cases_parent / case_id
        case_root.mkdir()
        (case_root / "preflight").mkdir()
        (case_root / "preflight" / "preflight.json").write_bytes(source_preflight)
        (case_root / "preflight" / "preflight-checkpoint.json").write_bytes(
            source_preflight_checkpoint
        )
        (case_root / "execution-receipt.json").write_bytes(
            (positive_root / "execution-receipt.json").read_bytes()
        )
        (case_root / "execution-checkpoint.json").write_bytes(source_checkpoint)

        for index in range(1, L6_REPEAT_COUNT + 1):
            source_run = positive_root / f"run-{index:03d}"
            case_run = case_root / f"run-{index:03d}"
            case_run.mkdir()
            engine = _read_json_file(source_run / "engine-receipt.json")
            if engine is None:
                _fail("adversarial_aggregate", "positive_engine_receipt_invalid")
            engine = copy.deepcopy(engine)
            factual = engine["factual_replay"]
            candidate = engine["candidates"][0]
            for label, replay in (("factual", factual), ("candidate", candidate)):
                source_replay_path = _owned_file(source_run, replay.get("artifact"))
                if source_replay_path is None:
                    _fail("adversarial_aggregate", "positive_runtime_artifact_invalid")
                local_name = f"{label}-replay.json"
                (case_run / local_name).write_bytes(source_replay_path.read_bytes())
                replay["artifact"] = local_name
            engine["attribution"] = copy.deepcopy(candidate)

            if index == 1:
                if case_id == "N1":
                    factual["renderer_capture"]["actual_capture_timestamps_us"] = []
                elif case_id in {"N2", "N3"}:
                    replay_path = case_run / "factual-replay.json"
                    replay = _read_json_file(replay_path)
                    if replay is None:
                        _fail("adversarial_aggregate", "fixture_build_failed")
                    if case_id == "N2":
                        replay["runtime_events"][0]["script_path"] = "res://wrong.gd"
                    else:
                        replay["runtime_events"] = []
                    replay_path.write_bytes(_canonical_json(replay))
                elif case_id == "N4":
                    candidate["action_acknowledgements"] = [
                        {"frame": 0, "status": "MISSING"}
                    ]
                    engine["attribution"] = copy.deepcopy(candidate)
                elif case_id == "N5":
                    factual["hazardous"] = False
                    factual["max_risk"] = 0.0
                    factual["hazard_frame_indices"] = []
                    factual["hazard_kinds"] = []
                elif case_id == "N6":
                    candidate["replacement"] = 1.0
                    engine["attribution"] = copy.deepcopy(candidate)
            (case_run / "engine-receipt.json").write_bytes(_canonical_json(engine))

        diagnostic = f"FAIL_CLOSED:{gate}:{reason}\n"
        result = _adversarial_result(gate, reason)
        records.append(
            {
                "expected": {
                    "exit": 2,
                    "result_json": result,
                    "result_json_sha256": _sha256_bytes(_canonical_json(result)),
                    "stderr_sha256": _sha256_bytes(diagnostic.encode("utf-8")),
                    "stdout_sha256": _sha256_bytes(b""),
                },
                "fixture": f"cases/{case_id}",
                "id": case_id,
                "pass_receipt": f"cases/{case_id}/verification-pass.json",
                "raw_input_sha256": _tree_sha256(case_root),
            }
        )

    manifest = {
        "cases": records,
        "schema": ADVERSARIAL_MANIFEST_SCHEMA,
        "source_positive_execution_receipt_sha256": positive_result["case_sha256"],
    }
    destination.write_bytes(_canonical_json(manifest))
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m flashpatch.l6_verify")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", type=Path)
    mode.add_argument("--aggregate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.case is not None:
            result = verify_case(arguments.case)
            sys.stdout.buffer.write(_canonical_json(result))
        else:
            verify_aggregate(arguments.aggregate)
            sys.stdout.write("PASS: verified 6 independent L6 adversarial cases\n")
    except VerificationFailure as exc:
        sys.stderr.write(exc.diagnostic + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
