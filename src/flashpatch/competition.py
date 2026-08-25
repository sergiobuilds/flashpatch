"""Fail-closed validators for the FlashPatch competition evidence contract."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .external_league import (
    DIRECT_DETECTOR_POPULATION,
    EA_IRIS_RELEASE_ORACLE_ID,
    EA_IRIS_SOURCE_ADAPTER_ID,
    KAYA_PROTOTYPE_ID,
)
from .l8_league import (
    L8LeagueError,
    aggregate_artifact_league,
    prepare_artifact_league,
    reveal_artifact_league,
)
from .renderer_artifact import RendererArtifactError, open_renderer_artifact, renderer_rgb_sha256


SHA256 = re.compile(r"^[0-9a-f]{64}$")
DECISIONS = {"PASS", "SAFE", "FAIL", "INCONCLUSIVE"}
REQUIRED_RECEIPT_HASHES = (
    "input_sha256",
    "trace_sha256",
    "factual_frame_sha256",
    "candidate_frame_sha256",
    "source_snapshot_sha256",
    "diff_sha256",
)
INDEPENDENT_GOLD_SCHEMA = "flashpatch-l7-independent-gold-v2"
INDEPENDENT_GOLD_TRUST_SCHEMA = "flashpatch-l7-gold-trust-policy-v1"
INDEPENDENT_GOLD_NOT_VERIFIED = "NOT_SCOREABLE reason=independent_gold_not_verified"
NATIVE_MAIN_BLINDED_GOLD_INTAKE_SCHEMA = "flashpatch-l7-native-main-blinded-gold-intake-v1"
NATIVE_MAIN_BLINDED_GOLD_INTAKE_ASSESSMENT_SCHEMA = "flashpatch-l7-native-main-blinded-gold-intake-assessment-v1"
NATIVE_MAIN_INDEPENDENT_GOLD_RETURN_SCHEMA = "flashpatch-l7-native-main-independent-gold-return-v1"
NATIVE_MAIN_INDEPENDENT_GOLD_RETURN_ASSESSMENT_SCHEMA = "flashpatch-l7-native-main-independent-gold-return-assessment-v1"
NATIVE_MAIN_CANDIDATE_START_GATE_SCHEMA = "flashpatch-l7-native-main-candidate-start-gate-v1"
NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA = "flashpatch-l7-native-main-candidate-start-gate-assessment-v1"
NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_SCHEMA = "flashpatch-l7-native-main-adjudicator-packet-manifest-v1"
NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_ASSESSMENT_SCHEMA = (
    "flashpatch-l7-native-main-adjudicator-packet-manifest-assessment-v1"
)
L7_EXTERNAL_HOST_REQUEST_SCHEMA_V2 = "flashpatch-l7-external-host-witness-request-v2"
OPENSSL_EXECUTABLE = Path("/usr/bin/openssl")


@dataclass(frozen=True)
class ApprovedGoldTrustPolicy:
    policy_id: str
    roster_id: str
    policy_sha256: str
    approval_record_sha256: str
    external_timestamp_token_sha256: str
    verifier_source_commit: str
    approved_at: str
    valid_from: str
    valid_until: str | None
    revoked_at: str | None
    predecessor_policy_sha256: str | None
    normative_rule_sha256: str
    openssl_binary_sha256: str
    approval_record_path: str
    approval_authority_id: str
    approval_authority_public_key_path: str
    approval_authority_public_key_sha256: str
    approval_authority_spki_sha256: str
    external_timestamp_token_path: str
    timestamp_authority_id: str
    timestamp_authority_public_key_path: str
    timestamp_authority_public_key_sha256: str
    timestamp_authority_spki_sha256: str
    verifier_build_manifest_path: str
    verifier_build_manifest_sha256: str
    predecessor_approval_record_sha256: str | None


# This deployment has no externally approved adjudicator authority yet. Entries
# may be added only by a separately reviewed source change made before candidate
# execution; evidence submitters cannot supply or override this registry.
APPROVED_GOLD_TRUST_POLICIES: tuple[ApprovedGoldTrustPolicy, ...] = (
    ApprovedGoldTrustPolicy(
        policy_id="ea-iris-native-main-v7-gold-policy-20260804",
        roster_id="ea-iris-native-main-v7-gold-roster-20260804",
        policy_sha256="220958b8a70a06fd132770e46158fa58a8c793717ebd96251a520c38cfd39eb0",
        approval_record_sha256="a97e137aeac77432873375934b998940ec1782516b2ff6be6a74ee53186a424d",
        external_timestamp_token_sha256="cda4bf88244d692e8430b4a8c541d25ecad6ddd962546fdd489293933d886c95",
        verifier_source_commit="cc64ad92394d269e9242c2e22e63655deab7aab8",
        approved_at="2026-08-04T04:39:30Z",
        valid_from="2026-08-04T04:39:31Z",
        valid_until=None,
        revoked_at=None,
        predecessor_policy_sha256=None,
        normative_rule_sha256="679a6c315b1ae638ea82e248d5c77e90d4dfdb897c0990561075bc68b14d78c7",
        openssl_binary_sha256="b86b739329008369aebe1f7cff6c2adb18965609d68a19456fca55232f2908f5",
        approval_record_path="ea-iris-native-main-v7/approval-record.json",
        approval_authority_id="ea-iris-policy-approval-authority",
        approval_authority_public_key_path="ea-iris-native-main-v7/keys/ea-iris-policy-approval-authority.public.pem",
        approval_authority_public_key_sha256="7609cafb589007ecd049a1938b360593d18f92bbddad29d48a5f826e99762b79",
        approval_authority_spki_sha256="77c2e75fa89b8adbd1f810b12da33a9e42f1513f35dfb1bb4cf84dc0e1f5811f",
        external_timestamp_token_path="ea-iris-native-main-v7/timestamp-token.json",
        timestamp_authority_id="ea-iris-policy-timestamp-authority",
        timestamp_authority_public_key_path="ea-iris-native-main-v7/keys/ea-iris-policy-timestamp-authority.public.pem",
        timestamp_authority_public_key_sha256="4811055d2a25c00d25a680ada67f9eec7b0a2f4c1ed3c6487e5c80213c6ce4a0",
        timestamp_authority_spki_sha256="94bbe996525b51a07fc990f4d03342a7f620d129b9b3eddcf61c95f8c576ea65",
        verifier_build_manifest_path="ea-iris-native-main-v7/verifier-build.json",
        verifier_build_manifest_sha256="1b9eba7c8b24192e920cc3a416db97733a13aaf6b745b9141c9e945465d48ba1",
        predecessor_approval_record_sha256=None,
    ),
)
GOLD_APPROVAL_ARTIFACT_ROOT = Path(__file__).resolve().parent / "_gold_approvals"
GOLD_CASE_CLASSES = {"natural_external", "controlled_external"}
GOLD_CLAIM_TIERS = {"L9_ELIGIBLE", "CONTROLLED_ONLY"}
GOLD_ARTIFACT_FIELDS = (
    "source_tree_sha256",
    "trace_sha256",
    "renderer_execution_receipt_sha256",
    "renderer_rgb_raw_sha256",
    "timestamps_sha256",
    "rubric_sha256",
    "corpus_manifest_sha256",
    "blinded_gold_input_sha256",
    "candidate_start_receipt_sha256",
)
L7_DIRECT_CANDIDATE_TOOLS = list(DIRECT_DETECTOR_POPULATION)
GOLD_EXCLUDED_TOOLS = {
    *L7_DIRECT_CANDIDATE_TOOLS,
    EA_IRIS_RELEASE_ORACLE_ID,
    EA_IRIS_SOURCE_ADAPTER_ID,
    KAYA_PROTOTYPE_ID,
    "FFmpeg",
}
BLINDED_INTAKE_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:candidate_id|candidate_predictions|candidate_results|detector_output|"
    r"detector_outputs|prediction|predictions|decision|hazard|risk|score|winner|"
    r"ranking|repository_url|repo_url|repository_name|source_repository|candidate_repository|"
    r"candidate_name|flashpatch|kaya|tooflashy|iris|ffmpeg)(?:_|$)",
    re.IGNORECASE,
)
NATIVE_MAIN_GOLD_RETURN_FORBIDDEN_KEYS = {
    "candidate_output",
    "candidate_outputs",
    "candidate_prediction",
    "candidate_predictions",
    "candidate_result",
    "candidate_results",
    "detector_decision",
    "detector_output",
    "detector_outputs",
    "repository_identity_seen",
    "source_repository_seen",
}
CHECKPOINT_LEAVES = frozenset({"P0", "V1", "V2", "V3", "V4", "L6", "L7", "L8", "L9"})
CHECKPOINT_SCHEMA = "flashpatch-competition-checkpoint-v1"


class ContractError(ValueError):
    """An evidence artifact did not meet the fail-closed contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"invalid JSON: {path}") from exc
    return _load_json_bytes(value, str(path))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ContractError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json_bytes(value: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON root must be an object: {context}")
    return payload


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(root: Path, leaf: str) -> Path:
    if leaf not in CHECKPOINT_LEAVES:
        raise ContractError(f"unsupported checkpoint leaf: {leaf}")
    return root / ("preflight.json" if leaf == "P0" else f"{leaf}.json")


def _checkpoint_hashes(
    immutable_input: object, command: object, environment: object, receipt: object
) -> dict[str, str]:
    """Hash every frozen checkpoint input before it can be reused."""
    return {
        "input_sha256": _sha256(immutable_input),
        "command_sha256": _sha256(command),
        "environment_sha256": _sha256(environment),
        "receipt_sha256": _sha256(receipt),
    }


def write_checkpoint(
    root: Path,
    leaf: str,
    *,
    immutable_input: object,
    command: object,
    environment: object,
    receipt: object,
    status: str = "COMPLETED",
    exclusion_reason: str | None = None,
) -> Path:
    """Atomically publish a checkpoint, including an interrupted failure receipt.

    Only a completed checkpoint is eligible for reuse.  An interruption therefore
    remains inspectable on disk but can never be mistaken for a successful run.
    """
    if status not in {"COMPLETED", "INCONCLUSIVE"}:
        raise ContractError("invalid checkpoint status")
    if status == "INCONCLUSIVE" and not exclusion_reason:
        raise ContractError("INCONCLUSIVE checkpoint requires exclusion_reason")
    destination = _checkpoint_path(root, leaf)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "leaf": leaf,
        "status": status,
        "exclusion_reason": exclusion_reason,
        **_checkpoint_hashes(immutable_input, command, environment, receipt),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    return destination


def checkpoint_is_reusable(
    root: Path,
    leaf: str,
    *,
    immutable_input: object,
    command: object,
    environment: object,
    receipt: object,
) -> bool:
    """Return true only when the full immutable checkpoint identity matches."""
    path = _checkpoint_path(root, leaf)
    try:
        checkpoint = _load_json(path)
    except ContractError:
        return False
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("leaf") != leaf:
        return False
    if checkpoint.get("status") != "COMPLETED" or checkpoint.get("exclusion_reason") is not None:
        return False
    return all(checkpoint.get(key) == value for key, value in _checkpoint_hashes(
        immutable_input, command, environment, receipt
    ).items())


def _plan_revision(path: Path) -> str:
    """Get the committed revision that owns the plan, never executing its text."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=Path.cwd(), capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise ContractError("unable to inspect plan revision") from exc
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        raise ContractError("unable to inspect plan revision")
    return revision


def preflight(plan: Path, expected_commit: str, checkpoints: Path) -> str:
    """Freeze the committed execution plan in the P0 checkpoint."""
    revision = _plan_revision(plan)
    command = {"plan": str(plan), "expected_commit": expected_commit}
    environment = {"python": sys.version, "platform": sys.platform}
    if not revision.startswith(expected_commit):
        receipt = {"status": "INCONCLUSIVE", "exclusion_reason": "stale_plan", "revision": revision}
        write_checkpoint(
            checkpoints, "P0", immutable_input=plan.read_text(encoding="utf-8"), command=command,
            environment=environment, receipt=receipt, status="INCONCLUSIVE", exclusion_reason="stale_plan",
        )
        raise ContractError("exclusion_reason=stale_plan")
    receipt = {"status": "PASS", "revision": revision}
    write_checkpoint(
        checkpoints, "P0", immutable_input=plan.read_text(encoding="utf-8"), command=command,
        environment=environment, receipt=receipt,
    )
    return "VALID preflight"


def validate_plan(path: Path) -> str:
    """Verify the global freshness rule without executing untrusted plan text."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read plan: {path}") from exc
    if not re.search(r"^\s*(runtime|global_runtime):\s*$", text, re.MULTILINE):
        raise ContractError("missing global_runtime.freshness_rule")
    runtime_match = re.search(
        r"^\s*(?:runtime|global_runtime):\s*$([\s\S]*?)(?=^\S|\Z)", text, re.MULTILINE
    )
    if runtime_match is None or not re.search(r"^\s+freshness_rule:\s*\S", runtime_match.group(1), re.MULTILINE):
        raise ContractError("missing global_runtime.freshness_rule")
    schema = re.search(r"^schema_version:\s*([^\s]+)", text, re.MULTILINE)
    if schema is None:
        raise ContractError("missing schema_version")
    return f"VALID plan={schema.group(1)}"


def validate_case(path: Path) -> str:
    case = _load_json(path)
    case_id = case.get("case_id")
    decision = case.get("decision")
    if not isinstance(case_id, str) or not case_id:
        raise ContractError("missing case_id")
    if decision not in DECISIONS:
        raise ContractError("invalid decision")
    patch = case.get("patch")
    if decision == "SAFE" and patch is not None:
        raise ContractError("SAFE requires patch=null")
    if decision == "SAFE" and case.get("failure_reason") is not None:
        raise ContractError("SAFE requires failure_reason=null")
    if decision == "PASS":
        for field in ("source_file", "source_line", "exported_parameter", "factual_value", "candidate_value", "diff_sha256"):
            if field not in case:
                raise ContractError(f"PASS missing {field}")
    if decision in {"FAIL", "INCONCLUSIVE"} and not isinstance(case.get("failure_reason"), str):
        raise ContractError(f"{decision} requires failure_reason")
    return f"VALID case_id={case_id}"


def validate_provenance(path: Path) -> str:
    manifest = _load_json(path)
    source = manifest.get("source")
    observed = manifest.get("observed")
    if not isinstance(source, dict) or not isinstance(observed, dict):
        raise ContractError("missing source or observed provenance")
    if source.get("source_revision") != observed.get("source_revision"):
        raise ContractError("source revision mismatch")
    if source.get("repository_url") != observed.get("repository_url"):
        raise ContractError("repository URL mismatch")
    if not isinstance(source.get("license"), str) or not source["license"]:
        raise ContractError("missing source license")
    return "VALID provenance"


def verify_receipt(path: Path) -> str:
    receipt = _load_json(path)
    decision = receipt.get("decision")
    if decision not in DECISIONS:
        raise ContractError("invalid receipt decision")
    if decision == "SAFE":
        if receipt.get("patch") is not None:
            raise ContractError("SAFE requires patch=null")
        if receipt.get("failure_reason") is not None:
            raise ContractError("SAFE requires failure_reason=null")
    if decision == "PASS":
        if not isinstance(receipt.get("patch"), dict):
            raise ContractError("PASS requires patch object")
        for field in ("source_file", "source_line", "exported_parameter", "factual_value", "candidate_value", "diff_sha256"):
            if field not in receipt:
                raise ContractError(f"PASS missing {field}")
    if decision in {"FAIL", "INCONCLUSIVE"} and not isinstance(receipt.get("failure_reason"), str):
        raise ContractError(f"{decision} requires failure_reason")
    for field in REQUIRED_RECEIPT_HASHES:
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            if field in {"factual_frame_sha256", "candidate_frame_sha256"}:
                raise ContractError("frame hash mismatch")
            raise ContractError(f"invalid {field}")
    _verify_receipt_artifacts(receipt, path)
    return f"VALID receipt decision={decision}"


def _required_object(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ContractError(f"missing {field}")
    return value


def _validate_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"invalid {field}")
    return value


def _verify_receipt_artifacts(receipt: dict[str, Any], receipt_path: Path) -> None:
    """Recompute declared artifact hashes from receipt-local files.

    A correctly shaped SHA-256 string is not evidence.  Artifact paths are
    relative to the receipt directory and cannot resolve outside it.
    """
    artifacts = receipt.get("artifacts")
    if artifacts is None:
        return
    if not isinstance(artifacts, dict):
        raise ContractError("invalid artifacts")
    root = receipt_path.parent.resolve()
    for hash_field, relative_path in artifacts.items():
        if hash_field not in REQUIRED_RECEIPT_HASHES:
            raise ContractError("unknown artifact hash field")
        if not isinstance(relative_path, str) or not relative_path:
            raise ContractError("invalid artifact path")
        candidate = receipt_path.parent / relative_path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ContractError("artifact unavailable") from exc
        if candidate.is_symlink() or root not in resolved.parents:
            raise ContractError("artifact path escapes receipt root")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if receipt.get(hash_field) != actual:
            message = "frame hash mismatch" if "frame" in hash_field else f"{hash_field} mismatch"
            raise ContractError(message)


def _verify_hash_bound_artifacts(
    payload: dict[str, Any],
    artifact_paths: dict[str, Any],
    root_path: Path,
    *,
    fields: tuple[str, ...],
) -> dict[str, bytes]:
    """Recompute a named hash set from relative, non-symlinked files only."""
    root = root_path.parent.resolve()
    verified: dict[str, bytes] = {}
    for field in fields:
        expected = _validate_hash(payload.get(field), field)
        relative_path = artifact_paths.get(field)
        if not isinstance(relative_path, str) or not relative_path:
            raise ContractError(f"missing gold artifact path: {field}")
        candidate = root_path.parent / relative_path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"gold artifact unavailable: {field}") from exc
        if candidate.is_symlink() or root not in resolved.parents:
            raise ContractError(f"gold artifact path escapes receipt root: {field}")
        try:
            artifact_bytes = resolved.read_bytes()
        except OSError as exc:
            raise ContractError(f"gold artifact unavailable: {field}") from exc
        if hashlib.sha256(artifact_bytes).hexdigest() != expected:
            raise ContractError(f"gold artifact hash mismatch: {field}")
        verified[field] = artifact_bytes
    return verified


def _validate_natural_renderer_execution(
    execution_bytes: bytes,
    freeze: dict[str, Any],
) -> None:
    """Bind a natural gold label to an actual unmutated engine receipt."""
    execution = _load_json_bytes(execution_bytes, "natural renderer execution receipt")
    if execution.get("schema") != "flashpatch-renderer-engine-receipt-v1":
        raise ContractError("natural gold renderer execution receipt schema is invalid")
    if execution.get("controlled_mutation") is not False:
        raise ContractError("natural gold renderer execution receipt is controlled or undeclared")
    upstream = _required_object(execution, "upstream")
    expected = {
        "repository_url": freeze["public_repository_url"],
        "source_revision": freeze["source_revision"],
        "license": freeze["license"],
        "project_path": freeze["project_subpath"],
    }
    if any(upstream.get(field) != value for field, value in expected.items()):
        raise ContractError("natural gold renderer execution provenance mismatch")
    factual = _required_object(execution, "factual_replay")
    if factual.get("renderer_rgb_raw_sha256") != freeze["renderer_rgb_raw_sha256"]:
        raise ContractError("natural gold renderer RGB hash does not match execution receipt")
    if factual.get("timestamps_sha256") != freeze["timestamps_sha256"]:
        raise ContractError("natural gold renderer timestamps do not match execution receipt")
    if factual.get("frame_count") != freeze["frame_count"]:
        raise ContractError("natural gold renderer frame count does not match execution receipt")
    capture = _required_object(factual, "renderer_capture")
    trace_hash = capture.get("trace_sha256")
    if trace_hash != f"sha256:{freeze['trace_sha256']}":
        raise ContractError("natural gold renderer trace does not match execution receipt")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _approved_registry_snapshot_sha256() -> str:
    rows = sorted((asdict(record) for record in APPROVED_GOLD_TRUST_POLICIES), key=lambda row: row["policy_id"])
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _read_build_owned_artifact(relative_path: str, expected_sha256: str, context: str) -> bytes:
    expected = _validate_hash(expected_sha256, context)
    if not isinstance(relative_path, str) or not relative_path:
        raise ContractError(f"{context} path is invalid")
    try:
        root = GOLD_APPROVAL_ARTIFACT_ROOT.resolve(strict=True)
        candidate = GOLD_APPROVAL_ARTIFACT_ROOT / relative_path
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{context} is unavailable") from exc
    if candidate.is_symlink() or root not in resolved.parents or not resolved.is_file():
        raise ContractError(f"{context} escapes build-owned approval root")
    try:
        value = resolved.read_bytes()
    except OSError as exc:
        raise ContractError(f"{context} is unavailable") from exc
    if hashlib.sha256(value).hexdigest() != expected:
        raise ContractError(f"{context} hash mismatch")
    return value


def _read_approval_authority_key(
    *,
    path: str,
    file_sha256: str,
    spki_sha256: str,
    context: str,
) -> bytes:
    key_bytes = _read_build_owned_artifact(path, file_sha256, context)
    expected_spki = _validate_hash(spki_sha256, f"{context} SPKI")
    if hashlib.sha256(_canonical_ed25519_spki(key_bytes)).hexdigest() != expected_spki:
        raise ContractError(f"{context} SPKI hash mismatch")
    return key_bytes


def _approval_verifier_source_sha256() -> str:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    normalized = re.sub(
        r"APPROVED_GOLD_TRUST_POLICIES: tuple\[ApprovedGoldTrustPolicy, \.\.\.\] = \([\s\S]*?\)\n"
        r"GOLD_APPROVAL_ARTIFACT_ROOT = ",
        "APPROVED_GOLD_TRUST_POLICIES: tuple[ApprovedGoldTrustPolicy, ...] = (<registry>)\n"
        "GOLD_APPROVAL_ARTIFACT_ROOT = ",
        source,
        count=1,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_approved_policy_record(
    record: ApprovedGoldTrustPolicy,
    *,
    visited: frozenset[str] = frozenset(),
) -> tuple[datetime, datetime, datetime | None, dict[str, Any]]:
    _identity(record.policy_id, "approved trust policy")
    _identity(record.roster_id, "approved adjudicator roster")
    approval_authority_id = _identity(record.approval_authority_id, "approval authority")
    timestamp_authority_id = _identity(record.timestamp_authority_id, "timestamp authority")
    if approval_authority_id == timestamp_authority_id:
        raise ContractError("approval and timestamp authorities are not independent")
    if record.approval_authority_spki_sha256 == record.timestamp_authority_spki_sha256:
        raise ContractError("approval and timestamp authorities reuse one cryptographic identity")
    if record.policy_id in visited:
        raise ContractError("approved trust policy predecessor cycle detected")
    for field in (
        "policy_sha256",
        "approval_record_sha256",
        "external_timestamp_token_sha256",
        "normative_rule_sha256",
        "openssl_binary_sha256",
        "approval_authority_public_key_sha256",
        "approval_authority_spki_sha256",
        "timestamp_authority_public_key_sha256",
        "timestamp_authority_spki_sha256",
        "verifier_build_manifest_sha256",
    ):
        _validate_hash(getattr(record, field), f"approved trust policy {field}")
    if re.fullmatch(r"[0-9a-f]{40}", record.verifier_source_commit) is None:
        raise ContractError("approved trust policy verifier commit is invalid")
    try:
        openssl_bytes = OPENSSL_EXECUTABLE.read_bytes()
    except OSError as exc:
        raise ContractError("approved Ed25519 verifier is unavailable") from exc
    if hashlib.sha256(openssl_bytes).hexdigest() != record.openssl_binary_sha256:
        raise ContractError("approved Ed25519 verifier binary hash mismatch")
    if record.predecessor_policy_sha256 is not None:
        _validate_hash(record.predecessor_policy_sha256, "approved trust policy predecessor")
    if (record.predecessor_policy_sha256 is None) != (record.predecessor_approval_record_sha256 is None):
        raise ContractError("approved trust policy predecessor binding is incomplete")
    if record.predecessor_approval_record_sha256 is not None:
        _validate_hash(record.predecessor_approval_record_sha256, "approved predecessor record")
    approved_at = _parse_utc_timestamp(record.approved_at, "approved trust policy")
    valid_from = _parse_utc_timestamp(record.valid_from, "approved trust policy validity")
    valid_until = (
        _parse_utc_timestamp(record.valid_until, "approved trust policy validity")
        if record.valid_until is not None
        else None
    )
    if approved_at > valid_from or (valid_until is not None and valid_until <= valid_from):
        raise ContractError("approved trust policy validity window is invalid")
    if record.revoked_at is not None:
        raise ContractError("approved trust policy is revoked")

    build_bytes = _read_build_owned_artifact(
        record.verifier_build_manifest_path,
        record.verifier_build_manifest_sha256,
        "verifier build manifest",
    )
    build = _load_json_bytes(build_bytes, "verifier build manifest")
    _require_exact_keys(
        build,
        {"schema", "verifier_source_commit", "competition_py_sha256", "built_at"},
        "verifier build manifest",
    )
    if (
        build.get("schema") != "flashpatch-l7-gold-verifier-build-v1"
        or build.get("verifier_source_commit") != record.verifier_source_commit
        or build.get("competition_py_sha256") != _approval_verifier_source_sha256()
    ):
        raise ContractError("verifier build identity does not match executing source")
    built_at = _parse_utc_timestamp(build.get("built_at"), "verifier build")
    if built_at > approved_at:
        raise ContractError("trust policy approval predates verifier build")

    approval_key = _read_approval_authority_key(
        path=record.approval_authority_public_key_path,
        file_sha256=record.approval_authority_public_key_sha256,
        spki_sha256=record.approval_authority_spki_sha256,
        context="approval authority public key",
    )
    approval_bytes = _read_build_owned_artifact(
        record.approval_record_path,
        record.approval_record_sha256,
        "external approval record",
    )
    approval = _load_json_bytes(approval_bytes, "external approval record")
    _require_exact_keys(
        approval,
        {
            "schema",
            "policy_id",
            "roster_id",
            "policy_sha256",
            "approved_at",
            "valid_from",
            "valid_until",
            "revoked_at",
            "predecessor_policy_sha256",
            "predecessor_approval_record_sha256",
            "normative_rule_sha256",
            "verifier_source_commit",
            "verifier_build_manifest_sha256",
            "approval_authority_id",
            "signature_algorithm",
            "signature_base64",
        },
        "external approval record",
    )
    mirrored = {
        "policy_id": record.policy_id,
        "roster_id": record.roster_id,
        "policy_sha256": record.policy_sha256,
        "approved_at": record.approved_at,
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
        "revoked_at": record.revoked_at,
        "predecessor_policy_sha256": record.predecessor_policy_sha256,
        "predecessor_approval_record_sha256": record.predecessor_approval_record_sha256,
        "normative_rule_sha256": record.normative_rule_sha256,
        "verifier_source_commit": record.verifier_source_commit,
        "verifier_build_manifest_sha256": record.verifier_build_manifest_sha256,
        "approval_authority_id": record.approval_authority_id,
    }
    if approval.get("schema") != "flashpatch-l7-external-policy-approval-v1" or any(
        approval.get(field) != value for field, value in mirrored.items()
    ):
        raise ContractError("external approval record does not match registry")
    if approval.get("signature_algorithm") != "Ed25519":
        raise ContractError("external approval signature algorithm is invalid")
    approval_payload = {
        key: value
        for key, value in approval.items()
        if key not in {"signature_algorithm", "signature_base64"}
    }
    _verify_ed25519_signature(
        approval_key,
        {"domain": "FlashPatch/L7/trust-policy-approval/v1", "payload": approval_payload},
        approval.get("signature_base64"),
    )

    timestamp_key = _read_approval_authority_key(
        path=record.timestamp_authority_public_key_path,
        file_sha256=record.timestamp_authority_public_key_sha256,
        spki_sha256=record.timestamp_authority_spki_sha256,
        context="timestamp authority public key",
    )
    token_bytes = _read_build_owned_artifact(
        record.external_timestamp_token_path,
        record.external_timestamp_token_sha256,
        "external approval timestamp token",
    )
    token = _load_json_bytes(token_bytes, "external approval timestamp token")
    _require_exact_keys(
        token,
        {
            "schema",
            "timestamp_authority_id",
            "subject_policy_sha256",
            "subject_approval_record_sha256",
            "approved_at",
            "token_serial",
            "signature_algorithm",
            "signature_base64",
        },
        "external approval timestamp token",
    )
    if (
        token.get("schema") != "flashpatch-l7-external-approval-timestamp-v1"
        or token.get("timestamp_authority_id") != record.timestamp_authority_id
        or token.get("subject_policy_sha256") != record.policy_sha256
        or token.get("subject_approval_record_sha256") != record.approval_record_sha256
        or token.get("approved_at") != record.approved_at
        or not isinstance(token.get("token_serial"), str)
        or re.fullmatch(r"tsa-[0-9a-f]{32}", token["token_serial"]) is None
        or token.get("signature_algorithm") != "Ed25519"
    ):
        raise ContractError("external approval timestamp subject or time is invalid")
    token_payload = {
        key: value
        for key, value in token.items()
        if key not in {"signature_algorithm", "signature_base64"}
    }
    _verify_ed25519_signature(
        timestamp_key,
        {"domain": "FlashPatch/L7/external-approval-timestamp/v1", "payload": token_payload},
        token.get("signature_base64"),
    )

    if record.predecessor_policy_sha256 is not None:
        predecessors = [
            predecessor
            for predecessor in APPROVED_GOLD_TRUST_POLICIES
            if predecessor.policy_sha256 == record.predecessor_policy_sha256
            and predecessor.approval_record_sha256 == record.predecessor_approval_record_sha256
        ]
        if len(predecessors) != 1:
            raise ContractError("approved trust policy predecessor is absent or ambiguous")
        predecessor_approved_at, _, _, _ = _validate_approved_policy_record(
            predecessors[0],
            visited=visited | {record.policy_id},
        )
        if predecessor_approved_at >= approved_at:
            raise ContractError("approved trust policy predecessor ordering is invalid")
    return approved_at, valid_from, valid_until, approval


def _require_exact_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        raise ContractError(f"{context} fields are not exact")


def _parse_utc_timestamp(value: object, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"{context} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{context} timestamp is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ContractError(f"{context} timestamp is not UTC")
    return parsed


def _validate_gold_intervals(
    value: object,
    decision: str,
    context: str,
    *,
    capture_start: float,
    capture_end: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{context} intervals are invalid")
    if decision == "SAFE":
        if value:
            raise ContractError("safe independent gold must not invent hazard intervals")
        return value
    if not value:
        raise ContractError("hazardous independent gold requires intervals")
    previous_end = -1.0
    for interval in value:
        if not isinstance(interval, dict):
            raise ContractError(f"{context} interval is invalid")
        _require_exact_keys(interval, {"kind", "start_seconds", "end_seconds"}, f"{context} interval")
        if interval.get("kind") not in {"flash", "red_flash", "regular_pattern"}:
            raise ContractError(f"{context} interval kind is invalid")
        start = interval.get("start_seconds")
        end = interval.get("end_seconds")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < capture_start
            or float(end) > capture_end
            or float(start) >= float(end)
            or float(start) < previous_end
        ):
            raise ContractError(f"{context} interval bounds are invalid")
        previous_end = float(end)
    return value


def _validate_gold_timestamps(value: bytes, frame_count: int) -> tuple[float, float]:
    receipt = _load_json_bytes(value, "gold timestamps")
    _require_exact_keys(receipt, {"schema", "timestamps_seconds"}, "gold timestamps")
    timestamps = receipt.get("timestamps_seconds")
    if (
        receipt.get("schema") != "flashpatch-l7-renderer-timestamps-v1"
        or not isinstance(timestamps, list)
        or len(timestamps) != frame_count
        or frame_count < 2
    ):
        raise ContractError("gold timestamps do not match frozen frame count")
    normalized: list[float] = []
    for value in timestamps:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError("gold timestamp is invalid")
        normalized.append(float(value))
    if normalized[0] < 0 or any(current <= previous for previous, current in zip(normalized, normalized[1:])):
        raise ContractError("gold timestamps are not strictly increasing")
    return normalized[0], normalized[-1]


def _find_project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "docs" / "MASTER-MAP.md").is_file() and (parent / "src" / "flashpatch").is_dir():
            return parent
    raise ContractError("project root is unavailable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_bound_file(project_root: Path, value: object, expected_sha256: object, context: str) -> Path:
    if not isinstance(value, str) or value.startswith("/") or "\x00" in value:
        raise ContractError(f"{context} path is invalid")
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ContractError(f"{context} path escapes project") from exc
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{context} is unavailable")
    if _sha256_file(path) != _validate_hash(expected_sha256, context):
        raise ContractError(f"{context} hash mismatch")
    return path


def _receipt_bound_file(receipt_root: Path, value: object, expected_sha256: object, context: str) -> Path:
    if not isinstance(value, str) or value.startswith("/") or "\x00" in value:
        raise ContractError(f"{context} path is invalid")
    path = (receipt_root / value).resolve()
    try:
        path.relative_to(receipt_root)
    except ValueError as exc:
        raise ContractError(f"{context} path escapes intake receipt") from exc
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{context} is unavailable")
    if _sha256_file(path) != _validate_hash(expected_sha256, context):
        raise ContractError(f"{context} hash mismatch")
    return path


def _reject_blinded_intake_leaks(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if BLINDED_INTAKE_FORBIDDEN_KEY.search(key):
                raise ContractError(f"blinded gold intake leaks candidate output: {context}:{key}")
            _reject_blinded_intake_leaks(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_blinded_intake_leaks(child, f"{context}[{index}]")
    elif isinstance(value, str):
        if context.endswith(".schema"):
            return
        lowered = value.lower()
        forbidden = (
            "flashpatch",
            "kaya",
            "tooflashy",
            "ea_iris",
            "ffmpeg",
            "safe_scenario_ready",
            "hazardous_attribution_pending",
        )
        if any(token in lowered for token in forbidden):
            raise ContractError(f"blinded gold intake leaks candidate output: {context}")


def _renderer_hash_projection(frame_artifact: Path) -> tuple[str, str, list[float], int]:
    try:
        with open_renderer_artifact(frame_artifact) as artifact:
            rgb_sha256 = renderer_rgb_sha256(artifact.frames)
            timestamps_sha256 = hashlib.sha256(artifact.timestamps.tobytes()).hexdigest()
            timestamps = [float(value) for value in artifact.timestamps.tolist()]
            frame_count = int(len(artifact.frames))
    except RendererArtifactError as exc:
        raise ContractError("renderer artifact cannot be projected for blinded gold intake") from exc
    return rgb_sha256, timestamps_sha256, timestamps, frame_count


def _blind_case_id(seed: bytes, used: set[str]) -> str:
    counter = 0
    while True:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).hexdigest()[:32]
        candidate = f"blind-{digest}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def prepare_native_main_blinded_gold_intake(
    summary_path: Path,
    output_dir: Path,
    *,
    frozen_at: str,
) -> Path:
    """Create a non-scoreable packet for future independent gold adjudication."""

    from .l7_verify import verify_native_main_qualification_summary

    summary_file = summary_path.resolve(strict=True)
    project_root = _find_project_root(summary_file)
    verify_native_main_qualification_summary(summary_file)
    frozen_timestamp = _parse_utc_timestamp(frozen_at, "blinded gold intake freeze")
    summary = _load_json(summary_file)
    rows = [row for row in summary.get("results", []) if isinstance(row, dict) and row.get("receipt_created") is True]
    if not rows:
        raise ContractError("native-main summary has no complete renderer receipts")
    destination = output_dir.resolve()
    if destination.exists():
        raise ContractError("blinded gold intake output already exists")
    packet = destination / "adjudicator-packet"
    frames_dir = packet / "frames"
    timestamps_dir = packet / "timestamps"
    inputs_dir = packet / "blinded-inputs"
    for directory in (frames_dir, timestamps_dir, inputs_dir):
        directory.mkdir(parents=True, exist_ok=False)
    rubric = {
        "schema": "flashpatch-l7-native-main-gold-rubric-v1",
        "profile_id": "wcag22-g19-v1",
        "source_url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G19.html",
        "instruction": "Label only the provided rendered frames and timestamps as SAFE or HAZARDOUS, with intervals when visible evidence supports them.",
    }
    rubric_path = packet / "rubric.json"
    rubric_path.write_text(json.dumps(rubric, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rubric_sha256 = _sha256_file(rubric_path)
    adjudicator_instructions = {
        "schema": "flashpatch-l7-native-main-adjudicator-instructions-v1",
        "allowed_inputs": [
            "blinded rendered RGB frame artifact",
            "blinded timestamp projection",
            "rubric",
        ],
        "forbidden_inputs": [
            "candidate detector output",
            "candidate tool identity",
            "source repository identity",
            "previous qualification decision",
            "score or ranking",
        ],
        "label_set": ["SAFE", "HAZARDOUS"],
        "interval_policy": "include intervals only when visible rendered evidence supports them",
        "uncertainty_policy": "return INCONCLUSIVE for cases that cannot be labeled from rendered frames and timestamps alone",
        "no_leak_attestation_required": True,
    }
    _reject_blinded_intake_leaks(adjudicator_instructions, "adjudicator_instructions")
    instructions_path = packet / "adjudicator-instructions.json"
    instructions_path.write_text(
        json.dumps(adjudicator_instructions, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return_contract = {
        "schema": "flashpatch-l7-native-main-independent-gold-return-contract-v1",
        "required_return_files": [
            "external-trust-policy.json",
            "independent-gold.json",
            "candidate-start-receipt.json",
        ],
        "required_gold_receipt_schema": INDEPENDENT_GOLD_SCHEMA,
        "required_trust_policy_schema": INDEPENDENT_GOLD_TRUST_SCHEMA,
        "required_candidate_start_schema": "flashpatch-l7-candidate-start-receipt-v1",
        "required_packet_manifest_binding": {
            "required_return_manifest_fields": [
                "adjudicator_packet_manifest_sha256",
                "adjudicator_packet_root_sha256",
            ],
            "adjudicator_packet_manifest_schema": NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_SCHEMA,
            "packet_root_hash_field": "packet_manifest_sha256",
            "binding_rule": "return_manifest_must_echo_manifest_file_sha256_and_packet_root_sha256",
        },
        "required_independent_gold_skeleton": {
            "required_top_level_fields": [
                "schema",
                "case_id",
                "case_class",
                "claim_tier",
                "controlled_mutation",
                "case_freeze",
                "gold_authority",
                "pre_candidate_commitment",
                "commitment_witness",
                "adjudication",
                "candidate_start_witness",
            ],
            "required_pre_candidate_commitment_fields": [
                "corpus_manifest_sha256",
                "blinded_gold_input_sha256",
                "blind_mapping_sha256",
                "case_freeze_sha256",
                "trust_policy_sha256",
                "committed_at",
            ],
            "required_signature_fields": [
                "signer_id",
                "signer_role",
                "signed_at",
                "signature_algorithm",
                "signature_base64",
            ],
            "binding_rule": "per_case_gold_must_bind_blinded_input_mapping_case_freeze_trust_policy_and_candidate_start_before_approval",
        },
        "scoreability_note": "returned files remain not scoreable until an external approval registry pins the trust policy before candidate execution",
        "case_binding": {
            "blind_case_ids_sha256": hashlib.sha256(_canonical_bytes([])).hexdigest(),
            "corpus_commitment_pending": True,
        },
    }
    return_contract_path = packet / "expected-return-contract.json"
    return_contract_path.write_text(
        json.dumps(return_contract, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trust_policy_template = {
        "schema": "flashpatch-l7-native-main-trust-policy-template-v1",
        "minimum_adjudicator_count": 3,
        "minimum_witness_count": 2,
        "required_adjudicator_role": "external_adjudicator",
        "required_witness_role": "external_timestamp_witness",
        "identity_disjointness_required": True,
        "key_disjointness_required": True,
        "external_approval_required": True,
        "revocation_field_required": True,
    }
    trust_policy_template_path = packet / "trust-policy-template.json"
    trust_policy_template_path.write_text(
        json.dumps(trust_policy_template, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    blind_ids: list[str] = []
    blinded_inputs: list[dict[str, Any]] = []
    sealed_mapping_rows: list[dict[str, Any]] = []
    used_blind_ids: set[str] = set()
    summary_sha256 = _sha256_file(summary_file)
    for index, row in enumerate(rows):
        frame_source = _project_bound_file(
            project_root,
            row.get("frame_artifact_path"),
            row.get("frame_artifact_sha256"),
            f"native-main row {index} frame artifact",
        )
        rgb_sha256, timestamps_f64_sha256, timestamps, frame_count = _renderer_hash_projection(frame_source)
        if frame_count != row.get("frame_count"):
            raise ContractError("native-main row frame count mismatches renderer artifact")
        blind_id = _blind_case_id(
            _canonical_bytes(
                {
                    "summary_sha256": summary_sha256,
                    "row_ordinal": index,
                    "frame_artifact_sha256": row.get("frame_artifact_sha256"),
                    "renderer_rgb_raw_sha256": rgb_sha256,
                    "timestamps_f64_sha256": timestamps_f64_sha256,
                }
            ),
            used_blind_ids,
        )
        blind_ids.append(blind_id)
        frame_target = frames_dir / f"{blind_id}.npz"
        shutil.copy2(frame_source, frame_target, follow_symlinks=False)
        timestamp_projection = {
            "schema": "flashpatch-l7-renderer-timestamps-v1",
            "timestamps_seconds": timestamps,
        }
        timestamp_target = timestamps_dir / f"{blind_id}.json"
        timestamp_target.write_text(
            json.dumps(timestamp_projection, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        blinded_input = {
            "schema": "flashpatch-l7-native-main-blinded-gold-input-v1",
            "blind_case_id": blind_id,
            "native_main_qualification_summary_sha256": summary_sha256,
            "frame_artifact_path": f"frames/{blind_id}.npz",
            "frame_artifact_sha256": _sha256_file(frame_target),
            "renderer_rgb_raw_sha256": rgb_sha256,
            "timestamps_path": f"timestamps/{blind_id}.json",
            "timestamps_projection_sha256": _sha256_file(timestamp_target),
            "timestamps_f64_sha256": timestamps_f64_sha256,
            "frame_count": frame_count,
            "color_space": "sRGB_BT709",
            "rubric_sha256": rubric_sha256,
        }
        _reject_blinded_intake_leaks(blinded_input, blind_id)
        input_path = inputs_dir / f"{blind_id}.json"
        input_path.write_text(json.dumps(blinded_input, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        blinded_inputs.append(
            {
                "blind_case_id": blind_id,
                "path": f"adjudicator-packet/blinded-inputs/{blind_id}.json",
                "sha256": _sha256_file(input_path),
            }
        )
        sealed_mapping_rows.append(
            {
                "blind_case_id": blind_id,
                "source_summary_sha256": summary_sha256,
                "row_ordinal": index,
                "source_row_receipt_sha256": row.get("raw_receipt_sha256"),
                "source_row_frame_artifact_sha256": row.get("frame_artifact_sha256"),
            }
        )
    corpus = {
        "schema": "flashpatch-l7-native-main-blinded-corpus-commitment-v1",
        "native_main_qualification_summary_sha256": summary_sha256,
        "blind_case_ids": blind_ids,
        "selection_rule_sha256": hashlib.sha256(
            b"all complete receipt_created native-main v7 qualification rows in summary order"
        ).hexdigest(),
        "stop_rule_sha256": _validate_hash(summary.get("discovery_manifest_sha256"), "native-main discovery manifest"),
        "frozen_at": frozen_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate_outputs_observed": False,
        "qualification_decisions_included": False,
    }
    _reject_blinded_intake_leaks(corpus, "corpus")
    corpus_path = packet / "blinded-corpus-commitment.json"
    corpus_path.write_text(json.dumps(corpus, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return_contract["case_binding"] = {
        "blind_case_ids_sha256": hashlib.sha256(_canonical_bytes(blind_ids)).hexdigest(),
        "corpus_commitment_sha256": _sha256_file(corpus_path),
    }
    return_contract_path.write_text(
        json.dumps(return_contract, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": NATIVE_MAIN_BLINDED_GOLD_INTAKE_SCHEMA,
        "status": "INDEPENDENT_GOLD_MISSING",
        "scoreable": False,
        "external_claim_authorized": False,
        "reason": "blinded_packet_prepared_but_no_independent_signer_or_trust_policy",
        "source_summary_path": str(summary_file.relative_to(project_root)),
        "source_summary_sha256": summary_sha256,
        "adjudicator_packet_root": "adjudicator-packet",
        "corpus_commitment_path": "adjudicator-packet/blinded-corpus-commitment.json",
        "corpus_commitment_sha256": _sha256_file(corpus_path),
        "rubric_path": "adjudicator-packet/rubric.json",
        "rubric_sha256": rubric_sha256,
        "adjudicator_instructions_path": "adjudicator-packet/adjudicator-instructions.json",
        "adjudicator_instructions_sha256": _sha256_file(instructions_path),
        "expected_return_contract_path": "adjudicator-packet/expected-return-contract.json",
        "expected_return_contract_sha256": _sha256_file(return_contract_path),
        "trust_policy_template_path": "adjudicator-packet/trust-policy-template.json",
        "trust_policy_template_sha256": _sha256_file(trust_policy_template_path),
        "case_count": len(blind_ids),
        "blinded_inputs": blinded_inputs,
        "sealed_mapping_sha256": hashlib.sha256(_canonical_bytes(sealed_mapping_rows)).hexdigest(),
        "score_blockers": [
            "independent_gold_missing",
            "independent_trust_policy_not_approved",
            "candidate_start_witness_missing",
            "same_condition_9_slot_execution_missing",
            "receipt_bound_score_bundle_missing",
        ],
    }
    receipt_path = destination / "gold-intake-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify_native_main_blinded_gold_intake(receipt_path)
    return receipt_path


def verify_native_main_blinded_gold_intake(receipt_path: Path) -> dict[str, Any]:
    """Verify the blind intake packet and keep it outside the score path."""

    receipt_file = receipt_path.resolve(strict=True)
    receipt_root = receipt_file.parent
    project_root = _find_project_root(receipt_file)
    receipt = _load_json(receipt_file)
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "scoreable",
            "external_claim_authorized",
            "reason",
            "source_summary_path",
            "source_summary_sha256",
            "adjudicator_packet_root",
            "corpus_commitment_path",
            "corpus_commitment_sha256",
            "rubric_path",
            "rubric_sha256",
            "adjudicator_instructions_path",
            "adjudicator_instructions_sha256",
            "expected_return_contract_path",
            "expected_return_contract_sha256",
            "trust_policy_template_path",
            "trust_policy_template_sha256",
            "case_count",
            "blinded_inputs",
            "sealed_mapping_sha256",
            "score_blockers",
        },
        "native-main blinded gold intake receipt",
    )
    if (
        receipt.get("schema") != NATIVE_MAIN_BLINDED_GOLD_INTAKE_SCHEMA
        or receipt.get("status") != "INDEPENDENT_GOLD_MISSING"
        or receipt.get("scoreable") is not False
        or receipt.get("external_claim_authorized") is not False
    ):
        raise ContractError("native-main blinded gold intake scoreability claim is invalid")
    blockers = receipt.get("score_blockers")
    if not isinstance(blockers, list) or "independent_gold_missing" not in blockers:
        raise ContractError("native-main blinded gold intake blockers are invalid")
    summary_file = _project_bound_file(
        project_root,
        receipt.get("source_summary_path"),
        receipt.get("source_summary_sha256"),
        "native-main qualification summary",
    )
    from .l7_verify import verify_native_main_qualification_summary

    verify_native_main_qualification_summary(summary_file)
    summary = _load_json(summary_file)
    rows = [row for row in summary.get("results", []) if isinstance(row, dict) and row.get("receipt_created") is True]
    if receipt.get("case_count") != len(rows):
        raise ContractError("native-main blinded gold intake case count mismatch")
    corpus_path = _receipt_bound_file(
        receipt_root,
        receipt.get("corpus_commitment_path"),
        receipt.get("corpus_commitment_sha256"),
        "blinded corpus commitment",
    )
    rubric_path = _receipt_bound_file(
        receipt_root,
        receipt.get("rubric_path"),
        receipt.get("rubric_sha256"),
        "blinded gold rubric",
    )
    instructions_path = _receipt_bound_file(
        receipt_root,
        receipt.get("adjudicator_instructions_path"),
        receipt.get("adjudicator_instructions_sha256"),
        "blinded gold adjudicator instructions",
    )
    return_contract_path = _receipt_bound_file(
        receipt_root,
        receipt.get("expected_return_contract_path"),
        receipt.get("expected_return_contract_sha256"),
        "blinded gold expected return contract",
    )
    trust_template_path = _receipt_bound_file(
        receipt_root,
        receipt.get("trust_policy_template_path"),
        receipt.get("trust_policy_template_sha256"),
        "blinded gold trust policy template",
    )
    corpus = _load_json(corpus_path)
    rubric = _load_json(rubric_path)
    instructions = _load_json(instructions_path)
    return_contract = _load_json(return_contract_path)
    trust_template = _load_json(trust_template_path)
    _reject_blinded_intake_leaks(corpus, "corpus")
    _reject_blinded_intake_leaks(rubric, "rubric")
    _reject_blinded_intake_leaks(instructions, "adjudicator_instructions")
    _require_exact_keys(
        corpus,
        {
            "schema",
            "native_main_qualification_summary_sha256",
            "blind_case_ids",
            "selection_rule_sha256",
            "stop_rule_sha256",
            "frozen_at",
            "candidate_outputs_observed",
            "qualification_decisions_included",
        },
        "native-main blinded corpus commitment",
    )
    if (
        corpus.get("schema") != "flashpatch-l7-native-main-blinded-corpus-commitment-v1"
        or corpus.get("native_main_qualification_summary_sha256") != receipt["source_summary_sha256"]
        or corpus.get("candidate_outputs_observed") is not False
        or corpus.get("qualification_decisions_included") is not False
    ):
        raise ContractError("native-main blinded corpus commitment is invalid")
    blind_case_ids = corpus.get("blind_case_ids")
    if (
        not isinstance(blind_case_ids, list)
        or len(blind_case_ids) != receipt["case_count"]
        or len(blind_case_ids) != len(set(blind_case_ids))
        or any(not isinstance(value, str) or re.fullmatch(r"blind-[0-9a-f]{32}", value) is None for value in blind_case_ids)
    ):
        raise ContractError("native-main blinded corpus blind identities are invalid")
    _parse_utc_timestamp(corpus.get("frozen_at"), "native-main blinded corpus")
    _validate_hash(corpus.get("selection_rule_sha256"), "native-main blinded corpus selection rule")
    _validate_hash(corpus.get("stop_rule_sha256"), "native-main blinded corpus stop rule")
    _require_exact_keys(
        rubric,
        {"schema", "profile_id", "source_url", "instruction"},
        "native-main blinded gold rubric",
    )
    if rubric.get("schema") != "flashpatch-l7-native-main-gold-rubric-v1":
        raise ContractError("native-main blinded gold rubric is invalid")
    _require_exact_keys(
        instructions,
        {
            "schema",
            "allowed_inputs",
            "forbidden_inputs",
            "label_set",
            "interval_policy",
            "uncertainty_policy",
            "no_leak_attestation_required",
        },
        "native-main adjudicator instructions",
    )
    if (
        instructions.get("schema") != "flashpatch-l7-native-main-adjudicator-instructions-v1"
        or instructions.get("label_set") != ["SAFE", "HAZARDOUS"]
        or instructions.get("no_leak_attestation_required") is not True
    ):
        raise ContractError("native-main adjudicator instructions are invalid")
    forbidden_inputs = instructions.get("forbidden_inputs")
    if not isinstance(forbidden_inputs, list) or not {
        "candidate detector output",
        "candidate tool identity",
        "source repository identity",
        "previous qualification decision",
        "score or ranking",
    }.issubset(set(forbidden_inputs)):
        raise ContractError("native-main adjudicator instructions do not block leakage")
    _require_exact_keys(
        return_contract,
        {
            "schema",
            "required_return_files",
            "required_gold_receipt_schema",
            "required_trust_policy_schema",
            "required_candidate_start_schema",
            "required_packet_manifest_binding",
            "required_independent_gold_skeleton",
            "scoreability_note",
            "case_binding",
        },
        "native-main independent gold return contract",
    )
    case_binding = return_contract.get("case_binding")
    packet_binding = return_contract.get("required_packet_manifest_binding")
    gold_skeleton = return_contract.get("required_independent_gold_skeleton")
    if (
        return_contract.get("schema") != "flashpatch-l7-native-main-independent-gold-return-contract-v1"
        or return_contract.get("required_gold_receipt_schema") != INDEPENDENT_GOLD_SCHEMA
        or return_contract.get("required_trust_policy_schema") != INDEPENDENT_GOLD_TRUST_SCHEMA
        or return_contract.get("required_candidate_start_schema") != "flashpatch-l7-candidate-start-receipt-v1"
        or return_contract.get("required_return_files") != [
            "external-trust-policy.json",
            "independent-gold.json",
            "candidate-start-receipt.json",
        ]
        or not isinstance(case_binding, dict)
        or case_binding.get("corpus_commitment_sha256") != receipt["corpus_commitment_sha256"]
    ):
        raise ContractError("native-main independent gold return contract is invalid")
    if (
        not isinstance(packet_binding, dict)
        or packet_binding.get("required_return_manifest_fields")
        != [
            "adjudicator_packet_manifest_sha256",
            "adjudicator_packet_root_sha256",
        ]
        or packet_binding.get("adjudicator_packet_manifest_schema")
        != NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_SCHEMA
        or packet_binding.get("packet_root_hash_field") != "packet_manifest_sha256"
        or packet_binding.get("binding_rule")
        != "return_manifest_must_echo_manifest_file_sha256_and_packet_root_sha256"
    ):
        raise ContractError("native-main independent gold return contract packet manifest binding is invalid")
    if (
        not isinstance(gold_skeleton, dict)
        or gold_skeleton.get("required_top_level_fields")
        != [
            "schema",
            "case_id",
            "case_class",
            "claim_tier",
            "controlled_mutation",
            "case_freeze",
            "gold_authority",
            "pre_candidate_commitment",
            "commitment_witness",
            "adjudication",
            "candidate_start_witness",
        ]
        or gold_skeleton.get("required_pre_candidate_commitment_fields")
        != [
            "corpus_manifest_sha256",
            "blinded_gold_input_sha256",
            "blind_mapping_sha256",
            "case_freeze_sha256",
            "trust_policy_sha256",
            "committed_at",
        ]
        or gold_skeleton.get("required_signature_fields")
        != [
            "signer_id",
            "signer_role",
            "signed_at",
            "signature_algorithm",
            "signature_base64",
        ]
        or gold_skeleton.get("binding_rule")
        != "per_case_gold_must_bind_blinded_input_mapping_case_freeze_trust_policy_and_candidate_start_before_approval"
    ):
        raise ContractError("native-main independent gold return contract skeleton binding is invalid")
    _validate_hash(case_binding.get("blind_case_ids_sha256"), "native-main return contract blind ids")
    _require_exact_keys(
        trust_template,
        {
            "schema",
            "minimum_adjudicator_count",
            "minimum_witness_count",
            "required_adjudicator_role",
            "required_witness_role",
            "identity_disjointness_required",
            "key_disjointness_required",
            "external_approval_required",
            "revocation_field_required",
        },
        "native-main trust policy template",
    )
    if (
        trust_template.get("schema") != "flashpatch-l7-native-main-trust-policy-template-v1"
        or trust_template.get("minimum_adjudicator_count") != 3
        or trust_template.get("minimum_witness_count") != 2
        or trust_template.get("required_adjudicator_role") != "external_adjudicator"
        or trust_template.get("required_witness_role") != "external_timestamp_witness"
        or trust_template.get("identity_disjointness_required") is not True
        or trust_template.get("key_disjointness_required") is not True
        or trust_template.get("external_approval_required") is not True
        or trust_template.get("revocation_field_required") is not True
    ):
        raise ContractError("native-main trust policy template is invalid")
    inputs = receipt.get("blinded_inputs")
    if not isinstance(inputs, list) or len(inputs) != len(blind_case_ids):
        raise ContractError("native-main blinded input list is invalid")
    expected_tuples: list[tuple[str, str, str, int]] = []
    for row in rows:
        frame_source = _project_bound_file(
            project_root,
            row.get("frame_artifact_path"),
            row.get("frame_artifact_sha256"),
            "native-main source frame artifact",
        )
        rgb_sha256, timestamps_f64_sha256, _, frame_count = _renderer_hash_projection(frame_source)
        expected_tuples.append((str(row["frame_artifact_sha256"]), rgb_sha256, timestamps_f64_sha256, frame_count))
    observed_tuples: list[tuple[str, str, str, int]] = []
    seen_inputs: set[str] = set()
    for entry in inputs:
        if not isinstance(entry, dict):
            raise ContractError("native-main blinded input entry is invalid")
        _require_exact_keys(entry, {"blind_case_id", "path", "sha256"}, "native-main blinded input entry")
        blind_id = entry.get("blind_case_id")
        if blind_id not in blind_case_ids or blind_id in seen_inputs:
            raise ContractError("native-main blinded input identity is invalid")
        seen_inputs.add(str(blind_id))
        input_path = _receipt_bound_file(receipt_root, entry.get("path"), entry.get("sha256"), "native-main blinded input")
        blinded = _load_json(input_path)
        _reject_blinded_intake_leaks(blinded, str(blind_id))
        _require_exact_keys(
            blinded,
            {
                "schema",
                "blind_case_id",
                "native_main_qualification_summary_sha256",
                "frame_artifact_path",
                "frame_artifact_sha256",
                "renderer_rgb_raw_sha256",
                "timestamps_path",
                "timestamps_projection_sha256",
                "timestamps_f64_sha256",
                "frame_count",
                "color_space",
                "rubric_sha256",
            },
            "native-main blinded input",
        )
        if (
            blinded.get("schema") != "flashpatch-l7-native-main-blinded-gold-input-v1"
            or blinded.get("blind_case_id") != blind_id
            or blinded.get("native_main_qualification_summary_sha256") != receipt["source_summary_sha256"]
            or blinded.get("rubric_sha256") != receipt["rubric_sha256"]
            or blinded.get("color_space") != "sRGB_BT709"
        ):
            raise ContractError("native-main blinded input binding is invalid")
        frame_copy = _receipt_bound_file(
            input_path.parent.parent,
            blinded.get("frame_artifact_path"),
            blinded.get("frame_artifact_sha256"),
            "native-main blinded frame artifact",
        )
        timestamp_file = _receipt_bound_file(
            input_path.parent.parent,
            blinded.get("timestamps_path"),
            blinded.get("timestamps_projection_sha256"),
            "native-main blinded timestamps",
        )
        rgb_sha256, timestamps_f64_sha256, timestamps, frame_count = _renderer_hash_projection(frame_copy)
        if (
            blinded.get("renderer_rgb_raw_sha256") != rgb_sha256
            or blinded.get("timestamps_f64_sha256") != timestamps_f64_sha256
            or blinded.get("frame_count") != frame_count
        ):
            raise ContractError("native-main blinded input renderer projection mismatch")
        timestamp_projection = _load_json(timestamp_file)
        if timestamp_projection != {
            "schema": "flashpatch-l7-renderer-timestamps-v1",
            "timestamps_seconds": timestamps,
        }:
            raise ContractError("native-main blinded timestamp projection mismatch")
        observed_tuples.append((str(blinded["frame_artifact_sha256"]), rgb_sha256, timestamps_f64_sha256, frame_count))
    if sorted(expected_tuples) != sorted(observed_tuples):
        raise ContractError("native-main blinded intake does not match complete renderer receipts")
    if case_binding.get("blind_case_ids_sha256") != hashlib.sha256(_canonical_bytes(blind_case_ids)).hexdigest():
        raise ContractError("native-main return contract blind identities mismatch")
    return {
        "schema": NATIVE_MAIN_BLINDED_GOLD_INTAKE_ASSESSMENT_SCHEMA,
        "status": "INDEPENDENT_GOLD_MISSING",
        "scoreable": False,
        "case_count": len(observed_tuples),
        "source_summary_sha256": receipt["source_summary_sha256"],
        "corpus_commitment_sha256": receipt["corpus_commitment_sha256"],
        "rubric_sha256": receipt["rubric_sha256"],
        "adjudicator_instructions_sha256": receipt["adjudicator_instructions_sha256"],
        "expected_return_contract_sha256": receipt["expected_return_contract_sha256"],
        "trust_policy_template_sha256": receipt["trust_policy_template_sha256"],
        "score_blockers": blockers,
        "external_claim_authorized": False,
    }


def _native_main_packet_files(receipt_root: Path, packet_root: Path) -> list[dict[str, Any]]:
    try:
        packet_root.relative_to(receipt_root)
    except ValueError as exc:
        raise ContractError("native-main adjudicator packet root escapes intake receipt") from exc
    if packet_root.is_symlink() or not packet_root.is_dir():
        raise ContractError("native-main adjudicator packet root is invalid")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(packet_root.rglob("*")):
        if candidate.is_symlink():
            raise ContractError("native-main adjudicator packet contains symlink")
        if not candidate.is_file():
            continue
        relative_path = candidate.relative_to(receipt_root).as_posix()
        if relative_path in seen:
            raise ContractError("native-main adjudicator packet contains duplicate path")
        seen.add(relative_path)
        files.append(
            {
                "path": relative_path,
                "sha256": _sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    if not files:
        raise ContractError("native-main adjudicator packet is empty")
    return files


def _native_main_packet_root_hash(files: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(files)).hexdigest()


def seal_native_main_adjudicator_packet(
    *,
    intake_receipt: Path,
    output_path: Path,
) -> Path:
    """Write a whole-packet manifest for the blinded native-main adjudicator packet.

    The manifest is still non-scoreable.  It only proves that the packet handed
    to a future external adjudicator has a single root hash and can be reopened
    without trusting the caller's file list.
    """

    intake_file = intake_receipt.resolve(strict=True)
    intake_root = intake_file.parent
    project_root = _find_project_root(intake_file)
    output = output_path.resolve()
    try:
        output.relative_to(project_root)
    except ValueError as exc:
        raise ContractError("native-main adjudicator packet manifest output escapes project") from exc
    if output.exists() or output.is_symlink():
        raise ContractError("native-main adjudicator packet manifest output already exists")
    assessment = verify_native_main_blinded_gold_intake(intake_file)
    intake = _load_json(intake_file)
    packet_root_value = intake.get("adjudicator_packet_root")
    if not isinstance(packet_root_value, str) or packet_root_value.startswith("/") or "\x00" in packet_root_value:
        raise ContractError("native-main adjudicator packet root is invalid")
    packet_root = (intake_root / packet_root_value).resolve()
    files = _native_main_packet_files(intake_root, packet_root)
    packet_bytes = sum(int(row["size_bytes"]) for row in files)
    packet_manifest_sha256 = _native_main_packet_root_hash(files)
    manifest = {
        "schema": NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_SCHEMA,
        "status": "INDEPENDENT_GOLD_MISSING",
        "scoreable": False,
        "external_claim_authorized": False,
        "reason": "adjudicator_packet_sealed_but_no_external_signer_or_approved_trust_policy",
        "intake_receipt_path": str(intake_file.relative_to(project_root)),
        "intake_receipt_sha256": _sha256_file(intake_file),
        "packet_root": packet_root_value,
        "packet_file_count": len(files),
        "packet_bytes": packet_bytes,
        "packet_manifest_sha256": packet_manifest_sha256,
        "case_count": assessment["case_count"],
        "source_summary_sha256": assessment["source_summary_sha256"],
        "corpus_commitment_sha256": assessment["corpus_commitment_sha256"],
        "rubric_sha256": assessment["rubric_sha256"],
        "adjudicator_instructions_sha256": assessment["adjudicator_instructions_sha256"],
        "expected_return_contract_sha256": assessment["expected_return_contract_sha256"],
        "trust_policy_template_sha256": assessment["trust_policy_template_sha256"],
        "score_blockers": assessment["score_blockers"],
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify_native_main_adjudicator_packet_manifest(output)
    return output


def verify_native_main_adjudicator_packet_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify a sealed native-main adjudicator packet manifest."""

    manifest_file = manifest_path.resolve(strict=True)
    project_root = _find_project_root(manifest_file)
    manifest = _load_json(manifest_file)
    _require_exact_keys(
        manifest,
        {
            "schema",
            "status",
            "scoreable",
            "external_claim_authorized",
            "reason",
            "intake_receipt_path",
            "intake_receipt_sha256",
            "packet_root",
            "packet_file_count",
            "packet_bytes",
            "packet_manifest_sha256",
            "case_count",
            "source_summary_sha256",
            "corpus_commitment_sha256",
            "rubric_sha256",
            "adjudicator_instructions_sha256",
            "expected_return_contract_sha256",
            "trust_policy_template_sha256",
            "score_blockers",
            "files",
        },
        "native-main adjudicator packet manifest",
    )
    if (
        manifest.get("schema") != NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_SCHEMA
        or manifest.get("status") != "INDEPENDENT_GOLD_MISSING"
        or manifest.get("scoreable") is not False
        or manifest.get("external_claim_authorized") is not False
        or manifest.get("reason") != "adjudicator_packet_sealed_but_no_external_signer_or_approved_trust_policy"
    ):
        raise ContractError("native-main adjudicator packet manifest scoreability claim is invalid")
    intake_file = _project_bound_file(
        project_root,
        manifest.get("intake_receipt_path"),
        manifest.get("intake_receipt_sha256"),
        "native-main adjudicator packet intake receipt",
    )
    intake_root = intake_file.parent
    intake = _load_json(intake_file)
    assessment = verify_native_main_blinded_gold_intake(intake_file)
    if (
        manifest.get("case_count") != assessment["case_count"]
        or manifest.get("source_summary_sha256") != assessment["source_summary_sha256"]
        or manifest.get("corpus_commitment_sha256") != assessment["corpus_commitment_sha256"]
        or manifest.get("rubric_sha256") != assessment["rubric_sha256"]
        or manifest.get("adjudicator_instructions_sha256") != assessment["adjudicator_instructions_sha256"]
        or manifest.get("expected_return_contract_sha256") != assessment["expected_return_contract_sha256"]
        or manifest.get("trust_policy_template_sha256") != assessment["trust_policy_template_sha256"]
        or manifest.get("score_blockers") != assessment["score_blockers"]
    ):
        raise ContractError("native-main adjudicator packet manifest intake snapshot mismatch")
    packet_root_value = manifest.get("packet_root")
    if packet_root_value != intake.get("adjudicator_packet_root"):
        raise ContractError("native-main adjudicator packet manifest root mismatch")
    if not isinstance(packet_root_value, str) or packet_root_value.startswith("/") or "\x00" in packet_root_value:
        raise ContractError("native-main adjudicator packet manifest root is invalid")
    packet_root = (intake_root / packet_root_value).resolve()
    actual_files = _native_main_packet_files(intake_root, packet_root)
    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise ContractError("native-main adjudicator packet manifest file list is invalid")
    normalized_declared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in declared_files:
        if not isinstance(entry, dict):
            raise ContractError("native-main adjudicator packet manifest file entry is invalid")
        _require_exact_keys(entry, {"path", "sha256", "size_bytes"}, "native-main adjudicator packet manifest file entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or "\x00" in relative:
            raise ContractError("native-main adjudicator packet manifest file path is invalid")
        path = (intake_root / relative).resolve()
        try:
            path.relative_to(packet_root)
        except ValueError as exc:
            raise ContractError("native-main adjudicator packet manifest file escapes packet") from exc
        if relative in seen:
            raise ContractError("native-main adjudicator packet manifest duplicate file")
        seen.add(relative)
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError("native-main adjudicator packet manifest file size is invalid")
        normalized_declared.append(
            {
                "path": relative,
                "sha256": _validate_hash(entry.get("sha256"), "native-main adjudicator packet manifest file"),
                "size_bytes": size,
            }
        )
    normalized_declared.sort(key=lambda row: row["path"])
    if normalized_declared != actual_files:
        raise ContractError("native-main adjudicator packet manifest file list mismatch")
    if (
        manifest.get("packet_file_count") != len(actual_files)
        or manifest.get("packet_bytes") != sum(int(row["size_bytes"]) for row in actual_files)
        or manifest.get("packet_manifest_sha256") != _native_main_packet_root_hash(actual_files)
    ):
        raise ContractError("native-main adjudicator packet manifest hash mismatch")
    return {
        "schema": NATIVE_MAIN_ADJUDICATOR_PACKET_MANIFEST_ASSESSMENT_SCHEMA,
        "status": "INDEPENDENT_GOLD_MISSING",
        "scoreable": False,
        "external_claim_authorized": False,
        "case_count": assessment["case_count"],
        "intake_receipt_sha256": _sha256_file(intake_file),
        "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "packet_file_count": len(actual_files),
        "packet_bytes": manifest["packet_bytes"],
        "score_blockers": assessment["score_blockers"],
        "reason": "adjudicator_packet_manifest_verified_but_independent_gold_missing",
    }


def _reject_native_main_gold_return_contamination(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in NATIVE_MAIN_GOLD_RETURN_FORBIDDEN_KEYS:
                raise ContractError(f"native-main independent gold return leaks candidate output: {context}:{key}")
            _reject_native_main_gold_return_contamination(child, f"{context}:{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_native_main_gold_return_contamination(child, f"{context}[{index}]")


def _native_main_return_bound_file(root: Path, value: object, expected_sha256: object, context: str) -> Path:
    return _receipt_bound_file(root, value, expected_sha256, f"native-main independent gold return {context}")


def _validate_native_main_return_signature_shape(
    value: object,
    *,
    context: str,
    signer_role: str,
) -> datetime:
    if not isinstance(value, dict):
        raise ContractError(f"native-main independent gold return {context} signature is missing")
    for field in ("signer_id", "signer_role", "signed_at", "signature_algorithm", "signature_base64"):
        if field not in value:
            raise ContractError(f"native-main independent gold return {context} signature is missing")
    signer_id = value.get("signer_id")
    if not isinstance(signer_id, str) or not signer_id:
        raise ContractError(f"native-main independent gold return {context} signer identity is invalid")
    if value.get("signer_role") != signer_role:
        raise ContractError(f"native-main independent gold return {context} signer role is invalid")
    if value.get("signature_algorithm") != "Ed25519":
        raise ContractError(f"native-main independent gold return {context} signature algorithm is invalid")
    signature = value.get("signature_base64")
    if not isinstance(signature, str):
        raise ContractError(f"native-main independent gold return {context} signature is missing")
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError(f"native-main independent gold return {context} signature encoding is invalid") from exc
    if len(signature_bytes) != 64:
        raise ContractError(f"native-main independent gold return {context} signature length is invalid")
    return _parse_utc_timestamp(value.get("signed_at"), f"native-main independent gold return {context}")


def _native_main_return_policy_identity_ids(
    policy: dict[str, Any],
    field: str,
) -> set[str]:
    rows = policy.get(field)
    if not isinstance(rows, list):
        raise ContractError("native-main independent gold trust policy identity roster is invalid")
    identities: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("native-main independent gold trust policy identity roster is invalid")
        identity = row.get("id", row.get("identity_id"))
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ContractError("native-main independent gold trust policy identity roster is invalid")
        identities.add(identity)
    return identities


def _native_main_policy_public_keys(
    policy_path: Path,
    policy: dict[str, Any],
    field: str,
    expected_role: str,
) -> dict[str, bytes]:
    rows = policy.get(field)
    if not isinstance(rows, list) or not rows:
        raise ContractError("native-main independent gold trust policy identity roster is invalid")
    keys: dict[str, bytes] = {}
    seen_spki: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError("native-main independent gold trust policy identity roster is invalid")
        identity = row.get("id", row.get("identity_id"))
        if not isinstance(identity, str) or not identity or identity in keys:
            raise ContractError("native-main independent gold trust policy identity roster is invalid")
        if row.get("role") != expected_role:
            raise ContractError("native-main independent gold trust policy identity role is invalid")
        key_entry = {
            "identity_id": identity,
            "role": expected_role,
            "public_key_path": row.get("public_key_path"),
            "public_key_sha256": row.get("public_key_sha256"),
            "ed25519_spki_sha256": row.get("ed25519_spki_sha256"),
        }
        key = _read_pinned_public_key(policy_path, key_entry, f"native-main {field}[{index}]")
        spki = str(key_entry["ed25519_spki_sha256"])
        if spki in seen_spki:
            raise ContractError("native-main independent gold trust policy reuses a key")
        seen_spki.add(spki)
        keys[identity] = key
    return keys


def _native_main_return_policy_operator_ids(policy: dict[str, Any]) -> set[str]:
    identities: set[str] = set()
    for field in ("flashpatch_operator_ids", "candidate_operator_ids"):
        rows = policy.get(field)
        if not isinstance(rows, list) or any(not isinstance(value, str) or not value for value in rows):
            raise ContractError("native-main independent gold trust policy operator roster is invalid")
        identities.update(str(value) for value in rows)
    return identities


def _validate_native_main_return_signer_independence(
    policy: dict[str, Any],
) -> tuple[set[str], set[str]]:
    adjudicator_ids = _native_main_return_policy_identity_ids(policy, "adjudicators")
    witness_ids = _native_main_return_policy_identity_ids(policy, "timestamp_witnesses")
    operator_ids = _native_main_return_policy_operator_ids(policy)
    candidate_identities = {
        identity.casefold()
        for identity in (*GOLD_EXCLUDED_TOOLS, *operator_ids)
    }
    if (
        adjudicator_ids & witness_ids
        or adjudicator_ids & operator_ids
        or witness_ids & operator_ids
        or any(identity.casefold() in candidate_identities for identity in adjudicator_ids | witness_ids)
    ):
        raise ContractError("native-main independent gold return signer is not independent")
    return adjudicator_ids, witness_ids


def _validate_native_main_return_gold_skeleton(
    *,
    gold: dict[str, Any],
    blind_id: str,
    intake: dict[str, Any],
    blinded_input: dict[str, Any],
    trust_policy: dict[str, Any],
    trust_policy_sha256: object,
    candidate_start_file: Path,
) -> None:
    """Reject unsigned or unbound native-main gold before approval is considered."""

    approved_adjudicators, approved_witnesses = _validate_native_main_return_signer_independence(
        trust_policy,
    )
    _require_exact_keys(
        gold,
        {
            "schema",
            "case_id",
            "case_class",
            "claim_tier",
            "controlled_mutation",
            "case_freeze",
            "gold_authority",
            "pre_candidate_commitment",
            "commitment_witness",
            "adjudication",
            "candidate_start_witness",
        },
        "native-main independent gold return receipt skeleton",
    )
    if (
        gold.get("schema") != INDEPENDENT_GOLD_SCHEMA
        or gold.get("case_id") != blind_id
        or gold.get("case_class") != "natural_external"
        or gold.get("claim_tier") != "L9_ELIGIBLE"
        or gold.get("controlled_mutation") is not False
    ):
        raise ContractError("native-main independent gold return receipt skeleton is invalid")

    freeze = _required_object(gold, "case_freeze")
    _require_exact_keys(
        freeze,
        {
            "blind_case_id",
            "source_summary_sha256",
            "corpus_commitment_sha256",
            "frame_artifact_sha256",
            "renderer_rgb_raw_sha256",
            "timestamps_f64_sha256",
            "frame_count",
            "color_space",
            "frozen_at",
        },
        "native-main independent gold return case freeze",
    )
    if (
        freeze.get("blind_case_id") != blind_id
        or freeze.get("source_summary_sha256") != intake.get("source_summary_sha256")
        or freeze.get("corpus_commitment_sha256") != intake.get("corpus_commitment_sha256")
        or freeze.get("frame_artifact_sha256") != blinded_input.get("frame_artifact_sha256")
        or freeze.get("renderer_rgb_raw_sha256") != blinded_input.get("renderer_rgb_raw_sha256")
        or freeze.get("timestamps_f64_sha256") != blinded_input.get("timestamps_f64_sha256")
        or freeze.get("frame_count") != blinded_input.get("frame_count")
        or freeze.get("color_space") != blinded_input.get("color_space")
    ):
        raise ContractError("native-main independent gold return case freeze is not bound to intake")
    _parse_utc_timestamp(freeze.get("frozen_at"), "native-main independent gold return case freeze")
    case_freeze_sha256 = hashlib.sha256(_canonical_bytes(freeze)).hexdigest()

    authority = _required_object(gold, "gold_authority")
    _require_exact_keys(
        authority,
        {
            "kind",
            "candidate_tools_excluded",
            "required_independent_submissions",
            "trust_policy_sha256",
        },
        "native-main independent gold return authority",
    )
    if (
        authority.get("kind") != "independent_adjudication"
        or not isinstance(authority.get("candidate_tools_excluded"), list)
        or set(authority.get("candidate_tools_excluded", [])) != GOLD_EXCLUDED_TOOLS
        or authority.get("required_independent_submissions") != trust_policy.get("required_independent_submissions")
        or authority.get("trust_policy_sha256") != trust_policy_sha256
    ):
        raise ContractError("native-main independent gold return authority is invalid")

    commitment = _required_object(gold, "pre_candidate_commitment")
    _require_exact_keys(
        commitment,
        {
            "corpus_manifest_sha256",
            "blinded_gold_input_sha256",
            "blind_mapping_sha256",
            "case_freeze_sha256",
            "trust_policy_sha256",
            "committed_at",
        },
        "native-main independent gold return pre-candidate commitment",
    )
    if (
        commitment.get("corpus_manifest_sha256") != intake.get("corpus_commitment_sha256")
        or commitment.get("blinded_gold_input_sha256") != blinded_input.get("_entry_sha256")
        or commitment.get("blind_mapping_sha256") != intake.get("sealed_mapping_sha256")
        or commitment.get("case_freeze_sha256") != case_freeze_sha256
        or commitment.get("trust_policy_sha256") != trust_policy_sha256
    ):
        raise ContractError("native-main independent gold return commitment is not bound to intake")
    committed_at = _parse_utc_timestamp(
        commitment.get("committed_at"),
        "native-main independent gold return pre-candidate commitment",
    )
    commitment_sha256 = hashlib.sha256(_canonical_bytes(commitment)).hexdigest()

    commitment_witness = _required_object(gold, "commitment_witness")
    if commitment_witness.get("pre_candidate_commitment_sha256") != commitment_sha256:
        raise ContractError("native-main independent gold return commitment witness is not bound")
    commitment_witness_id = commitment_witness.get("signer_id")
    if commitment_witness_id not in approved_witnesses:
        raise ContractError("native-main independent gold return signer is not independent")
    commitment_witness_at = _validate_native_main_return_signature_shape(
        commitment_witness,
        context="commitment witness",
        signer_role="external_timestamp_witness",
    )
    if commitment_witness_at < committed_at:
        raise ContractError("native-main independent gold return commitment witness predates commitment")

    adjudication = _required_object(gold, "adjudication")
    _require_exact_keys(
        adjudication,
        {"method", "result", "intervals", "submissions"},
        "native-main independent gold return adjudication",
    )
    decision = adjudication.get("result")
    if adjudication.get("method") != "independent-blinded-renderer-review" or decision not in {"SAFE", "HAZARDOUS"}:
        raise ContractError("native-main independent gold return adjudication is invalid")
    intervals = _validate_gold_intervals(
        adjudication.get("intervals"),
        str(decision),
        "native-main independent gold return adjudication",
        capture_start=0.0,
        capture_end=float("inf"),
    )
    submissions = adjudication.get("submissions")
    required_submissions = trust_policy.get("required_independent_submissions")
    if (
        isinstance(required_submissions, bool)
        or not isinstance(required_submissions, int)
        or not isinstance(submissions, list)
        or len(submissions) < required_submissions
    ):
        raise ContractError("native-main independent gold return lacks required submissions")
    signer_ids: set[str] = set()
    for index, submission in enumerate(submissions):
        if not isinstance(submission, dict):
            raise ContractError("native-main independent gold return submission is invalid")
        _require_exact_keys(
            submission,
            {
                "signer_id",
                "signer_role",
                "decision",
                "intervals",
                "pre_candidate_commitment_sha256",
                "signed_at",
                "signature_algorithm",
                "signature_base64",
            },
            "native-main independent gold return submission",
        )
        signer = str(submission.get("signer_id"))
        if signer in signer_ids or signer not in approved_adjudicators:
            raise ContractError("native-main independent gold return submissions are not independent")
        signer_ids.add(signer)
        if (
            submission.get("decision") != decision
            or submission.get("intervals") != intervals
            or submission.get("pre_candidate_commitment_sha256") != commitment_sha256
        ):
            raise ContractError("native-main independent gold return submission is not bound")
        signed_at = _validate_native_main_return_signature_shape(
            submission,
            context=f"submission {index}",
            signer_role="external_adjudicator",
        )
        if signed_at < committed_at:
            raise ContractError("native-main independent gold return submission predates commitment")

    candidate_start = _load_json(candidate_start_file)
    _require_exact_keys(
        candidate_start,
        {
            "schema",
            "blind_case_id",
            "started_at",
            "state",
            "candidate_outputs_observed",
        },
        "native-main independent gold return candidate start receipt",
    )
    if (
        candidate_start.get("schema") != "flashpatch-l7-candidate-start-receipt-v1"
        or candidate_start.get("blind_case_id") != blind_id
        or candidate_start.get("state") != "STARTED_OUTPUT_UNOPENED"
        or candidate_start.get("candidate_outputs_observed") is not False
    ):
        raise ContractError("native-main independent gold return candidate start receipt is invalid")
    candidate_start_witness = _required_object(gold, "candidate_start_witness")
    if (
        candidate_start_witness.get("case_id") != blind_id
        or candidate_start_witness.get("pre_candidate_commitment_sha256") != commitment_sha256
        or candidate_start_witness.get("candidate_start_receipt_sha256") != _sha256_file(candidate_start_file)
    ):
        raise ContractError("native-main independent gold return candidate-start witness is not bound")
    candidate_start_witness_id = candidate_start_witness.get("signer_id")
    if (
        candidate_start_witness_id not in approved_witnesses
        or candidate_start_witness_id == commitment_witness_id
    ):
        raise ContractError("native-main independent gold return signer is not independent")
    witness_at = _validate_native_main_return_signature_shape(
        candidate_start_witness,
        context="candidate-start witness",
        signer_role="external_timestamp_witness",
    )
    started_at = _parse_utc_timestamp(
        candidate_start.get("started_at"),
        "native-main independent gold return candidate start receipt",
    )
    if witness_at < started_at or started_at < committed_at:
        raise ContractError("native-main independent gold return candidate-start timing is invalid")


def _native_main_signature_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: child
        for key, child in value.items()
        if key not in {"signature_algorithm", "signature_base64"}
    }


def _native_main_signature_envelope(
    *,
    message_kind: str,
    approval: ApprovedGoldTrustPolicy,
    registry_snapshot_sha256: str,
    policy: dict[str, Any],
    signer_id: str,
    signer_role: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": "FlashPatch/L7/native-main-independent-gold/v1",
        "message_kind": message_kind,
        "trust_policy_sha256": approval.policy_sha256,
        "approval_record_sha256": approval.approval_record_sha256,
        "external_timestamp_token_sha256": approval.external_timestamp_token_sha256,
        "registry_snapshot_sha256": registry_snapshot_sha256,
        "verifier_source_commit": approval.verifier_source_commit,
        "openssl_binary_sha256": approval.openssl_binary_sha256,
        "policy_id": policy["policy_id"],
        "roster_id": policy["roster_id"],
        "signer_id": signer_id,
        "signer_role": signer_role,
        "payload": payload,
    }


def _verify_native_main_return_signatures(
    *,
    gold: dict[str, Any],
    policy_path: Path,
    policy: dict[str, Any],
    approval: ApprovedGoldTrustPolicy,
) -> None:
    _validate_approved_policy_record(approval)
    registry_snapshot_sha256 = _approved_registry_snapshot_sha256()
    adjudicator_keys = _native_main_policy_public_keys(
        policy_path,
        policy,
        "adjudicators",
        "external_adjudicator",
    )
    witness_keys = _native_main_policy_public_keys(
        policy_path,
        policy,
        "timestamp_witnesses",
        "external_timestamp_witness",
    )
    adjudication = _required_object(gold, "adjudication")
    for index, submission in enumerate(adjudication.get("submissions", [])):
        if not isinstance(submission, dict):
            raise ContractError("native-main independent gold return submission is invalid")
        signer_id = submission.get("signer_id")
        if signer_id not in adjudicator_keys:
            raise ContractError("native-main independent gold return signer is not approved")
        payload = _native_main_signature_payload(submission)
        _verify_ed25519_signature(
            adjudicator_keys[str(signer_id)],
            _native_main_signature_envelope(
                message_kind="adjudicator_submission",
                approval=approval,
                registry_snapshot_sha256=registry_snapshot_sha256,
                policy=policy,
                signer_id=str(signer_id),
                signer_role="external_adjudicator",
                payload=payload,
            ),
            submission.get("signature_base64"),
        )
    for field, message_kind in (
        ("commitment_witness", "pre_candidate_commitment_witness"),
        ("candidate_start_witness", "candidate_start_witness"),
    ):
        witness = _required_object(gold, field)
        signer_id = witness.get("signer_id")
        if signer_id not in witness_keys:
            raise ContractError("native-main independent gold return witness is not approved")
        payload = _native_main_signature_payload(witness)
        _verify_ed25519_signature(
            witness_keys[str(signer_id)],
            _native_main_signature_envelope(
                message_kind=message_kind,
                approval=approval,
                registry_snapshot_sha256=registry_snapshot_sha256,
                policy=policy,
                signer_id=str(signer_id),
                signer_role="external_timestamp_witness",
                payload=payload,
            ),
            witness.get("signature_base64"),
        )


def _validate_native_main_return_trust_policy(path: Path, expected_sha256: object) -> tuple[dict[str, Any], bool]:
    policy_file = _native_main_return_bound_file(path.parent, path.name, expected_sha256, "trust policy")
    policy = _load_json(policy_file)
    _require_exact_keys(
        policy,
        {
            "schema",
            "policy_id",
            "roster_id",
            "required_independent_submissions",
            "adjudicators",
            "timestamp_witnesses",
            "flashpatch_operator_ids",
            "candidate_operator_ids",
            "revoked_at",
        },
        "native-main independent gold trust policy",
    )
    if policy.get("schema") != INDEPENDENT_GOLD_TRUST_SCHEMA:
        raise ContractError("native-main independent gold trust policy schema is invalid")
    required = policy.get("required_independent_submissions")
    adjudicators = policy.get("adjudicators")
    witnesses = policy.get("timestamp_witnesses")
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or required < 3
        or not isinstance(adjudicators, list)
        or len(adjudicators) < required
        or not isinstance(witnesses, list)
        or len(witnesses) < 2
        or policy.get("revoked_at") is not None
    ):
        raise ContractError("native-main independent gold trust policy is invalid")
    approved = any(
        record.policy_id == policy.get("policy_id")
        and record.policy_sha256 == expected_sha256
        for record in APPROVED_GOLD_TRUST_POLICIES
    )
    return policy, approved


def verify_native_main_independent_gold_return(
    *,
    intake_receipt: Path,
    return_root: Path,
    packet_manifest: Path,
) -> dict[str, Any]:
    """Verify a returned native-main gold package against its blind intake packet.

    This is a narrow G3 bridge.  It proves that returned files bind to the
    blinded native-main packet before the general externally approved
    independent-gold verifier can authorize score inputs.
    """

    intake_file = intake_receipt.resolve(strict=True)
    intake_root = intake_file.parent
    intake_assessment = verify_native_main_blinded_gold_intake(intake_file)
    intake = _load_json(intake_file)
    packet_manifest_file = packet_manifest.resolve(strict=True)
    packet_assessment = verify_native_main_adjudicator_packet_manifest(packet_manifest_file)
    if packet_assessment.get("intake_receipt_sha256") != _sha256_file(intake_file):
        raise ContractError("native-main independent gold return packet manifest intake mismatch")
    return_dir = return_root.resolve(strict=True)
    if not return_dir.is_dir() or return_dir.is_symlink():
        raise ContractError("native-main independent gold return root is invalid")
    manifest_path = return_dir / "native-main-independent-gold-return.json"
    manifest = _load_json(manifest_path)
    _reject_native_main_gold_return_contamination(manifest, "return_manifest")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "status",
            "scoreable",
            "external_claim_authorized",
            "intake_receipt_sha256",
            "adjudicator_packet_manifest_sha256",
            "adjudicator_packet_root_sha256",
            "source_summary_sha256",
            "corpus_commitment_sha256",
            "expected_return_contract_sha256",
            "case_count",
            "gold_receipts",
        },
        "native-main independent gold return manifest",
    )
    if (
        manifest.get("schema") != NATIVE_MAIN_INDEPENDENT_GOLD_RETURN_SCHEMA
        or manifest.get("scoreable") is not False
        or manifest.get("external_claim_authorized") is not False
        or manifest.get("intake_receipt_sha256") != _sha256_file(intake_file)
        or manifest.get("adjudicator_packet_manifest_sha256") != _sha256_file(packet_manifest_file)
        or manifest.get("adjudicator_packet_root_sha256") != packet_assessment["packet_manifest_sha256"]
        or manifest.get("source_summary_sha256") != intake["source_summary_sha256"]
        or manifest.get("corpus_commitment_sha256") != intake["corpus_commitment_sha256"]
        or manifest.get("expected_return_contract_sha256") != intake["expected_return_contract_sha256"]
        or manifest.get("case_count") != intake["case_count"]
    ):
        raise ContractError("native-main independent gold return manifest is invalid")
    if manifest.get("status") not in {"INDEPENDENT_GOLD_RETURNED_UNAPPROVED", "INDEPENDENT_GOLD_VERIFIED"}:
        raise ContractError("native-main independent gold return status is invalid")

    corpus_path = _receipt_bound_file(
        intake_root,
        intake.get("corpus_commitment_path"),
        intake.get("corpus_commitment_sha256"),
        "native-main blinded corpus commitment",
    )
    corpus = _load_json(corpus_path)
    blind_case_ids = corpus.get("blind_case_ids")
    if not isinstance(blind_case_ids, list) or manifest["case_count"] != len(blind_case_ids):
        raise ContractError("native-main independent gold return case count mismatch")
    intake_inputs = {
        entry["blind_case_id"]: entry
        for entry in intake.get("blinded_inputs", [])
        if isinstance(entry, dict) and isinstance(entry.get("blind_case_id"), str)
    }
    if set(intake_inputs) != set(blind_case_ids):
        raise ContractError("native-main independent gold return blind intake binding is invalid")
    entries = manifest.get("gold_receipts")
    if not isinstance(entries, list) or len(entries) != len(blind_case_ids):
        raise ContractError("native-main independent gold return receipt count mismatch")

    approved_count = 0
    seen: set[str] = set()
    trust_policy_approved = True
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError("native-main independent gold return entry is invalid")
        _require_exact_keys(
            entry,
            {
                "blind_case_id",
                "independent_gold_path",
                "independent_gold_sha256",
                "trust_policy_path",
                "trust_policy_sha256",
                "candidate_start_receipt_path",
                "candidate_start_receipt_sha256",
            },
            "native-main independent gold return entry",
        )
        blind_id = entry.get("blind_case_id")
        if blind_id not in blind_case_ids or blind_id in seen:
            raise ContractError("native-main independent gold return blind case identity is invalid")
        seen.add(str(blind_id))
        gold_file = _native_main_return_bound_file(
            return_dir,
            entry.get("independent_gold_path"),
            entry.get("independent_gold_sha256"),
            "independent gold",
        )
        policy_file = _native_main_return_bound_file(
            return_dir,
            entry.get("trust_policy_path"),
            entry.get("trust_policy_sha256"),
            "trust policy",
        )
        candidate_start_file = _native_main_return_bound_file(
            return_dir,
            entry.get("candidate_start_receipt_path"),
            entry.get("candidate_start_receipt_sha256"),
            "candidate start receipt",
        )
        gold = _load_json(gold_file)
        _reject_native_main_gold_return_contamination(gold, str(blind_id))
        _policy, approved = _validate_native_main_return_trust_policy(policy_file, entry.get("trust_policy_sha256"))
        intake_entry = intake_inputs[str(blind_id)]
        input_file = _receipt_bound_file(
            intake_root,
            intake_entry.get("path"),
            intake_entry.get("sha256"),
            "native-main independent gold return blinded input",
        )
        blinded_input = {
            **_load_json(input_file),
            "_entry_sha256": intake_entry["sha256"],
        }
        _validate_native_main_return_gold_skeleton(
            gold=gold,
            blind_id=str(blind_id),
            intake=intake,
            blinded_input=blinded_input,
            trust_policy=_policy,
            trust_policy_sha256=entry.get("trust_policy_sha256"),
            candidate_start_file=candidate_start_file,
        )
        trust_policy_approved = trust_policy_approved and approved
        if approved:
            matching_approvals = [
                record
                for record in APPROVED_GOLD_TRUST_POLICIES
                if record.policy_id == _policy.get("policy_id")
                and record.policy_sha256 == entry.get("trust_policy_sha256")
            ]
            if len(matching_approvals) != 1:
                raise ContractError("native-main independent gold policy approval is ambiguous")
            _verify_native_main_return_signatures(
                gold=gold,
                policy_path=policy_file,
                policy=_policy,
                approval=matching_approvals[0],
            )
            approved_count += 1

    if set(seen) != set(blind_case_ids):
        raise ContractError("native-main independent gold return case set mismatch")
    score_blockers = [
        "candidate_start_witness_missing",
        "same_condition_9_slot_execution_missing",
        "receipt_bound_score_bundle_missing",
    ]
    status = "INDEPENDENT_GOLD_VERIFIED" if approved_count == len(entries) else "INDEPENDENT_GOLD_UNVERIFIED"
    if not trust_policy_approved:
        score_blockers.insert(0, "independent_trust_policy_not_approved")
    return {
        "schema": NATIVE_MAIN_INDEPENDENT_GOLD_RETURN_ASSESSMENT_SCHEMA,
        "status": status,
        "scoreable": False,
        "case_count": len(entries),
        "approved_gold_count": approved_count,
        "intake_receipt_sha256": _sha256_file(intake_file),
        "adjudicator_packet_manifest_sha256": _sha256_file(packet_manifest_file),
        "adjudicator_packet_root_sha256": packet_assessment["packet_manifest_sha256"],
        "source_summary_sha256": intake_assessment["source_summary_sha256"],
        "corpus_commitment_sha256": intake_assessment["corpus_commitment_sha256"],
        "score_blockers": score_blockers,
        "external_claim_authorized": False,
    }


def verify_native_main_candidate_start_gate(
    *,
    intake_receipt: Path,
    return_root: Path,
    packet_manifest: Path,
    start_receipt: Path,
) -> dict[str, Any]:
    """Gate native-main 9-slot start on verified independent gold first."""

    intake_file = intake_receipt.resolve(strict=True)
    return_dir = return_root.resolve(strict=True)
    start_file = start_receipt.resolve(strict=True)
    if not start_file.is_file() or start_file.is_symlink():
        raise ContractError("native-main candidate-start gate receipt is unavailable")
    gold_return = verify_native_main_independent_gold_return(
        intake_receipt=intake_file,
        return_root=return_dir,
        packet_manifest=packet_manifest,
    )
    packet_manifest_file = packet_manifest.resolve(strict=True)
    manifest_path = return_dir / "native-main-independent-gold-return.json"
    start = _load_json(start_file)
    _reject_native_main_gold_return_contamination(start, "candidate_start_gate")
    _require_exact_keys(
        start,
        {
            "schema",
            "status",
            "scoreable",
            "external_claim_authorized",
            "intake_receipt_sha256",
            "adjudicator_packet_manifest_sha256",
            "adjudicator_packet_root_sha256",
            "gold_return_manifest_sha256",
            "source_summary_sha256",
            "corpus_commitment_sha256",
            "detector_population",
            "slot_count",
            "repeat_count_per_detector",
            "request_schema",
            "wall_clock_budget_seconds",
            "retry_rule",
            "process_started_at",
            "process_start_monotonic_ns",
            "state",
            "candidate_outputs_observed",
        },
        "native-main candidate-start gate",
    )
    slot_count = start.get("slot_count")
    repeats = start.get("repeat_count_per_detector")
    budget = start.get("wall_clock_budget_seconds")
    monotonic_ns = start.get("process_start_monotonic_ns")
    if (
        start.get("schema") != NATIVE_MAIN_CANDIDATE_START_GATE_SCHEMA
        or start.get("status") != "CANDIDATE_START_REQUESTED"
        or start.get("scoreable") is not False
        or start.get("external_claim_authorized") is not False
        or start.get("intake_receipt_sha256") != _sha256_file(intake_file)
        or start.get("adjudicator_packet_manifest_sha256") != gold_return["adjudicator_packet_manifest_sha256"]
        or start.get("adjudicator_packet_manifest_sha256") != _sha256_file(packet_manifest_file)
        or start.get("adjudicator_packet_root_sha256") != gold_return["adjudicator_packet_root_sha256"]
        or start.get("gold_return_manifest_sha256") != _sha256_file(manifest_path)
        or start.get("source_summary_sha256") != gold_return["source_summary_sha256"]
        or start.get("corpus_commitment_sha256") != gold_return["corpus_commitment_sha256"]
        or start.get("detector_population") != L7_DIRECT_CANDIDATE_TOOLS
        or slot_count != 9
        or repeats != 3
        or start.get("request_schema") != L7_EXTERNAL_HOST_REQUEST_SCHEMA_V2
        or not isinstance(start.get("retry_rule"), str)
        or not start.get("retry_rule")
        or start.get("state") != "STARTED_OUTPUT_UNOPENED"
        or start.get("candidate_outputs_observed") is not False
        or isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) <= 0.0
        or isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns <= 0
    ):
        raise ContractError("native-main candidate-start gate is invalid")
    started_at = _parse_utc_timestamp(start.get("process_started_at"), "native-main candidate start")
    if started_at > datetime.now(timezone.utc):
        raise ContractError("native-main candidate start is in the future")
    if gold_return.get("status") != "INDEPENDENT_GOLD_VERIFIED" or gold_return.get("approved_gold_count") != gold_return.get("case_count"):
        blockers = list(gold_return.get("score_blockers", []))
        if "independent_gold_not_verified" not in blockers:
            blockers.insert(0, "independent_gold_not_verified")
        return {
            "schema": NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA,
            "status": "CANDIDATE_START_BLOCKED",
            "scoreable": False,
            "reason": "independent_gold_not_verified_before_candidate_start",
            "case_count": gold_return["case_count"],
            "intake_receipt_sha256": _sha256_file(intake_file),
            "gold_return_manifest_sha256": _sha256_file(manifest_path),
            "candidate_start_receipt_sha256": _sha256_file(start_file),
            "detector_population": start["detector_population"],
            "slot_count": slot_count,
            "repeat_count_per_detector": repeats,
            "request_schema": start["request_schema"],
            "score_blockers": blockers,
            "external_claim_authorized": False,
        }
    return {
        "schema": NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA,
        "status": "CANDIDATE_START_WITNESS_VERIFIED",
        "scoreable": False,
        "reason": "candidate_start_verified_but_9_slot_execution_and_score_bundle_missing",
        "case_count": gold_return["case_count"],
        "intake_receipt_sha256": _sha256_file(intake_file),
        "gold_return_manifest_sha256": _sha256_file(manifest_path),
        "candidate_start_receipt_sha256": _sha256_file(start_file),
        "detector_population": start["detector_population"],
        "slot_count": slot_count,
        "repeat_count_per_detector": repeats,
        "request_schema": start["request_schema"],
        "score_blockers": [
            "same_condition_9_slot_execution_missing",
            "receipt_bound_score_bundle_missing",
        ],
        "external_claim_authorized": False,
    }


def _identity(value: object, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@/-]{2,127}", value) is None:
        raise ContractError(f"{context} identity is invalid")
    return value.casefold()


def _canonical_ed25519_spki(public_key_bytes: bytes) -> bytes:
    if not OPENSSL_EXECUTABLE.is_file():
        raise ContractError("Ed25519 verifier is unavailable")
    with tempfile.TemporaryDirectory(prefix="flashpatch-gold-key-") as temporary:
        public_key_path = Path(temporary) / "public-key.pem"
        public_key_path.write_bytes(public_key_bytes)
        try:
            result = subprocess.run(
                [
                    str(OPENSSL_EXECUTABLE),
                    "pkey",
                    "-pubin",
                    "-in",
                    str(public_key_path),
                    "-outform",
                    "DER",
                ],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError("Ed25519 public key normalization failed") from exc
    if result.returncode != 0 or not result.stdout.startswith(bytes.fromhex("302a300506032b6570032100")):
        raise ContractError("approved public key is not Ed25519")
    return result.stdout


def _read_pinned_public_key(policy_path: Path, entry: dict[str, Any], context: str) -> bytes:
    _require_exact_keys(
        entry,
        {
            "identity_id",
            "role",
            "public_key_path",
            "public_key_sha256",
            "ed25519_spki_sha256",
        },
        context,
    )
    _identity(entry.get("identity_id"), context)
    relative = entry.get("public_key_path")
    if not isinstance(relative, str) or not relative:
        raise ContractError(f"{context} public key path is invalid")
    expected = _validate_hash(entry.get("public_key_sha256"), f"{context} public key")
    candidate = policy_path.parent / relative
    root = policy_path.parent.resolve()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{context} public key is unavailable") from exc
    if candidate.is_symlink() or root not in resolved.parents or not resolved.is_file():
        raise ContractError(f"{context} public key escapes trust policy root")
    try:
        key_bytes = resolved.read_bytes()
    except OSError as exc:
        raise ContractError(f"{context} public key is unavailable") from exc
    if hashlib.sha256(key_bytes).hexdigest() != expected:
        raise ContractError(f"{context} public key hash mismatch")
    expected_spki = _validate_hash(entry.get("ed25519_spki_sha256"), f"{context} Ed25519 SPKI")
    if hashlib.sha256(_canonical_ed25519_spki(key_bytes)).hexdigest() != expected_spki:
        raise ContractError(f"{context} Ed25519 SPKI hash mismatch")
    return key_bytes


def _load_gold_trust_policy(
    policy_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], bytes]], dict[str, tuple[dict[str, Any], bytes]]]:
    expected = _validate_hash(expected_sha256, "independent gold trust policy")
    try:
        policy_bytes = policy_path.read_bytes()
    except OSError as exc:
        raise ContractError("independent gold trust policy is unavailable") from exc
    if policy_path.is_symlink() or hashlib.sha256(policy_bytes).hexdigest() != expected:
        raise ContractError("independent gold trust policy is not externally pinned")
    policy = _load_json_bytes(policy_bytes, "independent gold trust policy")
    _require_exact_keys(
        policy,
        {
            "schema",
            "policy_id",
            "roster_id",
            "required_independent_submissions",
            "flashpatch_operator_ids",
            "candidate_operator_ids",
            "adjudicators",
            "witnesses",
        },
        "independent gold trust policy",
    )
    if policy.get("schema") != INDEPENDENT_GOLD_TRUST_SCHEMA:
        raise ContractError("independent gold trust policy schema is invalid")
    _identity(policy.get("policy_id"), "trust policy")
    _identity(policy.get("roster_id"), "adjudicator roster")
    required = policy.get("required_independent_submissions")
    if isinstance(required, bool) or not isinstance(required, int) or required < 3:
        raise ContractError("independent gold trust policy requires fewer than three adjudicators")

    operator_ids: set[str] = set()
    for field in ("flashpatch_operator_ids", "candidate_operator_ids"):
        values = policy.get(field)
        if not isinstance(values, list) or not values:
            raise ContractError(f"independent gold trust policy {field} is invalid")
        normalized = {_identity(value, field) for value in values}
        if len(normalized) != len(values) or operator_ids & normalized:
            raise ContractError("FlashPatch and candidate operator identities overlap")
        operator_ids.update(normalized)

    role_maps: list[dict[str, tuple[dict[str, Any], bytes]]] = []
    used_identities = set(operator_ids)
    used_keys: set[str] = set()
    for field, expected_role in (("adjudicators", "external_adjudicator"), ("witnesses", "external_timestamp_witness")):
        entries = policy.get(field)
        if not isinstance(entries, list) or not entries:
            raise ContractError(f"independent gold trust policy {field} is invalid")
        role_map: dict[str, tuple[dict[str, Any], bytes]] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ContractError(f"independent gold trust policy {field} entry is invalid")
            key_bytes = _read_pinned_public_key(policy_path, entry, f"{field}[{index}]")
            identity = _identity(entry.get("identity_id"), field)
            key_hash = str(entry["ed25519_spki_sha256"])
            if entry.get("role") != expected_role:
                raise ContractError(f"independent gold trust policy {field} role is invalid")
            if identity in used_identities or key_hash in used_keys:
                raise ContractError("adjudicator, witness, and operator identities are not disjoint")
            used_identities.add(identity)
            used_keys.add(key_hash)
            role_map[identity] = (entry, key_bytes)
        role_maps.append(role_map)
    adjudicators, witnesses = role_maps
    if len(adjudicators) < required or len(witnesses) < 2:
        raise ContractError("independent gold trust policy roster is incomplete")
    return policy, adjudicators, witnesses


def _verify_ed25519_signature(public_key_bytes: bytes, payload: object, signature: object) -> None:
    if not isinstance(signature, str):
        raise ContractError("independent gold signature is missing")
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError("independent gold signature encoding is invalid") from exc
    if len(signature_bytes) != 64:
        raise ContractError("independent gold signature length is invalid")
    if not OPENSSL_EXECUTABLE.is_file():
        raise ContractError("Ed25519 verifier is unavailable")
    with tempfile.TemporaryDirectory(prefix="flashpatch-gold-signature-") as temporary:
        public_key_path = Path(temporary) / "public-key.pem"
        input_path = Path(temporary) / "payload.json"
        signature_path = Path(temporary) / "signature.bin"
        public_key_path.write_bytes(public_key_bytes)
        input_path.write_bytes(_canonical_bytes(payload))
        signature_path.write_bytes(signature_bytes)
        try:
            result = subprocess.run(
                [
                    str(OPENSSL_EXECUTABLE),
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_key_path),
                    "-rawin",
                    "-in",
                    str(input_path),
                    "-sigfile",
                    str(signature_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError("Ed25519 verification failed to execute") from exc
    if result.returncode != 0:
        raise ContractError("independent gold signature is invalid")


def _signature_envelope(
    *,
    message_kind: str,
    approval: ApprovedGoldTrustPolicy,
    registry_snapshot_sha256: str,
    policy: dict[str, Any],
    signer_id: str,
    signer_role: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": "FlashPatch/L7/independent-gold/v2",
        "message_kind": message_kind,
        "trust_policy_sha256": approval.policy_sha256,
        "approval_record_sha256": approval.approval_record_sha256,
        "external_timestamp_token_sha256": approval.external_timestamp_token_sha256,
        "registry_snapshot_sha256": registry_snapshot_sha256,
        "verifier_source_commit": approval.verifier_source_commit,
        "openssl_binary_sha256": approval.openssl_binary_sha256,
        "policy_id": policy["policy_id"],
        "roster_id": policy["roster_id"],
        "signer_id": signer_id,
        "signer_role": signer_role,
        "payload": payload,
    }


def _validate_blinded_gold_input(value: bytes, receipt: dict[str, Any], freeze: dict[str, Any]) -> str:
    blinded = _load_json_bytes(value, "blinded gold input")
    _require_exact_keys(
        blinded,
        {
            "schema",
            "blind_case_id",
            "corpus_manifest_sha256",
            "source_tree_sha256",
            "trace_sha256",
            "renderer_rgb_raw_sha256",
            "timestamps_sha256",
            "frame_count",
            "color_space",
            "rubric_sha256",
        },
        "blinded gold input",
    )
    if blinded.get("schema") != "flashpatch-l7-blinded-gold-input-v1":
        raise ContractError("blinded gold input schema is invalid")
    blind_case_id = blinded.get("blind_case_id")
    if not isinstance(blind_case_id, str) or re.fullmatch(r"blind-[0-9a-f]{32}", blind_case_id) is None:
        raise ContractError("blinded gold input identity is invalid")
    expected = {
        "corpus_manifest_sha256": _required_object(receipt, "pre_candidate_commitment").get("corpus_manifest_sha256"),
        "source_tree_sha256": freeze.get("source_tree_sha256"),
        "trace_sha256": freeze.get("trace_sha256"),
        "renderer_rgb_raw_sha256": freeze.get("renderer_rgb_raw_sha256"),
        "timestamps_sha256": freeze.get("timestamps_sha256"),
        "frame_count": freeze.get("frame_count"),
        "color_space": freeze.get("color_space"),
        "rubric_sha256": _required_object(receipt, "standard_profile").get("rubric_sha256"),
    }
    if any(blinded.get(field) != value for field, value in expected.items()):
        raise ContractError("blinded gold input does not match frozen evidence")
    return blind_case_id


def _validate_gold_corpus_manifest(value: bytes, blind_case_id: str) -> datetime:
    corpus = _load_json_bytes(value, "blinded gold corpus manifest")
    _require_exact_keys(
        corpus,
        {
            "schema",
            "discovery_manifest_sha256",
            "blind_case_ids",
            "selection_rule_sha256",
            "stop_rule_sha256",
            "frozen_at",
            "candidate_outputs_observed",
        },
        "blinded gold corpus manifest",
    )
    if corpus.get("schema") != "flashpatch-l7-blinded-corpus-commitment-v1":
        raise ContractError("blinded gold corpus manifest schema is invalid")
    for field in ("discovery_manifest_sha256", "selection_rule_sha256", "stop_rule_sha256"):
        _validate_hash(corpus.get(field), f"blinded gold corpus {field}")
    blind_case_ids = corpus.get("blind_case_ids")
    if (
        not isinstance(blind_case_ids, list)
        or not blind_case_ids
        or any(not isinstance(value, str) or re.fullmatch(r"blind-[0-9a-f]{32}", value) is None for value in blind_case_ids)
        or len(blind_case_ids) != len(set(blind_case_ids))
        or blind_case_id not in blind_case_ids
        or corpus.get("candidate_outputs_observed") is not False
    ):
        raise ContractError("blinded gold corpus contains post-candidate or unblinded evidence")
    return _parse_utc_timestamp(corpus.get("frozen_at"), "blinded gold corpus")


def _validate_gold_rubric(value: bytes, profile: dict[str, Any]) -> str:
    rubric = _load_json_bytes(value, "independent gold rubric")
    _require_exact_keys(
        rubric,
        {"schema", "profile_id", "source_url", "normative_rule_sha256"},
        "independent gold rubric",
    )
    if (
        rubric.get("schema") != "flashpatch-l7-gold-rubric-v1"
        or rubric.get("profile_id") != profile.get("id")
        or rubric.get("source_url") != profile.get("source_url")
    ):
        raise ContractError("independent gold rubric identity is invalid")
    return _validate_hash(rubric.get("normative_rule_sha256"), "independent gold normative rule")


def _validate_candidate_start_receipt(
    value: bytes,
    *,
    case_id: str,
    freeze: dict[str, Any],
    commitment_sha256: str,
    approval: ApprovedGoldTrustPolicy,
    registry_snapshot_sha256: str,
) -> tuple[str, datetime]:
    start = _load_json_bytes(value, "candidate start receipt")
    _require_exact_keys(
        start,
        {
            "schema",
            "case_id",
            "run_id",
            "candidate_tools",
            "renderer_rgb_raw_sha256",
            "timestamps_sha256",
            "pre_candidate_commitment_sha256",
            "trust_policy_id",
            "approval_record_sha256",
            "external_timestamp_token_sha256",
            "registry_snapshot_sha256",
            "verifier_source_commit",
            "openssl_binary_sha256",
            "process_started_at",
            "process_start_monotonic_ns",
            "state",
        },
        "candidate start receipt",
    )
    run_id = start.get("run_id")
    if (
        start.get("schema") != "flashpatch-l7-candidate-start-receipt-v1"
        or start.get("case_id") != case_id
        or not isinstance(run_id, str)
        or re.fullmatch(r"run-[0-9a-f]{32}", run_id) is None
        or start.get("candidate_tools") != L7_DIRECT_CANDIDATE_TOOLS
        or start.get("renderer_rgb_raw_sha256") != freeze.get("renderer_rgb_raw_sha256")
        or start.get("timestamps_sha256") != freeze.get("timestamps_sha256")
        or start.get("pre_candidate_commitment_sha256") != commitment_sha256
        or start.get("trust_policy_id") != approval.policy_id
        or start.get("approval_record_sha256") != approval.approval_record_sha256
        or start.get("external_timestamp_token_sha256") != approval.external_timestamp_token_sha256
        or start.get("registry_snapshot_sha256") != registry_snapshot_sha256
        or start.get("verifier_source_commit") != approval.verifier_source_commit
        or start.get("openssl_binary_sha256") != approval.openssl_binary_sha256
        or start.get("state") != "STARTED_OUTPUT_UNOPENED"
    ):
        raise ContractError("candidate start receipt is not bound to the frozen run")
    monotonic_ns = start.get("process_start_monotonic_ns")
    if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns <= 0:
        raise ContractError("candidate start monotonic clock is invalid")
    started_at = _parse_utc_timestamp(start.get("process_started_at"), "candidate process start")
    if started_at > datetime.now(timezone.utc):
        raise ContractError("candidate process start is in the future")
    return run_id, started_at


def _validate_independent_gold_strict(
    path: Path,
    trust_policy_path: Path,
    approval: ApprovedGoldTrustPolicy,
) -> dict[str, Any]:
    """Strict G3 verifier whose root of trust is outside the gold package."""
    approved_at, policy_valid_from, policy_valid_until, approval_record = _validate_approved_policy_record(approval)
    registry_snapshot_sha256 = _approved_registry_snapshot_sha256()
    receipt = _load_json(path)
    _require_exact_keys(
        receipt,
        {
            "schema",
            "case_id",
            "case_class",
            "claim_tier",
            "controlled_mutation",
            "case_freeze",
            "standard_profile",
            "gold_authority",
            "pre_candidate_commitment",
            "commitment_witness",
            "adjudication",
            "candidate_start_witness",
            "artifacts",
        },
        "independent gold receipt",
    )
    if receipt.get("schema") != INDEPENDENT_GOLD_SCHEMA:
        raise ContractError("invalid independent gold schema")
    case_id = receipt.get("case_id")
    case_class = receipt.get("case_class")
    claim_tier = receipt.get("claim_tier")
    if not isinstance(case_id, str) or not case_id or case_class not in GOLD_CASE_CLASSES or claim_tier not in GOLD_CLAIM_TIERS:
        raise ContractError("invalid independent gold case identity")
    if (case_class == "natural_external") != (claim_tier == "L9_ELIGIBLE"):
        raise ContractError("gold case class and claim tier conflict")
    if case_class == "natural_external" and receipt.get("controlled_mutation") is not False:
        raise ContractError("natural external gold must explicitly reject controlled mutation")
    if case_class == "controlled_external" and receipt.get("controlled_mutation") is not True:
        raise ContractError("controlled external gold must be explicitly labelled")

    freeze = _required_object(receipt, "case_freeze")
    _require_exact_keys(
        freeze,
        {
            "public_repository_url",
            "source_revision",
            "license",
            "project_subpath",
            "source_tree_sha256",
            "trace_sha256",
            "renderer_execution_receipt_sha256",
            "renderer_rgb_raw_sha256",
            "timestamps_sha256",
            "frame_count",
            "color_space",
        },
        "gold case freeze",
    )
    for field in ("public_repository_url", "source_revision", "license", "project_subpath", "color_space"):
        if not isinstance(freeze.get(field), str) or not freeze[field]:
            raise ContractError(f"gold case freeze missing {field}")
    if re.fullmatch(r"[0-9a-f]{40}", freeze["source_revision"]) is None:
        raise ContractError("gold case freeze source revision is not immutable")
    if freeze["color_space"] != "sRGB_BT709":
        raise ContractError("gold case freeze color space mismatch")
    if isinstance(freeze.get("frame_count"), bool) or not isinstance(freeze.get("frame_count"), int) or freeze["frame_count"] <= 0:
        raise ContractError("gold case freeze frame count invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("independent gold artifacts missing")
    _require_exact_keys(artifacts, set(GOLD_ARTIFACT_FIELDS), "independent gold artifacts")
    profile = _required_object(receipt, "standard_profile")
    _require_exact_keys(profile, {"id", "source_url", "rubric_sha256"}, "independent gold standard profile")
    merged_hashes = {
        **{field: freeze.get(field) for field in GOLD_ARTIFACT_FIELDS[:5]},
        "rubric_sha256": profile.get("rubric_sha256"),
        "corpus_manifest_sha256": _required_object(receipt, "pre_candidate_commitment").get("corpus_manifest_sha256"),
        "blinded_gold_input_sha256": _required_object(receipt, "pre_candidate_commitment").get("blinded_gold_input_sha256"),
        "candidate_start_receipt_sha256": _required_object(receipt, "candidate_start_witness").get("candidate_start_receipt_sha256"),
    }
    verified_artifacts = _verify_hash_bound_artifacts(
        merged_hashes,
        artifacts,
        path,
        fields=GOLD_ARTIFACT_FIELDS,
    )
    if case_class == "natural_external":
        _validate_natural_renderer_execution(
            verified_artifacts["renderer_execution_receipt_sha256"],
            freeze,
        )

    if (
        profile.get("id") != "wcag22-g19-v1"
        or profile.get("source_url") != "https://www.w3.org/WAI/WCAG22/Techniques/general/G19.html"
    ):
        raise ContractError("independent gold standard profile is invalid")
    blind_case_id = _validate_blinded_gold_input(
        verified_artifacts["blinded_gold_input_sha256"],
        receipt,
        freeze,
    )
    corpus_frozen_at = _validate_gold_corpus_manifest(
        verified_artifacts["corpus_manifest_sha256"],
        blind_case_id,
    )
    normative_rule_sha256 = _validate_gold_rubric(
        verified_artifacts["rubric_sha256"],
        profile,
    )
    if normative_rule_sha256 != approval.normative_rule_sha256:
        raise ContractError("independent gold rubric is not in the approved registry")
    capture_start, capture_end = _validate_gold_timestamps(
        verified_artifacts["timestamps_sha256"],
        int(freeze["frame_count"]),
    )

    policy, approved_adjudicators, approved_witnesses = _load_gold_trust_policy(
        trust_policy_path,
        approval.policy_sha256,
    )
    policy_identities = set(approved_adjudicators) | set(approved_witnesses)
    policy_identities.update(
        _identity(value, "policy operator")
        for field in ("flashpatch_operator_ids", "candidate_operator_ids")
        for value in policy[field]
    )
    if {
        _identity(approval.approval_authority_id, "approval authority"),
        _identity(approval.timestamp_authority_id, "timestamp authority"),
    } & policy_identities:
        raise ContractError("approval authorities overlap adjudicator, witness, or operator identity")
    policy_spki_hashes = {
        str(entry["ed25519_spki_sha256"])
        for entry, _ in (*approved_adjudicators.values(), *approved_witnesses.values())
    }
    if {
        approval.approval_authority_spki_sha256,
        approval.timestamp_authority_spki_sha256,
    } & policy_spki_hashes:
        raise ContractError("approval authority key overlaps adjudicator or witness key")
    if policy.get("roster_id") != approval.roster_id or approval_record.get("roster_id") != approval.roster_id:
        raise ContractError("approved roster identity does not match trust policy")
    authority = _required_object(receipt, "gold_authority")
    _require_exact_keys(
        authority,
        {
            "kind",
            "candidate_tools_excluded",
            "trust_policy_id",
            "roster_id",
            "required_independent_submissions",
            "flashpatch_operator_ids",
            "candidate_operator_ids",
        },
        "independent gold authority",
    )
    if authority.get("kind") != "independent_adjudication":
        raise ContractError("independent gold authority is invalid")
    excluded = authority.get("candidate_tools_excluded")
    if (
        not isinstance(excluded, list)
        or len(excluded) != len(GOLD_EXCLUDED_TOOLS)
        or not all(isinstance(value, str) for value in excluded)
        or set(excluded) != GOLD_EXCLUDED_TOOLS
    ):
        raise ContractError("independent gold must exclude every candidate tool")
    policy_bindings = {
        "trust_policy_id": policy.get("policy_id"),
        "roster_id": policy.get("roster_id"),
        "required_independent_submissions": policy.get("required_independent_submissions"),
        "flashpatch_operator_ids": policy.get("flashpatch_operator_ids"),
        "candidate_operator_ids": policy.get("candidate_operator_ids"),
    }
    for field, expected_value in policy_bindings.items():
        if authority.get(field) != expected_value:
            raise ContractError("independent gold authority does not match the externally pinned roster")

    commitment = _required_object(receipt, "pre_candidate_commitment")
    _require_exact_keys(
        commitment,
        {
            "schema",
            "case_id",
            "corpus_manifest_sha256",
            "blinded_gold_input_sha256",
            "rubric_sha256",
            "gold_decision_commitment_sha256",
            "committed_at",
        },
        "pre-candidate gold commitment",
    )
    if commitment.get("schema") != "flashpatch-l7-pre-candidate-gold-commitment-v1" or commitment.get("case_id") != case_id:
        raise ContractError("pre-candidate gold commitment identity is invalid")
    for field in ("corpus_manifest_sha256", "blinded_gold_input_sha256", "rubric_sha256", "gold_decision_commitment_sha256"):
        _validate_hash(commitment.get(field), f"pre-candidate gold commitment {field}")
    if commitment.get("rubric_sha256") != profile.get("rubric_sha256"):
        raise ContractError("pre-candidate gold commitment rubric mismatch")
    committed_at = _parse_utc_timestamp(commitment.get("committed_at"), "pre-candidate gold commitment")
    if not (approved_at <= policy_valid_from <= corpus_frozen_at <= committed_at):
        raise ContractError("corpus or gold commitment predates approved policy or is post-selected")
    commitment_sha256 = hashlib.sha256(_canonical_bytes(commitment)).hexdigest()

    adjudication = _required_object(receipt, "adjudication")
    _require_exact_keys(
        adjudication,
        {"method", "resolution", "result", "intervals", "opening_nonce_base64", "submissions"},
        "independent gold adjudication",
    )
    submissions = adjudication.get("submissions")
    decision = adjudication.get("result")
    if adjudication.get("method") != "three-independent-submissions-plus-resolution" or decision not in {"SAFE", "HAZARDOUS"}:
        raise ContractError("independent gold adjudication is invalid")
    required_submissions = policy["required_independent_submissions"]
    if adjudication.get("resolution") != "unanimous" or not isinstance(submissions, list) or len(submissions) < required_submissions:
        raise ContractError("independent gold lacks required unanimous submissions")
    resolution_intervals = _validate_gold_intervals(
        adjudication.get("intervals"),
        str(decision),
        "independent gold resolution",
        capture_start=capture_start,
        capture_end=capture_end,
    )
    gold_decision = {"decision": decision, "intervals": resolution_intervals}
    opening_nonce = adjudication.get("opening_nonce_base64")
    if not isinstance(opening_nonce, str):
        raise ContractError("independent gold decision opening is missing")
    try:
        opening_nonce_bytes = base64.b64decode(opening_nonce, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContractError("independent gold decision opening is invalid") from exc
    if len(opening_nonce_bytes) != 32:
        raise ContractError("independent gold decision opening is invalid")
    opened_decision_hash = hashlib.sha256(
        b"FlashPatch/L7/gold-decision/v1\x00"
        + opening_nonce_bytes
        + _canonical_bytes(gold_decision)
    ).hexdigest()
    if opened_decision_hash != commitment["gold_decision_commitment_sha256"]:
        raise ContractError("pre-candidate gold decision commitment mismatch")

    used_adjudicators: set[str] = set()
    submission_times: list[datetime] = []
    for submission in submissions:
        if not isinstance(submission, dict):
            raise ContractError("independent gold submission is invalid")
        _require_exact_keys(
            submission,
            {
                "adjudicator_id",
                "blinded_gold_input_sha256",
                "pre_candidate_commitment_sha256",
                "decision",
                "intervals",
                "signed_at",
                "submission_payload_sha256",
                "signature_algorithm",
                "signature_base64",
            },
            "independent gold submission",
        )
        adjudicator_id = _identity(submission.get("adjudicator_id"), "adjudicator")
        if adjudicator_id in used_adjudicators or adjudicator_id not in approved_adjudicators:
            raise ContractError("independent gold raters are not independent")
        used_adjudicators.add(adjudicator_id)
        if submission.get("blinded_gold_input_sha256") != commitment["blinded_gold_input_sha256"] or submission.get("pre_candidate_commitment_sha256") != commitment_sha256:
            raise ContractError("independent gold submission is not bound to blinded input and commitment")
        if submission.get("decision") != decision:
            raise ContractError("independent gold submissions are not unanimous")
        intervals = _validate_gold_intervals(
            submission.get("intervals"),
            str(decision),
            "independent gold submission",
            capture_start=capture_start,
            capture_end=capture_end,
        )
        if intervals != resolution_intervals:
            raise ContractError("independent gold submission intervals are not unanimous")
        signed_at = _parse_utc_timestamp(submission.get("signed_at"), "independent gold submission")
        submission_times.append(signed_at)
        signed_payload = {
            key: value
            for key, value in submission.items()
            if key not in {"submission_payload_sha256", "signature_algorithm", "signature_base64"}
        }
        if hashlib.sha256(_canonical_bytes(signed_payload)).hexdigest() != submission.get("submission_payload_sha256"):
            raise ContractError("independent gold submission payload hash mismatch")
        if submission.get("signature_algorithm") != "Ed25519":
            raise ContractError("independent gold signature algorithm is invalid")
        _verify_ed25519_signature(
            approved_adjudicators[adjudicator_id][1],
            _signature_envelope(
                message_kind="adjudicator_submission",
                approval=approval,
                registry_snapshot_sha256=registry_snapshot_sha256,
                policy=policy,
                signer_id=str(submission["adjudicator_id"]),
                signer_role="external_adjudicator",
                payload=signed_payload,
            ),
            submission.get("signature_base64"),
        )

    witness_ids: set[str] = set()
    commitment_witness = _required_object(receipt, "commitment_witness")
    _require_exact_keys(
        commitment_witness,
        {
            "witness_id",
            "pre_candidate_commitment_sha256",
            "attested_at",
            "signature_algorithm",
            "signature_base64",
        },
        "commitment witness",
    )
    commitment_witness_id = _identity(commitment_witness.get("witness_id"), "commitment witness")
    if commitment_witness_id not in approved_witnesses:
        raise ContractError("commitment witness is not externally approved")
    witness_ids.add(commitment_witness_id)
    if commitment_witness.get("pre_candidate_commitment_sha256") != commitment_sha256:
        raise ContractError("commitment witness substitution detected")
    commitment_attested_at = _parse_utc_timestamp(commitment_witness.get("attested_at"), "commitment witness")
    if commitment_attested_at < committed_at:
        raise ContractError("commitment witness predates commitment")
    if commitment_witness.get("signature_algorithm") != "Ed25519":
        raise ContractError("commitment witness signature algorithm is invalid")
    commitment_witness_payload = {
        key: value
        for key, value in commitment_witness.items()
        if key not in {"signature_algorithm", "signature_base64"}
    }
    _verify_ed25519_signature(
        approved_witnesses[commitment_witness_id][1],
        _signature_envelope(
            message_kind="pre_candidate_commitment_witness",
            approval=approval,
            registry_snapshot_sha256=registry_snapshot_sha256,
            policy=policy,
            signer_id=str(commitment_witness["witness_id"]),
            signer_role="external_timestamp_witness",
            payload=commitment_witness_payload,
        ),
        commitment_witness.get("signature_base64"),
    )

    candidate_start = _required_object(receipt, "candidate_start_witness")
    _require_exact_keys(
        candidate_start,
        {
            "witness_id",
            "case_id",
            "run_id",
            "pre_candidate_commitment_sha256",
            "gold_decision_commitment_sha256",
            "candidate_start_receipt_sha256",
            "candidate_started_at",
            "signature_algorithm",
            "signature_base64",
        },
        "candidate-start witness",
    )
    start_witness_id = _identity(candidate_start.get("witness_id"), "candidate-start witness")
    if start_witness_id not in approved_witnesses or start_witness_id in witness_ids:
        raise ContractError("candidate-start witness is not independently approved")
    if (
        candidate_start.get("case_id") != case_id
        or candidate_start.get("pre_candidate_commitment_sha256") != commitment_sha256
        or candidate_start.get("gold_decision_commitment_sha256") != commitment["gold_decision_commitment_sha256"]
    ):
        raise ContractError("candidate-start witness substitution detected")
    candidate_run_id, process_started_at = _validate_candidate_start_receipt(
        verified_artifacts["candidate_start_receipt_sha256"],
        case_id=str(case_id),
        freeze=freeze,
        commitment_sha256=commitment_sha256,
        approval=approval,
        registry_snapshot_sha256=registry_snapshot_sha256,
    )
    if candidate_start.get("run_id") != candidate_run_id:
        raise ContractError("candidate-start witness run identity mismatch")
    candidate_started_at = _parse_utc_timestamp(candidate_start.get("candidate_started_at"), "candidate start")
    if candidate_started_at != process_started_at:
        raise ContractError("candidate-start witness time does not match process receipt")
    if policy_valid_until is not None and candidate_started_at >= policy_valid_until:
        raise ContractError("approved trust policy expired before candidate start")
    if not (
        committed_at <= commitment_attested_at < candidate_started_at
        and all(committed_at <= signed_at < candidate_started_at for signed_at in submission_times)
    ):
        raise ContractError("gold was not sealed before candidate start")
    if candidate_start.get("signature_algorithm") != "Ed25519":
        raise ContractError("candidate-start witness signature algorithm is invalid")
    candidate_start_payload = {
        key: value
        for key, value in candidate_start.items()
        if key not in {"signature_algorithm", "signature_base64"}
    }
    _verify_ed25519_signature(
        approved_witnesses[start_witness_id][1],
        _signature_envelope(
            message_kind="candidate_start_witness",
            approval=approval,
            registry_snapshot_sha256=registry_snapshot_sha256,
            policy=policy,
            signer_id=str(candidate_start["witness_id"]),
            signer_role="external_timestamp_witness",
            payload=candidate_start_payload,
        ),
        candidate_start.get("signature_base64"),
    )
    timestamps_payload = _load_json_bytes(
        verified_artifacts["timestamps_sha256"],
        "gold timestamps projection",
    )
    return {
        "schema": "flashpatch-l7-independent-gold-projection-v1",
        "gold_verified": True,
        "case_id": case_id,
        "case_class": case_class,
        "claim_tier": claim_tier,
        "controlled_mutation": receipt["controlled_mutation"],
        "case_freeze": dict(freeze),
        "decision": decision,
        "intervals": resolution_intervals,
        "timestamps_seconds": timestamps_payload["timestamps_seconds"],
        "receipt": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "trust_policy": {
            "path": str(trust_policy_path.resolve()),
            "sha256": approval.policy_sha256,
            "policy_id": approval.policy_id,
            "registry_snapshot_sha256": registry_snapshot_sha256,
        },
        "league_score_authorized": False,
    }


def project_independent_gold(
    path: Path,
    *,
    trust_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Return score inputs only after the external-root G3 contract verifies."""
    if trust_policy_path is None:
        raise ContractError("independent gold trust policy is required")
    policy_bytes = trust_policy_path.read_bytes()
    policy = _load_json_bytes(policy_bytes, "independent gold trust policy")
    policy_id = policy.get("policy_id")
    matching = [
        record
        for record in APPROVED_GOLD_TRUST_POLICIES
        if record.policy_id == policy_id
    ]
    if len(matching) != 1:
        raise ContractError("independent gold policy is not externally approved")
    approval = matching[0]
    if hashlib.sha256(policy_bytes).hexdigest() != approval.policy_sha256:
        raise ContractError("independent gold policy hash is not externally approved")
    return _validate_independent_gold_strict(path, trust_policy_path, approval)


def validate_independent_gold(
    path: Path,
    *,
    trust_policy_path: Path | None = None,
) -> str:
    """Classify G3 evidence, never treating a receipt-contained identity as trust.

    A receipt is scoreable only when its policy ID and hash were compiled into
    the deployment's external-approval registry before candidate execution.
    Evidence submitters can point at a policy artifact, but cannot choose its
    expected hash. Missing or invalid evidence collapses to one non-scoreable
    result; it never becomes a weak form of independent gold.
    """
    if trust_policy_path is None:
        return INDEPENDENT_GOLD_NOT_VERIFIED
    try:
        projection = project_independent_gold(
            path,
            trust_policy_path=trust_policy_path,
        )
        return (
            "VALID independent_gold "
            f"case_id={projection['case_id']} tier={projection['claim_tier']}"
        )
    except Exception:
        return INDEPENDENT_GOLD_NOT_VERIFIED


def verify_chain(case_path: Path) -> str:
    """Validate the L6 controlled-mutation chain without judging upstream code.

    The case declares the original public project separately.  Only the explicitly
    labelled mutation may produce a PASS; its upstream record remains SAFE or
    INCONCLUSIVE and is never classified as an upstream defect.
    """
    case = _load_json(case_path)
    source = _required_object(case, "source")
    expected_source = {
        "repository_url": "https://github.com/godotengine/godot-demo-projects",
        "source_revision": "52e3004",
        "license": "MIT",
        "project_path": "2d/pong",
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ContractError(f"controlled source {field} mismatch")
    if case.get("controlled_mutation") is not True:
        raise ContractError("L6 requires controlled_mutation=true")
    upstream = _required_object(case, "original_upstream")
    if upstream.get("decision") not in {"SAFE", "INCONCLUSIVE"}:
        raise ContractError("original upstream must be SAFE or INCONCLUSIVE")
    if upstream.get("upstream_defect") is not False:
        raise ContractError("controlled mutation must not be an upstream defect")

    evidence_root = case_path.parents[1]
    freeze_path = evidence_root / "freeze-manifest.json"
    factual_path = evidence_root / "factual-receipt.json"
    candidate_path = evidence_root / "candidate-receipt.json"
    chain_path = evidence_root / "chain-receipt.json"
    freeze = _load_json(freeze_path)
    factual = _load_json(factual_path)
    candidate = _load_json(candidate_path)
    chain = _load_json(chain_path)
    for field in ("trace_hash", "input_hash", "environment_digest", "source_snapshot_sha256", "mutation_diff_sha256"):
        _validate_hash(freeze.get(field), field)
    if freeze.get("source") != source:
        raise ContractError("source binding mismatch")
    if freeze.get("controlled_mutation") is not True:
        raise ContractError("freeze manifest lacks controlled mutation label")

    verify_receipt(factual_path)
    verify_receipt(candidate_path)
    if not isinstance(factual.get("artifacts"), dict) or not isinstance(candidate.get("artifacts"), dict):
        raise ContractError("renderer artifacts unavailable")
    if factual.get("decision") != "PASS" or candidate.get("decision") != "PASS":
        raise ContractError("controlled chain requires PASS receipts")
    for receipt in (factual, candidate):
        if receipt.get("controlled_mutation") is not True:
            raise ContractError("receipt lacks controlled mutation label")
        if receipt.get("trace_sha256") != freeze["trace_hash"] or receipt.get("input_sha256") != freeze["input_hash"]:
            raise ContractError("trace or input binding mismatch")
        if receipt.get("source_snapshot_sha256") != freeze["source_snapshot_sha256"]:
            raise ContractError("source snapshot mismatch")
        if receipt.get("diff_sha256") != freeze["mutation_diff_sha256"]:
            raise ContractError("mutation diff mismatch")
    patch = candidate.get("patch")
    edits = patch.get("edits") if isinstance(patch, dict) else None
    if not isinstance(edits, list) or len(edits) != 1 or edits[0].get("exported") is not True:
        raise ContractError("patch_count_not_one")
    if factual.get("factual_value", 0) < 1.0 or candidate.get("candidate_value") != 0.0:
        raise ContractError("risk threshold mismatch")
    binding_fields = ("source_file", "source_line", "exported_parameter")
    if any(factual.get(field) != candidate.get(field) for field in binding_fields):
        raise ContractError("runtime source attribution mismatch")
    if factual.get("action_sequence_sha256") != candidate.get("action_sequence_sha256"):
        raise ContractError("exact action sequence mismatch")
    if factual.get("gameplay_state_fingerprint") != candidate.get("gameplay_state_fingerprint"):
        raise ContractError("invariant mismatch")
    if chain.get("decision") != "PASS" or chain.get("claimable") is not False:
        raise ContractError("invalid L6 chain decision")
    if chain.get("controlled_mutation") is not True or chain.get("source_revision") != source["source_revision"]:
        raise ContractError("invalid L6 chain provenance")
    return "PASS controlled_mutation"


def compare_detectors(path: Path) -> tuple[int, str]:
    """Refuse an L7 win until pinned comparators have hash-bound raw runs."""
    if not path.is_file():
        return 1, "NOT_CLAIMABLE"
    try:
        manifest = _load_json(path)
    except ContractError:
        return 1, "NOT_CLAIMABLE"
    required = ("comparators", "raw_runs", "aggregate")
    if manifest.get("frozen") is not True or any(field not in manifest for field in required):
        return 1, "NOT_CLAIMABLE"
    comparators = manifest.get("comparators")
    if not isinstance(comparators, list) or len(comparators) < 3:
        return 1, "NOT_CLAIMABLE"
    # A scoreable result requires L7's later actual runner.  The L5 contract
    # intentionally has no authority to convert absent evidence into a win.
    return 1, "NOT_CLAIMABLE"


def _league_paths(run: Path) -> tuple[Path, Path, Path, Path]:
    return run / "freeze-manifest.json", run / "cell-packets.json", run / "private" / "identity-mapping.json", run / "aggregate.json"


def _validate_league(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze, packets, identities, _ = _league_paths(run)
    manifest, cells, mapping = _load_json(freeze), _load_json(packets), _load_json(identities)
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        raise ContractError("league requires exactly three balanced lanes")
    if not isinstance(cells.get("cells"), list) or len(cells["cells"]) != 3:
        raise ContractError("league requires three cell packets")
    leaked = json.dumps(cells, sort_keys=True).lower()
    identity_values = mapping.get("identities")
    if not isinstance(identity_values, dict):
        raise ContractError("missing private identity mapping")
    for identity in identity_values.values():
        if isinstance(identity, str) and identity.lower() in leaked:
            raise ContractError("identity leakage")
    return manifest, cells, mapping


def prepare_league(run: Path) -> str:
    manifest, cells, _ = _validate_league(run)
    packet_lanes = {cell.get("lane") for cell in cells["cells"] if isinstance(cell, dict)}
    if set(manifest["lanes"]) != packet_lanes:
        raise ContractError("unbalanced lane assignment")
    return "SEALED lanes=3"


def aggregate_league(run: Path, results_path: Path) -> str:
    manifest, cells, _ = _validate_league(run)
    results = _load_json(results_path)
    scores = results.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(manifest["lanes"]):
        raise ContractError("missing score")
    aggregate = {
        "schema": "flashpatch-blind-league-aggregate-v1",
        "sealed_assignment_sha256": _sha256(cells),
        "scores": scores,
        "seal": "SEALED",
    }
    destination = run / "aggregate.json"
    destination.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "SEALED aggregate"


def reveal_league(run: Path, aggregate_path: Path) -> str:
    _, cells, mapping = _validate_league(run)
    aggregate = _load_json(aggregate_path)
    if aggregate.get("seal") != "SEALED" or aggregate.get("sealed_assignment_sha256") != _sha256(cells):
        raise ContractError("unsealed reveal")
    revealed = {"schema": "flashpatch-blind-league-revealed-v1", "mapping": mapping["identities"], "scores": aggregate["scores"]}
    (run / "revealed.json").write_text(json.dumps(revealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "REVEALED after seal"


def claim_gate(path: Path) -> tuple[int, str]:
    """Open a hash-bound L7 score receipt and fail closed at the L9 boundary.

    Manifest scalar summaries are deliberately ignored.  Counts and repository
    identities are derived from the referenced receipt, and this gate has no
    positive external-superiority result while that receipt says
    ``external_claim_authorized=false``.
    """
    try:
        manifest_path = _l9_regular_file(path, "L9 manifest")
        manifest = _load_json(manifest_path)
        if manifest.get("frozen") is not True:
            return 1, "NOT_CLAIMABLE unfrozen input"
        if manifest.get("schema") != "flashpatch-l9-evidence-bundle-v1":
            return 1, "NOT_CLAIMABLE"

        reference = manifest.get("l7_score_receipt")
        if isinstance(reference, str):
            # Compatibility for the original diagnostic interface.  An
            # un-hashed legacy reference may explain NOT_SCOREABLE, but can
            # never reach the scoreable L9 verification path.
            receipt_path = _l9_legacy_contained_file(manifest_path, reference)
            receipt = _load_json(receipt_path)
            if receipt.get("scoreable") is not True:
                return 1, "NOT_CLAIMABLE: L7 is NOT_SCOREABLE"
            return 1, "NOT_CLAIMABLE: L7 receipt hash is required"

        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            return 1, "NOT_CLAIMABLE: missing L7 score receipt"
        receipt_path = _l9_relative_bound_file(
            manifest_path,
            reference.get("path"),
            reference.get("sha256"),
        )
        receipt = _load_json(receipt_path)
        case_count, repository_count = _verify_l9_score_receipt(receipt)
    except ContractError as exc:
        return 1, f"NOT_CLAIMABLE: {exc}"

    if receipt["scoreable"] is not True:
        return 1, "NOT_CLAIMABLE: L7 is NOT_SCOREABLE"
    if receipt["external_claim_authorized"] is False:
        return (
            1,
            "NOT_CLAIMABLE: external_claim_authorized=false "
            f"cases={case_count} repositories={repository_count}",
        )
    # No positive claim surface exists in this verifier.  A future authority
    # grant requires a separately reviewed L9 receipt/verdict contract.
    return 1, "NOT_CLAIMABLE: positive L9 authorization is unsupported"


def _l9_regular_file(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ContractError(f"{context} symlink is forbidden")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{context} is unavailable") from exc
    if not resolved.is_file():
        raise ContractError(f"{context} is unavailable")
    return resolved


def _l9_legacy_contained_file(manifest_path: Path, value: str) -> Path:
    if not value or "\x00" in value:
        raise ContractError("L7 score receipt path is invalid")
    root = manifest_path.parent.resolve()
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError("L7 score receipt path escapes L9 bundle") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ContractError("L7 score receipt is unavailable")
    return resolved


def _l9_relative_bound_file(
    manifest_path: Path,
    value: object,
    expected_sha256: object,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\x00" in value
        or ".." in Path(value).parts
    ):
        raise ContractError("L7 score receipt path must be bundle-relative")
    expected = _validate_hash(expected_sha256, "L7 score receipt sha256")
    root = manifest_path.parent.resolve()
    candidate = root / value
    current = root
    for part in Path(value).parts:
        current = current / part
        if current.is_symlink():
            raise ContractError("L7 score receipt symlink is forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError("L7 score receipt is unavailable or escapes L9 bundle") from exc
    if not resolved.is_file():
        raise ContractError("L7 score receipt is unavailable")
    if _sha256_file(resolved) != expected:
        raise ContractError("L7 score receipt hash mismatch")
    return resolved


def _canonical_l9_repository(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("L7 case repository URL is invalid")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ContractError("L7 case repository URL is invalid")
    host = parsed.hostname.lower() if parsed.hostname else ""
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not host or len(parts) < 2:
        raise ContractError("L7 case repository URL is invalid")
    owner = parts[0].lower()
    repository = parts[1].lower()
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise ContractError("L7 case repository URL is invalid")
    return f"{host}/{owner}/{repository}"


def _verify_l9_score_receipt(receipt: dict[str, Any]) -> tuple[int, int]:
    expected_population = list(DIRECT_DETECTOR_POPULATION)
    expected_population_sha256 = _sha256(expected_population)
    required_top_level = {
        "schema",
        "status",
        "claim_status",
        "scoreable",
        "external_claim_authorized",
        "case_count",
        "public_repository_count",
        "minimum_scoreable_natural_cases",
        "minimum_scoreable_public_repositories",
        "detector_population",
        "detector_population_sha256",
        "case_receipts",
        "bootstrap_lanes_separate",
    }
    if not required_top_level.issubset(receipt):
        raise ContractError("L7 score receipt schema is incomplete")
    if receipt.get("schema") != "flashpatch-l7-receipt-bound-statistics-v1":
        raise ContractError("invalid L7 score receipt schema")
    if (
        receipt.get("status") != "RECEIPT_BOUND_STATISTICS_VERIFIED"
        or receipt.get("bootstrap_lanes_separate") is not True
        or receipt.get("detector_population") != expected_population
        or receipt.get("detector_population_sha256") != expected_population_sha256
        or receipt.get("minimum_scoreable_natural_cases") != 9
        or receipt.get("minimum_scoreable_public_repositories") != 3
        or not isinstance(receipt.get("scoreable"), bool)
        or not isinstance(receipt.get("external_claim_authorized"), bool)
    ):
        raise ContractError("L7 score receipt contract mismatch")

    rows = receipt.get("case_receipts")
    if not isinstance(rows, list):
        raise ContractError("L7 case receipts are invalid")
    case_ids: set[str] = set()
    repositories: set[str] = set()
    required_row_hashes = {
        "natural_case_ledger_sha256",
        "renderer_rgb_sha256",
        "timestamps_sha256",
        "canonical_video_sha256",
        "fair_runtime_input_sha256",
        "fair_runtime_external_slot_child_joins_sha256",
        "independent_gold_receipt_sha256",
        "tool_parity_receipt_sha256",
        "fair_runtime_receipt_sha256",
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("L7 case receipt row is invalid")
        case_id = row.get("case_id")
        revision = row.get("repository_revision")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ContractError("L7 case IDs are invalid or duplicated")
        if not isinstance(revision, str) or not revision:
            raise ContractError("L7 case repository revision is invalid")
        if (
            row.get("detector_population") != expected_population
            or row.get("detector_population_sha256") != expected_population_sha256
        ):
            raise ContractError("L7 case detector population mismatch")
        for field in required_row_hashes:
            _validate_hash(row.get(field), f"L7 case {field}")
        join_count = row.get("fair_runtime_external_slot_child_join_count")
        if isinstance(join_count, bool) or not isinstance(join_count, int) or join_count != 9:
            raise ContractError("L7 case external slot join count mismatch")
        case_ids.add(case_id)
        repositories.add(_canonical_l9_repository(row.get("repository_url")))

    case_count = len(case_ids)
    repository_count = len(repositories)
    if receipt.get("case_count") != case_count:
        raise ContractError("L7 case count mismatch")
    if receipt.get("public_repository_count") != repository_count:
        raise ContractError("L7 public repository count mismatch")
    expected_scoreable = case_count >= 9 and repository_count >= 3
    if (
        receipt.get("scoreable") is not expected_scoreable
        or receipt.get("claim_status")
        != ("SCOREABLE" if expected_scoreable else "NOT_SCOREABLE")
    ):
        raise ContractError("L7 scoreability does not match receipt-derived counts")
    return case_count, repository_count


MATRIX_ROW = re.compile(
    r"\{command:\s*(?P<command>[^,{}]+?)\s*,\s*exit:\s*(?P<exit>\d+)\s*,\s*contains:\s*(?P<contains>[^{}]+?)\s*\}"
)
MATRIX_COMMAND_PREFIX = ("python", "-m", "flashpatch.competition")
MATRIX_TIMEOUT_SECONDS = 900


def _matrix_rows(plan: Path) -> list[dict[str, Any]]:
    """Read every declared validator row from the committed plan as data."""
    try:
        text = plan.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read plan: {plan}") from exc
    block = re.search(r"^validators:\s*$([\s\S]*?)(?=^\S|\Z)", text, re.MULTILINE)
    if block is None:
        raise ContractError("matrix rows missing: no validators block")
    rows: list[dict[str, Any]] = []
    validator_id = None
    follow_up_index = 0
    for line in block.group(1).splitlines():
        id_match = re.match(r"\s*-\s*id:\s*(\S+)\s*$", line)
        if id_match:
            validator_id = id_match.group(1)
            follow_up_index = 0
            continue
        row_match = MATRIX_ROW.search(line)
        if row_match is None:
            continue
        if validator_id is None:
            raise ContractError("matrix row without validator id")
        kind_match = re.match(r"\s*(pass|fail):", line)
        if kind_match:
            kind = kind_match.group(1)
        else:
            follow_up_index += 1
            kind = f"follow_up[{follow_up_index}]"
        contains = row_match.group("contains").strip()
        if len(contains) >= 2 and contains[0] == contains[-1] and contains[0] in "'\"":
            contains = contains[1:-1]
        rows.append(
            {
                "id": validator_id,
                "kind": kind,
                "command": row_match.group("command").strip().split(),
                "exit": int(row_match.group("exit")),
                "contains": contains,
            }
        )
    if not rows:
        raise ContractError("matrix rows missing: no rows parsed")
    per_validator: dict[str, set[str]] = {}
    for row in rows:
        per_validator.setdefault(row["id"], set()).add(row["kind"])
    for validator, kinds in per_validator.items():
        if not {"pass", "fail"} <= kinds:
            raise ContractError(f"matrix rows missing: validator {validator} lacks pass/fail pair")
    return rows


def run_matrix(plan: Path) -> str:
    """Execute every declared matrix row with its exact exit and output assertion."""
    rows = _matrix_rows(plan)
    for row in rows:
        label = f"{row['id']}.{row['kind']}"
        command = row["command"]
        if tuple(command[: len(MATRIX_COMMAND_PREFIX)]) != MATRIX_COMMAND_PREFIX:
            raise ContractError(f"MATRIX FAIL row={label} untrusted command rejected")
        try:
            result = subprocess.run(
                [sys.executable, *command[1:]],
                capture_output=True,
                text=True,
                check=False,
                timeout=MATRIX_TIMEOUT_SECONDS,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError(f"MATRIX FAIL row={label} exclusion_reason=timeout") from exc
        if result.returncode != row["exit"]:
            raise ContractError(
                f"MATRIX FAIL row={label} exit={result.returncode} declared={row['exit']}"
            )
        if row["contains"] in result.stdout:
            stream = "stdout"
        elif row["contains"] in result.stderr:
            stream = "stderr"
        else:
            raise ContractError(f"MATRIX FAIL row={label} missing declared output in stdout/stderr")
        print(f"ROW {label} exit={result.returncode} declared={row['exit']} match={stream}")
    return f"MATRIX PASS rows={len(rows)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashpatch.competition")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, argument in (("validate-plan", "--plan"), ("validate-case", "--case"), ("validate-provenance", "--manifest"), ("verify-receipt", "--receipt"), ("claim-gate", "--manifest"), ("compare-detectors", "--manifest")):
        sub = commands.add_parser(name)
        sub.add_argument(argument, type=Path, required=True)
    gold = commands.add_parser("validate-independent-gold")
    gold.add_argument("--receipt", type=Path, required=True)
    gold.add_argument("--trust-policy", type=Path)
    gold_intake = commands.add_parser("prepare-native-main-gold-intake")
    gold_intake.add_argument("--summary", type=Path, required=True)
    gold_intake.add_argument("--output", type=Path, required=True)
    gold_intake.add_argument("--frozen-at", required=True)
    verify_gold_intake = commands.add_parser("verify-native-main-gold-intake")
    verify_gold_intake.add_argument("--receipt", type=Path, required=True)
    seal_packet = commands.add_parser("seal-native-main-adjudicator-packet")
    seal_packet.add_argument("--intake-receipt", type=Path, required=True)
    seal_packet.add_argument("--output", type=Path, required=True)
    verify_packet_manifest = commands.add_parser("verify-native-main-adjudicator-packet")
    verify_packet_manifest.add_argument("--manifest", type=Path, required=True)
    verify_gold_return = commands.add_parser("verify-native-main-gold-return")
    verify_gold_return.add_argument("--intake-receipt", type=Path, required=True)
    verify_gold_return.add_argument("--packet-manifest", type=Path, required=True)
    verify_gold_return.add_argument("--return-root", type=Path, required=True)
    verify_start_gate = commands.add_parser("verify-native-main-candidate-start-gate")
    verify_start_gate.add_argument("--intake-receipt", type=Path, required=True)
    verify_start_gate.add_argument("--packet-manifest", type=Path, required=True)
    verify_start_gate.add_argument("--return-root", type=Path, required=True)
    verify_start_gate.add_argument("--start-receipt", type=Path, required=True)
    prepare = commands.add_parser("prepare-league")
    prepare.add_argument("--run", type=Path, required=True)
    aggregate = commands.add_parser("aggregate-league")
    aggregate.add_argument("--run", type=Path, required=True)
    aggregate.add_argument("--results", type=Path, required=True)
    reveal = commands.add_parser("reveal-league")
    reveal.add_argument("--run", type=Path, required=True)
    reveal.add_argument("--aggregate", type=Path, required=True)
    prepare_artifact = commands.add_parser("prepare-artifact-league")
    prepare_artifact.add_argument("--candidates", type=Path, required=True)
    prepare_artifact.add_argument("--anchors", type=Path, required=True)
    prepare_artifact.add_argument("--rubric", type=Path, required=True)
    prepare_artifact.add_argument("--preregistration", type=Path, required=True)
    prepare_artifact.add_argument("--reconstruction", type=Path, required=True)
    prepare_artifact.add_argument("--out", type=Path, required=True)
    prepare_artifact.add_argument("--seed", type=int, required=True)
    prepare_artifact.add_argument("--replicas", type=int, default=3)
    aggregate_artifact = commands.add_parser("aggregate-artifact-league")
    aggregate_artifact.add_argument("--run", type=Path, required=True)
    aggregate_artifact.add_argument("--results", type=Path, required=True)
    aggregate_artifact.add_argument("--out", type=Path)
    reveal_artifact = commands.add_parser("reveal-artifact-league")
    reveal_artifact.add_argument("--run", type=Path, required=True)
    reveal_artifact.add_argument("--aggregate", type=Path, required=True)
    reveal_artifact.add_argument("--out", type=Path)
    chain = commands.add_parser("verify-chain")
    chain.add_argument("--case", type=Path, required=True)
    preflight_command = commands.add_parser("preflight")
    preflight_command.add_argument("--plan", type=Path, required=True)
    preflight_command.add_argument("--expected-commit", required=True)
    preflight_command.add_argument("--checkpoints", type=Path, default=Path("evidence/competition/checkpoints"))
    matrix = commands.add_parser("run-matrix")
    matrix.add_argument("--plan", type=Path, default=Path("specs/flashpatch-competition-evidence-v1.yaml"))
    resume = commands.add_parser("resume")
    resume.add_argument("--leaf", choices=sorted(CHECKPOINT_LEAVES), required=True)
    resume.add_argument("--checkpoints", type=Path, default=Path("evidence/competition/checkpoints"))
    resume.add_argument("--input", default="")
    resume.add_argument("--command", dest="checkpoint_command", default="")
    resume.add_argument("--environment", default="")
    resume.add_argument("--receipt", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            output, code = preflight(
                args.plan, args.expected_commit, args.checkpoints
            ), 0
        elif args.command == "resume":
            if checkpoint_is_reusable(
                args.checkpoints, args.leaf, immutable_input=args.input, command=args.checkpoint_command,
                environment=args.environment, receipt=args.receipt,
            ):
                output, code = f"REUSED checkpoint leaf={args.leaf}", 0
            else:
                output, code = f"INCONCLUSIVE checkpoint leaf={args.leaf} not reusable", 1
        elif args.command == "run-matrix":
            output, code = run_matrix(args.plan), 0
        elif args.command == "validate-plan":
            output, code = validate_plan(args.plan), 0
        elif args.command == "validate-case":
            output, code = validate_case(args.case), 0
        elif args.command == "validate-provenance":
            output, code = validate_provenance(args.manifest), 0
        elif args.command == "verify-receipt":
            output, code = verify_receipt(args.receipt), 0
        elif args.command == "validate-independent-gold":
            output = validate_independent_gold(
                args.receipt,
                trust_policy_path=args.trust_policy,
            )
            code = 0 if output.startswith("VALID ") else 1
        elif args.command == "prepare-native-main-gold-intake":
            receipt = prepare_native_main_blinded_gold_intake(
                args.summary,
                args.output,
                frozen_at=args.frozen_at,
            )
            output, code = f"INDEPENDENT_GOLD_MISSING intake_receipt={receipt}", 0
        elif args.command == "verify-native-main-gold-intake":
            output, code = json.dumps(
                verify_native_main_blinded_gold_intake(args.receipt),
                sort_keys=True,
                indent=2,
            ), 0
        elif args.command == "seal-native-main-adjudicator-packet":
            manifest_path = seal_native_main_adjudicator_packet(
                intake_receipt=args.intake_receipt,
                output_path=args.output,
            )
            output, code = json.dumps(
                verify_native_main_adjudicator_packet_manifest(manifest_path),
                sort_keys=True,
                indent=2,
            ), 0
        elif args.command == "verify-native-main-adjudicator-packet":
            output, code = json.dumps(
                verify_native_main_adjudicator_packet_manifest(args.manifest),
                sort_keys=True,
                indent=2,
            ), 0
        elif args.command == "verify-native-main-gold-return":
            output, code = json.dumps(
                verify_native_main_independent_gold_return(
                    intake_receipt=args.intake_receipt,
                    packet_manifest=args.packet_manifest,
                    return_root=args.return_root,
                ),
                sort_keys=True,
                indent=2,
            ), 0
        elif args.command == "verify-native-main-candidate-start-gate":
            output, code = json.dumps(
                verify_native_main_candidate_start_gate(
                    intake_receipt=args.intake_receipt,
                    packet_manifest=args.packet_manifest,
                    return_root=args.return_root,
                    start_receipt=args.start_receipt,
                ),
                sort_keys=True,
                indent=2,
            ), 0
        elif args.command == "prepare-league":
            output, code = prepare_league(args.run), 0
        elif args.command == "aggregate-league":
            output, code = aggregate_league(args.run, args.results), 0
        elif args.command == "reveal-league":
            output, code = reveal_league(args.run, args.aggregate), 0
        elif args.command == "prepare-artifact-league":
            prepared = prepare_artifact_league(
                candidates_path=args.candidates,
                anchors_path=args.anchors,
                rubric_path=args.rubric,
                preregistration_path=args.preregistration,
                reconstruction_path=args.reconstruction,
                out=args.out,
                seed=args.seed,
                replicas=args.replicas,
            )
            output, code = (
                f"SEALED artifact-league lanes=3 candidates={prepared['candidate_count']}",
                0,
            )
        elif args.command == "aggregate-artifact-league":
            sealed = aggregate_artifact_league(
                run=args.run, results_path=args.results, out=args.out
            )
            output, code = (
                f"SEALED aggregate candidates={len(sealed['candidates'])} "
                f"warnings={len(sealed['warnings'])}",
                0,
            )
        elif args.command == "reveal-artifact-league":
            revealed = reveal_artifact_league(
                run=args.run, aggregate_path=args.aggregate, out=args.out
            )
            output, code = f"REVEALED after seal candidates={len(revealed['candidates'])}", 0
        elif args.command == "verify-chain":
            output, code = verify_chain(args.case), 0
        elif args.command == "compare-detectors":
            code, output = compare_detectors(args.manifest)
        else:
            code, output = claim_gate(args.manifest)
    except (ContractError, L8LeagueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
