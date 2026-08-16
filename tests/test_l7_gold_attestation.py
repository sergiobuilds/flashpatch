from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import flashpatch.competition as competition
from flashpatch.competition import (
    ContractError,
    INDEPENDENT_GOLD_NOT_VERIFIED,
    ApprovedGoldTrustPolicy,
    prepare_native_main_blinded_gold_intake,
    seal_native_main_adjudicator_packet,
    validate_independent_gold,
    verify_native_main_adjudicator_packet_manifest,
    verify_native_main_candidate_start_gate,
    verify_native_main_blinded_gold_intake,
    verify_native_main_independent_gold_return,
    project_independent_gold,
)


ROOT = Path(__file__).parents[1]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _generate_test_identity(root: Path, identity_id: str) -> tuple[Path, Path]:
    private_key = root / f"{identity_id}.private.pem"
    public_key = root / f"{identity_id}.public.pem"
    subprocess.run(
        ["/usr/bin/openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_key)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "/usr/bin/openssl",
            "pkey",
            "-in",
            str(private_key),
            "-pubout",
            "-out",
            str(public_key),
        ],
        check=True,
        capture_output=True,
    )
    return private_key, public_key


def _sign(root: Path, name: str, private_key: Path, payload: object) -> str:
    payload_path = root / f"{name}.payload"
    signature_path = root / f"{name}.signature"
    payload_path.write_bytes(_canonical_bytes(payload))
    subprocess.run(
        [
            "/usr/bin/openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            str(private_key),
            "-rawin",
            "-in",
            str(payload_path),
            "-out",
            str(signature_path),
        ],
        check=True,
        capture_output=True,
    )
    return base64.b64encode(signature_path.read_bytes()).decode("ascii")


def _spki_sha256(root: Path, name: str, public_key: Path) -> str:
    der_path = root / f"{name}.spki.der"
    subprocess.run(
        [
            "/usr/bin/openssl",
            "pkey",
            "-pubin",
            "-in",
            str(public_key),
            "-outform",
            "DER",
            "-out",
            str(der_path),
        ],
        check=True,
        capture_output=True,
    )
    return _sha256_bytes(der_path.read_bytes())


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


def _build_attested_gold(
    tmp_path: Path,
    *,
    committed_at: str = "2026-08-02T00:00:00Z",
    submission_at: str = "2026-08-02T00:01:00Z",
    commitment_attested_at: str = "2026-08-02T00:02:00Z",
    candidate_started_at: str = "2026-08-02T00:03:00Z",
    corpus_frozen_at: str = "2026-08-01T23:55:00Z",
    decision: str = "SAFE",
    intervals: list[dict[str, Any]] | None = None,
    duplicate_adjudicator: bool = False,
    overlap_operator: bool = False,
    missing_signature: bool = False,
    invalid_signature: bool = False,
    substitute_witness: bool = False,
    missing_candidate_start_receipt: bool = False,
    invalid_approval_signature: bool = False,
    timestamp_subject_override: str | None = None,
    timestamp_approved_at_override: str | None = None,
    same_root_key: bool = False,
    root_overlaps_adjudicator_key: bool = False,
    blinded_extra: dict[str, Any] | None = None,
) -> tuple[Path, Path, ApprovedGoldTrustPolicy, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = tmp_path / "keys"
    keys.mkdir()
    private_keys: dict[str, Path] = {}
    public_keys: dict[str, Path] = {}
    for identity_id in ("adj-0", "adj-1", "adj-2", "wit-0", "wit-1"):
        private_keys[identity_id], public_keys[identity_id] = _generate_test_identity(keys, identity_id)
    approval_root = tmp_path / "approval-root"
    approval_keys = approval_root / "keys"
    approval_keys.mkdir(parents=True)
    for identity_id in ("root-approval", "root-timestamp"):
        private_keys[identity_id], public_keys[identity_id] = _generate_test_identity(
            approval_keys,
            identity_id,
        )
    if same_root_key:
        private_keys["root-timestamp"] = private_keys["root-approval"]
        public_keys["root-timestamp"] = public_keys["root-approval"]
    if root_overlaps_adjudicator_key:
        private_keys["root-approval"].write_bytes(private_keys["adj-0"].read_bytes())
        public_keys["root-approval"].write_bytes(public_keys["adj-0"].read_bytes())

    policy = {
        "schema": "flashpatch-l7-gold-trust-policy-v1",
        "policy_id": "external-policy-2026",
        "roster_id": "external-roster-2026",
        "required_independent_submissions": 3,
        "flashpatch_operator_ids": ["adj-0" if overlap_operator else "flashpatch-operator"],
        "candidate_operator_ids": ["candidate-operator"],
        "adjudicators": [
            {
                "identity_id": identity_id,
                "role": "external_adjudicator",
                "public_key_path": str(public_keys[identity_id].relative_to(tmp_path)),
                "public_key_sha256": _sha256_bytes(public_keys[identity_id].read_bytes()),
                "ed25519_spki_sha256": _spki_sha256(tmp_path, identity_id, public_keys[identity_id]),
            }
            for identity_id in ("adj-0", "adj-1", "adj-2")
        ],
        "witnesses": [
            {
                "identity_id": identity_id,
                "role": "external_timestamp_witness",
                "public_key_path": str(public_keys[identity_id].relative_to(tmp_path)),
                "public_key_sha256": _sha256_bytes(public_keys[identity_id].read_bytes()),
                "ed25519_spki_sha256": _spki_sha256(tmp_path, identity_id, public_keys[identity_id]),
            }
            for identity_id in ("wit-0", "wit-1")
        ],
    }
    policy_path = tmp_path / "external-trust-policy.json"
    policy_path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    policy_sha256 = _sha256_bytes(policy_path.read_bytes())
    verifier_source_commit = "018d58a3c021c28a097b89eb9f0a0b46abab28f1"
    build_manifest = {
        "schema": "flashpatch-l7-gold-verifier-build-v1",
        "verifier_source_commit": verifier_source_commit,
        "competition_py_sha256": competition._approval_verifier_source_sha256(),
        "built_at": "2026-08-01T23:45:00Z",
    }
    build_manifest_path = approval_root / "verifier-build.json"
    build_manifest_path.write_text(json.dumps(build_manifest, sort_keys=True), encoding="utf-8")
    build_manifest_sha256 = _sha256_bytes(build_manifest_path.read_bytes())
    approval_payload = {
        "schema": "flashpatch-l7-external-policy-approval-v1",
        "policy_id": str(policy["policy_id"]),
        "roster_id": str(policy["roster_id"]),
        "policy_sha256": policy_sha256,
        "approved_at": "2026-08-01T23:50:00Z",
        "valid_from": "2026-08-01T23:51:00Z",
        "valid_until": None,
        "revoked_at": None,
        "predecessor_policy_sha256": None,
        "predecessor_approval_record_sha256": None,
        "normative_rule_sha256": "b" * 64,
        "verifier_source_commit": verifier_source_commit,
        "verifier_build_manifest_sha256": build_manifest_sha256,
        "approval_authority_id": "root-approval",
    }
    approval_record = {
        **approval_payload,
        "signature_algorithm": "Ed25519",
        "signature_base64": _sign(
            approval_root,
            "approval-record",
            private_keys["root-approval"],
            {"domain": "FlashPatch/L7/trust-policy-approval/v1", "payload": approval_payload},
        ),
    }
    if invalid_approval_signature:
        approval_record["signature_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    approval_record_path = approval_root / "approval-record.json"
    approval_record_path.write_text(json.dumps(approval_record, sort_keys=True), encoding="utf-8")
    approval_record_sha256 = _sha256_bytes(approval_record_path.read_bytes())
    timestamp_payload = {
        "schema": "flashpatch-l7-external-approval-timestamp-v1",
        "timestamp_authority_id": "root-timestamp",
        "subject_policy_sha256": timestamp_subject_override or policy_sha256,
        "subject_approval_record_sha256": approval_record_sha256,
        "approved_at": timestamp_approved_at_override or approval_payload["approved_at"],
        "token_serial": "tsa-0123456789abcdef0123456789abcdef",
    }
    timestamp_token = {
        **timestamp_payload,
        "signature_algorithm": "Ed25519",
        "signature_base64": _sign(
            approval_root,
            "timestamp-token",
            private_keys["root-timestamp"],
            {"domain": "FlashPatch/L7/external-approval-timestamp/v1", "payload": timestamp_payload},
        ),
    }
    timestamp_token_path = approval_root / "timestamp-token.json"
    timestamp_token_path.write_text(json.dumps(timestamp_token, sort_keys=True), encoding="utf-8")
    timestamp_token_sha256 = _sha256_bytes(timestamp_token_path.read_bytes())
    approval = ApprovedGoldTrustPolicy(
        policy_id=str(policy["policy_id"]),
        roster_id=str(policy["roster_id"]),
        policy_sha256=policy_sha256,
        approval_record_sha256=approval_record_sha256,
        external_timestamp_token_sha256=timestamp_token_sha256,
        verifier_source_commit=verifier_source_commit,
        approved_at=str(approval_payload["approved_at"]),
        valid_from=str(approval_payload["valid_from"]),
        valid_until=None,
        revoked_at=None,
        predecessor_policy_sha256=None,
        normative_rule_sha256="b" * 64,
        openssl_binary_sha256=_sha256_bytes(Path("/usr/bin/openssl").read_bytes()),
        approval_record_path=approval_record_path.name,
        approval_authority_id="root-approval",
        approval_authority_public_key_path=str(public_keys["root-approval"].relative_to(approval_root)),
        approval_authority_public_key_sha256=_sha256_bytes(public_keys["root-approval"].read_bytes()),
        approval_authority_spki_sha256=_spki_sha256(approval_root, "root-approval", public_keys["root-approval"]),
        external_timestamp_token_path=timestamp_token_path.name,
        timestamp_authority_id="root-timestamp",
        timestamp_authority_public_key_path=str(public_keys["root-timestamp"].relative_to(approval_root)),
        timestamp_authority_public_key_sha256=_sha256_bytes(public_keys["root-timestamp"].read_bytes()),
        timestamp_authority_spki_sha256=_spki_sha256(approval_root, "root-timestamp", public_keys["root-timestamp"]),
        verifier_build_manifest_path=build_manifest_path.name,
        verifier_build_manifest_sha256=build_manifest_sha256,
        predecessor_approval_record_sha256=None,
    )
    registry_snapshot_sha256 = _sha256_bytes(_canonical_bytes([asdict(approval)]))

    blind_case_id = "blind-0123456789abcdef0123456789abcdef"
    rubric_bytes = json.dumps(
        {
            "schema": "flashpatch-l7-gold-rubric-v1",
            "profile_id": "wcag22-g19-v1",
            "source_url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G19.html",
            "normative_rule_sha256": "b" * 64,
        },
        sort_keys=True,
    ).encode("utf-8")
    corpus_bytes = json.dumps(
        {
            "schema": "flashpatch-l7-blinded-corpus-commitment-v1",
            "discovery_manifest_sha256": "c" * 64,
            "blind_case_ids": [blind_case_id],
            "selection_rule_sha256": "d" * 64,
            "stop_rule_sha256": "e" * 64,
            "frozen_at": corpus_frozen_at,
            "candidate_outputs_observed": False,
        },
        sort_keys=True,
    ).encode("utf-8")
    timestamps_bytes = json.dumps(
        {
            "schema": "flashpatch-l7-renderer-timestamps-v1",
            "timestamps_seconds": [0.0, 0.016, 0.033],
        },
        sort_keys=True,
    ).encode("utf-8")
    payloads = {
        "source_tree_sha256": ("source-tree.bin", b"pinned-source-tree"),
        "trace_sha256": ("trace.json", b'{"fixed_fps":60}'),
        "renderer_rgb_raw_sha256": ("renderer.rgb", b"renderer-rgb"),
        "timestamps_sha256": ("timestamps.json", timestamps_bytes),
        "rubric_sha256": ("rubric.json", rubric_bytes),
        "corpus_manifest_sha256": ("corpus-manifest.json", corpus_bytes),
    }
    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for field, (name, content) in payloads.items():
        target = tmp_path / name
        target.write_bytes(content)
        artifacts[field] = name
        hashes[field] = _sha256_bytes(content)

    execution = {
        "schema": "flashpatch-renderer-engine-receipt-v1",
        "controlled_mutation": False,
        "upstream": {
            "repository_url": "https://github.com/example/game",
            "source_revision": "a" * 40,
            "license": "MIT",
            "project_path": ".",
        },
        "factual_replay": {
            "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
            "timestamps_sha256": hashes["timestamps_sha256"],
            "frame_count": 3,
            "renderer_capture": {"trace_sha256": f"sha256:{hashes['trace_sha256']}"},
        },
    }
    execution_path = tmp_path / "renderer-execution.json"
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")
    artifacts["renderer_execution_receipt_sha256"] = execution_path.name
    hashes["renderer_execution_receipt_sha256"] = _sha256_bytes(execution_path.read_bytes())

    blinded_input: dict[str, Any] = {
        "schema": "flashpatch-l7-blinded-gold-input-v1",
        "blind_case_id": blind_case_id,
        "corpus_manifest_sha256": hashes["corpus_manifest_sha256"],
        "source_tree_sha256": hashes["source_tree_sha256"],
        "trace_sha256": hashes["trace_sha256"],
        "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
        "timestamps_sha256": hashes["timestamps_sha256"],
        "frame_count": 3,
        "color_space": "sRGB_BT709",
        "rubric_sha256": hashes["rubric_sha256"],
    }
    if blinded_extra:
        blinded_input.update(blinded_extra)
    blinded_path = tmp_path / "blinded-gold-input.json"
    blinded_path.write_text(json.dumps(blinded_input, sort_keys=True), encoding="utf-8")
    artifacts["blinded_gold_input_sha256"] = blinded_path.name
    hashes["blinded_gold_input_sha256"] = _sha256_bytes(blinded_path.read_bytes())

    gold_intervals = [] if intervals is None else intervals
    gold_decision = {"decision": decision, "intervals": gold_intervals}
    opening_nonce_bytes = b"0" * 32
    gold_decision_commitment = _sha256_bytes(
        b"FlashPatch/L7/gold-decision/v1\x00"
        + opening_nonce_bytes
        + _canonical_bytes(gold_decision)
    )
    commitment = {
        "schema": "flashpatch-l7-pre-candidate-gold-commitment-v1",
        "case_id": "natural-sparta-safe-001",
        "corpus_manifest_sha256": hashes["corpus_manifest_sha256"],
        "blinded_gold_input_sha256": hashes["blinded_gold_input_sha256"],
        "rubric_sha256": hashes["rubric_sha256"],
        "gold_decision_commitment_sha256": gold_decision_commitment,
        "committed_at": committed_at,
    }
    commitment_sha256 = _sha256_bytes(_canonical_bytes(commitment))

    candidate_start_receipt = {
        "schema": "flashpatch-l7-candidate-start-receipt-v1",
        "case_id": "natural-sparta-safe-001",
        "run_id": "run-0123456789abcdef0123456789abcdef",
        "candidate_tools": list(competition.L7_DIRECT_CANDIDATE_TOOLS),
        "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
        "timestamps_sha256": hashes["timestamps_sha256"],
        "pre_candidate_commitment_sha256": commitment_sha256,
        "trust_policy_id": approval.policy_id,
        "approval_record_sha256": approval.approval_record_sha256,
        "external_timestamp_token_sha256": approval.external_timestamp_token_sha256,
        "registry_snapshot_sha256": registry_snapshot_sha256,
        "verifier_source_commit": approval.verifier_source_commit,
        "openssl_binary_sha256": approval.openssl_binary_sha256,
        "process_started_at": candidate_started_at,
        "process_start_monotonic_ns": 123456789,
        "state": "STARTED_OUTPUT_UNOPENED",
    }
    candidate_start_receipt_path = tmp_path / "candidate-start-receipt.json"
    candidate_start_receipt_path.write_text(
        json.dumps(candidate_start_receipt, sort_keys=True),
        encoding="utf-8",
    )
    if not missing_candidate_start_receipt:
        artifacts["candidate_start_receipt_sha256"] = candidate_start_receipt_path.name
    hashes["candidate_start_receipt_sha256"] = _sha256_bytes(candidate_start_receipt_path.read_bytes())

    submissions: list[dict[str, Any]] = []
    for index in range(3):
        identity_id = "adj-0" if duplicate_adjudicator and index == 1 else f"adj-{index}"
        signed_payload = {
            "adjudicator_id": identity_id,
            "blinded_gold_input_sha256": hashes["blinded_gold_input_sha256"],
            "pre_candidate_commitment_sha256": commitment_sha256,
            "decision": decision,
            "intervals": gold_intervals,
            "signed_at": submission_at,
        }
        submission = {
            **signed_payload,
            "submission_payload_sha256": _sha256_bytes(_canonical_bytes(signed_payload)),
            "signature_algorithm": "Ed25519",
            "signature_base64": _sign(
                tmp_path,
                f"submission-{index}",
                private_keys[identity_id],
                _signature_envelope(
                    message_kind="adjudicator_submission",
                    approval=approval,
                    registry_snapshot_sha256=registry_snapshot_sha256,
                    policy=policy,
                    signer_id=identity_id,
                    signer_role="external_adjudicator",
                    payload=signed_payload,
                ),
            ),
        }
        submissions.append(submission)
    if missing_signature:
        submissions[0].pop("signature_base64")
    if invalid_signature:
        submissions[0]["signature_base64"] = base64.b64encode(b"x" * 64).decode("ascii")

    commitment_witness_payload = {
        "witness_id": "wit-0",
        "pre_candidate_commitment_sha256": commitment_sha256,
        "attested_at": commitment_attested_at,
    }
    commitment_witness = {
        **commitment_witness_payload,
        "signature_algorithm": "Ed25519",
        "signature_base64": _sign(
            tmp_path,
            "commitment-witness",
            private_keys["wit-0"],
            _signature_envelope(
                message_kind="pre_candidate_commitment_witness",
                approval=approval,
                registry_snapshot_sha256=registry_snapshot_sha256,
                policy=policy,
                signer_id="wit-0",
                signer_role="external_timestamp_witness",
                payload=commitment_witness_payload,
            ),
        ),
    }
    start_witness_id = "adj-0" if substitute_witness else "wit-1"
    candidate_start_payload = {
        "witness_id": start_witness_id,
        "case_id": "natural-sparta-safe-001",
        "run_id": candidate_start_receipt["run_id"],
        "pre_candidate_commitment_sha256": commitment_sha256,
        "gold_decision_commitment_sha256": commitment["gold_decision_commitment_sha256"],
        "candidate_start_receipt_sha256": hashes["candidate_start_receipt_sha256"],
        "candidate_started_at": candidate_started_at,
    }
    candidate_start = {
        **candidate_start_payload,
        "signature_algorithm": "Ed25519",
        "signature_base64": _sign(
            tmp_path,
            "candidate-start-witness",
            private_keys[start_witness_id],
            _signature_envelope(
                message_kind="candidate_start_witness",
                approval=approval,
                registry_snapshot_sha256=registry_snapshot_sha256,
                policy=policy,
                signer_id=start_witness_id,
                signer_role="external_timestamp_witness",
                payload=candidate_start_payload,
            ),
        ),
    }

    receipt = {
        "schema": "flashpatch-l7-independent-gold-v2",
        "case_id": "natural-sparta-safe-001",
        "case_class": "natural_external",
        "claim_tier": "L9_ELIGIBLE",
        "controlled_mutation": False,
        "case_freeze": {
            "public_repository_url": "https://github.com/example/game",
            "source_revision": "a" * 40,
            "license": "MIT",
            "project_subpath": ".",
            "source_tree_sha256": hashes["source_tree_sha256"],
            "trace_sha256": hashes["trace_sha256"],
            "renderer_execution_receipt_sha256": hashes["renderer_execution_receipt_sha256"],
            "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
            "timestamps_sha256": hashes["timestamps_sha256"],
            "frame_count": 3,
            "color_space": "sRGB_BT709",
        },
        "standard_profile": {
            "id": "wcag22-g19-v1",
            "source_url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G19.html",
            "rubric_sha256": hashes["rubric_sha256"],
        },
        "gold_authority": {
            "kind": "independent_adjudication",
            "candidate_tools_excluded": sorted(competition.GOLD_EXCLUDED_TOOLS),
            "trust_policy_id": policy["policy_id"],
            "roster_id": policy["roster_id"],
            "required_independent_submissions": policy["required_independent_submissions"],
            "flashpatch_operator_ids": policy["flashpatch_operator_ids"],
            "candidate_operator_ids": policy["candidate_operator_ids"],
        },
        "pre_candidate_commitment": commitment,
        "commitment_witness": commitment_witness,
        "adjudication": {
            "method": "three-independent-submissions-plus-resolution",
            "resolution": "unanimous",
            "result": decision,
            "intervals": gold_intervals,
            "opening_nonce_base64": base64.b64encode(opening_nonce_bytes).decode("ascii"),
            "submissions": submissions,
        },
        "candidate_start_witness": candidate_start,
        "artifacts": artifacts,
    }
    receipt_path = tmp_path / "independent-gold.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return receipt_path, policy_path, approval, approval_root


def _validate_as_preapproved(
    package: tuple[Path, Path, ApprovedGoldTrustPolicy, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    receipt, policy, approval, approval_root = package
    monkeypatch.setattr(
        competition,
        "APPROVED_GOLD_TRUST_POLICIES",
        (approval,),
    )
    monkeypatch.setattr(competition, "GOLD_APPROVAL_ARTIFACT_ROOT", approval_root)
    return validate_independent_gold(
        receipt,
        trust_policy_path=policy,
    )


def test_fully_self_created_policy_and_signatures_cannot_activate_empty_approval_registry(tmp_path: Path) -> None:
    receipt, policy, _, _ = _build_attested_gold(tmp_path)
    assert validate_independent_gold(receipt, trust_policy_path=policy) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_gold_exclusion_policy_covers_canonical_historical_and_iris_aliases() -> None:
    assert competition.GOLD_EXCLUDED_TOOLS == {
        "FlashPatch",
        "KAYA_PSE_DETECTION_CORRECTION_0776EA3",
        "KAYA_SOURCE_DIRECT_INPUT_PROTOTYPE_0776EA3E_UNSCORED",
        "TooFlashy",
        "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU",
        "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC",
        "FFmpeg",
    }


@pytest.mark.parametrize(
    "omitted_identity",
    [
        "KAYA_PSE_DETECTION_CORRECTION_0776EA3",
        "KAYA_SOURCE_DIRECT_INPUT_PROTOTYPE_0776EA3E_UNSCORED",
        "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU",
        "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC",
    ],
)
def test_independent_gold_rejects_omitted_candidate_or_alias_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    omitted_identity: str,
) -> None:
    package = _build_attested_gold(tmp_path)
    receipt_path = package[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    exclusions = receipt["gold_authority"]["candidate_tools_excluded"]
    exclusions.remove(omitted_identity)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_cli_cannot_accept_a_caller_supplied_trust_hash(tmp_path: Path) -> None:
    receipt, policy, approval, _ = _build_attested_gold(tmp_path)
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.competition",
            "validate-independent-gold",
            "--receipt",
            str(receipt),
            "--trust-policy",
            str(policy),
            "--trust-policy-sha256",
            approval.policy_sha256,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --trust-policy-sha256" in result.stderr


def test_receipt_cannot_self_attest_independence(tmp_path: Path) -> None:
    receipt, _, _, _ = _build_attested_gold(tmp_path)
    assert validate_independent_gold(receipt) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_preapproved_gold_has_a_structured_receipt_bound_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        decision="HAZARDOUS",
        intervals=[{"kind": "flash", "start_seconds": 0.0, "end_seconds": 0.02}],
    )
    receipt, policy, approval, approval_root = package
    monkeypatch.setattr(competition, "APPROVED_GOLD_TRUST_POLICIES", (approval,))
    monkeypatch.setattr(competition, "GOLD_APPROVAL_ARTIFACT_ROOT", approval_root)

    projection = project_independent_gold(receipt, trust_policy_path=policy)

    assert projection["gold_verified"] is True
    assert projection["case_class"] == "natural_external"
    assert projection["controlled_mutation"] is False
    assert projection["decision"] == "HAZARDOUS"
    assert projection["intervals"] == [
        {"kind": "flash", "start_seconds": 0.0, "end_seconds": 0.02}
    ]
    assert projection["timestamps_seconds"] == [0.0, 0.016, 0.033]
    assert projection["receipt"]["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert projection["trust_policy"]["sha256"] == approval.policy_sha256
    assert projection["league_score_authorized"] is False


def test_post_candidate_gold_seal_is_not_scoreable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _build_attested_gold(
        tmp_path,
        candidate_started_at="2026-08-01T23:59:59Z",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize("mode", ["duplicate", "operator_overlap"])
def test_duplicate_or_operator_overlapping_adjudicator_is_not_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        duplicate_adjudicator=mode == "duplicate",
        overlap_operator=mode == "operator_overlap",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize("mode", ["missing", "invalid"])
def test_missing_or_invalid_cryptographic_signature_is_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        missing_signature=mode == "missing",
        invalid_signature=mode == "invalid",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_adjudicator_cannot_substitute_for_external_timestamp_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(tmp_path, substitute_witness=True)
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_fully_resigned_blinded_bundle_with_candidate_output_leak_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        blinded_extra={"candidate_predictions": {"FlashPatch": "SAFE"}},
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_missing_candidate_start_receipt_is_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(tmp_path, missing_candidate_start_receipt=True)
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_future_dated_candidate_start_is_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        committed_at="2099-01-01T00:00:00Z",
        submission_at="2099-01-01T00:01:00Z",
        commitment_attested_at="2099-01-01T00:02:00Z",
        candidate_started_at="2099-01-01T00:03:00Z",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_corpus_frozen_after_candidate_start_is_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        corpus_frozen_at="2026-08-02T00:04:00Z",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_hazard_interval_outside_renderer_timestamp_range_is_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        decision="HAZARDOUS",
        intervals=[{"kind": "flash", "start_seconds": 0.0, "end_seconds": 100.0}],
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_record_sha256", "0" * 64),
        ("approved_at", "2026-08-02T00:10:00Z"),
        ("verifier_source_commit", "2" * 40),
    ],
)
def test_arbitrary_registry_hash_time_or_commit_is_not_an_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    receipt, policy, approval, approval_root = _build_attested_gold(tmp_path)
    forged = replace(approval, **{field: value})
    assert _validate_as_preapproved(
        (receipt, policy, forged, approval_root),
        monkeypatch,
    ) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize("mode", ["signature", "subject", "time"])
def test_external_approval_signature_and_timestamp_subject_time_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        invalid_approval_signature=mode == "signature",
        timestamp_subject_override="3" * 64 if mode == "subject" else None,
        timestamp_approved_at_override="2026-08-02T00:10:00Z" if mode == "time" else None,
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize("mode", ["same_root", "root_overlaps_adjudicator"])
def test_approval_timestamp_and_adjudicator_keys_are_cryptographically_disjoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    package = _build_attested_gold(
        tmp_path,
        same_root_key=mode == "same_root",
        root_overlaps_adjudicator_key=mode == "root_overlaps_adjudicator",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_duplicate_json_keys_in_signed_receipt_are_not_scoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _build_attested_gold(tmp_path)
    receipt = package[0]
    receipt.write_text(
        receipt.read_text(encoding="utf-8").replace("{", '{"schema":"forged",', 1),
        encoding="utf-8",
    )
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_path_shadowed_openssl_cannot_validate_a_bad_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_openssl = fake_bin / "openssl"
    fake_openssl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_openssl.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    package = _build_attested_gold(tmp_path / "package", invalid_signature=True)
    assert _validate_as_preapproved(package, monkeypatch) == INDEPENDENT_GOLD_NOT_VERIFIED


def _write_renderer_npz(path: Path, *, seed: int) -> tuple[str, str, int]:
    frames = np.full((3, 2, 2, 3), seed, dtype=np.uint8)
    timestamps = np.asarray([0.0, 0.016, 0.033], dtype=np.float64)
    np.savez(path, frames=frames, timestamps=timestamps)
    return (
        hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
        hashlib.sha256(np.ascontiguousarray(timestamps).tobytes()).hexdigest(),
        int(len(frames)),
    )


def _native_main_summary_for_gold_intake(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    (project / "src" / "flashpatch").mkdir(parents=True)
    (project / "docs" / "MASTER-MAP.md").write_text("# map\n", encoding="utf-8")
    manifest = project / "discovery.json"
    manifest.write_text(json.dumps({"schema": "test-discovery"}, sort_keys=True), encoding="utf-8")
    preflight = project / "preflight-summary.json"
    preflight.write_text(json.dumps({"schema": "test-preflight"}, sort_keys=True), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for index, decision in enumerate(["SAFE_SCENARIO_READY", "HAZARDOUS_ATTRIBUTION_PENDING"]):
        case_root = project / "artifacts" / f"case-{index}"
        case_root.mkdir(parents=True)
        frame = case_root / "renderer-frames.npz"
        _, _, frame_count = _write_renderer_npz(frame, seed=index)
        receipt = case_root / "native-main-capture-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "flashpatch-godot-native-main-capture-v1",
                    "decision": decision,
                    "qualification_only": True,
                    "scoreable": False,
                    "native_equivalence": "NOT_ESTABLISHED",
                    "frame_count": frame_count,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "candidate_id": f"source-case-{index}",
                "status": f"{decision}_QUALIFICATION_ONLY",
                "decision": decision,
                "scoreable": False,
                "external_claim_authorized": False,
                "receipt_created": True,
                "frame_count": frame_count,
                "frame_artifact_path": str(frame.relative_to(project)),
                "frame_artifact_sha256": _sha256_bytes(frame.read_bytes()),
                "raw_receipt_path": str(receipt.relative_to(project)),
                "raw_receipt_sha256": _sha256_bytes(receipt.read_bytes()),
            }
        )
    rows.append(
        {
            "candidate_id": "failed-case",
            "status": "FAILED_TIMEOUT",
            "scoreable": False,
            "external_claim_authorized": False,
            "receipt_created": False,
        }
    )
    summary = {
        "schema": "flashpatch-l7-native-main-qualification-summary-v1",
        "summary_id": "native-main-test-summary",
        "discovery_manifest_path": str(manifest.relative_to(project)),
        "discovery_manifest_sha256": _sha256_bytes(manifest.read_bytes()),
        "source_preflight_summary_path": str(preflight.relative_to(project)),
        "source_preflight_summary_sha256": _sha256_bytes(preflight.read_bytes()),
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "external_claim_authorized": False,
        "candidate_count": 3,
        "executed_complete_receipt_count": 2,
        "failed_count": 1,
        "safe_qualification_only_count": 1,
        "hazardous_attribution_pending_count": 1,
        "qualified_case_count": 0,
        "score_blockers": [
            "qualification_only_capture",
            "native_equivalence_not_established",
            "independent_gold_missing",
            "qualified_natural_case_count_below_nine",
            "same_condition_9_slot_execution_missing",
            "receipt_bound_score_bundle_missing",
        ],
        "results": rows,
    }
    summary_path = project / "evidence" / "l7" / "native-main" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return summary_path


def test_native_main_blinded_gold_intake_packages_complete_receipts_without_score_authority(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )

    result = verify_native_main_blinded_gold_intake(receipt)

    assert result["status"] == "INDEPENDENT_GOLD_MISSING"
    assert result["scoreable"] is False
    assert result["case_count"] == 2
    assert "independent_gold_missing" in result["score_blockers"]
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["adjudicator_instructions_sha256"] == receipt_payload["adjudicator_instructions_sha256"]
    assert result["expected_return_contract_sha256"] == receipt_payload["expected_return_contract_sha256"]
    assert result["trust_policy_template_sha256"] == receipt_payload["trust_policy_template_sha256"]
    for field in (
        "adjudicator_instructions",
        "expected_return_contract",
        "trust_policy_template",
    ):
        path = receipt.parent / receipt_payload[f"{field}_path"]
        assert path.is_file()
        assert _sha256_bytes(path.read_bytes()) == receipt_payload[f"{field}_sha256"]
    contract_path = receipt.parent / receipt_payload["expected_return_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["required_packet_manifest_binding"] == {
        "required_return_manifest_fields": [
            "adjudicator_packet_manifest_sha256",
            "adjudicator_packet_root_sha256",
        ],
        "adjudicator_packet_manifest_schema": "flashpatch-l7-native-main-adjudicator-packet-manifest-v1",
        "packet_root_hash_field": "packet_manifest_sha256",
        "binding_rule": "return_manifest_must_echo_manifest_file_sha256_and_packet_root_sha256",
    }
    for entry in receipt_payload["blinded_inputs"]:
        blinded = json.loads((receipt.parent / entry["path"]).read_text(encoding="utf-8"))
        assert "candidate_id" not in json.dumps(blinded).lower()
        assert "hazardous_attribution_pending" not in json.dumps(blinded).lower()
        assert "safe_scenario_ready" not in json.dumps(blinded).lower()


def test_native_main_blinded_gold_intake_rejects_candidate_prediction_leak(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["blinded_inputs"][0]["path"]
    blinded = json.loads(target.read_text(encoding="utf-8"))
    blinded["candidate_predictions"] = {"FlashPatch": "HAZARDOUS"}
    target.write_text(json.dumps(blinded, sort_keys=True), encoding="utf-8")
    payload["blinded_inputs"][0]["sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="blinded gold intake leaks candidate output"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_repository_identity_leak(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["blinded_inputs"][0]["path"]
    blinded = json.loads(target.read_text(encoding="utf-8"))
    blinded["repository_url"] = "https://example.invalid/source-case-0"
    target.write_text(json.dumps(blinded, sort_keys=True), encoding="utf-8")
    payload["blinded_inputs"][0]["sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="blinded gold intake leaks candidate output"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_missing_frame_copy(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["blinded_inputs"][0]["path"]
    blinded = json.loads(target.read_text(encoding="utf-8"))
    (target.parent.parent / blinded["frame_artifact_path"]).unlink()

    with pytest.raises(ContractError, match="blinded frame artifact is unavailable"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_timestamp_projection_tamper(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["blinded_inputs"][0]["path"]
    blinded = json.loads(target.read_text(encoding="utf-8"))
    timestamp_path = target.parent.parent / blinded["timestamps_path"]
    timestamp_payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
    timestamp_payload["timestamps_seconds"][1] = 0.017
    timestamp_path.write_text(json.dumps(timestamp_payload, sort_keys=True), encoding="utf-8")
    blinded["timestamps_projection_sha256"] = _sha256_bytes(timestamp_path.read_bytes())
    target.write_text(json.dumps(blinded, sort_keys=True), encoding="utf-8")
    payload["blinded_inputs"][0]["sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="timestamp projection mismatch"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_return_contract_tamper(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["expected_return_contract_path"]
    contract = json.loads(target.read_text(encoding="utf-8"))
    contract["required_return_files"].remove("candidate-start-receipt.json")
    target.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    payload["expected_return_contract_sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="return contract is invalid"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_return_contract_packet_binding_tamper(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["expected_return_contract_path"]
    contract = json.loads(target.read_text(encoding="utf-8"))
    contract["required_packet_manifest_binding"]["required_return_manifest_fields"] = [
        "adjudicator_packet_manifest_sha256"
    ]
    target.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    payload["expected_return_contract_sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="return contract packet manifest binding is invalid"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_weak_trust_policy_template(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    target = receipt.parent / payload["trust_policy_template_path"]
    template = json.loads(target.read_text(encoding="utf-8"))
    template["external_approval_required"] = False
    target.write_text(json.dumps(template, sort_keys=True), encoding="utf-8")
    payload["trust_policy_template_sha256"] = _sha256_bytes(target.read_bytes())
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="trust policy template is invalid"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_blinded_gold_intake_rejects_scoreable_tamper(tmp_path: Path) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["scoreable"] = True
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="scoreability claim is invalid"):
        verify_native_main_blinded_gold_intake(receipt)


def test_native_main_adjudicator_packet_manifest_seals_full_packet_without_score_authority(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    manifest = seal_native_main_adjudicator_packet(
        intake_receipt=receipt,
        output_path=receipt.parent / "adjudicator-packet-manifest.json",
    )

    result = verify_native_main_adjudicator_packet_manifest(manifest)

    assert result["schema"] == "flashpatch-l7-native-main-adjudicator-packet-manifest-assessment-v1"
    assert result["status"] == "INDEPENDENT_GOLD_MISSING"
    assert result["scoreable"] is False
    assert result["external_claim_authorized"] is False
    assert result["case_count"] == 2
    assert result["packet_file_count"] == 11
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in payload["files"]}
    assert "adjudicator-packet/rubric.json" in paths
    assert "adjudicator-packet/expected-return-contract.json" in paths
    assert sum(path.startswith("adjudicator-packet/frames/") for path in paths) == 2
    assert sum(path.startswith("adjudicator-packet/timestamps/") for path in paths) == 2
    assert "independent_gold_missing" in result["score_blockers"]


def test_native_main_adjudicator_packet_manifest_rejects_packet_tamper(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    manifest = seal_native_main_adjudicator_packet(
        intake_receipt=receipt,
        output_path=receipt.parent / "adjudicator-packet-manifest.json",
    )
    target = receipt.parent / "adjudicator-packet" / "rubric.json"
    rubric = json.loads(target.read_text(encoding="utf-8"))
    rubric["instruction"] = "changed after seal"
    target.write_text(json.dumps(rubric, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="blinded gold rubric hash mismatch|manifest file list mismatch"):
        verify_native_main_adjudicator_packet_manifest(manifest)


def test_native_main_adjudicator_packet_manifest_rejects_missing_file(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    manifest = seal_native_main_adjudicator_packet(
        intake_receipt=receipt,
        output_path=receipt.parent / "adjudicator-packet-manifest.json",
    )
    target = next((receipt.parent / "adjudicator-packet" / "frames").glob("*.npz"))
    target.unlink()

    with pytest.raises(ContractError, match="blinded frame artifact is unavailable|manifest file list mismatch"):
        verify_native_main_adjudicator_packet_manifest(manifest)


def test_native_main_adjudicator_packet_manifest_cli_reports_not_scoreable(
    tmp_path: Path,
) -> None:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    receipt = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    manifest = receipt.parent / "adjudicator-packet-manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.competition",
            "seal-native-main-adjudicator-packet",
            "--intake-receipt",
            str(receipt),
            "--output",
            str(manifest),
        ],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "INDEPENDENT_GOLD_MISSING"
    assert payload["scoreable"] is False
    assert payload["packet_file_count"] == 11


def _native_main_unapproved_gold_return(tmp_path: Path) -> tuple[Path, Path, Path]:
    summary = _native_main_summary_for_gold_intake(tmp_path)
    intake = prepare_native_main_blinded_gold_intake(
        summary,
        tmp_path / "project" / "evidence" / "l7" / "gold-intake" / "v1",
        frozen_at="2026-08-04T00:00:00Z",
    )
    packet_manifest = seal_native_main_adjudicator_packet(
        intake_receipt=intake,
        output_path=intake.parent / "adjudicator-packet-manifest.json",
    )
    intake_payload = json.loads(intake.read_text(encoding="utf-8"))
    packet_payload = json.loads(packet_manifest.read_text(encoding="utf-8"))
    return_root = tmp_path / "project" / "evidence" / "l7" / "gold-return" / "v1"
    return_root.mkdir(parents=True)
    trust_policy = {
        "schema": "flashpatch-l7-gold-trust-policy-v1",
        "policy_id": "external-policy-test",
        "roster_id": "external-roster-test",
        "required_independent_submissions": 3,
        "adjudicators": [{"id": "judge-a"}, {"id": "judge-b"}, {"id": "judge-c"}],
        "timestamp_witnesses": [{"id": "witness-a"}, {"id": "witness-b"}],
        "flashpatch_operator_ids": ["flashpatch-operator"],
        "candidate_operator_ids": ["candidate-operator"],
        "revoked_at": None,
    }
    trust_policy_path = return_root / "external-trust-policy.json"
    trust_policy_path.write_text(json.dumps(trust_policy, sort_keys=True), encoding="utf-8")
    trust_policy_sha256 = _sha256_bytes(trust_policy_path.read_bytes())
    placeholder_signature = base64.b64encode(b"x" * 64).decode("ascii")
    entries = []
    for input_entry in intake_payload["blinded_inputs"]:
        blind_id = input_entry["blind_case_id"]
        case_root = return_root / blind_id
        case_root.mkdir()
        blinded_input = json.loads((intake.parent / input_entry["path"]).read_text(encoding="utf-8"))
        candidate_start = {
            "schema": "flashpatch-l7-candidate-start-receipt-v1",
            "blind_case_id": blind_id,
            "started_at": "2026-08-04T00:05:00Z",
            "state": "STARTED_OUTPUT_UNOPENED",
            "candidate_outputs_observed": False,
        }
        candidate_start_path = case_root / "candidate-start-receipt.json"
        candidate_start_path.write_text(json.dumps(candidate_start, sort_keys=True), encoding="utf-8")
        case_freeze = {
            "blind_case_id": blind_id,
            "source_summary_sha256": intake_payload["source_summary_sha256"],
            "corpus_commitment_sha256": intake_payload["corpus_commitment_sha256"],
            "frame_artifact_sha256": blinded_input["frame_artifact_sha256"],
            "renderer_rgb_raw_sha256": blinded_input["renderer_rgb_raw_sha256"],
            "timestamps_f64_sha256": blinded_input["timestamps_f64_sha256"],
            "frame_count": blinded_input["frame_count"],
            "color_space": blinded_input["color_space"],
            "frozen_at": "2026-08-04T00:00:00Z",
        }
        commitment = {
            "corpus_manifest_sha256": intake_payload["corpus_commitment_sha256"],
            "blinded_gold_input_sha256": input_entry["sha256"],
            "blind_mapping_sha256": intake_payload["sealed_mapping_sha256"],
            "case_freeze_sha256": _sha256_bytes(_canonical_bytes(case_freeze)),
            "trust_policy_sha256": trust_policy_sha256,
            "committed_at": "2026-08-04T00:01:00Z",
        }
        commitment_sha256 = _sha256_bytes(_canonical_bytes(commitment))
        gold = {
            "schema": "flashpatch-l7-independent-gold-v2",
            "case_id": blind_id,
            "case_class": "natural_external",
            "claim_tier": "L9_ELIGIBLE",
            "controlled_mutation": False,
            "case_freeze": case_freeze,
            "gold_authority": {
                "kind": "independent_adjudication",
                "candidate_tools_excluded": sorted(competition.GOLD_EXCLUDED_TOOLS),
                "required_independent_submissions": trust_policy["required_independent_submissions"],
                "trust_policy_sha256": trust_policy_sha256,
            },
            "pre_candidate_commitment": commitment,
            "commitment_witness": {
                "signer_id": "witness-a",
                "signer_role": "external_timestamp_witness",
                "pre_candidate_commitment_sha256": commitment_sha256,
                "signed_at": "2026-08-04T00:03:00Z",
                "signature_algorithm": "Ed25519",
                "signature_base64": placeholder_signature,
            },
            "adjudication": {
                "method": "independent-blinded-renderer-review",
                "result": "SAFE",
                "intervals": [],
                "submissions": [
                    {
                        "signer_id": f"judge-{index}",
                        "signer_role": "external_adjudicator",
                        "decision": "SAFE",
                        "intervals": [],
                        "pre_candidate_commitment_sha256": commitment_sha256,
                        "signed_at": "2026-08-04T00:02:00Z",
                        "signature_algorithm": "Ed25519",
                        "signature_base64": placeholder_signature,
                    }
                    for index in ("a", "b", "c")
                ],
            },
            "candidate_start_witness": {
                "signer_id": "witness-b",
                "signer_role": "external_timestamp_witness",
                "case_id": blind_id,
                "pre_candidate_commitment_sha256": commitment_sha256,
                "candidate_start_receipt_sha256": _sha256_bytes(candidate_start_path.read_bytes()),
                "signed_at": "2026-08-04T00:05:00Z",
                "signature_algorithm": "Ed25519",
                "signature_base64": placeholder_signature,
            },
        }
        gold_path = case_root / "independent-gold.json"
        gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
        entries.append(
            {
                "blind_case_id": blind_id,
                "independent_gold_path": str(gold_path.relative_to(return_root)),
                "independent_gold_sha256": _sha256_bytes(gold_path.read_bytes()),
                "trust_policy_path": str(trust_policy_path.relative_to(return_root)),
                "trust_policy_sha256": trust_policy_sha256,
                "candidate_start_receipt_path": str(candidate_start_path.relative_to(return_root)),
                "candidate_start_receipt_sha256": _sha256_bytes(candidate_start_path.read_bytes()),
            }
        )
    manifest = {
        "schema": "flashpatch-l7-native-main-independent-gold-return-v1",
        "status": "INDEPENDENT_GOLD_RETURNED_UNAPPROVED",
        "scoreable": False,
        "external_claim_authorized": False,
        "intake_receipt_sha256": _sha256_bytes(intake.read_bytes()),
        "adjudicator_packet_manifest_sha256": _sha256_bytes(packet_manifest.read_bytes()),
        "adjudicator_packet_root_sha256": packet_payload["packet_manifest_sha256"],
        "source_summary_sha256": intake_payload["source_summary_sha256"],
        "corpus_commitment_sha256": intake_payload["corpus_commitment_sha256"],
        "expected_return_contract_sha256": intake_payload["expected_return_contract_sha256"],
        "case_count": intake_payload["case_count"],
        "gold_receipts": entries,
    }
    (return_root / "native-main-independent-gold-return.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return intake, packet_manifest, return_root


def test_native_main_independent_gold_return_binds_blind_packet_but_stays_unapproved(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)

    result = verify_native_main_independent_gold_return(
        intake_receipt=intake,
        packet_manifest=packet_manifest,
        return_root=return_root,
    )

    assert result["schema"] == "flashpatch-l7-native-main-independent-gold-return-assessment-v1"
    assert result["status"] == "INDEPENDENT_GOLD_UNVERIFIED"
    assert result["scoreable"] is False
    assert result["approved_gold_count"] == 0
    assert result["adjudicator_packet_manifest_sha256"] == _sha256_bytes(packet_manifest.read_bytes())
    assert "independent_trust_policy_not_approved" in result["score_blockers"]


def test_native_main_independent_gold_return_rejects_candidate_output_leak(tmp_path: Path) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["candidate_predictions"] = {"FlashPatch": "SAFE"}
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="leaks candidate output"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_candidate_identity_as_signer(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["adjudication"]["submissions"][0]["signer_id"] = "FlashPatch"
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="not independent"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_operator_as_witness(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["candidate_start_witness"]["signer_id"] = "candidate-operator"
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="not independent"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_candidate_start_output_observed(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    start_path = return_root / entry["candidate_start_receipt_path"]
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["candidate_outputs_observed"] = True
    start_path.write_text(json.dumps(start, sort_keys=True), encoding="utf-8")
    entry["candidate_start_receipt_sha256"] = _sha256_bytes(start_path.read_bytes())
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["candidate_start_witness"]["candidate_start_receipt_sha256"] = entry[
        "candidate_start_receipt_sha256"
    ]
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="candidate start receipt is invalid"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_candidate_start_identity_mismatch(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    start_path = return_root / entry["candidate_start_receipt_path"]
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["blind_case_id"] = "blind-00000000000000000000000000000000"
    start_path.write_text(json.dumps(start, sort_keys=True), encoding="utf-8")
    entry["candidate_start_receipt_sha256"] = _sha256_bytes(start_path.read_bytes())
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["candidate_start_witness"]["candidate_start_receipt_sha256"] = entry[
        "candidate_start_receipt_sha256"
    ]
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="candidate start receipt is invalid"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_blinded_input_mismatch(tmp_path: Path) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["pre_candidate_commitment"]["blinded_gold_input_sha256"] = "0" * 64
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="not bound to intake"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_packet_manifest_mismatch(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adjudicator_packet_root_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="return manifest is invalid"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_rejects_unsigned_gold_skeleton(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    manifest_path = return_root / "native-main-independent-gold-return.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["gold_receipts"][0]
    gold_path = return_root / entry["independent_gold_path"]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    del gold["adjudication"]["submissions"][0]["signature_base64"]
    gold_path.write_text(json.dumps(gold, sort_keys=True), encoding="utf-8")
    entry["independent_gold_sha256"] = _sha256_bytes(gold_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="fields are not exact|signature"):
        verify_native_main_independent_gold_return(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
        )


def test_native_main_independent_gold_return_cli_reports_unapproved_not_scoreable(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.competition",
            "verify-native-main-gold-return",
            "--intake-receipt",
            str(intake),
            "--packet-manifest",
            str(packet_manifest),
            "--return-root",
            str(return_root),
        ],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "INDEPENDENT_GOLD_UNVERIFIED"
    assert payload["scoreable"] is False


def _native_main_candidate_start_gate_receipt(
    intake: Path,
    packet_manifest: Path,
    return_root: Path,
    *,
    detector_population: list[str] | None = None,
    slot_count: int = 9,
    repeats: int = 3,
) -> Path:
    intake_payload = json.loads(intake.read_text(encoding="utf-8"))
    packet_payload = json.loads(packet_manifest.read_text(encoding="utf-8"))
    manifest_path = return_root / "native-main-independent-gold-return.json"
    receipt = {
        "schema": "flashpatch-l7-native-main-candidate-start-gate-v1",
        "status": "CANDIDATE_START_REQUESTED",
        "scoreable": False,
        "external_claim_authorized": False,
        "intake_receipt_sha256": _sha256_bytes(intake.read_bytes()),
        "adjudicator_packet_manifest_sha256": _sha256_bytes(packet_manifest.read_bytes()),
        "adjudicator_packet_root_sha256": packet_payload["packet_manifest_sha256"],
        "gold_return_manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "source_summary_sha256": intake_payload["source_summary_sha256"],
        "corpus_commitment_sha256": intake_payload["corpus_commitment_sha256"],
        "detector_population": detector_population
        if detector_population is not None
        else [
            "FlashPatch",
            "KAYA_PSE_DETECTION_CORRECTION_0776EA3",
            "TooFlashy",
        ],
        "slot_count": slot_count,
        "repeat_count_per_detector": repeats,
        "request_schema": "flashpatch-l7-external-host-witness-request-v2",
        "wall_clock_budget_seconds": 60,
        "retry_rule": "exactly_one_visible_attempt_per_slot_no_hidden_retry",
        "process_started_at": "2026-08-03T00:06:00Z",
        "process_start_monotonic_ns": 123456789,
        "state": "STARTED_OUTPUT_UNOPENED",
        "candidate_outputs_observed": False,
    }
    path = return_root / "candidate-start-gate.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return path


def test_native_main_candidate_start_gate_blocks_until_independent_gold_verified(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    start = _native_main_candidate_start_gate_receipt(intake, packet_manifest, return_root)

    result = verify_native_main_candidate_start_gate(
        intake_receipt=intake,
        packet_manifest=packet_manifest,
        return_root=return_root,
        start_receipt=start,
    )

    assert result["status"] == "CANDIDATE_START_BLOCKED"
    assert result["scoreable"] is False
    assert result["reason"] == "independent_gold_not_verified_before_candidate_start"
    assert "independent_gold_not_verified" in result["score_blockers"]
    assert "independent_trust_policy_not_approved" in result["score_blockers"]


def test_native_main_candidate_start_gate_rejects_six_slot_or_missing_kaya_start(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    start = _native_main_candidate_start_gate_receipt(
        intake,
        packet_manifest,
        return_root,
        detector_population=["FlashPatch", "TooFlashy"],
        slot_count=6,
        repeats=3,
    )

    with pytest.raises(ContractError, match="candidate-start gate is invalid"):
        verify_native_main_candidate_start_gate(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
            start_receipt=start,
        )


def test_native_main_candidate_start_gate_rejects_observed_candidate_outputs(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    start = _native_main_candidate_start_gate_receipt(intake, packet_manifest, return_root)
    payload = json.loads(start.read_text(encoding="utf-8"))
    payload["candidate_outputs_observed"] = True
    start.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="candidate-start gate is invalid"):
        verify_native_main_candidate_start_gate(
            intake_receipt=intake,
            packet_manifest=packet_manifest,
            return_root=return_root,
            start_receipt=start,
        )


def test_native_main_candidate_start_gate_cli_reports_blocked_not_scoreable(
    tmp_path: Path,
) -> None:
    intake, packet_manifest, return_root = _native_main_unapproved_gold_return(tmp_path)
    start = _native_main_candidate_start_gate_receipt(intake, packet_manifest, return_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.competition",
            "verify-native-main-candidate-start-gate",
            "--intake-receipt",
            str(intake),
            "--packet-manifest",
            str(packet_manifest),
            "--return-root",
            str(return_root),
            "--start-receipt",
            str(start),
        ],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "CANDIDATE_START_BLOCKED"
    assert payload["scoreable"] is False
