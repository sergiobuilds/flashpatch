from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flashpatch.competition import claim_gate
from flashpatch.external_league import DIRECT_DETECTOR_POPULATION


HASH_FIELDS = (
    "natural_case_ledger_sha256",
    "renderer_rgb_sha256",
    "timestamps_sha256",
    "canonical_video_sha256",
    "fair_runtime_input_sha256",
    "fair_runtime_external_slot_child_joins_sha256",
    "independent_gold_receipt_sha256",
    "tool_parity_receipt_sha256",
    "fair_runtime_receipt_sha256",
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_receipt(*, external_claim_authorized: bool = False) -> dict[str, object]:
    population = list(DIRECT_DETECTOR_POPULATION)
    population_sha256 = hashlib.sha256(
        json.dumps(population, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for ordinal in range(9):
        row: dict[str, object] = {
            "case_id": f"blind-{ordinal}",
            "repository_url": f"https://github.com/example/repository-{ordinal}",
            "repository_revision": f"revision-{ordinal}",
            "detector_population": population,
            "detector_population_sha256": population_sha256,
            "fair_runtime_external_slot_child_join_count": 9,
        }
        row.update({field: f"{ordinal + 1:064x}" for field in HASH_FIELDS})
        rows.append(row)
    return {
        "schema": "flashpatch-l7-receipt-bound-statistics-v1",
        "status": "RECEIPT_BOUND_STATISTICS_VERIFIED",
        "claim_status": "SCOREABLE",
        "scoreable": True,
        "external_claim_authorized": external_claim_authorized,
        "case_count": 9,
        "public_repository_count": 9,
        "minimum_scoreable_natural_cases": 9,
        "minimum_scoreable_public_repositories": 3,
        "detector_population": population,
        "detector_population_sha256": population_sha256,
        "case_receipts": rows,
        "bootstrap_lanes_separate": True,
    }


def _write_bound_manifest(tmp_path: Path, receipt: dict[str, object]) -> tuple[Path, Path]:
    receipt_path = tmp_path / "score-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema": "flashpatch-l9-evidence-bundle-v1",
        "frozen": True,
        "l7_score_receipt": {
            "path": receipt_path.name,
            "sha256": _sha256_bytes(receipt_path),
        },
    }
    manifest_path = tmp_path / "claim-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, receipt_path


def test_claim_gate_rejects_unscoreable_l7(tmp_path: Path) -> None:
    l7_bundle = {
        "schema": "flashpatch-l7-receipt-bound-statistics-v1",
        "scoreable": False,
        "claim_status": "NOT_SCOREABLE",
    }
    l7_path = tmp_path / "score-receipt.json"
    l7_path.write_text(json.dumps(l7_bundle))

    l9_bundle = {
        "schema": "flashpatch-l9-evidence-bundle-v1",
        "frozen": True,
        "l7_score_receipt": str(l7_path),
    }
    l9_path = tmp_path / "claim-manifest.json"
    l9_path.write_text(json.dumps(l9_bundle))

    code, msg = claim_gate(l9_path)
    assert code == 1
    assert "NOT_CLAIMABLE: L7 is NOT_SCOREABLE" in msg


def test_claim_gate_rejects_tampered_hash_bound_receipt(tmp_path: Path) -> None:
    manifest_path, receipt_path = _write_bound_manifest(tmp_path, _score_receipt())
    receipt_path.write_text(receipt_path.read_text() + "\n", encoding="utf-8")

    code, msg = claim_gate(manifest_path)

    assert code == 1
    assert msg == "NOT_CLAIMABLE: L7 score receipt hash mismatch"


def test_claim_gate_rejects_missing_receipt(tmp_path: Path) -> None:
    manifest = {
        "schema": "flashpatch-l9-evidence-bundle-v1",
        "frozen": True,
        "l7_score_receipt": {"path": "missing.json", "sha256": "0" * 64},
    }
    manifest_path = tmp_path / "claim-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code, msg = claim_gate(manifest_path)

    assert code == 1
    assert msg.startswith("NOT_CLAIMABLE: L7 score receipt is unavailable")


def test_claim_gate_recomputes_counts_and_honors_false_authorization(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_bound_manifest(tmp_path, _score_receipt())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "case_count": 999,
            "public_repository_count": 999,
            "statistical_gate_passed": True,
            "external_claim_authorized": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code, msg = claim_gate(manifest_path)

    assert code == 1
    assert msg == (
        "NOT_CLAIMABLE: external_claim_authorized=false cases=9 repositories=9"
    )


def test_claim_gate_rejects_receipt_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-score-receipt.json"
    outside.write_text(json.dumps(_score_receipt()), encoding="utf-8")
    manifest = {
        "schema": "flashpatch-l9-evidence-bundle-v1",
        "frozen": True,
        "l7_score_receipt": {
            "path": f"../{outside.name}",
            "sha256": _sha256_bytes(outside),
        },
    }
    manifest_path = tmp_path / "claim-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code, msg = claim_gate(manifest_path)

    assert code == 1
    assert msg == "NOT_CLAIMABLE: L7 score receipt path must be bundle-relative"
