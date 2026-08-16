from __future__ import annotations

import json
from pathlib import Path

import pytest
from flashpatch.competition import ContractError
from flashpatch.external_league import DIRECT_DETECTOR_POPULATION
from flashpatch.l7_score import (
    BOOTSTRAP_ITERATIONS,
    DIRECT_DETECTOR_POPULATION_SHA256,
    L7ScoreError,
    generate_l8_handoff,
)

def test_generate_l8_handoff_rejects_unscoreable_bundle(tmp_path: Path) -> None:
    bundle = {
        "schema": "flashpatch-l7-receipt-bound-statistics-v1",
        "scoreable": False,
        "claim_status": "NOT_SCOREABLE",
    }
    bundle_path = tmp_path / "score-receipt.json"
    bundle_path.write_text(json.dumps(bundle))

    with pytest.raises(ContractError, match="NO_HANDOFF"):
        generate_l8_handoff(bundle_path, tmp_path / "handoffs")


def _scoreable_receipt() -> dict[str, object]:
    primary = {
        comparator: {
            "case_count": 9,
            "exact_matches": 9,
            "exact_agreement": 1.0,
            "false_negative_count": 0,
            "false_positive_count": 0,
            "disagreement_case_count": 0,
            "disagreement_cases": [],
            "bootstrap": {
                "iterations": BOOTSTRAP_ITERATIONS,
                "metric": "case_level_exact_agreement",
                "percentile_95_interval": [1.0, 1.0],
            },
        }
        for comparator in DIRECT_DETECTOR_POPULATION
    }
    secondary = {
        comparator: {
            "eligible_case_count": 9,
            "true_positive_frames": 0,
            "false_positive_frames": 0,
            "false_negative_frames": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_absolute_onset_error_frames": None,
            "onset_comparable_case_count": 0,
            "bootstrap": {
                "iterations": BOOTSTRAP_ITERATIONS,
                "lane": "secondary_interval_only",
                "percentile_95_intervals": {},
            },
        }
        for comparator in DIRECT_DETECTOR_POPULATION
    }
    runtime = {
        comparator: {
            "case_count": 9,
            "repeat_count": 27,
            "total_wall_time_ns": 27_000,
            "mean_wall_time_ns": 1_000.0,
            "min_wall_time_ns": 900,
            "max_wall_time_ns": 1_100,
        }
        for comparator in DIRECT_DETECTOR_POPULATION
    }
    return {
        "schema": "flashpatch-l7-receipt-bound-statistics-v1",
        "status": "RECEIPT_BOUND_STATISTICS_VERIFIED",
        "case_count": 9,
        "minimum_scoreable_natural_cases": 9,
        "minimum_scoreable_public_repositories": 3,
        "public_repository_count": 3,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "detector_population_sha256": DIRECT_DETECTOR_POPULATION_SHA256,
        "case_receipts": [
            {
                "case_id": f"case-{index}",
                "natural_case_ledger_sha256": f"{index + 1:064x}",
            }
            for index in range(9)
        ],
        "primary_case_level": primary,
        "secondary_interval": secondary,
        "secondary_case_diagnostics": {
            comparator: [] for comparator in DIRECT_DETECTOR_POPULATION
        },
        "runtime": runtime,
        "bootstrap_lanes_separate": True,
        "claim_status": "SCOREABLE",
        "scoreable": True,
        "scoreability_blockers": [],
        "external_claim_authorized": False,
    }


def test_generate_l8_handoff_emits_identity_free_detector_card_only(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "score-receipt.json"
    bundle_path.write_text(json.dumps(_scoreable_receipt(), sort_keys=True), encoding="utf-8")

    outputs = generate_l8_handoff(bundle_path, tmp_path / "handoffs")

    assert len(outputs) == 1
    handoff_text = outputs[0].read_text(encoding="utf-8")
    handoff = json.loads(handoff_text)
    assert handoff["schema"] == "flashpatch-l8-identity-free-detector-handoff-v1"
    assert handoff["lane"] == "direct_detector"
    assert handoff["identity_free"] is True
    assert handoff["detector_identity_mapping_included"] is False
    assert handoff["external_claim_authorized"] is False
    assert handoff["combined_winner_authorized"] is False
    assert "winner" not in handoff
    assert "ranking" not in handoff
    assert [cell["cell_id"] for cell in handoff["cells"]] == [
        "detector_cell_1",
        "detector_cell_2",
        "detector_cell_3",
    ]
    for comparator in DIRECT_DETECTOR_POPULATION:
        assert comparator not in handoff_text


@pytest.mark.parametrize(
    "patch",
    [
        {"external_claim_authorized": True},
        {"claim_status": "NOT_SCOREABLE"},
        {"ranking": ["FlashPatch"]},
        {"winner": "FlashPatch"},
    ],
)
def test_generate_l8_handoff_rejects_claim_or_winner_surfaces(
    tmp_path: Path, patch: dict[str, object]
) -> None:
    receipt = _scoreable_receipt()
    receipt.update(patch)
    bundle_path = tmp_path / "score-receipt.json"
    bundle_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    with pytest.raises(L7ScoreError):
        generate_l8_handoff(bundle_path, tmp_path / "handoffs")
