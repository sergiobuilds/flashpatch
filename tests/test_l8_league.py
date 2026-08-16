from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from flashpatch.l8_league import (
    L8LeagueError,
    aggregate_artifact_league,
    prepare_artifact_league,
    reveal_artifact_league,
)


ROOT = Path(__file__).resolve().parents[1]
AXES = ["axis_1", "axis_2", "axis_3", "axis_4"]
NARRATIVE = [
    "problem",
    "user",
    "experience",
    "mechanism",
    "required_technology",
    "business_case",
    "build_scope",
    "proof_plan",
]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _card(tag: str) -> dict[str, Any]:
    card = {
        field: (
            f"{field} gives a symmetric, evidence-bounded artifact account for {tag}; "
            "the wording and disclosure budget remain neutral."
        )
        for field in NARRATIVE
    }
    card.update(
        {
            "observed_evidence": [f"frozen observation for {tag}"],
            "claims": [f"bounded source claim for {tag}"],
            "inferences": [f"neutral inference for {tag}"],
            "missing_evidence": [f"common missing proof for {tag}"],
        }
    )
    return card


def _inputs(tmp_path: Path) -> dict[str, Any]:
    candidates = [
        {
            "identity": {"name": "Vendor One", "url": "https://one.invalid/source"},
            "card": _card("candidate-a"),
        },
        {
            "identity": {"name": "Vendor Two", "url": "https://two.invalid/source"},
            "card": _card("candidate-b"),
        },
    ]
    anchors = [
        {"level": "low", "expected_total": 20, "card": _card("anchor-low")},
        {"level": "mid", "expected_total": 50, "card": _card("anchor-mid")},
        {"level": "high", "expected_total": 80, "card": _card("anchor-high")},
    ]
    rubric = {
        "axes": [
            {
                "id": axis,
                "label": f"Artifact axis {index}",
                "min_score": 0,
                "max_score": 25,
                "anchors": {"0": "no verified support", "25": "strong verified support"},
            }
            for index, axis in enumerate(AXES, start=1)
        ],
        "hard_gates": ["artifact opens at frozen revision"],
    }
    preregistration = {
        "evaluation_class": "ARTIFACT",
        "decision": "select the strongest measured artifact",
        "population_label": "two frozen artifacts at 2026-08-11",
        "evidence_cutoff": "2026-08-11T00:00:00Z",
        "information_parity": {
            "min_total_characters": 700,
            "max_total_characters": 1400,
            "field_min_characters": 50,
            "field_max_characters": 180,
            "narrative_fields": NARRATIVE,
        },
        "scorer_model_identifiers": ["frozen-scorer-v1"],
        "disagreement_rule": {
            "total_raw_range": 20,
            "axis_raw_range": 8,
            "pairwise_position_win_rate": 0.65,
        },
        "tie_rule": "report tied rank without a winner",
        "audit_promotion_count": 2,
        "pairwise_plan": "all finalists in A/B and B/A orientation",
        "source_locked_sensitivity_plan": "truncate to common source floor",
        "parity_reconstructed_sensitivity_plan": "repeat with reconstructed cards",
    }
    reconstruction = [
        {
            "candidate_index": index,
            "status": "SCOREABLE",
            "source_tier": "repository_and_demo",
            "research_budget_units": 3,
            "assumptions": [],
            "card_sha256": _canonical_hash(candidate["card"]),
        }
        for index, candidate in enumerate(candidates)
    ]
    return {
        "candidates": candidates,
        "anchors": anchors,
        "rubric": rubric,
        "preregistration": preregistration,
        "reconstruction": reconstruction,
        "paths": {
            "candidates": _write(tmp_path / "candidates.json", candidates),
            "anchors": _write(tmp_path / "anchors.json", anchors),
            "rubric": _write(tmp_path / "rubric.json", rubric),
            "preregistration": _write(tmp_path / "preregistration.json", preregistration),
            "reconstruction": _write(tmp_path / "reconstruction.json", reconstruction),
        },
    }


def _prepare(tmp_path: Path, name: str = "run") -> tuple[Path, dict[str, Any], dict[str, Any]]:
    inputs = _inputs(tmp_path)
    run = tmp_path / name
    manifest = prepare_artifact_league(
        candidates_path=inputs["paths"]["candidates"],
        anchors_path=inputs["paths"]["anchors"],
        rubric_path=inputs["paths"]["rubric"],
        preregistration_path=inputs["paths"]["preregistration"],
        reconstruction_path=inputs["paths"]["reconstruction"],
        out=run,
        seed=20260811,
    )
    return run, manifest, inputs


def _results(manifest: dict[str, Any], *, pairwise: bool = True) -> dict[str, Any]:
    candidate_ids = sorted(
        {item["blind_id"] for item in manifest["assignments"] if item["kind"] == "candidate"}
    )
    candidate_base = {candidate_ids[0]: 80.0, candidate_ids[1]: 60.0}
    observed_lane_bias = {"lane-1": -4.0, "lane-2": 0.0, "lane-3": 4.0}
    results = []
    for assignment in manifest["assignments"]:
        if assignment["kind"] == "anchor":
            total = assignment["expected_total"] + observed_lane_bias[assignment["lane"]]
        else:
            total = candidate_base[assignment["blind_id"]] + observed_lane_bias[assignment["lane"]]
        results.append(
            {
                "assignment_id": assignment["assignment_id"],
                "blind_id": assignment["blind_id"],
                "scores": {axis: total / 4 for axis in AXES},
                "reason": "frozen evidence supports this score",
                "evidence_gaps": [],
            }
        )
    pairwise_results = []
    if pairwise:
        high, low = candidate_ids
        pairwise_results = [
            {
                "pair_id": "pair-ab",
                "left_blind_id": high,
                "right_blind_id": low,
                "winner_blind_id": high,
            },
            {
                "pair_id": "pair-ba",
                "left_blind_id": low,
                "right_blind_id": high,
                "winner_blind_id": high,
            },
        ]
    return {"results": results, "pairwise_results": pairwise_results}


def test_prepare_is_deterministic_private_and_balanced(tmp_path: Path) -> None:
    first, first_manifest, inputs = _prepare(tmp_path, "run-a")
    second = tmp_path / "run-b"
    second_manifest = prepare_artifact_league(
        candidates_path=inputs["paths"]["candidates"],
        anchors_path=inputs["paths"]["anchors"],
        rubric_path=inputs["paths"]["rubric"],
        preregistration_path=inputs["paths"]["preregistration"],
        reconstruction_path=inputs["paths"]["reconstruction"],
        out=second,
        seed=20260811,
    )
    assert first_manifest == second_manifest
    assert (first / "cell-packets.json").read_bytes() == (second / "cell-packets.json").read_bytes()
    for lane in ("lane-1", "lane-2", "lane-3"):
        packet = json.loads((first / "cells" / f"{lane}.json").read_text(encoding="utf-8"))
        candidates = [item for item in packet["items"] if item["kind"] == "candidate"]
        anchors = [item for item in packet["items"] if item["kind"] == "anchor"]
        assert len(candidates) == 2
        assert len({item["blind_id"] for item in candidates}) == 2
        assert {item["blind_id"] for item in anchors} == {
            "ANCHOR-LOW",
            "ANCHOR-MID",
            "ANCHOR-HIGH",
        }
        public_text = json.dumps(packet).lower()
        assert "vendor one" not in public_text
        assert "vendor two" not in public_text
    assert (first / "private" / "identity-mapping.json").is_file()
    manifest_text = (first / "freeze-manifest.json").read_text(encoding="utf-8").lower()
    assert "vendor one" not in manifest_text
    assert "vendor two" not in manifest_text


def test_aggregate_corrects_anchor_severity_and_reveals_only_after_seal(tmp_path: Path) -> None:
    run, manifest, _ = _prepare(tmp_path)
    result_path = _write(tmp_path / "results.json", _results(manifest))
    aggregate = aggregate_artifact_league(run=run, results_path=result_path)
    assert aggregate["state"] == "SEALED"
    assert aggregate["lane_anchor_corrections"] == {
        "lane-1": 4.0,
        "lane-2": 0.0,
        "lane-3": -4.0,
    }
    assert [item["corrected_total_median"] for item in aggregate["candidates"]] == [80.0, 60.0]
    assert aggregate["warnings"] == []
    public_text = (run / "aggregate.json").read_text(encoding="utf-8").lower()
    assert "vendor one" not in public_text and "vendor two" not in public_text

    revealed = reveal_artifact_league(run=run, aggregate_path=run / "aggregate.json")
    assert revealed["state"] == "REVEALED_AFTER_SEAL"
    assert {item["identity"]["name"] for item in revealed["candidates"]} == {
        "Vendor One",
        "Vendor Two",
    }


@pytest.mark.parametrize("mutation", ["unscoreable", "unequal_budget", "identity_leak"])
def test_prepare_rejects_reconstruction_or_identity_asymmetry(
    tmp_path: Path, mutation: str
) -> None:
    inputs = _inputs(tmp_path)
    if mutation == "unscoreable":
        inputs["reconstruction"][1]["status"] = "UNSCORABLE"
        _write(inputs["paths"]["reconstruction"], inputs["reconstruction"])
    elif mutation == "unequal_budget":
        inputs["reconstruction"][1]["research_budget_units"] = 4
        _write(inputs["paths"]["reconstruction"], inputs["reconstruction"])
    else:
        inputs["candidates"][0]["card"]["problem"] += " Vendor One"
        _write(inputs["paths"]["candidates"], inputs["candidates"])
    with pytest.raises(L8LeagueError):
        prepare_artifact_league(
            candidates_path=inputs["paths"]["candidates"],
            anchors_path=inputs["paths"]["anchors"],
            rubric_path=inputs["paths"]["rubric"],
            preregistration_path=inputs["paths"]["preregistration"],
            reconstruction_path=inputs["paths"]["reconstruction"],
            out=tmp_path / "run",
            seed=1,
        )


def test_aggregate_requires_every_assignment_and_all_four_axes(tmp_path: Path) -> None:
    run, manifest, _ = _prepare(tmp_path)
    result_document = _results(manifest)
    result_document["results"][0]["scores"].pop("axis_4")
    result_path = _write(tmp_path / "results.json", result_document)
    with pytest.raises(L8LeagueError, match="exact four axes"):
        aggregate_artifact_league(run=run, results_path=result_path)
    assert not (run / "aggregate.json").exists()
    assert not (run / "sealed-results.json").exists()


def test_disagreement_and_pairwise_position_bias_are_sealed_as_warnings(tmp_path: Path) -> None:
    run, manifest, _ = _prepare(tmp_path)
    result_document = _results(manifest)
    candidate_id = next(
        item["blind_id"] for item in manifest["assignments"] if item["kind"] == "candidate"
    )
    candidate_rows = [
        row for row in result_document["results"] if row["blind_id"] == candidate_id
    ]
    candidate_rows[0]["scores"] = {axis: 0 for axis in AXES}
    candidate_rows[1]["scores"] = {axis: 25 for axis in AXES}
    for row in result_document["pairwise_results"]:
        row["winner_blind_id"] = row["left_blind_id"]
    result_path = _write(tmp_path / "results.json", result_document)
    aggregate = aggregate_artifact_league(run=run, results_path=result_path)
    codes = {warning["code"] for warning in aggregate["warnings"]}
    assert "TOTAL_DISAGREEMENT_ADJUDICATION_REQUIRED" in codes
    assert "AXIS_DISAGREEMENT_ADJUDICATION_REQUIRED" in codes
    assert "PAIRWISE_ORIENTATION_BIAS" in codes


def test_two_candidate_pairwise_contract_rejects_missing_reverse_orientation(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["preregistration"]["pairwise_contract"] = "two_candidate_reversed"
    _write(inputs["paths"]["preregistration"], inputs["preregistration"])
    run = tmp_path / "run"
    manifest = prepare_artifact_league(
        candidates_path=inputs["paths"]["candidates"],
        anchors_path=inputs["paths"]["anchors"],
        rubric_path=inputs["paths"]["rubric"],
        preregistration_path=inputs["paths"]["preregistration"],
        reconstruction_path=inputs["paths"]["reconstruction"],
        out=run,
        seed=20260811,
    )
    requirements = {
        lane: json.loads((run / "cells" / f"{lane}.json").read_text())["pairwise_requirements"]
        for lane in ("lane-1", "lane-2", "lane-3")
    }
    assert len(requirements["lane-1"]) == len(requirements["lane-2"]) == 1
    assert requirements["lane-3"] == []
    results = _results(manifest)
    results["pairwise_results"] = results["pairwise_results"][:1]
    with pytest.raises(L8LeagueError, match="frozen A/B and B/A"):
        aggregate_artifact_league(run=run, results_path=_write(tmp_path / "results.json", results))


def test_reveal_rejects_aggregate_tampering(tmp_path: Path) -> None:
    run, manifest, _ = _prepare(tmp_path)
    result_path = _write(tmp_path / "results.json", _results(manifest))
    aggregate_artifact_league(run=run, results_path=result_path)
    aggregate_path = run / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["candidates"][0]["corrected_total_median"] = 0
    _write(aggregate_path, aggregate)
    with pytest.raises(L8LeagueError, match="aggregate seal"):
        reveal_artifact_league(run=run, aggregate_path=aggregate_path)


def test_project_local_cli_is_available_through_competition(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.competition",
            "prepare-artifact-league",
            "--candidates",
            str(inputs["paths"]["candidates"]),
            "--anchors",
            str(inputs["paths"]["anchors"]),
            "--rubric",
            str(inputs["paths"]["rubric"]),
            "--preregistration",
            str(inputs["paths"]["preregistration"]),
            "--reconstruction",
            str(inputs["paths"]["reconstruction"]),
            "--out",
            str(tmp_path / "cli-run"),
            "--seed",
            "20260811",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "SEALED artifact-league lanes=3 candidates=2\n"
