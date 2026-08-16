from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flashpatch import l7_verify
import flashpatch.l7_score as l7_score
from flashpatch.external_league import DIRECT_DETECTOR_POPULATION
from flashpatch.external_league import EA_IRIS_RELEASE_ORACLE_ID
from flashpatch.l7_score import (
    BOOTSTRAP_ITERATIONS,
    CAPABILITY_FAILURE,
    DIRECT_DETECTOR_POPULATION_SHA256,
    GOLD_FAILURE,
    L7ScoreError,
    SCORE_BUNDLE_SCHEMA,
    verify_score_bundle,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _case(
    case_id: str,
    gold: str,
    predictions: dict[str, str],
    *,
    repository_index: int = 0,
    repository_url: object | None = None,
    repository_revision: object | None = None,
) -> dict[str, object]:
    observations = {
        comparator: {
            "prediction": predictions[comparator],
            "hazard_frame_indices": (
                None
                if comparator == "TooFlashy"
                else ([1, 2] if predictions[comparator] == "HAZARDOUS" else [])
            ),
        }
        for comparator in DIRECT_DETECTOR_POPULATION
    }
    intervals = (
        [{"kind": "flash", "start_seconds": 0.01, "end_seconds": 0.04}]
        if gold == "HAZARDOUS"
        else []
    )
    renderer_rgb_sha256 = hashlib.sha256((case_id + "rgb").encode()).hexdigest()
    timestamps_sha256 = hashlib.sha256((case_id + "timestamps").encode()).hexdigest()
    canonical_video_sha256 = hashlib.sha256((case_id + "video").encode()).hexdigest()
    fair_join_rows = [
        {
            "slot": slot,
            "comparator": DIRECT_DETECTOR_POPULATION[(slot - 1) // 3],
            "repeat_ordinal": ((slot - 1) % 3) + 1,
            "external_result_sha256": f"{slot:064x}",
            "child_receipt_sha256": hashlib.sha256(f"{case_id}-child-{slot}".encode()).hexdigest(),
        }
        for slot in range(1, 10)
    ]
    return {
        "natural": {
            "assessment": {
                "ledger_sha256": hashlib.sha256(case_id.encode()).hexdigest(),
                "renderer_rgb_sha256": renderer_rgb_sha256,
                "timestamps_sha256": timestamps_sha256,
            },
            "ledger": {"controlled_mutation": False},
            "repository": {
                "url": (
                    repository_url
                    if repository_url is not None
                    else f"https://github.com/example/game-{repository_index}"
                ),
                "revision": (
                    repository_revision
                    if repository_revision is not None
                    else f"{repository_index:040d}"[-40:]
                ),
            },
        },
        "gold": {
            "case_id": case_id,
            "controlled_mutation": False,
            "decision": gold,
            "intervals": intervals,
            "timestamps_seconds": [0.0, 0.02, 0.03, 0.05],
            "receipt": {"sha256": hashlib.sha256((case_id + "gold").encode()).hexdigest()},
        },
        "parity": {
            "receipt": {"sha256": hashlib.sha256((case_id + "parity").encode()).hexdigest()},
            "contract": {"canonical_video": {"sha256": canonical_video_sha256}},
            "observations": observations,
        },
        "fair": {
            "receipt": {"sha256": hashlib.sha256((case_id + "fair").encode()).hexdigest()},
            "input_sha256": canonical_video_sha256,
            "external_slot_child_joins": fair_join_rows,
            "observations": copy.deepcopy(observations),
            "runtime_wall_time_ns": {
                comparator: [
                    (repository_index + 1) * 1_000 + repeat
                    for repeat in (1, 2, 3)
                ]
                for comparator in DIRECT_DETECTOR_POPULATION
            },
        },
    }


def _mock_reopeners(monkeypatch: pytest.MonkeyPatch, cases: list[dict[str, object]]) -> None:
    by_token = {str(index): case for index, case in enumerate(cases)}
    monkeypatch.setattr(l7_score, "_natural_projection", lambda ref: by_token[ref["path"]]["natural"])
    monkeypatch.setattr(l7_score, "_gold_projection", lambda ref: by_token[ref["receipt"]]["gold"])
    monkeypatch.setattr(l7_score, "_reopen_tool_parity", lambda ref: by_token[ref["path"]]["parity"])
    monkeypatch.setattr(l7_score, "_reopen_fair_runtime", lambda ref: by_token[ref["path"]]["fair"])
    monkeypatch.setattr(l7_score, "_match_natural_and_gold", lambda natural, gold: None)
    monkeypatch.setattr(l7_score, "_match_case_inputs", lambda gold, parity, fair: None)


def _bundle(path: Path, count: int, *, extra: dict[str, object] | None = None) -> Path:
    payload: dict[str, object] = {
        "schema": SCORE_BUNDLE_SCHEMA,
        "cases": [
            {
                "natural_case": {"path": str(index), "sha256": "a" * 64},
                "independent_gold": {"receipt": str(index), "trust_policy": str(index)},
                "tool_parity": {"path": str(index), "sha256": "b" * 64},
                "fair_runtime": {"path": str(index), "sha256": "c" * 64},
            }
            for index in range(count)
        ],
    }
    if extra:
        payload.update(extra)
    return _write(path, payload)


def test_l7_verify_bundle_cli_reports_missing_bundle_not_scoreable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "score_bundle.json"

    code = l7_verify.main(["--bundle", str(missing), "--bootstrap", "10000"])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {
        "status": "NOT_SCOREABLE",
        "reason": "RECEIPT_BOUND_SCORE_BUNDLE_MISSING",
        "l8_handoff_eligible": False,
    }
    assert "score bundle missing" in captured.err


def test_l7_verify_bundle_cli_rejects_noncanonical_bootstrap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _write(tmp_path / "score_bundle.json", {"schema": SCORE_BUNDLE_SCHEMA, "cases": []})

    code = l7_verify.main(["--bundle", str(bundle), "--bootstrap", "9999"])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {
        "status": "NOT_SCOREABLE",
        "reason": "BOOTSTRAP_ITERATION_COUNT_INVALID",
        "l8_handoff_eligible": False,
    }
    assert "bootstrap must be exactly 10000" in captured.err


def test_l7_verify_bundle_cli_emits_scoreable_handoff_only_from_scoreable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _write(tmp_path / "score_bundle.json", {"schema": SCORE_BUNDLE_SCHEMA, "cases": []})
    monkeypatch.setattr(
        l7_score,
        "verify_score_bundle",
        lambda path: {
            "scoreable": True,
            "claim_status": "SCOREABLE",
            "case_count": 9,
        },
    )

    code = l7_verify.main(["--bundle", str(bundle), "--bootstrap", "10000"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == (
        '{"status":"SCOREABLE","cases":9,"slots":9,'
        '"bootstrap":10000,"l8_handoff_eligible":true}\n'
    )
    assert captured.err == ""


def test_l7_verify_bundle_cli_accepts_more_than_minimum_scoreable_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _write(tmp_path / "score_bundle.json", {"schema": SCORE_BUNDLE_SCHEMA, "cases": []})
    monkeypatch.setattr(
        l7_score,
        "verify_score_bundle",
        lambda path: {
            "scoreable": True,
            "claim_status": "SCOREABLE",
            "case_count": 10,
        },
    )

    code = l7_verify.main(["--bundle", str(bundle), "--bootstrap", "10000"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == (
        '{"status":"SCOREABLE","cases":10,"slots":9,'
        '"bootstrap":10000,"l8_handoff_eligible":true}\n'
    )
    assert captured.err == ""


def test_l7_verify_bundle_cli_keeps_gold_failure_not_scoreable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _write(tmp_path / "score_bundle.json", {"schema": SCORE_BUNDLE_SCHEMA, "cases": []})

    def fail_gold(path: Path) -> dict[str, object]:
        raise L7ScoreError(GOLD_FAILURE)

    monkeypatch.setattr(l7_score, "verify_score_bundle", fail_gold)

    code = l7_verify.main(["--bundle", str(bundle), "--bootstrap", "10000"])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out) == {
        "status": "NOT_SCOREABLE",
        "reason": "INDEPENDENT_GOLD_MISSING",
        "l8_handoff_eligible": False,
    }
    assert GOLD_FAILURE in captured.err


def test_receipt_bound_statistics_keep_primary_secondary_and_bootstrap_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        _case(
            "case-safe",
            "SAFE",
            {"FlashPatch": "SAFE", DIRECT_DETECTOR_POPULATION[1]: "SAFE", "TooFlashy": "HAZARDOUS"},
        ),
        _case(
            "case-hazard",
            "HAZARDOUS",
            {"FlashPatch": "HAZARDOUS", DIRECT_DETECTOR_POPULATION[1]: "SAFE", "TooFlashy": "HAZARDOUS"},
        ),
    ]
    _mock_reopeners(monkeypatch, cases)
    bundle = _bundle(tmp_path / "bundle.json", len(cases))

    first = verify_score_bundle(bundle)
    second = verify_score_bundle(bundle)

    assert first == second
    assert first["status"] == "RECEIPT_BOUND_STATISTICS_VERIFIED"
    assert first["claim_status"] == "NOT_SCOREABLE"
    assert first["scoreable"] is False
    assert first["minimum_scoreable_natural_cases"] == 9
    assert first["minimum_scoreable_public_repositories"] == 3
    assert first["public_repository_count"] == 1
    assert first["detector_population"] == list(DIRECT_DETECTOR_POPULATION)
    assert first["detector_population_sha256"] == DIRECT_DETECTOR_POPULATION_SHA256
    assert first["scoreability_blockers"] == [
        "minimum_nine_qualified_natural_cases_not_met",
        "minimum_three_public_repositories_not_met",
    ]
    assert first["external_claim_authorized"] is False
    assert first["primary_case_level"]["FlashPatch"]["exact_agreement"] == 1.0
    assert first["primary_case_level"][DIRECT_DETECTOR_POPULATION[1]]["exact_agreement"] == 0.5
    assert first["primary_case_level"][DIRECT_DETECTOR_POPULATION[1]]["false_negative_count"] == 1
    assert first["primary_case_level"][DIRECT_DETECTOR_POPULATION[1]]["false_positive_count"] == 0
    assert first["primary_case_level"][DIRECT_DETECTOR_POPULATION[1]]["disagreement_cases"] == [
        {
            "case_id": "case-hazard",
            "gold_decision": "HAZARDOUS",
            "predicted_decision": "SAFE",
            "error_type": "false_negative",
        }
    ]
    assert first["primary_case_level"]["TooFlashy"]["false_positive_count"] == 1
    assert first["primary_case_level"]["TooFlashy"]["false_negative_count"] == 0
    assert first["secondary_interval"]["FlashPatch"]["precision"] == 1.0
    assert first["secondary_interval"]["FlashPatch"]["recall"] == 1.0
    assert first["secondary_interval"]["FlashPatch"]["f1"] == 1.0
    assert first["secondary_case_diagnostics"]["FlashPatch"][1] == {
        "case_id": "case-hazard",
        "predicted_hazard_frame_indices": [1, 2],
        "gold_hazard_frame_indices": [1, 2],
        "false_positive_frames": [],
        "false_negative_frames": [],
        "onset_error_seconds": 0.0,
    }
    assert first["secondary_case_diagnostics"][DIRECT_DETECTOR_POPULATION[1]][1] == {
        "case_id": "case-hazard",
        "predicted_hazard_frame_indices": [],
        "gold_hazard_frame_indices": [1, 2],
        "false_positive_frames": [],
        "false_negative_frames": [1, 2],
        "onset_error_seconds": None,
    }
    assert first["secondary_interval"]["TooFlashy"]["eligible_case_count"] == 0
    assert first["secondary_interval"]["TooFlashy"]["bootstrap"] is None
    assert first["secondary_case_diagnostics"]["TooFlashy"] == []
    assert first["runtime"]["FlashPatch"] == {
        "case_count": 2,
        "repeat_count": 6,
        "total_wall_time_ns": 6_012,
        "mean_wall_time_ns": 1002.0,
        "min_wall_time_ns": 1001,
        "max_wall_time_ns": 1003,
    }
    first_case = first["case_receipts"][0]
    assert first_case["repository_url"] == "https://github.com/example/game-0"
    assert first_case["repository_revision"] == "0000000000000000000000000000000000000000"
    assert first_case["detector_population"] == list(DIRECT_DETECTOR_POPULATION)
    assert first_case["detector_population_sha256"] == DIRECT_DETECTOR_POPULATION_SHA256
    assert first_case["renderer_rgb_sha256"] == hashlib.sha256(b"case-safergb").hexdigest()
    assert first_case["timestamps_sha256"] == hashlib.sha256(b"case-safetimestamps").hexdigest()
    assert first_case["canonical_video_sha256"] == hashlib.sha256(b"case-safevideo").hexdigest()
    assert first_case["fair_runtime_input_sha256"] == hashlib.sha256(b"case-safevideo").hexdigest()
    assert first_case["fair_runtime_external_slot_child_join_count"] == 9
    assert len(first_case["fair_runtime_external_slot_child_joins_sha256"]) == 64
    primary_bootstrap = first["primary_case_level"]["FlashPatch"]["bootstrap"]
    secondary_bootstrap = first["secondary_interval"]["FlashPatch"]["bootstrap"]
    assert primary_bootstrap["iterations"] == BOOTSTRAP_ITERATIONS
    assert secondary_bootstrap["iterations"] == BOOTSTRAP_ITERATIONS
    assert primary_bootstrap["seed"] != secondary_bootstrap["seed"]
    assert first["bootstrap_lanes_separate"] is True
    assert not {"rank", "ranking", "winner"}.intersection(first)


def test_nine_receipt_bound_natural_gold_cases_make_l7_scoreable_without_claim_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        _case(
            f"case-{index}",
            "SAFE" if index % 2 == 0 else "HAZARDOUS",
            {
                "FlashPatch": "SAFE" if index % 2 == 0 else "HAZARDOUS",
                DIRECT_DETECTOR_POPULATION[1]: "SAFE",
                "TooFlashy": "HAZARDOUS",
            },
            repository_index=index % 3,
        )
        for index in range(9)
    ]
    _mock_reopeners(monkeypatch, cases)

    result = verify_score_bundle(_bundle(tmp_path / "bundle.json", len(cases)))

    assert result["status"] == "RECEIPT_BOUND_STATISTICS_VERIFIED"
    assert result["case_count"] == 9
    assert result["minimum_scoreable_natural_cases"] == 9
    assert result["minimum_scoreable_public_repositories"] == 3
    assert result["public_repository_count"] == 3
    assert result["claim_status"] == "SCOREABLE"
    assert result["scoreable"] is True
    assert result["scoreability_blockers"] == []
    assert result["external_claim_authorized"] is False
    assert not {"rank", "ranking", "winner"}.intersection(result)


@pytest.mark.parametrize(
    "duplicate_field",
    [
        "natural_ledger",
        "renderer_input",
        "gold_receipt",
        "tool_parity_receipt",
        "fair_runtime_receipt",
    ],
)
def test_duplicate_raw_artifacts_cannot_expand_score_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_field: str
) -> None:
    cases = [
        _case(
            f"case-{index}",
            "SAFE",
            {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
            repository_index=index % 3,
        )
        for index in range(9)
    ]
    if duplicate_field == "natural_ledger":
        cases[1]["natural"]["assessment"]["ledger_sha256"] = cases[0]["natural"]["assessment"]["ledger_sha256"]
    elif duplicate_field == "renderer_input":
        cases[1]["natural"]["assessment"]["renderer_rgb_sha256"] = cases[0]["natural"]["assessment"]["renderer_rgb_sha256"]
        cases[1]["natural"]["assessment"]["timestamps_sha256"] = cases[0]["natural"]["assessment"]["timestamps_sha256"]
    elif duplicate_field == "gold_receipt":
        cases[1]["gold"]["receipt"]["sha256"] = cases[0]["gold"]["receipt"]["sha256"]
    elif duplicate_field == "tool_parity_receipt":
        cases[1]["parity"]["receipt"]["sha256"] = cases[0]["parity"]["receipt"]["sha256"]
    else:
        cases[1]["fair"]["receipt"]["sha256"] = cases[0]["fair"]["receipt"]["sha256"]
    _mock_reopeners(monkeypatch, cases)

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", len(cases)))


def test_nine_cases_from_one_public_repository_remain_not_scoreable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        _case(
            f"case-{index}",
            "SAFE",
            {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
            repository_index=0,
        )
        for index in range(9)
    ]
    _mock_reopeners(monkeypatch, cases)

    result = verify_score_bundle(_bundle(tmp_path / "bundle.json", len(cases)))

    assert result["case_count"] == 9
    assert result["public_repository_count"] == 1
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["scoreability_blockers"] == [
        "minimum_three_public_repositories_not_met"
    ]


def test_three_revisions_of_one_public_repository_remain_not_scoreable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        _case(
            f"case-{index}",
            "SAFE",
            {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
            repository_url="https://github.com/example/same-game",
            repository_revision=f"{index:040d}"[-40:],
        )
        for index in range(9)
    ]
    _mock_reopeners(monkeypatch, cases)

    result = verify_score_bundle(_bundle(tmp_path / "bundle.json", len(cases)))

    assert result["case_count"] == 9
    assert result["public_repository_count"] == 1
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["scoreability_blockers"] == [
        "minimum_three_public_repositories_not_met"
    ]


def test_url_variants_of_one_public_repository_remain_not_scoreable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variants = [
        "https://github.com/Example/Same-Game",
        "https://github.com/example/same-game/",
        "https://github.com/example/same-game.git",
    ]
    cases = [
        _case(
            f"case-{index}",
            "SAFE",
            {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
            repository_url=variants[index % len(variants)],
            repository_revision=f"{index:040d}"[-40:],
        )
        for index in range(9)
    ]
    _mock_reopeners(monkeypatch, cases)

    result = verify_score_bundle(_bundle(tmp_path / "bundle.json", len(cases)))

    assert result["case_count"] == 9
    assert result["public_repository_count"] == 1
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["scoreability_blockers"] == [
        "minimum_three_public_repositories_not_met"
    ]


@pytest.mark.parametrize(
    ("repository_url", "repository_revision"),
    [
        ("", "0" * 40),
        (123, "0" * 40),
        ("https://github.com/example/game", ""),
        ("https://github.com/example/game", 123),
        ("git@github.com:example/game.git", "0" * 40),
        ("file:///tmp/game", "0" * 40),
        ("https://github.com/example/game?mirror=1", "0" * 40),
    ],
)
def test_invalid_repository_identity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_url: object,
    repository_revision: object,
) -> None:
    case = _case(
        "case-safe",
        "SAFE",
        {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
        repository_url=repository_url,
        repository_revision=repository_revision,
    )
    _mock_reopeners(monkeypatch, [case])

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


@pytest.mark.parametrize("field", ["score", "scores", "score_summary", "ranking", "winner"])
def test_caller_supplied_score_or_rank_surface_is_rejected(tmp_path: Path, field: str) -> None:
    bundle = _bundle(tmp_path / "bundle.json", 1, extra={field: {"FlashPatch": 1.0}})
    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(bundle)


def test_release_oracle_cannot_enter_detector_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        "case-safe",
        "SAFE",
        {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION},
    )
    case["parity"]["observations"] = dict(case["parity"]["observations"])
    case["fair"]["observations"] = dict(case["fair"]["observations"])
    case["fair"]["runtime_wall_time_ns"] = dict(case["fair"]["runtime_wall_time_ns"])
    case["parity"]["observations"].pop(DIRECT_DETECTOR_POPULATION[1])
    case["fair"]["observations"].pop(DIRECT_DETECTOR_POPULATION[1])
    case["fair"]["runtime_wall_time_ns"].pop(DIRECT_DETECTOR_POPULATION[1])
    case["parity"]["observations"][EA_IRIS_RELEASE_ORACLE_ID] = {
        "prediction": "SAFE",
        "hazard_frame_indices": [],
    }
    case["fair"]["observations"][EA_IRIS_RELEASE_ORACLE_ID] = {
        "prediction": "SAFE",
        "hazard_frame_indices": [],
    }
    case["fair"]["runtime_wall_time_ns"][EA_IRIS_RELEASE_ORACLE_ID] = [
        1_001,
        1_002,
        1_003,
    ]
    _mock_reopeners(monkeypatch, [case])

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


def test_missing_or_invalid_gold_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case("case-safe", "SAFE", {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION})
    _mock_reopeners(monkeypatch, [case])
    monkeypatch.setattr(
        l7_score,
        "_gold_projection",
        lambda ref: (_ for _ in ()).throw(L7ScoreError(GOLD_FAILURE)),
    )
    with pytest.raises(L7ScoreError, match=GOLD_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


@pytest.mark.parametrize(
    "gold_patch",
    [
        {"timestamps_seconds": [0.0, "0.02", 0.03]},
        {"timestamps_seconds": [0.0, float("nan"), 0.03]},
        {"timestamps_seconds": [0.0, float("inf"), 0.03]},
        {"timestamps_seconds": [0.0, 0.02, 0.02, 0.05]},
        {"timestamps_seconds": [0.0, 0.03, 0.02, 0.05]},
        {"intervals": [{"kind": "flash", "end_seconds": 0.04}]},
        {"intervals": [{"kind": "flash", "start_seconds": 0.04, "end_seconds": 0.04}]},
        {"intervals": [{"kind": "flash", "start_seconds": False, "end_seconds": 0.04}]},
        {"intervals": [{"kind": "flash", "start_seconds": 0.01, "end_seconds": float("inf")}]},
    ],
)
def test_malformed_gold_timestamps_or_intervals_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gold_patch: dict[str, object],
) -> None:
    case = _case("case-safe", "HAZARDOUS", {name: "HAZARDOUS" for name in DIRECT_DETECTOR_POPULATION})
    case["gold"].update(gold_patch)
    _mock_reopeners(monkeypatch, [case])

    with pytest.raises(L7ScoreError, match=GOLD_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


def test_nonstandard_json_numeric_constants_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        '{"schema":"flashpatch-l7-receipt-bound-score-bundle-v1","cases":NaN}\n',
        encoding="utf-8",
    )

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(bundle)


def test_score_bundle_reopen_rejects_direct_symlink_path(tmp_path: Path) -> None:
    real_bundle = _write(
        tmp_path / "real-bundle.json",
        {"schema": SCORE_BUNDLE_SCHEMA, "cases": []},
    )
    linked_bundle = tmp_path / "linked-bundle.json"
    linked_bundle.symlink_to(real_bundle)

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(linked_bundle)


def test_score_bundle_reopen_rejects_parent_symlink_path(tmp_path: Path) -> None:
    real_dir = tmp_path / "real-bundle-dir"
    real_dir.mkdir()
    _write(real_dir / "bundle.json", {"schema": SCORE_BUNDLE_SCHEMA, "cases": []})
    linked_dir = tmp_path / "linked-bundle-dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(linked_dir / "bundle.json")


def test_artifact_reopen_rejects_parent_symlink_path(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    artifact = _write(real_dir / "artifact.json", {"schema": "receipt"})
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._artifact(
            {
                "path": str(linked_dir / "artifact.json"),
                "sha256": l7_score._sha256_file(artifact),
            }
        )


@pytest.mark.parametrize("mixture", ["controlled", "mitigation", "unbound"])
def test_controlled_mitigation_and_unbound_receipt_mixtures_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mixture: str
) -> None:
    case = _case("case-safe", "SAFE", {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION})
    if mixture == "controlled":
        case["natural"]["ledger"]["controlled_mutation"] = True
    elif mixture == "mitigation":
        case["parity"]["observations"]["FFmpeg vf_photosensitivity"] = {
            "prediction": "SAFE", "hazard_frame_indices": []
        }
    else:
        case["fair"]["observations"]["FlashPatch"]["prediction"] = "HAZARDOUS"
    _mock_reopeners(monkeypatch, [case])

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


@pytest.mark.parametrize("bad_case_id", ["", 123, None])
def test_gold_case_id_must_remain_nonempty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_case_id: object
) -> None:
    case = _case("case-safe", "SAFE", {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION})
    case["gold"]["case_id"] = bad_case_id
    _mock_reopeners(monkeypatch, [case])

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        verify_score_bundle(_bundle(tmp_path / "bundle.json", 1))


@pytest.mark.parametrize(
    ("tamper_field", "tamper_value"),
    [
        ("schema", None),
        ("schema", "flashpatch-l7-independent-gold-projection-v0"),
        ("receipt", None),
        ("receipt_sha256", "0" * 64),
        ("receipt_path", "/tmp/other-gold.json"),
        ("trust_policy_sha256", "1" * 64),
        ("trust_policy_path", "/tmp/other-policy.json"),
        ("trust_policy_policy_id", ""),
        ("trust_policy_registry_snapshot_sha256", "not-a-sha"),
    ],
)
def test_gold_projection_must_rebind_to_reopened_receipt_and_trust_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_field: str,
    tamper_value: object,
) -> None:
    receipt = _write(tmp_path / "gold.json", {"schema": "gold"})
    policy = _write(tmp_path / "policy.json", {"schema": "policy"})
    projection: dict[str, object] = {
        "schema": "flashpatch-l7-independent-gold-projection-v1",
        "gold_verified": True,
        "case_class": "natural_external",
        "controlled_mutation": False,
        "league_score_authorized": False,
        "receipt": {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)},
        "trust_policy": {
            "path": str(policy),
            "sha256": l7_score._sha256_file(policy),
            "policy_id": "gold-policy-v1",
            "registry_snapshot_sha256": "2" * 64,
        },
    }
    if tamper_field == "schema":
        projection["schema"] = tamper_value
    elif tamper_field == "receipt":
        projection["receipt"] = tamper_value
    elif tamper_field == "receipt_sha256":
        projection["receipt"]["sha256"] = tamper_value
    elif tamper_field == "receipt_path":
        projection["receipt"]["path"] = tamper_value
    elif tamper_field == "trust_policy_sha256":
        projection["trust_policy"]["sha256"] = tamper_value
    elif tamper_field == "trust_policy_path":
        projection["trust_policy"]["path"] = tamper_value
    elif tamper_field == "trust_policy_policy_id":
        projection["trust_policy"]["policy_id"] = tamper_value
    else:
        projection["trust_policy"]["registry_snapshot_sha256"] = tamper_value
    monkeypatch.setattr(
        l7_score,
        "project_independent_gold",
        lambda *args, **kwargs: copy.deepcopy(projection),
    )

    with pytest.raises(L7ScoreError, match=GOLD_FAILURE):
        l7_score._gold_projection(
            {
                "receipt": {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)},
                "trust_policy": {"path": str(policy), "sha256": l7_score._sha256_file(policy)},
            }
        )


@pytest.mark.parametrize(
    ("freeze_field", "tampered_value"),
    [
        ("source_revision", "f" * 40),
        ("renderer_rgb_raw_sha256", "e" * 64),
    ],
)
def test_natural_case_source_revision_or_rgb_hash_drift_fails_closed(
    freeze_field: str,
    tampered_value: str,
) -> None:
    case = _case("case-safe", "SAFE", {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION})
    natural = {
        "assessment": {
            **case["natural"]["assessment"],
            "case_id": case["gold"]["case_id"],
            "renderer_execution_receipt_sha256": "a" * 64,
        },
        "repository": {
            **case["natural"]["repository"],
            "license": "MIT",
            "project_subpath": ".",
        },
        "source": {"source_tree_sha256": "b" * 64},
        "renderer": {"frame_count": 4},
        "trace": {"sha256": "c" * 64},
    }
    gold = {
        **case["gold"],
        "case_freeze": {
            "public_repository_url": case["natural"]["repository"]["url"],
            "source_revision": case["natural"]["repository"]["revision"],
            "license": "MIT",
            "project_subpath": ".",
            "source_tree_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "renderer_execution_receipt_sha256": "a" * 64,
            "renderer_rgb_raw_sha256": case["natural"]["assessment"]["renderer_rgb_sha256"],
            "timestamps_sha256": case["natural"]["assessment"]["timestamps_sha256"],
            "frame_count": 4,
        },
    }
    gold["case_freeze"][freeze_field] = tampered_value

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._match_natural_and_gold(natural, gold)


def test_natural_case_source_revision_and_rgb_hash_match_allows_binding() -> None:
    case = _case("case-safe", "SAFE", {name: "SAFE" for name in DIRECT_DETECTOR_POPULATION})
    natural = {
        "assessment": {
            **case["natural"]["assessment"],
            "case_id": case["gold"]["case_id"],
            "renderer_execution_receipt_sha256": "a" * 64,
        },
        "repository": {
            **case["natural"]["repository"],
            "license": "MIT",
            "project_subpath": ".",
        },
        "source": {"source_tree_sha256": "b" * 64},
        "renderer": {"frame_count": 4},
        "trace": {"sha256": "c" * 64},
    }
    gold = {
        **case["gold"],
        "case_freeze": {
            "public_repository_url": case["natural"]["repository"]["url"],
            "source_revision": case["natural"]["repository"]["revision"],
            "license": "MIT",
            "project_subpath": ".",
            "source_tree_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "renderer_execution_receipt_sha256": "a" * 64,
            "renderer_rgb_raw_sha256": case["natural"]["assessment"]["renderer_rgb_sha256"],
            "timestamps_sha256": case["natural"]["assessment"]["timestamps_sha256"],
            "frame_count": 4,
        },
    }

    l7_score._match_natural_and_gold(natural, gold)


def test_tool_parity_receipt_is_recomputed_and_child_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "canonical.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"ffv1")
    _write(conversion, {"schema": "conversion"})
    rows = []
    for index, comparator in enumerate(DIRECT_DETECTOR_POPULATION):
        child = _write(tmp_path / f"parity-child-{index}.json", {"comparator": comparator})
        secondary = (
            {"status": "NOT_VERIFIED", "reason": "native_tool_does_not_expose_interval_endpoint"}
            if comparator == "TooFlashy"
            else {"status": "VERIFIED", "hazard_frame_indices": [1]}
        )
        rows.append(
            {
                "comparator": comparator,
                "run_receipt": {"path": str(child), "sha256": l7_score._sha256_file(child)},
                "status": "VERIFIED",
                "decoder_timeline": {"parity_status": "VERIFIED", "native_artifacts": []},
                "primary_case_level_endpoint": {"status": "VERIFIED", "prediction": "HAZARDOUS"},
                "secondary_interval_endpoint": secondary,
            }
        )
    stored = {
        "schema": l7_score.DECODER_TIMELINE_PARITY_SCHEMA,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "canonical_contract": {
            "canonical_video": {"path": str(video), "sha256": l7_score._sha256_file(video)},
            "conversion_receipt": {"path": str(conversion), "sha256": l7_score._sha256_file(conversion)},
        },
        "comparators": rows,
        "failures": [],
    }
    receipt = _write(tmp_path / "parity.json", stored)
    monkeypatch.setattr(l7_score, "verify_decoder_timeline_parity", lambda *args, **kwargs: copy.deepcopy(stored))

    reopened = l7_score._reopen_tool_parity(
        {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
    )

    assert reopened["observations"]["FlashPatch"] == {
        "prediction": "HAZARDOUS", "hazard_frame_indices": [1]
    }
    assert reopened["observations"]["TooFlashy"]["hazard_frame_indices"] is None
    Path(rows[0]["run_receipt"]["path"]).write_text("tampered\n")
    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_tool_parity(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_native_main_summary_can_bind_to_independent_gold_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "MASTER-MAP.md").write_text("map\n", encoding="utf-8")
    (tmp_path / "src" / "flashpatch").mkdir(parents=True)
    frame_path = tmp_path / "artifacts" / "case-a" / "output" / "renderer-frames.npz"
    frame_path.parent.mkdir(parents=True)
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    timestamps = np.arange(4, dtype=np.float64) / 60.0
    np.savez_compressed(frame_path, frames=frames, timestamps=timestamps)
    receipt = _write(
        tmp_path / "artifacts" / "case-a" / "output" / "native-main-capture-receipt.json",
        {"schema": "flashpatch-godot-native-main-capture-v1"},
    )
    summary = {
        "schema": "flashpatch-l7-native-main-qualification-summary-v1",
        "results": [
            {
                "candidate_id": "case-a",
                "receipt_created": True,
                "repository_url": "https://github.com/example/native-main-game",
                "revision": "1" * 40,
                "license": "MIT",
                "frame_artifact_path": frame_path.relative_to(tmp_path).as_posix(),
                "frame_artifact_sha256": l7_score._sha256_file(frame_path),
                "raw_receipt_path": receipt.relative_to(tmp_path).as_posix(),
                "raw_receipt_sha256": l7_score._sha256_file(receipt),
                "frame_count": 4,
            }
        ],
    }
    summary_path = tmp_path / "evidence" / "l7" / "native-main" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path = _write(summary_path, summary)
    monkeypatch.setattr(
        l7_score,
        "verify_native_main_qualification_summary",
        lambda path: {
            "status": "NOT_SCOREABLE",
            "scoreable": False,
            "qualified_case_count": 0,
            "external_claim_authorized": False,
        },
    )

    natural = l7_score._natural_projection(
        {
            "native_main_summary": {
                "path": str(summary_path),
                "sha256": l7_score._sha256_file(summary_path),
            },
            "blind_case_id": "blind-native-case",
            "frame_artifact_sha256": l7_score._sha256_file(frame_path),
        }
    )
    gold = {
        "case_id": "blind-native-case",
        "case_freeze": {
            "source_summary_sha256": l7_score._sha256_file(summary_path),
            "frame_artifact_sha256": l7_score._sha256_file(frame_path),
            "renderer_rgb_raw_sha256": natural["assessment"]["renderer_rgb_sha256"],
            "timestamps_f64_sha256": natural["assessment"]["timestamps_sha256"],
            "frame_count": 4,
        },
    }

    l7_score._match_natural_and_gold(natural, gold)


def test_native_main_gold_return_projects_only_after_return_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_root = tmp_path / "intake"
    packet = intake_root / "adjudicator-packet"
    blinded = packet / "blinded-inputs" / "blind-native-case.json"
    timestamps = packet / "timestamps" / "blind-native-case.json"
    timestamps.parent.mkdir(parents=True)
    blinded.parent.mkdir(parents=True)
    _write(timestamps, {"timestamps_seconds": [0.0, 1 / 60, 2 / 60, 3 / 60]})
    _write(
        blinded,
        {
            "blind_case_id": "blind-native-case",
            "timestamps_path": "timestamps/blind-native-case.json",
        },
    )
    intake = _write(
        intake_root / "gold-intake-receipt.json",
        {
            "schema": "flashpatch-l7-native-main-blinded-gold-intake-v1",
            "blinded_inputs": [
                {
                    "blind_case_id": "blind-native-case",
                    "path": "adjudicator-packet/blinded-inputs/blind-native-case.json",
                    "sha256": l7_score._sha256_file(blinded),
                }
            ],
        },
    )
    packet_manifest = _write(tmp_path / "packet-manifest.json", {"schema": "packet"})
    return_root = tmp_path / "return"
    return_root.mkdir()
    policy = _write(
        return_root / "external-trust-policy.json",
        {"schema": "policy", "policy_id": "approved-native-policy"},
    )
    (return_root / "blind-native-case").mkdir()
    receipt = _write(
        return_root / "blind-native-case" / "independent-gold.json",
        {
            "schema": "flashpatch-l7-independent-gold-v2",
            "case_id": "blind-native-case",
            "case_class": "natural_external",
            "claim_tier": "L9_ELIGIBLE",
            "controlled_mutation": False,
            "case_freeze": {
                "source_summary_sha256": "1" * 64,
                "frame_artifact_sha256": "2" * 64,
                "renderer_rgb_raw_sha256": "3" * 64,
                "timestamps_f64_sha256": "4" * 64,
                "frame_count": 4,
            },
            "adjudication": {"result": "SAFE", "intervals": []},
        },
    )
    manifest = _write(
        return_root / "native-main-independent-gold-return.json",
        {
            "schema": "flashpatch-l7-native-main-independent-gold-return-v1",
            "gold_receipts": [
                {
                    "blind_case_id": "blind-native-case",
                    "independent_gold_path": "blind-native-case/independent-gold.json",
                    "independent_gold_sha256": l7_score._sha256_file(receipt),
                    "trust_policy_path": "external-trust-policy.json",
                    "trust_policy_sha256": l7_score._sha256_file(policy),
                }
            ],
        },
    )
    calls: list[tuple[Path, Path, Path]] = []

    def fake_verify(*, intake_receipt: Path, return_root: Path, packet_manifest: Path) -> dict[str, object]:
        calls.append((intake_receipt, return_root, packet_manifest))
        return {
            "status": "INDEPENDENT_GOLD_VERIFIED",
            "scoreable": False,
            "external_claim_authorized": False,
        }

    monkeypatch.setattr(l7_score, "verify_native_main_independent_gold_return", fake_verify)

    projected = l7_score._gold_projection(
        {
            "receipt": {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)},
            "trust_policy": {"path": str(policy), "sha256": l7_score._sha256_file(policy)},
            "native_main_gold_return": {
                "intake_receipt": {"path": str(intake), "sha256": l7_score._sha256_file(intake)},
                "packet_manifest": {
                    "path": str(packet_manifest),
                    "sha256": l7_score._sha256_file(packet_manifest),
                },
                "return_manifest": {"path": str(manifest), "sha256": l7_score._sha256_file(manifest)},
            },
        }
    )

    assert calls == [(intake.resolve(), return_root.resolve(), packet_manifest.resolve())]
    assert projected["schema"] == "flashpatch-l7-independent-gold-projection-v1"
    assert projected["case_id"] == "blind-native-case"
    assert projected["decision"] == "SAFE"


def test_tool_parity_reopen_rejects_noncanonical_interval_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "canonical.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"ffv1")
    _write(conversion, {"schema": "conversion"})
    child = _write(tmp_path / "parity-child.json", {"comparator": "FlashPatch"})
    rows = [
        {
            "comparator": comparator,
            "run_receipt": {"path": str(child), "sha256": l7_score._sha256_file(child)},
            "status": "VERIFIED",
            "decoder_timeline": {"parity_status": "VERIFIED", "native_artifacts": []},
            "primary_case_level_endpoint": {"status": "VERIFIED", "prediction": "HAZARDOUS"},
            "secondary_interval_endpoint": {
                "status": "VERIFIED",
                "hazard_frame_indices": [False],
            },
        }
        for comparator in DIRECT_DETECTOR_POPULATION
    ]
    stored = {
        "schema": l7_score.DECODER_TIMELINE_PARITY_SCHEMA,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "canonical_contract": {
            "canonical_video": {"path": str(video), "sha256": l7_score._sha256_file(video)},
            "conversion_receipt": {"path": str(conversion), "sha256": l7_score._sha256_file(conversion)},
        },
        "comparators": rows,
        "failures": [],
    }
    receipt = _write(tmp_path / "parity.json", stored)
    monkeypatch.setattr(
        l7_score,
        "verify_decoder_timeline_parity",
        lambda *args, **kwargs: copy.deepcopy(stored),
    )

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_tool_parity(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_tool_parity_reopen_rejects_canonical_video_parent_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_dir = tmp_path / "real-video"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked-video"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    video = real_dir / "canonical.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"ffv1")
    _write(conversion, {"schema": "conversion"})
    child = _write(tmp_path / "parity-child.json", {"comparator": "FlashPatch"})
    stored = {
        "schema": l7_score.DECODER_TIMELINE_PARITY_SCHEMA,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "canonical_contract": {
            "canonical_video": {
                "path": str(linked_dir / "canonical.mkv"),
                "sha256": l7_score._sha256_file(video),
            },
            "conversion_receipt": {
                "path": str(conversion),
                "sha256": l7_score._sha256_file(conversion),
            },
        },
        "comparators": [
            {
                "comparator": comparator,
                "run_receipt": {"path": str(child), "sha256": l7_score._sha256_file(child)},
                "status": "VERIFIED",
                "decoder_timeline": {"parity_status": "VERIFIED", "native_artifacts": []},
                "primary_case_level_endpoint": {"status": "VERIFIED", "prediction": "SAFE"},
                "secondary_interval_endpoint": {"status": "VERIFIED", "hazard_frame_indices": []},
            }
            for comparator in DIRECT_DETECTOR_POPULATION
        ],
        "failures": [],
    }
    receipt = _write(tmp_path / "parity.json", stored)
    monkeypatch.setattr(
        l7_score,
        "verify_decoder_timeline_parity",
        lambda *args, **kwargs: copy.deepcopy(stored),
    )

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_tool_parity(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_full_population_fair_runtime_receipt_is_recomputed_from_three_raw_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_rows = []
    join_rows = []
    slot = 1
    for comparator_index, comparator in enumerate(DIRECT_DETECTOR_POPULATION):
        runs = []
        for ordinal in range(1, 4):
            child = _write(
                tmp_path / f"fair-child-{comparator_index}-{ordinal}.json",
                {
                    "observation": {
                        "prediction": "HAZARDOUS",
                        "hazard_frame_indices": None if comparator == "TooFlashy" else [1, 2],
                    },
                    "wall_time_ns": (comparator_index + 1) * 1_000 + ordinal,
                },
            )
            child_sha256 = l7_score._sha256_file(child)
            runs.append({
                "repeat": ordinal,
                "receipt": str(child),
                "receipt_sha256": child_sha256,
            })
            join_rows.append({
                "slot": slot,
                "comparator": comparator,
                "repeat_ordinal": ordinal,
                "external_result_sha256": f"{slot:064x}",
                "child_receipt_sha256": child_sha256,
            })
            slot += 1
        aggregate = _write(
            tmp_path / f"repeat-{comparator_index}.json",
            {"comparator": comparator, "runs": runs},
        )
        receipt_rows.append(
            {"receipt": str(aggregate), "sha256": l7_score._sha256_file(aggregate)}
        )
    schedule = _write(tmp_path / "schedule.json", {"schema": "schedule", "input_sha256": "d" * 64})
    request = _write(tmp_path / "request.json", {"schema": "request"})
    witness_receipt = _write(tmp_path / "witness.json", {"schema": "witness"})
    stored = {
        "schema": l7_score.FAIR_RUNTIME_BUNDLE_SCHEMA,
        "receipts": receipt_rows,
        "schedule": {
            "path": str(schedule),
            "artifact_sha256": l7_score._sha256_file(schedule),
        },
        "external_host_witness": {"request": str(request), "receipt": str(witness_receipt)},
        "receipts_verified": True,
        "fair_runtime_verified": True,
        "independent_execution_witness_verified": True,
        "failures": [],
        "external_slot_child_joins": join_rows,
    }
    receipt = _write(tmp_path / "fair.json", stored)
    monkeypatch.setattr(l7_score, "verify_fair_runtime_receipts", lambda *args, **kwargs: copy.deepcopy(stored))

    reopened = l7_score._reopen_fair_runtime(
        {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
    )

    assert set(reopened["observations"]) == set(DIRECT_DETECTOR_POPULATION)
    assert reopened["input_sha256"] == "d" * 64
    assert reopened["observations"]["FlashPatch"]["hazard_frame_indices"] == [1, 2]
    assert reopened["observations"]["TooFlashy"]["hazard_frame_indices"] is None
    assert reopened["runtime_wall_time_ns"]["FlashPatch"] == [1001, 1002, 1003]


@pytest.mark.parametrize(
    "child_patch",
    [
        {},
        {
            "wall_time_ns": 10,
            "fair_runtime": {
                "started_monotonic_ns": 1,
                "finished_monotonic_ns": 12,
                "wall_time_ns": 11,
            },
        },
        {
            "fair_runtime": {
                "started_monotonic_ns": 1,
                "finished_monotonic_ns": 12,
                "wall_time_ns": 10,
            },
        },
    ],
)
def test_fair_runtime_reopen_rejects_missing_or_inconsistent_child_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_patch: dict[str, object],
) -> None:
    receipt_rows = []
    join_rows = []
    slot = 1
    for comparator_index, comparator in enumerate(DIRECT_DETECTOR_POPULATION):
        runs = []
        for ordinal in range(1, 4):
            payload: dict[str, object] = {
                "observation": {"prediction": "SAFE", "hazard_frame_indices": []},
                "wall_time_ns": (comparator_index + 1) * 1_000 + ordinal,
            }
            if comparator_index == 0 and ordinal == 1:
                payload = {
                    "observation": {"prediction": "SAFE", "hazard_frame_indices": []},
                    **child_patch,
                }
            child = _write(
                tmp_path / f"runtime-child-{comparator_index}-{ordinal}.json",
                payload,
            )
            child_sha256 = l7_score._sha256_file(child)
            runs.append({
                "repeat": ordinal,
                "receipt": str(child),
                "receipt_sha256": child_sha256,
            })
            join_rows.append({
                "slot": slot,
                "comparator": comparator,
                "repeat_ordinal": ordinal,
                "external_result_sha256": f"{slot:064x}",
                "child_receipt_sha256": child_sha256,
            })
            slot += 1
        aggregate = _write(
            tmp_path / f"runtime-repeat-{comparator_index}.json",
            {"comparator": comparator, "runs": runs},
        )
        receipt_rows.append({"receipt": str(aggregate), "sha256": l7_score._sha256_file(aggregate)})
    schedule = _write(tmp_path / "runtime-schedule.json", {"schema": "schedule", "input_sha256": "d" * 64})
    request = _write(tmp_path / "runtime-request.json", {"schema": "request"})
    witness_receipt = _write(tmp_path / "runtime-witness.json", {"schema": "witness"})
    stored = {
        "schema": l7_score.FAIR_RUNTIME_BUNDLE_SCHEMA,
        "receipts": receipt_rows,
        "schedule": {
            "path": str(schedule),
            "artifact_sha256": l7_score._sha256_file(schedule),
        },
        "external_host_witness": {"request": str(request), "receipt": str(witness_receipt)},
        "receipts_verified": True,
        "fair_runtime_verified": True,
        "independent_execution_witness_verified": True,
        "failures": [],
        "external_slot_child_joins": join_rows,
    }
    receipt = _write(tmp_path / "runtime-fair.json", stored)
    monkeypatch.setattr(l7_score, "verify_fair_runtime_receipts", lambda *args, **kwargs: copy.deepcopy(stored))

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_fair_runtime(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_fair_runtime_reopen_rejects_schedule_parent_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_rows = []
    join_rows = []
    slot = 1
    for comparator_index, comparator in enumerate(DIRECT_DETECTOR_POPULATION):
        runs = []
        for ordinal in range(1, 4):
            child = _write(
                tmp_path / f"symlink-child-{comparator_index}-{ordinal}.json",
                {
                    "observation": {"prediction": "SAFE", "hazard_frame_indices": []},
                    "wall_time_ns": (comparator_index + 1) * 1_000 + ordinal,
                },
            )
            child_sha256 = l7_score._sha256_file(child)
            runs.append({
                "repeat": ordinal,
                "receipt": str(child),
                "receipt_sha256": child_sha256,
            })
            join_rows.append({
                "slot": slot,
                "comparator": comparator,
                "repeat_ordinal": ordinal,
                "external_result_sha256": f"{slot:064x}",
                "child_receipt_sha256": child_sha256,
            })
            slot += 1
        aggregate = _write(
            tmp_path / f"symlink-repeat-{comparator_index}.json",
            {"comparator": comparator, "runs": runs},
        )
        receipt_rows.append({"receipt": str(aggregate), "sha256": l7_score._sha256_file(aggregate)})
    real_schedule_dir = tmp_path / "real-schedule"
    real_schedule_dir.mkdir()
    linked_schedule_dir = tmp_path / "linked-schedule"
    linked_schedule_dir.symlink_to(real_schedule_dir, target_is_directory=True)
    schedule = _write(real_schedule_dir / "schedule.json", {"schema": "schedule", "input_sha256": "d" * 64})
    request = _write(tmp_path / "request.json", {"schema": "request"})
    witness_receipt = _write(tmp_path / "witness.json", {"schema": "witness"})
    stored = {
        "schema": l7_score.FAIR_RUNTIME_BUNDLE_SCHEMA,
        "receipts": receipt_rows,
        "schedule": {
            "path": str(linked_schedule_dir / "schedule.json"),
            "artifact_sha256": l7_score._sha256_file(schedule),
        },
        "external_host_witness": {"request": str(request), "receipt": str(witness_receipt)},
        "receipts_verified": True,
        "fair_runtime_verified": True,
        "independent_execution_witness_verified": True,
        "failures": [],
        "external_slot_child_joins": join_rows,
    }
    receipt = _write(tmp_path / "fair.json", stored)
    monkeypatch.setattr(l7_score, "verify_fair_runtime_receipts", lambda *args, **kwargs: copy.deepcopy(stored))

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_fair_runtime(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_fair_runtime_reopen_rejects_unbound_child_receipt_inside_repeat_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _write(
        tmp_path / "fair-child.json",
        {
            "observation": {"prediction": "SAFE", "hazard_frame_indices": []},
            "wall_time_ns": 100,
        },
    )
    repeat = _write(
        tmp_path / "repeat.json",
        {
            "comparator": "FlashPatch",
            "runs": [
                {"repeat": 1, "receipt": str(child)},
                {"repeat": 2, "receipt": str(child), "receipt_sha256": "0" * 64},
                {"repeat": 3, "receipt": str(child), "receipt_sha256": l7_score._sha256_file(child)},
            ],
        },
    )
    schedule = _write(tmp_path / "schedule.json", {"schema": "schedule", "input_sha256": "d" * 64})
    request = _write(tmp_path / "request.json", {"schema": "request"})
    witness_receipt = _write(tmp_path / "witness.json", {"schema": "witness"})
    stored = {
        "schema": l7_score.FAIR_RUNTIME_BUNDLE_SCHEMA,
        "receipts": [{"receipt": str(repeat), "sha256": l7_score._sha256_file(repeat)}],
        "schedule": {
            "path": str(schedule),
            "artifact_sha256": l7_score._sha256_file(schedule),
        },
        "external_host_witness": {"request": str(request), "receipt": str(witness_receipt)},
        "receipts_verified": True,
        "fair_runtime_verified": True,
        "independent_execution_witness_verified": True,
        "failures": [],
        "external_slot_child_joins": [
            {
                "slot": slot,
                "comparator": DIRECT_DETECTOR_POPULATION[(slot - 1) // 3],
                "repeat_ordinal": ((slot - 1) % 3) + 1,
                "external_result_sha256": f"{slot:064x}",
                "child_receipt_sha256": f"{slot + 10:064x}",
            }
            for slot in range(1, 10)
        ],
    }
    receipt = _write(tmp_path / "fair.json", stored)
    monkeypatch.setattr(l7_score, "verify_fair_runtime_receipts", lambda *args, **kwargs: copy.deepcopy(stored))

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_fair_runtime(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )


def test_fair_runtime_reopen_rejects_external_join_child_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_rows = []
    join_rows = []
    slot = 1
    for comparator_index, comparator in enumerate(DIRECT_DETECTOR_POPULATION):
        runs = []
        for ordinal in range(1, 4):
            child = _write(
                tmp_path / f"join-child-{comparator_index}-{ordinal}.json",
                {
                    "observation": {"prediction": "SAFE", "hazard_frame_indices": []},
                    "wall_time_ns": (comparator_index + 1) * 1_000 + ordinal,
                },
            )
            child_sha256 = l7_score._sha256_file(child)
            runs.append({
                "repeat": ordinal,
                "receipt": str(child),
                "receipt_sha256": child_sha256,
            })
            join_rows.append({
                "slot": slot,
                "comparator": comparator,
                "repeat_ordinal": ordinal,
                "external_result_sha256": f"{slot:064x}",
                "child_receipt_sha256": child_sha256,
            })
            slot += 1
        aggregate = _write(
            tmp_path / f"join-repeat-{comparator_index}.json",
            {"comparator": comparator, "runs": runs},
        )
        receipt_rows.append({"receipt": str(aggregate), "sha256": l7_score._sha256_file(aggregate)})
    join_rows[0] = {**join_rows[0], "child_receipt_sha256": "f" * 64}
    schedule = _write(tmp_path / "join-schedule.json", {"schema": "schedule", "input_sha256": "d" * 64})
    request = _write(tmp_path / "join-request.json", {"schema": "request"})
    witness_receipt = _write(tmp_path / "join-witness.json", {"schema": "witness"})
    stored = {
        "schema": l7_score.FAIR_RUNTIME_BUNDLE_SCHEMA,
        "receipts": receipt_rows,
        "schedule": {
            "path": str(schedule),
            "artifact_sha256": l7_score._sha256_file(schedule),
        },
        "external_host_witness": {"request": str(request), "receipt": str(witness_receipt)},
        "receipts_verified": True,
        "fair_runtime_verified": True,
        "independent_execution_witness_verified": True,
        "failures": [],
        "external_slot_child_joins": join_rows,
    }
    receipt = _write(tmp_path / "join-fair.json", stored)
    monkeypatch.setattr(l7_score, "verify_fair_runtime_receipts", lambda *args, **kwargs: copy.deepcopy(stored))

    with pytest.raises(L7ScoreError, match=CAPABILITY_FAILURE):
        l7_score._reopen_fair_runtime(
            {"path": str(receipt), "sha256": l7_score._sha256_file(receipt)}
        )
