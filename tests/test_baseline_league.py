from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashpatch.baseline_league import BaselineLeague


RESULTS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "results.json"


def test_baseline_league_verifies_equal_budgets_and_compiler_wins(tmp_path: Path) -> None:
    output = tmp_path / "baseline-league.json"
    league = BaselineLeague.from_results(RESULTS).verify(output)

    assert json.loads(output.read_text(encoding="utf-8")) == league
    assert league["schema"] == "aigame-psebench-baseline-league-v1"
    assert league["budget_fairness"] == "PASS"
    assert league["source_attribution_winner"] == "flashpatch-neurosafety-compiler"
    assert league["source_patch_winner"] == "flashpatch-neurosafety-compiler"
    assert league["hazard_discovery_result"] == "TIE"
    assert league["video_only_runtime_filter"] is True
    assert league["verdict"] == "PASS"


def test_baseline_league_rejects_episode_budget_overrun(tmp_path: Path) -> None:
    tampered = json.loads(RESULTS.read_text(encoding="utf-8"))
    tampered["methods"]["random-exploration"]["families"][0]["episodes_used"] = 13
    input_path = tmp_path / "tampered.json"
    input_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="episode budget exceeded"):
        BaselineLeague.from_results(input_path).verify(tmp_path / "league.json")
