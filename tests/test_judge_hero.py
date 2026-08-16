from __future__ import annotations

import json
from pathlib import Path

from flashpatch.hero import JudgeHeroReplay


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_judge_hero_replays_hidden_interaction_and_source_fix(tmp_path: Path) -> None:
    output = tmp_path / "hero-replay.json"
    hero = JudgeHeroReplay(CORPUS, workspace=tmp_path / "work").run(output)

    assert json.loads(output.read_text(encoding="utf-8")) == hero
    assert hero["schema"] == "flashpatch-judge-hero-v1"
    assert hero["family_id"] == "interaction-burst"
    assert hero["hazard"]["found"] is True
    assert hero["hazard"]["max_risk_before"] == 1.0
    assert hero["attribution"]["node"] == "/root/InteractionBurst"
    assert hero["attribution"]["parameter"] == "burst_intensity"
    assert hero["attribution"]["causal_contribution"] == 1.0
    assert hero["patch"]["changed_source_parameters"] == 1
    assert hero["patch"]["diff_file"].endswith(".diff")
    assert hero["safe_replay"]["max_risk_after"] == 0.0
    assert hero["safe_replay"]["same_action_trace"] is True
    assert hero["safe_replay"]["timing_preserved"] is True
    assert hero["safe_replay"]["runtime_filter_used"] is False
    assert hero["verdict"] == "PASS"
