from __future__ import annotations

from pathlib import Path

from flashpatch.corpus import GodotCorpus
from flashpatch.explorer import RiskSeekingExplorer


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_risk_seeking_explorer_finds_hidden_hazards_within_budget(tmp_path: Path) -> None:
    explorer = RiskSeekingExplorer(GodotCorpus(CORPUS), workspace=tmp_path)

    pulse = explorer.explore("pulse-light", episode_budget=2)
    interaction = explorer.explore("interaction-burst", episode_budget=12)

    assert pulse.hazard_found
    assert pulse.max_risk == 1.0
    assert pulse.episodes_used <= 2
    assert interaction.hazard_found
    assert interaction.max_risk == 1.0
    assert interaction.episodes_used <= 12
    assert any(action.get("charge") for action in interaction.action_trace)
    assert any(action.get("fire") for action in interaction.action_trace)
