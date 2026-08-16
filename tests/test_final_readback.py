from __future__ import annotations

import json
from pathlib import Path

from flashpatch.readback import FinalReadback


REPOSITORY = Path(__file__).parents[1]


def test_final_readback_verifies_campaign_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "final-readback.json"
    result = FinalReadback(REPOSITORY).run(output)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["schema"] == "flashpatch-final-readback-v1"
    assert result["campaign_id"] == "flashpatch-ai-game-neurosafety-2026"
    assert result["scope_partition"]["remaining_ids"] == []
    assert result["selection"]["verdict"] == "NO_WINNER"
    assert result["selection"]["nonterminal"] is True
    assert result["benchmark"]["verdict"] == "PASS"
    assert result["hero_replay"]["verdict"] == "PASS"
    assert result["terminal_evidence"]["verified_records"] == 18
    assert result["verdict"] == "PASS"
