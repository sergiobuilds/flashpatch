from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FinalReadback:
    def __init__(self, repository: Path, *, state_path: Path | None = None) -> None:
        self.repository = Path(repository).resolve()
        archived_state = (
            self.repository
            / ".closure"
            / "archive"
            / "flashpatch-ai-game-neurosafety-2026.completed.json"
        )
        self.state_path = Path(state_path).resolve() if state_path else archived_state

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"expected a JSON object: {path}")
        return value

    def run(self, output: Path) -> dict[str, Any]:
        state = self._read_json(self.state_path)
        scope = set(state["scope_ids"])
        completed = set(state["completed_ids"])
        rejected = set(state["rejected_ids"])
        deferred = set(state["deferred_ids"])
        if completed & rejected or completed & deferred or rejected & deferred:
            raise RuntimeError("Closure terminal sets overlap")
        remaining = sorted(scope - completed - rejected - deferred)
        expected_remaining = [] if state["status"] == "completed" else ["final-readback"]
        if remaining != expected_remaining:
            raise RuntimeError(f"unexpected remaining Closure scope: {remaining}")

        records = state["terminal_records"]
        terminal_ids = completed | rejected | deferred
        record_ids = {record["unit_id"] for record in records}
        if record_ids != terminal_ids:
            raise RuntimeError("Closure terminal records do not match terminal sets")
        for record in records:
            artifact = self.repository / record["artifact"]
            expected = str(record["artifact_sha256"]).removeprefix("sha256:")
            if not artifact.is_file() or self._sha256(artifact) != expected:
                raise RuntimeError(f"terminal evidence readback failed: {record['unit_id']}")

        benchmark_path = self.repository / str(state["canonical_artifact"])
        benchmark = self._read_json(benchmark_path)
        if benchmark.get("schema") != "aigame-psebench-results-v1" or benchmark.get("verdict") != "PASS":
            raise RuntimeError("AIGame-PSEBench readback failed")

        hero_path = self.repository / "artifacts" / "judge-hero" / "hero-replay.json"
        hero = self._read_json(hero_path)
        if hero.get("schema") != "flashpatch-judge-hero-v1" or hero.get("verdict") != "PASS":
            raise RuntimeError("judge hero readback failed")

        selection = state["selection_decision"]
        selection_nonterminal = (
            selection.get("verdict") == "NO_WINNER"
            and selection.get("implementation_must_not_advance") is False
        )
        if not selection_nonterminal:
            raise RuntimeError("sealed selection decision conflicts with restart contract")

        result = {
            "schema": "flashpatch-final-readback-v1",
            "campaign_id": state["campaign_id"],
            "scope_partition": {
                "scope_count": len(scope),
                "terminal_count": len(terminal_ids),
                "remaining_ids": remaining,
            },
            "selection": {
                "verdict": selection["verdict"],
                "nonterminal": selection_nonterminal,
                "artifact": "evidence/wf-selection.json",
                "sha256": self._sha256(self.repository / "evidence" / "wf-selection.json"),
            },
            "benchmark": {
                "artifact": str(benchmark_path.relative_to(self.repository)),
                "sha256": self._sha256(benchmark_path),
                "verdict": benchmark["verdict"],
            },
            "hero_replay": {
                "artifact": str(hero_path.relative_to(self.repository)),
                "sha256": self._sha256(hero_path),
                "verdict": hero["verdict"],
            },
            "terminal_evidence": {
                "verified_records": len(records),
            },
            "verdict": "PASS",
        }
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
