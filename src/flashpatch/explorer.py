from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from .corpus import GodotCorpus


@dataclass(frozen=True)
class ExplorationResult:
    family_id: str
    hazard_found: bool
    max_risk: float
    episodes_used: int
    action_trace: tuple[dict[str, object], ...]


class RiskSeekingExplorer:
    def __init__(self, corpus: GodotCorpus, *, workspace: Path) -> None:
        self.corpus = corpus
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def explore(self, family_id: str, *, episode_budget: int) -> ExplorationResult:
        if episode_budget <= 0:
            raise ValueError("episode_budget must be positive")
        family = self.corpus.family(family_id)
        best_risk = float("-inf")
        best_trace: tuple[dict[str, object], ...] = ()
        episodes_used = 0

        candidates = itertools.product(family.action_space, repeat=family.horizon)
        for episode, candidate in enumerate(candidates, start=1):
            if episode > episode_budget:
                break
            actions = tuple({"frame": frame, **action} for frame, action in enumerate(candidate))
            trace_path = self.workspace / f"{family_id}-episode-{episode}-trace.json"
            output_path = self.workspace / f"{family_id}-episode-{episode}-result.json"
            trace_path.write_text(
                json.dumps({"fixed_fps": 60, "actions": actions}, sort_keys=True),
                encoding="utf-8",
            )
            replay = self.corpus.replay(family_id, output_path, trace=trace_path)
            observations = replay.get("observations")
            if not isinstance(observations, list):
                raise RuntimeError(f"replay for {family_id} omitted observations")
            risk = max((float(value) for value in observations), default=0.0)
            episodes_used = episode
            if risk > best_risk:
                best_risk = risk
                best_trace = actions
            if risk >= family.risk_threshold:
                break

        return ExplorationResult(
            family_id=family_id,
            hazard_found=best_risk >= family.risk_threshold,
            max_risk=max(best_risk, 0.0),
            episodes_used=episodes_used,
            action_trace=best_trace,
        )
