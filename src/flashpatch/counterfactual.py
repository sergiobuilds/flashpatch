from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .corpus import GodotCorpus
from .godot import GodotReplayRunner
from .provenance import RenderContributor


@dataclass(frozen=True)
class CounterfactualResult:
    family_id: str
    parameter: str
    replacement: object
    factual_max_risk: float
    counterfactual_max_risk: float
    causal_contribution: float
    hazard_removed: bool
    modified_source: Path


class CounterfactualReplay:
    def __init__(self, corpus: GodotCorpus, *, workspace: Path) -> None:
        self.corpus = corpus
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        family_id: str,
        action_trace: tuple[dict[str, object], ...],
        contributor: RenderContributor,
        *,
        replacement: object,
    ) -> CounterfactualResult:
        family = self.corpus.family(family_id)
        trace_path = self.workspace / f"{family_id}-trace.json"
        trace_path.write_text(
            json.dumps({"fixed_fps": 60, "actions": action_trace}, sort_keys=True),
            encoding="utf-8",
        )
        factual = self.corpus.replay(
            family_id,
            self.workspace / f"{family_id}-factual.json",
            trace=trace_path,
        )

        project_copy = self.workspace / f"{family_id}-project"
        if project_copy.exists():
            shutil.rmtree(project_copy)
        shutil.copytree(family.project_file.parent, project_copy)
        source_relative = contributor.source_file.relative_to(family.project_file.parent)
        modified_source = project_copy / source_relative
        self._replace_exported_parameter(
            modified_source,
            contributor.parameter,
            replacement,
        )
        counterfactual = GodotReplayRunner(project_copy).replay(
            trace_path,
            self.workspace / f"{family_id}-counterfactual.json",
        )

        factual_max = self._max_risk(factual)
        counterfactual_max = self._max_risk(counterfactual)
        return CounterfactualResult(
            family_id=family_id,
            parameter=contributor.parameter,
            replacement=replacement,
            factual_max_risk=factual_max,
            counterfactual_max_risk=counterfactual_max,
            causal_contribution=factual_max - counterfactual_max,
            hazard_removed=counterfactual_max < family.risk_threshold,
            modified_source=modified_source,
        )

    @staticmethod
    def _max_risk(replay: dict[str, object]) -> float:
        observations = replay.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError("counterfactual replay omitted observations")
        return max((float(value) for value in observations), default=0.0)

    @staticmethod
    def _replace_exported_parameter(source: Path, parameter: str, replacement: object) -> None:
        pattern = re.compile(
            rf"^(?P<prefix>\s*@export\s+var\s+{re.escape(parameter)}\s*(?::[^=]+)?=\s*).+$"
        )
        replacement_text = str(replacement).lower() if isinstance(replacement, bool) else str(replacement)
        lines = source.read_text(encoding="utf-8").splitlines()
        matches = 0
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                lines[index] = match.group("prefix") + replacement_text
                matches += 1
        if matches != 1:
            raise ValueError(f"expected one exported parameter {parameter!r}, found {matches}")
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
