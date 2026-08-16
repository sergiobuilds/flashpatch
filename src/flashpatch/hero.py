from __future__ import annotations

import json
from pathlib import Path

from .corpus import GodotCorpus
from .counterfactual import CounterfactualReplay
from .explorer import RiskSeekingExplorer
from .godot import GodotReplayRunner
from .patching import SourcePatchSynthesizer
from .provenance import RenderProvenanceCollector
from .psebench import AIGamePSEBench


class JudgeHeroReplay:
    FAMILY_ID = "interaction-burst"

    def __init__(self, corpus_root: Path, *, workspace: Path) -> None:
        self.corpus = GodotCorpus(corpus_root)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, output: Path) -> dict[str, object]:
        family = self.corpus.family(self.FAMILY_ID)
        exploration = RiskSeekingExplorer(
            self.corpus,
            workspace=self.workspace / "exploration",
        ).explore(
            self.FAMILY_ID,
            episode_budget=AIGamePSEBench.BUDGETS[self.FAMILY_ID],
        )
        provenance = RenderProvenanceCollector(
            self.corpus,
            workspace=self.workspace / "provenance",
        ).collect(self.FAMILY_ID, exploration.action_trace)
        counterfactual = CounterfactualReplay(
            self.corpus,
            workspace=self.workspace / "counterfactual",
        )
        ranked = []
        for contributor in provenance.contributors:
            replacement = AIGamePSEBench.SAFE_REPLACEMENTS.get(contributor.parameter)
            if replacement is None:
                continue
            replay = counterfactual.evaluate(
                self.FAMILY_ID,
                exploration.action_trace,
                contributor,
                replacement=replacement,
            )
            ranked.append((replay.causal_contribution, contributor, replay))
        ranked.sort(key=lambda item: (-item[0], item[1].parameter))
        if not ranked:
            raise RuntimeError("hero replay found no causal source parameter")
        _, contributor, causal_replay = ranked[0]

        patch = SourcePatchSynthesizer(
            self.corpus,
            allowed_parameters=set(AIGamePSEBench.SAFE_REPLACEMENTS),
        ).synthesize(
            self.FAMILY_ID,
            contributor,
            replacement=causal_replay.replacement,
            output_dir=self.workspace / "source-patch",
        )
        trace_path = self.workspace / "hero-trace.json"
        trace_path.write_text(
            json.dumps(
                {"fixed_fps": 60, "actions": exploration.action_trace},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        safe_replay = GodotReplayRunner(patch.project_dir).replay(
            trace_path,
            self.workspace / "safe-replay.json",
        )
        observations = safe_replay.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError("hero safe replay omitted observations")
        max_risk_after = max(map(float, observations), default=0.0)

        result = {
            "schema": "flashpatch-judge-hero-v1",
            "family_id": self.FAMILY_ID,
            "hazard": {
                "found": exploration.hazard_found,
                "max_risk_before": exploration.max_risk,
                "episodes_used": exploration.episodes_used,
            },
            "attribution": {
                "node": contributor.node_path,
                "parameter": contributor.parameter,
                "causal_contribution": causal_replay.causal_contribution,
            },
            "patch": {
                "changed_source_parameters": patch.changed_parameter_count,
                "diff_file": str(patch.patch_file),
            },
            "safe_replay": {
                "max_risk_after": max_risk_after,
                "same_action_trace": True,
                "timing_preserved": len(observations) == len(exploration.action_trace),
                "runtime_filter_used": False,
            },
            "verdict": "PASS"
            if (
                exploration.hazard_found
                and causal_replay.hazard_removed
                and max_risk_after < family.risk_threshold
                and len(observations) == len(exploration.action_trace)
            )
            else "FAIL",
        }
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
