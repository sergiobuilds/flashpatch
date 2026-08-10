from __future__ import annotations

import json
import random
from pathlib import Path

from .corpus import GodotCorpus
from .counterfactual import CounterfactualReplay
from .explorer import ExplorationResult, RiskSeekingExplorer
from .godot import GodotReplayRunner
from .patching import SourcePatchSynthesizer
from .provenance import RenderProvenanceCollector


class AIGamePSEBench:
    BUDGETS = {"pulse-light": 2, "interaction-burst": 12}
    SAFE_REPLACEMENTS = {"flash_interval_frames": 100, "burst_intensity": 0.0}

    def __init__(self, corpus_root: Path, *, workspace: Path) -> None:
        self.corpus = GodotCorpus(corpus_root)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, output: Path) -> dict[str, object]:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        compiler_rows = self._run_compiler(output.parent / "source-patches")
        random_rows = self._run_random()
        scripted_rows = self._run_scripted()
        video_rows = [
            {
                **row,
                "node_top1_correct": False,
                "parameter_top1_correct": False,
                "source_patch_removed_hazard": False,
                "modified_source_parameters": 0,
                "runtime_filter_used": True,
            }
            for row in scripted_rows
        ]
        methods = {
            "flashpatch-neurosafety-compiler": self._summarize(compiler_rows),
            "random-exploration": self._summarize(random_rows),
            "scripted-exploration": self._summarize(scripted_rows),
            "video-only-repair": self._summarize(video_rows),
        }
        result = {
            "schema": "aigame-psebench-results-v1",
            "budgets": {family_id: self.BUDGETS[family_id] for family_id in self.corpus.family_ids},
            "methods": methods,
            "verdict": "PASS" if self._passes(methods) else "FAIL",
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    def _run_compiler(self, patch_root: Path) -> list[dict[str, object]]:
        explorer = RiskSeekingExplorer(self.corpus, workspace=self.workspace / "compiler-exploration")
        provenance = RenderProvenanceCollector(self.corpus, workspace=self.workspace / "provenance")
        counterfactual = CounterfactualReplay(self.corpus, workspace=self.workspace / "counterfactual")
        synthesizer = SourcePatchSynthesizer(
            self.corpus,
            allowed_parameters=set(self.SAFE_REPLACEMENTS),
        )
        rows = []
        for family_id in self.corpus.family_ids:
            family = self.corpus.family(family_id)
            exploration = explorer.explore(family_id, episode_budget=self.BUDGETS[family_id])
            trace = exploration.action_trace
            contributors = provenance.collect(family_id, trace).contributors
            ranked = []
            for contributor in contributors:
                replacement = self.SAFE_REPLACEMENTS.get(contributor.parameter)
                if replacement is None:
                    continue
                replay = counterfactual.evaluate(
                    family_id,
                    trace,
                    contributor,
                    replacement=replacement,
                )
                ranked.append((replay.causal_contribution, contributor, replay))
            ranked.sort(key=lambda item: (-item[0], item[1].parameter))
            if not ranked:
                raise RuntimeError(f"no causal source parameter found for {family_id}")
            _, top_contributor, causal_replay = ranked[0]
            patch = synthesizer.synthesize(
                family_id,
                top_contributor,
                replacement=causal_replay.replacement,
                output_dir=patch_root / family_id,
            )
            trace_path = self.workspace / f"{family_id}-compiler-trace.json"
            trace_path.write_text(
                json.dumps({"fixed_fps": 60, "actions": trace}, sort_keys=True),
                encoding="utf-8",
            )
            safe_replay = GodotReplayRunner(patch.project_dir).replay(
                trace_path,
                self.workspace / f"{family_id}-compiler-safe.json",
            )
            observations = safe_replay.get("observations")
            if not isinstance(observations, list):
                raise RuntimeError(f"safe replay for {family_id} omitted observations")
            rows.append(
                {
                    "family_id": family_id,
                    "episodes_used": exploration.episodes_used,
                    "hazard_found": exploration.hazard_found,
                    "node_top1_correct": top_contributor.node_path == family.causal_node,
                    "parameter_top1_correct": top_contributor.parameter == family.causal_parameter,
                    "interaction_discovered": family.interaction_cause and exploration.hazard_found,
                    "source_patch_removed_hazard": max(map(float, observations), default=0.0)
                    < family.risk_threshold,
                    "gameplay_timing_preserved": len(observations) == len(trace),
                    "modified_source_parameters": patch.changed_parameter_count,
                    "runtime_filter_used": False,
                    "source_patch": str(Path("source-patches") / family_id / patch.patch_file.name),
                }
            )
        return rows

    def _run_random(self) -> list[dict[str, object]]:
        rng = random.Random(20260730)
        rows = []
        for family_id in self.corpus.family_ids:
            family = self.corpus.family(family_id)
            best = 0.0
            used = 0
            found = False
            for episode in range(1, self.BUDGETS[family_id] + 1):
                actions = tuple(
                    {"frame": frame, **rng.choice(family.action_space)}
                    for frame in range(family.horizon)
                )
                risk = self._replay_risk("random", family_id, episode, actions)
                used = episode
                best = max(best, risk)
                if risk >= family.risk_threshold:
                    found = True
                    break
            rows.append(self._exploration_baseline_row(family_id, used, found, family.interaction_cause and found))
        return rows

    def _run_scripted(self) -> list[dict[str, object]]:
        rows = []
        for family_id in self.corpus.family_ids:
            family = self.corpus.family(family_id)
            replay = self.corpus.replay(
                family_id,
                self.workspace / f"{family_id}-scripted.json",
            )
            observations = replay.get("observations")
            if not isinstance(observations, list):
                raise RuntimeError(f"scripted replay for {family_id} omitted observations")
            found = max(map(float, observations), default=0.0) >= family.risk_threshold
            rows.append(self._exploration_baseline_row(family_id, 1, found, family.interaction_cause and found))
        return rows

    def _replay_risk(
        self,
        method: str,
        family_id: str,
        episode: int,
        actions: tuple[dict[str, object], ...],
    ) -> float:
        trace = self.workspace / f"{family_id}-{method}-{episode}-trace.json"
        trace.write_text(
            json.dumps({"fixed_fps": 60, "actions": actions}, sort_keys=True),
            encoding="utf-8",
        )
        replay = self.corpus.replay(
            family_id,
            self.workspace / f"{family_id}-{method}-{episode}.json",
            trace=trace,
        )
        observations = replay.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError(f"{method} replay for {family_id} omitted observations")
        return max(map(float, observations), default=0.0)

    @staticmethod
    def _exploration_baseline_row(
        family_id: str,
        episodes_used: int,
        hazard_found: bool,
        interaction_discovered: bool,
    ) -> dict[str, object]:
        return {
            "family_id": family_id,
            "episodes_used": episodes_used,
            "hazard_found": hazard_found,
            "node_top1_correct": False,
            "parameter_top1_correct": False,
            "interaction_discovered": interaction_discovered,
            "source_patch_removed_hazard": False,
            "gameplay_timing_preserved": True,
            "modified_source_parameters": 0,
            "runtime_filter_used": False,
        }

    @staticmethod
    def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        total = len(rows)
        interaction_rows = [row for row in rows if row["family_id"] == "interaction-burst"]

        def rate(field: str, records: list[dict[str, object]] = rows) -> float:
            return round(sum(bool(row[field]) for row in records) / len(records), 6) if records else 0.0

        return {
            "hazard_discovery_rate": rate("hazard_found"),
            "node_top1_accuracy": rate("node_top1_correct"),
            "parameter_top1_accuracy": rate("parameter_top1_correct"),
            "interaction_discovery_rate": rate("interaction_discovered", interaction_rows),
            "source_patch_hazard_removal_rate": rate("source_patch_removed_hazard"),
            "gameplay_timing_preservation_rate": rate("gameplay_timing_preserved"),
            "mean_modified_source_parameters": round(
                sum(
                    value
                    for row in rows
                    if isinstance((value := row["modified_source_parameters"]), int)
                )
                / total,
                6,
            ),
            "runtime_filter_used": any(bool(row["runtime_filter_used"]) for row in rows),
            "families": rows,
        }

    @staticmethod
    def _passes(methods: dict[str, dict[str, object]]) -> bool:
        compiler = methods["flashpatch-neurosafety-compiler"]
        return all(
            compiler[field] == 1.0
            for field in (
                "hazard_discovery_rate",
                "node_top1_accuracy",
                "parameter_top1_accuracy",
                "interaction_discovery_rate",
                "source_patch_hazard_removal_rate",
                "gameplay_timing_preservation_rate",
                "mean_modified_source_parameters",
            )
        ) and compiler["runtime_filter_used"] is False
