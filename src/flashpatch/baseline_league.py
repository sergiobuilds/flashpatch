from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BaselineLeague:
    EXPECTED_METHODS = {
        "flashpatch-neurosafety-compiler",
        "random-exploration",
        "scripted-exploration",
        "video-only-repair",
    }

    def __init__(self, results: dict[str, object], *, source_sha256: str) -> None:
        self.results = results
        self.source_sha256 = source_sha256

    @classmethod
    def from_results(cls, path: Path) -> BaselineLeague:
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), source_sha256=hashlib.sha256(raw).hexdigest())

    def verify(self, output: Path) -> dict[str, object]:
        budgets = self.results.get("budgets")
        methods = self.results.get("methods")
        if not isinstance(budgets, dict) or not isinstance(methods, dict):
            raise ValueError("benchmark results omitted budgets or methods")
        if set(methods) != self.EXPECTED_METHODS:
            raise ValueError("benchmark method slate is not frozen")
        self._verify_budgets(budgets, methods)

        attribution_winner = self._unique_winner(methods, "parameter_top1_accuracy")
        patch_winner = self._unique_winner(methods, "source_patch_hazard_removal_rate")
        discovery_values = {
            name: self._metric(record, "hazard_discovery_rate")
            for name, record in methods.items()
        }
        best_discovery = max(discovery_values.values())
        discovery_winners = sorted(
            name for name, value in discovery_values.items() if value == best_discovery
        )
        video = methods["video-only-repair"]
        if not isinstance(video, dict):
            raise ValueError("video-only baseline record is invalid")
        result = {
            "schema": "aigame-psebench-baseline-league-v1",
            "source_results_sha256": f"sha256:{self.source_sha256}",
            "methods": sorted(methods),
            "budgets": budgets,
            "budget_fairness": "PASS",
            "source_attribution_winner": attribution_winner,
            "source_patch_winner": patch_winner,
            "hazard_discovery_result": "TIE" if len(discovery_winners) > 1 else discovery_winners[0],
            "hazard_discovery_leaders": discovery_winners,
            "video_only_runtime_filter": video.get("runtime_filter_used") is True,
            "verdict": "PASS",
        }
        if attribution_winner != "flashpatch-neurosafety-compiler":
            result["verdict"] = "FAIL"
        if patch_winner != "flashpatch-neurosafety-compiler":
            result["verdict"] = "FAIL"
        if not result["video_only_runtime_filter"]:
            result["verdict"] = "FAIL"
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def _verify_budgets(
        budgets: dict[object, object],
        methods: dict[object, object],
    ) -> None:
        for method_name, method in methods.items():
            if not isinstance(method, dict):
                raise ValueError(f"invalid method record: {method_name}")
            families = method.get("families")
            if not isinstance(families, list):
                raise ValueError(f"method omitted family records: {method_name}")
            if {row.get("family_id") for row in families if isinstance(row, dict)} != set(budgets):
                raise ValueError(f"method family slate differs: {method_name}")
            for row in families:
                if not isinstance(row, dict):
                    raise ValueError(f"invalid family record: {method_name}")
                family_id = row["family_id"]
                budget = budgets[family_id]
                episodes = row.get("episodes_used")
                if not isinstance(budget, int) or not isinstance(episodes, int):
                    raise ValueError("episode budgets and usage must be integers")
                if episodes > budget:
                    raise ValueError(
                        f"episode budget exceeded: {method_name}/{family_id} used {episodes}>{budget}"
                    )

    @classmethod
    def _unique_winner(cls, methods: dict[object, object], metric: str) -> str:
        scores = {str(name): cls._metric(record, metric) for name, record in methods.items()}
        best = max(scores.values())
        winners = [name for name, value in scores.items() if value == best]
        return winners[0] if len(winners) == 1 else "TIE"

    @staticmethod
    def _metric(record: object, name: str) -> float:
        if not isinstance(record, dict):
            raise ValueError("invalid method record")
        value = record.get(name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"method omitted numeric metric: {name}")
        return float(value)
