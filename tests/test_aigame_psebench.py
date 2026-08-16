from __future__ import annotations

import json
from pathlib import Path

from flashpatch.psebench import AIGamePSEBench


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_aigame_psebench_emits_machine_results_and_beats_baselines(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    results = AIGamePSEBench(CORPUS, workspace=tmp_path / "work").run(output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == results
    assert results["schema"] == "aigame-psebench-results-v1"
    assert results["budgets"] == {"interaction-burst": 12, "pulse-light": 2}
    assert set(results["methods"]) == {
        "flashpatch-neurosafety-compiler",
        "random-exploration",
        "scripted-exploration",
        "video-only-repair",
    }

    compiler = results["methods"]["flashpatch-neurosafety-compiler"]
    assert compiler["hazard_discovery_rate"] == 1.0
    assert compiler["node_top1_accuracy"] == 1.0
    assert compiler["parameter_top1_accuracy"] == 1.0
    assert compiler["interaction_discovery_rate"] == 1.0
    assert compiler["source_patch_hazard_removal_rate"] == 1.0
    assert compiler["gameplay_timing_preservation_rate"] == 1.0
    assert compiler["mean_modified_source_parameters"] == 1.0
    assert compiler["runtime_filter_used"] is False

    for baseline_name in ("random-exploration", "scripted-exploration", "video-only-repair"):
        baseline = results["methods"][baseline_name]
        assert compiler["parameter_top1_accuracy"] > baseline["parameter_top1_accuracy"]
        assert compiler["source_patch_hazard_removal_rate"] > baseline["source_patch_hazard_removal_rate"]

    assert {row["family_id"] for row in compiler["families"]} == {
        "interaction-burst",
        "pulse-light",
    }
    assert all(row["source_patch"].endswith(".diff") for row in compiler["families"])
    assert results["verdict"] == "PASS"
