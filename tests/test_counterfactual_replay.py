from __future__ import annotations

from pathlib import Path

from flashpatch.corpus import GodotCorpus
from flashpatch.counterfactual import CounterfactualReplay
from flashpatch.explorer import RiskSeekingExplorer
from flashpatch.provenance import RenderProvenanceCollector


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_counterfactual_replay_attributes_risk_to_source_parameter(tmp_path: Path) -> None:
    corpus = GodotCorpus(CORPUS)
    explorer = RiskSeekingExplorer(corpus, workspace=tmp_path / "exploration")
    collector = RenderProvenanceCollector(corpus, workspace=tmp_path / "provenance")
    replay = CounterfactualReplay(corpus, workspace=tmp_path / "counterfactual")

    replacements = {"pulse-light": 100, "interaction-burst": 0.0}
    original_sources = {
        family_id: corpus.family(family_id).source_file.read_text(encoding="utf-8")
        for family_id in replacements
    }

    for family_id, replacement in replacements.items():
        exploration = explorer.explore(family_id, episode_budget=12)
        contributor = collector.collect(family_id, exploration.action_trace).contributors[0]
        result = replay.evaluate(
            family_id,
            exploration.action_trace,
            contributor,
            replacement=replacement,
        )

        assert result.factual_max_risk == 1.0
        assert result.counterfactual_max_risk < corpus.family(family_id).risk_threshold
        assert result.causal_contribution > 0.0
        assert result.hazard_removed
        assert result.modified_source.is_file()
        assert str(replacement) in result.modified_source.read_text(encoding="utf-8")
        assert corpus.family(family_id).source_file.read_text(encoding="utf-8") == original_sources[family_id]
