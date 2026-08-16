from __future__ import annotations

from pathlib import Path

from flashpatch.corpus import GodotCorpus
from flashpatch.explorer import RiskSeekingExplorer
from flashpatch.provenance import RenderProvenanceCollector


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_render_provenance_maps_hazard_to_node_and_source_parameter(tmp_path: Path) -> None:
    corpus = GodotCorpus(CORPUS)
    explorer = RiskSeekingExplorer(corpus, workspace=tmp_path / "exploration")
    collector = RenderProvenanceCollector(corpus, workspace=tmp_path / "provenance")

    for family_id, budget in (("pulse-light", 2), ("interaction-burst", 12)):
        exploration = explorer.explore(family_id, episode_budget=budget)
        result = collector.collect(family_id, exploration.action_trace)
        family = corpus.family(family_id)

        assert result.hazard_frames
        assert result.contributors
        top = result.contributors[0]
        assert top.node_path == family.causal_node
        assert top.parameter == family.causal_parameter
        assert top.source_file == family.source_file
        assert top.source_line > 0
        assert top.value is not None
        assert top.hazard_frames == result.hazard_frames
