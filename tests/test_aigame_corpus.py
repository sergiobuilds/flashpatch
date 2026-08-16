from __future__ import annotations

from pathlib import Path

from flashpatch.corpus import GodotCorpus


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_control_corpus_runs_two_known_cause_scene_families(tmp_path: Path) -> None:
    corpus = GodotCorpus(CORPUS)

    assert corpus.family_ids == ("interaction-burst", "pulse-light")
    for family_id in corpus.family_ids:
        family = corpus.family(family_id)
        assert family.project_file.is_file()
        assert family.source_file.is_file()
        assert family.causal_node
        assert family.causal_parameter

    pulse = corpus.replay("pulse-light", tmp_path / "pulse.json")
    assert pulse["observations"] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    interaction = corpus.replay("interaction-burst", tmp_path / "interaction.json")
    assert interaction["observations"] == [0.0, 0.0, 1.0, 0.0]
    assert corpus.family("interaction-burst").interaction_cause
