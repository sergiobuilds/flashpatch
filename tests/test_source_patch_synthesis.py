from __future__ import annotations

from pathlib import Path

import pytest

from flashpatch.corpus import GodotCorpus
from flashpatch.godot import GodotReplayRunner
from flashpatch.patching import SourcePatchSynthesizer
from flashpatch.provenance import RenderContributor


CORPUS = Path(__file__).parents[1] / "benchmarks" / "aigame-psebench" / "corpus"


def test_source_patch_synthesis_emits_minimal_tscn_diff_and_safe_replay(tmp_path: Path) -> None:
    corpus = GodotCorpus(CORPUS)
    synthesizer = SourcePatchSynthesizer(
        corpus,
        allowed_parameters={"flash_interval_frames", "burst_intensity"},
    )

    for family_id, replacement in (("pulse-light", 100), ("interaction-burst", 0.0)):
        family = corpus.family(family_id)
        contributor = RenderContributor(
            node_path=family.causal_node,
            parameter=family.causal_parameter,
            source_file=family.source_file,
            source_line=3,
            value=None,
            hazard_frames=(1,),
        )
        result = synthesizer.synthesize(
            family_id,
            contributor,
            replacement=replacement,
            output_dir=tmp_path / family_id,
        )

        assert result.changed_parameter_count == 1
        assert result.patch_file.suffix == ".diff"
        assert "main.tscn" in result.patch_file.read_text(encoding="utf-8")
        assert f"+{family.causal_parameter} = {replacement}" in result.patch_file.read_text(encoding="utf-8")
        assert result.patched_scene.suffix == ".tscn"

        replay = GodotReplayRunner(result.project_dir).replay(
            family.trace_file,
            tmp_path / f"{family_id}-safe.json",
        )
        observations = replay.get("observations")
        assert isinstance(observations, list)
        assert max(float(value) for value in observations) < family.risk_threshold


def test_source_patch_synthesis_rejects_parameter_outside_allowlist(tmp_path: Path) -> None:
    corpus = GodotCorpus(CORPUS)
    family = corpus.family("pulse-light")
    synthesizer = SourcePatchSynthesizer(corpus, allowed_parameters={"flash_interval_frames"})
    contributor = RenderContributor(
        node_path=family.causal_node,
        parameter="unapproved_filter",
        source_file=family.source_file,
        source_line=3,
        value=None,
        hazard_frames=(1,),
    )

    with pytest.raises(ValueError, match="not allowed"):
        synthesizer.synthesize(
            "pulse-light",
            contributor,
            replacement=0,
            output_dir=tmp_path,
        )
