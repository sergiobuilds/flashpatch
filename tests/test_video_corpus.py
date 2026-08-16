from __future__ import annotations

from pathlib import Path

import numpy as np

from flashpatch.core import analyze
from flashpatch.video_corpus import SealedVideoCorpus


CORPUS_ROOT = Path(__file__).parents[1] / "corpus" / "competition"


def test_sealed_corpus_has_family_split_rights_hashes_and_reproducible_gold() -> None:
    corpus = SealedVideoCorpus(CORPUS_ROOT)

    assert set(corpus.split_ids("development"))
    assert set(corpus.split_ids("sealed"))
    assert set(corpus.split_ids("development")).isdisjoint(corpus.split_ids("sealed"))
    assert {case.category for case in corpus.cases} == {
        "real",
        "synthetic",
        "boundary",
        "safe-negative",
        "transformed",
    }
    assert len({case.source_family for case in corpus.cases}) < len(corpus.cases)

    for case in corpus.cases:
        assert case.rights.spdx_id
        assert case.rights.redistribution in {"allowed", "generated-in-project"}
        frames, timestamps, gold = corpus.load(case.case_id)
        assert corpus.verify_hashes(case.case_id)
        assert gold.shape == frames.shape[:3]
        detection = analyze(frames, timestamps)
        np.testing.assert_array_equal(detection.hazard_mask, gold)
        assert detection.hazardous is case.hazardous


def test_transformed_family_cannot_cross_the_sealed_split() -> None:
    corpus = SealedVideoCorpus(CORPUS_ROOT)

    family_splits: dict[str, set[str]] = {}
    for case in corpus.cases:
        family_splits.setdefault(case.source_family, set()).add(case.split)

    assert all(len(splits) == 1 for splits in family_splits.values())
