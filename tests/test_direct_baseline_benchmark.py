from __future__ import annotations

import json
from pathlib import Path

from flashpatch.direct_baseline import DirectBaselineBenchmark

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "corpus" / "competition"


def test_direct_baseline_runs_frozen_sealed_league_and_reproduces(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = DirectBaselineBenchmark(CORPUS).run(first_path)
    second = DirectBaselineBenchmark(CORPUS).run(second_path)

    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert first == second
    assert first["schema"] == "flashpatch-direct-baseline-v1"
    assert first["corpus"]["split"] == "sealed"
    assert first["corpus"]["case_ids"] == [
        "sealed-real-sintel-opening",
        "sealed-synthetic-red",
        "sealed-transformed-red-vfr",
    ]
    assert set(first["detection_league"]) == {
        "flashpatch",
        "ea-iris",
        "epi-lens",
    }
    assert set(first["repair_league"]) == {"flashpatch", "ffmpeg-photosensitivity"}
    assert first["fairness"]["same_case_ids"] is True
    assert first["fairness"]["sealed_inputs_hash_verified"] is True
    assert first["reproduction"]["runs_equal"] is True
    assert first["verdict"] == "PASS"


def test_direct_baseline_rejects_unavailable_required_adapter(tmp_path: Path) -> None:
    benchmark = DirectBaselineBenchmark(CORPUS, iris_command=("missing-iris",))

    try:
        benchmark.run(tmp_path / "result.json")
    except RuntimeError as error:
        assert "required baseline adapter unavailable" in str(error)
    else:
        raise AssertionError("missing required adapter did not fail closed")