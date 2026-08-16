from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from flashpatch.core import analyze


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "l7_product_errors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_l7_product_fixtures_are_byte_bound_and_have_fail_closed_provenance() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "flashpatch-l7-product-provenance-v2"
    assert len(manifest["cases"]) == 2
    expected = {
        "gchess-fn": "GOLD_CAPTURE_CONFLICT_REQUIRES_REVIEW",
        "super-mario-solar-engine-fp": "RESOLVED_HISTORICAL_FALSE_POSITIVE",
    }
    for case in manifest["cases"]:
        fixture = FIXTURE_ROOT / case["fixture"]
        assert _sha256(fixture) == case["sha256"]
        assert case["classification"] == expected[case["id"]]
        assert case["fresh_l7"]["run_id"] == "20260810-8f11da0"
        assert len(case["fresh_l7"]["contract_sha256"]) == 64


def test_gchess_disagreement_is_static_small_area_pattern_under_current_detector() -> None:
    data = np.load(FIXTURE_ROOT / "gchess-blind-23583b652b8c31faa39089555f8fee97.npz")
    frames = data["frames"]
    adjacent_changes = np.any(frames[1:] != frames[:-1], axis=-1)
    result = analyze(data["frames"], data["timestamps"])

    # The oracle's 0.00--0.55s regular-pattern label cannot supply temporal
    # evidence for this byte-bound capture: every frame is identical.
    assert np.array_equal(frames, np.broadcast_to(frames[0], frames.shape))
    assert not np.any(adjacent_changes)
    assert result.hazardous is False
    assert result.windows == ()
    assert result.max_flash_count == 0.0
    assert 0.079 < result.max_affected_fraction < 0.081


def test_super_mario_smooth_regular_pattern_flow_remains_safe() -> None:
    data = np.load(
        FIXTURE_ROOT / "super-mario-solar-engine-blind-1f16ef6168ce276bc5f7da2fa7e18d01.npz"
    )
    result = analyze(data["frames"], data["timestamps"])
    assert result.hazardous is False
    assert result.windows == ()
    assert 0.39 < result.max_affected_fraction < 0.40
