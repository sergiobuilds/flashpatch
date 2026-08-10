from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flashpatch.standards import (
    BT709_RELATIVE_LUMINANCE_COEFFICIENTS,
    WCAG_SMALL_SAFE_AREA_PIXELS,
    general_flash_transition_is_flash,
    normalize_color_profile,
    red_flash_transition_is_flash,
    regular_pattern_is_hazardous,
    saturated_red_threshold_is_met,
    evaluate_boundary_vector,
    wcag_small_area_exemption_applies,
)


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "docs" / "research" / "standards-boundary-vectors.json"


def test_general_flash_boundaries_are_inclusive_for_delta_and_strict_for_dark_state() -> None:
    assert not general_flash_transition_is_flash(0.2, 0.299)
    assert general_flash_transition_is_flash(0.2, 0.3)
    assert not general_flash_transition_is_flash(0.8, 0.9)


def test_saturated_red_boundaries_are_inclusive_for_ratio_and_strict_for_chroma_distance() -> None:
    assert saturated_red_threshold_is_met(0.8, np.nextafter(0.2, np.inf))
    assert not saturated_red_threshold_is_met(np.nextafter(0.8, 0.0), 0.21)
    assert not saturated_red_threshold_is_met(0.8, 0.2)
    assert red_flash_transition_is_flash((255, 0, 0), (0, 255, 0))
    assert not red_flash_transition_is_flash((0, 0, 0), (0, 255, 0))


def test_wcag_small_area_boundary_is_strict() -> None:
    assert WCAG_SMALL_SAFE_AREA_PIXELS == 21_824
    assert wcag_small_area_exemption_applies(21_823)
    assert not wcag_small_area_exemption_applies(21_824)


def test_regular_pattern_boundaries_follow_fixed_itu_profile() -> None:
    assert not regular_pattern_is_hazardous(
        stripe_pairs=5, affected_fraction=0.26, dynamic=True
    )
    assert not regular_pattern_is_hazardous(
        stripe_pairs=6, affected_fraction=0.25, dynamic=True
    )
    assert regular_pattern_is_hazardous(
        stripe_pairs=6, affected_fraction=np.nextafter(0.25, 1.0), dynamic=True
    )
    assert not regular_pattern_is_hazardous(
        stripe_pairs=6, affected_fraction=0.40, dynamic=False
    )
    assert regular_pattern_is_hazardous(
        stripe_pairs=6, affected_fraction=np.nextafter(0.40, 1.0), dynamic=False
    )
    assert not regular_pattern_is_hazardous(
        stripe_pairs=6,
        affected_fraction=1.0,
        dynamic=True,
        smooth_one_direction=True,
    )


def test_product_color_contract_accepts_only_explicit_srgb_bt709() -> None:
    assert normalize_color_profile("srgb") == "srgb-bt709"
    assert normalize_color_profile("bt709") == "srgb-bt709"
    assert BT709_RELATIVE_LUMINANCE_COEFFICIENTS == (0.2126, 0.7152, 0.0722)
    for unsupported in ("bt601", "unknown", "", None):
        with pytest.raises(ValueError, match="sRGB/BT.709"):
            normalize_color_profile(unsupported)


def test_executable_boundary_vector_manifest_covers_frozen_standard_contract() -> None:
    manifest = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["profile"] == "flashpatch-sdr-srgb-bt709-v1"
    assert set(manifest["source_claim_ids"]) == {
        "wcag22-general-flash-threshold",
        "wcag22-red-flash-threshold",
        "wcag22-threshold-vectors",
        "itu-bt1702-flash-gate",
        "itu-bt1702-pattern-gate",
        "itu-bt1702-michelson-boundary-conflict",
        "itu-bt709-luma-matrix-precision",
        "iso-9241-391-spatial-stripe-hazard",
    }
    vectors = {vector["id"]: vector for vector in manifest["vectors"]}
    assert set(vectors) == {
        "general-delta-below",
        "general-delta-at",
        "general-dark-at",
        "red-ratio-below",
        "red-ratio-at",
        "red-distance-at",
        "area-21823",
        "area-21824",
        "frequency-three-closed-same-state",
        "frequency-three-closed-different-state",
        "frequency-four",
        "itu-dark-below-delta-below",
        "itu-dark-below-delta-at",
        "itu-michelson-strict-at",
        "itu-michelson-inclusive-at",
        "pattern-five-pairs-dynamic-above-area",
        "pattern-six-pairs-dynamic-at-area",
        "pattern-six-pairs-dynamic-above-area",
        "pattern-six-pairs-static-at-area",
        "pattern-six-pairs-static-above-area",
        "pattern-smooth-flow-exempt",
        "color-srgb-bt709",
        "color-bt601-rejected",
    }
    assert all(vector["source_claim_id"] in manifest["source_claim_ids"] for vector in vectors.values())
    assert all("expected" in vector and "input" in vector for vector in vectors.values())
    for vector in vectors.values():
        assert evaluate_boundary_vector(vector) == vector["expected"]
