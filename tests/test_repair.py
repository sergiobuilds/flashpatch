from __future__ import annotations

import numpy as np

from flashpatch.core import analyze, repair_with_evidence


def _general_clip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.full((60, 24, 32, 3), 96, dtype=np.uint8)
    region = np.zeros((24, 32), dtype=bool)
    region[4:20, 6:26] = True
    for index in range(len(frames)):
        frames[index, region] = 255 if (index // 3) % 2 else 0
    return frames, np.arange(len(frames), dtype=np.float64) / 30.0, region


def test_feathered_local_repair_closes_hazard_and_preserves_nonhazard_pixels() -> None:
    frames, timestamps, region = _general_clip()
    detection = analyze(frames, timestamps)

    outcome = repair_with_evidence(frames, timestamps, detection)

    assert outcome.verified
    assert not analyze(outcome.frames, timestamps).hazardous
    assert 0.0 < outcome.changed_fraction < 0.25
    assert outcome.outside_hazard_unchanged
    assert np.array_equal(outcome.frames[:, ~region], frames[:, ~region])
    assert np.any((outcome.feather_mask > 0.0) & (outcome.feather_mask < 1.0))
    assert outcome.structural_similarity > 0.20
    assert outcome.hazard_band_power_after <= outcome.hazard_band_power_before + 1e-12


def test_red_and_regular_pattern_repairs_use_bounded_fallback_when_needed() -> None:
    red = np.full((12, 16, 16, 3), 24, dtype=np.uint8)
    red[:, :, :12] = np.array(
        [(255, 0, 0) if index % 2 else (0, 64, 64) for index in range(len(red))]
    )[:, None, None, :]
    red_times = np.arange(len(red), dtype=np.float64) / 10.0

    pattern = np.full((8, 16, 32, 3), 96, dtype=np.uint8)
    for frame_index in range(len(pattern)):
        for stripe in range(12):
            pattern[frame_index, :, stripe * 2 : stripe * 2 + 2] = (
                240 if (stripe + frame_index) % 2 else 8
            )
    pattern_times = np.arange(len(pattern), dtype=np.float64) / 10.0

    for frames, timestamps in ((red, red_times), (pattern, pattern_times)):
        detection = analyze(frames, timestamps)
        outcome = repair_with_evidence(frames, timestamps, detection)
        assert outcome.verified
        assert not analyze(outcome.frames, timestamps).hazardous
        assert outcome.outside_hazard_unchanged

    fallback_clip = np.full((8, 16, 16, 3), 89, dtype=np.uint8)
    low = np.array((26, 97, 83), dtype=np.uint8)
    high = np.array((157, 228, 200), dtype=np.uint8)
    stripe_values = (32, 32, 255, 0, 255, 0, 255, 255)
    for index, stripe_value in enumerate(stripe_values):
        fallback_clip[index, 5:9, 3:11] = high if index % 2 else low
        fallback_clip[index, :, ::2] = stripe_value
    fallback_times = np.arange(len(fallback_clip), dtype=np.float64) / 15.0
    fallback = repair_with_evidence(
        fallback_clip,
        fallback_times,
        analyze(fallback_clip, fallback_times),
    )
    assert fallback.fallback_used
    assert fallback.verified
