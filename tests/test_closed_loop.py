from __future__ import annotations

from time import monotonic
from typing import cast

import numpy as np
import pytest

from flashpatch.core import (
    _SPATIAL_TILE_EDGE,
    HazardWindow,
    _alternating_stripe_span,
    _luminance_transition_directions,
    _longest_opposing_run,
    _red_transition_directions,
    _regular_pattern_localization,
    _temporal_localization,
    _tiled_temporal_localization,
    _window_longest_opposing_runs,
    analyze,
    itu_luminance_change_is_flash,
    relative_luminance,
    repair,
    srgb_to_linear,
)
from flashpatch.standards import regular_pattern_is_hazardous
from flashpatch.verify import verify


def _vectorized_general_directions(frames: np.ndarray, *, luminance_delta: float) -> np.ndarray:
    luminance = relative_luminance(frames)
    delta = np.diff(luminance, axis=0)
    darker = np.minimum(luminance[:-1], luminance[1:])
    return np.where((np.abs(delta) >= luminance_delta) & (darker < 0.8), np.sign(delta), 0).astype(np.int8)


def _vectorized_red_directions(frames: np.ndarray) -> np.ndarray:
    encoded = frames.astype(np.float64) / 255.0
    totals = np.sum(encoded, axis=-1)
    ratio = np.divide(encoded[..., 0], totals, out=np.zeros_like(totals), where=totals > 0.0)
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    x = 0.4124 * linear[..., 0] + 0.3576 * linear[..., 1] + 0.1805 * linear[..., 2]
    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    z = 0.0193 * linear[..., 0] + 0.1192 * linear[..., 1] + 0.9505 * linear[..., 2]
    denominator = x + 15.0 * y + 3.0 * z
    u = np.divide(4.0 * x, denominator, out=np.zeros_like(x), where=denominator > 0.0)
    v = np.divide(9.0 * y, denominator, out=np.zeros_like(y), where=denominator > 0.0)
    changed = (np.maximum(ratio[:-1], ratio[1:]) >= 0.8) & (np.hypot(np.diff(u, axis=0), np.diff(v, axis=0)) > 0.2)
    return np.where(changed, np.sign(np.diff(ratio, axis=0)), 0).astype(np.int8)


def _whole_frame_regular_pattern_reference(
    frame_array: np.ndarray, time_array: np.ndarray
) -> tuple[np.ndarray, list[HazardWindow], float]:
    """Frozen reference for the pre-strip whole-video implementation."""
    candidates = np.zeros(frame_array.shape[:3], dtype=bool)
    for frame_index, frame in enumerate(frame_array):
        plane = relative_luminance(frame[None, ...])[0]
        for row_index, row in enumerate(plane):
            span = _alternating_stripe_span(row)
            if span is not None:
                candidates[frame_index, row_index, span[0] : span[1]] = True
        for column_index, column in enumerate(plane.T):
            span = _alternating_stripe_span(column)
            if span is not None:
                candidates[frame_index, span[0] : span[1], column_index] = True

    dynamic = False
    for frame_index in range(1, len(frame_array)):
        changed = np.any(
            frame_array[frame_index] != frame_array[frame_index - 1], axis=-1
        )
        if np.any(
            changed & (candidates[frame_index - 1] | candidates[frame_index])
        ):
            dynamic = True
            break

    mask = np.zeros_like(candidates)
    max_fraction = 0.0
    for index, candidate in enumerate(candidates):
        fraction = float(np.mean(candidate))
        max_fraction = max(max_fraction, fraction)
        if regular_pattern_is_hazardous(
            stripe_pairs=6,
            affected_fraction=fraction,
            dynamic=dynamic,
        ):
            mask[index] = candidate

    windows: list[HazardWindow] = []
    active_frames = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if len(active_frames):
        windows.append(
            HazardWindow(
                start=float(time_array[active_frames[0]]),
                end=float(time_array[active_frames[-1]]),
                affected_fraction=max_fraction,
                flash_count=0.0,
                kind="regular_pattern",
            )
        )
    return mask, windows, max_fraction


def _vectorized_reference(frames: np.ndarray, timestamps: np.ndarray, *, delta: float, limit: int, area: float) -> dict[str, object]:
    transitions = 2 * limit + 1
    general_mask, general_windows, general_count, general_fraction = _temporal_localization(
        _vectorized_general_directions(frames, luminance_delta=delta), timestamps,
        failure_transition_count=transitions, area_threshold=area, kind="general_flash",
    )
    red_mask, red_windows, red_count, red_fraction = _temporal_localization(
        _vectorized_red_directions(frames), timestamps,
        failure_transition_count=transitions, area_threshold=area, kind="red_flash",
    )
    pattern_mask, pattern_windows, pattern_fraction = (
        _whole_frame_regular_pattern_reference(frames, timestamps)
    )
    kind_masks = {name: mask for name, mask in (("general_flash", general_mask), ("red_flash", red_mask), ("regular_pattern", pattern_mask)) if np.any(mask)}
    hazard_mask = np.zeros(frames.shape[:3], dtype=bool)
    for mask in kind_masks.values():
        hazard_mask |= mask
    return {
        "hazardous": bool([*general_windows, *red_windows, *pattern_windows]),
        "hazard_mask": hazard_mask,
        "kind_masks": kind_masks,
        "windows": tuple([*general_windows, *red_windows, *pattern_windows]),
        "max_flash_count": max(general_count, red_count) / 2.0,
        "max_affected_fraction": max(general_fraction, red_fraction, pattern_fraction),
    }


def _assert_matches_vectorized_reference(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    delta: float,
    limit: int,
    area: float,
) -> None:
    expected = _vectorized_reference(
        frames, timestamps, delta=delta, limit=limit, area=area
    )
    actual = analyze(
        frames,
        timestamps,
        luminance_delta=delta,
        max_flashes_per_second=limit,
        area_threshold=area,
    )
    assert actual.hazardous is expected["hazardous"]
    np.testing.assert_array_equal(actual.hazard_mask, expected["hazard_mask"])
    assert actual.windows == expected["windows"]
    assert actual.max_flash_count == expected["max_flash_count"]
    assert actual.max_affected_fraction == expected["max_affected_fraction"]
    assert set(actual.kind_masks) == set(expected["kind_masks"])
    for kind, mask in actual.kind_masks.items():
        np.testing.assert_array_equal(mask, expected["kind_masks"][kind])


def test_streaming_detector_is_exactly_equivalent_to_vectorized_reference() -> None:
    rng = np.random.default_rng(20260801)
    random_frames = rng.integers(0, 256, size=(9, 12, 13, 3), dtype=np.uint8)
    red_frames = np.zeros((9, 12, 13, 3), dtype=np.uint8)
    red_frames[1::2, :, :, 0] = 255
    threshold_frames = np.full((9, 12, 13, 3), 32, dtype=np.uint8)
    threshold_frames[1::2] = 180
    timestamps = np.arange(9, dtype=np.float64) / 60.0
    for frames, delta, limit, area in ((random_frames, 0.1, 3, 0.25), (red_frames, 0.1, 3, 0.25), (threshold_frames, 0.1, 0, 0.0)):
        expected_general = _vectorized_general_directions(frames, luminance_delta=delta)
        expected_red = _vectorized_red_directions(frames)
        actual_general = _luminance_transition_directions(frames, luminance_delta=delta)
        actual_red = _red_transition_directions(frames)
        assert actual_general.shape == (len(frames) - 1, *frames.shape[1:3])
        assert actual_red.shape == (len(frames) - 1, *frames.shape[1:3])
        np.testing.assert_array_equal(actual_general, expected_general)
        np.testing.assert_array_equal(actual_red, expected_red)
        _assert_matches_vectorized_reference(
            frames, timestamps, delta=delta, limit=limit, area=area
        )


def test_uint8_srgb_lookup_is_bit_exact_to_reference_formula() -> None:
    values = np.arange(256, dtype=np.uint8)
    encoded = np.asarray(values, dtype=np.float64) / 255.0
    expected = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )

    np.testing.assert_array_equal(srgb_to_linear(values), expected)


def test_bounded_window_runs_are_exact_for_arbitrary_zero_and_repeat_events() -> None:
    rng = np.random.default_rng(20260802)
    directions = rng.integers(-1, 2, size=(80, 17, 19), dtype=np.int8)
    left_indices = [max(0, right - (right % 23)) for right in range(80)]

    actual = dict(
        _window_longest_opposing_runs(directions, left_indices)
    )

    assert list(actual) == list(range(len(directions)))
    for right, left in enumerate(left_indices):
        np.testing.assert_array_equal(
            actual[right],
            _longest_opposing_run(directions[left : right + 1]),
        )


def test_grouped_window_runs_preserve_requested_right_order_and_exactness() -> None:
    directions = np.array(
        [
            [[1]], [[-1]], [[0]], [[1]], [[1]], [[-1]], [[0]], [[1]],
        ],
        dtype=np.int8,
    )
    left_indices = [0, 0, 0, 0, 0, 1, 2, 3]
    requested = [7, 1, 5, 3]

    actual = list(
        _window_longest_opposing_runs(
            directions,
            left_indices,
            rights=requested,
        )
    )

    assert [right for right, _ in actual] == requested
    for right, counts in actual:
        np.testing.assert_array_equal(
            counts,
            _longest_opposing_run(directions[left_indices[right] : right + 1]),
        )


def test_ninety_frame_large_smoke_completes_inside_bounded_runtime() -> None:
    frames = np.zeros((90, 384, 640, 3), dtype=np.uint8)
    timestamps = np.arange(len(frames), dtype=np.float64) / 60.0

    started = monotonic()
    result = analyze(frames, timestamps)

    assert monotonic() - started < 15.0
    assert result.hazardous is False
    assert result.windows == ()
    assert result.max_flash_count == 0.0
    assert result.max_affected_fraction == 0.0


def test_bounded_window_runs_do_not_overflow_on_long_trace_prefix() -> None:
    directions = np.ones((32_829, 1, 1), dtype=np.int8)
    directions[1::2] = -1
    right = len(directions) - 1
    left_indices = [0] * len(directions)
    left_indices[right] = right - 60

    actual = dict(
        _window_longest_opposing_runs(
            directions,
            left_indices,
            rights=[right],
        )
    )[right]
    expected = _longest_opposing_run(
        directions[left_indices[right] : right + 1]
    )

    np.testing.assert_array_equal(actual, expected)
    assert int(actual[0, 0]) == 61


@pytest.mark.parametrize("red_flash", [False, True])
def test_tiled_detector_matches_reference_at_spatial_area_and_time_boundaries(
    red_flash: bool,
) -> None:
    height, width = 131, 133
    affected = np.zeros((height, width), dtype=bool)
    affected[60:, 60:] = True
    frames = np.zeros((8, height, width, 3), dtype=np.uint8)
    if red_flash:
        frames[:, affected] = np.array([0, 64, 64], dtype=np.uint8)
        frames[1::2, affected] = np.array([255, 0, 0], dtype=np.uint8)
    else:
        frames[1::2, affected] = 255

    boundary_times = np.array([1.0, 4 / 3, 1.4, 1.5, 1.6, 1.7, 1.8, 7 / 3])
    outside_times = boundary_times.copy()
    outside_times[-1] += 0.000001
    exact_fraction = float(np.mean(affected))
    below_fraction = float(np.nextafter(exact_fraction, -np.inf))

    for timestamps in (boundary_times, outside_times):
        for area in (exact_fraction, below_fraction):
            _assert_matches_vectorized_reference(
                frames,
                timestamps,
                delta=0.1,
                limit=3,
                area=area,
            )


def test_tiled_localizer_bounds_direction_planes_and_uses_two_passes() -> None:
    frames = np.zeros((3, 257, 259, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0])
    observed_shapes: list[tuple[int, ...]] = []

    def alternating_directions(tile: np.ndarray) -> np.ndarray:
        observed_shapes.append(tile.shape)
        directions = np.ones((2, *tile.shape[1:3]), dtype=np.int8)
        directions[1] = -1
        return directions

    mask, windows, _, _ = _tiled_temporal_localization(
        frames,
        timestamps,
        direction_builder=alternating_directions,
        failure_transition_count=2,
        area_threshold=0.0,
        kind="general_flash",
    )

    tiles_per_pass = 3 * 3
    assert len(observed_shapes) == 2 * tiles_per_pass
    assert all(
        shape[1] <= _SPATIAL_TILE_EDGE and shape[2] <= _SPATIAL_TILE_EDGE
        for shape in observed_shapes
    )
    assert windows
    assert not np.any(mask[0])
    assert np.all(mask[1:])


def test_safe_tiled_localizer_returns_non_owning_zero_mask() -> None:
    frames = np.zeros((3, 257, 259, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.5, 1.0])
    observed_shapes: list[tuple[int, ...]] = []

    def no_directions(tile: np.ndarray) -> np.ndarray:
        observed_shapes.append(tile.shape)
        return np.zeros((2, *tile.shape[1:3]), dtype=np.int8)

    mask, windows, _, _ = _tiled_temporal_localization(
        frames,
        timestamps,
        direction_builder=no_directions,
        failure_transition_count=2,
        area_threshold=0.0,
        kind="general_flash",
    )

    assert len(observed_shapes) == 3 * 3
    assert not windows
    assert not np.any(mask)
    assert not mask.flags.owndata
    assert mask.strides == (0, 0, 0)


def _striped_pattern_clip(
    *,
    height: int,
    width: int,
    axis: int,
    start: int,
    stripe_count: int,
    dynamic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    frames = np.full((4, height, width, 3), 96, dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        phase = frame_index if dynamic else 0
        for stripe in range(stripe_count):
            value = 240 if (stripe + phase) % 2 else 8
            stripe_start = start + 2 * stripe
            if axis == 1:
                frame[:, stripe_start : stripe_start + 2] = value
            else:
                frame[stripe_start : stripe_start + 2, :] = value
    timestamps = np.arange(len(frames), dtype=np.float64) / 10.0
    return frames, timestamps


def test_optimized_analysis_matches_frozen_reference_for_each_hazard_family() -> None:
    general_frames = np.zeros((9, 16, 20, 3), dtype=np.uint8)
    general_frames[1::2] = 255

    red_frames = np.empty_like(general_frames)
    red_frames[::2] = np.array([0, 64, 64], dtype=np.uint8)
    red_frames[1::2] = np.array([255, 0, 0], dtype=np.uint8)

    regular_frames, regular_timestamps = _striped_pattern_clip(
        height=16,
        width=95,
        axis=1,
        start=0,
        stripe_count=12,
        dynamic=True,
    )
    flash_timestamps = np.arange(len(general_frames), dtype=np.float64) / 60.0
    cases = (
        (general_frames, flash_timestamps, "general_flash"),
        (red_frames, flash_timestamps, "red_flash"),
        (regular_frames, regular_timestamps, "regular_pattern"),
    )

    for frames, timestamps, expected_kind in cases:
        _assert_matches_vectorized_reference(
            frames,
            timestamps,
            delta=0.1,
            limit=3,
            area=0.25,
        )
        assert expected_kind in analyze(frames, timestamps).kind_masks


@pytest.mark.parametrize("axis", [1, 2])
def test_regular_pattern_strips_match_whole_frame_across_128_boundary(
    axis: int,
) -> None:
    frames, timestamps = _striped_pattern_clip(
        height=263,
        width=263,
        axis=axis,
        start=96,
        stripe_count=36,
        dynamic=True,
    )

    expected = _whole_frame_regular_pattern_reference(frames, timestamps)
    actual = _regular_pattern_localization(frames, timestamps)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert np.any(actual[0])


@pytest.mark.parametrize(
    ("width", "dynamic", "hazardous"),
    [
        (96, True, False),
        (95, True, True),
        (60, False, False),
        (59, False, True),
    ],
)
def test_regular_pattern_strips_preserve_strict_area_thresholds(
    width: int, dynamic: bool, hazardous: bool
) -> None:
    frames, timestamps = _striped_pattern_clip(
        height=16,
        width=width,
        axis=1,
        start=0,
        stripe_count=12,
        dynamic=dynamic,
    )

    expected = _whole_frame_regular_pattern_reference(frames, timestamps)
    actual = _regular_pattern_localization(frames, timestamps)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert bool(actual[1]) is hazardous


def test_safe_regular_pattern_bounds_luminance_strips_and_zero_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashpatch.core as core

    frames = np.full((3, 260, 263, 3), 96, dtype=np.uint8)
    timestamps = np.array([0.0, 0.1, 0.2])
    observed_shapes: list[tuple[int, ...]] = []
    original = core.relative_luminance

    def recording_luminance(values: np.ndarray) -> np.ndarray:
        observed_shapes.append(values.shape)
        return original(values)

    monkeypatch.setattr(core, "relative_luminance", recording_luminance)
    mask, windows, fraction = _regular_pattern_localization(frames, timestamps)

    # Three row strips for stripes, three column strips for stripes, and
    # three row strips for the orthogonal-lattice edge preflight per frame.
    assert len(observed_shapes) == 3 * (3 + 3 + 3)
    assert all(
        shape[0] == 1
        and (
            (shape[1] <= _SPATIAL_TILE_EDGE and shape[2] == frames.shape[2])
            or (
                shape[1] == frames.shape[1]
                and shape[2] <= _SPATIAL_TILE_EDGE
            )
        )
        for shape in observed_shapes
    )
    assert not windows
    assert fraction == 0.0
    assert not mask.flags.owndata
    assert mask.strides == (0, 0, 0)


def test_regular_pattern_prefilter_skips_signals_below_minimum_transition_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than eleven threshold crossings cannot form six stripe pairs."""
    import flashpatch.core as core

    frames = np.full((3, 16, 32, 3), 96, dtype=np.uint8)
    calls = 0
    original = core._alternating_stripe_span

    def counted(signal: np.ndarray) -> tuple[int, int] | None:
        nonlocal calls
        calls += 1
        return original(signal)

    monkeypatch.setattr(core, "_alternating_stripe_span", counted)
    mask, windows, fraction = _regular_pattern_localization(
        frames, np.arange(len(frames), dtype=np.float64) / 60.0
    )

    assert calls == 0
    assert fraction == 0.0
    assert not windows
    assert not np.any(mask)


def test_regular_pattern_prefilter_preserves_alternating_stripe_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashpatch.core as core

    frames, timestamps = _striped_pattern_clip(
        height=16,
        width=32,
        axis=1,
        start=0,
        stripe_count=12,
        dynamic=True,
    )
    calls = 0
    original = core._alternating_stripe_span

    def counted(signal: np.ndarray) -> tuple[int, int] | None:
        nonlocal calls
        calls += 1
        return original(signal)

    monkeypatch.setattr(core, "_alternating_stripe_span", counted)
    actual = _regular_pattern_localization(frames, timestamps)
    expected = _whole_frame_regular_pattern_reference(frames, timestamps)

    # Equal edge positions with globally reversed phase share one exact span
    # per frame, and packed masks avoid a second materialization pass.
    assert calls == len(frames)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_lattice_sequence_search_has_a_fixed_candidate_bound() -> None:
    import flashpatch.core as core

    centers = list(range(3, 4096, 3))

    sequences = core._regular_edge_sequences(centers, minimum=3)

    assert len(sequences) <= core._LATTICE_SEQUENCE_LIMIT


def make_clip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.full((60, 48, 64, 3), 96, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=bool)
    mask[8:40, 8:56] = True
    for index in range(len(frames)):
        frames[index, mask] = 255 if (index // 3) % 2 else 0
    timestamps = np.arange(len(frames), dtype=np.float64) / 30.0
    return frames, timestamps, mask


def alternating_clip(transition_count: int, *, affected_columns: int = 4) -> tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((transition_count + 1, 4, 4, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, :affected_columns] = 255 if index % 2 else 0
    timestamps = np.linspace(0.0, 1.0, len(frames))
    return frames, timestamps


def test_detect_repair_verify_closed_loop() -> None:
    frames, timestamps, expected_mask = make_clip()

    detection = analyze(frames, timestamps)
    assert detection.hazardous
    assert detection.max_flash_count > 3
    assert detection.hazard_mask.shape == frames.shape[:3]
    assert np.mean(np.any(detection.hazard_mask, axis=0)) > 0.25
    assert not verify(frames, timestamps).passed

    repaired = repair(frames, timestamps, detection)
    assert verify(repaired, timestamps).passed
    assert np.array_equal(repaired[:, ~expected_mask], frames[:, ~expected_mask])
    assert np.array_equal(repaired[~detection.hazard_mask], frames[~detection.hazard_mask])


def test_safe_clip_is_unchanged() -> None:
    frames = np.full((30, 16, 16, 3), 120, dtype=np.uint8)
    timestamps = np.arange(30, dtype=np.float64) / 30.0

    detection = analyze(frames, timestamps)
    assert not detection.hazardous
    assert verify(frames, timestamps).passed
    assert np.array_equal(repair(frames, timestamps, detection), frames)


@pytest.mark.parametrize(
    ("transition_count", "hazardous"),
    [(6, False), (7, True), (8, True)],
)
def test_transition_boundary_agrees_with_independent_verifier(
    transition_count: int, hazardous: bool
) -> None:
    frames, timestamps = alternating_clip(transition_count)
    detection = analyze(frames, timestamps)
    independent = verify(frames, timestamps)

    assert detection.hazardous is hazardous
    assert independent.passed is (not hazardous)


def test_closed_one_second_window_includes_both_endpoints() -> None:
    frames, timestamps = alternating_clip(7)
    assert timestamps[-1] - timestamps[0] == 1.0
    assert analyze(frames, timestamps).hazardous
    assert not verify(frames, timestamps).passed


def test_closed_one_second_window_survives_fractional_frame_rounding() -> None:
    frames = np.zeros((32, 4, 4, 3), dtype=np.uint8)
    value = 0
    for index in range(len(frames)):
        if index in (1, 6, 11, 16, 21, 26, 31):
            value = 255 - value
        frames[index] = value
    timestamps = np.arange(len(frames), dtype=np.float64) / 30.0

    assert timestamps[31] - timestamps[1] == 1.0
    assert analyze(frames, timestamps).hazardous
    assert not verify(frames, timestamps).passed


def test_closed_one_second_window_tolerates_subtraction_roundoff() -> None:
    timestamps = np.array([1.0, 4 / 3, 1.4, 1.5, 1.6, 1.7, 1.8, 7 / 3])
    frames = np.zeros((len(timestamps), 4, 4, 3), dtype=np.uint8)
    frames[1::2] = 255

    assert timestamps[-1] - timestamps[1] > 1.0
    assert analyze(frames, timestamps).hazardous
    assert not verify(frames, timestamps).passed


def test_transition_outside_one_second_rounding_tolerance_is_excluded() -> None:
    timestamps = np.array([1.0, 4 / 3, 1.4, 1.5, 1.6, 1.7, 1.8, 7 / 3 + 0.000001])
    frames = np.zeros((len(timestamps), 4, 4, 3), dtype=np.uint8)
    frames[1::2] = 255

    assert not analyze(frames, timestamps).hazardous
    assert verify(frames, timestamps).passed


def test_itu_luminance_threshold_vectors_preserve_boundary_conflict() -> None:
    assert not itu_luminance_change_is_flash(159.999, 179.998)
    assert itu_luminance_change_is_flash(159.999, 179.999)

    darker = 160.0
    exact = darker * (1.0 + 1.0 / 17.0) / (1.0 - 1.0 / 17.0)
    assert not itu_luminance_change_is_flash(darker, exact)
    assert itu_luminance_change_is_flash(darker, exact, michelson_inclusive=True)
    assert itu_luminance_change_is_flash(darker, exact + 0.001)


def test_area_threshold_requires_more_than_twenty_five_percent() -> None:
    exact_frames, timestamps = alternating_clip(7, affected_columns=1)
    above_frames, _ = alternating_clip(7, affected_columns=2)

    assert not analyze(exact_frames, timestamps).hazardous
    assert verify(exact_frames, timestamps).passed
    assert analyze(above_frames, timestamps).hazardous
    assert not verify(above_frames, timestamps).passed


def test_one_direction_ratchet_is_not_a_flash_sequence() -> None:
    linear_levels = [0.0]
    for index in range(7):
        linear_levels.extend([0.11 + 0.02 * index, 0.02 + 0.02 * index])
    linear_levels = linear_levels[:-1]
    linear = np.array(linear_levels)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    values = np.rint(srgb * 255.0).astype(np.uint8)
    frames = np.repeat(values[:, None, None, None], 4 * 4 * 3, axis=1).reshape(-1, 4, 4, 3)
    timestamps = np.linspace(0.0, 1.0, len(frames))

    assert not analyze(frames, timestamps).hazardous
    assert verify(frames, timestamps).passed


def test_rejects_unsupported_and_degenerate_inputs() -> None:
    uint8_frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    float_frames = uint8_frames.astype(np.float32)
    empty_space = np.zeros((3, 0, 2, 3), dtype=np.uint8)
    short_frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="uint8"):
        analyze(float_frames, np.array([0.0, 0.1, 0.2]))
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze(uint8_frames, np.array([0.0, 0.1, 0.1]))
    with pytest.raises(ValueError, match="positive"):
        analyze(empty_space, np.array([0.0, 0.1, 0.2]))
    with pytest.raises(ValueError, match="positive"):
        verify(empty_space, np.array([0.0, 0.1, 0.2]))
    with pytest.raises(ValueError, match="three frames"):
        verify(short_frames, np.array([0.0, 0.1]))
    for invalid in (float("nan"), float("inf"), 3.5, True):
        invalid_limit = cast(int, invalid)
        with pytest.raises(ValueError, match="non-negative integer"):
            analyze(uint8_frames, np.array([0.0, 0.1, 0.2]), max_flashes_per_second=invalid_limit)
        with pytest.raises(ValueError, match="non-negative integer"):
            verify(uint8_frames, np.array([0.0, 0.1, 0.2]), max_flashes_per_second=invalid_limit)
