from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .standards import regular_pattern_is_hazardous


_SPATIAL_TILE_EDGE = 128
_REGULAR_PATTERN_WORKERS = 4
_LATTICE_EDGE_SUPPORT = 0.25
_LATTICE_MIN_BOUNDARIES = 6
_LATTICE_SPACING_TOLERANCE = 0.15
_LATTICE_PHASE_AGREEMENT = 0.90
_LATTICE_SEQUENCE_LIMIT = 32
_LATTICE_SAMPLE_ROW_CHUNK = 16
_SRGB8_ENCODED = np.arange(256, dtype=np.float64) / 255.0
_SRGB8_LINEAR = np.where(
    _SRGB8_ENCODED <= 0.04045,
    _SRGB8_ENCODED / 12.92,
    ((_SRGB8_ENCODED + 0.055) / 1.055) ** 2.4,
)


@dataclass(frozen=True)
class HazardWindow:
    start: float
    end: float
    affected_fraction: float
    flash_count: float
    kind: str = "general_flash"


@dataclass(frozen=True)
class Analysis:
    hazardous: bool
    hazard_mask: np.ndarray
    windows: tuple[HazardWindow, ...]
    max_flash_count: float
    max_affected_fraction: float
    kind_masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class RepairOutcome:
    frames: np.ndarray
    feather_mask: np.ndarray
    fallback_used: bool
    verified: bool
    changed_fraction: float
    outside_hazard_unchanged: bool
    structural_similarity: float
    hazard_band_power_before: float
    hazard_band_power_after: float


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype == np.uint8:
        return _SRGB8_LINEAR[array]
    values = np.asarray(array, dtype=np.float64) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def relative_luminance(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape [time, height, width, 3]")
    linear = srgb_to_linear(frames)
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def _luminance_transition_directions(frame_array: np.ndarray, *, luminance_delta: float) -> np.ndarray:
    """Compute general-flash directions one frame pair at a time.

    Keeping a full float64 luminance tensor beside RGB input makes a real
    720p replay need multiple gigabytes before any safety decision is made.
    The detector only needs adjacent luminance planes, while the temporal
    localizer needs the compact int8 direction tensor.  This preserves the
    same per-pixel calculation without making frame count and resolution
    multiply peak float memory.
    """
    directions = np.zeros((len(frame_array) - 1, *frame_array.shape[1:3]), dtype=np.int8)
    previous = relative_luminance(frame_array[:1])[0]
    for index in range(1, len(frame_array)):
        current = relative_luminance(frame_array[index : index + 1])[0]
        delta = current - previous
        darker = np.minimum(previous, current)
        transitions = (np.abs(delta) >= luminance_delta) & (darker < 0.8)
        directions[index - 1] = np.where(transitions, np.sign(delta), 0).astype(np.int8)
        previous = current
    return directions


def itu_luminance_change_is_flash(
    first_cd_m2: float,
    second_cd_m2: float,
    *,
    michelson_inclusive: bool = False,
) -> bool:
    darker, lighter = sorted((float(first_cd_m2), float(second_cd_m2)))
    if darker < 160.0:
        return lighter - darker >= 20.0
    contrast = (lighter - darker) / (lighter + darker)
    if michelson_inclusive:
        return contrast >= 1.0 / 17.0
    return contrast > 1.0 / 17.0


def _longest_opposing_run(directions: np.ndarray) -> np.ndarray:
    previous = np.zeros(directions.shape[1:], dtype=np.int8)
    current_run = np.zeros(directions.shape[1:], dtype=np.int16)
    longest_run = np.zeros(directions.shape[1:], dtype=np.int16)
    for direction in directions:
        active = direction != 0
        first = active & (previous == 0)
        opposing = active & (previous == -direction)
        repeated = active & ~(first | opposing)
        current_run[first] = 1
        current_run[opposing] += 1
        current_run[repeated] = 1
        previous[active] = direction[active]
        longest_run = np.maximum(longest_run, current_run)
    return longest_run


def _window_longest_opposing_runs(
    directions: np.ndarray,
    left_indices: list[int],
    rights: Iterator[int] | range | list[int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield exact opposing-run maxima for selected closed windows.

    Non-zero directions form event streams independently at every pixel.
    Repeated signs start a new alternating segment and zero directions leave
    the current segment unchanged.  A wide prefix count says how many events
    fall inside ``[left, right]``; a bounded per-event segment length says how
    many belong to its current alternating segment.  Their minimum is the
    exact window-local run.  This removes the former repeated scan of up to
    one second for every right edge while retaining the same int16 saturation
    and a tile-bounded working set.
    """
    active = directions != 0
    count_dtype = (
        np.int32
        if len(directions) <= np.iinfo(np.int32).max
        else np.int64
    )
    event_counts = np.cumsum(active, axis=0, dtype=count_dtype)
    segment_lengths = np.zeros(directions.shape, dtype=np.uint16)
    previous = np.zeros(directions.shape[1:], dtype=np.int8)
    current_run = np.zeros(directions.shape[1:], dtype=np.uint16)

    for index, direction in enumerate(directions):
        current_active = active[index]
        first = current_active & (previous == 0)
        opposing = current_active & (previous == -direction)
        repeated = current_active & ~(first | opposing)
        current_run[first] = 1
        current_run[opposing] = np.minimum(
            current_run[opposing] + np.uint16(1),
            np.iinfo(np.int16).max,
        )
        current_run[repeated] = 1
        segment_lengths[index][current_active] = current_run[current_active]
        previous[current_active] = direction[current_active]

    # ``left_indices`` is monotonic because every right edge advances in
    # timestamp order.  Adjacent windows therefore commonly share their left
    # edge (the first second of a 60 fps capture is one large group).  The
    # previous implementation rebuilt and reduced every prefix in that group,
    # making the work quadratic in the number of frames per second.  For a
    # fixed left edge, the exact per-event values are fixed too; a cumulative
    # maximum yields every closed right-edge result in one streaming pass.
    #
    # This retains the original definition exactly: at each event it bounds
    # the alternating segment length by the number of events since the window
    # start, zeroes non-events, then takes the maximum in [left, right].
    zero_counts = np.zeros(directions.shape[1:], dtype=count_dtype)
    selected_rights = list(range(len(directions)) if rights is None else rights)
    if not selected_rights:
        return

    ordered_rights = sorted(selected_rights)
    results: dict[int, np.ndarray] = {}
    group_start = 0
    while group_start < len(ordered_rights):
        left = left_indices[ordered_rights[group_start]]
        group_stop = group_start + 1
        while (
            group_stop < len(ordered_rights)
            and left_indices[ordered_rights[group_stop]] == left
        ):
            group_stop += 1
        group_rights = ordered_rights[group_start:group_stop]
        last_right = group_rights[-1]
        before_window = zero_counts if left == 0 else event_counts[left - 1]
        window_counts = event_counts[left : last_right + 1] - before_window
        np.minimum(
            window_counts,
            segment_lengths[left : last_right + 1],
            out=window_counts,
        )
        window_counts[~active[left : last_right + 1]] = 0
        if len(group_rights) == 1:
            right = group_rights[0]
            results[right] = window_counts.max(axis=0, initial=0).astype(
                np.int16, copy=False
            )
        else:
            wanted = set(group_rights)
            running_max = np.zeros(directions.shape[1:], dtype=count_dtype)
            for offset, values in enumerate(window_counts):
                np.maximum(running_max, values, out=running_max)
                right = left + offset
                if right in wanted:
                    results[right] = running_max.astype(np.int16)
        group_start = group_stop

    for right in selected_rights:
        yield right, results[right]


def _validate(frames: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_array = np.asarray(frames)
    time_array = np.asarray(timestamps, dtype=np.float64)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError("frames must have shape [time, height, width, 3]")
    if frame_array.shape[1] == 0 or frame_array.shape[2] == 0:
        raise ValueError("frame height and width must be positive")
    if frame_array.dtype != np.uint8:
        raise ValueError("bootstrap profile accepts uint8 RGB frames only")
    if len(frame_array) < 3:
        raise ValueError("at least three frames are required")
    if time_array.shape != (len(frame_array),):
        raise ValueError("timestamps must contain one value per frame")
    if not np.all(np.isfinite(time_array)) or np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return frame_array, time_array


def _temporal_localization(
    directions: np.ndarray,
    time_array: np.ndarray,
    *,
    failure_transition_count: int,
    area_threshold: float,
    kind: str,
) -> tuple[np.ndarray, list[HazardWindow], int, float]:
    transition_times = time_array[1:]
    hazard_mask = np.zeros((len(time_array), *directions.shape[1:]), dtype=bool)
    windows: list[HazardWindow] = []
    max_transition_count = 0
    max_fraction = 0.0
    for right, end_time in enumerate(transition_times):
        window_times = transition_times[: right + 1]
        scale = np.maximum(np.maximum(np.abs(end_time), np.abs(window_times)), 1.0)
        tolerance = 4.0 * np.finfo(np.float64).eps * scale
        left = int(np.count_nonzero(end_time - window_times > 1.0 + tolerance))
        counts = _longest_opposing_run(directions[left : right + 1])
        local_max = int(counts.max(initial=0))
        violating = counts >= failure_transition_count
        fraction = float(np.mean(violating))
        max_transition_count = max(max_transition_count, local_max)
        max_fraction = max(max_fraction, fraction)
        if fraction > area_threshold:
            start_time = float(transition_times[left])
            frame_selection = (time_array >= start_time) & (time_array <= end_time)
            hazard_mask[frame_selection] |= violating
            windows.append(
                HazardWindow(
                    start=start_time,
                    end=float(end_time),
                    affected_fraction=fraction,
                    flash_count=local_max / 2.0,
                    kind=kind,
                )
            )
    return hazard_mask, windows, max_transition_count, max_fraction


def _spatial_tiles(height: int, width: int) -> Iterator[tuple[slice, slice]]:
    for row_start in range(0, height, _SPATIAL_TILE_EDGE):
        row_stop = min(row_start + _SPATIAL_TILE_EDGE, height)
        for column_start in range(0, width, _SPATIAL_TILE_EDGE):
            column_stop = min(column_start + _SPATIAL_TILE_EDGE, width)
            yield slice(row_start, row_stop), slice(column_start, column_stop)


def _zero_mask_view(shape: tuple[int, ...]) -> np.ndarray:
    """Return a full-shaped zero mask without allocating its logical extent."""
    return np.broadcast_to(np.array(False, dtype=bool), shape)


def _tiled_temporal_localization(
    frame_array: np.ndarray,
    time_array: np.ndarray,
    *,
    direction_builder: Callable[[np.ndarray], np.ndarray],
    failure_transition_count: int,
    area_threshold: float,
    kind: str,
) -> tuple[np.ndarray, list[HazardWindow], int, float]:
    """Localize temporal hazards without a full-frame direction tensor.

    The first pass accumulates the exact number of violating pixels and the
    exact maximum transition run for every closed one-second window.  Once
    those global decisions are fixed, the second pass revisits only hazardous
    windows and materializes their pixels in the public full-resolution mask.
    Each tile is independent because both general- and red-flash tests are
    temporal per-pixel operations; regular-pattern analysis remains on its
    separate whole-row/whole-column path.
    """
    transition_times = time_array[1:]
    left_indices: list[int] = []
    for right, end_time in enumerate(transition_times):
        window_times = transition_times[: right + 1]
        scale = np.maximum(np.maximum(np.abs(end_time), np.abs(window_times)), 1.0)
        tolerance = 4.0 * np.finfo(np.float64).eps * scale
        left_indices.append(
            int(np.count_nonzero(end_time - window_times > 1.0 + tolerance))
        )

    height, width = frame_array.shape[1:3]
    violating_pixels = np.zeros(len(transition_times), dtype=np.int64)
    local_maxima = np.zeros(len(transition_times), dtype=np.int64)
    for row_slice, column_slice in _spatial_tiles(height, width):
        directions = direction_builder(frame_array[:, row_slice, column_slice])
        for right, counts in _window_longest_opposing_runs(
            directions, left_indices
        ):
            local_maxima[right] = max(
                int(local_maxima[right]), int(counts.max(initial=0))
            )
            violating_pixels[right] += np.count_nonzero(
                counts >= failure_transition_count
            )

    total_pixels = height * width
    fractions = [float(count / total_pixels) for count in violating_pixels]
    max_transition_count = int(local_maxima.max(initial=0))
    max_fraction = max(fractions, default=0.0)
    hazardous_rights = [
        right for right, fraction in enumerate(fractions) if fraction > area_threshold
    ]
    windows = [
        HazardWindow(
            start=float(transition_times[left_indices[right]]),
            end=float(transition_times[right]),
            affected_fraction=fractions[right],
            flash_count=int(local_maxima[right]) / 2.0,
            kind=kind,
        )
        for right in hazardous_rights
    ]
    if not hazardous_rights:
        return (
            _zero_mask_view(frame_array.shape[:3]),
            windows,
            max_transition_count,
            max_fraction,
        )

    hazard_mask = np.zeros(frame_array.shape[:3], dtype=bool)
    for row_slice, column_slice in _spatial_tiles(height, width):
        directions = direction_builder(frame_array[:, row_slice, column_slice])
        for right, counts in _window_longest_opposing_runs(
            directions, left_indices, hazardous_rights
        ):
            violating = counts >= failure_transition_count
            start_time = transition_times[left_indices[right]]
            frame_selection = (time_array >= start_time) & (
                time_array <= transition_times[right]
            )
            hazard_mask[frame_selection, row_slice, column_slice] |= violating
    return hazard_mask, windows, max_transition_count, max_fraction


def _red_chromaticity(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the red ratio and CIE u/v planes for one RGB frame only."""
    if frame.dtype == np.uint8:
        encoded = _SRGB8_ENCODED[frame]
    else:
        encoded = frame.astype(np.float64) / 255.0
    totals = np.sum(encoded, axis=-1)
    red_ratio = np.divide(
        encoded[..., 0], totals, out=np.zeros_like(totals), where=totals > 0.0
    )
    if frame.dtype == np.uint8:
        linear = _SRGB8_LINEAR[frame]
    else:
        linear = np.where(
            encoded <= 0.04045,
            encoded / 12.92,
            ((encoded + 0.055) / 1.055) ** 2.4,
        )
    x = 0.4124 * linear[..., 0] + 0.3576 * linear[..., 1] + 0.1805 * linear[..., 2]
    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    z = 0.0193 * linear[..., 0] + 0.1192 * linear[..., 1] + 0.9505 * linear[..., 2]
    denominator = x + 15.0 * y + 3.0 * z
    u = np.divide(4.0 * x, denominator, out=np.zeros_like(x), where=denominator > 0.0)
    v = np.divide(9.0 * y, denominator, out=np.zeros_like(y), where=denominator > 0.0)
    return red_ratio, u, v


def _red_transition_directions(frame_array: np.ndarray) -> np.ndarray:
    """Compute red-flash directions without retaining whole-video float RGB."""
    directions = np.zeros((len(frame_array) - 1, *frame_array.shape[1:3]), dtype=np.int8)
    previous_ratio, previous_u, previous_v = _red_chromaticity(frame_array[0])
    for index in range(1, len(frame_array)):
        current_ratio, current_u, current_v = _red_chromaticity(frame_array[index])
        distance = np.hypot(current_u - previous_u, current_v - previous_v)
        saturated = np.maximum(previous_ratio, current_ratio) >= 0.8
        transitions = saturated & (distance > 0.2)
        direction = np.sign(current_ratio - previous_ratio).astype(np.int8)
        directions[index - 1] = np.where(transitions, direction, 0).astype(np.int8)
        previous_ratio, previous_u, previous_v = current_ratio, current_u, current_v
    return directions


def _alternating_stripe_span(signal: np.ndarray) -> tuple[int, int] | None:
    differences = np.diff(signal)
    positions = np.flatnonzero(np.abs(differences) >= 0.1)
    if len(positions) < 11:
        return None
    directions = np.sign(differences[positions])
    best_start = 0
    best_end = 0
    current_start = 0
    for index in range(1, len(positions)):
        if directions[index] != -directions[index - 1]:
            current_start = index
        if index - current_start > best_end - best_start:
            best_start, best_end = current_start, index
    alternating = positions[best_start : best_end + 1]
    if len(alternating) < 11:
        return None
    spacing = max(1, int(round(float(np.median(np.diff(alternating))))))
    trailing_width = len(signal) - int(alternating[-1]) - 1
    if len(alternating) > 11 and trailing_width > 2 * spacing:
        alternating = alternating[:-1]
    stripe_pairs = (len(alternating) + 1) // 2
    if stripe_pairs <= 5:
        return None
    start = max(0, int(alternating[0]) - spacing + 1)
    end = min(len(signal), int(alternating[-1]) + spacing + 1)
    return start, end


def _strip_slices(length: int) -> Iterator[slice]:
    for start in range(0, length, _SPATIAL_TILE_EDGE):
        yield slice(start, min(start + _SPATIAL_TILE_EDGE, length))


def _alternating_stripe_eligible_signals(plane: np.ndarray, *, axis: int) -> np.ndarray:
    """Return signals that can contain eleven alternating significant edges.

    This is only a necessary-condition filter.  It preserves the historical
    one-dimensional span calculation for every surviving signal, but avoids
    allocating positions arrays and running Python loops for signals which
    cannot possibly contain the six stripe pairs required by the standard.
    """
    deltas = np.diff(plane, axis=axis)
    if axis == 0:
        deltas = deltas.T
    active = np.abs(deltas) >= 0.1
    event_counts = np.count_nonzero(active, axis=1)
    eligible = np.zeros(len(active), dtype=bool)
    if not np.any(event_counts >= 11):
        return eligible

    directions = np.sign(deltas).astype(np.int8, copy=False)
    positions = np.broadcast_to(np.arange(active.shape[1]), active.shape)
    latest_event = np.maximum.accumulate(np.where(active, positions, -1), axis=1)
    previous_event = np.full_like(latest_event, -1)
    previous_event[:, 1:] = latest_event[:, :-1]
    previous_direction = np.take_along_axis(
        directions, np.maximum(previous_event, 0), axis=1
    )
    continues = active & (previous_event >= 0) & (directions == -previous_direction)

    # Compact each ragged sequence of significant edges to its own contiguous
    # axis.  A sequence of eleven edges has ten adjacent "continues" values.
    event_rank = np.cumsum(active, axis=1, dtype=np.int32) - 1
    widest = int(event_counts.max(initial=0))
    compact = np.zeros((len(active), widest), dtype=bool)
    rows, columns = np.nonzero(active)
    compact[rows, event_rank[rows, columns]] = continues[rows, columns]
    if widest < 11:
        return eligible
    consecutive = np.lib.stride_tricks.sliding_window_view(compact, 10, axis=1)
    return np.any(np.all(consecutive, axis=-1), axis=1)


def _uniform_stripe_span(
    plane: np.ndarray, *, axis: int, eligible: np.ndarray
) -> tuple[int, int] | None:
    """Reuse one span when every signal has the same edges and phase up to sign."""
    if not len(eligible) or not np.all(eligible):
        return None
    differences = np.diff(plane, axis=axis)
    if axis == 0:
        differences = differences.T
    active = np.abs(differences) >= 0.1
    if not np.all(active == active[0]):
        return None
    directions = np.sign(differences)
    reference = directions[0]
    same_phase = np.all(directions == reference, axis=1)
    opposite_phase = np.all(directions == -reference, axis=1)
    if not np.all(same_phase | opposite_phase):
        return None
    return _alternating_stripe_span(plane[0] if axis == 1 else plane[:, 0])


def _supported_edge_centers(support: np.ndarray) -> list[int]:
    """Collapse contiguous, well-supported edge pixels to boundary centers."""
    positions = np.flatnonzero(support >= _LATTICE_EDGE_SUPPORT)
    if not len(positions):
        return []
    centers: list[int] = []
    for group in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
        center = int(round(float(np.average(group, weights=support[group])))) + 1
        centers.append(center)
    return centers


def _regular_edge_sequences(
    centers: list[int], *, minimum: int
) -> tuple[tuple[int, ...], ...]:
    """Find a bounded set of equal-pitch edge runs in near-linear time."""
    if len(centers) < minimum:
        return ()
    values = np.asarray(centers, dtype=np.int64)
    pitch_differences = np.concatenate(
        [
            values[stride:] - values[:-stride]
            for stride in range(1, min(8, len(values) - 1) + 1)
        ]
    )
    possible = np.unique(pitch_differences[pitch_differences >= 3])
    if not len(possible):
        return ()
    scored_pitches = sorted(
        (
            (
                int(
                    np.count_nonzero(
                        np.abs(pitch_differences - pitch)
                        <= max(2, pitch * _LATTICE_SPACING_TOLERANCE)
                    )
                ),
                int(pitch),
            )
            for pitch in possible
        ),
        reverse=True,
    )
    pitches: list[int] = []
    for _, pitch in scored_pitches:
        if any(
            abs(pitch - selected)
            <= max(2, selected * _LATTICE_SPACING_TOLERANCE)
            for selected in pitches
        ):
            continue
        pitches.append(pitch)
        if len(pitches) == 8:
            break

    def match(target: float, tolerance: float) -> int | None:
        insertion = int(np.searchsorted(values, target))
        options = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(values) and abs(values[index] - target) <= tolerance
        ]
        return min(options, key=lambda index: abs(values[index] - target)) if options else None

    sequences: set[tuple[int, ...]] = set()
    for pitch in pitches:
        tolerance = max(2.0, pitch * _LATTICE_SPACING_TOLERANCE)
        for root, center in enumerate(values):
            predecessor = match(float(center - pitch), tolerance)
            if predecessor is not None and predecessor < root:
                continue
            indices = [root]
            while True:
                following = match(float(values[indices[-1]] + pitch), tolerance)
                if following is None or following <= indices[-1]:
                    break
                indices.append(following)
            if len(indices) < minimum:
                continue
            sequence = tuple(int(values[index]) for index in indices)
            sequences.add(sequence)
            if len(sequence) > minimum:
                for start in range(len(sequence) - minimum + 1):
                    sequences.add(sequence[start : start + minimum])
    ranked = sorted(
        sequences,
        key=lambda sequence: (sequence[-1] - sequence[0], len(sequence)),
        reverse=True,
    )
    return tuple(ranked[:_LATTICE_SEQUENCE_LIMIT])


def _boundary_variants(sequence: tuple[int, ...], limit: int) -> tuple[tuple[int, ...], ...]:
    spacing = float(np.median(np.diff(sequence)))
    prepend = max(0, int(round(sequence[0] - spacing)))
    append = min(limit, int(round(sequence[-1] + spacing)))
    variants = {sequence}
    if sequence[0] - prepend >= 0.6 * spacing:
        variants.add((prepend, *sequence))
    if append - sequence[-1] >= 0.6 * spacing:
        variants.add((*sequence, append))
    if (
        sequence[0] - prepend >= 0.6 * spacing
        and append - sequence[-1] >= 0.6 * spacing
    ):
        variants.add((prepend, *sequence, append))
    return tuple(sorted(variants))


def _sample_cell_luminance(
    frame: np.ndarray, y_edges: tuple[int, ...], x_edges: tuple[int, ...]
) -> np.ndarray:
    fractions = np.asarray((0.2, 0.4, 0.6, 0.8))
    y_top = np.asarray(y_edges[:-1])[:, None]
    y_bottom = np.asarray(y_edges[1:])[:, None]
    x_left = np.asarray(x_edges[:-1])[:, None]
    x_right = np.asarray(x_edges[1:])[:, None]
    y = np.minimum(y_bottom - 1, y_top + ((y_bottom - y_top) * fractions).astype(int))
    x = np.minimum(x_right - 1, x_left + ((x_right - x_left) * fractions).astype(int))
    means = np.empty((len(y), len(x)), dtype=np.float64)
    for start in range(0, len(y), _LATTICE_SAMPLE_ROW_CHUNK):
        stop = min(start + _LATTICE_SAMPLE_ROW_CHUNK, len(y))
        samples = frame[y[start:stop, :, None, None], x[None, None, :, :]]
        linear = _SRGB8_LINEAR[samples]
        luminance = (
            0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2]
        )
        means[start:stop] = np.mean(luminance, axis=(1, 3))
    return means


def _lattice_block_is_coherent(means: np.ndarray) -> bool:
    low, high = np.quantile(means, (0.25, 0.75))
    if high - low < 0.1 or means.shape[0] < 3 or means.shape[1] < 3:
        return False
    horizontal_directions = np.sign(np.diff(means, axis=1))
    vertical_directions = np.sign(np.diff(means, axis=0))
    horizontal_reversals = (
        horizontal_directions[:, 1:] == -horizontal_directions[:, :-1]
    )
    vertical_reversals = vertical_directions[1:] == -vertical_directions[:-1]
    horizontal_contrast = np.abs(np.diff(means, axis=1)) >= 0.1
    vertical_contrast = np.abs(np.diff(means, axis=0)) >= 0.1
    return bool(
        np.all(np.mean(horizontal_reversals, axis=0) >= _LATTICE_PHASE_AGREEMENT)
        and np.all(np.mean(vertical_reversals, axis=1) >= _LATTICE_PHASE_AGREEMENT)
        and np.all(np.mean(horizontal_contrast, axis=0) >= _LATTICE_PHASE_AGREEMENT)
        and np.all(np.mean(vertical_contrast, axis=1) >= _LATTICE_PHASE_AGREEMENT)
    )


def _covered_expansion_is_coherent(
    frame: np.ndarray,
    x_sequence: tuple[int, ...],
    y_sequence: tuple[int, ...],
    x_edges: tuple[int, ...],
    y_edges: tuple[int, ...],
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], bool],
) -> bool:
    bands: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    if x_edges[0] != x_sequence[0]:
        bands.append((y_edges, x_edges[:4]))
    if x_edges[-1] != x_sequence[-1]:
        bands.append((y_edges, x_edges[-4:]))
    if y_edges[0] != y_sequence[0]:
        bands.append((y_edges[:4], x_edges))
    if y_edges[-1] != y_sequence[-1]:
        bands.append((y_edges[-4:], x_edges))
    if not bands:
        return False
    for band_y, band_x in bands:
        key = (band_y, band_x)
        coherent = cache.get(key)
        if coherent is None:
            coherent = _lattice_block_is_coherent(
                _sample_cell_luminance(frame, band_y, band_x)
            )
            cache[key] = coherent
        if not coherent:
            return False
    return True


def _coherent_lattice_mask(
    frame: np.ndarray,
    x_sequences: tuple[tuple[int, ...], ...],
    y_sequences: tuple[tuple[int, ...], ...],
    *,
    expand_outer: bool,
    tracking_margin: bool = False,
    covered: np.ndarray | None = None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    candidates: list[
        tuple[
            int,
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            bool,
        ]
    ] = []
    for x_sequence in x_sequences:
        for y_sequence in y_sequences:
            base_covered = False
            if covered is not None:
                base_covered = np.all(
                    covered[
                        y_sequence[0] : y_sequence[-1],
                        x_sequence[0] : x_sequence[-1],
                    ]
                )
                if base_covered and not expand_outer:
                    continue
            x_variants = _boundary_variants(x_sequence, width) if expand_outer else (x_sequence,)
            y_variants = _boundary_variants(y_sequence, height) if expand_outer else (y_sequence,)
            for x_edges in x_variants:
                for y_edges in y_variants:
                    left, right = x_edges[0], x_edges[-1]
                    top, bottom = y_edges[0], y_edges[-1]
                    if tracking_margin:
                        x_margin = int(round(float(np.median(np.diff(x_edges))) / 2.0))
                        y_margin = int(round(float(np.median(np.diff(y_edges))) / 2.0))
                        left, right = max(0, left - x_margin), min(width, right + x_margin)
                        top, bottom = max(0, top - y_margin), min(height, bottom + y_margin)
                    area = (right - left) * (bottom - top)
                    candidates.append(
                        (
                            area,
                            x_edges,
                            y_edges,
                            x_sequence,
                            y_sequence,
                            bool(base_covered),
                        )
                    )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    result = (
        np.zeros((height, width), dtype=bool)
        if covered is None
        else np.array(covered, dtype=bool, copy=True)
    )
    coherence_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] = {}
    for _, x_edges, y_edges, x_sequence, y_sequence, base_covered in candidates:
        left, right = x_edges[0], x_edges[-1]
        top, bottom = y_edges[0], y_edges[-1]
        if tracking_margin:
            x_margin = int(round(float(np.median(np.diff(x_edges))) / 2.0))
            y_margin = int(round(float(np.median(np.diff(y_edges))) / 2.0))
            left, right = max(0, left - x_margin), min(width, right + x_margin)
            top, bottom = max(0, top - y_margin), min(height, bottom + y_margin)
        if np.all(result[top:bottom, left:right]):
            continue
        if base_covered:
            coherent = _covered_expansion_is_coherent(
                frame,
                x_sequence,
                y_sequence,
                x_edges,
                y_edges,
                coherence_cache,
            )
        else:
            key = (y_edges, x_edges)
            coherent = coherence_cache.get(key)
            if coherent is None:
                coherent = _lattice_block_is_coherent(
                    _sample_cell_luminance(frame, y_edges, x_edges)
                )
                coherence_cache[key] = coherent
        if not coherent:
            continue
        result[top:bottom, left:right] = True
    return result


@dataclass(frozen=True)
class _LatticeCandidate:
    mask: np.ndarray
    tracking_mask: np.ndarray


def _regular_lattice_candidate(
    frame: np.ndarray, *, covered: np.ndarray | None = None
) -> _LatticeCandidate:
    """Localize a coherent two-dimensional alternating-cell lattice.

    The stripe lane intentionally requires six one-dimensional light/dark
    pairs.  A checker or tiled lattice distributes those alternations across
    both axes, so no single row supplies eleven significant edges.  This lane
    separately requires repeated, equal-pitch boundaries on both axes and a
    high-agreement alternating phase across the enclosed cells.
    """
    height, width = frame.shape[:2]
    vertical_counts = np.zeros(max(0, width - 1), dtype=np.int64)
    horizontal_counts = np.zeros(max(0, height - 1), dtype=np.int64)
    previous_row: np.ndarray | None = None
    for row_slice in _strip_slices(height):
        plane = relative_luminance(frame[None, row_slice, :, :])[0]
        vertical_counts += np.count_nonzero(np.abs(np.diff(plane, axis=1)) >= 0.1, axis=0)
        if previous_row is not None:
            horizontal_counts[row_slice.start - 1] = np.count_nonzero(
                np.abs(plane[0] - previous_row) >= 0.1
            )
        if len(plane) > 1:
            horizontal_counts[row_slice.start : row_slice.stop - 1] = np.count_nonzero(
                np.abs(np.diff(plane, axis=0)) >= 0.1, axis=1
            )
        previous_row = plane[-1]

    x_centers = _supported_edge_centers(vertical_counts / max(1, height))
    y_centers = _supported_edge_centers(horizontal_counts / max(1, width))
    tracking_x = _regular_edge_sequences(x_centers, minimum=4)
    tracking_y = _regular_edge_sequences(y_centers, minimum=4)
    tracking = _coherent_lattice_mask(
        frame,
        tracking_x,
        tracking_y,
        expand_outer=False,
        tracking_margin=True,
        covered=covered,
    )
    eligible_x = _regular_edge_sequences(
        x_centers, minimum=_LATTICE_MIN_BOUNDARIES
    )
    eligible_y = _regular_edge_sequences(
        y_centers, minimum=_LATTICE_MIN_BOUNDARIES
    )
    eligible = _coherent_lattice_mask(
        frame,
        eligible_x,
        eligible_y,
        expand_outer=True,
        covered=covered,
    )
    return _LatticeCandidate(mask=eligible, tracking_mask=tracking)


@dataclass(frozen=True)
class _RegularPatternCandidate:
    mask: np.ndarray
    tracking_mask: np.ndarray


def _regular_pattern_candidate(frame: np.ndarray) -> _RegularPatternCandidate:
    """Find one frame's pattern pixels while bounding float luminance planes.

    A row is always inspected across the complete frame width, and a column is
    always inspected across the complete frame height.  Only the number of
    rows or columns converted to float luminance at once is bounded, so a
    stripe crossing a 128-pixel strip boundary has exactly the same semantics
    as the former whole-frame implementation.
    """
    height, width = frame.shape[:2]
    candidate = np.zeros((height, width), dtype=bool)
    for row_slice in _strip_slices(height):
        plane = relative_luminance(frame[None, row_slice, :, :])[0]
        row_eligible = _alternating_stripe_eligible_signals(plane, axis=1)
        uniform_span = _uniform_stripe_span(plane, axis=1, eligible=row_eligible)
        if uniform_span is not None:
            candidate[
                row_slice, uniform_span[0] : uniform_span[1]
            ] = True
        else:
            for row_index in np.flatnonzero(row_eligible):
                row = plane[row_index]
                span = _alternating_stripe_span(row)
                if span is not None:
                    candidate[
                        row_slice.start + row_index, span[0] : span[1]
                    ] = True

    if np.all(candidate):
        return _RegularPatternCandidate(mask=candidate, tracking_mask=candidate.copy())

    for column_slice in _strip_slices(width):
        plane = relative_luminance(frame[None, :, column_slice, :])[0]
        column_eligible = _alternating_stripe_eligible_signals(plane, axis=0)
        uniform_span = _uniform_stripe_span(
            plane, axis=0, eligible=column_eligible
        )
        if uniform_span is not None:
            candidate[
                uniform_span[0] : uniform_span[1], column_slice
            ] = True
        else:
            for column_index in np.flatnonzero(column_eligible):
                column = plane[:, column_index]
                span = _alternating_stripe_span(column)
                if span is not None:
                    candidate[
                        span[0] : span[1], column_slice.start + column_index
                    ] = True
    lattice = _regular_lattice_candidate(frame, covered=candidate)
    candidate |= lattice.mask
    return _RegularPatternCandidate(
        mask=candidate,
        tracking_mask=candidate | lattice.tracking_mask,
    )


def _regular_pattern_candidates(
    frame_array: np.ndarray,
    indices: Iterator[int] | range,
) -> Iterator[tuple[int, _RegularPatternCandidate]]:
    """Yield candidates in trace order with a bounded parallel work queue."""
    ordered_indices = iter(indices)
    worker_count = min(_REGULAR_PATTERN_WORKERS, len(frame_array))
    if worker_count < 2:
        for index in ordered_indices:
            yield index, _regular_pattern_candidate(frame_array[index])
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: dict[int, Future[_RegularPatternCandidate]] = {}
        for _ in range(worker_count):
            try:
                index = next(ordered_indices)
            except StopIteration:
                break
            pending[index] = executor.submit(_regular_pattern_candidate, frame_array[index])
        while pending:
            index = next(iter(pending))
            candidate = pending.pop(index).result()
            try:
                next_index = next(ordered_indices)
            except StopIteration:
                pass
            else:
                pending[next_index] = executor.submit(
                    _regular_pattern_candidate, frame_array[next_index]
                )
            yield index, candidate


def _pattern_contrast_is_reversed(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    active: np.ndarray,
) -> bool:
    """Return whether the same pattern region reverses its light/dark phase."""
    count = int(np.count_nonzero(active))
    if count < 2:
        return False
    changed = False
    for row_slice in _strip_slices(len(active)):
        local_active = active[row_slice]
        if np.any(local_active) and np.any(
            previous_frame[row_slice][local_active]
            != current_frame[row_slice][local_active]
        ):
            changed = True
            break
    if not changed:
        return False
    sum_previous = 0.0
    sum_current = 0.0
    sum_previous_squared = 0.0
    sum_current_squared = 0.0
    sum_product = 0.0
    for row_slice in _strip_slices(len(active)):
        local_active = active[row_slice]
        if not np.any(local_active):
            continue
        previous = relative_luminance(previous_frame[None, row_slice])[0][
            local_active
        ]
        current = relative_luminance(current_frame[None, row_slice])[0][
            local_active
        ]
        sum_previous += float(np.sum(previous))
        sum_current += float(np.sum(current))
        sum_previous_squared += float(np.dot(previous, previous))
        sum_current_squared += float(np.dot(current, current))
        sum_product += float(np.dot(previous, current))
    covariance = sum_product - (sum_previous * sum_current / count)
    previous_variance = sum_previous_squared - sum_previous**2 / count
    current_variance = sum_current_squared - sum_current**2 / count
    if previous_variance <= 0.0 or current_variance <= 0.0:
        return False
    correlation = covariance / np.sqrt(previous_variance * current_variance)
    return bool(correlation <= -_LATTICE_PHASE_AGREEMENT)


def _matching_pattern_translation(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_tracking: np.ndarray,
    current_tracking: np.ndarray,
) -> tuple[float, float] | None:
    """Return a verified mask-and-pixel translation between adjacent frames."""
    previous_rows = np.flatnonzero(np.any(previous_tracking, axis=1))
    previous_columns = np.flatnonzero(np.any(previous_tracking, axis=0))
    current_rows = np.flatnonzero(np.any(current_tracking, axis=1))
    current_columns = np.flatnonzero(np.any(current_tracking, axis=0))
    if not (
        len(previous_rows)
        and len(previous_columns)
        and len(current_rows)
        and len(current_columns)
    ):
        return None
    previous_center = np.array(
        (
            (previous_rows[0] + previous_rows[-1]) / 2.0,
            (previous_columns[0] + previous_columns[-1]) / 2.0,
        )
    )
    current_center = np.array(
        (
            (current_rows[0] + current_rows[-1]) / 2.0,
            (current_columns[0] + current_columns[-1]) / 2.0,
        )
    )
    displacement = current_center - previous_center
    rounded = np.rint(displacement).astype(int)
    if (
        not np.any(rounded)
        or np.max(np.abs(displacement - rounded)) > 0.25
    ):
        return None

    height, width = previous_tracking.shape
    dy, dx = int(rounded[0]), int(rounded[1])
    source_top, source_bottom = max(0, -dy), min(height, height - dy)
    source_left, source_right = max(0, -dx), min(width, width - dx)
    target_top, target_bottom = source_top + dy, source_bottom + dy
    target_left, target_right = source_left + dx, source_right + dx
    if source_top >= source_bottom or source_left >= source_right:
        return None

    shifted_previous = previous_tracking[
        source_top:source_bottom, source_left:source_right
    ]
    target_tracking = current_tracking[
        target_top:target_bottom, target_left:target_right
    ]
    intersection = shifted_previous & target_tracking
    union = shifted_previous | target_tracking
    if (
        not np.any(intersection)
        or np.count_nonzero(intersection) / np.count_nonzero(union) < 0.90
    ):
        return None
    previous_pixels = previous_frame[
        source_top:source_bottom, source_left:source_right
    ][intersection]
    current_pixels = current_frame[
        target_top:target_bottom, target_left:target_right
    ][intersection]
    if float(np.mean(np.all(previous_pixels == current_pixels, axis=-1))) < 0.90:
        return None
    return float(displacement[0]), float(displacement[1])


def _smooth_translation_frames(
    translations: list[tuple[float, float] | None],
) -> list[bool]:
    """Mark frames belonging to an obvious, sustained one-direction flow."""
    smooth = [False] * len(translations)
    start = 1
    while start < len(translations):
        first = translations[start]
        if first is None:
            start += 1
            continue
        stop = start + 1
        previous = np.asarray(first, dtype=np.float64)
        while stop < len(translations):
            current_value = translations[stop]
            if current_value is None:
                break
            current = np.asarray(current_value, dtype=np.float64)
            cosine = float(
                np.dot(previous, current)
                / (np.linalg.norm(previous) * np.linalg.norm(current))
            )
            if cosine < _LATTICE_PHASE_AGREEMENT:
                break
            previous = current
            stop += 1
        if stop - start >= 3:
            for frame_index in range(start - 1, stop):
                smooth[frame_index] = True
        start = stop
    return smooth


def _regular_pattern_localization(
    frame_array: np.ndarray, time_array: np.ndarray
) -> tuple[np.ndarray, list[HazardWindow], float]:
    """Localize regular patterns with frame-local motion and a selective pass two."""
    fractions: list[float] = []
    dynamic_frames: list[bool] = []
    spatial_updates: list[float] = []
    fraction_deltas: list[float] = []
    packed_candidates: dict[int, np.ndarray] = {}
    previous_tracking: np.ndarray | None = None
    previous_frame: np.ndarray | None = None
    tracking_fractions: list[float] = []
    translations: list[tuple[float, float] | None] = []
    contrast_reversals: list[bool] = []
    for index, candidate_result in _regular_pattern_candidates(
        frame_array, range(len(frame_array))
    ):
        frame = frame_array[index]
        candidate = candidate_result.mask
        tracking = candidate_result.tracking_mask
        fraction = float(np.mean(candidate))
        tracking_fraction = float(np.mean(tracking))
        fractions.append(fraction)
        if fraction > 0.25:
            packed_candidates[index] = np.packbits(candidate.reshape(-1))
        tracking_fractions.append(tracking_fraction)
        dynamic = False
        spatial_update = 0.0
        fraction_delta = 0.0
        contrast_reversed = False
        if previous_tracking is not None and previous_frame is not None:
            spatial_update = float(np.mean(previous_tracking ^ tracking))
            fraction_delta = tracking_fraction - tracking_fractions[-2]
            contrast_reversed = _pattern_contrast_is_reversed(
                previous_frame,
                frame,
                previous_tracking & tracking,
            )
            dynamic = spatial_update > 0.0 or contrast_reversed
            translations.append(
                _matching_pattern_translation(
                    previous_frame,
                    frame,
                    previous_tracking,
                    tracking,
                )
            )
        else:
            translations.append(None)
        dynamic_frames.append(dynamic)
        contrast_reversals.append(contrast_reversed)
        spatial_updates.append(spatial_update)
        fraction_deltas.append(fraction_delta)
        previous_tracking = tracking
        previous_frame = frame

    max_fraction = max(fractions, default=0.0)
    smooth_frames = [False] * len(fractions)
    for index in range(1, len(fractions)):
        prior_flow = index > 1 and smooth_frames[index - 1]
        if spatial_updates[index] == 0.0:
            smooth_frames[index] = prior_flow and dynamic_frames[index]
            continue
        expanding = fraction_deltas[index] > 0.0
        small_boundary_adjustment = spatial_updates[index] <= 0.02
        following_expansion = (
            index + 1 < len(fractions)
            and spatial_updates[index + 1] > 0.0
            and fraction_deltas[index + 1] > 0.0
        )
        smooth_frames[index] = expanding or (
            small_boundary_adjustment and (prior_flow or following_expansion)
        )
    # A single expansion is not a directional flow.  Require three adjacent
    # spatial expansions so alternating full-frame patterns cannot inherit the
    # smooth-flow exemption from their every-other-frame appearance.
    has_smooth_one_direction_flow = any(
        all(
            spatial_updates[flow_index] > 0.0
            and fraction_deltas[flow_index] > 0.0
            for flow_index in range(start, start + 3)
        )
        for start in range(1, max(1, len(fractions) - 2))
    )
    translation_frames = _smooth_translation_frames(translations)
    smooth_frames = [
        expansion or translation
        for expansion, translation in zip(
            smooth_frames, translation_frames, strict=True
        )
    ]
    has_smooth_one_direction_flow |= any(translation_frames)
    # Spatial motion applies to its arrival frame.  A verified contrast
    # reversal has two pattern endpoints, so include the preceding frame too.
    # This does not let an unrelated later UI update lower the static area
    # threshold for an earlier frame.
    dynamic_pattern_frames = [
        dynamic_frames[index]
        or (
            index + 1 < len(contrast_reversals)
            and contrast_reversals[index + 1]
        )
        for index in range(len(dynamic_frames))
    ]
    qualifying_frames = [
        index
        for index, fraction in enumerate(fractions)
        if regular_pattern_is_hazardous(
            stripe_pairs=6,
            affected_fraction=fraction,
            dynamic=dynamic_pattern_frames[index],
            smooth_one_direction=(
                smooth_frames[index] if has_smooth_one_direction_flow else False
            ),
        )
    ]
    if not qualifying_frames:
        return _zero_mask_view(frame_array.shape[:3]), [], max_fraction

    mask = np.zeros(frame_array.shape[:3], dtype=bool)
    active_frames: list[int] = []
    pixels_per_frame = frame_array.shape[1] * frame_array.shape[2]
    for index in qualifying_frames:
        packed = packed_candidates.get(index)
        if packed is not None:
            mask[index] = np.unpackbits(packed, count=pixels_per_frame).reshape(
                frame_array.shape[1:3]
            )
            active_frames.append(index)

    windows: list[HazardWindow] = []
    if active_frames:
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


def analyze(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    luminance_delta: float = 0.1,
    max_flashes_per_second: int = 3,
    area_threshold: float = 0.25,
) -> Analysis:
    frame_array, time_array = _validate(frames, timestamps)
    if not 0.0 < luminance_delta <= 1.0:
        raise ValueError("luminance_delta must be in (0, 1]")
    if (
        isinstance(max_flashes_per_second, (bool, np.bool_))
        or not isinstance(max_flashes_per_second, (int, np.integer))
        or max_flashes_per_second < 0
    ):
        raise ValueError("max_flashes_per_second must be a non-negative integer")
    if not 0.0 <= area_threshold <= 1.0:
        raise ValueError("area_threshold must be in [0, 1]")

    failure_transition_count = 2 * max_flashes_per_second + 1
    general_mask, general_windows, general_max_count, general_max_fraction = _tiled_temporal_localization(
        frame_array,
        time_array,
        direction_builder=lambda tile: _luminance_transition_directions(
            tile, luminance_delta=luminance_delta
        ),
        failure_transition_count=failure_transition_count,
        area_threshold=area_threshold,
        kind="general_flash",
    )
    red_mask, red_windows, red_max_count, red_max_fraction = _tiled_temporal_localization(
        frame_array,
        time_array,
        direction_builder=_red_transition_directions,
        failure_transition_count=failure_transition_count,
        area_threshold=area_threshold,
        kind="red_flash",
    )
    pattern_mask, pattern_windows, pattern_max_fraction = _regular_pattern_localization(
        frame_array, time_array
    )
    kind_masks = {
        kind: mask
        for kind, mask in (
            ("general_flash", general_mask),
            ("red_flash", red_mask),
            ("regular_pattern", pattern_mask),
        )
        if np.any(mask)
    }
    hazard_mask = np.zeros(frame_array.shape[:3], dtype=bool)
    for mask in kind_masks.values():
        hazard_mask |= mask
    windows = [*general_windows, *red_windows, *pattern_windows]
    max_transition_count = max(general_max_count, red_max_count)
    max_fraction = max(general_max_fraction, red_max_fraction, pattern_max_fraction)

    return Analysis(
        hazardous=bool(windows),
        hazard_mask=hazard_mask,
        windows=tuple(windows),
        max_flash_count=max_transition_count / 2.0,
        max_affected_fraction=max_fraction,
        kind_masks=kind_masks,
    )


def _feather_mask(mask: np.ndarray, *, passes: int = 2) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    soft = hard.astype(np.float64)
    for _ in range(passes):
        padded = np.pad(soft, ((0, 0), (1, 1), (1, 1)), mode="edge")
        blurred = np.zeros_like(soft)
        for row_offset in range(3):
            for column_offset in range(3):
                blurred += padded[
                    :, row_offset : row_offset + soft.shape[1], column_offset : column_offset + soft.shape[2]
                ]
        soft = blurred / 9.0
    return np.where(hard, soft, 0.0)


def _desaturate(candidate: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> None:
    if not np.any(mask):
        return
    luminance = relative_luminance(candidate)
    encoded_gray = np.where(
        luminance <= 0.0031308,
        luminance * 12.92,
        1.055 * np.power(luminance, 1.0 / 2.4) - 0.055,
    )
    target = np.repeat(encoded_gray[..., None], 3, axis=-1) * 255.0
    weight = (alpha * mask)[..., None]
    candidate[:] = np.rint(candidate * (1.0 - weight) + target * weight).astype(np.uint8)


def _flatten_patterns(candidate: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> None:
    for frame_index, frame_mask in enumerate(mask):
        if not np.any(frame_mask):
            continue
        target = np.mean(candidate[frame_index, frame_mask].astype(np.float64), axis=0)
        weight = (alpha[frame_index, frame_mask])[:, None]
        candidate[frame_index, frame_mask] = np.rint(
            candidate[frame_index, frame_mask] * (1.0 - weight) + target * weight
        ).astype(np.uint8)


def _clamp_temporal_steps(
    candidate: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
    *,
    max_luminance_step: float,
) -> None:
    normalized = candidate.astype(np.float64) / 255.0
    for index in range(1, len(normalized)):
        active = mask[index]
        if not np.any(active):
            continue
        previous_luminance = relative_luminance(
            np.rint(normalized[index - 1 : index] * 255.0).astype(np.uint8)
        )[0]
        current_luminance = relative_luminance(
            np.rint(normalized[index : index + 1] * 255.0).astype(np.uint8)
        )[0]
        step = np.abs(current_luminance - previous_luminance)
        over = active & (step > max_luminance_step)
        blend = np.ones_like(step)
        blend[over] = max_luminance_step / step[over]
        target = normalized[index].copy()
        target[over] = (
            normalized[index - 1, over] * (1.0 - blend[over, None])
            + normalized[index, over] * blend[over, None]
        )
        weight = (alpha[index] * over)[..., None]
        normalized[index] = normalized[index] * (1.0 - weight) + target * weight
    candidate[:] = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def _structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    x = first.astype(np.float64) / 255.0
    y = second.astype(np.float64) / 255.0
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    variance_x = float(np.var(x))
    variance_y = float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    )


def _hazard_band_power(frames: np.ndarray, timestamps: np.ndarray) -> float:
    signal = np.mean(relative_luminance(frames), axis=(1, 2))
    step = float(np.median(np.diff(timestamps)))
    uniform_times = np.arange(timestamps[0], timestamps[-1] + step * 0.5, step)
    uniform_signal = np.interp(uniform_times, timestamps, signal)
    spectrum = np.fft.rfft(uniform_signal - np.mean(uniform_signal))
    frequencies = np.fft.rfftfreq(len(uniform_signal), d=step)
    band = (frequencies >= 3.0) & (frequencies <= 30.0)
    return float(np.sum(np.abs(spectrum[band]) ** 2) / max(1, len(uniform_signal) ** 2))


def repair_with_evidence(
    frames: np.ndarray,
    timestamps: np.ndarray,
    analysis: Analysis,
    *,
    max_luminance_step: float = 0.08,
) -> RepairOutcome:
    frame_array, _ = _validate(frames, timestamps)
    if not 0.0 < max_luminance_step < 0.1:
        raise ValueError("max_luminance_step must be in (0, 0.1)")
    if analysis.hazard_mask.shape != frame_array.shape[:3]:
        raise ValueError("analysis hazard mask does not match the input clip")
    hard_mask = analysis.hazard_mask.copy()
    feather = _feather_mask(hard_mask)
    candidate = frame_array.copy()
    if analysis.hazardous:
        _flatten_patterns(
            candidate,
            analysis.kind_masks.get("regular_pattern", np.zeros_like(hard_mask)),
            feather,
        )
        _clamp_temporal_steps(
            candidate,
            hard_mask,
            feather,
            max_luminance_step=max_luminance_step,
        )

    fallback_used = False
    for _ in range(6):
        residual = analyze(candidate, timestamps)
        if not residual.hazardous:
            break
        fallback_used = True
        bounded_masks = {
            kind: mask & hard_mask for kind, mask in residual.kind_masks.items()
        }
        full_weight = hard_mask.astype(np.float64)
        _flatten_patterns(
            candidate,
            bounded_masks.get("regular_pattern", np.zeros_like(hard_mask)),
            full_weight,
        )
        _desaturate(
            candidate,
            bounded_masks.get("red_flash", np.zeros_like(hard_mask)),
            full_weight,
        )
        _clamp_temporal_steps(
            candidate,
            residual.hazard_mask & hard_mask,
            full_weight,
            max_luminance_step=max_luminance_step,
        )

    verified = not analyze(candidate, timestamps).hazardous
    changed = np.any(candidate != frame_array, axis=-1)
    before_power = _hazard_band_power(frame_array, np.asarray(timestamps, dtype=np.float64))
    after_power = _hazard_band_power(candidate, np.asarray(timestamps, dtype=np.float64))
    return RepairOutcome(
        frames=candidate,
        feather_mask=feather,
        fallback_used=fallback_used,
        verified=verified,
        changed_fraction=float(np.mean(changed)),
        outside_hazard_unchanged=bool(np.array_equal(candidate[~hard_mask], frame_array[~hard_mask])),
        structural_similarity=_structural_similarity(frame_array, candidate),
        hazard_band_power_before=before_power,
        hazard_band_power_after=after_power,
    )


def repair(
    frames: np.ndarray,
    timestamps: np.ndarray,
    analysis: Analysis,
    *,
    max_luminance_step: float = 0.08,
) -> np.ndarray:
    return repair_with_evidence(
        frames,
        timestamps,
        analysis,
        max_luminance_step=max_luminance_step,
    ).frames
