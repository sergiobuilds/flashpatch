from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verification:
    passed: bool
    max_transition_count: int
    max_affected_fraction: float


def _linear_luminance(frames: np.ndarray) -> np.ndarray:
    values = frames.astype(np.float64) / 255.0
    linear = np.empty_like(values)
    low = values <= 0.04045
    linear[low] = values[low] / 12.92
    linear[~low] = ((values[~low] + 0.055) / 1.055) ** 2.4
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def verify(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    luminance_delta: float = 0.1,
    max_flashes_per_second: int = 3,
    area_threshold: float = 0.25,
) -> Verification:
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
        raise ValueError("timestamps must match frames")
    if not np.all(np.isfinite(time_array)) or np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
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

    luminance = _linear_luminance(frame_array)
    difference = np.diff(luminance, axis=0)
    changes = np.abs(difference)
    darker = np.minimum(luminance[:-1], luminance[1:])
    transitions = (changes >= luminance_delta) & (darker < 0.8)
    signed = np.where(transitions, np.sign(difference), 0).astype(np.int8)
    transition_times = time_array[1:]

    max_count = 0
    max_fraction = 0.0
    failed = False
    for right, end_time in enumerate(transition_times):
        window_times = transition_times[: right + 1]
        scale = np.maximum(np.maximum(np.abs(end_time), np.abs(window_times)), 1.0)
        tolerance = 4.0 * np.finfo(np.float64).eps * scale
        left = int(np.count_nonzero(end_time - window_times > 1.0 + tolerance))
        sequence = np.zeros(frame_array.shape[1:3], dtype=np.int16)
        best = np.zeros_like(sequence)
        last = np.zeros(frame_array.shape[1:3], dtype=np.int8)
        for step in signed[left : right + 1]:
            changed = step != 0
            alternates = changed & (last != 0) & (step == -last)
            starts = changed & ~alternates
            sequence[alternates] += 1
            sequence[starts] = 1
            last[changed] = step[changed]
            best = np.maximum(best, sequence)
        counts = best
        max_count = max(max_count, int(counts.max(initial=0)))
        affected = counts >= 2 * max_flashes_per_second + 1
        fraction = float(np.mean(affected))
        max_fraction = max(max_fraction, fraction)
        if fraction > area_threshold:
            failed = True

    return Verification(
        passed=not failed,
        max_transition_count=max_count,
        max_affected_fraction=max_fraction,
    )
