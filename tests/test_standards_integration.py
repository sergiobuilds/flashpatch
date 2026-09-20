from __future__ import annotations

import math

import numpy as np

from flashpatch.core import (
    _luminance_transition_directions,
    _red_chromaticity,
    _red_transition_directions,
    relative_luminance,
)
from flashpatch.standards import (
    general_flash_transition_is_flash,
    saturated_red_threshold_is_met,
)


def test_vectorized_general_flash_matches_standard_contract() -> None:
    frames = np.array(
        [
            [
                [(0, 0, 0), (120, 120, 120), (230, 230, 230), (230, 230, 230)],
            ],
            [
                [(255, 255, 255), (255, 255, 255), (255, 255, 255), (230, 230, 230)],
            ],
        ],
        dtype=np.uint8,
    )

    luminance = relative_luminance(frames)
    expected = np.vectorize(general_flash_transition_is_flash)(
        luminance[0], luminance[1]
    )
    actual = _luminance_transition_directions(
        frames, luminance_delta=0.1
    )[0] != 0

    np.testing.assert_array_equal(actual, expected)


def test_vectorized_red_flash_matches_standard_contract() -> None:
    frames = np.array(
        [
            [[(255, 0, 0), (0, 0, 0), (0, 255, 0)]],
            [[(0, 255, 0), (0, 255, 0), (0, 255, 0)]],
        ],
        dtype=np.uint8,
    )

    previous_ratio, previous_u, previous_v = _red_chromaticity(frames[0])
    current_ratio, current_u, current_v = _red_chromaticity(frames[1])
    expected = np.zeros(previous_ratio.shape, dtype=bool)
    for index in np.ndindex(expected.shape):
        distance = math.hypot(
            float(current_u[index] - previous_u[index]),
            float(current_v[index] - previous_v[index]),
        )
        expected[index] = saturated_red_threshold_is_met(
            max(float(previous_ratio[index]), float(current_ratio[index])),
            distance,
        )

    actual = _red_transition_directions(frames)[0] != 0

    np.testing.assert_array_equal(actual, expected)
