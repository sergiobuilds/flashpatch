from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from flashpatch.core import analyze
from flashpatch.media import write_video


def _iou(actual: np.ndarray, expected: np.ndarray) -> float:
    intersection = np.count_nonzero(actual & expected)
    union = np.count_nonzero(actual | expected)
    return intersection / union


def test_red_flash_localizes_gold_time_and_space() -> None:
    frames = np.full((12, 8, 8, 3), 24, dtype=np.uint8)
    spatial_gold = np.zeros((8, 8), dtype=bool)
    spatial_gold[:, :6] = True
    for index in range(len(frames)):
        frames[index, spatial_gold] = (255, 0, 0) if index % 2 else (0, 64, 64)
    timestamps = np.arange(len(frames), dtype=np.float64) / 10.0
    temporal_gold = np.zeros(frames.shape[:3], dtype=bool)
    temporal_gold[1:, spatial_gold] = True

    detection = analyze(frames, timestamps)

    assert "red_flash" in detection.kind_masks
    assert _iou(detection.kind_masks["red_flash"], temporal_gold) > 0.95
    assert any(window.kind == "red_flash" for window in detection.windows)


def test_dynamic_regular_pattern_localizes_gold_region() -> None:
    frames = np.full((8, 16, 32, 3), 96, dtype=np.uint8)
    spatial_gold = np.zeros((16, 32), dtype=bool)
    spatial_gold[:, :24] = True
    for frame_index in range(len(frames)):
        for stripe in range(12):
            value = 240 if (stripe + frame_index) % 2 else 8
            frames[frame_index, :, stripe * 2 : stripe * 2 + 2] = value
    timestamps = np.arange(len(frames), dtype=np.float64) / 10.0
    temporal_gold = np.broadcast_to(spatial_gold, frames.shape[:3])

    detection = analyze(frames, timestamps)

    assert "regular_pattern" in detection.kind_masks
    assert _iou(detection.kind_masks["regular_pattern"], temporal_gold) > 0.95
    assert any(window.kind == "regular_pattern" for window in detection.windows)


def test_smooth_one_direction_regular_pattern_reveal_remains_safe() -> None:
    frames = np.full((8, 16, 128, 3), 96, dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        stripe_count = 18 + frame_index
        for stripe in range(stripe_count):
            value = 240 - frame_index if stripe % 2 else 8 + frame_index
            frame[:, stripe * 2 : stripe * 2 + 2] = value
    timestamps = np.arange(len(frames), dtype=np.float64) / 10.0

    detection = analyze(frames, timestamps)

    assert detection.hazardous is False
    assert "regular_pattern" not in detection.kind_masks
    assert 0.28 < detection.max_affected_fraction < 0.40


def test_safe_smooth_gradient_has_no_pattern_localization() -> None:
    row = np.linspace(16, 224, 32, dtype=np.uint8)
    frame = np.repeat(row[None, :, None], 16, axis=0)
    frame = np.repeat(frame, 3, axis=2)
    frames = np.repeat(frame[None], 8, axis=0)
    timestamps = np.arange(len(frames), dtype=np.float64) / 10.0

    detection = analyze(frames, timestamps)

    assert "regular_pattern" not in detection.kind_masks


def test_real_mp4_scan_exports_red_and_pattern_masks(tmp_path: Path) -> None:
    red_frames = np.full((12, 16, 16, 3), 24, dtype=np.uint8)
    red_gold = np.zeros((16, 16), dtype=bool)
    red_gold[:, :12] = True
    for index in range(len(red_frames)):
        red_frames[index, red_gold] = (255, 0, 0) if index % 2 else (0, 64, 64)
    red_times = np.arange(len(red_frames), dtype=np.float64) / 10.0

    pattern_frames = np.full((8, 16, 32, 3), 96, dtype=np.uint8)
    pattern_gold = np.zeros((16, 32), dtype=bool)
    pattern_gold[:, :24] = True
    for frame_index in range(len(pattern_frames)):
        for stripe in range(12):
            value = 240 if (stripe + frame_index) % 2 else 8
            pattern_frames[frame_index, :, stripe * 2 : stripe * 2 + 2] = value
    pattern_times = np.arange(len(pattern_frames), dtype=np.float64) / 10.0

    for kind, frames, timestamps, spatial_gold in (
        ("red_flash", red_frames, red_times, red_gold),
        ("regular_pattern", pattern_frames, pattern_times, pattern_gold),
    ):
        source = tmp_path / f"{kind}.mp4"
        mask_path = tmp_path / f"{kind}-mask.npz"
        write_video(source, frames, timestamps)
        scan = subprocess.run(
            [sys.executable, "-m", "flashpatch.cli", "scan", str(source), "--mask", str(mask_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert kind in json.loads(scan.stdout)["hazard_kinds"]
        with np.load(mask_path) as masks:
            localized_space = np.any(masks[kind], axis=0)
            assert _iou(localized_space, spatial_gold) > 0.90
