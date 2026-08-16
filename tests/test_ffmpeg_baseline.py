from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from flashpatch.core import analyze, repair
from flashpatch.verify import verify


def test_ffmpeg_global_repair_changes_safe_moving_region(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    if "ffmpeg version 6.1.1-3ubuntu5 " not in version:
        pytest.skip("baseline is pinned to FFmpeg 6.1.1-3ubuntu5")

    width, height, fps, frame_count = 64, 48, 30, 60
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        frames[index] = 80 + index
    hazard_region = np.zeros((height, width), dtype=bool)
    hazard_region[8:40, 8:56] = True
    for index in range(frame_count):
        frames[index, hazard_region] = 255 if (index // 3) % 2 else 0

    raw_path = tmp_path / "input.rgb"
    input_path = tmp_path / "input.mkv"
    output_path = tmp_path / "ffmpeg.mkv"
    decoded_path = tmp_path / "ffmpeg.rgb"
    raw_path.write_bytes(frames.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", str(raw_path), "-c:v", "ffv1", "-level", "3",
            str(input_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
            "-vf", "photosensitivity=frames=30:threshold=1:skip=1",
            "-c:v", "ffv1", "-level", "3", str(output_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(output_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", str(decoded_path),
        ],
        check=True,
    )

    ffmpeg_frames = np.frombuffer(decoded_path.read_bytes(), dtype=np.uint8).reshape(frames.shape)
    ffmpeg_changed = np.any(ffmpeg_frames != frames, axis=-1)
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    detection = analyze(frames, timestamps)
    repaired = repair(frames, timestamps, detection)
    flashpatch_changed = np.any(repaired != frames, axis=-1)

    assert not verify(frames, timestamps).passed
    assert verify(repaired, timestamps).passed
    assert ffmpeg_changed[:, ~hazard_region].mean() == pytest.approx(0.95)
    assert not flashpatch_changed[:, ~hazard_region].any()
