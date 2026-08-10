from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from flashpatch.media import VideoMetadata, read_video, write_video


def test_mp4_round_trip_preserves_frame_pts_and_dimensions(tmp_path: Path) -> None:
    frames = np.zeros((4, 8, 10, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 32
    frames[1, :, :, 1] = 96
    frames[2, :, :, 2] = 160
    frames[3, :, :, :] = 224
    timestamps = np.array([1.25, 1.283, 1.354, 1.455], dtype=np.float64)
    output = tmp_path / "variable-frame-rate.mp4"

    write_video(output, frames, timestamps)
    decoded = read_video(output)


    assert decoded.frames.shape == frames.shape
    assert decoded.frames.dtype == np.uint8
    assert np.mean(np.abs(decoded.frames.astype(int) - frames.astype(int))) < 3.0
    np.testing.assert_allclose(decoded.timestamps, timestamps, rtol=0.0, atol=1e-6)
    if shutil.which("ffprobe") is not None:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probed_timestamps = np.array(
            [
                float(frame["best_effort_timestamp_time"])
                for frame in json.loads(probe.stdout)["frames"]
            ]
        )
        np.testing.assert_allclose(probed_timestamps, timestamps, rtol=0.0, atol=1e-6)


def test_mp4_writer_rejects_timestamp_collisions_at_microsecond_timebase(tmp_path: Path) -> None:
    frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.0000004, 0.000002], dtype=np.float64)

    with pytest.raises(ValueError, match="microsecond"):
        write_video(tmp_path / "collision.mp4", frames, timestamps)


def test_mp4_writer_rejects_timestamps_beyond_exact_float_microsecond_range(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    timestamps = np.array([0.0, (2**53 + 2) / 1_000_000], dtype=np.float64)

    with pytest.raises(ValueError, match="distinguishable microsecond"):
        write_video(tmp_path / "out-of-range.mp4", frames, timestamps)


def test_mp4_reader_enforces_decoded_byte_limit(tmp_path: Path) -> None:
    frames = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    timestamps = np.array([0.0, 0.04, 0.08], dtype=np.float64)
    output = tmp_path / "bounded.mp4"
    write_video(output, frames, timestamps)

    with pytest.raises(ValueError, match="decoded array byte limit"):
        read_video(output, max_decoded_array_bytes=frames.nbytes + frames[0].nbytes - 1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="rotation fixture requires ffmpeg")
def test_mp4_reader_applies_display_rotation(tmp_path: Path) -> None:
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    frames[:, :3, :4, 0] = 255
    timestamps = np.array([0.0, 0.04], dtype=np.float64)
    source = tmp_path / "source.mp4"
    rotated = tmp_path / "rotated.mp4"
    normalized = tmp_path / "normalized.mp4"
    write_video(source, frames, timestamps)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )

    source_decoded = read_video(source)
    decoded = read_video(rotated)

    assert decoded.frames.shape == (2, 10, 8, 3)
    np.testing.assert_array_equal(decoded.frames, np.rot90(source_decoded.frames, k=1, axes=(1, 2)))
    write_video(normalized, decoded.frames, decoded.timestamps, metadata=decoded.metadata)
    reopened = read_video(normalized)
    assert reopened.frames.shape == decoded.frames.shape
    assert np.mean(np.abs(reopened.frames.astype(int) - decoded.frames.astype(int))) < 3.0
    np.testing.assert_allclose(reopened.timestamps, decoded.timestamps, rtol=0.0, atol=1e-6)


def test_file_cli_scans_repairs_and_verifies_vfr_mp4_with_color_metadata(tmp_path: Path) -> None:
    frames = np.zeros((12, 8, 10, 3), dtype=np.uint8)
    frames[1::2] = 255
    timestamps = np.array(
        [0.0, 0.08, 0.19, 0.27, 0.39, 0.48, 0.57, 0.69, 0.78, 0.88, 0.97, 1.09],
        dtype=np.float64,
    )
    metadata = VideoMetadata(color_range=1, colorspace=1, color_primaries=1, color_trc=1)
    source = tmp_path / "hazard-vfr.mp4"
    repaired = tmp_path / "repaired-vfr.mp4"
    receipt = tmp_path / "repair-receipt.json"
    mask = tmp_path / "scan-mask.npz"
    write_video(source, frames, timestamps, metadata=metadata)

    scan = subprocess.run(
        [sys.executable, "-m", "flashpatch.cli", "scan", str(source), "--mask", str(mask)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(scan.stdout)["hazardous"] is True
    with np.load(mask) as localized:
        assert localized["aggregate"].shape == frames.shape[:3]
        np.testing.assert_allclose(localized["timestamps"], timestamps, rtol=0.0, atol=1e-6)

    repair = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.cli",
            "repair",
            str(source),
            str(repaired),
            "--receipt",
            str(receipt),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    repair_payload = json.loads(repair.stdout)
    assert repair_payload["status"] == "VERIFIED"
    assert repair_payload["repair_verified"] is True
    assert 0.0 < repair_payload["changed_fraction"] <= 1.0
    assert repair_payload["outside_hazard_unchanged"] is True
    assert repair_payload["structural_similarity"] > 0.0
    assert (
        repair_payload["hazard_band_power_after"]
        <= repair_payload["hazard_band_power_before"] + 1e-12
    )
    assert isinstance(repair_payload["fallback_used"], bool)
    assert json.loads(receipt.read_text(encoding="utf-8")) == repair_payload

    verification = subprocess.run(
        [sys.executable, "-m", "flashpatch.cli", "verify", str(repaired)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verification.stdout)["passed"] is True

    reopened = read_video(repaired)
    np.testing.assert_allclose(reopened.timestamps, timestamps, rtol=0.0, atol=1e-6)
    assert reopened.metadata == metadata
