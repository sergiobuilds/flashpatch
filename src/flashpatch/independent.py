from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .media import VideoFrames
from .verify import verify


@dataclass(frozen=True)
class IndependentVideoVerification:
    passed: bool
    decoder: str
    decoder_agreement: bool
    disagreement_reasons: tuple[str, ...]
    frame_count: int
    max_transition_count: int
    max_affected_fraction: float


def _probe_timestamps(path: Path) -> np.ndarray:
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
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [
        float(frame["best_effort_timestamp_time"])
        for frame in json.loads(probe.stdout)["frames"]
    ]
    return np.asarray(values, dtype=np.float64)


def _decode_with_opencv(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("independent decoder could not open the video")
    frames: list[np.ndarray] = []
    try:
        while True:
            decoded, bgr = capture.read()
            if not decoded:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError("independent decoder produced no frames")
    return np.stack(frames).astype(np.uint8, copy=False)


def verify_video_independently(
    path: str | Path,
    primary: VideoFrames,
    *,
    pixel_mean_absolute_tolerance: float = 4.0,
) -> IndependentVideoVerification:
    source = Path(path)
    reasons: list[str] = []
    try:
        frames = _decode_with_opencv(source)
        timestamps = _probe_timestamps(source)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return IndependentVideoVerification(
            passed=False,
            decoder="opencv-videoio",
            decoder_agreement=False,
            disagreement_reasons=("decode_failure",),
            frame_count=0,
            max_transition_count=0,
            max_affected_fraction=0.0,
        )

    if len(frames) != len(primary.frames) or len(timestamps) != len(primary.timestamps):
        reasons.append("frame_count")
    if frames.shape[1:] != primary.frames.shape[1:]:
        reasons.append("display_dimensions")
    comparable = not reasons
    if comparable and not np.allclose(timestamps, primary.timestamps, rtol=0.0, atol=1e-6):
        reasons.append("timestamps")
    if comparable:
        pixel_error = float(
            np.mean(np.abs(frames.astype(np.int16) - primary.frames.astype(np.int16)))
        )
        if pixel_error > pixel_mean_absolute_tolerance:
            reasons.append("decoded_pixels")

    try:
        result = verify(frames, timestamps)
    except ValueError:
        reasons.append("verification_input")
        result = None
    agreement = not reasons
    return IndependentVideoVerification(
        passed=bool(agreement and result is not None and result.passed),
        decoder="opencv-videoio",
        decoder_agreement=agreement,
        disagreement_reasons=tuple(reasons),
        frame_count=len(frames),
        max_transition_count=0 if result is None else result.max_transition_count,
        max_affected_fraction=0.0 if result is None else result.max_affected_fraction,
    )
