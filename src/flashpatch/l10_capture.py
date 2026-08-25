"""Deterministically pack engine frame directories into L10 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .core import analyze


class L10CaptureError(ValueError):
    """Engine output cannot be promoted to an L10 capture artifact."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def pack_engine_capture(source: Path, output: Path) -> dict[str, object]:
    """Validate exact PNG/timestamp output, run the detector, and bind all bytes."""
    root = source.resolve()
    if not root.is_dir() or root.is_symlink() or output.exists():
        raise L10CaptureError("capture source is unsafe or output already exists")
    frames = sorted(root.glob("[0-9][0-9][0-9][0-9].png"))
    expected = [root / f"{index:04d}.png" for index in range(len(frames))]
    if len(frames) < 2 or frames != expected or any(path.is_symlink() for path in frames):
        raise L10CaptureError("capture PNG sequence is incomplete or noncanonical")
    timestamp_path = root / "timestamps.txt"
    if timestamp_path.is_symlink() or not timestamp_path.is_file():
        raise L10CaptureError("capture timestamps are missing or unsafe")
    try:
        timestamps = np.array(
            [float(item) for item in timestamp_path.read_text(encoding="utf-8").splitlines()],
            dtype=np.float64,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise L10CaptureError("capture timestamps are unreadable") from exc
    if (
        timestamps.shape != (len(frames),)
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise L10CaptureError("capture timestamps are invalid")
    decoded: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise L10CaptureError("capture PNG is not a decodable RGB frame")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if shape is None:
            shape = rgb.shape
        elif rgb.shape != shape:
            raise L10CaptureError("capture PNG dimensions changed")
        decoded.append(rgb)
    array = np.stack(decoded).astype(np.uint8, copy=False)
    result = analyze(array, timestamps)
    output.mkdir(parents=True)
    frame_artifact = output / "frames.npz"
    np.savez_compressed(frame_artifact, frames=array, timestamps=timestamps)
    timestamp_artifact = output / "timestamps.json"
    timestamp_artifact.write_bytes(_canonical(timestamps.tolist()))
    png_manifest = output / "png-manifest.json"
    png_manifest.write_bytes(_canonical([
        {"path": path.name, "sha256": _sha256(path)} for path in frames
    ]))
    detector = {
        "schema": "flashpatch-l10-detector-receipt-v1",
        "frames_rgb_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "timestamps_sha256": hashlib.sha256(timestamps.tobytes(order="C")).hexdigest(),
        "result": "HAZARDOUS" if result.hazardous else "SAFE",
        "hazard_frames": sorted({
            index
            for window in result.windows
            for index, timestamp in enumerate(timestamps)
            if window.start <= timestamp <= window.end
        }),
        "max_flash_count": result.max_flash_count,
        "max_affected_fraction": result.max_affected_fraction,
    }
    detector_path = output / "detector.json"
    detector_path.write_bytes(_canonical(detector))
    receipt = {
        "schema": "flashpatch-l10-capture-pack-v1",
        "frame_count": len(frames),
        "width": int(array.shape[2]),
        "height": int(array.shape[1]),
        "frames": {"path": frame_artifact.name, "sha256": _sha256(frame_artifact)},
        "timestamps": {"path": timestamp_artifact.name, "sha256": _sha256(timestamp_artifact)},
        "png_manifest": {"path": png_manifest.name, "sha256": _sha256(png_manifest)},
        "detector": {"path": detector_path.name, "sha256": _sha256(detector_path)},
        "result": detector["result"],
        "hazard_frames": detector["hazard_frames"],
    }
    (output / "capture-pack.json").write_bytes(_canonical(receipt))
    return receipt
