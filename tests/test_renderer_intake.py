from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flashpatch.renderer_intake import inspect_renderer_capture


def _write_capture(path: Path, frames: np.ndarray, timestamps: np.ndarray) -> None:
    np.savez_compressed(path, frames=frames, timestamps=timestamps)


def test_intake_records_factual_safe_capture_and_scope(tmp_path: Path) -> None:
    source = tmp_path / "safe.npz"
    frames = np.full((12, 8, 10, 3), 96, dtype=np.uint8)
    timestamps = np.arange(12, dtype=np.float64) / 30.0
    _write_capture(source, frames, timestamps)
    receipt = inspect_renderer_capture(source)
    assert receipt["schema"] == "flashpatch-renderer-intake-receipt-v1"
    assert receipt["detector"]["result"] == "SAFE"
    assert "engine identity or version" in receipt["scope"]["not_established"]


def test_cli_publishes_hazard_receipt_only_for_valid_capture(tmp_path: Path) -> None:
    source = tmp_path / "hazard.npz"
    receipt_path = tmp_path / "receipt.json"
    frames = np.zeros((18, 16, 16, 3), dtype=np.uint8)
    frames[1::2] = 255
    _write_capture(source, frames, np.arange(18, dtype=np.float64) / 30.0)
    completed = subprocess.run(
        [sys.executable, "-m", "flashpatch.cli", "renderer-intake", str(source), "--receipt", str(receipt_path)],
        check=True, text=True, capture_output=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["detector"]["result"] == "HAZARDOUS"
    assert receipt == json.loads(receipt_path.read_text(encoding="utf-8"))


def test_cli_fails_closed_without_publishing_for_invalid_capture(tmp_path: Path) -> None:
    source = tmp_path / "invalid.npz"
    receipt_path = tmp_path / "receipt.json"
    np.savez_compressed(source, frames=np.zeros((2, 2, 2, 3), dtype=np.uint8))
    completed = subprocess.run(
        [sys.executable, "-m", "flashpatch.cli", "renderer-intake", str(source), "--receipt", str(receipt_path)],
        text=True, capture_output=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["verdict"] == "INCONCLUSIVE"
    assert not receipt_path.exists()
