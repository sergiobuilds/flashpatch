from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import pytest

from flashpatch.l10_capture import L10CaptureError, pack_engine_capture


def _frames(root: Path, *, hazardous: bool = False) -> None:
    root.mkdir()
    for index in range(18):
        value = 255 if hazardous and index % 2 else 0
        image = np.full((16, 20, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(root / f"{index:04d}.png"), image)
    (root / "timestamps.txt").write_text("\n".join(str(index / 30) for index in range(18)) + "\n")


def test_packs_hazardous_png_sequence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _frames(source, hazardous=True)
    receipt = pack_engine_capture(source, tmp_path / "packed")
    assert receipt["result"] == "HAZARDOUS"
    assert receipt["frame_count"] == 18


def test_rejects_missing_frame(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _frames(source)
    (source / "0008.png").unlink()
    with pytest.raises(L10CaptureError, match="incomplete"):
        pack_engine_capture(source, tmp_path / "packed")


def test_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _frames(source)
    (source / "timestamps.txt").write_text("0\n" * 18)
    with pytest.raises(L10CaptureError, match="timestamps"):
        pack_engine_capture(source, tmp_path / "packed")


def test_rejects_stale_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _frames(source)
    output = tmp_path / "packed"
    output.mkdir()
    with pytest.raises(L10CaptureError, match="already exists"):
        pack_engine_capture(source, output)
