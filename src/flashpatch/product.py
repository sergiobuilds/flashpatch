from __future__ import annotations

from pathlib import Path

from .cli import _repair_file, _scan_file, _verify_file


def scan_video(path: str | Path, *, mask: str | Path | None = None) -> dict[str, object]:
    source = Path(path)
    mask_path = None if mask is None else Path(mask)
    return _scan_file(source, mask_path)


def repair_video(
    source: str | Path,
    destination: str | Path,
    *,
    receipt: str | Path,
) -> dict[str, object]:
    return _repair_file(Path(source), Path(destination), Path(receipt))


def verify_video(path: str | Path) -> dict[str, object]:
    return _verify_file(Path(path))