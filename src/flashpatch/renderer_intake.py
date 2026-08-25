"""Fail-closed intake for externally captured renderer frames.
The intake validates only the supplied ``frame_npz_v1`` artifact.  It deliberately
does not infer an engine, source revision, replay trace, display behaviour, or a
repair result from those pixels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .core import analyze
from .renderer_artifact import (
    open_renderer_artifact,
    renderer_rgb_sha256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    return _sha256_file(Path(__file__).with_name("core.py"))


def _window_payload(window: Any) -> dict[str, float | str]:
    return {
        "start_seconds": window.start,
        "end_seconds": window.end,
        "affected_fraction": window.affected_fraction,
        "flash_count": window.flash_count,
        "kind": window.kind,
    }


def inspect_renderer_capture(path: Path) -> dict[str, object]:
    """Return a factual detector receipt for one strict ``frame_npz_v1`` capture.

    ``RendererArtifactError`` is intentionally left to the CLI boundary: malformed
    evidence produces INCONCLUSIVE there and no receipt is published.
    """
    source = path.resolve()
    with open_renderer_artifact(source) as artifact:
        analysis = analyze(artifact.frames, artifact.timestamps)
        return {
            "schema": "flashpatch-renderer-intake-receipt-v1",
            "input_contract": "frame_npz_v1",
            "input": {
                "sha256": _sha256_file(source),
                "frames_rgb_sha256": renderer_rgb_sha256(artifact.frames),
                "timestamps_sha256": hashlib.sha256(artifact.timestamps.tobytes()).hexdigest(),
                "frame_count": int(artifact.frames.shape[0]),
                "height": int(artifact.frames.shape[1]),
                "width": int(artifact.frames.shape[2]),
            },
            "detector": {
                "source_sha256": _source_sha256(),
                "result": "HAZARDOUS" if analysis.hazardous else "SAFE",
                "max_flash_count": analysis.max_flash_count,
                "max_affected_fraction": analysis.max_affected_fraction,
                "hazard_windows": [_window_payload(window) for window in analysis.windows],
            },
            "scope": {
                "observed": "The supplied RGB frames and timestamps were validated and analyzed.",
                "not_established": [
                    "engine identity or version",
                    "source revision or runtime-to-source causality",
                    "replay trace or gameplay preservation",
                    "display output or individual safety",
                    "repair or mitigation effectiveness",
                ],
            },
        }


def write_renderer_intake_receipt(receipt: dict[str, object], destination: Path) -> None:
    """Atomically publish an already-complete factual intake receipt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
