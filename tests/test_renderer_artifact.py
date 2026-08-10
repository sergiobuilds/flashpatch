from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from flashpatch.renderer_artifact import (
    RendererArtifactError,
    open_renderer_artifact,
    renderer_rgb_sha256,
    renderer_visual_change_ratio,
)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def test_renderer_artifact_streams_to_memmaps_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    frames = np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3)
    timestamps = np.arange(4, dtype=np.float64) / 60.0
    source = tmp_path / "renderer.npz"
    temporary_parent = tmp_path / "temporary"
    temporary_parent.mkdir()
    np.savez_compressed(source, frames=frames, timestamps=timestamps)

    with open_renderer_artifact(source, temp_parent=temporary_parent) as artifact:
        assert isinstance(artifact.frames, np.memmap)
        assert isinstance(artifact.timestamps, np.memmap)
        assert np.array_equal(artifact.frames, frames)
        assert np.array_equal(artifact.timestamps, timestamps)
        assert len(list(temporary_parent.iterdir())) == 1

    assert list(temporary_parent.iterdir()) == []


def test_renderer_artifact_rejects_unexpected_members_and_cleans_on_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renderer.npz"
    temporary_parent = tmp_path / "temporary"
    temporary_parent.mkdir()
    np.savez_compressed(
        source,
        frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
        timestamps=np.arange(3, dtype=np.float64),
        injected=np.zeros(1, dtype=np.uint8),
    )

    with pytest.raises(RendererArtifactError, match="only frames.npy and timestamps.npy"):
        with open_renderer_artifact(source, temp_parent=temporary_parent):
            pass

    assert list(temporary_parent.iterdir()) == []


def test_renderer_artifact_rejects_truncated_npy_payload_before_extraction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renderer.npz"
    temporary_parent = tmp_path / "temporary"
    temporary_parent.mkdir()
    frame_bytes = _npy_bytes(np.zeros((3, 2, 2, 3), dtype=np.uint8))
    timestamp_bytes = _npy_bytes(np.arange(3, dtype=np.float64))
    with zipfile.ZipFile(source, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("frames.npy", frame_bytes[:-1])
        archive.writestr("timestamps.npy", timestamp_bytes)

    with pytest.raises(RendererArtifactError, match="payload size does not match"):
        with open_renderer_artifact(source, temp_parent=temporary_parent):
            pass

    assert list(temporary_parent.iterdir()) == []


@pytest.mark.parametrize(
    ("frames", "timestamps", "message"),
    [
        (
            np.zeros((3, 2, 2, 3), dtype=np.float32),
            np.arange(3, dtype=np.float64),
            "uint8 RGB",
        ),
        (
            np.zeros((3, 2, 2, 3), dtype=np.uint8),
            np.arange(3, dtype=np.float32),
            "float64 seconds",
        ),
        (
            np.zeros((3, 2, 2), dtype=np.uint8),
            np.arange(3, dtype=np.float64),
            "shape",
        ),
    ],
)
def test_renderer_artifact_rejects_noncanonical_shape_and_dtype(
    tmp_path: Path,
    frames: np.ndarray,
    timestamps: np.ndarray,
    message: str,
) -> None:
    source = tmp_path / "renderer.npz"
    np.savez_compressed(source, frames=frames, timestamps=timestamps)

    with pytest.raises(RendererArtifactError, match=message):
        with open_renderer_artifact(source):
            pass


def test_renderer_artifact_enforces_declared_decompressed_byte_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renderer.npz"
    np.savez_compressed(
        source,
        frames=np.zeros((3, 16, 16, 3), dtype=np.uint8),
        timestamps=np.arange(3, dtype=np.float64),
    )

    with pytest.raises(RendererArtifactError, match="decompressed byte limit"):
        with open_renderer_artifact(source, max_decompressed_bytes=128):
            pass


def test_renderer_artifact_hash_and_visual_diff_match_in_memory_reference(
    tmp_path: Path,
) -> None:
    first = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    second = first.copy()
    second[1, 2, 3] = 0
    timestamps = np.arange(3, dtype=np.float64) / 60.0
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    np.savez_compressed(first_path, frames=first, timestamps=timestamps)
    np.savez_compressed(second_path, frames=second, timestamps=timestamps)

    with open_renderer_artifact(first_path) as factual:
        with open_renderer_artifact(second_path) as candidate:
            assert renderer_rgb_sha256(factual.frames) == __import__("hashlib").sha256(
                first.tobytes()
            ).hexdigest()
            assert renderer_visual_change_ratio(
                factual.frames, candidate.frames
            ) == pytest.approx(float(np.mean(np.any(first != second, axis=-1))))
