"""Bounded-memory access to renderer RGB frame artifacts."""

from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


DEFAULT_MAX_DECOMPRESSED_BYTES = 8 * 1024**3
_EXPECTED_MEMBERS = frozenset({"frames.npy", "timestamps.npy"})
_COPY_BUFFER_BYTES = 1024 * 1024


class RendererArtifactError(ValueError):
    """A packed renderer artifact violates the fail-closed frame contract."""


@dataclass(frozen=True)
class RendererFrames:
    frames: np.memmap
    timestamps: np.memmap


@dataclass(frozen=True)
class _NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype[object]
    fortran_order: bool
    payload_offset: int

    @property
    def payload_bytes(self) -> int:
        elements = math.prod(self.shape)
        return elements * self.dtype.itemsize

    @property
    def file_bytes(self) -> int:
        return self.payload_offset + self.payload_bytes


def _read_npy_header(stream: object, member: str) -> _NpyHeader:
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise RendererArtifactError(
                f"renderer artifact member {member!r} uses unsupported NPY version {version}"
            )
        payload_offset = stream.tell()
    except RendererArtifactError:
        raise
    except (EOFError, OSError, ValueError) as exc:
        raise RendererArtifactError(
            f"renderer artifact member {member!r} has an invalid NPY header"
        ) from exc
    normalized = np.dtype(dtype)
    if normalized.hasobject:
        raise RendererArtifactError(
            f"renderer artifact member {member!r} must not contain object data"
        )
    if not isinstance(shape, tuple) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in shape
    ):
        raise RendererArtifactError(
            f"renderer artifact member {member!r} has an invalid array shape"
        )
    return _NpyHeader(
        shape=shape,
        dtype=normalized,
        fortran_order=bool(fortran_order),
        payload_offset=payload_offset,
    )


def _validate_headers(
    frames: _NpyHeader,
    timestamps: _NpyHeader,
    *,
    max_decompressed_bytes: int,
) -> None:
    if frames.dtype != np.dtype(np.uint8):
        raise RendererArtifactError("renderer frames must use uint8 RGB pixels")
    if len(frames.shape) != 4 or frames.shape[-1] != 3:
        raise RendererArtifactError(
            "renderer frames must have shape [time, height, width, 3]"
        )
    if len(frames.shape) == 0 or any(dimension <= 0 for dimension in frames.shape):
        raise RendererArtifactError("renderer frame dimensions must be positive")
    if timestamps.dtype != np.dtype(np.float64):
        raise RendererArtifactError("renderer timestamps must use float64 seconds")
    if timestamps.shape != (frames.shape[0],):
        raise RendererArtifactError(
            "renderer timestamps must contain one value per frame"
        )
    total_bytes = frames.file_bytes + timestamps.file_bytes
    if total_bytes > max_decompressed_bytes:
        raise RendererArtifactError(
            "renderer artifact exceeds the decompressed byte limit"
        )


def _inspect_archive(
    archive: zipfile.ZipFile,
    *,
    max_decompressed_bytes: int,
) -> tuple[dict[str, zipfile.ZipInfo], dict[str, _NpyHeader]]:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise RendererArtifactError("renderer artifact contains duplicate ZIP members")
    if frozenset(names) != _EXPECTED_MEMBERS or len(names) != len(_EXPECTED_MEMBERS):
        raise RendererArtifactError(
            "renderer artifact must contain only frames.npy and timestamps.npy"
        )
    infos = {entry.filename: entry for entry in entries}
    if any(entry.is_dir() or entry.flag_bits & 0x1 for entry in entries):
        raise RendererArtifactError(
            "renderer artifact members must be unencrypted regular files"
        )
    declared_bytes = sum(entry.file_size for entry in entries)
    if declared_bytes > max_decompressed_bytes:
        raise RendererArtifactError(
            "renderer artifact exceeds the decompressed byte limit"
        )
    headers: dict[str, _NpyHeader] = {}
    for name, info in infos.items():
        try:
            with archive.open(info, mode="r") as stream:
                header = _read_npy_header(stream, name)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            raise RendererArtifactError(
                f"renderer artifact member {name!r} cannot be read"
            ) from exc
        if header.file_bytes != info.file_size:
            raise RendererArtifactError(
                f"renderer artifact member {name!r} payload size does not match its NPY header"
            )
        headers[name] = header
    _validate_headers(
        headers["frames.npy"],
        headers["timestamps.npy"],
        max_decompressed_bytes=max_decompressed_bytes,
    )
    return infos, headers


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> None:
    try:
        with archive.open(info, mode="r") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=_COPY_BUFFER_BYTES)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise RendererArtifactError(
            f"renderer artifact member {info.filename!r} could not be extracted"
        ) from exc
    if destination.stat().st_size != info.file_size:
        raise RendererArtifactError(
            f"renderer artifact member {info.filename!r} was truncated during extraction"
        )


@contextmanager
def open_renderer_artifact(
    path: str | Path,
    *,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    temp_parent: str | Path | None = None,
) -> Iterator[RendererFrames]:
    """Yield verified mmap arrays and remove all extracted bytes on exit."""
    source = Path(path)
    if isinstance(max_decompressed_bytes, bool) or max_decompressed_bytes <= 0:
        raise ValueError("max_decompressed_bytes must be positive")
    if not source.is_file():
        raise RendererArtifactError(f"renderer artifact is missing: {source}")
    parent = None if temp_parent is None else str(Path(temp_parent))
    try:
        with tempfile.TemporaryDirectory(
            prefix="flashpatch-renderer-npz-", dir=parent
        ) as temporary:
            temporary_root = Path(temporary)
            try:
                with zipfile.ZipFile(source, mode="r") as archive:
                    infos, headers = _inspect_archive(
                        archive,
                        max_decompressed_bytes=max_decompressed_bytes,
                    )
                    for name in ("frames.npy", "timestamps.npy"):
                        _extract_member(archive, infos[name], temporary_root / name)
            except RendererArtifactError:
                raise
            except (OSError, zipfile.BadZipFile) as exc:
                raise RendererArtifactError(
                    "renderer artifact is not a readable NPZ archive"
                ) from exc

            try:
                frames = np.load(
                    temporary_root / "frames.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                timestamps = np.load(
                    temporary_root / "timestamps.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
            except (OSError, ValueError) as exc:
                raise RendererArtifactError(
                    "renderer artifact extracted arrays cannot be memory mapped"
                ) from exc
            if not isinstance(frames, np.memmap) or not isinstance(timestamps, np.memmap):
                raise RendererArtifactError(
                    "renderer artifact arrays were not opened as memory maps"
                )
            if (
                frames.shape != headers["frames.npy"].shape
                or frames.dtype != headers["frames.npy"].dtype
                or timestamps.shape != headers["timestamps.npy"].shape
                or timestamps.dtype != headers["timestamps.npy"].dtype
            ):
                raise RendererArtifactError(
                    "renderer artifact arrays do not match their verified headers"
                )
            if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
                raise RendererArtifactError(
                    "renderer timestamps must be finite and strictly increasing"
                )
            try:
                yield RendererFrames(frames=frames, timestamps=timestamps)
            finally:
                frames._mmap.close()
                timestamps._mmap.close()
    except RendererArtifactError:
        raise
    except OSError as exc:
        raise RendererArtifactError(
            "renderer artifact temporary extraction failed"
        ) from exc


def renderer_rgb_sha256(frames: np.ndarray) -> str:
    """Hash C-order RGB bytes while retaining at most one frame at a time."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def renderer_visual_change_ratio(first: np.ndarray, second: np.ndarray) -> float:
    """Compare matching RGB arrays with row-strip bounded temporary state."""
    if first.shape != second.shape or first.dtype != np.uint8 or second.dtype != np.uint8:
        raise RendererArtifactError(
            "renderer candidate frames must match factual uint8 RGB frame shape"
        )
    changed_pixels = 0
    total_pixels = math.prod(first.shape[:3])
    for frame_index in range(first.shape[0]):
        for row_start in range(0, first.shape[1], 128):
            row_stop = min(row_start + 128, first.shape[1])
            changed = np.any(
                first[frame_index, row_start:row_stop]
                != second[frame_index, row_start:row_stop],
                axis=-1,
            )
            changed_pixels += int(np.count_nonzero(changed))
    return float(changed_pixels / total_pixels)
