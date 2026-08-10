from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


_TIME_BASE = Fraction(1, 1_000_000)


@dataclass(frozen=True)
class VideoMetadata:
    color_range: int
    colorspace: int
    color_primaries: int
    color_trc: int


@dataclass(frozen=True)
class VideoFrames:
    frames: np.ndarray
    timestamps: np.ndarray
    metadata: VideoMetadata


def read_video(path: str | Path, *, max_decoded_array_bytes: int = 1_073_741_824) -> VideoFrames:
    if max_decoded_array_bytes <= 0:
        raise ValueError("max_decoded_array_bytes must be positive")
    decoded_frames: np.ndarray | None = None
    timestamps: list[float] = []
    with av.open(str(Path(path)), mode="r") as container:
        if not container.streams.video:
            raise ValueError("input has no video stream")
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        metadata = VideoMetadata(
            color_range=int(codec_context.color_range),
            colorspace=int(codec_context.colorspace),
            color_primaries=int(codec_context.color_primaries),
            color_trc=int(codec_context.color_trc),
        )
        expected_frames = int(stream.frames)
        if expected_frames <= 0:
            raise ValueError("video does not declare a bounded frame count")
        frame_bytes = stream.width * stream.height * 3
        if frame_bytes * (expected_frames + 1) > max_decoded_array_bytes:
            raise ValueError("video exceeds decoded array byte limit")
        decoded_count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError("decoded video frame has no presentation timestamp")
            rotation = frame.rotation % 360
            if rotation not in (0, 90, 180, 270):
                raise ValueError(f"unsupported display rotation: {frame.rotation}")
            displayed_shape = (
                (frame.width, frame.height, 3)
                if rotation in (90, 270)
                else (frame.height, frame.width, 3)
            )
            converted_bytes = frame.width * frame.height * 3
            retained_bytes = 0 if decoded_frames is None else decoded_frames.nbytes
            if retained_bytes + converted_bytes > max_decoded_array_bytes:
                raise ValueError("video exceeds decoded array byte limit")
            if decoded_frames is not None and displayed_shape != decoded_frames.shape[1:]:
                raise RuntimeError("decoded frames do not match declared video dimensions")
            pixels = frame.to_ndarray(format="rgb24")
            if rotation:
                pixels = np.rot90(pixels, k=rotation // 90)
            if decoded_frames is None:
                peak_array_bytes = pixels.nbytes * (expected_frames + 1)
                if peak_array_bytes > max_decoded_array_bytes:
                    raise ValueError("video exceeds decoded array byte limit")
                decoded_frames = np.empty((expected_frames, *pixels.shape), dtype=np.uint8)
            if decoded_count >= expected_frames:
                raise RuntimeError("decoded frame count exceeds declared video frame count")
            decoded_frames[decoded_count] = pixels
            decoded_count += 1
            timestamps.append(float(frame.pts * frame.time_base))

    if decoded_frames is None:
        raise ValueError("input contains no decoded video frames")
    if decoded_count != len(decoded_frames):
        raise RuntimeError("decoded frame count does not match declared video frame count")
    time_array = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(time_array) <= 0):
        raise RuntimeError("video timestamps are not strictly increasing")
    return VideoFrames(frames=decoded_frames, timestamps=time_array, metadata=metadata)


def write_video(
    path: str | Path,
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    metadata: VideoMetadata | None = None,
) -> None:
    destination = Path(path)
    frame_array = np.asarray(frames)
    time_array = np.asarray(timestamps, dtype=np.float64)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3 or frame_array.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB with shape [time, height, width, 3]")
    if time_array.shape != (len(frame_array),) or not np.all(np.isfinite(time_array)):
        raise ValueError("timestamps must contain one finite value per frame")
    if len(frame_array) == 0 or np.any(np.diff(time_array) <= 0) or time_array[0] < 0:
        raise ValueError("timestamps must be non-negative and strictly increasing")
    if np.max(time_array) >= 2**32:
        raise ValueError("timestamps exceed the distinguishable microsecond range")
    quantized_pts = [round(float(timestamp) / float(_TIME_BASE)) for timestamp in time_array]
    if any(current <= previous for previous, current in zip(quantized_pts, quantized_pts[1:])):
        raise ValueError("timestamps collide at the microsecond timebase")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(destination), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264")
        stream.width = frame_array.shape[2]
        stream.height = frame_array.shape[1]
        stream.pix_fmt = "yuv444p"
        stream.time_base = _TIME_BASE
        stream.codec_context.time_base = _TIME_BASE
        if metadata is not None:
            stream.codec_context.color_range = metadata.color_range
            stream.codec_context.colorspace = metadata.colorspace
            stream.codec_context.color_primaries = metadata.color_primaries
            stream.codec_context.color_trc = metadata.color_trc
        for pixels, pts in zip(frame_array, quantized_pts, strict=True):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = int(pts)
            frame.time_base = _TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
