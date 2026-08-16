from __future__ import annotations

from pathlib import Path

import numpy as np

from flashpatch.core import analyze, repair_with_evidence
from flashpatch.independent import verify_video_independently
from flashpatch.media import VideoFrames, read_video, write_video


def test_independent_decoder_fails_original_passes_repair_and_rejects_disagreement(
    tmp_path: Path,
) -> None:
    frames = np.zeros((12, 12, 16, 3), dtype=np.uint8)
    frames[1::2] = 255
    timestamps = np.array(
        [0.0, 0.08, 0.19, 0.27, 0.39, 0.48, 0.57, 0.69, 0.78, 0.88, 0.97, 1.09],
        dtype=np.float64,
    )
    source = tmp_path / "source.mp4"
    repaired_path = tmp_path / "repaired.mp4"
    write_video(source, frames, timestamps)
    primary_source = read_video(source)
    repair = repair_with_evidence(
        primary_source.frames,
        primary_source.timestamps,
        analyze(primary_source.frames, primary_source.timestamps),
    )
    write_video(
        repaired_path,
        repair.frames,
        primary_source.timestamps,
        metadata=primary_source.metadata,
    )
    primary_repaired = read_video(repaired_path)

    original_check = verify_video_independently(source, primary_source)
    repaired_check = verify_video_independently(repaired_path, primary_repaired)

    assert original_check.decoder == "opencv-videoio"
    assert original_check.decoder_agreement
    assert not original_check.passed
    assert repaired_check.decoder_agreement
    assert repaired_check.passed

    missing_frame_reference = VideoFrames(
        frames=primary_repaired.frames[:-1],
        timestamps=primary_repaired.timestamps[:-1],
        metadata=primary_repaired.metadata,
    )
    disagreement = verify_video_independently(repaired_path, missing_frame_reference)
    assert not disagreement.decoder_agreement
    assert not disagreement.passed
    assert "frame_count" in disagreement.disagreement_reasons
