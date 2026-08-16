from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import flashpatch.external_league as external_league
from flashpatch.external_league import (
    EA_IRIS_MINIMAL_OPENCV_MODULES,
    EA_IRIS_PATTERN_NEGATIVE_FIXTURE,
    EA_IRIS_REQUIRED_CONFORMANCE_ROLES,
    EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES,
    EA_IRIS_REALTIME_SEMANTIC_PROBE_SCHEMA,
    ExternalLeagueError,
    _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP,
    _audit_iris_direct_binary_boundary,
    _audit_iris_source_decoder_timeline,
    _iris_dynamic_closure,
    _iris_semantic_parity_evidence,
    _iris_source_link_flags,
    _load_iris_realtime_semantic_probe,
    _load_iris_direct_rgb_input,
    _materialize_iris_direct_rgb_input,
    _sha256_file,
    materialize_cfr_ffv1,
)


def _canonical_input(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    frames = np.zeros((8, 12, 16, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, :, 0] = index * 17
        frames[index, :, :, 1] = np.arange(16, dtype=np.uint8)
        frames[index, :, :, 2] = np.arange(12, dtype=np.uint8)[:, None]
    source = tmp_path / "renderer.npz"
    np.savez(source, frames=frames, timestamps=np.arange(len(frames)) / 60.0)
    conversion = materialize_cfr_ffv1(source, tmp_path / "conversion", fps=60)
    video = Path(str(conversion["receipt"])).parent / "canonical.ffv1.mkv"
    return video, Path(str(conversion["receipt"])), conversion


def test_iris_direct_link_flags_exclude_native_video_decoder_stack(tmp_path: Path) -> None:
    direct = _iris_source_link_flags(tmp_path, direct_adapter=True)
    source_video = _iris_source_link_flags(tmp_path, direct_adapter=False)

    assert "-Wl,-rpath,$ORIGIN/../lib" in direct
    assert "-lopencv_features2d" in direct
    assert "-lopencv_features2d" in source_video
    assert "-lopencv_highgui" not in source_video
    assert "-l:libtbb.so.12" not in direct
    assert not any(
        flag in direct
        for flag in (
            "-lopencv_videoio",
            "-lopencv_imgcodecs",
            "-lopencv_highgui",
            "-lavformat",
            "-lavcodec",
            "-lavutil",
            "-lswscale",
        )
    )
    assert {
        "-lopencv_videoio",
        "-lopencv_imgcodecs",
        "-lavformat",
        "-lavcodec",
        "-lavutil",
        "-lswscale",
    } <= set(source_video)
    assert not {
        "-lopencv_highgui",
    } & set(source_video)
    runtime_search_flags = [
        flag for flag in direct + source_video
        if flag.startswith("-Wl,-rpath,")
    ]
    assert runtime_search_flags == [
        "-Wl,-rpath,$ORIGIN/../lib",
        "-Wl,-rpath,$ORIGIN/../lib",
    ]


def test_iris_direct_child_source_has_only_raw_rgb_input_boundary() -> None:
    assert "cv::VideoCapture" not in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "opencv2/videoio" not in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "ReadOnlyMappedFile" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "O_NOFOLLOW" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "RealTimeInit" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "AnalyseFrame" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "runtime_timing_eligible\", false" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP
    assert "direct RGB input changed while IRIS consumed it" in _EA_IRIS_SOURCE_FRAME_ADAPTER_CPP


def test_iris_direct_rgb_input_replays_exact_frozen_ffmpeg_command(tmp_path: Path) -> None:
    video, conversion, _ = _canonical_input(tmp_path)
    root = tmp_path / "direct"
    root.mkdir()
    frozen = _materialize_iris_direct_rgb_input(video, conversion, root)
    raw = Path(str(frozen["raw_rgb"]["path"]))
    timeline = Path(str(frozen["timeline"]["path"]))

    reopened = _load_iris_direct_rgb_input(
        raw,
        timeline,
        video=video,
        conversion=conversion,
        expected_raw_sha256=str(frozen["raw_rgb"]["sha256"]),
        expected_timeline_sha256=str(frozen["timeline"]["sha256"]),
    )

    assert reopened == frozen["payload"]
    assert reopened["decoder"]["command"][-4:] == ["rawvideo", "-pix_fmt", "rgb24", "-"]
    assert reopened["raw_rgb"]["sha256"] == reopened["renderer_source"]["rgb_sha256"]


def test_iris_direct_rgb_input_rejects_coherently_rehashed_decoder_argv(tmp_path: Path) -> None:
    video, conversion, _ = _canonical_input(tmp_path)
    root = tmp_path / "direct"
    root.mkdir()
    frozen = _materialize_iris_direct_rgb_input(video, conversion, root)
    raw = Path(str(frozen["raw_rgb"]["path"]))
    timeline = Path(str(frozen["timeline"]["path"]))
    payload = json.loads(timeline.read_text(encoding="utf-8"))
    payload["decoder"]["command"].insert(-1, "-an")
    timeline.chmod(0o644)
    timeline.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timeline.chmod(0o444)

    with pytest.raises(ExternalLeagueError, match="fresh canonical replay"):
        _load_iris_direct_rgb_input(
            raw,
            timeline,
            video=video,
            conversion=conversion,
            expected_raw_sha256=_sha256_file(raw),
            expected_timeline_sha256=_sha256_file(timeline),
        )


@pytest.mark.parametrize("mutation", ["truncate", "append"])
def test_iris_direct_rgb_input_rejects_byte_count_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    video, conversion, _ = _canonical_input(tmp_path)
    root = tmp_path / "direct"
    root.mkdir()
    frozen = _materialize_iris_direct_rgb_input(video, conversion, root)
    raw = Path(str(frozen["raw_rgb"]["path"]))
    timeline = Path(str(frozen["timeline"]["path"]))
    data = raw.read_bytes()
    raw.chmod(0o644)
    raw.write_bytes(data[:-1] if mutation == "truncate" else data + b"\x00")
    raw.chmod(0o444)

    with pytest.raises(ExternalLeagueError, match="fresh canonical replay"):
        _load_iris_direct_rgb_input(
            raw,
            timeline,
            video=video,
            conversion=conversion,
            expected_raw_sha256=_sha256_file(raw),
            expected_timeline_sha256=_sha256_file(timeline),
        )


def test_iris_strict_dynamic_closure_rejects_non_system_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "adapter"
    binary.write_bytes(b"elf")
    library_root = tmp_path / "sysroot" / "usr" / "lib" / "x86_64-linux-gnu"
    library_root.mkdir(parents=True)
    host_library = tmp_path / "host" / "libforbidden.so.1"
    host_library.parent.mkdir()
    host_library.write_bytes(b"host")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=f"libforbidden.so.1 => {host_library} (0x1)\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExternalLeagueError, match="non-system host libraries"):
        _iris_dynamic_closure(binary, ldd=Path("/usr/bin/ldd"), library_root=library_root)


def test_iris_direct_elf_gate_rejects_videoio_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        (
            b" 0x1 (NEEDED) Shared library: [libopencv_videoio.so.408]\n",
            b"Symbol table contains no decoder symbol\n",
        )
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args[0], 0, stdout=next(outputs), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ExternalLeagueError, match="forbidden video decoder dependency"):
        _audit_iris_direct_binary_boundary(tmp_path / "adapter", readelf=Path("/usr/bin/readelf"))


def test_iris_minimal_module_and_conformance_boundaries_are_complete() -> None:
    assert EA_IRIS_MINIMAL_OPENCV_MODULES == (
        "opencv_core",
        "opencv_imgproc",
        "opencv_features2d",
        "opencv_imgcodecs",
        "opencv_videoio",
    )
    assert set(EA_IRIS_REQUIRED_CONFORMANCE_ROLES) == {
        "SAFE_CONTROL",
        "LUMINANCE_FLASH",
        "RED_FLASH",
        "PATTERN",
    }
    assert EA_IRIS_REQUIRED_TEMPORAL_BOUNDARIES == {
        "PATTERN_PERSISTENCE_FRAMES": (29, 30, 31),
        "ONE_SECOND_FLASH_FRAMES": (59, 60, 61),
        "FOUR_SECOND_EXTENDED_FRAMES": (239, 240, 241),
        "FIVE_SECOND_WINDOW_FRAMES": (299, 300, 301),
        "TRANSITION_COUNT": (3, 4, 6, 7),
    }
    assert EA_IRIS_PATTERN_NEGATIVE_FIXTURE["path"].endswith("20stripes.png")
    assert EA_IRIS_PATTERN_NEGATIVE_FIXTURE["frame_count"] == 31


def test_iris_terminal_agreement_cannot_hide_pattern_timing_mismatch() -> None:
    direct = [
        {"frame_index": index, "luminance_result": 0, "red_result": 0, "pattern_result": 1}
        for index in range(31)
    ]
    source = [
        {
            "frame_index": index,
            "luminance_result": 0,
            "red_result": 0,
            "pattern_result": 1 if index >= 29 else 0,
        }
        for index in range(31)
    ]

    evidence = _iris_semantic_parity_evidence(
        direct,
        source,
        direct_prediction="HAZARDOUS",
        direct_warning=False,
        source_prediction="HAZARDOUS",
        source_warning=False,
    )

    assert evidence["verified"] is False
    assert evidence["comparison"] == {
        "terminal_agreement": True,
        "frame_categories_exact": False,
        "mismatch_frame_indices": list(range(29)),
        "first_direct_pattern_fail_frame": 0,
        "first_source_video_pattern_fail_frame": 29,
        "normalization_applied": False,
    }


def test_iris_semantic_mismatch_receipt_cannot_authorize_participant(tmp_path: Path) -> None:
    receipt = tmp_path / "probe.json"
    receipt.write_text(
        json.dumps({
            "schema": EA_IRIS_REALTIME_SEMANTIC_PROBE_SCHEMA,
            "identity": "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC",
            "classification": "MANDATORY_NEGATIVE_CONFORMANCE_NOT_SCORING",
            "status": "SEMANTIC_MISMATCH_NOT_VERIFIED",
            "direct_participant_authorized": True,
            "scoreable": False,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ExternalLeagueError, match="claim boundary"):
        _load_iris_realtime_semantic_probe(receipt)


def test_iris_decoder_audit_requires_semantic_receipt_and_mismatch_never_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_path = tmp_path / "child.json"
    conformance_path = tmp_path / "conformance.json"
    semantic_path = tmp_path / "semantic.json"
    build_path = tmp_path / "build.json"
    for path in (child_path, conformance_path, semantic_path, build_path):
        path.write_text("{}\n", encoding="utf-8")
    video = tmp_path / "canonical.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"video")
    conversion.write_text("{}\n", encoding="utf-8")
    contract = {
        "canonical_video": {"sha256": _sha256_file(video)},
        "conversion_receipt": {"sha256": _sha256_file(conversion)},
        "frame_count": 1,
        "fps": 60,
        "frame_map": [{
            "frame_index": 0,
            "cfr_timestamp_us": 0,
            "renderer_timestamp_us": 0,
            "rgb_sha256": "a" * 64,
        }],
        "frame_map_sha256": "b" * 64,
    }
    build_ref = {"path": str(build_path), "sha256": _sha256_file(build_path)}
    run = {
        "input": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
        "build_receipt": build_ref,
    }
    raw = {"frames": [{
        "frame_index": 0,
        "cfr_timestamp_us_rounded": 0,
        "renderer_timestamp_us": 0,
        "rgb_sha256": "a" * 64,
    }]}
    conformance = {
        "source_build": build_ref,
        "status": "LOCAL_CONFORMANCE_MATCH",
        "local_fixture_match": True,
    }
    build = {
        "binaries": {
            "source_frame_adapter": {"sha256": "c" * 64},
            "source_video_oracle": {"sha256": "d" * 64},
        },
    }
    semantic = {
        "status": "SEMANTIC_MISMATCH_NOT_VERIFIED",
        "direct_participant_authorized": False,
        "scoreable": False,
        "source_build": {
            **build_ref,
            "direct_binary_sha256": "c" * 64,
            "source_video_binary_sha256": "d" * 64,
        },
        "comparison": {
            "frame_categories_exact": False,
            "terminal_agreement": True,
            "normalization_applied": False,
        },
    }
    monkeypatch.setattr(
        external_league,
        "_load_iris_source_adapter_run",
        lambda *args, **kwargs: (run, raw, {"runtime_timing_eligible": False}, child_path),
    )
    monkeypatch.setattr(
        external_league,
        "_load_iris_source_conformance_receipt",
        lambda *args, **kwargs: (conformance, conformance_path),
    )
    monkeypatch.setattr(
        external_league,
        "_load_iris_source_build_receipt",
        lambda *args, **kwargs: (build, build_path),
    )

    with pytest.raises(ExternalLeagueError, match="semantic conformance receipt is required"):
        _audit_iris_source_decoder_timeline(
            child_path, {}, video, conversion, contract, conformance_path, None,
        )

    monkeypatch.setattr(
        external_league,
        "_load_iris_realtime_semantic_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(ExternalLeagueError("forged semantic receipt")),
    )
    with pytest.raises(ExternalLeagueError, match="forged semantic receipt"):
        _audit_iris_source_decoder_timeline(
            child_path, {}, video, conversion, contract, conformance_path, semantic_path,
        )

    monkeypatch.setattr(
        external_league,
        "_load_iris_realtime_semantic_probe",
        lambda *args, **kwargs: (semantic, semantic_path),
    )
    result = _audit_iris_source_decoder_timeline(
        child_path, {}, video, conversion, contract, conformance_path, semantic_path,
    )
    assert result["parity_status"] == "NOT_VERIFIED"
    assert result["comparison_eligible"] is False
    assert result["semantic_conformance"] == {
        "status": "SEMANTIC_MISMATCH_NOT_VERIFIED",
        "frame_categories_exact": False,
        "terminal_agreement": True,
        "normalization_applied": False,
    }
