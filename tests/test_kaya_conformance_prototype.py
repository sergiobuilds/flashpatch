from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import flashpatch.external_league as external_league
from flashpatch.external_league import (
    DIRECT_DETECTOR_POPULATION,
    KAYA_CONFORMANCE_PROTOTYPE_SCHEMA,
    KAYA_DIRECT_PARTICIPANT_ID,
    KAYA_DIRECT_INPUT_SCHEMA,
    KAYA_NATURAL_CASE_PARITY_SCHEMA,
    KAYA_PARTICIPANT_CONFORMANCE_SCHEMA,
    KAYA_PYTHON_SHA256,
    KAYA_PROTOTYPE_ID,
    KAYA_REQUIRED_FIXTURE_IDS,
    KAYA_REQUIRED_SOURCE_HASHES,
    KAYA_SOURCE_REVISION,
    KAYA_SOURCE_TREE,
    ExternalLeagueError,
    _KAYA_CONFORMANCE_CHILD_SCRIPT,
    _canonical_json_sha256,
    _finalize_kaya_verification,
    _kaya_semantic_conformance_failures,
    _kaya_natural_case_exact_failures,
    _kaya_pre_replay_status,
    _load_kaya_direct_input_manifest,
    _materialize_kaya_direct_input,
    _sha256_bytes,
    _sha256_file,
    _verify_kaya_distribution_closure,
    _verify_kaya_replay_storage_independence,
    materialize_cfr_ffv1,
    verify_kaya_conformance_prototype,
    verify_kaya_participant_conformance_receipt,
    verify_kaya_natural_case_parity_receipt,
    execute_kaya_natural_case_parity,
    write_kaya_participant_conformance_receipt,
)


def _direct_input(tmp_path: Path) -> tuple[Path, Path, Path]:
    frames = np.zeros((4, 8, 12, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    frames[2, :, :, 1] = 127
    frames[3, :, :, 2] = 63
    source = tmp_path / "renderer.npz"
    np.savez(source, frames=frames, timestamps=np.arange(len(frames)) / 60.0)
    conversion = materialize_cfr_ffv1(source, tmp_path / "conversion", fps=60)
    conversion_path = Path(str(conversion["receipt"])).resolve()
    video = conversion_path.parent / "canonical.ffv1.mkv"
    direct = _materialize_kaya_direct_input(
        "safe", video, conversion_path, tmp_path / "direct"
    )
    return Path(str(direct["manifest"])), video, conversion_path


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_kaya_historical_and_participant_identities_are_distinct_and_pinned() -> None:
    assert KAYA_PROTOTYPE_ID not in DIRECT_DETECTOR_POPULATION
    assert KAYA_DIRECT_PARTICIPANT_ID in DIRECT_DETECTOR_POPULATION
    assert KAYA_DIRECT_PARTICIPANT_ID == "KAYA_PSE_DETECTION_CORRECTION_0776EA3"
    assert KAYA_SOURCE_REVISION == "0776ea3e6949a62d5becb8027a2765961b515793"
    assert KAYA_SOURCE_TREE == "34bef92fcc2bbd7c2e779475b84094493ae23aa1"
    assert KAYA_REQUIRED_SOURCE_HASHES["LICENSE"] == (
        "cd18b47f83e5cb4640272cc95e0144206be9ed55d8458573eba3cf0b49534da8"
    )
    assert "UNSCORED_CONFORMANCE_ONLY" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert '"scoreable": False' in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert '"population_authorized": False' in _KAYA_CONFORMANCE_CHILD_SCRIPT


def test_kaya_child_calls_unmodified_upstream_pipeline_without_main_or_moviepy() -> None:
    assert "w3c.w3c_guideline.analyse_file(" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "CustomVideo.frame_intervals" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "direct_frames[..., ::-1]" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "rgb_to_bgr_channel_reverse_once" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "main.py" not in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "moviepy" not in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "sys.flags.no_site != 1" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert 'sys.path[:] = [str(checkout), *standard_library_paths, str(site_packages)]' in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "DISTRIBUTION_CLOSURES" in _KAYA_CONFORMANCE_CHILD_SCRIPT
    assert "CALLABLE_SOURCE_HASHES" in _KAYA_CONFORMANCE_CHILD_SCRIPT


def test_kaya_participant_receipt_promotes_only_verified_unscored_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prototype = tmp_path / "prototype.json"
    prototype.write_text("{}\n", encoding="utf-8")
    conformance = {
        "native_api": "unmodified_GuidelineProcess.analyse_file",
        "direct_api": "same_unmodified_GuidelineProcess.analyse_file_with_receipt_bound_DirectCapture",
        "rgb_to_bgr": "exactly_one_channel_reverse_before_upstream_consumption",
        "result_equality": "exact_raw_arrays_and_exact_upstream_interval_tuples_no_normalization",
        "eof_open_both_bug": "preserved_unmodified",
        "failures": [],
    }
    primary_environment = {
        "root": str(tmp_path / "primary"),
        "python": str(tmp_path / "primary" / "bin" / "python"),
        "python_sha256": KAYA_PYTHON_SHA256,
        "required_distributions": dict(external_league.KAYA_REQUIRED_DISTRIBUTIONS),
    }
    replay_environment = {
        "root": str(tmp_path / "replay"),
        "python": str(tmp_path / "replay" / "bin" / "python"),
        "python_sha256": KAYA_PYTHON_SHA256,
    }
    storage = {
        "primary_identity_count": 20,
        "replay_identity_count": 20,
        "primary_counts": {"dependencies": 10, "runtime": 10},
        "replay_counts": {"dependencies": 10, "runtime": 10},
        "shared_inode_count": 0,
    }

    def verified_prototype(*args: object, **kwargs: object) -> dict[str, object]:
        fresh_replay = kwargs.get("fresh_replay", True)
        return {
            "identity": KAYA_PROTOTYPE_ID,
            "environment": primary_environment,
            "conformance": conformance,
            "scoreable": False,
            "verification_status": "VERIFIED" if fresh_replay else "NOT_VERIFIED",
            "fresh_replay_verified": bool(fresh_replay),
            "replay_environment": replay_environment,
            "dependency_storage_independence": storage,
        }

    monkeypatch.setattr(
        external_league,
        "verify_kaya_conformance_prototype",
        verified_prototype,
    )
    destination = tmp_path / "participant.json"
    written = write_kaya_participant_conformance_receipt(
        prototype,
        tmp_path / "replay" / "bin" / "python",
        destination,
    )

    assert written["identity"] == KAYA_DIRECT_PARTICIPANT_ID
    assert written["prototype_identity"] == KAYA_PROTOTYPE_ID
    assert written["status"] == "VERIFIED"
    assert written["scoreable"] is False
    assert written["unscored_population_authorized"] is True
    assert written["external_claim_authorized"] is False
    reopened = verify_kaya_participant_conformance_receipt(destination)
    assert reopened["schema"] == KAYA_PARTICIPANT_CONFORMANCE_SCHEMA


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scoreable", True),
        ("external_claim_authorized", True),
        ("identity", KAYA_PROTOTYPE_ID),
    ],
)
def test_kaya_participant_receipt_rejects_score_claim_or_identity_conflation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "participant.json"
    path.write_text(
        json.dumps(
            {
                "schema": KAYA_PARTICIPANT_CONFORMANCE_SCHEMA,
                "identity": KAYA_DIRECT_PARTICIPANT_ID,
                "prototype_identity": KAYA_PROTOTYPE_ID,
                "classification": "UNSCORED_DIRECT_PARTICIPANT_CONFORMANCE",
                "upstream": {
                    "repository_url": external_league.KAYA_REPOSITORY_URL,
                    "revision": KAYA_SOURCE_REVISION,
                    "tree": KAYA_SOURCE_TREE,
                },
                "prototype_receipt": {"path": "missing", "sha256": "0" * 64},
                "verification": {},
                "status": "VERIFIED",
                "claim_status": "NOT_SCOREABLE",
                "scoreable": False,
                "unscored_population_authorized": True,
                "external_claim_authorized": False,
                "scoreable_blockers": list(
                    external_league._KAYA_PARTICIPANT_SCOREABLE_BLOCKERS
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _rewrite(path, payload)

    with pytest.raises(ExternalLeagueError, match="identity or claim boundary"):
        verify_kaya_participant_conformance_receipt(path)


def test_kaya_direct_manifest_accepts_only_exact_canonical_rgb_60_cfr(tmp_path: Path) -> None:
    manifest, video, conversion = _direct_input(tmp_path)
    loaded = _load_kaya_direct_input_manifest(
        manifest, expected_video=video, expected_conversion=conversion
    )
    assert loaded["schema"] == KAYA_DIRECT_INPUT_SCHEMA
    assert loaded["fps"] == 60
    assert loaded["dtype"] == "uint8"
    assert loaded["pixel_format"] == "rgb24"


@pytest.mark.parametrize("bad_fps", [59.94, 59, 60.0, True])
def test_kaya_direct_manifest_rejects_non_exact_integer_60_cfr(
    tmp_path: Path,
    bad_fps: object,
) -> None:
    manifest, video, conversion = _direct_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["fps"] = bad_fps
    _rewrite(manifest, payload)
    with pytest.raises(ExternalLeagueError, match="exact 60 CFR"):
        _load_kaya_direct_input_manifest(
            manifest, expected_video=video, expected_conversion=conversion
        )


@pytest.mark.parametrize("attack", ["missing", "duplicate", "timestamp", "pixel-format"])
def test_kaya_direct_manifest_rejects_coherently_rehashed_ledger_attacks(
    tmp_path: Path,
    attack: str,
) -> None:
    manifest, video, conversion = _direct_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    ledger = payload["ledger"]
    if attack == "missing":
        ledger.pop(1)
    elif attack == "duplicate":
        ledger[1]["index"] = 0
    elif attack == "timestamp":
        ledger[1]["cfr_timestamp_us"] += 1
    else:
        ledger[1]["pixel_format"] = "bgr24"
    payload["ledger_sha256"] = _canonical_json_sha256(ledger)
    _rewrite(manifest, payload)
    with pytest.raises(ExternalLeagueError, match="ledger"):
        _load_kaya_direct_input_manifest(
            manifest, expected_video=video, expected_conversion=conversion
        )


def test_kaya_direct_manifest_rejects_malformed_dtype_even_when_hashes_are_updated(
    tmp_path: Path,
) -> None:
    manifest, video, conversion = _direct_input(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    frames_path = Path(payload["frames"])
    frames = np.load(frames_path, allow_pickle=False).astype(np.float32)
    np.save(frames_path, frames, allow_pickle=False)
    payload["frames_file_sha256"] = _sha256_file(frames_path)
    payload["raw_rgb_sha256"] = _sha256_bytes(np.ascontiguousarray(frames).tobytes())
    payload["dtype"] = "float32"
    _rewrite(manifest, payload)
    with pytest.raises(ExternalLeagueError, match="uint8 RGB"):
        _load_kaya_direct_input_manifest(
            manifest, expected_video=video, expected_conversion=conversion
        )


def _result(
    general: list[float],
    red: list[float],
    intervals: list[list[object]],
) -> dict[str, object]:
    return {
        "raw": {"General Flashes": general, "Red Flashes": red},
        "interval_tuples": intervals,
        "interval_semantics": "unmodified_CustomVideo.frame_intervals_including_eof_open_both_bug",
    }


def _semantic_rows() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for fixture_id in KAYA_REQUIRED_FIXTURE_IDS:
        frame_count = {
            "safe": 61,
            "rgb-channel-trap": 72,
            "flash-threshold": 90,
            "history-59": 59,
            "history-60": 60,
            "history-61": 61,
            "letterbox": 61,
            "state-reuse": 61,
        }[fixture_id]
        general = [0.0] * frame_count
        red = [0.0] * frame_count
        intervals: list[list[object]] = []
        if fixture_id == "rgb-channel-trap":
            red[28:] = [4.0] * (frame_count - 28)
            intervals = [["red", 28, frame_count]]
        elif fixture_id in {"flash-threshold", "history-59", "history-60", "history-61", "letterbox"}:
            start = 28 if fixture_id != "letterbox" else 35
            if fixture_id == "flash-threshold":
                general[24:28] = [3.0] * 4
                general[start:] = [3.5] * (frame_count - start)
            else:
                general[start:] = [4.0] * (frame_count - start)
            intervals = [["general", start, frame_count]]
        elif fixture_id == "state-reuse":
            general[-1] = 4.0
            red[-1] = 4.0
            # Upstream intentionally omits an EOF-open `both` interval.
            intervals = []
        result = _result(general, red, intervals)
        ledger = [{"index": index, "bgr_sha256": str(index)} for index in range(frame_count)]
        native = {"results": [copy.deepcopy(result)], "capture_runs": [{"ledger": copy.deepcopy(ledger)}]}
        direct_results = [copy.deepcopy(result), copy.deepcopy(result)] if fixture_id == "state-reuse" else [copy.deepcopy(result)]
        direct = {"results": direct_results, "capture_runs": [{"ledger": copy.deepcopy(ledger)}]}
        rows[fixture_id] = {
            "input": {"shape": [frame_count, 36, 180, 3] if fixture_id == "letterbox" else [frame_count, 72, 96, 3]},
            "native_output": native,
            "direct_output": direct,
        }
    return rows


def test_kaya_semantic_gate_accepts_only_unmodified_exact_native_direct_evidence() -> None:
    assert _kaya_semantic_conformance_failures(_semantic_rows()) == []


def test_kaya_semantic_gate_rejects_channel_swap_before_consumption() -> None:
    rows = _semantic_rows()
    rows["rgb-channel-trap"]["direct_output"]["capture_runs"][0]["ledger"][0]["bgr_sha256"] = "swapped"
    assert "native_direct_preconsumption_bgr_mismatch:rgb-channel-trap" in _kaya_semantic_conformance_failures(rows)


def test_kaya_semantic_gate_rejects_raw_array_drift_without_normalization() -> None:
    rows = _semantic_rows()
    rows["flash-threshold"]["direct_output"]["results"][0]["raw"]["General Flashes"][30] = 4.5
    assert "native_direct_exact_result_mismatch:flash-threshold" in _kaya_semantic_conformance_failures(rows)


def test_kaya_semantic_gate_rejects_interval_normalization() -> None:
    rows = _semantic_rows()
    rows["state-reuse"]["direct_output"]["results"][0]["interval_tuples"] = [["both", 60, 61]]
    failures = _kaya_semantic_conformance_failures(rows)
    assert "native_direct_exact_result_mismatch:state-reuse" in failures
    assert "state_reuse_changed_upstream_result" in failures


def test_kaya_semantic_gate_rejects_state_reuse_drift() -> None:
    rows = _semantic_rows()
    rows["state-reuse"]["direct_output"]["results"][1]["raw"]["Red Flashes"][-1] = 3.5
    failures = _kaya_semantic_conformance_failures(rows)
    assert "native_direct_exact_result_mismatch:state-reuse" in failures
    assert "state_reuse_changed_upstream_result" in failures


def test_kaya_semantic_gate_requires_verbatim_eof_open_both_bug() -> None:
    rows = _semantic_rows()
    rows["state-reuse"]["native_output"]["results"][0]["interval_tuples"] = [["both", 60, 61]]
    rows["state-reuse"]["direct_output"]["results"][0]["interval_tuples"] = [["both", 60, 61]]
    rows["state-reuse"]["direct_output"]["results"][1]["interval_tuples"] = [["both", 60, 61]]
    assert "eof_open_both_upstream_bug_not_exercised_verbatim" in _kaya_semantic_conformance_failures(rows)


def test_kaya_natural_case_exact_gate_accepts_existing_conformance_shape() -> None:
    rows = _semantic_rows()
    native = rows["flash-threshold"]["native_output"]
    direct = rows["flash-threshold"]["direct_output"]
    assert _kaya_natural_case_exact_failures(native, direct) == []
    assert KAYA_NATURAL_CASE_PARITY_SCHEMA.endswith("-v1")


@pytest.mark.parametrize("attack", ["channel-ledger", "raw-array", "interval"])
def test_kaya_natural_case_exact_gate_rejects_channel_or_upstream_output_drift(
    attack: str,
) -> None:
    rows = _semantic_rows()
    native = rows["flash-threshold"]["native_output"]
    direct = copy.deepcopy(rows["flash-threshold"]["direct_output"])
    if attack == "channel-ledger":
        direct["capture_runs"][0]["ledger"][0]["bgr_sha256"] = "wrong-channel-order"
        expected = "native_direct_preconsumption_bgr_ledger_mismatch"
    elif attack == "raw-array":
        direct["results"][0]["raw"]["General Flashes"][30] = 4.5
        expected = "native_direct_raw_arrays_or_intervals_mismatch"
    else:
        direct["results"][0]["interval_tuples"] = [["general", 31, 90]]
        expected = "native_direct_raw_arrays_or_intervals_mismatch"
    assert expected in _kaya_natural_case_exact_failures(native, direct)


def _natural_case_output() -> dict[str, object]:
    return {
        "capture_runs": [{"ledger": [{"index": 0, "bgr_sha256": "frame-0"}]}],
        "results": [{"raw": {"General Flashes": [0.0], "Red Flashes": [0.0]}, "interval_tuples": []}],
    }


def _natural_case_process(
    root: Path,
    *,
    mode: str,
    command: list[str],
) -> dict[str, object]:
    work = root / mode
    work.mkdir(parents=True)
    streams: dict[str, dict[str, object]] = {}
    for label, filename in (("stdout", "stdout.bin"), ("stderr", "stderr.bin")):
        stream = work / filename
        stream.write_bytes(b"")
        streams[label] = {"path": str(stream), "sha256": _sha256_file(stream)}
    child_output = work / "child-output.json"
    child_output.write_text("{}\n", encoding="utf-8")
    environment = {"PYTHONPATH": "frozen"}
    return {
        "command": command,
        "command_sha256": _canonical_json_sha256(command),
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
        "working_directory": str(work),
        "started_monotonic_ns": 10,
        "finished_monotonic_ns": 20,
        "wall_time_ns": 10,
        "timeout_seconds": 120,
        "timed_out": False,
        "exit_code": 0,
        **streams,
        "output": {"path": str(child_output), "exists": True, "sha256": _sha256_file(child_output)},
    }


def _natural_case_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    """Build a fully hash-bound receipt while mocking only unavailable Kaya execution."""
    root = tmp_path / "receipt-root"
    root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"frozen-python")
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    participant = tmp_path / "participant.json"
    direct = root / "direct-input.json"
    for path in (video, conversion, participant, direct):
        path.write_text("{}\n", encoding="utf-8")
    contract = {
        "canonical_video": {"path": str(video), "sha256": _sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": _sha256_file(conversion)},
        "decoder_contract": "frozen-test-contract",
        "fps": 60,
        "frame_count": 1,
        "shape": [1, 1, 1, 3],
        "frame_map": [
            {
                "frame_index": 0,
                "cfr_timestamp_us": 0,
                "renderer_timestamp_us": 0,
                "rgb_sha256": "0" * 64,
            }
        ],
        "frame_map_sha256": "1" * 64,
    }
    provenance = {"repository_url": "https://example.invalid/kaya", "revision": "pinned", "tree": "tree"}
    native_output = _natural_case_output()
    direct_output = copy.deepcopy(native_output)
    monkeypatch.setattr(external_league, "KAYA_PYTHON_SHA256", _sha256_file(python))
    monkeypatch.setattr(external_league, "verify_kaya_participant_conformance_receipt", lambda _: {})
    monkeypatch.setattr(external_league, "_audit_kaya_source_checkout", lambda _: provenance)
    monkeypatch.setattr(external_league, "_canonical_decoder_timeline_contract", lambda *_: ({}, contract))
    monkeypatch.setattr(external_league, "_load_kaya_direct_input_manifest", lambda *_, **__: {})
    monkeypatch.setattr(external_league, "_load_kaya_child_output", lambda _: {})
    monkeypatch.setattr(
        external_league,
        "_validate_kaya_child_output",
        lambda _, *, mode, **__: native_output if mode == "native" else direct_output,
    )
    script_hash = _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8"))
    native_command = [
        str(python), "-S", "-X", f"pycache_prefix={root / 'native' / 'pycache'}", "-c",
        _KAYA_CONFORMANCE_CHILD_SCRIPT, "native", str(checkout), str(video), "-",
        str(root / "native" / "child-output.json"), script_hash, "1",
    ]
    direct_command = [
        str(python), "-S", "-X", f"pycache_prefix={root / 'direct' / 'pycache'}", "-c",
        _KAYA_CONFORMANCE_CHILD_SCRIPT, "direct", str(checkout), str(video), str(direct),
        str(root / "direct" / "child-output.json"), script_hash, "1",
    ]
    receipt = {
        "schema": KAYA_NATURAL_CASE_PARITY_SCHEMA,
        "identity": KAYA_DIRECT_PARTICIPANT_ID,
        "prototype_identity": KAYA_PROTOTYPE_ID,
        "classification": "NATURAL_CASE_INPUT_PARITY_NOT_SCORING",
        "participant_conformance": {"path": str(participant), "sha256": _sha256_file(participant)},
        "source_checkout": str(checkout),
        "upstream": provenance,
        "runtime": {"python": str(python), "python_sha256": _sha256_file(python)},
        "canonical_contract": contract,
        "direct_input": {"path": str(direct), "sha256": _sha256_file(direct)},
        "native_process": _natural_case_process(root, mode="native", command=native_command),
        "direct_process": _natural_case_process(root, mode="direct", command=direct_command),
        "native_output": native_output,
        "direct_output": direct_output,
        "parity_failures": [],
        "status": "VERIFIED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "natural_case_parity_is_not_an_independent_execution_witness",
            "independent_gold_receipts_missing",
            "equal_budget_three_repeat_receipts_missing",
        ],
    }
    path = root / "kaya-natural-case-parity-receipt.json"
    _rewrite(path, receipt)
    return path, receipt


def _decoder_router_process_receipt(
    path: Path,
    *,
    name: str,
    contract: dict[str, object],
) -> Path:
    observation = {"prediction": "SAFE", "hazard_frame_indices": []}
    payload = {
        "schema": (
            "flashpatch-l7-direct-detector-run-v1"
            if name == "FlashPatch"
            else "flashpatch-external-comparator-run-v1"
        ),
        "comparator": {"name": name},
        "conversion_receipt": dict(contract["conversion_receipt"]),
        "status": "PROCESS_VALID",
        "observation" if name == "FlashPatch" else "parsed_observation": observation,
    }
    _rewrite(path, payload)
    return path


def _patch_decoder_router_process_audits(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    observation = {"prediction": "SAFE", "hazard_frame_indices": []}
    monkeypatch.setattr(
        external_league,
        "_reopen_child_normalized_observation",
        lambda _path, _child, name, _sha: calls.append(f"normalize:{name}") or observation,
    )

    def decoder(name: str, *, comparison_eligible: bool) -> dict[str, object]:
        calls.append(f"audit:{name}")
        return {
            "parity_status": "VERIFIED",
            "parity_reason": None,
            "timestamp_precision_us": 1,
            "comparison_eligible": comparison_eligible,
        }

    monkeypatch.setattr(
        external_league,
        "_audit_flashpatch_decoder_timeline",
        lambda *_: decoder("FlashPatch", comparison_eligible=True),
    )
    monkeypatch.setattr(
        external_league,
        "_audit_tooflashy_decoder_timeline",
        lambda *_: decoder("TooFlashy", comparison_eligible=False),
    )


def test_decoder_router_selects_kaya_natural_receipt_without_tooflashy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kaya_path, kaya = _natural_case_receipt(tmp_path, monkeypatch)
    contract = kaya["canonical_contract"]
    assert isinstance(contract, dict)
    flashpatch = _decoder_router_process_receipt(
        tmp_path / "flashpatch.json", name="FlashPatch", contract=contract
    )
    tooflashy = _decoder_router_process_receipt(
        tmp_path / "tooflashy.json", name="TooFlashy", contract=contract
    )
    calls: list[str] = []
    _patch_decoder_router_process_audits(monkeypatch, calls)
    real_verify = verify_kaya_natural_case_parity_receipt

    def verify_kaya(path: Path | str) -> dict[str, object]:
        calls.append("verify:Kaya")
        return real_verify(path)

    monkeypatch.setattr(
        external_league,
        "verify_kaya_natural_case_parity_receipt",
        verify_kaya,
    )

    result = external_league.verify_decoder_timeline_parity(
        [flashpatch, kaya_path, tooflashy],
        Path(str(contract["canonical_video"]["path"])),
        Path(str(contract["conversion_receipt"]["path"])),
    )

    rows = {row["comparator"]: row for row in result["comparators"]}
    assert calls == [
        "normalize:FlashPatch",
        "audit:FlashPatch",
        "verify:Kaya",
        "normalize:TooFlashy",
        "audit:TooFlashy",
    ]
    assert rows[KAYA_DIRECT_PARTICIPANT_ID]["status"] == "VERIFIED"
    assert rows[KAYA_DIRECT_PARTICIPANT_ID]["decoder_timeline"]["comparison_eligible"] is False
    assert rows["TooFlashy"]["decoder_timeline"]["comparison_eligible"] is False
    assert result["status"] == "NOT_VERIFIED"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert not {"rank", "ranking", "winner", "scores"}.intersection(result)


def test_decoder_router_rejects_generic_process_receipt_for_kaya(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, kaya = _natural_case_receipt(tmp_path, monkeypatch)
    contract = kaya["canonical_contract"]
    assert isinstance(contract, dict)
    flashpatch = _decoder_router_process_receipt(
        tmp_path / "flashpatch.json", name="FlashPatch", contract=contract
    )
    generic_kaya = _decoder_router_process_receipt(
        tmp_path / "generic-kaya.json",
        name=KAYA_DIRECT_PARTICIPANT_ID,
        contract=contract,
    )
    tooflashy = _decoder_router_process_receipt(
        tmp_path / "tooflashy.json", name="TooFlashy", contract=contract
    )
    calls: list[str] = []
    _patch_decoder_router_process_audits(monkeypatch, calls)

    result = external_league.verify_decoder_timeline_parity(
        [flashpatch, generic_kaya, tooflashy],
        Path(str(contract["canonical_video"]["path"])),
        Path(str(contract["conversion_receipt"]["path"])),
    )

    assert any(
        failure.startswith(f"run_receipt_kind_invalid:{KAYA_DIRECT_PARTICIPANT_ID}:")
        for failure in result["failures"]
    )
    assert f"run_receipt_missing:{KAYA_DIRECT_PARTICIPANT_ID}" in result["failures"]
    assert all("KAYA" not in call for call in calls)
    assert result["status"] == "NOT_VERIFIED"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False


@pytest.mark.parametrize(
    ("mutation", "failure_text"),
    [
        ("claim", "identity or claim boundary"),
        ("contract", "canonical contract drifted"),
    ],
)
def test_decoder_router_rejects_kaya_tamper_without_promoting_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failure_text: str,
) -> None:
    kaya_path, kaya = _natural_case_receipt(tmp_path, monkeypatch)
    contract = kaya["canonical_contract"]
    assert isinstance(contract, dict)
    flashpatch = _decoder_router_process_receipt(
        tmp_path / "flashpatch.json", name="FlashPatch", contract=contract
    )
    tooflashy = _decoder_router_process_receipt(
        tmp_path / "tooflashy.json", name="TooFlashy", contract=contract
    )
    calls: list[str] = []
    _patch_decoder_router_process_audits(monkeypatch, calls)
    tampered = copy.deepcopy(kaya)
    if mutation == "claim":
        tampered["claim_status"] = "SCOREABLE"
    else:
        tampered["canonical_contract"]["frame_map_sha256"] = "f" * 64
    _rewrite(kaya_path, tampered)

    result = external_league.verify_decoder_timeline_parity(
        [flashpatch, kaya_path, tooflashy],
        Path(str(contract["canonical_video"]["path"])),
        Path(str(contract["conversion_receipt"]["path"])),
    )

    kaya_row = next(
        row for row in result["comparators"]
        if row["comparator"] == KAYA_DIRECT_PARTICIPANT_ID
    )
    assert kaya_row["status"] == "NOT_VERIFIED"
    assert failure_text in kaya_row["failure"]
    assert "audit:TooFlashy" in calls
    assert result["status"] == "NOT_VERIFIED"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False


def test_kaya_natural_case_receipt_rejects_promoted_scoring_or_comparison_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt = _natural_case_receipt(tmp_path, monkeypatch)
    for field, promoted in (("claim_status", "SCOREABLE"), ("scoreable", True), ("comparison_eligible", True)):
        tampered = copy.deepcopy(receipt)
        tampered[field] = promoted
        _rewrite(path, tampered)
        with pytest.raises(ExternalLeagueError, match="identity or claim boundary"):
            verify_kaya_natural_case_parity_receipt(path)


def test_kaya_natural_case_receipt_rejects_contract_or_direct_input_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt = _natural_case_receipt(tmp_path, monkeypatch)
    assert verify_kaya_natural_case_parity_receipt(path)["claim_status"] == "NOT_SCOREABLE"

    contract_tampered = copy.deepcopy(receipt)
    contract_tampered["canonical_contract"]["decoder_contract"] = "changed"
    _rewrite(path, contract_tampered)
    with pytest.raises(ExternalLeagueError, match="canonical contract drifted"):
        verify_kaya_natural_case_parity_receipt(path)

    hash_tampered = copy.deepcopy(receipt)
    hash_tampered["direct_input"]["sha256"] = "0" * 64
    _rewrite(path, hash_tampered)
    with pytest.raises(ExternalLeagueError, match="direct input hash mismatches"):
        verify_kaya_natural_case_parity_receipt(path)


def test_execute_kaya_natural_case_parity_never_promotes_a_matching_run_to_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_python = tmp_path / "base" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"frozen-python")
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base_python)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    participant = tmp_path / "participant.json"
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    manifest = tmp_path / "direct-input.json"
    for path in (participant, video, conversion, manifest):
        path.write_text("{}\n", encoding="utf-8")
    output = _natural_case_output()
    provenance = {"repository_url": "https://example.invalid/kaya", "revision": "pinned", "tree": "tree"}
    contract = {"canonical_video": {"path": str(video)}, "conversion_receipt": {"path": str(conversion)}}
    monkeypatch.setattr(external_league, "KAYA_PYTHON_SHA256", _sha256_file(base_python))
    monkeypatch.setattr(external_league, "verify_kaya_participant_conformance_receipt", lambda _: {})
    monkeypatch.setattr(external_league, "_audit_kaya_source_checkout", lambda _: provenance)
    monkeypatch.setattr(external_league, "_canonical_decoder_timeline_contract", lambda *_: ({}, contract))
    monkeypatch.setattr(external_league, "_materialize_kaya_direct_input", lambda *_, **__: {"manifest": str(manifest)})
    invoked_pythons: list[Path] = []

    def fake_execute(*, mode: str, python: Path, **_: object) -> dict[str, object]:
        invoked_pythons.append(python)
        return {"mode": mode}

    monkeypatch.setattr(external_league, "_execute_kaya_child", fake_execute)
    monkeypatch.setattr(external_league, "_load_kaya_child_output", lambda _: {})
    monkeypatch.setattr(external_league, "_validate_kaya_child_output", lambda *_, **__: copy.deepcopy(output))

    result = execute_kaya_natural_case_parity(
        participant, checkout=checkout, python_executable=python, canonical_video=video,
        conversion_receipt=conversion, output_root=tmp_path / "run",
    )

    assert result["status"] == "VERIFIED"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert invoked_pythons == [python.absolute(), python.absolute()]
    assert result["runtime"]["python"] == str(python.absolute())
    persisted = json.loads(Path(str(result["receipt"])).read_text(encoding="utf-8"))
    assert persisted["claim_blockers"] == result["claim_blockers"]


def test_kaya_population_uses_only_the_canonical_participant_identity() -> None:
    assert DIRECT_DETECTOR_POPULATION == (
        "FlashPatch",
        KAYA_DIRECT_PARTICIPANT_ID,
        "TooFlashy",
    )
    assert external_league.EA_IRIS_SOURCE_ADAPTER_ID not in DIRECT_DETECTOR_POPULATION
    assert KAYA_PROTOTYPE_ID not in DIRECT_DETECTOR_POPULATION


@pytest.mark.parametrize("field", ["scoreable", "population_authorized", "population_mutated"])
def test_kaya_receipt_rejects_any_scoring_or_population_authority_claim(
    tmp_path: Path,
    field: str,
) -> None:
    receipt = {
        "schema": KAYA_CONFORMANCE_PROTOTYPE_SCHEMA,
        "identity": KAYA_PROTOTYPE_ID,
        "classification": "UNSCORED_CONFORMANCE_ONLY",
        "upstream": {},
        "adapter_source_sha256": _sha256_bytes(_KAYA_CONFORMANCE_CHILD_SCRIPT.encode("utf-8")),
        "environment": {},
        "fixture_ids": list(KAYA_REQUIRED_FIXTURE_IDS),
        "fixtures": {},
        "conformance": {},
        "status": "VERIFIED",
        "claim_status": "UNSCORED_CONFORMANCE_ONLY",
        "scoreable": False,
        "population_authorized": False,
        "population_mutated": False,
        "scoreable_blockers": [
            "fixed_L7_population_unchanged",
            "independent_execution_witness_missing",
            "natural_case_and_independent_gold_missing",
            "fair_three_repeat_runtime_missing",
        ],
    }
    receipt[field] = True
    path = tmp_path / "receipt.json"
    _rewrite(path, receipt)
    with pytest.raises(ExternalLeagueError, match="claim boundary"):
        verify_kaya_conformance_prototype(path, fresh_replay=False)


def test_kaya_no_replay_can_never_return_verified() -> None:
    assert _kaya_pre_replay_status() == "NOT_VERIFIED"
    result = _finalize_kaya_verification(
        {"status": "VERIFIED"},
        expected_status="VERIFIED",
        failures=[],
        fresh_replay=False,
    )
    assert result["status"] == "NOT_VERIFIED"
    assert result["verification_status"] == "NOT_VERIFIED"
    assert result["verification_blockers"] == ["fresh_replay_disabled"]
    assert result["fresh_replay_verified"] is False


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def _fake_distribution(
    root: Path,
    *,
    module_bytes: bytes,
) -> tuple[dict[str, object], Path]:
    site = root / "lib/python3.10/site-packages"
    package = site / "demo_pkg"
    dist_info = site / "demo_pkg-1.0.dist-info"
    package.mkdir(parents=True, exist_ok=True)
    dist_info.mkdir()
    module = package / "__init__.py"
    module.write_bytes(module_bytes)
    record = dist_info / "RECORD"
    record.write_text(
        "demo_pkg/__init__.py,sha256="
        + _record_digest(module_bytes)
        + f",{len(module_bytes)}\n"
        + "demo_pkg-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    installed_rows = [
        {
            "path": str(module.relative_to(root)),
            "bytes": len(module_bytes),
            "sha256": _sha256_bytes(module_bytes),
        },
        {
            "path": str(record.relative_to(root)),
            "bytes": record.stat().st_size,
            "sha256": _sha256_file(record),
        },
    ]
    installed_rows.sort(key=lambda row: row["path"])
    portable_rows = [
        {
            "path": str(module.relative_to(site)),
            "bytes": len(module_bytes),
            "sha256": _sha256_bytes(module_bytes),
        }
    ]
    normalized_record_rows = sorted([
        [
            "demo_pkg/__init__.py",
            "sha256=" + _record_digest(module_bytes),
            str(len(module_bytes)),
        ],
        ["demo_pkg-1.0.dist-info/RECORD", "", ""],
    ])
    evidence = {
        "version": "1.0",
        "record": {
            "path": str(record.relative_to(root)),
            "sha256": _sha256_file(record),
            "entry_count": 2,
        },
        "owned_roots": sorted(
            [str(package.relative_to(root)), str(dist_info.relative_to(root))]
        ),
        "file_count": 2,
        "files_sha256": _canonical_json_sha256(installed_rows),
        "portable_file_count": len(portable_rows),
        "portable_files_sha256": _canonical_json_sha256(portable_rows),
        "normalized_record_sha256": _canonical_json_sha256(normalized_record_rows),
        "record_hashes_verified": True,
        "unrecorded_files_absent": True,
    }
    return evidence, module


def _runtime_base(root: Path) -> tuple[Path, Path, dict[str, object]]:
    base = root / "runtime-base"
    python = base / "bin/python"
    stdlib = base / "lib/python3.10/os.py"
    python.parent.mkdir(parents=True)
    stdlib.parent.mkdir(parents=True)
    python.write_bytes(b"fake-python\n")
    stdlib.write_bytes(b"fake-stdlib\n")
    rows = [
        {
            "path": path.relative_to(base).as_posix(),
            "type": "file",
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted((python, stdlib), key=lambda value: value.relative_to(base).as_posix())
    ]
    evidence = {
        "classification": "PINNED_IMMUTABLE_RUNTIME_BASE_REQUIRES_STORAGE_INDEPENDENCE",
        "entry_count": len(rows),
        "content_bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": _canonical_json_sha256(rows),
    }
    return base, python, evidence


def _fake_runtime_payload(
    root: Path,
    dependency: dict[str, object],
    module: Path,
    *,
    base: Path,
    python: Path,
    runtime_base: dict[str, object],
) -> dict[str, object]:
    site_packages = root / "lib/python3.10/site-packages"
    loaded_rows = [
        {
            "module": "demo_pkg",
            "classification": "site_packages",
            "path": module.relative_to(site_packages).as_posix(),
            "sha256": _sha256_file(module),
            "shared_object": False,
        }
    ]
    return {
        "runtime": {
            "base_prefix": str(base),
            "site_packages": str(site_packages),
            "python_executable": str(python),
            "runtime_base": runtime_base,
            "dependencies": {"demo": dependency},
            "loaded_modules": {
                "rows": loaded_rows,
                "rows_sha256": _canonical_json_sha256(loaded_rows),
                "module_count": 1,
                "shared_object_count": 0,
                "import_hooks": {},
            },
        }
    }


def test_kaya_distribution_closure_rejects_unrecorded_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, module = _fake_distribution(tmp_path, module_bytes=b"original\n")
    monkeypatch.setitem(external_league.KAYA_REQUIRED_DISTRIBUTIONS, "demo", "1.0")
    monkeypatch.setitem(
        external_league.KAYA_DISTRIBUTION_CLOSURES,
        "demo",
        {
            "normalized_record_sha256": evidence["normalized_record_sha256"],
            "portable_file_count": evidence["portable_file_count"],
            "portable_files_sha256": evidence["portable_files_sha256"],
            "module_path": str(module.relative_to(tmp_path)),
            "module_sha256": _sha256_file(module),
        },
    )
    (module.parent / "shadow.py").write_text("poison = True\n", encoding="utf-8")
    with pytest.raises(ExternalLeagueError, match="unrecorded extra file"):
        _verify_kaya_distribution_closure("demo", evidence, environment_root=tmp_path)


def test_kaya_distribution_closure_rejects_coherently_rewritten_poisoned_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen, frozen_module = _fake_distribution(tmp_path, module_bytes=b"original\n")
    immutable = {
        "normalized_record_sha256": frozen["normalized_record_sha256"],
        "portable_file_count": frozen["portable_file_count"],
        "portable_files_sha256": frozen["portable_files_sha256"],
        "module_path": str(frozen_module.relative_to(tmp_path)),
        "module_sha256": _sha256_file(frozen_module),
    }
    poisoned_root = tmp_path / "poisoned"
    poisoned, _ = _fake_distribution(poisoned_root, module_bytes=b"poisoned\n")
    monkeypatch.setitem(external_league.KAYA_REQUIRED_DISTRIBUTIONS, "demo", "1.0")
    monkeypatch.setitem(external_league.KAYA_DISTRIBUTION_CLOSURES, "demo", immutable)
    with pytest.raises(ExternalLeagueError, match="frozen wheel closure"):
        _verify_kaya_distribution_closure("demo", poisoned, environment_root=poisoned_root)


def test_kaya_fresh_replay_rejects_hardlinked_dependency_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_root = tmp_path / "primary"
    replay_root = tmp_path / "replay"
    primary, primary_module = _fake_distribution(primary_root, module_bytes=b"same\n")
    replay, replay_module = _fake_distribution(replay_root, module_bytes=b"same\n")
    replay_module.unlink()
    os.link(primary_module, replay_module)
    monkeypatch.setattr(external_league, "KAYA_REQUIRED_DISTRIBUTIONS", {"demo": "1.0"})
    primary_base, primary_python, runtime_base = _runtime_base(primary_root)
    replay_base, replay_python, replay_runtime_base = _runtime_base(replay_root)
    assert runtime_base == replay_runtime_base
    monkeypatch.setattr(external_league, "KAYA_COMMON_BASE_RUNTIME_CLOSURE", runtime_base)
    primary_payload = _fake_runtime_payload(
        primary_root, primary, primary_module,
        base=primary_base, python=primary_python, runtime_base=runtime_base,
    )
    replay_payload = _fake_runtime_payload(
        replay_root, replay, replay_module,
        base=replay_base, python=replay_python, runtime_base=replay_runtime_base,
    )
    with pytest.raises(ExternalLeagueError, match="storage is aliased"):
        _verify_kaya_replay_storage_independence(
            primary_payload,
            replay_payload,
            primary_environment_root=primary_root,
            replay_environment_root=replay_root,
        )


def test_kaya_fresh_replay_rejects_shared_interpreter_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_root = tmp_path / "primary"
    replay_root = tmp_path / "replay"
    primary, primary_module = _fake_distribution(primary_root, module_bytes=b"same\n")
    replay, replay_module = _fake_distribution(replay_root, module_bytes=b"same\n")
    shared_base, shared_python, runtime_base = _runtime_base(tmp_path / "shared")
    monkeypatch.setattr(external_league, "KAYA_REQUIRED_DISTRIBUTIONS", {"demo": "1.0"})
    monkeypatch.setattr(external_league, "KAYA_COMMON_BASE_RUNTIME_CLOSURE", runtime_base)
    primary_payload = _fake_runtime_payload(
        primary_root, primary, primary_module,
        base=shared_base, python=shared_python, runtime_base=runtime_base,
    )
    replay_payload = _fake_runtime_payload(
        replay_root, replay, replay_module,
        base=shared_base, python=shared_python, runtime_base=runtime_base,
    )
    with pytest.raises(ExternalLeagueError, match="runtime storage is aliased"):
        _verify_kaya_replay_storage_independence(
            primary_payload,
            replay_payload,
            primary_environment_root=primary_root,
            replay_environment_root=replay_root,
        )


def test_kaya_fresh_replay_accepts_independently_copied_runtime_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_root = tmp_path / "primary"
    replay_root = tmp_path / "replay"
    primary, primary_module = _fake_distribution(primary_root, module_bytes=b"same\n")
    replay, replay_module = _fake_distribution(replay_root, module_bytes=b"same\n")
    primary_base, primary_python, runtime_base = _runtime_base(primary_root)
    replay_base, replay_python, replay_runtime_base = _runtime_base(replay_root)
    assert runtime_base == replay_runtime_base
    monkeypatch.setattr(external_league, "KAYA_REQUIRED_DISTRIBUTIONS", {"demo": "1.0"})
    monkeypatch.setattr(external_league, "KAYA_COMMON_BASE_RUNTIME_CLOSURE", runtime_base)
    evidence = _verify_kaya_replay_storage_independence(
        _fake_runtime_payload(
            primary_root, primary, primary_module,
            base=primary_base, python=primary_python, runtime_base=runtime_base,
        ),
        _fake_runtime_payload(
            replay_root, replay, replay_module,
            base=replay_base, python=replay_python, runtime_base=replay_runtime_base,
        ),
        primary_environment_root=primary_root,
        replay_environment_root=replay_root,
    )
    assert evidence["shared_inode_count"] == 0
    assert evidence["primary_counts"]["runtime_base_entries"] == 2
    assert evidence["replay_counts"]["loaded_runtime_modules"] == 1
