from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flashpatch.external_league as external_league
from flashpatch.external_league import (
    TOOFLASHY_COPIED_REPLAY_WITNESS_SCHEMA,
    TOOFLASHY_OFFICIAL_JSON_FIELDS,
    TOOFLASHY_PARITY_ADAPTER_SCHEMA,
    TOOFLASHY_PARITY_CALLABLE_HASHES,
    TOOFLASHY_PARITY_SOURCE_HASHES,
    materialize_cfr_ffv1,
    verify_tooflashy_copied_replay_witness,
    verify_tooflashy_parity_adapter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frames(path: Path) -> Path:
    frames = np.arange(8 * 4 * 6 * 3, dtype=np.uint8).reshape(8, 4, 6, 3)
    timestamps = np.arange(8, dtype=np.float64) / 60.0
    np.savez_compressed(path, frames=frames, timestamps=timestamps)
    return path


def _valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    video = (tmp_path / "lane" / str(lane["canonical_video"]["path"])).resolve()
    conversion = Path(str(lane["receipt"])).resolve()
    second_frames = 255 - np.arange(8 * 4 * 6 * 3, dtype=np.uint8).reshape(8, 4, 6, 3)
    second_timestamps = np.arange(8, dtype=np.float64) / 60.0
    second_source = tmp_path / "conformance-frames.npz"
    np.savez_compressed(second_source, frames=second_frames, timestamps=second_timestamps)
    materialize_cfr_ffv1(second_source, tmp_path / "conformance-lane", fps=60)
    second_video = tmp_path / "conformance-lane" / "canonical.ffv1.mkv"
    conformance_manifest_path = tmp_path / "conformance-manifest.json"
    conformance_manifest_payload = {
        "schema": external_league.TOOFLASHY_CONFORMANCE_FIXTURES_SCHEMA,
        "fixtures": [{
            "id": "inverse-rgb",
            "path": str(second_video.relative_to(tmp_path)),
            "sha256": _sha256(second_video),
            "bytes": second_video.stat().st_size,
        }],
    }
    _write_json(conformance_manifest_path, conformance_manifest_payload)
    frozen_conformance = external_league._load_tooflashy_conformance_manifest(conformance_manifest_path)
    root = tmp_path / "adapter"
    root.mkdir()
    checkout = tmp_path / "TooFlashy"
    checkout.mkdir()
    upstream = {
        "path": str(checkout),
        "repository_url": "https://github.com/hashb/TooFlashy",
        "revision": external_league.TOOFLASHY_PARITY_ADAPTER_REVISION,
        "tree": external_league.TOOFLASHY_PARITY_ADAPTER_TREE,
        "license": "Apache-2.0",
        "source_sha256": dict(TOOFLASHY_PARITY_SOURCE_HASHES),
    }
    uv = Path(shutil.which("uv") or "").resolve()
    assert uv.is_file()
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps({"artifact_root": str(tmp_path / "census-artifacts")}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(external_league, "_verify_tooflashy_adapter_checkout", lambda entry: copy.deepcopy(upstream))
    monkeypatch.setattr(
        external_league,
        "_load_execution_census_entry",
        lambda receipt, artifact_root, comparator: ({
            "repository_url": upstream["repository_url"],
            "revision": upstream["revision"],
            "license": upstream["license"],
            "binary_sha256": _sha256(uv),
        }, census),
    )
    _, canonical = external_league._canonical_decoder_timeline_contract(video, conversion)
    environment = {
        "HOME": str(root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_PROJECT": str(checkout),
        "UV_PROJECT_ENVIRONMENT": str(root / "uv-environment"),
    }
    adapter_hash = hashlib.sha256(external_league._TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode()).hexdigest()
    rows = []
    fresh_replays: dict[str, dict[str, object]] = {}
    for ordinal, fixture in enumerate((video, second_video)):
        fixture_root = root / f"fixture-{ordinal:03d}"
        fixture_root.mkdir()
        if ordinal == 0:
            ledger = [
                {
                    "index": row["frame_index"],
                    "cfr_timestamp_us": row["cfr_timestamp_us"],
                    "shape": list(canonical["shape"][1:]),
                    "pixel_format": "rgb24",
                    "rgb_sha256": row["rgb_sha256"],
                }
                for row in canonical["frame_map"]
            ]
        else:
            ledger = [
                {
                    "index": row["frame_index"],
                    "cfr_timestamp_us": row["cfr_timestamp_us"],
                    "shape": list(canonical["shape"][1:]),
                    "pixel_format": "rgb24",
                    "rgb_sha256": row["rgb_sha256"],
                }
                for row in canonical["frame_map"]
            ]
        official = {
            "path": str(fixture),
            "passes": True,
            "fps": 60.0,
            "frame_count": len(ledger),
            "event_count": 0,
            "failures": [],
        }
        assert set(official) == TOOFLASHY_OFFICIAL_JSON_FIELDS
        public_api = {
            "iter_video_frames": {
                "module": "tooflashy.video",
                "qualname": "iter_video_frames",
                "module_path": str(checkout / "src/tooflashy/video.py"),
                "module_sha256": TOOFLASHY_PARITY_SOURCE_HASHES["src/tooflashy/video.py"],
                "callable_source_sha256": TOOFLASHY_PARITY_CALLABLE_HASHES["iter_video_frames"],
            },
            "analyze_frames": {
                "module": "tooflashy.analysis",
                "qualname": "analyze_frames",
                "module_path": str(checkout / "src/tooflashy/analysis.py"),
                "module_sha256": TOOFLASHY_PARITY_SOURCE_HASHES["src/tooflashy/analysis.py"],
                "callable_source_sha256": TOOFLASHY_PARITY_CALLABLE_HASHES["analyze_frames"],
            },
        }
        child_environment = {
            **{key: value for key, value in environment.items() if key != "PATH"},
            "PATH": f"{root / 'uv-environment'}/bin:/usr/bin:/bin",
            "LC_CTYPE": "C.UTF-8",
            "LD_LIBRARY_PATH": str(root / "uv-environment/lib/python3.12/site-packages/cv2/../../lib64") + ":",
            "QT_QPA_FONTDIR": str(root / "uv-environment/lib/python3.12/site-packages/cv2/qt/fonts"),
            "QT_QPA_PLATFORM_PLUGIN_PATH": str(root / "uv-environment/lib/python3.12/site-packages/cv2/qt/plugins"),
            "UV": str(uv),
            "UV_RUN_RECURSION_DEPTH": "1",
            "VIRTUAL_ENV": str(root / "uv-environment"),
        }
        child_python = Path(sys.executable).resolve()
        assert child_python.is_file()
        decoder_executables = {}
        for executable_name in ("ffmpeg", "ffprobe"):
            executable = Path(shutil.which(executable_name) or "").resolve()
            assert executable.is_file()
            version = subprocess.run([str(executable), "-version"], capture_output=True, check=True)
            decoder_executables[executable_name] = {
                "path": str(executable),
                "sha256": _sha256(executable),
                "version_stdout_sha256": hashlib.sha256(version.stdout).hexdigest(),
                "version_stderr_sha256": hashlib.sha256(version.stderr).hexdigest(),
            }
        adapter_payload = {
            "schema": "flashpatch-l7-tooflashy-child-adapter-v1",
            "evidence_origin": "live_generator_append_immediately_before_yield_v1",
            "adapter_source_sha256": adapter_hash,
            "input": {"path": str(fixture), "sha256": _sha256(fixture)},
            "public_api": public_api,
            "runtime": {
                "python_executable": str(child_python),
                "python_executable_sha256": _sha256(child_python),
                "python_version": sys.version,
                "environment": child_environment,
                "sys_path": [],
                "dependency_versions": {"tooflashy": "0.1.0", "numpy": "2.4.4", "opencv-python": "4.13.0.92", "opencv-python-headless": None},
                "dependency_evidence": {
                    "tooflashy": {"version": "0.1.0", "file_count": 1, "files_sha256": "1" * 64},
                    "numpy": {"version": "2.4.4", "file_count": 1, "files_sha256": "2" * 64},
                    "opencv-python": {"version": "4.13.0.92", "file_count": 1, "files_sha256": "3" * 64},
                    "opencv-python-headless": None,
                },
                "decoder_executables": decoder_executables,
            },
            "decode": {
                "engine": "ffmpeg",
                "fps": 60.0,
                "frame_count": len(ledger),
                "pixel_format": "rgb24",
                "ledger": ledger,
                "ledger_sha256": external_league._canonical_json_sha256(ledger),
            },
            "result": {
                **official,
                "event_representation": {"event_count": 0, "failures": []},
            },
        }
        adapter_output = fixture_root / "adapter-output.json"
        cli_output = fixture_root / "official-cli-output.json"
        _write_json(adapter_output, adapter_payload)
        _write_json(cli_output, official)
        adapter_stdout = fixture_root / "adapter.stdout.bin"
        adapter_stderr = fixture_root / "adapter.stderr.bin"
        cli_stdout = fixture_root / "official-cli.stdout.bin"
        cli_stderr = fixture_root / "official-cli.stderr.bin"
        adapter_stdout.write_bytes(b"")
        adapter_stderr.write_bytes(b"")
        cli_stdout.write_bytes(cli_output.read_bytes())
        cli_stderr.write_bytes(b"")
        adapter_command = [
            str(uv), "run", "--locked", "--project", str(checkout), "--directory", str(checkout),
            "python", "-c", external_league._TOOFLASHY_PARITY_ADAPTER_SCRIPT,
            str(fixture), str(adapter_output), adapter_hash,
        ]
        cli_command = [
            str(uv), "run", "--locked", "--project", str(checkout), "--directory", str(checkout),
            "tooflashy", "--json", str(fixture),
        ]

        def process(command: list[str], stdout: Path, stderr: Path, exit_code: int) -> dict[str, object]:
            return {
                "command": command,
                "working_directory": str(checkout),
                "environment_sha256": external_league._canonical_json_sha256(environment),
                "exit_code": exit_code,
                "timed_out": False,
                "stdout": {"path": stdout.name, "sha256": _sha256(stdout)},
                "stderr": {"path": stderr.name, "sha256": _sha256(stderr)},
            }

        rows.append({
            "ordinal": ordinal,
            "fixture_id": "canonical" if ordinal == 0 else "inverse-rgb",
            "role": "CANONICAL" if ordinal == 0 else "CONFORMANCE",
            "input": {"path": str(fixture), "sha256": _sha256(fixture), "bytes": fixture.stat().st_size},
            "adapter_process": process(adapter_command, adapter_stdout, adapter_stderr, 0),
            "adapter_output": {"path": str(adapter_output.relative_to(root)), "exists": True, "sha256": _sha256(adapter_output)},
            "official_cli_process": process(cli_command, cli_stdout, cli_stderr, 0),
            "official_cli_output": {"path": str(cli_output.relative_to(root)), "sha256": _sha256(cli_output)},
        })
        fresh_replays[str(fixture.resolve())] = {
            "adapter": copy.deepcopy(adapter_payload),
            "official_cli": copy.deepcopy(official),
            "official_cli_exit_code": 0,
            "root": str(root / "fresh-placeholder"),
            "environment": copy.deepcopy(environment),
            "runtime_frozen": True,
        }
    monkeypatch.setattr(
        external_league,
        "_fresh_replay_tooflashy_fixture",
        lambda fixture, **kwargs: copy.deepcopy(fresh_replays[str(Path(fixture).resolve())]),
    )
    receipt = {
        "schema": TOOFLASHY_PARITY_ADAPTER_SCHEMA,
        "adapter_source_sha256": adapter_hash,
        "upstream": copy.deepcopy(upstream),
        "dependency_lock": [
            {"path": relative, "sha256": TOOFLASHY_PARITY_SOURCE_HASHES[relative]}
            for relative in ("pyproject.toml", "uv.lock")
        ],
        "census_receipt": {"path": str(census), "sha256": _sha256(census), "artifact_root": str(tmp_path / "census-artifacts")},
        "canonical_input": {
            "video": canonical["canonical_video"],
            "conversion_receipt": canonical["conversion_receipt"],
            "frame_map_sha256": canonical["frame_map_sha256"],
        },
        "conformance_manifest": frozen_conformance,
        "environment_contract": {"environment": environment, "sha256": external_league._canonical_json_sha256(environment)},
        "runner": {"uv": str(uv), "uv_sha256": _sha256(uv), "timeout_seconds": 120},
        "fixtures": rows,
        "status": "NOT_VERIFIED",
        "parity_reason": "verification_pending",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "scoreable_blockers": ["detector_scoring_out_of_scope", "independent_gold_not_verified"],
    }
    receipt_path = root / "tooflashy-parity-adapter-receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path, video, conversion, receipt


def _rewrite_adapter_output(receipt_path: Path, receipt: dict[str, object], ordinal: int, mutate) -> None:
    row = receipt["fixtures"][ordinal]
    path = receipt_path.parent / row["adapter_output"]["path"]
    payload = json.loads(path.read_text())
    mutate(payload)
    _write_json(path, payload)
    row["adapter_output"]["sha256"] = _sha256(path)
    _write_json(receipt_path, receipt)


def _scheduled_adapter_child(
    tmp_path: Path,
    receipt_path: Path,
    receipt: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    child_root = tmp_path / "scheduled-child"
    child_root.mkdir()
    canonical_row = receipt["fixtures"][0]
    source = receipt_path.parent / canonical_row["adapter_output"]["path"]
    payload = json.loads(source.read_text())
    parent_environment = {
        "HOME": str(child_root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(child_root / "uv-cache"),
        "UV_PROJECT": receipt["upstream"]["path"],
        "UV_PROJECT_ENVIRONMENT": str(child_root / "uv-environment"),
    }
    payload["runtime"]["environment"] = {
        **{key: value for key, value in parent_environment.items() if key != "PATH"},
        "PATH": f"{child_root / 'uv-environment'}/bin:/usr/bin:/bin",
        "LC_CTYPE": "C.UTF-8",
        "LD_LIBRARY_PATH": str(child_root / "uv-environment/lib/python3.12/site-packages/cv2/../../lib64") + ":",
        "QT_QPA_FONTDIR": str(child_root / "uv-environment/lib/python3.12/site-packages/cv2/qt/fonts"),
        "QT_QPA_PLATFORM_PLUGIN_PATH": str(child_root / "uv-environment/lib/python3.12/site-packages/cv2/qt/plugins"),
        "UV": receipt["runner"]["uv"],
        "UV_RUN_RECURSION_DEPTH": "1",
        "VIRTUAL_ENV": str(child_root / "uv-environment"),
    }
    raw = child_root / "raw-output.bin"
    _write_json(raw, payload)
    stdout = child_root / "stdout.bin"
    stderr = child_root / "stderr.bin"
    stdout.write_bytes(b"")
    stderr.write_bytes(b"")
    canonical_video = Path(receipt["canonical_input"]["video"]["path"])
    normalized = external_league.parse_tooflashy_adapter_json(
        raw,
        canonical_video,
        expected_fps=60,
        expected_frame_count=8,
    )
    child = {
        "schema": "flashpatch-external-comparator-run-v1",
        "status": "PROCESS_VALID",
        "exit_code": 0,
        "parse_error": None,
        "parsed_observation": normalized,
        "command": [],
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "raw_output": {"path": raw.name, "mode": "file", "sha256": _sha256(raw)},
        "input": receipt["canonical_input"]["video"],
        "conversion_receipt": receipt["canonical_input"]["conversion_receipt"],
        "census_receipt": {
            "path": receipt["census_receipt"]["path"],
            "sha256": receipt["census_receipt"]["sha256"],
        },
        "comparator": {
            "name": "TooFlashy",
            "repository_url": receipt["upstream"]["repository_url"],
            "revision": receipt["upstream"]["revision"],
            "license": receipt["upstream"]["license"],
            "binary": receipt["runner"]["uv"],
            "binary_sha256": receipt["runner"]["uv_sha256"],
            "working_directory": receipt["upstream"]["path"],
        },
        "tooflashy_parity_adapter": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "status": "VERIFIED",
        },
        "adapter_build": {"environment": parent_environment},
    }
    child_path = child_root / "comparator-receipt.json"
    _write_json(child_path, child)
    return child_path, child


def _forge_coherent_local_execution_witness(
    child_path: Path,
    child: dict[str, object],
    receipt: dict[str, object],
) -> None:
    root = child_path.parent
    checkout = Path(receipt["upstream"]["path"])
    uv = Path(receipt["runner"]["uv"])
    input_path = Path(child["input"]["path"])
    raw_path = root / child["raw_output"]["path"]
    adapter_environment = child["adapter_build"]["environment"]
    build_stdout = root / "adapter-build.stdout.bin"
    build_stderr = root / "adapter-build.stderr.bin"
    build_stdout.write_bytes(b"")
    build_stderr.write_bytes(b"")
    child["adapter_build"] = {
        "command": [
            str(uv), "sync", "--locked", "--project", str(checkout),
            "--directory", str(checkout),
        ],
        "environment": adapter_environment,
        "environment_sha256": external_league._canonical_json_sha256(adapter_environment),
        "exit_code": 0,
        "stdout_sha256": _sha256(build_stdout),
        "stderr_sha256": _sha256(build_stderr),
    }
    witness = {
        "schema": "flashpatch-l7-child-runtime-probe-v1",
        "effective_environment": {
            "cache": {
                "policy": "WARM_INPUT_PRETOUCHED",
                "input_sha256": _sha256(input_path),
                "input_bytes": input_path.stat().st_size,
            },
        },
        "launcher_identity_environment": {"PWD": None, "UV_PROJECT": str(checkout)},
        "schedule_observation": None,
        "child_timing": {
            "probe_started_monotonic_ns": 1,
            "tool_started_monotonic_ns": 2,
            "tool_finished_monotonic_ns": 3,
        },
    }
    witness_path = root / "runtime-probe.json"
    _write_json(witness_path, witness)
    child["adapter_execution_witness"] = {
        "path": witness_path.name,
        "sha256": _sha256(witness_path),
        "observation": witness,
    }
    adapter_hash = hashlib.sha256(external_league._TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode()).hexdigest()
    adapter_tool = [
        str(Path("/usr/bin/env").resolve()), "-i",
        *[f"{key}={value}" for key, value in sorted(adapter_environment.items())],
        str(uv), "run", "--locked", "--no-sync", "--project", str(checkout),
        "--directory", str(checkout), "python", "-c",
        external_league._TOOFLASHY_PARITY_ADAPTER_SCRIPT,
        str(input_path), str(raw_path), adapter_hash,
    ]
    child["command"] = [
        str(Path(sys.executable).resolve()), "-c", external_league._RUNTIME_PROBE_SCRIPT,
        str(witness_path), str(input_path), "-", *adapter_tool,
    ]
    _write_json(child_path, child)


def test_tooflashy_adapter_verifies_exact_public_api_ledger_and_cli_conformance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, _ = _valid_bundle(tmp_path, monkeypatch)

    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "VERIFIED"
    assert result["parity_reason"] is None
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False


@pytest.mark.parametrize("api_name", ["iter_video_frames", "analyze_frames"])
def test_tooflashy_adapter_rejects_monkey_patched_public_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_name: str) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    _rewrite_adapter_output(
        receipt_path,
        receipt,
        0,
        lambda payload: payload["public_api"][api_name].__setitem__("callable_source_sha256", "0" * 64),
    )

    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert f"fixture_0:public_api_altered:{api_name}" in result["parity_reason"]


def test_tooflashy_adapter_rejects_post_consumption_or_synthetic_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    _rewrite_adapter_output(
        receipt_path,
        receipt,
        0,
        lambda payload: payload.__setitem__("evidence_origin", "post_hoc_redecode"),
    )

    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert "fixture_0:synthetic_post_hoc_audit" in result["parity_reason"]


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("index", 7, "fixture_0:frame_ledger_drift:0"),
        ("cfr_timestamp_us", 999, "fixture_0:frame_ledger_drift:0"),
        ("rgb_sha256", "f" * 64, "fixture_0:canonical_rgb_timeline_drift"),
    ],
)
def test_tooflashy_adapter_rejects_index_timestamp_and_rgb_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    failure: str,
) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)

    def mutate(payload: dict[str, object]) -> None:
        payload["decode"]["ledger"][0][field] = value
        payload["decode"]["ledger_sha256"] = external_league._canonical_json_sha256(payload["decode"]["ledger"])

    _rewrite_adapter_output(receipt_path, receipt, 0, mutate)
    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert failure in result["parity_reason"]


def test_tooflashy_adapter_rejects_adapter_cli_divergence_and_unsupported_cli_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    _rewrite_adapter_output(
        receipt_path,
        receipt,
        1,
        lambda payload: payload["result"].__setitem__("event_count", 2),
    )
    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)
    assert "fixture_1:adapter_cli_divergence" in result["parity_reason"]

    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path / "second", monkeypatch)
    row = receipt["fixtures"][1]
    cli_path = receipt_path.parent / row["official_cli_output"]["path"]
    cli = json.loads(cli_path.read_text())
    cli["invented_interval"] = [1, 2]
    _write_json(cli_path, cli)
    row["official_cli_output"]["sha256"] = _sha256(cli_path)
    _write_json(receipt_path, receipt)
    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)
    assert "fixture_1:official_cli_unsupported_output_field" in result["parity_reason"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("source", "upstream_receipt_drift"),
        ("dependency", "dependency_lock_drift"),
        ("provenance", "runner_provenance_drift"),
    ],
)
def test_tooflashy_adapter_rejects_source_dependency_and_runner_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    if mutation == "source":
        receipt["upstream"]["tree"] = "0" * 40
    elif mutation == "dependency":
        receipt["dependency_lock"][1]["sha256"] = "0" * 64
    else:
        receipt["runner"]["uv_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert expected in result["parity_reason"]


def test_tooflashy_adapter_rejects_fake_conformance_receipt_even_with_rehashed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    receipt["fixtures"][1]["official_cli_process"]["command"][-1] = str(video)
    receipt["status"] = "VERIFIED"
    receipt["parity_reason"] = None
    _write_json(receipt_path, receipt)

    result = verify_tooflashy_parity_adapter(receipt_path, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert "fixture_1:official_cli_command_drift" in result["parity_reason"]


def test_tooflashy_decoder_audit_rejects_copied_adapter_artifact_without_execution_witness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    child_path, child = _scheduled_adapter_child(tmp_path, receipt_path, receipt)
    _, contract = external_league._canonical_decoder_timeline_contract(video, conversion)

    with pytest.raises(external_league.ExternalLeagueError, match="ledger, result, API, or runtime drifted"):
        external_league._audit_tooflashy_decoder_timeline(
            child_path,
            child,
            contract,
            receipt_path,
        )


def test_public_decoder_parity_verifier_rejects_rehashed_copied_adapter_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    child_path, _ = _scheduled_adapter_child(tmp_path, receipt_path, receipt)

    result = external_league.verify_decoder_timeline_parity(
        [child_path],
        video,
        conversion,
        tooflashy_adapter_receipt=receipt_path,
    )

    row = next(item for item in result["comparators"] if item["comparator"] == "TooFlashy")
    assert result["status"] == "NOT_VERIFIED"
    assert row["status"] == "NOT_VERIFIED"
    assert "scheduled TooFlashy adapter ledger, result, API, or runtime drifted" in row["failure"]


def test_coherent_forged_local_witness_never_becomes_comparison_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    child_path, child = _scheduled_adapter_child(tmp_path, receipt_path, receipt)
    _forge_coherent_local_execution_witness(child_path, child, receipt)

    result = external_league.verify_decoder_timeline_parity(
        [child_path],
        video,
        conversion,
        tooflashy_adapter_receipt=receipt_path,
    )

    row = next(item for item in result["comparators"] if item["comparator"] == "TooFlashy")
    assert result["status"] == "NOT_VERIFIED"
    assert row["status"] == "VERIFIED"
    assert row["decoder_timeline"]["execution_witness_status"] == "LOCAL_RECEIPT_ONLY_NOT_INDEPENDENT"
    assert row["decoder_timeline"]["league_child_binding"] == "LOCAL_RECEIPT_CONSISTENT_SAME_PROCESS_EVIDENCE"
    assert row["primary_case_level_endpoint"]["comparison_eligible"] is False
    assert "decoder_comparison_ineligible:TooFlashy:independent_execution_witness_missing" in result["parity_blockers"]


def test_tooflashy_decoder_audit_rejects_matching_official_cli_child_without_same_process_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    child_path, child = _scheduled_adapter_child(tmp_path, receipt_path, receipt)
    canonical_row = receipt["fixtures"][0]
    official = receipt_path.parent / canonical_row["official_cli_output"]["path"]
    raw = child_path.parent / child["raw_output"]["path"]
    raw.write_bytes(official.read_bytes())
    child["raw_output"]["sha256"] = _sha256(raw)
    _, contract = external_league._canonical_decoder_timeline_contract(video, conversion)

    with pytest.raises(external_league.ExternalLeagueError, match="scheduled TooFlashy child"):
        external_league._audit_tooflashy_decoder_timeline(child_path, child, contract, receipt_path)


def test_tooflashy_decoder_audit_rejects_same_outcome_with_different_consumed_rgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    child_path, child = _scheduled_adapter_child(tmp_path, receipt_path, receipt)
    raw = child_path.parent / child["raw_output"]["path"]
    payload = json.loads(raw.read_text())
    payload["decode"]["ledger"][0]["rgb_sha256"] = "f" * 64
    payload["decode"]["ledger_sha256"] = external_league._canonical_json_sha256(payload["decode"]["ledger"])
    _write_json(raw, payload)
    child["raw_output"]["sha256"] = _sha256(raw)
    _, contract = external_league._canonical_decoder_timeline_contract(video, conversion)

    with pytest.raises(external_league.ExternalLeagueError, match="ledger, result, API, or runtime drifted"):
        external_league._audit_tooflashy_decoder_timeline(child_path, child, contract, receipt_path)


def test_tooflashy_adapter_rejects_fair_runtime_until_prebuilt_environment_is_schedule_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, video, conversion, receipt = _valid_bundle(tmp_path, monkeypatch)
    checkout = Path(receipt["upstream"]["path"])
    uv = Path(receipt["runner"]["uv"])
    entry = {
        "repository_url": receipt["upstream"]["repository_url"],
        "revision": receipt["upstream"]["revision"],
        "license": receipt["upstream"]["license"],
        "distribution": "fixed-source-checkout",
        "distribution_revision": receipt["upstream"]["revision"],
        "configuration_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "source_checkout": str(checkout),
        "binary_sha256": _sha256(uv),
    }
    monkeypatch.setattr(
        external_league,
        "_load_execution_census_entry",
        lambda *args, **kwargs: (entry, Path(receipt["census_receipt"]["path"])),
    )
    monkeypatch.setattr(
        external_league,
        "_checkout_provenance",
        lambda spec: {"status": "VERIFIED", "path": str(checkout), "head": spec.revision, "clean": True},
    )
    spec = external_league.ComparatorSpec(
        name="TooFlashy",
        repository_url=entry["repository_url"],
        revision=entry["revision"],
        license=entry["license"],
        mode="detection",
        command=(str(uv), "run", "tooflashy", "--json", "{input}"),
        raw_output_mode="stdout",
        working_directory=checkout,
        source_checkout=checkout,
        distribution=entry["distribution"],
        distribution_revision=entry["distribution_revision"],
        configuration_sha256=entry["configuration_sha256"],
        environment_sha256=entry["environment_sha256"],
    )
    protocol = external_league.capture_fair_runtime_protocol(
        concurrency_lock_path=tmp_path / "fair.lock",
        timeout_seconds=120,
        cpu_affinity=sorted(__import__("os").sched_getaffinity(0))[:1],
        thread_limit=1,
    )
    monkeypatch.setattr(
        external_league,
        "_fresh_replay_tooflashy_fixture",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fair mode must reject before replay")),
    )

    with pytest.raises(external_league.ExternalLeagueError, match="does not authorize fair-runtime scoring"):
        external_league.execute_comparator(
            spec,
            video,
            conversion,
            tmp_path / "forbidden-fair-run",
            census_receipt=receipt["census_receipt"]["path"],
            census_artifact_root=receipt["census_receipt"]["artifact_root"],
            runtime_protocol=protocol,
            scheduled_repeat_ordinal=1,
            tooflashy_parity_adapter_receipt=receipt_path,
        )


def _copied_replay_stub(
    root: Path,
    *,
    video: Path,
    conversion: Path,
) -> tuple[Path, dict[str, object], Path]:
    checkout = root / "checkout"
    adapter_root = root / "adapter"
    environment_root = adapter_root / "uv-environment"
    cache_root = adapter_root / "uv-cache"
    for directory in (checkout, adapter_root / "home", environment_root, cache_root):
        directory.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for relative in TOOFLASHY_PARITY_SOURCE_HASHES:
        source = checkout / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative + "\n", encoding="utf-8")
        source_hashes[relative] = _sha256(source)
    environment_file = environment_root / "site-packages" / "dependency.bin"
    environment_file.parent.mkdir(parents=True)
    environment_file.write_bytes(b"locked dependency closure")
    (cache_root / "wheel.bin").write_bytes(b"locked wheel cache")

    site_packages = environment_root / "lib" / "python3.12" / "site-packages"
    dist_info = site_packages / "tooflashy-0.1.0.dist-info"
    dist_info.mkdir(parents=True)
    for name, payload in external_league.TOOFLASHY_EDITABLE_SEMANTIC_FILES.items():
        (dist_info / name).write_bytes(payload)
    direct_url = dist_info / "direct_url.json"
    direct_url.write_text(
        json.dumps(
            {"url": checkout.resolve().as_uri(), "dir_info": {"editable": True}},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    pth = site_packages / "tooflashy.pth"
    pth.write_text(str(checkout.resolve() / "src"), encoding="utf-8")
    build_timestamp = {
        "secs_since_epoch": 100 if root.name == "primary" else 200,
        "nanos_since_epoch": 123,
    }
    uv_cache = dist_info / "uv_cache.json"
    uv_cache.write_text(
        json.dumps(
            {
                "timestamp": build_timestamp,
                "commit": None,
                "tags": None,
                "env": {},
                "directories": {"src": build_timestamp},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    console = environment_root / "bin" / "tooflashy"
    console.parent.mkdir(parents=True)
    console.write_bytes(
        b"#!" + str(environment_root / "bin" / "python").encode("utf-8") + b"\n"
        + external_league.TOOFLASHY_EDITABLE_CONSOLE_BODY
    )

    dist_prefix = dist_info.name
    record_paths = [
        "../../../bin/tooflashy",
        *[f"{dist_prefix}/{name}" for name in external_league.TOOFLASHY_EDITABLE_SEMANTIC_FILES],
        f"{dist_prefix}/RECORD",
        f"{dist_prefix}/direct_url.json",
        f"{dist_prefix}/uv_cache.json",
        "tooflashy.pth",
    ]
    record_rows: list[list[str]] = []
    installed_rows: list[dict[str, object]] = []
    for relative in sorted(record_paths):
        path = (site_packages / relative).resolve()
        if relative.endswith("/RECORD"):
            record_rows.append([relative, "", ""])
            continue
        payload = path.read_bytes()
        record_rows.append([
            relative,
            f"sha256={external_league._tooflashy_record_digest(payload)}",
            str(len(payload)),
        ])
        if (
            path.is_relative_to(site_packages.resolve())
            and path.name not in {"INSTALLER", "REQUESTED"}
        ):
            installed_rows.append({
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    record = dist_info / "RECORD"
    record.write_text(
        "".join(
            f"{row[0]},{row[1]},{row[2]}\n"
            for row in record_rows
        ),
        encoding="utf-8",
    )
    tooflashy_dependency_evidence = {
        "version": "0.1.0",
        "file_count": len(installed_rows),
        "files_sha256": external_league._canonical_json_sha256(installed_rows),
    }

    adapter_hash = hashlib.sha256(
        external_league._TOOFLASHY_PARITY_ADAPTER_SCRIPT.encode("utf-8")
    ).hexdigest()
    public_api = {
        "iter_video_frames": {
            "module": "tooflashy.video",
            "qualname": "iter_video_frames",
            "module_path": str(checkout / "src/tooflashy/video.py"),
            "module_sha256": TOOFLASHY_PARITY_SOURCE_HASHES["src/tooflashy/video.py"],
            "callable_source_sha256": TOOFLASHY_PARITY_CALLABLE_HASHES["iter_video_frames"],
        },
        "analyze_frames": {
            "module": "tooflashy.analysis",
            "qualname": "analyze_frames",
            "module_path": str(checkout / "src/tooflashy/analysis.py"),
            "module_sha256": TOOFLASHY_PARITY_SOURCE_HASHES["src/tooflashy/analysis.py"],
            "callable_source_sha256": TOOFLASHY_PARITY_CALLABLE_HASHES["analyze_frames"],
        },
    }
    dependency_versions = {
        "tooflashy": "0.1.0",
        "numpy": "2.4.4",
        "opencv-python": "4.13.0.92",
        "opencv-python-headless": None,
    }
    dependency_evidence = {
        "tooflashy": tooflashy_dependency_evidence,
        "numpy": {"version": "2.4.4", "file_count": 100, "files_sha256": "2" * 64},
        "opencv-python": {"version": "4.13.0.92", "file_count": 50, "files_sha256": "3" * 64},
        "opencv-python-headless": None,
    }
    decoder_executables = {
        "ffmpeg": {
            "path": "/usr/bin/ffmpeg",
            "sha256": "4" * 64,
            "version_stdout_sha256": "5" * 64,
            "version_stderr_sha256": "6" * 64,
        },
        "ffprobe": {
            "path": "/usr/bin/ffprobe",
            "sha256": "7" * 64,
            "version_stdout_sha256": "8" * 64,
            "version_stderr_sha256": "9" * 64,
        },
    }
    decode = {
        "engine": "ffmpeg",
        "fps": 60.0,
        "frame_count": 1,
        "pixel_format": "rgb24",
        "ledger": [{
            "index": 0,
            "cfr_timestamp_us": 0,
            "shape": [4, 6, 3],
            "pixel_format": "rgb24",
            "rgb_sha256": "a" * 64,
        }],
    }
    decode["ledger_sha256"] = external_league._canonical_json_sha256(decode["ledger"])
    official = {
        "path": str(video),
        "passes": True,
        "fps": 60.0,
        "frame_count": 1,
        "event_count": 0,
        "failures": [],
    }
    adapter_payload = {
        "schema": "flashpatch-l7-tooflashy-child-adapter-v1",
        "evidence_origin": "live_generator_append_immediately_before_yield_v1",
        "adapter_source_sha256": adapter_hash,
        "input": {"path": str(video), "sha256": "b" * 64},
        "public_api": public_api,
        "runtime": {
            "python_executable": "/usr/bin/python3.12",
            "python_executable_sha256": "c" * 64,
            "python_version": "3.12-test",
            "environment": {},
            "sys_path": [],
            "dependency_versions": dependency_versions,
            "dependency_evidence": dependency_evidence,
            "decoder_executables": decoder_executables,
        },
        "decode": decode,
        "result": {
            **official,
            "event_representation": {"event_count": 0, "failures": []},
        },
    }
    fixture_root = adapter_root / "fixture-000"
    fixture_root.mkdir()
    adapter_output = fixture_root / "adapter-output.json"
    cli_output = fixture_root / "official-cli-output.json"
    _write_json(adapter_output, adapter_payload)
    _write_json(cli_output, official)
    canonical_input = {
        "video": {"path": str(video), "sha256": "b" * 64, "bytes": 10},
        "conversion_receipt": {"path": str(conversion), "sha256": "d" * 64},
        "frame_map_sha256": "e" * 64,
    }
    environment = {
        "HOME": str(adapter_root / "home"),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": str(cache_root),
        "UV_PROJECT": str(checkout),
        "UV_PROJECT_ENVIRONMENT": str(environment_root),
    }
    fixture = {
        "ordinal": 0,
        "fixture_id": "canonical",
        "role": "CANONICAL",
        "input": {"path": str(video), "sha256": "b" * 64, "bytes": 10},
        "adapter_output": {
            "path": str(adapter_output.relative_to(adapter_root)),
            "exists": True,
            "sha256": _sha256(adapter_output),
        },
        "official_cli_output": {
            "path": str(cli_output.relative_to(adapter_root)),
            "sha256": _sha256(cli_output),
        },
    }
    upstream = {
        "path": str(checkout),
        "repository_url": "https://github.com/hashb/TooFlashy",
        "revision": external_league.TOOFLASHY_PARITY_ADAPTER_REVISION,
        "tree": external_league.TOOFLASHY_PARITY_ADAPTER_TREE,
        "license": "Apache-2.0",
        "source_sha256": source_hashes,
    }
    verified = {
        "schema": TOOFLASHY_PARITY_ADAPTER_SCHEMA,
        "adapter_source_sha256": adapter_hash,
        "upstream": upstream,
        "dependency_lock": [
            {"path": relative, "sha256": TOOFLASHY_PARITY_SOURCE_HASHES[relative]}
            for relative in ("pyproject.toml", "uv.lock")
        ],
        "canonical_input": canonical_input,
        "environment_contract": {
            "environment": environment,
            "sha256": external_league._canonical_json_sha256(environment),
        },
        "runner": {"uv": "/usr/local/bin/uv", "uv_sha256": "f" * 64, "timeout_seconds": 120},
        "fixtures": [fixture],
        "status": "VERIFIED",
        "parity_reason": None,
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
    }
    receipt_path = adapter_root / "tooflashy-parity-adapter-receipt.json"
    _write_json(receipt_path, {"fixture": root.name})
    return receipt_path, verified, environment_file


def _install_copied_replay_stub_verifier(
    monkeypatch: pytest.MonkeyPatch,
    pairs: list[tuple[Path, dict[str, object]]],
) -> None:
    by_path = {str(path.resolve()): payload for path, payload in pairs}
    monkeypatch.setattr(
        external_league,
        "verify_tooflashy_parity_adapter",
        lambda receipt, canonical_video, conversion_receipt: copy.deepcopy(
            by_path[str(Path(receipt).resolve())]
        ),
    )


def _copied_replay_install_paths(receipt: Path) -> dict[str, Path]:
    environment = receipt.parent / "uv-environment"
    dist_infos = list(
        environment.glob("lib/python*/site-packages/tooflashy-0.1.0.dist-info")
    )
    assert len(dist_infos) == 1
    dist_info = dist_infos[0]
    return {
        "environment": environment,
        "dist_info": dist_info,
        "site_packages": dist_info.parent,
        "console": environment / "bin" / "tooflashy",
    }


def test_tooflashy_copied_replay_requires_distinct_storage_and_remains_unscoreable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"canonical")
    conversion.write_text("{}\n", encoding="utf-8")
    primary_path, primary, _ = _copied_replay_stub(
        tmp_path / "primary", video=video, conversion=conversion
    )
    replay_path, replay, _ = _copied_replay_stub(
        tmp_path / "replay", video=video, conversion=conversion
    )
    _install_copied_replay_stub_verifier(
        monkeypatch,
        [(primary_path, primary), (replay_path, replay)],
    )

    result = verify_tooflashy_copied_replay_witness(
        primary_path,
        replay_path,
        video,
        conversion,
        destination=tmp_path / "witness.json",
    )

    assert result["schema"] == TOOFLASHY_COPIED_REPLAY_WITNESS_SCHEMA
    assert result["execution_witness_status"] == "COPIED_LOCKED_RUNTIME_REPLAY_VERIFIED"
    assert result["private_storage_independence"]["shared_inode_count"] == 0
    assert result["status"] == "VERIFIED"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert "independent_gold_not_verified" in result["claim_blockers"]
    assert "equal_budget_three_repeat_fair_runtime_receipts_missing" in result["claim_blockers"]
    primary_output = primary_path.parent / primary["fixtures"][0]["adapter_output"]["path"]
    replay_output = replay_path.parent / replay["fixtures"][0]["adapter_output"]["path"]
    primary_raw = json.loads(primary_output.read_text(encoding="utf-8"))
    replay_raw = json.loads(replay_output.read_text(encoding="utf-8"))
    assert (
        primary_raw["runtime"]["dependency_evidence"]["tooflashy"]["files_sha256"]
        != replay_raw["runtime"]["dependency_evidence"]["tooflashy"]["files_sha256"]
    )
    normalized = result["replay_agreement"]["fixtures"][0]["dependency_evidence"]["tooflashy"]
    assert normalized["direct_url"] == {"editable": True, "target": "OWN_CHECKOUT"}
    assert normalized["pth"] == {"target": "OWN_CHECKOUT/src"}
    assert normalized["uv_cache"]["timestamp"] == "UV_BUILD_TIMESTAMP"


def test_tooflashy_copied_replay_rejects_hardlinked_private_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"canonical")
    conversion.write_text("{}\n", encoding="utf-8")
    primary_path, primary, primary_environment_file = _copied_replay_stub(
        tmp_path / "primary", video=video, conversion=conversion
    )
    replay_path, replay, replay_environment_file = _copied_replay_stub(
        tmp_path / "replay", video=video, conversion=conversion
    )
    replay_environment_file.unlink()
    __import__("os").link(primary_environment_file, replay_environment_file)
    _install_copied_replay_stub_verifier(
        monkeypatch,
        [(primary_path, primary), (replay_path, replay)],
    )

    with pytest.raises(external_league.ExternalLeagueError, match="shares inodes"):
        verify_tooflashy_copied_replay_witness(
            primary_path,
            replay_path,
            video,
            conversion,
        )


def test_tooflashy_copied_replay_rejects_reused_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"canonical")
    conversion.write_text("{}\n", encoding="utf-8")
    primary_path, primary, _ = _copied_replay_stub(
        tmp_path / "primary", video=video, conversion=conversion
    )
    replay_path, replay, _ = _copied_replay_stub(
        tmp_path / "replay", video=video, conversion=conversion
    )
    replay["upstream"]["path"] = primary["upstream"]["path"]
    _install_copied_replay_stub_verifier(
        monkeypatch,
        [(primary_path, primary), (replay_path, replay)],
    )

    with pytest.raises(external_league.ExternalLeagueError, match="second source checkout"):
        verify_tooflashy_copied_replay_witness(
            primary_path,
            replay_path,
            video,
            conversion,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("source", "source drifted"),
        ("metadata", "semantic file drifted: METADATA"),
        ("entry_point", "semantic file drifted: entry_points.txt"),
        ("direct_url", "direct_url target is not its own checkout"),
        ("pth", "PTH target is not its own checkout src"),
        ("uv_cache_semantic", "uv_cache semantic fields drifted"),
        ("console", "console script drifted"),
        ("record", "RECORD contains unallowlisted drift"),
        ("unallowlisted_file", "dist-info contains unallowlisted drift"),
        ("version", "dependency evidence is invalid"),
    ],
)
def test_tooflashy_copied_replay_rejects_editable_install_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"canonical")
    conversion.write_text("{}\n", encoding="utf-8")
    primary_path, primary, _ = _copied_replay_stub(
        tmp_path / "primary", video=video, conversion=conversion
    )
    replay_path, replay, _ = _copied_replay_stub(
        tmp_path / "replay", video=video, conversion=conversion
    )
    paths = _copied_replay_install_paths(replay_path)
    checkout = Path(str(replay["upstream"]["path"]))
    if drift == "source":
        (checkout / "pyproject.toml").write_text("mutated\n", encoding="utf-8")
    elif drift == "metadata":
        (paths["dist_info"] / "METADATA").write_bytes(b"mutated metadata")
    elif drift == "entry_point":
        (paths["dist_info"] / "entry_points.txt").write_text(
            "[console_scripts]\ntooflashy = attacker:main\n",
            encoding="utf-8",
        )
    elif drift == "direct_url":
        (paths["dist_info"] / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": (tmp_path / "outside-checkout").resolve().as_uri(),
                    "dir_info": {"editable": True},
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    elif drift == "pth":
        (paths["site_packages"] / "tooflashy.pth").write_text(
            str((tmp_path / "outside-checkout" / "src").resolve()),
            encoding="utf-8",
        )
    elif drift == "uv_cache_semantic":
        cache_path = paths["dist_info"] / "uv_cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["commit"] = "unexpected"
        cache_path.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    elif drift == "console":
        paths["console"].write_bytes(paths["console"].read_bytes() + b"# injected\n")
    elif drift == "record":
        record = paths["dist_info"] / "RECORD"
        record.write_text(
            record.read_text(encoding="utf-8") + "rogue.py,,\n",
            encoding="utf-8",
        )
    elif drift == "unallowlisted_file":
        (paths["dist_info"] / "rogue.json").write_text("{}", encoding="utf-8")
    else:
        fixture = replay["fixtures"][0]
        adapter_output = replay_path.parent / fixture["adapter_output"]["path"]
        adapter = json.loads(adapter_output.read_text(encoding="utf-8"))
        adapter["runtime"]["dependency_evidence"]["tooflashy"]["version"] = "9.9.9"
        _write_json(adapter_output, adapter)
    _install_copied_replay_stub_verifier(
        monkeypatch,
        [(primary_path, primary), (replay_path, replay)],
    )

    with pytest.raises(external_league.ExternalLeagueError, match=message):
        verify_tooflashy_copied_replay_witness(
            primary_path,
            replay_path,
            video,
            conversion,
        )


@pytest.mark.parametrize(
    "drift",
    ["source_tree", "lockfile", "adapter", "canonical", "runner", "dependency", "ffmpeg"],
)
def test_tooflashy_copied_replay_rejects_frozen_identity_and_closure_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    video = tmp_path / "canonical.ffv1.mkv"
    conversion = tmp_path / "conversion.json"
    video.write_bytes(b"canonical")
    conversion.write_text("{}\n", encoding="utf-8")
    primary_path, primary, _ = _copied_replay_stub(
        tmp_path / "primary", video=video, conversion=conversion
    )
    replay_path, replay, _ = _copied_replay_stub(
        tmp_path / "replay", video=video, conversion=conversion
    )
    if drift == "source_tree":
        replay["upstream"]["tree"] = "0" * 40
    elif drift == "lockfile":
        replay["dependency_lock"][1]["sha256"] = "0" * 64
    elif drift == "adapter":
        replay["adapter_source_sha256"] = "0" * 64
    elif drift == "canonical":
        replay["canonical_input"]["frame_map_sha256"] = "0" * 64
    elif drift == "runner":
        replay["runner"]["uv_sha256"] = "0" * 64
    else:
        fixture = replay["fixtures"][0]
        adapter_output = replay_path.parent / fixture["adapter_output"]["path"]
        adapter_payload = json.loads(adapter_output.read_text(encoding="utf-8"))
        if drift == "dependency":
            adapter_payload["runtime"]["dependency_evidence"]["numpy"]["files_sha256"] = "0" * 64
        else:
            adapter_payload["runtime"]["decoder_executables"]["ffmpeg"]["sha256"] = "0" * 64
        _write_json(adapter_output, adapter_payload)
    _install_copied_replay_stub_verifier(
        monkeypatch,
        [(primary_path, primary), (replay_path, replay)],
    )

    with pytest.raises(
        external_league.ExternalLeagueError,
        match="frozen identity differs|dependency closure or result differs",
    ):
        verify_tooflashy_copied_replay_witness(
            primary_path,
            replay_path,
            video,
            conversion,
        )
