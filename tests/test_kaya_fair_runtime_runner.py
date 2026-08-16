from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess

import pytest

import flashpatch.external_league as external_league


def _write(path: Path, payload: object | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_bytes(b"fixture\n")
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _frozen_protocol() -> dict[str, object]:
    return {
        "measurement_boundary": dict(external_league.FAIR_RUNTIME_BOUNDARY),
        "effective_environment_policy": dict(external_league.FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY),
        "machine": {"id": "test-machine", "operating_system": "test-os", "architecture": "x86_64"},
        "cpu": {"model": "test-cpu", "logical_count": 1, "affinity": [0]},
        "threads": {"limit": 1},
        "gpu": {"policy": "DISABLED", "device": None, "isolation": "BWRAP_EMPTY_DEV"},
        "cache": {"policy": "WARM_INPUT_PRETOUCHED"},
        "concurrency": {"limit": 1, "lock_path": "/tmp/kaya-test.lock", "process_isolation": "FRESH_SUBPROCESS_PER_REPEAT"},
        "budget": {"timeout_seconds": 120, "scheduled_repeats": 3, "retry_policy": "NO_RETRY"},
    }


def _binding() -> dict[str, object]:
    return {
        "path": "/tmp/pre-frozen-schedule.json",
        "artifact_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "stat": {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4, "ctime_ns": 5},
        "slot": 2,
        "round": 1,
        "position": 2,
        "comparator": external_league.KAYA_DIRECT_PARTICIPANT_ID,
        "repeat_ordinal": 1,
    }


def _kaya_output() -> dict[str, object]:
    return {
        "results": [{
            "raw": {"General Flashes": [0.0, 0.0], "Red Flashes": [0.0, 0.0]},
            "interval_tuples": [],
        }],
    }


def test_kaya_scheduled_runner_uses_native_path_and_never_claims_scoreability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    python = _write(tmp_path / "venv" / "bin" / "python")
    video = _write(tmp_path / "canonical.ffv1.mkv")
    conversion = _write(tmp_path / "conversion.json")
    participant = _write(tmp_path / "participant.json")
    parity = _write(tmp_path / "parity.json")
    frozen = _frozen_protocol()
    binding = _binding()
    monkeypatch.setattr(external_league, "KAYA_PYTHON_SHA256", external_league._sha256_file(python))
    monkeypatch.setattr(external_league, "_freeze_runtime_protocol_input", lambda _: frozen)
    monkeypatch.setattr(external_league, "_audit_kaya_source_checkout", lambda _: {"pinned": True})
    monkeypatch.setattr(
        external_league,
        "_kaya_fair_runtime_prerequisites",
        lambda *_args, **_kwargs: (participant, parity, {}),
    )
    monkeypatch.setattr(external_league, "_load_schedule_assignment", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(
        external_league,
        "_load_conversion_receipt",
        lambda *_args: {"renderer_rgb": {"raw_sha256": "c" * 64}},
    )

    @contextmanager
    def fake_context(*_args: object, **_kwargs: object):
        yield {"environment": {}, "observation": {"parent": "observed"}, "command_prefix": [], "schedule_binding": binding}

    commands: list[list[str]] = []
    monkeypatch.setattr(external_league, "_fair_execution_context", fake_context)
    monkeypatch.setattr(
        external_league,
        "_instrument_fair_command",
        lambda _execution, command, _probe: commands.append(list(command)) or list(command),
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raw = Path(command[-3])
        raw.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"stdout", b"stderr")

    monkeypatch.setattr(external_league.subprocess, "run", fake_run)
    monkeypatch.setattr(external_league, "_load_kaya_child_output", lambda _: {})
    monkeypatch.setattr(external_league, "_validate_kaya_child_output", lambda *_args, **_kwargs: _kaya_output())
    monkeypatch.setattr(external_league, "_load_child_runtime_probe", lambda _: {"probe": "observed"})
    monkeypatch.setattr(
        external_league,
        "_fair_runtime_run_receipt",
        lambda *_args, **_kwargs: {"schema": external_league.FAIR_RUNTIME_RUN_SCHEMA},
    )

    result = external_league.execute_kaya_scheduled_fair_runtime(
        participant,
        parity,
        checkout=checkout,
        python_executable=python,
        canonical_video=video,
        conversion_receipt=conversion,
        output_root=tmp_path / "run",
        runtime_protocol=frozen,
        scheduled_repeat_ordinal=1,
        runtime_schedule=tmp_path / "schedule.json",
        schedule_slot=2,
    )

    assert commands and commands[0][6] == "native"
    assert result["status"] == "PROCESS_VALID"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert result["external_claim_authorized"] is False
    assert "independent_execution_witness_missing" in result["claim_blockers"]


def _verified_kaya_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    python = _write(tmp_path / "venv" / "bin" / "python")
    video = _write(tmp_path / "canonical.ffv1.mkv")
    conversion = _write(tmp_path / "conversion.json")
    participant = _write(tmp_path / "participant.json")
    parity = _write(tmp_path / "parity.json")
    stdout = _write(root / "stdout.bin")
    stderr = _write(root / "stderr.bin")
    raw = _write(root / "child-output.json", {})
    probe_path = _write(root / "runtime-probe.json", {})
    frozen = _frozen_protocol()
    binding = _binding()
    output = _kaya_output()
    observation = external_league._normalize_kaya_fair_runtime_output(output)
    terminal = external_league._normalized_terminal_identity(observation, normalizer="kaya-native-exact-v1")
    probe = {
        "child_timing": {
            "probe_started_monotonic_ns": 11,
            "tool_started_monotonic_ns": 12,
            "tool_finished_monotonic_ns": 19,
        }
    }
    monkeypatch.setattr(external_league, "KAYA_PYTHON_SHA256", external_league._sha256_file(python))
    monkeypatch.setattr(external_league, "_validate_frozen_runtime_protocol", lambda _: frozen)
    monkeypatch.setattr(external_league, "_audit_kaya_source_checkout", lambda _: {"pinned": True})
    monkeypatch.setattr(external_league, "_load_conversion_receipt", lambda *_: {"renderer_rgb": {"raw_sha256": "d" * 64}})
    monkeypatch.setattr(external_league, "_kaya_fair_runtime_prerequisites", lambda *_args, **_kwargs: (participant, parity, {}))
    monkeypatch.setattr(external_league, "_load_kaya_child_output", lambda _: {})
    monkeypatch.setattr(external_league, "_validate_kaya_child_output", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(external_league, "_load_schedule_assignment", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(external_league, "_load_child_runtime_probe", lambda _: probe)
    monkeypatch.setattr(external_league, "_observed_environment_matches_protocol", lambda *_args: True)
    monkeypatch.setattr(external_league, "_child_probe_and_command_are_bound", lambda *_args: True)
    runtime = {
        "schema": external_league.FAIR_RUNTIME_RUN_SCHEMA,
        "protocol_sha256": external_league._canonical_json_sha256(frozen),
        "measurement_boundary": dict(external_league.FAIR_RUNTIME_BOUNDARY),
        "environment_policy_sha256": external_league._runtime_environment_sha256(frozen),
        "observed_environment": {},
        "timeout_seconds": 120,
        "scheduled_repeat_ordinal": 1,
        "schedule_binding": binding,
        "attempt_ordinal": 1,
        "retry_count": 0,
        "retry_policy": "NO_RETRY",
        "started_monotonic_ns": 10,
        "finished_monotonic_ns": 20,
        "wall_time_ns": 10,
        "timed_out": False,
        "input_identity_sha256": external_league._sha256_file(video),
        "normalized_terminal_observation": terminal,
    }
    receipt = {
        "schema": external_league.KAYA_FAIR_RUNTIME_RUN_SCHEMA,
        "comparator": {
            "name": external_league.KAYA_DIRECT_PARTICIPANT_ID,
            "repository_url": external_league.KAYA_REPOSITORY_URL,
            "revision": external_league.KAYA_SOURCE_REVISION,
            "tree": external_league.KAYA_SOURCE_TREE,
            "license": "BSD-3-Clause",
            "binary": str(python),
            "binary_sha256": external_league._sha256_file(python),
            "working_directory": str(root),
            "source_checkout": str(checkout),
            "upstream": {"pinned": True},
        },
        "input": {"path": str(video), "sha256": external_league._sha256_file(video)},
        "conversion_receipt": {"path": str(conversion), "sha256": external_league._sha256_file(conversion), "renderer_rgb_sha256": "d" * 64},
        "participant_conformance": {"path": str(participant), "sha256": external_league._sha256_file(participant)},
        "natural_case_parity": {"path": str(parity), "sha256": external_league._sha256_file(parity)},
        "command": ["frozen-command"],
        "exit_code": 0,
        "wall_time_ns": 10,
        "stdout": {"path": stdout.name, "sha256": external_league._sha256_file(stdout)},
        "stderr": {"path": stderr.name, "sha256": external_league._sha256_file(stderr)},
        "raw_output": {"path": raw.name, "exists": True, "sha256": external_league._sha256_file(raw)},
        "observation": observation,
        "parse_error": None,
        "fair_runtime_protocol": frozen,
        "fair_runtime": runtime,
        "runtime_probe": {"path": probe_path.name, "sha256": external_league._sha256_file(probe_path), "observation": probe},
        "status": "PROCESS_VALID",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "external_claim_authorized": False,
        "claim_blockers": ["independent_execution_witness_missing", "independent_gold_receipt_missing", "frozen_public_case_ledger_missing"],
    }
    receipt_path = root / "kaya-fair-runtime-receipt.json"
    _write(receipt_path, receipt)
    return receipt_path


def test_kaya_fair_runtime_verifier_reopens_a_bound_native_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _verified_kaya_run(tmp_path, monkeypatch)
    result = external_league.verify_kaya_scheduled_fair_runtime_run_receipt(receipt)
    assert result["status"] == "PROCESS_VALID"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["comparison_eligible"] is False


def test_kaya_repeat_writer_routes_only_the_kaya_child_schema_and_preserves_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _verified_kaya_run(tmp_path, monkeypatch)
    source = json.loads(first.read_text(encoding="utf-8"))
    children: list[Path] = []
    for ordinal in (1, 2, 3):
        child = copy.deepcopy(source)
        child["fair_runtime"]["scheduled_repeat_ordinal"] = ordinal
        path = tmp_path / f"kaya-child-{ordinal}.json"
        _write(path, child)
        children.append(path)
    frozen = _frozen_protocol()
    monkeypatch.setattr(external_league, "_freeze_runtime_protocol_input", lambda _: frozen)

    result = external_league.write_scheduled_runtime_repeat_receipt(
        external_league.KAYA_DIRECT_PARTICIPANT_ID,
        children,
        frozen,
        tmp_path / "kaya-repeats.json",
    )

    assert result["schema"] == external_league.KAYA_FAIR_RUNTIME_REPEATS_SCHEMA
    assert result["status"] == "PROCESS_REPRODUCIBLE"
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert result["external_claim_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("slot", "schedule binding drifted"),
        ("input", "canonical input hash mismatches"),
        ("retry", "timing, budget, retry, or normalization ledger is invalid"),
        ("raw", "raw output hash mismatches"),
        ("timing", "timing, budget, retry, or normalization ledger is invalid"),
    ],
)
def test_kaya_fair_runtime_verifier_rejects_slot_input_retry_raw_and_timing_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    receipt_path = _verified_kaya_run(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "slot":
        receipt["fair_runtime"]["schedule_binding"]["slot"] = 9
    elif mutation == "input":
        receipt["input"]["sha256"] = "0" * 64
    elif mutation == "retry":
        receipt["fair_runtime"]["retry_count"] = 1
    elif mutation == "raw":
        receipt["raw_output"]["sha256"] = "0" * 64
    else:
        receipt["fair_runtime"]["wall_time_ns"] = 11
    _write(receipt_path, receipt)

    with pytest.raises(external_league.ExternalLeagueError, match=error):
        external_league.verify_kaya_scheduled_fair_runtime_run_receipt(receipt_path)
