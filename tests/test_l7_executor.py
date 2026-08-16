from __future__ import annotations

import json
import pytest
from pathlib import Path

from flashpatch.l7_external_host import ExternalHostWitnessError
from flashpatch.l7_executor import CandidateStartGateInputs, run_external_host_9_slot_executor
from flashpatch.competition import NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA
from flashpatch.external_league import DIRECT_DETECTOR_POPULATION


def _candidate_start_gate_inputs(tmp_path: Path) -> CandidateStartGateInputs:
    return CandidateStartGateInputs(
        intake_receipt=tmp_path / "intake.json",
        return_root=tmp_path / "return",
        packet_manifest=tmp_path / "packet-manifest.json",
        start_receipt=tmp_path / "candidate-start.json",
    )


def _candidate_start_gate_assessment(*, status: str = "CANDIDATE_START_WITNESS_VERIFIED") -> dict[str, object]:
    from flashpatch.external_league import DIRECT_DETECTOR_POPULATION

    return {
        "schema": NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA,
        "status": status,
        "scoreable": False,
        "external_claim_authorized": False,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "slot_count": 9,
        "repeat_count_per_detector": 3,
        "request_schema": "flashpatch-l7-external-host-witness-request-v2",
        "score_blockers": [
            "same_condition_9_slot_execution_missing",
            "receipt_bound_score_bundle_missing",
        ],
    }


def _allow_candidate_start_gate(monkeypatch: pytest.MonkeyPatch, *, status: str = "CANDIDATE_START_WITNESS_VERIFIED") -> None:
    monkeypatch.setattr(
        "flashpatch.l7_executor.verify_native_main_candidate_start_gate",
        lambda **_: _candidate_start_gate_assessment(status=status),
    )


def test_executor_fails_without_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_candidate_start_gate(monkeypatch)
    with pytest.raises(ExternalHostWitnessError, match="request path does not exist"):
        run_external_host_9_slot_executor(
            tmp_path / "missing.json",
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )

def test_executor_requires_recomputed_candidate_start_gate(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v2"}')
    with pytest.raises(ExternalHostWitnessError, match="candidate-start gate verification failed"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )


def test_external_host_cli_rejects_legacy_assessment_only_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys
    from flashpatch import l7_external_host_cli

    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v2"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flashpatch-l7-external-host",
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--candidate-start-gate-assessment",
            str(tmp_path / "forged-assessment.json"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        l7_external_host_cli.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--intake-receipt" in captured.err
    assert "--return-root" in captured.err
    assert "--packet-manifest" in captured.err
    assert "--start-receipt" in captured.err


def test_external_host_cli_passes_original_gate_inputs_to_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys
    from flashpatch import l7_external_host_cli

    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v2"}')
    output_dir = tmp_path / "out"
    receipt_path = output_dir / "external-receipt.json"
    seen: dict[str, object] = {}

    def fake_executor(
        request: Path,
        output: Path,
        *,
        candidate_start_gate: CandidateStartGateInputs,
    ) -> Path:
        seen["request"] = request
        seen["output"] = output
        seen["candidate_start_gate"] = candidate_start_gate
        output.mkdir()
        receipt_path.write_text("{}\n", encoding="utf-8")
        return receipt_path

    monkeypatch.setattr(
        l7_external_host_cli,
        "run_external_host_9_slot_executor",
        fake_executor,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flashpatch-l7-external-host",
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--intake-receipt",
            str(tmp_path / "intake.json"),
            "--return-root",
            str(tmp_path / "return"),
            "--packet-manifest",
            str(tmp_path / "packet-manifest.json"),
            "--start-receipt",
            str(tmp_path / "candidate-start.json"),
        ],
    )

    assert l7_external_host_cli.main() == 0
    captured = capsys.readouterr()
    assert captured.out == f"Success: {receipt_path}\n"
    assert captured.err == ""
    assert seen["request"] == request_path
    assert seen["output"] == output_dir
    assert seen["candidate_start_gate"] == CandidateStartGateInputs(
        intake_receipt=tmp_path / "intake.json",
        return_root=tmp_path / "return",
        packet_manifest=tmp_path / "packet-manifest.json",
        start_receipt=tmp_path / "candidate-start.json",
    )


def test_executor_rejects_blocked_candidate_start_gate_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_candidate_start_gate(monkeypatch, status="CANDIDATE_START_BLOCKED")
    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v2"}')
    with pytest.raises(ExternalHostWitnessError, match="must verify independent gold"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )

def test_executor_rejects_non_v2_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_candidate_start_gate(monkeypatch)
    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v1"}')
    with pytest.raises(ExternalHostWitnessError, match="must be a frozen v2 request"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )

def test_executor_rejects_non_9_slot_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_candidate_start_gate(monkeypatch)
    request_path = tmp_path / "request.json"
    request_path.write_text('{"schema": "flashpatch-l7-external-host-witness-request-v2", "fair_runtime": {"slots": [1,2,3,4,5,6]}}')
    with pytest.raises(ExternalHostWitnessError, match="must have exactly 9 slots"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )


def test_executor_rejects_noncanonical_slot_ordinals_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_candidate_start_gate(monkeypatch)
    population = list(DIRECT_DETECTOR_POPULATION)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({
        "schema": "flashpatch-l7-external-host-witness-request-v2",
        "fair_runtime": {
            "slots": [{"slot": i, "comparator": population[(i - 10) % 3]} for i in range(10, 19)],
            "commands": [{"slot": i, "command": ["/usr/bin/true"]} for i in range(10, 19)],
        },
        "sources": [{"comparator": c} for c in population],
    }))

    with pytest.raises(ExternalHostWitnessError, match="slot identity is invalid"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )


def test_executor_rejects_missing_kaya_participant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flashpatch.external_league import DIRECT_DETECTOR_POPULATION

    _allow_candidate_start_gate(monkeypatch)
    population = list(DIRECT_DETECTOR_POPULATION)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({
        "schema": "flashpatch-l7-external-host-witness-request-v2",
        "fair_runtime": {
            "slots": (
                [{"slot": i, "comparator": population[0]} for i in range(1, 4)]
                + [{"slot": i, "comparator": population[1]} for i in range(4, 7)]
                + [{"slot": i, "comparator": population[2]} for i in range(7, 10)]
            ),
        },
        "sources": [{"comparator": population[0]}, {"comparator": population[2]}],
    }))
    with pytest.raises(ExternalHostWitnessError, match="must contain FlashPatch, TooFlashy, and Kaya participants"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )

def test_executor_rejects_missing_result_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from flashpatch.external_league import DIRECT_DETECTOR_POPULATION

    _allow_candidate_start_gate(monkeypatch)
    request_path = tmp_path / "request.json"
    population = list(DIRECT_DETECTOR_POPULATION)
    request_path.write_text(json.dumps({
        "schema": "flashpatch-l7-external-host-witness-request-v2",
        "expected_tool_fingerprints": [],
        "source_population_sha256": "pop123",
        "canonical_input": {
            "ffv1_sha256": "ffv1_123",
            "conversion_receipt_sha256": "conv_123",
        },
        "fair_runtime": {
            "protocol_sha256": "proto_123",
            "schedule_sha256": "sched_123",
            "slots": [{"slot": i, "comparator": population[(i - 1) % 3]} for i in range(1, 10)],
            "commands": [{"slot": i, "command": ["/usr/bin/true"]} for i in range(1, 10)],
        },
        "sources": [{"comparator": c} for c in population],
    }))

    class MockProcess:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MockProcess())

    with pytest.raises(ExternalHostWitnessError, match="did not produce result artifact"):
        run_external_host_9_slot_executor(
            request_path,
            tmp_path / "out",
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )


def test_executor_rejects_preexisting_output_artifact_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    _allow_candidate_start_gate(monkeypatch)
    request_path = tmp_path / "request.json"
    population = list(DIRECT_DETECTOR_POPULATION)
    request_path.write_text(json.dumps({
        "schema": "flashpatch-l7-external-host-witness-request-v2",
        "expected_tool_fingerprints": [],
        "source_population_sha256": "pop123",
        "canonical_input": {
            "ffv1_sha256": "ffv1_123",
            "conversion_receipt_sha256": "conv_123",
        },
        "fair_runtime": {
            "protocol_sha256": "proto_123",
            "schedule_sha256": "sched_123",
            "slots": [{"slot": i, "comparator": population[(i - 1) % 3]} for i in range(1, 10)],
            "commands": [{"slot": i, "command": ["/usr/bin/true"]} for i in range(1, 10)],
        },
        "sources": [{"comparator": c} for c in population],
    }))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "slot-1.json").write_text('{"stale": true}\n', encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("executor must reject stale output before running commands")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    with pytest.raises(ExternalHostWitnessError, match="output directory must be empty"):
        run_external_host_9_slot_executor(
            request_path,
            output_dir,
            candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
        )


def test_executor_executes_9_slots_and_produces_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    from flashpatch.external_league import DIRECT_DETECTOR_POPULATION

    _allow_candidate_start_gate(monkeypatch)
    request_path = tmp_path / "request.json"
    request_data = {
        "schema": "flashpatch-l7-external-host-witness-request-v2",
        "expected_tool_fingerprints": [],
        "source_population_sha256": "pop123",
        "canonical_input": {
            "ffv1_sha256": "ffv1_123",
            "conversion_receipt_sha256": "conv_123",
        },
        "fair_runtime": {
            "protocol_sha256": "proto_123",
            "schedule_sha256": "sched_123",
            "slots": [
                {"slot": i, "comparator": c}
                for i, c in enumerate(list(DIRECT_DETECTOR_POPULATION) * 3, start=1)
            ],
            "commands": [{"slot": i, "command": ["echo", "test", str(i)]} for i in range(1, 10)]
        },
        "sources": [{"comparator": c} for c in DIRECT_DETECTOR_POPULATION]
    }
    request_path.write_text(json.dumps(request_data))
    
    class MockProcess:
        returncode = 0
        stdout = b""
        stderr = b""

    def mock_run(command, *args, **kwargs):
        slot_id = int(command[-1])
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        (out / f"slot-{slot_id}.json").write_text(
            json.dumps({"slot": slot_id, "result": "PROCESS_VALID"}),
            encoding="utf-8",
        )
        return MockProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(
        "flashpatch.l7_executor.capture_host_identity",
        lambda: {
            "machine_id": "test-machine",
            "hostname": "test-host",
            "operating_system": "Linux",
            "kernel": "test",
            "architecture": "x86_64",
            "cpu_model": "test-cpu",
            "logical_cpu_count": 1,
        },
    )
    
    receipt_path = run_external_host_9_slot_executor(
        request_path,
        tmp_path / "out",
        candidate_start_gate=_candidate_start_gate_inputs(tmp_path),
    )
    
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "flashpatch-l7-external-host-witness-receipt-v2"
    assert len(receipt["slots"]) == 9
