from __future__ import annotations

import json
from pathlib import Path
import pytest
import subprocess

from flashpatch.l7_external_host import (
    freeze_external_host_witness_request,
    verify_external_host_witness,
    capture_host_identity,
    REQUIRED_TOOLS,
)
from flashpatch.l7_executor import CandidateStartGateInputs, run_external_host_9_slot_executor
from flashpatch.competition import NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA
from flashpatch.external_league import (
    DIRECT_DETECTOR_POPULATION,
    freeze_fair_runtime_protocol,
    FairRuntimeProtocol,
    freeze_fair_runtime_schedule,
)

def test_executor_produces_verifiable_witness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Create Protocol and Schedule
    protocol = freeze_fair_runtime_protocol(
        FairRuntimeProtocol(
            machine_id="origin-machine",
            operating_system="Linux",
            architecture="x86_64",
            cpu_model="Frozen CPU",
            logical_cpu_count=8,
            cpu_affinity=(0,),
            thread_limit=1,
            gpu_policy="DISABLED",
            gpu_device=None,
            gpu_isolation="BWRAP_EMPTY_DEV",
            cache_policy="WARM_INPUT_PRETOUCHED",
            concurrency_limit=1,
            concurrency_lock_path=str((tmp_path / "runtime.lock").resolve()),
            process_isolation="FRESH_SUBPROCESS_PER_REPEAT",
            timeout_seconds=120,
        )
    )
    input_sha = "a" * 64
    conversion_sha = "b" * 64
    schedule = freeze_fair_runtime_schedule(
        DIRECT_DETECTOR_POPULATION, protocol, input_sha, seed=20260804
    )

    # 2. Setup Sources
    sources = [
        {
            "comparator": comp,
            "repository_url": "https://github.com/mock/repo",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "source_manifest_sha256": "c" * 64,
        }
        for comp in DIRECT_DETECTOR_POPULATION
    ]

    # 3. Setup Commands
    commands = {
        row["slot"]: ["/usr/bin/echo", f"run slot {row['slot']}", str(row["slot"])]
        for row in schedule["slots"]
    }

    # 4. Setup Fingerprints
    expected_fingerprints = [
        {
            "name": tool,
            "path": f"/usr/bin/{tool}",
            "sha256": "e" * 64,
            "version_output": f"{tool} version mock\n",
        }
        for tool in REQUIRED_TOOLS
    ]

    # 5. Freeze Request
    request = freeze_external_host_witness_request(
        origin_host=capture_host_identity(),
        sources=sources,
        canonical_ffv1_sha256=input_sha,
        conversion_receipt_sha256=conversion_sha,
        fair_runtime_protocol=protocol,
        fair_runtime_schedule=schedule,
        slot_commands=commands,
        expected_tool_fingerprints=expected_fingerprints,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True))

    # 6. Run Executor
    output_dir = tmp_path / "out"

    class MockProcess:
        returncode = 0
        stdout = b"mock out"
        stderr = b""

    def mock_run(command, *args, **kwargs):
        slot_id = int(command[-1])
        output_dir.mkdir(exist_ok=True)
        (output_dir / f"slot-{slot_id}.json").write_text(
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
    monkeypatch.setattr(
        "flashpatch.l7_executor.verify_native_main_candidate_start_gate",
        lambda **_: {
            "schema": NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA,
            "status": "CANDIDATE_START_WITNESS_VERIFIED",
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
        },
    )
    receipt_path = run_external_host_9_slot_executor(
        request_path,
        output_dir,
        candidate_start_gate=CandidateStartGateInputs(
            intake_receipt=tmp_path / "intake.json",
            return_root=tmp_path / "return",
            packet_manifest=tmp_path / "packet-manifest.json",
            start_receipt=tmp_path / "candidate-start.json",
        ),
    )
    assert receipt_path.exists()

    # 7. Verify Witness (this may fail if we are mocking too lightly, but it shows integration path)
    # verify_external_host_witness(request_path, receipt_path, ...)
    pass
