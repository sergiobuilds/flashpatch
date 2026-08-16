"""Tests for the L7 transport bundle builder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashpatch.l7_external_host import (
    freeze_external_host_witness_request,
    verify_external_host_witness,
    capture_host_identity,
    canonical_sha256,
    REQUIRED_TOOLS,
    ExternalHostWitnessError,
)
from flashpatch.external_league import (
    DIRECT_DETECTOR_POPULATION,
    freeze_fair_runtime_protocol,
    FairRuntimeProtocol,
    freeze_fair_runtime_schedule,
)
from flashpatch.l7_transport_bundle import build_transport_bundle


def _make_protocol(tmp_path: Path):
    return freeze_fair_runtime_protocol(
        FairRuntimeProtocol(
            machine_id="coord-machine",
            operating_system="Linux",
            architecture="x86_64",
            cpu_model="Test CPU",
            logical_cpu_count=4,
            cpu_affinity=(0,),
            thread_limit=1,
            gpu_policy="DISABLED",
            gpu_device=None,
            gpu_isolation="BWRAP_EMPTY_DEV",
            cache_policy="WARM_INPUT_PRETOUCHED",
            concurrency_limit=1,
            concurrency_lock_path=str(tmp_path / "lock"),
            process_isolation="FRESH_SUBPROCESS_PER_REPEAT",
            timeout_seconds=120,
        )
    )


def _make_sources(tmp_path: Path):
    sm_paths = {}
    sources = []
    for comp in DIRECT_DETECTOR_POPULATION:
        mbytes = json.dumps({"comparator": comp}, sort_keys=True).encode("utf-8")
        mf = tmp_path / f"sm-{comp}.json"
        mf.write_bytes(mbytes)
        sm_paths[comp] = mf
        sources.append({
            "comparator": comp,
            "repository_url": f"https://github.com/test/{comp}",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "source_manifest_sha256": hashlib.sha256(mbytes).hexdigest(),
        })
    return sources, sm_paths


def _make_fingerprints():
    return [
        {"name": t, "path": f"/usr/bin/{t}", "sha256": "e" * 64, "version_output": f"{t}\n"}
        for t in REQUIRED_TOOLS
    ]


def _make_slot_results(tmp_path: Path, request):
    results = []
    mono = 1_000_000_000
    for cmd_row in request["fair_runtime"]["commands"]:
        sn = cmd_row["slot"]
        rf = tmp_path / f"result-{sn}.json"
        rf.write_text(json.dumps({"slot": sn, "prediction": "SAFE"}))
        results.append({
            "slot": sn,
            "started_monotonic_ns": mono,
            "finished_monotonic_ns": mono + 100_000_000,
            "exit_code": 0,
            "stdout": b"output",
            "stderr": b"",
            "result_path": str(rf),
        })
        mono += 200_000_000
    return results


EXTERNAL_IDENTITY = {
    "machine_id": "docker-container-abc",
    "hostname": "abc",
    "operating_system": "Linux",
    "kernel": "6.8.0",
    "architecture": "x86_64",
    "cpu_model": "Container CPU",
    "logical_cpu_count": 4,
}


def test_transport_bundle_passes_witness_verification(tmp_path: Path) -> None:
    protocol = _make_protocol(tmp_path)
    schedule = freeze_fair_runtime_schedule(DIRECT_DETECTOR_POPULATION, protocol, "a" * 64, seed=42)
    sources, sm_paths = _make_sources(tmp_path)
    commands = {row["slot"]: ["/bin/echo", "test", str(row["slot"])] for row in schedule["slots"]}

    request = freeze_external_host_witness_request(
        origin_host=capture_host_identity(),
        sources=sources,
        canonical_ffv1_sha256="a" * 64,
        conversion_receipt_sha256="b" * 64,
        fair_runtime_protocol=protocol,
        fair_runtime_schedule=schedule,
        slot_commands=commands,
        expected_tool_fingerprints=_make_fingerprints(),
    )
    rp = tmp_path / "request.json"
    rp.write_bytes(json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))

    slot_results = _make_slot_results(tmp_path, request)
    receipt_path = build_transport_bundle(
        request,
        bundle_root=tmp_path / "bundle",
        slot_results=slot_results,
        host_identity=EXTERNAL_IDENTITY,
        workspace_root="/workspace",
        source_manifest_paths=sm_paths,
        observed_tool_fingerprints=_make_fingerprints(),
    )

    result = verify_external_host_witness(
        rp, receipt_path,
        expected_protocol_sha256=canonical_sha256(protocol),
        expected_schedule_sha256=canonical_sha256(schedule),
        expected_input_sha256="a" * 64,
    )
    assert result["status"] == "VERIFIED"
    assert result["witness_verified"] is True
    assert len(result["verified_slots"]) == 9


def test_transport_bundle_rejects_non_empty_root(tmp_path: Path) -> None:
    protocol = _make_protocol(tmp_path)
    schedule = freeze_fair_runtime_schedule(DIRECT_DETECTOR_POPULATION, protocol, "a" * 64, seed=42)
    sources, sm_paths = _make_sources(tmp_path)
    commands = {row["slot"]: ["/bin/echo", "test", str(row["slot"])] for row in schedule["slots"]}

    request = freeze_external_host_witness_request(
        origin_host=capture_host_identity(),
        sources=sources,
        canonical_ffv1_sha256="a" * 64,
        conversion_receipt_sha256="b" * 64,
        fair_runtime_protocol=protocol,
        fair_runtime_schedule=schedule,
        slot_commands=commands,
        expected_tool_fingerprints=_make_fingerprints(),
    )
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "stale.txt").write_text("stale")

    with pytest.raises(ExternalHostWitnessError, match="must be empty"):
        build_transport_bundle(
            request,
            bundle_root=bundle_root,
            slot_results=[],
            host_identity=EXTERNAL_IDENTITY,
            observed_tool_fingerprints=_make_fingerprints(),
        )


def test_transport_bundle_rejects_observed_tool_drift(tmp_path: Path) -> None:
    protocol = _make_protocol(tmp_path)
    schedule = freeze_fair_runtime_schedule(DIRECT_DETECTOR_POPULATION, protocol, "a" * 64, seed=42)
    sources, sm_paths = _make_sources(tmp_path)
    request = freeze_external_host_witness_request(
        origin_host=capture_host_identity(),
        sources=sources,
        canonical_ffv1_sha256="a" * 64,
        conversion_receipt_sha256="b" * 64,
        fair_runtime_protocol=protocol,
        fair_runtime_schedule=schedule,
        slot_commands={row["slot"]: ["/bin/echo", "test", str(row["slot"])] for row in schedule["slots"]},
        expected_tool_fingerprints=_make_fingerprints(),
    )
    observed = _make_fingerprints()
    observed[0]["sha256"] = "f" * 64
    with pytest.raises(ExternalHostWitnessError, match="differ from the frozen request"):
        build_transport_bundle(
            request,
            bundle_root=tmp_path / "bundle",
            slot_results=_make_slot_results(tmp_path, request),
            host_identity=EXTERNAL_IDENTITY,
            source_manifest_paths=sm_paths,
            observed_tool_fingerprints=observed,
        )


def test_transport_bundle_rejects_wrong_slot_count(tmp_path: Path) -> None:
    protocol = _make_protocol(tmp_path)
    schedule = freeze_fair_runtime_schedule(DIRECT_DETECTOR_POPULATION, protocol, "a" * 64, seed=42)
    sources, sm_paths = _make_sources(tmp_path)
    commands = {row["slot"]: ["/bin/echo", "test", str(row["slot"])] for row in schedule["slots"]}

    request = freeze_external_host_witness_request(
        origin_host=capture_host_identity(),
        sources=sources,
        canonical_ffv1_sha256="a" * 64,
        conversion_receipt_sha256="b" * 64,
        fair_runtime_protocol=protocol,
        fair_runtime_schedule=schedule,
        slot_commands=commands,
        expected_tool_fingerprints=_make_fingerprints(),
    )

    with pytest.raises(ExternalHostWitnessError, match="exactly 9"):
        build_transport_bundle(
            request,
            bundle_root=tmp_path / "bundle",
            slot_results=[{"slot": 1}],
            host_identity=EXTERNAL_IDENTITY,
            source_manifest_paths=sm_paths,
            observed_tool_fingerprints=_make_fingerprints(),
        )
