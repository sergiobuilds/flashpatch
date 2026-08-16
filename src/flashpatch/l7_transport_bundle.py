"""Transport bundle builder for L7 external host witness evidence.

This module runs on the external host (e.g., inside a Docker container with a
distinct machine identity).  It takes a pre-frozen V2 request, executes the
9 slot commands, and packages the full evidence into a transport bundle that
``verify_external_host_witness`` on the coordinator host can verify.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .l7_external_host import (
    ExternalHostWitnessError,
    REQUEST_SCHEMA_V2,
    RECEIPT_SCHEMA_V2,
    MANIFEST_SCHEMA,
    PROBE_SCHEMA_V2,
    ROOT_FIELDS,
    canonical_sha256,
    capture_host_identity,
    _file_sha256,
)


def _sha256_file(path: Path) -> str:
    return _file_sha256(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_ref(bundle_root: Path, relative: str) -> dict[str, object]:
    path = bundle_root / relative
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def build_transport_bundle(
    request: Mapping[str, object],
    *,
    bundle_root: Path | str,
    slot_results: Sequence[Mapping[str, object]],
    host_identity: Mapping[str, object] | None = None,
    workspace_root: str = "/workspace",
    source_manifest_paths: Mapping[str, Path | str] | None = None,
    observed_tool_fingerprints: Sequence[Mapping[str, object]],
) -> Path:
    """Build a witness-verifiable transport bundle from slot execution results."""
    root = Path(bundle_root).resolve()
    if root.exists():
        if any(root.iterdir()):
            raise ExternalHostWitnessError("transport bundle root must be empty")
    else:
        root.mkdir(parents=True)

    if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA_V2:
        raise ExternalHostWitnessError("request must be a frozen V2 request")

    runtime = request.get("fair_runtime")
    if not isinstance(runtime, Mapping):
        raise ExternalHostWitnessError("request is missing fair_runtime")

    schedule_slots = runtime.get("slots")
    commands = runtime.get("commands")
    if not isinstance(schedule_slots, list) or len(schedule_slots) != 9:
        raise ExternalHostWitnessError("request schedule must have exactly 9 slots")
    if not isinstance(commands, list) or len(commands) != 9:
        raise ExternalHostWitnessError("request must have exactly 9 slot commands")

    if len(slot_results) != 9:
        raise ExternalHostWitnessError("must provide exactly 9 slot results")

    expected_tools = request.get("expected_tool_fingerprints")
    if not isinstance(expected_tools, list):
        raise ExternalHostWitnessError("V2 request must have expected_tool_fingerprints")

    observed_tools = list(observed_tool_fingerprints)
    if canonical_sha256(observed_tools) != canonical_sha256(expected_tools):
        raise ExternalHostWitnessError("external host tool fingerprints differ from the frozen request")

    identity = dict(host_identity) if host_identity is not None else capture_host_identity()
    identity_sha = canonical_sha256(identity)
    tool_fingerprints_sha = canonical_sha256(observed_tools)

    workspace_pure = PurePosixPath(workspace_root)
    if not workspace_pure.is_absolute():
        raise ExternalHostWitnessError("workspace_root must be absolute")

    request_bytes = json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    request_artifact_sha = hashlib.sha256(request_bytes).hexdigest()
    (root / "request.json").write_bytes(request_bytes)

    manifest_entries_for_request = [_artifact_ref(root, "request.json")]

    sources = request.get("sources")
    if not isinstance(sources, list):
        raise ExternalHostWitnessError("request sources are missing")
    source_by_comparator = {row["comparator"]: row for row in sources}
    source_manifest_entries: list[dict[str, object]] = []
    for source_row in sources:
        comparator = source_row["comparator"]
        expected_sha = source_row["source_manifest_sha256"]
        if source_manifest_paths is not None and comparator in source_manifest_paths:
            content = Path(source_manifest_paths[comparator]).read_bytes()
        else:
            content = json.dumps(
                source_row, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            raise ExternalHostWitnessError(
                f"source manifest sha256 mismatch for {comparator}"
            )
        relative = f"source-manifest-{comparator}.json"
        (root / relative).write_bytes(content)
        source_manifest_entries.append({
            "comparator": comparator,
            "artifact": _artifact_ref(root, relative),
        })
        manifest_entries_for_request.append(_artifact_ref(root, relative))

    manifest_entries: list[dict[str, object]] = list(manifest_entries_for_request)
    execution_slots: list[dict[str, object]] = []
    previous_finished: int | None = None

    slot_by_number = {row["slot"]: row for row in schedule_slots}
    command_by_slot = {row["slot"]: row["command"] for row in commands}

    for idx in range(9):
        slot_result = slot_results[idx]
        slot_number = slot_result["slot"]
        schedule_slot = slot_by_number[slot_number]
        command = command_by_slot[slot_number]
        comparator = schedule_slot["comparator"]
        started = slot_result["started_monotonic_ns"]
        finished = slot_result["finished_monotonic_ns"]
        wall = finished - started

        if previous_finished is not None and started < previous_finished:
            raise ExternalHostWitnessError("slot timing overlaps")
        previous_finished = finished

        env_roots = {
            "cache_root": f"{workspace_root}/cache/slot-{slot_number}",
            "checkout_root": f"{workspace_root}/checkout/slot-{slot_number}",
            "output_root": f"{workspace_root}/output/slot-{slot_number}",
            "runtime_root": f"{workspace_root}/runtime/slot-{slot_number}",
            "temporary_root": f"{workspace_root}/tmp/slot-{slot_number}",
            "working_directory": f"{workspace_root}/work/slot-{slot_number}",
        }

        raw_rel = f"slot-{slot_number}-stdout.bin"
        (root / raw_rel).write_bytes(slot_result.get("stdout", b""))
        raw_ref = _artifact_ref(root, raw_rel)
        manifest_entries.append(raw_ref)

        result_rel = f"slot-{slot_number}-result.json"
        (root / result_rel).write_bytes(
            Path(slot_result["result_path"]).read_bytes()
        )
        result_ref = _artifact_ref(root, result_rel)
        manifest_entries.append(result_ref)

        probe = {
            "schema": PROBE_SCHEMA_V2,
            "slot": slot_number,
            "host_identity_sha256": identity_sha,
            "source": source_by_comparator[comparator],
            "input_sha256": request["canonical_input"]["ffv1_sha256"],
            "conversion_receipt_sha256": request["canonical_input"]["conversion_receipt_sha256"],
            "protocol_sha256": runtime["protocol_sha256"],
            "schedule_sha256": runtime["schedule_sha256"],
            "command_sha256": canonical_sha256(command),
            "tool_fingerprints_sha256": tool_fingerprints_sha,
            "environment_roots": dict(env_roots),
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "wall_time_ns": wall,
            "tool_fingerprints": observed_tools,
        }
        probe_rel = f"slot-{slot_number}-probe.json"
        _write_json(root / probe_rel, probe)
        probe_ref = _artifact_ref(root, probe_rel)
        manifest_entries.append(probe_ref)

        execution_slots.append({
            "slot": slot_number,
            "round": schedule_slot["round"],
            "position": schedule_slot["position"],
            "comparator": comparator,
            "repeat_ordinal": schedule_slot["repeat_ordinal"],
            "command": list(command),
            "timeout_seconds": runtime["timeout_seconds"],
            "attempt_ordinal": 1,
            "retry_count": 0,
            "retry_policy": "NO_RETRY",
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "wall_time_ns": wall,
            "timed_out": False,
            "exit_code": slot_result["exit_code"],
            "environment_roots": dict(env_roots),
            "raw_outputs": [raw_ref],
            "probes": [probe_ref],
            "result": result_ref,
        })

    execution = {
        "request_sha256": request_artifact_sha,
        "protocol_sha256": runtime["protocol_sha256"],
        "schedule_sha256": runtime["schedule_sha256"],
        "input_sha256": request["canonical_input"]["ffv1_sha256"],
        "conversion_receipt_sha256": request["canonical_input"]["conversion_receipt_sha256"],
        "source_population_sha256": request["source_population_sha256"],
        "source_manifests": source_manifest_entries,
        "slots": execution_slots,
        "slots_sha256": canonical_sha256(execution_slots),
        "tool_fingerprints_sha256": tool_fingerprints_sha,
    }

    manifest_path = root / "manifest.json"
    _write_json(manifest_path, {
        "schema": MANIFEST_SCHEMA,
        "entries": sorted(manifest_entries, key=lambda e: e["path"]),
    })

    receipt = {
        "schema": RECEIPT_SCHEMA_V2,
        "request": _artifact_ref(root, "request.json"),
        "transport_manifest": {"path": "manifest.json", "sha256": _sha256_file(root / "manifest.json")},
        "host": {
            "identity": identity,
            "identity_sha256": identity_sha,
            "workspace_root": workspace_root,
            "tool_fingerprints": observed_tools,
        },
        "execution": execution,
        "status": "WITNESSED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
    }
    receipt_path = root / "receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path


def run_slot_command(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one slot command and capture full execution evidence."""
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
        )
        finished = time.monotonic_ns()
        return {
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        finished = time.monotonic_ns()
        return {
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "exit_code": 124,
            "stdout": exc.stdout or b"",
            "stderr": (exc.stderr or b"") + b"\nflashpatch: slot timeout",
            "timed_out": True,
        }
