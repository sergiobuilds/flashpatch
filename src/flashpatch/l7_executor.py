from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .competition import _load_json, ContractError
from .competition import NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA
from .competition import verify_native_main_candidate_start_gate
from .external_league import DIRECT_DETECTOR_POPULATION
from .l7_external_host import (
    REQUEST_SCHEMA_V2,
    RECEIPT_SCHEMA_V2,
    PROBE_SCHEMA_V2,
    ExternalHostWitnessError,
    canonical_sha256,
    capture_host_identity,
    _file_sha256,
)


@dataclass(frozen=True)
class CandidateStartGateInputs:
    intake_receipt: Path
    return_root: Path
    packet_manifest: Path
    start_receipt: Path

def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalHostWitnessError(f"invalid JSON: {path}") from exc

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def build_execution_probe(
    command_sequence: Sequence[str],
    stdout_sha256: str,
    stderr_sha256: str,
    exit_code: int,
    started: int,
    finished: int,
) -> dict[str, Any]:
    return {
        "command": list(command_sequence),
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "exit_code": exit_code,
        "started_monotonic_ns": started,
        "finished_monotonic_ns": finished,
    }

def run_external_host_9_slot_executor(
    request_path: Path,
    output_dir: Path,
    *,
    candidate_start_gate: CandidateStartGateInputs,
) -> Path:
    if not request_path.exists():
        raise ExternalHostWitnessError(f"request path does not exist: {request_path}")
    _verify_candidate_start_gate(candidate_start_gate)
    
    try:
        request = _load_json(request_path)
    except ContractError as exc:
        raise ExternalHostWitnessError(f"invalid request JSON: {exc}") from exc

    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA_V2:
        raise ExternalHostWitnessError("must be a frozen v2 request")

    fair_runtime = request.get("fair_runtime", {})
    if not isinstance(fair_runtime, dict):
        raise ExternalHostWitnessError("request is missing fair_runtime object")
        
    slots = fair_runtime.get("slots", [])
    if not isinstance(slots, list) or len(slots) != 9:
        raise ExternalHostWitnessError("must have exactly 9 slots")
    slot_ids: set[int] = set()
    comparator_counts = {name: 0 for name in DIRECT_DETECTOR_POPULATION}
    for row in slots:
        if not isinstance(row, Mapping):
            raise ExternalHostWitnessError("slot row is invalid")
        slot_id = row.get("slot")
        comparator = row.get("comparator")
        if (
            isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id < 1
            or slot_id > 9
            or slot_id in slot_ids
        ):
            raise ExternalHostWitnessError("slot identity is invalid")
        if comparator not in comparator_counts:
            raise ExternalHostWitnessError("slot comparator is outside the direct detector population")
        slot_ids.add(slot_id)
        comparator_counts[str(comparator)] += 1
    if slot_ids != set(range(1, 10)):
        raise ExternalHostWitnessError("slot identity is invalid")
    if any(count != 3 for count in comparator_counts.values()):
        raise ExternalHostWitnessError("must schedule exactly three repeats for every direct detector")

    sources = request.get("sources", [])
    if not isinstance(sources, list):
        raise ExternalHostWitnessError("request is missing sources list")
        
    comparators = {s.get("comparator") for s in sources if isinstance(s, dict)}
    if not set(DIRECT_DETECTOR_POPULATION).issubset(comparators):
        raise ExternalHostWitnessError("must contain FlashPatch, TooFlashy, and Kaya participants")
    commands = fair_runtime.get("commands", [])
    if not isinstance(commands, list) or len(commands) != 9:
        raise ExternalHostWitnessError("must provide exactly 9 slot commands")
    command_slots = {row.get("slot") for row in commands if isinstance(row, Mapping)}
    if command_slots != slot_ids:
        raise ExternalHostWitnessError("slot commands do not cover every scheduled slot")

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ExternalHostWitnessError("output directory must be empty before execution")
    execution_probes: list[dict[str, Any]] = []

    # Source mapping
    source_by_comparator = {row.get("comparator"): row for row in sources if isinstance(row, dict)}

    for slot in commands:
        if not isinstance(slot, Mapping):
            raise ExternalHostWitnessError("slot command row is invalid")
        slot_id = slot["slot"]
        command = slot["command"]
        if not isinstance(command, list) or not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ExternalHostWitnessError("slot command is invalid")
        
        # We need to find the comparator for this slot from the schedule
        slot_def = next((s for s in slots if s.get("slot") == slot_id), {})
        comparator = slot_def.get("comparator")

        started = time.monotonic_ns()
        proc = subprocess.run(command, capture_output=True)
        finished = time.monotonic_ns()

        out_bytes = proc.stdout if hasattr(proc, 'stdout') and isinstance(proc.stdout, bytes) else b""
        err_bytes = proc.stderr if hasattr(proc, 'stderr') and isinstance(proc.stderr, bytes) else b""

        out_sha = canonical_sha256(out_bytes.decode("utf-8", "replace"))
        err_sha = canonical_sha256(err_bytes.decode("utf-8", "replace"))

        probe = build_execution_probe(
            command_sequence=command,
            stdout_sha256=out_sha,
            stderr_sha256=err_sha,
            exit_code=proc.returncode,
            started=started,
            finished=finished,
        )
        
        result_path = output_dir / f"slot-{slot_id}.json"
        if not result_path.exists():
            raise ExternalHostWitnessError(f"slot {slot_id} did not produce result artifact")

        # Artifact Ref
        probe["result"] = {"path": result_path.name, "sha256": _file_sha256(result_path), "size": result_path.stat().st_size}
        
        probe["schema"] = PROBE_SCHEMA_V2
        probe["slot"] = slot_id
        probe["host_identity_sha256"] = request.get("origin_host_identity_sha256")
        probe["source"] = source_by_comparator.get(comparator, {})
        probe["input_sha256"] = request.get("canonical_input", {}).get("ffv1_sha256")
        probe["conversion_receipt_sha256"] = request.get("canonical_input", {}).get("conversion_receipt_sha256")
        probe["protocol_sha256"] = fair_runtime.get("protocol_sha256")
        probe["schedule_sha256"] = fair_runtime.get("schedule_sha256")
        probe["command_sha256"] = canonical_sha256(command)
        probe["tool_fingerprints_sha256"] = canonical_sha256(request.get("expected_tool_fingerprints", []))
        probe["tool_fingerprints"] = request.get("expected_tool_fingerprints", [])
        probe["environment_roots"] = {}

        execution_probes.append(probe)

    receipt = {
        "schema": RECEIPT_SCHEMA_V2,
        "request_sha256": canonical_sha256(request),
        "protocol_sha256": fair_runtime.get("protocol_sha256"),
        "schedule_sha256": fair_runtime.get("schedule_sha256"),
        "input_sha256": request.get("canonical_input", {}).get("ffv1_sha256"),
        "conversion_receipt_sha256": request.get("canonical_input", {}).get("conversion_receipt_sha256"),
        "source_population_sha256": request.get("source_population_sha256"),
        "tool_fingerprints_sha256": canonical_sha256(request.get("expected_tool_fingerprints", [])),
        "host_identity": capture_host_identity(),
        "source_manifests": request.get("sources", []),
        "slots": execution_probes,
        "slots_sha256": canonical_sha256(execution_probes),
    }

    receipt_path = output_dir / "external-receipt.json"
    _write_json(receipt_path, receipt)
    
    return receipt_path


def _verify_candidate_start_gate(inputs: CandidateStartGateInputs) -> dict[str, object]:
    try:
        assessment = verify_native_main_candidate_start_gate(
            intake_receipt=inputs.intake_receipt,
            return_root=inputs.return_root,
            packet_manifest=inputs.packet_manifest,
            start_receipt=inputs.start_receipt,
        )
    except (ContractError, OSError) as exc:
        raise ExternalHostWitnessError(f"candidate-start gate verification failed: {exc}") from exc
    if (
        assessment.get("schema") != NATIVE_MAIN_CANDIDATE_START_GATE_ASSESSMENT_SCHEMA
        or assessment.get("status") != "CANDIDATE_START_WITNESS_VERIFIED"
        or assessment.get("scoreable") is not False
        or assessment.get("external_claim_authorized") is not False
        or assessment.get("detector_population") != list(DIRECT_DETECTOR_POPULATION)
        or assessment.get("slot_count") != 9
        or assessment.get("repeat_count_per_detector") != 3
        or assessment.get("request_schema") != REQUEST_SCHEMA_V2
    ):
        raise ExternalHostWitnessError(
            "candidate-start gate must verify independent gold before 9-slot execution"
        )
    blockers = assessment.get("score_blockers")
    if not isinstance(blockers, list) or "same_condition_9_slot_execution_missing" not in blockers:
        raise ExternalHostWitnessError("candidate-start gate assessment blockers are invalid")
    return assessment
