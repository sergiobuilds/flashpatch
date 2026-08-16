"""Durable, contract-bound state for one L7 execution attempt.

The score verifier owns the evidence semantics.  This module owns only
orchestration safety: a run has one immutable execution contract, one output
directory, and receipt-bound completed-case checkpoints.  Narrative documents
are deliberately not part of the contract digest.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


EXECUTION_CONTRACT_SCHEMA = "flashpatch-l7-execution-contract-v1"
RUN_STATE_SCHEMA = "flashpatch-l7-durable-run-state-v1"
RUN_EVENT_SCHEMA = "flashpatch-l7-durable-run-event-v1"
_PHASES = frozenset({"prepare", "parity", "freeze_request", "execute", "parity_verify", "complete_case", "assemble", "discard_incomplete_case"})
_PHASE_STATES = frozenset({"STARTED", "COMPLETE", "FAILED"})


class L7RunStateError(ValueError):
    """Raised when a run cannot safely be initialized or resumed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = _canonical_json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise L7RunStateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise L7RunStateError(f"JSON object required: {path}")
    return value


def _owned_file(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise L7RunStateError(f"regular file required: {path}")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise L7RunStateError(f"path escapes run root: {path}") from exc
    return resolved_path


def execution_contract(
    project_root: Path | str,
    *,
    code_paths: Sequence[Path | str],
    comparator_census_sha256: str,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Freeze only execution-relevant inputs, never prose documentation.

    ``code_paths`` must be regular files below ``project_root``.  Callers pass
    the runner, verifier, comparator implementation, and container recipe as
    appropriate for the run.  A documentation-only edit therefore does not
    change this contract, while a runner or comparator edit does.
    """
    root = Path(project_root).resolve(strict=True)
    files: list[dict[str, str]] = []
    for raw_path in code_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = _owned_file(root, path)
        files.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)})
    if not files or len({row["path"] for row in files}) != len(files):
        raise L7RunStateError("non-empty unique code paths required")
    if not isinstance(comparator_census_sha256, str) or len(comparator_census_sha256) != 64:
        raise L7RunStateError("comparator census SHA-256 is required")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()):
        raise L7RunStateError("environment must contain string keys and values")
    payload: dict[str, object] = {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "code": sorted(files, key=lambda row: row["path"]),
        "comparator_census_sha256": comparator_census_sha256,
        "environment": dict(sorted(environment.items())),
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _validate_contract(contract: Mapping[str, object]) -> str:
    candidate = dict(contract)
    observed = candidate.pop("contract_sha256", None)
    if candidate.get("schema") != EXECUTION_CONTRACT_SCHEMA or not isinstance(observed, str):
        raise L7RunStateError("invalid execution contract")
    expected = hashlib.sha256(_canonical_json(candidate)).hexdigest()
    if observed != expected:
        raise L7RunStateError("execution contract digest mismatch")
    return observed


@contextmanager
def locked_run(run_root: Path | str) -> Iterator[Path]:
    """Acquire an exclusive lock for a single run directory."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise L7RunStateError("run root cannot be a symlink")
    lock_path = root / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise L7RunStateError("run is already active") from exc
        try:
            yield root.resolve(strict=True)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def execution_guard(run_root: Path | str) -> Iterator[Path]:
    """Reserve a run root for exactly one long-running orchestrator."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise L7RunStateError("run root cannot be a symlink")
    with (root / ".executor.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise L7RunStateError("run executor is already active") from exc
        try:
            yield root.resolve(strict=True)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_run(run_root: Path | str, contract: Mapping[str, object]) -> dict[str, object]:
    """Create a new empty run, refusing stale output reuse."""
    contract_sha256 = _validate_contract(contract)
    with locked_run(run_root) as root:
        state_path = root / "run-state.json"
        contract_path = root / "execution-contract.json"
        # The long-lived executor guard exists before initialization so that
        # two orchestrators cannot race to create the state.  It contains no
        # evidence and is the sole permitted pre-initialization entry.
        occupied = [
            path for path in root.iterdir()
            if path.name not in {".run.lock", ".executor.lock"}
        ]
        if occupied:
            raise L7RunStateError("run root must be empty before initialization")
        _atomic_json(contract_path, dict(contract))
        state: dict[str, object] = {
            "schema": RUN_STATE_SCHEMA,
            "status": "RUNNING",
            "contract_sha256": contract_sha256,
            "completed_cases": {},
            "phases": {},
        }
        _atomic_json(state_path, state)
        return state


def record_phase(
    run_root: Path | str,
    contract: Mapping[str, object],
    *,
    phase: str,
    status: str,
    case_id: str | None = None,
    detail: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Durably record a phase transition under the same contract and lock."""
    if phase not in _PHASES or status not in _PHASE_STATES:
        raise L7RunStateError("invalid phase transition")
    if case_id is not None and (not isinstance(case_id, str) or not case_id):
        raise L7RunStateError("invalid phase case id")
    contract_sha256 = _validate_contract(contract)
    with locked_run(run_root) as root:
        state_path = root / "run-state.json"
        state = _read_json(state_path)
        if state.get("status") != "RUNNING" or state.get("contract_sha256") != contract_sha256:
            raise L7RunStateError("run is not active under this execution contract")
        phases = state.get("phases")
        if not isinstance(phases, dict):
            raise L7RunStateError("invalid phase state")
        key = f"{case_id or 'run'}:{phase}"
        previous = phases.get(key)
        if isinstance(previous, str) and previous == "COMPLETE" and status != "COMPLETE":
            raise L7RunStateError("completed phase cannot regress")
        phases[key] = status
        state["phases"] = dict(sorted(phases.items()))
        _append_event(
            root / "events.jsonl",
            {
                "schema": RUN_EVENT_SCHEMA,
                "contract_sha256": contract_sha256,
                "phase": phase,
                "status": status,
                "case_id": case_id,
                "detail": dict(detail or {}),
            },
        )
        _atomic_json(state_path, state)
        return state


def record_completed_case(
    run_root: Path | str,
    contract: Mapping[str, object],
    *,
    case_id: str,
    receipt_path: Path | str,
) -> dict[str, object]:
    """Atomically record one complete case receipt exactly once."""
    if not isinstance(case_id, str) or not case_id:
        raise L7RunStateError("non-empty case id required")
    contract_sha256 = _validate_contract(contract)
    with locked_run(run_root) as root:
        state_path = root / "run-state.json"
        state = _read_json(state_path)
        if state.get("schema") != RUN_STATE_SCHEMA or state.get("status") != "RUNNING":
            raise L7RunStateError("run is not active")
        if state.get("contract_sha256") != contract_sha256:
            raise L7RunStateError("cannot resume with a different execution contract")
        receipt = _owned_file(root, Path(receipt_path))
        completed = state.get("completed_cases")
        if not isinstance(completed, dict):
            raise L7RunStateError("invalid completed case state")
        if case_id in completed:
            raise L7RunStateError("case is already checkpointed")
        completed[case_id] = {
            "path": receipt.relative_to(root).as_posix(),
            "sha256": _sha256_file(receipt),
        }
        state["completed_cases"] = dict(sorted(completed.items()))
        _atomic_json(state_path, state)
        return state


def discard_incomplete_case(
    run_root: Path | str,
    contract: Mapping[str, object],
    *,
    case_id: str,
) -> dict[str, object]:
    """Forget phase state for an uncheckpointed case before a clean rerun.

    The caller must remove that case's output directory independently.  This
    function never permits a checkpointed receipt to be overwritten; it only
    makes a previously interrupted, untrusted case eligible for a new attempt
    under the identical contract.
    """
    if not isinstance(case_id, str) or not case_id:
        raise L7RunStateError("non-empty case id required")
    contract_sha256 = _validate_contract(contract)
    with locked_run(run_root) as root:
        state_path = root / "run-state.json"
        state = _read_json(state_path)
        if state.get("schema") != RUN_STATE_SCHEMA or state.get("status") != "RUNNING":
            raise L7RunStateError("run is not active")
        if state.get("contract_sha256") != contract_sha256:
            raise L7RunStateError("cannot resume with a different execution contract")
        completed = state.get("completed_cases")
        phases = state.get("phases")
        if not isinstance(completed, dict) or not isinstance(phases, dict):
            raise L7RunStateError("invalid run state")
        if case_id in completed:
            raise L7RunStateError("checkpointed case cannot be discarded")
        prefix = f"{case_id}:"
        state["phases"] = {key: value for key, value in phases.items() if not key.startswith(prefix)}
        _append_event(
            root / "events.jsonl",
            {
                "schema": RUN_EVENT_SCHEMA,
                "contract_sha256": contract_sha256,
                "phase": "discard_incomplete_case",
                "status": "COMPLETE",
                "case_id": case_id,
                "detail": {"reason": "partial_case_output_not_reused"},
            },
        )
        _atomic_json(state_path, state)
        return state


def finalize_run(
    run_root: Path | str,
    contract: Mapping[str, object],
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, object]:
    """Close only a complete run with receipt files that still match their hashes."""
    contract_sha256 = _validate_contract(contract)
    expected = set(expected_case_ids)
    if not expected or len(expected) != len(expected_case_ids):
        raise L7RunStateError("unique expected case ids required")
    with locked_run(run_root) as root:
        state_path = root / "run-state.json"
        state = _read_json(state_path)
        if state.get("status") != "RUNNING" or state.get("contract_sha256") != contract_sha256:
            raise L7RunStateError("run cannot be finalized")
        completed = state.get("completed_cases")
        if not isinstance(completed, dict) or set(completed) != expected:
            raise L7RunStateError("all expected cases must be checkpointed")
        for row in completed.values():
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise L7RunStateError("invalid completed case receipt")
            receipt = _owned_file(root, root / row["path"])
            if row.get("sha256") != _sha256_file(receipt):
                raise L7RunStateError("completed case receipt changed after checkpoint")
        state["status"] = "COMPLETE"
        _atomic_json(state_path, state)
        return state
