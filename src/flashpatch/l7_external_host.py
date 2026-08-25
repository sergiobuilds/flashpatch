"""Portable, fail-closed evidence contract for an independent L7 host run.

The coordinator freezes a request before transport.  A remote runner may later
return a receipt bundle, but the consumer trusts only the original request and
content hashes.  Reported remote paths are evidence strings, never paths to be
opened on the verifier host.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


REQUEST_SCHEMA_V1 = "flashpatch-l7-external-host-witness-request-v1"
REQUEST_SCHEMA_V2 = "flashpatch-l7-external-host-witness-request-v2"
REQUEST_SCHEMA = REQUEST_SCHEMA_V2
RECEIPT_SCHEMA_V1 = "flashpatch-l7-external-host-witness-receipt-v1"
RECEIPT_SCHEMA_V2 = "flashpatch-l7-external-host-witness-receipt-v2"
RECEIPT_SCHEMA = RECEIPT_SCHEMA_V2
MANIFEST_SCHEMA = "flashpatch-l7-external-host-transport-manifest-v1"
PROBE_SCHEMA_V1 = "flashpatch-l7-external-host-execution-probe-v1"
PROBE_SCHEMA_V2 = "flashpatch-l7-external-host-execution-probe-v2"
PROBE_SCHEMA = PROBE_SCHEMA_V2
EXECUTION_BOUNDARY_SCHEMA = "flashpatch-l7-external-host-execution-boundary-v1"
PREPARATION_PROBE_SCHEMA = "flashpatch-l7-external-host-preparation-probe-v1"
PRIVILEGE_PROBE_SCHEMA = "flashpatch-l7-external-host-privilege-probe-v1"
VERIFICATION_SCHEMA_V1 = "flashpatch-l7-external-host-witness-verification-v1"
VERIFICATION_SCHEMA_V2 = "flashpatch-l7-external-host-witness-verification-v2"
VERIFICATION_SCHEMA = VERIFICATION_SCHEMA_V2
RUNTIME_PROTOCOL_SCHEMA = "flashpatch-l7-fair-runtime-protocol-v1"
RUNTIME_SCHEDULE_SCHEMA = "flashpatch-l7-fair-runtime-schedule-v1"
REQUIRED_TOOLS = ("ffmpeg", "git", "godot", "python", "uv")
ROOT_FIELDS = (
    "cache_root",
    "checkout_root",
    "output_root",
    "runtime_root",
    "temporary_root",
    "working_directory",
)
_HASH = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class ExternalHostWitnessError(ValueError):
    """The external-host evidence cannot be trusted."""


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ExternalHostWitnessError(f"{field} must be a lowercase SHA-256")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ExternalHostWitnessError(f"{field} must be non-empty text")
    return value


def capture_host_identity() -> dict[str, object]:
    """Capture the fields used to prove that origin and runner are distinct."""
    cpu_model = platform.processor().strip()
    if not cpu_model:
        try:
            cpu_model = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
                if line.startswith("model name") and ":" in line
            )
        except (OSError, StopIteration):
            cpu_model = "UNKNOWN"
    machine_id = ""
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            machine_id = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if machine_id:
            break
    if not machine_id:
        machine_id = socket.gethostname()
    return {
        "machine_id": machine_id,
        "hostname": socket.gethostname(),
        "operating_system": platform.system(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count() or 1,
    }


def _validate_host_identity(payload: object, field: str) -> dict[str, object]:
    expected = {
        "machine_id",
        "hostname",
        "operating_system",
        "kernel",
        "architecture",
        "cpu_model",
        "logical_cpu_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ExternalHostWitnessError(f"{field} fields are invalid")
    result = dict(payload)
    for name in expected - {"logical_cpu_count"}:
        _require_text(result[name], f"{field}.{name}")
    logical_count = result["logical_cpu_count"]
    if isinstance(logical_count, bool) or not isinstance(logical_count, int) or logical_count <= 0:
        raise ExternalHostWitnessError(f"{field}.logical_cpu_count is invalid")
    return result


def _validate_source_rows(payload: object, participants: Sequence[str]) -> list[dict[str, str]]:
    if not isinstance(payload, list) or len(payload) != len(participants):
        raise ExternalHostWitnessError("source population is incomplete")
    rows: list[dict[str, str]] = []
    expected_fields = {
        "comparator",
        "repository_url",
        "revision",
        "tree",
        "source_manifest_sha256",
    }
    for row in payload:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ExternalHostWitnessError("source identity fields are invalid")
        frozen = {key: _require_text(row[key], f"source.{key}") for key in expected_fields}
        if _GIT_OBJECT.fullmatch(frozen["revision"]) is None or _GIT_OBJECT.fullmatch(frozen["tree"]) is None:
            raise ExternalHostWitnessError("source revision or tree is not a Git object identity")
        _require_hash(frozen["source_manifest_sha256"], "source.source_manifest_sha256")
        rows.append(frozen)
    rows.sort(key=lambda row: row["comparator"])
    if [row["comparator"] for row in rows] != sorted(participants):
        raise ExternalHostWitnessError("source population differs from the frozen schedule")
    return rows


def _validate_commands(payload: object, slots: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not isinstance(payload, list) or len(payload) != len(slots):
        raise ExternalHostWitnessError("exact command population is incomplete")
    by_slot: dict[int, Mapping[str, object]] = {}
    for row in payload:
        if not isinstance(row, Mapping) or set(row) != {"slot", "command"}:
            raise ExternalHostWitnessError("exact command row fields are invalid")
        slot = row.get("slot")
        command = row.get("command")
        if (
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot in by_slot
            or not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part or "\x00" in part for part in command)
        ):
            raise ExternalHostWitnessError("exact command row is invalid")
        by_slot[slot] = row
    expected_slots = [int(row["slot"]) for row in slots]
    if sorted(by_slot) != sorted(expected_slots):
        raise ExternalHostWitnessError("exact commands do not cover every scheduled slot")
    return [{"slot": slot, "command": list(by_slot[slot]["command"])} for slot in sorted(by_slot)]


def _validate_expected_tool_fingerprints(
    payload: object,
    required_tools: Sequence[str],
    field: str = "expected_tool_fingerprints",
) -> list[dict[str, str]]:
    """Validate the ordered coordinator-owned executable identity contract."""
    if not isinstance(payload, list) or len(payload) != len(required_tools):
        raise ExternalHostWitnessError(f"{field} population is incomplete")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "path",
            "sha256",
            "version_output",
        }:
            raise ExternalHostWitnessError(f"{field} fields are invalid")
        name = _require_text(row.get("name"), f"{field}[{index}].name")
        binary_path = PurePosixPath(
            _require_text(row.get("path"), f"{field}[{index}].path")
        )
        if not binary_path.is_absolute() or any(
            part in {"", ".", ".."} for part in binary_path.parts
        ):
            raise ExternalHostWitnessError(f"{field} path is invalid")
        rows.append(
            {
                "name": name,
                "path": str(binary_path),
                "sha256": _require_hash(
                    row.get("sha256"), f"{field}[{index}].sha256"
                ),
                "version_output": _require_text(
                    row.get("version_output"),
                    f"{field}[{index}].version_output",
                ),
            }
        )
    expected_names = list(required_tools)
    observed_names = [row["name"] for row in rows]
    if observed_names != expected_names or len(observed_names) != len(set(observed_names)):
        raise ExternalHostWitnessError(
            f"{field} names or order differ from required_tools"
        )
    return rows


def _validate_execution_boundary(
    payload: object,
    *,
    commands: Sequence[Mapping[str, object]],
    required_tools: Sequence[str],
    tool_fingerprints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Freeze an explicit unprivileged-setup / privileged-timed boundary."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "preparation",
        "timed_execution",
    } or payload.get("schema") != EXECUTION_BOUNDARY_SCHEMA:
        raise ExternalHostWitnessError("external host execution boundary fields are invalid")
    preparation = payload.get("preparation")
    timed = payload.get("timed_execution")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "mode",
        "uid",
        "gid",
        "username",
        "source_setup_timing",
        "identity_probe_contracts",
    }:
        raise ExternalHostWitnessError("external host preparation boundary is invalid")
    if (
        preparation.get("mode") != "UNPRIVILEGED_USER"
        or preparation.get("source_setup_timing") != "OUTSIDE_MEASUREMENT_BOUNDARY"
        or isinstance(preparation.get("uid"), bool)
        or not isinstance(preparation.get("uid"), int)
        or int(preparation["uid"]) <= 0
        or isinstance(preparation.get("gid"), bool)
        or not isinstance(preparation.get("gid"), int)
        or int(preparation["gid"]) <= 0
    ):
        raise ExternalHostWitnessError("external host preparation identity is invalid")
    _require_text(preparation.get("username"), "execution_boundary.preparation.username")
    expected_identity_probe_contracts = [
        {
            "name": "uid",
            "command": ["/usr/bin/id", "-u"],
            "expected_stdout": f"{preparation['uid']}\n",
        },
        {
            "name": "gid",
            "command": ["/usr/bin/id", "-g"],
            "expected_stdout": f"{preparation['gid']}\n",
        },
        {
            "name": "username",
            "command": ["/usr/bin/id", "-un"],
            "expected_stdout": f"{preparation['username']}\n",
        },
    ]
    if preparation.get("identity_probe_contracts") != expected_identity_probe_contracts:
        raise ExternalHostWitnessError(
            "external host preparation identity probe contract is invalid"
        )
    if not isinstance(timed, Mapping) or set(timed) != {
        "mode",
        "launcher_prefix",
        "non_interactive",
        "preserve_environment",
        "required_effective_uid",
        "policy_scope",
        "probe_command",
        "probe_expected_stdout",
    }:
        raise ExternalHostWitnessError("external host timed execution boundary is invalid")
    prefix = timed.get("launcher_prefix")
    probe_command = timed.get("probe_command")
    if (
        timed.get("mode") != "SUDO_BWRAP_EXACT_COMMAND"
        or timed.get("non_interactive") is not True
        or timed.get("preserve_environment") is not False
        or timed.get("required_effective_uid") != 0
        or timed.get("policy_scope") != "NON_INTERACTIVE_EXECUTION_AVAILABLE_ONLY"
        or not isinstance(prefix, list)
        or len(prefix) != 3
        or prefix[1:] != ["-n", "--"]
        or not isinstance(prefix[0], str)
        or not PurePosixPath(prefix[0]).is_absolute()
        or not isinstance(probe_command, list)
        or probe_command != [*prefix, "/usr/bin/id", "-u"]
        or timed.get("probe_expected_stdout") != "0\n"
    ):
        raise ExternalHostWitnessError("external host sudo execution policy is invalid")
    if "sudo" not in required_tools:
        raise ExternalHostWitnessError("external host execution boundary requires sudo")
    sudo_rows = [row for row in tool_fingerprints if row.get("name") == "sudo"]
    if len(sudo_rows) != 1 or sudo_rows[0].get("path") != prefix[0]:
        raise ExternalHostWitnessError("external host sudo fingerprint differs from launcher")
    for row in commands:
        command = row.get("command")
        if not isinstance(command, list) or command[: len(prefix)] != prefix:
            raise ExternalHostWitnessError(
                "external host timed command omits the frozen privilege launcher"
            )
    return {
        "schema": EXECUTION_BOUNDARY_SCHEMA,
        "preparation": {
            **dict(preparation),
            "identity_probe_contracts": expected_identity_probe_contracts,
        },
        "timed_execution": {
            **dict(timed),
            "launcher_prefix": list(prefix),
            "probe_command": list(probe_command),
        },
    }


def freeze_external_host_witness_request(
    *,
    origin_host: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    canonical_ffv1_sha256: str,
    conversion_receipt_sha256: str,
    fair_runtime_protocol: Mapping[str, object],
    fair_runtime_schedule: Mapping[str, object],
    slot_commands: Mapping[int, Sequence[str]],
    required_tools: Sequence[str] = REQUIRED_TOOLS,
    expected_tool_fingerprints: Sequence[Mapping[str, object]] | None = None,
    execution_boundary: Mapping[str, object] | None = None,
    request_schema: str = REQUEST_SCHEMA,
) -> dict[str, object]:
    """Freeze the coordinator-owned request before it leaves the local host."""
    origin = _validate_host_identity(origin_host, "origin_host")
    if not isinstance(fair_runtime_protocol, Mapping) or fair_runtime_protocol.get("schema") != RUNTIME_PROTOCOL_SCHEMA:
        raise ExternalHostWitnessError("fair runtime protocol is not frozen")
    if not isinstance(fair_runtime_schedule, Mapping) or fair_runtime_schedule.get("schema") != RUNTIME_SCHEDULE_SCHEMA:
        raise ExternalHostWitnessError("fair runtime schedule is not frozen")
    protocol = dict(fair_runtime_protocol)
    schedule = dict(fair_runtime_schedule)
    protocol_sha = canonical_sha256(protocol)
    if schedule.get("protocol_sha256") != protocol_sha:
        raise ExternalHostWitnessError("schedule does not bind the supplied protocol")
    input_sha = _require_hash(canonical_ffv1_sha256, "canonical_ffv1_sha256")
    if schedule.get("input_sha256") != input_sha:
        raise ExternalHostWitnessError("schedule does not bind the canonical FFV1")
    conversion_sha = _require_hash(conversion_receipt_sha256, "conversion_receipt_sha256")
    participants = schedule.get("participants")
    slots = schedule.get("slots")
    if (
        not isinstance(participants, list)
        or len(participants) < 2
        or participants != sorted(participants)
        or len(set(participants)) != len(participants)
        or not isinstance(slots, list)
        or not slots
        or any(not isinstance(row, Mapping) for row in slots)
    ):
        raise ExternalHostWitnessError("fair runtime schedule population is invalid")
    source_rows = _validate_source_rows(list(sources), participants)
    command_rows = _validate_commands(
        [{"slot": slot, "command": list(command)} for slot, command in slot_commands.items()],
        slots,
    )
    budget = protocol.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("scheduled_repeats") != 3
        or budget.get("retry_policy") != "NO_RETRY"
        or isinstance(budget.get("timeout_seconds"), bool)
        or not isinstance(budget.get("timeout_seconds"), int)
        or int(budget["timeout_seconds"]) <= 0
    ):
        raise ExternalHostWitnessError("external host request requires three runs and retry zero")
    declared_tools = [
        _require_text(tool, "required_tools") for tool in required_tools
    ]
    if (
        len(declared_tools) != len(set(declared_tools))
        or not set(REQUIRED_TOOLS).issubset(declared_tools)
    ):
        raise ExternalHostWitnessError("external host request omits a required tool binary")
    if request_schema == REQUEST_SCHEMA_V1:
        if expected_tool_fingerprints is not None or execution_boundary is not None:
            raise ExternalHostWitnessError(
                "historical v1 request cannot carry v2 execution evidence"
            )
        tools = sorted(declared_tools)
        frozen_tool_fingerprints = None
        frozen_execution_boundary = None
    elif request_schema == REQUEST_SCHEMA_V2:
        tools = declared_tools
        if expected_tool_fingerprints is None:
            raise ExternalHostWitnessError(
                "v2 external host request requires explicit expected tool fingerprints"
            )
        frozen_tool_fingerprints = _validate_expected_tool_fingerprints(
            list(expected_tool_fingerprints), tools
        )
        frozen_execution_boundary = (
            _validate_execution_boundary(
                execution_boundary,
                commands=command_rows,
                required_tools=tools,
                tool_fingerprints=frozen_tool_fingerprints,
            )
            if execution_boundary is not None
            else None
        )
    else:
        raise ExternalHostWitnessError("external host request schema is unsupported")
    request = {
        "schema": request_schema,
        "freeze_state": "PRE_FROZEN",
        "origin_host": origin,
        "origin_host_identity_sha256": canonical_sha256(origin),
        "sources": source_rows,
        "source_population_sha256": canonical_sha256(source_rows),
        "canonical_input": {
            "format": "FFV1_RGB24_CFR",
            "ffv1_sha256": input_sha,
            "conversion_receipt_sha256": conversion_sha,
        },
        "fair_runtime": {
            "protocol_sha256": protocol_sha,
            "schedule_sha256": canonical_sha256(schedule),
            "timeout_seconds": int(budget["timeout_seconds"]),
            "scheduled_repeats": 3,
            "attempts_per_slot": 1,
            "retry_count": 0,
            "retry_policy": "NO_RETRY",
            "slots": [dict(row) for row in slots],
            "commands": command_rows,
        },
        "required_tools": tools,
        "transport_policy": {
            "relative_posix_artifact_paths_only": True,
            "reject_symlinks": True,
            "reject_unmanifested_files": True,
            "distinct_host_identity_required": True,
        },
    }
    if frozen_tool_fingerprints is not None:
        request["expected_tool_fingerprints"] = frozen_tool_fingerprints
    if frozen_execution_boundary is not None:
        request["execution_boundary"] = frozen_execution_boundary
    return _validate_request(request)


def _validate_request(payload: object) -> dict[str, object]:
    required_v1 = {
        "schema",
        "freeze_state",
        "origin_host",
        "origin_host_identity_sha256",
        "sources",
        "source_population_sha256",
        "canonical_input",
        "fair_runtime",
        "required_tools",
        "transport_policy",
    }
    required_v2 = required_v1 | {"expected_tool_fingerprints"}
    if not isinstance(payload, Mapping):
        raise ExternalHostWitnessError("external host request fields are invalid")
    schema = payload.get("schema")
    allowed = (
        {frozenset(required_v2), frozenset(required_v2 | {"execution_boundary"})}
        if schema == REQUEST_SCHEMA_V2
        else {frozenset(required_v1)}
    )
    if frozenset(payload) not in allowed:
        raise ExternalHostWitnessError("external host request fields are invalid")
    if schema not in {REQUEST_SCHEMA_V1, REQUEST_SCHEMA_V2} or payload.get("freeze_state") != "PRE_FROZEN":
        raise ExternalHostWitnessError("external host request is not pre-frozen")
    origin = _validate_host_identity(payload.get("origin_host"), "origin_host")
    if payload.get("origin_host_identity_sha256") != canonical_sha256(origin):
        raise ExternalHostWitnessError("origin host identity hash drifted")
    canonical_input = payload.get("canonical_input")
    runtime = payload.get("fair_runtime")
    if not isinstance(canonical_input, Mapping) or set(canonical_input) != {
        "format", "ffv1_sha256", "conversion_receipt_sha256"
    } or canonical_input.get("format") != "FFV1_RGB24_CFR":
        raise ExternalHostWitnessError("canonical input identity is invalid")
    _require_hash(canonical_input.get("ffv1_sha256"), "canonical_input.ffv1_sha256")
    _require_hash(canonical_input.get("conversion_receipt_sha256"), "canonical_input.conversion_receipt_sha256")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "protocol_sha256", "schedule_sha256", "timeout_seconds", "scheduled_repeats",
        "attempts_per_slot", "retry_count", "retry_policy", "slots", "commands",
    }:
        raise ExternalHostWitnessError("frozen fair runtime identity is invalid")
    _require_hash(runtime.get("protocol_sha256"), "fair_runtime.protocol_sha256")
    _require_hash(runtime.get("schedule_sha256"), "fair_runtime.schedule_sha256")
    if (
        isinstance(runtime.get("timeout_seconds"), bool)
        or not isinstance(runtime.get("timeout_seconds"), int)
        or int(runtime["timeout_seconds"]) <= 0
        or runtime.get("scheduled_repeats") != 3
        or runtime.get("attempts_per_slot") != 1
        or runtime.get("retry_count") != 0
        or runtime.get("retry_policy") != "NO_RETRY"
        or not isinstance(runtime.get("slots"), list)
    ):
        raise ExternalHostWitnessError("frozen fair runtime budget is invalid")
    slots = runtime["slots"]
    participants = sorted({str(row.get("comparator")) for row in slots if isinstance(row, Mapping)})
    if not slots or any(not isinstance(row, Mapping) for row in slots):
        raise ExternalHostWitnessError("frozen fair runtime slots are invalid")
    source_rows = _validate_source_rows(payload.get("sources"), participants)
    if payload.get("source_population_sha256") != canonical_sha256(source_rows):
        raise ExternalHostWitnessError("source population hash drifted")
    command_rows = _validate_commands(runtime.get("commands"), slots)
    tools = payload.get("required_tools")
    if (
        not isinstance(tools, list)
        or len(tools) != len(set(tools))
        or not all(isinstance(tool, str) and tool.strip() for tool in tools)
        or not set(REQUIRED_TOOLS).issubset(tools)
        or tools != sorted(tools)
    ):
        raise ExternalHostWitnessError("required tool population is invalid")
    if schema == REQUEST_SCHEMA_V2:
        frozen_fingerprints = _validate_expected_tool_fingerprints(
            payload.get("expected_tool_fingerprints"), tools
        )
        if "execution_boundary" in payload:
            _validate_execution_boundary(
                payload.get("execution_boundary"),
                commands=command_rows,
                required_tools=tools,
                tool_fingerprints=frozen_fingerprints,
            )
    policy = payload.get("transport_policy")
    if policy != {
        "relative_posix_artifact_paths_only": True,
        "reject_symlinks": True,
        "reject_unmanifested_files": True,
        "distinct_host_identity_required": True,
    }:
        raise ExternalHostWitnessError("transport policy drifted")
    return dict(payload)


def write_external_host_witness_request(path: Path | str, **kwargs: object) -> dict[str, object]:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"external host request already exists: {destination}")
    request = freeze_external_host_witness_request(**kwargs)  # type: ignore[arg-type]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**request, "request": str(destination), "request_artifact_sha256": _file_sha256(destination)}


def _safe_artifact(root: Path, relative: object) -> Path:
    text = _require_text(relative, "artifact.path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ExternalHostWitnessError("transport artifact path escapes bundle")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ExternalHostWitnessError("transport artifact is missing or a symlink")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ExternalHostWitnessError("transport artifact path escapes bundle") from exc
    return path


def _validate_artifact_ref(
    root: Path, payload: object, manifest: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {"path", "sha256", "size"}:
        raise ExternalHostWitnessError("transport artifact reference fields are invalid")
    path = _safe_artifact(root, payload.get("path"))
    relative = str(payload["path"])
    expected = manifest.get(relative)
    observed = {"path": relative, "sha256": _file_sha256(path), "size": path.stat().st_size}
    if expected is None or dict(payload) != observed or dict(expected) != observed:
        raise ExternalHostWitnessError("transport artifact hash or size drifted")
    return observed


def _validate_external_root(value: object, workspace_root: PurePosixPath, field: str) -> str:
    text = _require_text(value, field)
    path = PurePosixPath(text)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExternalHostWitnessError(f"{field} is not an absolute normalized host path")
    try:
        path.relative_to(workspace_root)
    except ValueError as exc:
        raise ExternalHostWitnessError(f"{field} escapes the external workspace") from exc
    return text


def _validate_boundary_probes(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Mapping[str, object]],
    execution: Mapping[str, object],
    request: Mapping[str, object],
    host_identity_sha256: str,
) -> tuple[int, int]:
    boundary = request["execution_boundary"]
    preparation_contract = boundary["preparation"]
    timed_contract = boundary["timed_execution"]
    preparation_ref = _validate_artifact_ref(
        bundle_root, execution.get("preparation_probe"), manifest
    )
    privilege_ref = _validate_artifact_ref(
        bundle_root, execution.get("privilege_probe"), manifest
    )
    try:
        preparation = json.loads(
            _safe_artifact(bundle_root, preparation_ref["path"]).read_text(
                encoding="utf-8"
            )
        )
        privilege = json.loads(
            _safe_artifact(bundle_root, privilege_ref["path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalHostWitnessError("external host boundary probe is unreadable") from exc
    expected_preparation_fields = {
        "schema",
        "request_sha256",
        "host_identity_sha256",
        "identity",
        "source_setup_timing",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "wall_time_ns",
        "detector_slots_started",
        "status",
        "identity_probes",
    }
    if (
        not isinstance(preparation, Mapping)
        or set(preparation) != expected_preparation_fields
        or preparation.get("schema") != PREPARATION_PROBE_SCHEMA
        or preparation.get("request_sha256") != canonical_sha256(request)
        or preparation.get("host_identity_sha256") != host_identity_sha256
        or preparation.get("identity") != preparation_contract
        or preparation.get("source_setup_timing")
        != "OUTSIDE_MEASUREMENT_BOUNDARY"
        or preparation.get("detector_slots_started") is not False
        or preparation.get("status") != "VERIFIED"
    ):
        raise ExternalHostWitnessError("external host preparation probe drifted")
    prep_started = preparation.get("started_monotonic_ns")
    prep_finished = preparation.get("finished_monotonic_ns")
    prep_wall = preparation.get("wall_time_ns")
    if (
        isinstance(prep_started, bool)
        or not isinstance(prep_started, int)
        or prep_started < 0
        or isinstance(prep_finished, bool)
        or not isinstance(prep_finished, int)
        or prep_finished <= prep_started
        or isinstance(prep_wall, bool)
        or not isinstance(prep_wall, int)
        or prep_wall != prep_finished - prep_started
    ):
        raise ExternalHostWitnessError("external host preparation timing is invalid")
    identity_probes = preparation.get("identity_probes")
    identity_contracts = preparation_contract["identity_probe_contracts"]
    if (
        not isinstance(identity_probes, list)
        or len(identity_probes) != len(identity_contracts)
    ):
        raise ExternalHostWitnessError(
            "external host preparation identity probes are incomplete"
        )
    prior_probe_finished: int | None = None
    expected_identity_probe_fields = {
        "name",
        "command",
        "command_sha256",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "wall_time_ns",
        "exit_code",
        "stdout",
        "stderr",
    }
    for contract, observed in zip(identity_contracts, identity_probes, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != expected_identity_probe_fields:
            raise ExternalHostWitnessError(
                "external host preparation identity probe fields are invalid"
            )
        command = contract["command"]
        started = observed.get("started_monotonic_ns")
        finished = observed.get("finished_monotonic_ns")
        wall = observed.get("wall_time_ns")
        if (
            observed.get("name") != contract["name"]
            or observed.get("command") != command
            or observed.get("command_sha256") != canonical_sha256(command)
            or observed.get("exit_code") != 0
        ):
            raise ExternalHostWitnessError(
                "external host preparation identity probe drifted"
            )
        if (
            isinstance(started, bool)
            or not isinstance(started, int)
            or started < prep_started
            or isinstance(finished, bool)
            or not isinstance(finished, int)
            or finished <= started
            or finished > prep_finished
            or isinstance(wall, bool)
            or not isinstance(wall, int)
            or wall != finished - started
            or (prior_probe_finished is not None and started < prior_probe_finished)
        ):
            raise ExternalHostWitnessError(
                "external host preparation identity probe timing is invalid"
            )
        stdout_ref = _validate_artifact_ref(
            bundle_root, observed.get("stdout"), manifest
        )
        stderr_ref = _validate_artifact_ref(
            bundle_root, observed.get("stderr"), manifest
        )
        stdout = _safe_artifact(bundle_root, stdout_ref["path"]).read_bytes()
        stderr = _safe_artifact(bundle_root, stderr_ref["path"]).read_bytes()
        if stdout != contract["expected_stdout"].encode("utf-8") or stderr != b"":
            raise ExternalHostWitnessError(
                "external host preparation identity probe output drifted"
            )
        prior_probe_finished = finished
    expected_privilege_fields = {
        "schema",
        "request_sha256",
        "host_identity_sha256",
        "preparation_identity",
        "command",
        "command_sha256",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "wall_time_ns",
        "exit_code",
        "stdout",
        "stderr",
        "observed_effective_uid",
    }
    if not isinstance(privilege, Mapping) or set(privilege) != expected_privilege_fields:
        raise ExternalHostWitnessError("external host privilege probe fields are invalid")
    command = timed_contract["probe_command"]
    if (
        privilege.get("schema") != PRIVILEGE_PROBE_SCHEMA
        or privilege.get("request_sha256") != canonical_sha256(request)
        or privilege.get("host_identity_sha256") != host_identity_sha256
        or privilege.get("preparation_identity") != preparation_contract
        or privilege.get("command") != command
        or privilege.get("command_sha256") != canonical_sha256(command)
        or privilege.get("exit_code") != 0
        or privilege.get("observed_effective_uid")
        != timed_contract["required_effective_uid"]
    ):
        raise ExternalHostWitnessError("external host privilege probe drifted")
    privilege_started = privilege.get("started_monotonic_ns")
    privilege_finished = privilege.get("finished_monotonic_ns")
    privilege_wall = privilege.get("wall_time_ns")
    if (
        isinstance(privilege_started, bool)
        or not isinstance(privilege_started, int)
        or privilege_started < prep_finished
        or isinstance(privilege_finished, bool)
        or not isinstance(privilege_finished, int)
        or privilege_finished <= privilege_started
        or isinstance(privilege_wall, bool)
        or not isinstance(privilege_wall, int)
        or privilege_wall != privilege_finished - privilege_started
    ):
        raise ExternalHostWitnessError("external host privilege probe timing is invalid")
    stdout_ref = _validate_artifact_ref(bundle_root, privilege.get("stdout"), manifest)
    stderr_ref = _validate_artifact_ref(bundle_root, privilege.get("stderr"), manifest)
    stdout = _safe_artifact(bundle_root, stdout_ref["path"]).read_bytes()
    stderr = _safe_artifact(bundle_root, stderr_ref["path"]).read_bytes()
    if (
        stdout != timed_contract["probe_expected_stdout"].encode("utf-8")
        or stderr != b""
    ):
        raise ExternalHostWitnessError("external host privilege probe output drifted")
    return prep_finished, privilege_finished


def verify_external_host_witness(
    request_path: Path | str,
    receipt_path: Path | str,
    *,
    expected_protocol_sha256: str,
    expected_schedule_sha256: str,
    expected_input_sha256: str,
    local_host_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Reopen a transported external-host bundle without trusting host paths."""
    failures: list[str] = []
    host_summary: dict[str, object] | None = None
    verified_slots: list[dict[str, object]] = []
    verification_schema = VERIFICATION_SCHEMA_V2
    trusted_request_path = Path(request_path).resolve()
    received_receipt_path = Path(receipt_path).resolve()
    try:
        request_bytes = trusted_request_path.read_bytes()
        request = _validate_request(json.loads(request_bytes))
        request_schema = request["schema"]
        has_execution_boundary = (
            request_schema == REQUEST_SCHEMA_V2
            and "execution_boundary" in request
        )
        verification_schema = (
            VERIFICATION_SCHEMA_V2
            if request_schema == REQUEST_SCHEMA_V2
            else VERIFICATION_SCHEMA_V1
        )
        request_artifact_sha = hashlib.sha256(request_bytes).hexdigest()
        local_identity = _validate_host_identity(
            capture_host_identity() if local_host_identity is None else local_host_identity,
            "local_host_identity",
        )
        if request["origin_host"] != local_identity:
            raise ExternalHostWitnessError("trusted request origin differs from the verifier host")
        runtime = request["fair_runtime"]
        canonical_input = request["canonical_input"]
        if runtime["protocol_sha256"] != _require_hash(expected_protocol_sha256, "expected_protocol_sha256"):
            raise ExternalHostWitnessError("external host request protocol drifted")
        if runtime["schedule_sha256"] != _require_hash(expected_schedule_sha256, "expected_schedule_sha256"):
            raise ExternalHostWitnessError("external host request schedule drifted")
        if canonical_input["ffv1_sha256"] != _require_hash(expected_input_sha256, "expected_input_sha256"):
            raise ExternalHostWitnessError("external host request input drifted")
        if received_receipt_path.is_symlink() or not received_receipt_path.is_file():
            raise ExternalHostWitnessError("external host receipt is missing or a symlink")
        bundle_root = received_receipt_path.parent.resolve()
        receipt = json.loads(received_receipt_path.read_text(encoding="utf-8"))
        expected_receipt_fields = {
            "schema", "request", "transport_manifest", "host", "execution",
            "status", "claim_status", "scoreable", "comparison_eligible",
        }
        expected_receipt_schema = (
            RECEIPT_SCHEMA_V2
            if request_schema == REQUEST_SCHEMA_V2
            else RECEIPT_SCHEMA_V1
        )
        if not isinstance(receipt, Mapping) or set(receipt) != expected_receipt_fields or receipt.get("schema") != expected_receipt_schema:
            raise ExternalHostWitnessError("external host receipt fields are invalid")
        if receipt.get("status") != "WITNESSED" or receipt.get("claim_status") != "NOT_SCOREABLE" or receipt.get("scoreable") is not False or receipt.get("comparison_eligible") is not False:
            raise ExternalHostWitnessError("external host receipt claim boundary drifted")
        manifest_ref = receipt.get("transport_manifest")
        if not isinstance(manifest_ref, Mapping) or set(manifest_ref) != {"path", "sha256"}:
            raise ExternalHostWitnessError("transport manifest reference is invalid")
        manifest_path = _safe_artifact(bundle_root, manifest_ref.get("path"))
        if manifest_ref.get("sha256") != _file_sha256(manifest_path):
            raise ExternalHostWitnessError("transport manifest hash drifted")
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, Mapping) or set(manifest_payload) != {"schema", "entries"} or manifest_payload.get("schema") != MANIFEST_SCHEMA:
            raise ExternalHostWitnessError("transport manifest fields are invalid")
        entries = manifest_payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ExternalHostWitnessError("transport manifest is empty")
        manifest: dict[str, Mapping[str, object]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256", "size"}:
                raise ExternalHostWitnessError("transport manifest entry fields are invalid")
            relative = _require_text(entry.get("path"), "transport_manifest.path")
            if relative in manifest:
                raise ExternalHostWitnessError("transport manifest contains duplicate paths")
            _require_hash(entry.get("sha256"), "transport_manifest.sha256")
            if isinstance(entry.get("size"), bool) or not isinstance(entry.get("size"), int) or int(entry["size"]) < 0:
                raise ExternalHostWitnessError("transport manifest size is invalid")
            _safe_artifact(bundle_root, relative)
            manifest[relative] = entry
        request_ref = _validate_artifact_ref(bundle_root, receipt.get("request"), manifest)
        request_copy = _safe_artifact(bundle_root, request_ref["path"])
        if request_ref["sha256"] != request_artifact_sha or request_copy.read_bytes() != request_bytes:
            raise ExternalHostWitnessError("received request differs from the trusted pre-frozen request")
        allowed_files = set(manifest) | {
            received_receipt_path.relative_to(bundle_root).as_posix(),
            manifest_path.relative_to(bundle_root).as_posix(),
        }
        observed_files: set[str] = set()
        for candidate in bundle_root.rglob("*"):
            if candidate.is_symlink():
                raise ExternalHostWitnessError("transport bundle contains a symlink")
            if candidate.is_file():
                observed_files.add(candidate.relative_to(bundle_root).as_posix())
        if observed_files != allowed_files:
            raise ExternalHostWitnessError("transport bundle contains unmanifested files")
        host = receipt.get("host")
        expected_host_fields = (
            {"identity", "identity_sha256", "workspace_root", "tool_fingerprints"}
            if request_schema == REQUEST_SCHEMA_V2
            else {"identity", "identity_sha256", "workspace_root", "tools"}
        )
        if not isinstance(host, Mapping) or set(host) != expected_host_fields:
            raise ExternalHostWitnessError("external host fingerprint fields are invalid")
        host_identity = _validate_host_identity(host.get("identity"), "external_host.identity")
        host_identity_sha = canonical_sha256(host_identity)
        if host.get("identity_sha256") != host_identity_sha:
            raise ExternalHostWitnessError("external host identity hash drifted")
        if host_identity["machine_id"] == local_identity["machine_id"] or host_identity_sha == request["origin_host_identity_sha256"]:
            raise ExternalHostWitnessError("external host identity is not distinct")
        workspace_text = _require_text(host.get("workspace_root"), "external_host.workspace_root")
        workspace_root = PurePosixPath(workspace_text)
        if not workspace_root.is_absolute() or any(part in {"", ".", ".."} for part in workspace_root.parts):
            raise ExternalHostWitnessError("external host workspace root is invalid")
        if request_schema == REQUEST_SCHEMA_V2:
            frozen_tools: object = _validate_expected_tool_fingerprints(
                host.get("tool_fingerprints"),
                request["required_tools"],
                "external_host.tool_fingerprints",
            )
            if frozen_tools != request["expected_tool_fingerprints"]:
                raise ExternalHostWitnessError(
                    "external host tool fingerprints differ from the pre-frozen request"
                )
            host_summary = {
                "identity": host_identity,
                "identity_sha256": host_identity_sha,
                "tool_fingerprints": frozen_tools,
            }
            if has_execution_boundary:
                host_summary["execution_boundary"] = request["execution_boundary"]
        else:
            tools = host.get("tools")
            if not isinstance(tools, Mapping) or set(tools) != set(request["required_tools"]):
                raise ExternalHostWitnessError("external host tool population drifted")
            legacy_frozen_tools: dict[str, object] = {}
            for name in sorted(tools):
                tool = tools[name]
                if not isinstance(tool, Mapping) or set(tool) != {"path", "sha256", "version"}:
                    raise ExternalHostWitnessError("external host tool fingerprint fields are invalid")
                binary_path = PurePosixPath(_require_text(tool.get("path"), f"tool.{name}.path"))
                if not binary_path.is_absolute() or any(part in {"", ".", ".."} for part in binary_path.parts):
                    raise ExternalHostWitnessError("external host tool path is invalid")
                _require_hash(tool.get("sha256"), f"tool.{name}.sha256")
                _require_text(tool.get("version"), f"tool.{name}.version")
                legacy_frozen_tools[name] = dict(tool)
            frozen_tools = legacy_frozen_tools
            host_summary = {"identity": host_identity, "identity_sha256": host_identity_sha, "tools": frozen_tools}
        execution = receipt.get("execution")
        expected_execution_fields = {
            "request_sha256", "protocol_sha256", "schedule_sha256", "input_sha256",
            "conversion_receipt_sha256", "source_population_sha256", "source_manifests",
            "slots", "slots_sha256",
        }
        if request_schema == REQUEST_SCHEMA_V2:
            expected_execution_fields.add("tool_fingerprints_sha256")
        if has_execution_boundary:
            expected_execution_fields.update(
                {"preparation_probe", "privilege_probe"}
            )
        if not isinstance(execution, Mapping) or set(execution) != expected_execution_fields:
            raise ExternalHostWitnessError("external host execution fields are invalid")
        expected_bindings = {
            "request_sha256": canonical_sha256(request),
            "protocol_sha256": runtime["protocol_sha256"],
            "schedule_sha256": runtime["schedule_sha256"],
            "input_sha256": canonical_input["ffv1_sha256"],
            "conversion_receipt_sha256": canonical_input["conversion_receipt_sha256"],
            "source_population_sha256": request["source_population_sha256"],
        }
        if request_schema == REQUEST_SCHEMA_V2:
            expected_bindings["tool_fingerprints_sha256"] = canonical_sha256(
                request["expected_tool_fingerprints"]
            )
        if any(execution.get(field) != value for field, value in expected_bindings.items()):
            raise ExternalHostWitnessError("external host execution binding drifted")
        privilege_finished: int | None = None
        if has_execution_boundary:
            _, privilege_finished = _validate_boundary_probes(
                bundle_root=bundle_root,
                manifest=manifest,
                execution=execution,
                request=request,
                host_identity_sha256=host_identity_sha,
            )
        source_manifests = execution.get("source_manifests")
        if not isinstance(source_manifests, list) or len(source_manifests) != len(request["sources"]):
            raise ExternalHostWitnessError("transported source manifest population is incomplete")
        source_by_comparator = {row["comparator"]: row for row in request["sources"]}
        observed_source_comparators: set[str] = set()
        for source_manifest in source_manifests:
            if not isinstance(source_manifest, Mapping) or set(source_manifest) != {"comparator", "artifact"}:
                raise ExternalHostWitnessError("transported source manifest fields are invalid")
            comparator = source_manifest.get("comparator")
            if not isinstance(comparator, str) or comparator in observed_source_comparators or comparator not in source_by_comparator:
                raise ExternalHostWitnessError("transported source manifest identity is invalid")
            artifact = _validate_artifact_ref(bundle_root, source_manifest.get("artifact"), manifest)
            if artifact["sha256"] != source_by_comparator[comparator]["source_manifest_sha256"]:
                raise ExternalHostWitnessError("transported source manifest drifted")
            observed_source_comparators.add(comparator)
        if observed_source_comparators != set(source_by_comparator):
            raise ExternalHostWitnessError("transported source manifest population drifted")
        slots = execution.get("slots")
        if not isinstance(slots, list) or len(slots) != len(runtime["slots"]):
            raise ExternalHostWitnessError("external host slot set is incomplete")
        expected_commands = {row["slot"]: row["command"] for row in runtime["commands"]}
        previous_finished: int | None = None
        expected_slot_fields = {
            "slot", "round", "position", "comparator", "repeat_ordinal", "command",
            "timeout_seconds", "attempt_ordinal", "retry_count", "retry_policy",
            "started_monotonic_ns", "finished_monotonic_ns", "wall_time_ns", "timed_out",
            "exit_code", "environment_roots", "raw_outputs", "probes", "result",
        }
        for expected_slot, observed_slot in zip(runtime["slots"], slots, strict=True):
            if not isinstance(observed_slot, Mapping) or set(observed_slot) != expected_slot_fields:
                raise ExternalHostWitnessError("external host slot fields are invalid")
            for field in ("slot", "round", "position", "comparator", "repeat_ordinal"):
                if observed_slot.get(field) != expected_slot.get(field):
                    raise ExternalHostWitnessError("external host slot assignment drifted")
            slot_number = observed_slot["slot"]
            if observed_slot.get("command") != expected_commands.get(slot_number):
                raise ExternalHostWitnessError("external host command drifted")
            if (
                observed_slot.get("timeout_seconds") != runtime["timeout_seconds"]
                or observed_slot.get("attempt_ordinal") != 1
                or observed_slot.get("retry_count") != 0
                or observed_slot.get("retry_policy") != "NO_RETRY"
                or observed_slot.get("timed_out") is not False
                or observed_slot.get("exit_code") != 0
            ):
                raise ExternalHostWitnessError("external host retry, budget, timeout, or exit evidence drifted")
            started = observed_slot.get("started_monotonic_ns")
            finished = observed_slot.get("finished_monotonic_ns")
            wall = observed_slot.get("wall_time_ns")
            if (
                isinstance(started, bool) or not isinstance(started, int) or started < 0
                or isinstance(finished, bool) or not isinstance(finished, int) or finished <= started
                or isinstance(wall, bool) or not isinstance(wall, int) or wall != finished - started
                or wall > int(runtime["timeout_seconds"]) * 1_000_000_000
                or (previous_finished is not None and started < previous_finished)
            ):
                raise ExternalHostWitnessError("external host timing or slot isolation drifted")
            if has_execution_boundary and previous_finished is None and (
                privilege_finished is None or started < privilege_finished
            ):
                raise ExternalHostWitnessError(
                    "external host preparation or privilege timing contaminated measured slots"
                )
            previous_finished = finished
            roots = observed_slot.get("environment_roots")
            if not isinstance(roots, Mapping) or set(roots) != set(ROOT_FIELDS):
                raise ExternalHostWitnessError("external host environment root set is incomplete")
            root_values = [
                _validate_external_root(roots[field], workspace_root, f"environment_roots.{field}")
                for field in ROOT_FIELDS
            ]
            if len(root_values) != len(set(root_values)):
                raise ExternalHostWitnessError("external host environment roots are not isolated")
            for collection_name in ("raw_outputs", "probes"):
                collection = observed_slot.get(collection_name)
                if not isinstance(collection, list) or not collection:
                    raise ExternalHostWitnessError(f"external host {collection_name} evidence is missing")
                for artifact in collection:
                    _validate_artifact_ref(bundle_root, artifact, manifest)
            probes = observed_slot["probes"]
            if len(probes) != 1:
                raise ExternalHostWitnessError("external host execution probe population is ambiguous")
            probe_ref = _validate_artifact_ref(bundle_root, probes[0], manifest)
            probe_path = _safe_artifact(bundle_root, probe_ref["path"])
            try:
                probe = json.loads(probe_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExternalHostWitnessError("external host execution probe is unreadable") from exc
            expected_probe = {
                "schema": (
                    PROBE_SCHEMA_V2
                    if request_schema == REQUEST_SCHEMA_V2
                    else PROBE_SCHEMA_V1
                ),
                "slot": slot_number,
                "host_identity_sha256": host_identity_sha,
                "source": source_by_comparator[observed_slot["comparator"]],
                "input_sha256": canonical_input["ffv1_sha256"],
                "conversion_receipt_sha256": canonical_input["conversion_receipt_sha256"],
                "protocol_sha256": runtime["protocol_sha256"],
                "schedule_sha256": runtime["schedule_sha256"],
                "command_sha256": canonical_sha256(observed_slot["command"]),
                "tool_fingerprints_sha256": canonical_sha256(frozen_tools),
                "environment_roots": dict(roots),
                "started_monotonic_ns": started,
                "finished_monotonic_ns": finished,
                "wall_time_ns": wall,
            }
            if request_schema == REQUEST_SCHEMA_V2:
                expected_probe["tool_fingerprints"] = request[
                    "expected_tool_fingerprints"
                ]
            if probe != expected_probe:
                raise ExternalHostWitnessError("external host execution probe drifted")
            result_ref = _validate_artifact_ref(
                bundle_root, observed_slot.get("result"), manifest
            )
            result_path = _safe_artifact(bundle_root, result_ref["path"])
            verified_slots.append(
                {
                    "slot": observed_slot["slot"],
                    "round": observed_slot["round"],
                    "position": observed_slot["position"],
                    "comparator": observed_slot["comparator"],
                    "repeat_ordinal": observed_slot["repeat_ordinal"],
                    "result": {
                        **result_ref,
                        "path": str(result_path.resolve()),
                    },
                }
            )
        if execution.get("slots_sha256") != canonical_sha256(slots):
            raise ExternalHostWitnessError("external host result or timing ledger hash drifted")
    except (OSError, json.JSONDecodeError, ExternalHostWitnessError, TypeError, ValueError) as exc:
        failures.append(str(exc) or exc.__class__.__name__)
    verified = not failures
    return {
        "schema": verification_schema,
        "status": "VERIFIED" if verified else "INCONCLUSIVE",
        "witness_verified": verified,
        "independent_host_identity_verified": verified,
        "host": host_summary,
        "request": str(trusted_request_path),
        "receipt": str(received_receipt_path),
        "verified_slots": verified_slots if verified else [],
        "failures": failures,
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "independent_gold_receipt_missing",
            "fair_population_receipt_conditions_unproven",
            "receipt_bound_score_verifier_missing",
        ],
    }
