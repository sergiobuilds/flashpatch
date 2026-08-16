from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from flashpatch import external_league
from flashpatch.external_league import (
    FairRuntimeProtocol,
    freeze_fair_runtime_protocol,
    freeze_fair_runtime_schedule,
)
from flashpatch.l7_external_host import (
    EXECUTION_BOUNDARY_SCHEMA,
    MANIFEST_SCHEMA,
    PREPARATION_PROBE_SCHEMA,
    PRIVILEGE_PROBE_SCHEMA,
    PROBE_SCHEMA,
    PROBE_SCHEMA_V1,
    RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_V1,
    REQUIRED_TOOLS,
    REQUEST_SCHEMA_V1,
    VERIFICATION_SCHEMA,
    VERIFICATION_SCHEMA_V1,
    canonical_sha256,
    capture_host_identity,
    verify_external_host_witness,
    write_external_host_witness_request,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _protocol(tmp_path: Path) -> dict[str, object]:
    return freeze_fair_runtime_protocol(
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


def test_fair_runtime_protocol_accepts_explicit_docker_empty_device_isolation(tmp_path: Path) -> None:
    protocol = freeze_fair_runtime_protocol(
        FairRuntimeProtocol(
            machine_id="docker-machine",
            operating_system="Linux",
            architecture="x86_64",
            cpu_model="Frozen CPU",
            logical_cpu_count=8,
            cpu_affinity=(0,),
            thread_limit=1,
            gpu_policy="DISABLED",
            gpu_device=None,
            gpu_isolation="DOCKER_EMPTY_DEV",
            cache_policy="WARM_INPUT_PRETOUCHED",
            concurrency_limit=1,
            concurrency_lock_path=str((tmp_path / "runtime.lock").resolve()),
            process_isolation="FRESH_SUBPROCESS_PER_REPEAT",
            timeout_seconds=120,
        )
    )
    assert protocol["gpu"] == {
        "policy": "DISABLED",
        "device": None,
        "isolation": "DOCKER_EMPTY_DEV",
    }


def test_docker_empty_device_protocol_cannot_be_captured_outside_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(external_league, "_inside_docker", lambda: False)
    with pytest.raises(external_league.ExternalLeagueError, match="inside Docker"):
        external_league.capture_fair_runtime_protocol(
            concurrency_lock_path=tmp_path / "runtime.lock",
            timeout_seconds=120,
            gpu_isolation="DOCKER_EMPTY_DEV",
        )


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "size": path.stat().st_size,
    }


def _case(
    tmp_path: Path, *, version: int = 2, privileged: bool = False
) -> dict[str, object]:
    protocol = _protocol(tmp_path)
    input_sha = "a" * 64
    conversion_sha = "b" * 64
    schedule = freeze_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy"], protocol, input_sha, seed=20260803
    )
    origin = capture_host_identity()
    source_manifest_bytes = {
        "FlashPatch": b"FlashPatch source manifest\n",
        "TooFlashy": b"TooFlashy source manifest\n",
    }
    launcher_prefix = ["/usr/bin/sudo", "-n", "--"]
    commands = {
        row["slot"]: [
            *(launcher_prefix if privileged else []),
            "/usr/bin/python3",
            "/opt/flashpatch/runner.py",
            "--slot",
            str(row["slot"]),
        ]
        for row in schedule["slots"]
    }
    sources = [
        {
            "comparator": "FlashPatch",
            "repository_url": "https://github.com/sergiobuilds/flashpatch",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "source_manifest_sha256": hashlib.sha256(source_manifest_bytes["FlashPatch"]).hexdigest(),
        },
        {
            "comparator": "TooFlashy",
            "repository_url": "https://github.com/hashb/TooFlashy",
            "revision": "4" * 40,
            "tree": "5" * 40,
            "source_manifest_sha256": hashlib.sha256(source_manifest_bytes["TooFlashy"]).hexdigest(),
        },
    ]
    trusted_request = tmp_path / "trusted-request.json"
    required_tools = tuple(
        sorted((*REQUIRED_TOOLS, "sudo")) if privileged else REQUIRED_TOOLS
    )
    expected_tool_fingerprints = [
        {
            "name": name,
            "path": f"/usr/bin/{name}",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "version_output": f"{name} test version\n",
        }
        for name in required_tools
    ]
    request_kwargs: dict[str, object] = {
        "origin_host": origin,
        "sources": sources,
        "canonical_ffv1_sha256": input_sha,
        "conversion_receipt_sha256": conversion_sha,
        "fair_runtime_protocol": protocol,
        "fair_runtime_schedule": schedule,
        "slot_commands": commands,
    }
    if version == 2:
        request_kwargs["expected_tool_fingerprints"] = expected_tool_fingerprints
        request_kwargs["required_tools"] = required_tools
        if privileged:
            request_kwargs["execution_boundary"] = {
                "schema": EXECUTION_BOUNDARY_SCHEMA,
                "preparation": {
                    "mode": "UNPRIVILEGED_USER",
                    "uid": 1000,
                    "gid": 1000,
                    "username": "external-runner",
                    "source_setup_timing": "OUTSIDE_MEASUREMENT_BOUNDARY",
                    "identity_probe_contracts": [
                        {
                            "name": "uid",
                            "command": ["/usr/bin/id", "-u"],
                            "expected_stdout": "1000\n",
                        },
                        {
                            "name": "gid",
                            "command": ["/usr/bin/id", "-g"],
                            "expected_stdout": "1000\n",
                        },
                        {
                            "name": "username",
                            "command": ["/usr/bin/id", "-un"],
                            "expected_stdout": "external-runner\n",
                        },
                    ],
                },
                "timed_execution": {
                    "mode": "SUDO_BWRAP_EXACT_COMMAND",
                    "launcher_prefix": launcher_prefix,
                    "non_interactive": True,
                    "preserve_environment": False,
                    "required_effective_uid": 0,
                    "policy_scope": "NON_INTERACTIVE_EXECUTION_AVAILABLE_ONLY",
                    "probe_command": [*launcher_prefix, "/usr/bin/id", "-u"],
                    "probe_expected_stdout": "0\n",
                },
            }
    elif version == 1:
        request_kwargs["request_schema"] = REQUEST_SCHEMA_V1
    else:
        raise AssertionError(f"unsupported test witness version: {version}")
    request = write_external_host_witness_request(
        trusted_request,
        **request_kwargs,
    )
    request_payload = {key: value for key, value in request.items() if key not in {"request", "request_artifact_sha256"}}
    bundle = tmp_path / "received"
    bundle.mkdir()
    request_copy = bundle / "request.json"
    request_copy.write_bytes(trusted_request.read_bytes())
    entries = [_artifact(request_copy, bundle)]
    source_manifest_refs: list[dict[str, object]] = []
    for comparator, content in source_manifest_bytes.items():
        source_path = bundle / "sources" / f"{comparator}.manifest"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(content)
        source_ref = _artifact(source_path, bundle)
        entries.append(source_ref)
        source_manifest_refs.append({"comparator": comparator, "artifact": source_ref})
    source_manifest_refs.sort(key=lambda row: row["comparator"])
    external_identity = {
        **origin,
        "machine_id": f"external-{origin['machine_id']}",
        "hostname": "external-runner",
    }
    legacy_tools = {
        name: {
            "path": f"/usr/bin/{name}",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "version": f"{name} test version",
        }
        for name in REQUIRED_TOOLS
    }
    source_by_comparator = {row["comparator"]: row for row in request_payload["sources"]}
    slots: list[dict[str, object]] = []
    cursor = 1_000_000_000
    for scheduled in schedule["slots"]:
        ordinal = scheduled["slot"]
        slot_root = bundle / "artifacts" / f"slot-{ordinal:02d}"
        slot_root.mkdir(parents=True)
        raw = slot_root / "raw.json"
        probe = slot_root / "probe.json"
        result = slot_root / "result.json"
        _write_json(raw, {"slot": ordinal, "raw": True})
        started = cursor
        finished = started + 10_000_000
        cursor = finished + 1_000_000
        roots = {
            field: f"/srv/flashpatch/slot-{ordinal:02d}/{field}"
            for field in (
                "cache_root",
                "checkout_root",
                "output_root",
                "runtime_root",
                "temporary_root",
                "working_directory",
            )
        }
        probe_payload = {
            "schema": PROBE_SCHEMA if version == 2 else PROBE_SCHEMA_V1,
            "slot": ordinal,
            "host_identity_sha256": canonical_sha256(external_identity),
            "source": source_by_comparator[scheduled["comparator"]],
            "input_sha256": input_sha,
            "conversion_receipt_sha256": conversion_sha,
            "protocol_sha256": canonical_sha256(protocol),
            "schedule_sha256": canonical_sha256(schedule),
            "command_sha256": canonical_sha256(commands[ordinal]),
            "tool_fingerprints_sha256": canonical_sha256(
                expected_tool_fingerprints if version == 2 else legacy_tools
            ),
            "environment_roots": roots,
            "started_monotonic_ns": started,
            "finished_monotonic_ns": finished,
            "wall_time_ns": finished - started,
        }
        if version == 2:
            probe_payload["tool_fingerprints"] = expected_tool_fingerprints
        _write_json(probe, probe_payload)
        _write_json(result, {"slot": ordinal, "result": "PROCESS_VALID"})
        raw_ref = _artifact(raw, bundle)
        probe_ref = _artifact(probe, bundle)
        result_ref = _artifact(result, bundle)
        entries.extend([raw_ref, probe_ref, result_ref])
        slots.append(
            {
                **scheduled,
                "command": commands[ordinal],
                "timeout_seconds": 120,
                "attempt_ordinal": 1,
                "retry_count": 0,
                "retry_policy": "NO_RETRY",
                "started_monotonic_ns": started,
                "finished_monotonic_ns": finished,
                "wall_time_ns": finished - started,
                "timed_out": False,
                "exit_code": 0,
                "environment_roots": roots,
                "raw_outputs": [raw_ref],
                "probes": [probe_ref],
                "result": result_ref,
            }
        )
    boundary_execution: dict[str, object] = {}
    if privileged:
        boundary_root = bundle / "boundary"
        boundary_root.mkdir()
        preparation_path = boundary_root / "preparation.json"
        privilege_path = boundary_root / "privilege.json"
        stdout_path = boundary_root / "id-u.stdout.bin"
        stderr_path = boundary_root / "id-u.stderr.bin"
        stdout_path.write_bytes(b"0\n")
        stderr_path.write_bytes(b"")
        identity_probes: list[dict[str, object]] = []
        identity_artifacts: list[dict[str, object]] = []
        identity_contracts = request_payload["execution_boundary"]["preparation"]["identity_probe_contracts"]
        for index, contract in enumerate(identity_contracts, start=1):
            identity_stdout = boundary_root / f"preparation-{contract['name']}.stdout.bin"
            identity_stderr = boundary_root / f"preparation-{contract['name']}.stderr.bin"
            identity_stdout.write_text(contract["expected_stdout"], encoding="utf-8")
            identity_stderr.write_bytes(b"")
            started = 100_000_000 + index * 10_000_000
            finished = started + 1_000_000
            identity_probes.append(
                {
                    "name": contract["name"],
                    "command": contract["command"],
                    "command_sha256": canonical_sha256(contract["command"]),
                    "started_monotonic_ns": started,
                    "finished_monotonic_ns": finished,
                    "wall_time_ns": finished - started,
                    "exit_code": 0,
                    "stdout": _artifact(identity_stdout, bundle),
                    "stderr": _artifact(identity_stderr, bundle),
                }
            )
            identity_artifacts.extend(
                [_artifact(identity_stdout, bundle), _artifact(identity_stderr, bundle)]
            )
        preparation = {
            "schema": PREPARATION_PROBE_SCHEMA,
            "request_sha256": canonical_sha256(request_payload),
            "host_identity_sha256": canonical_sha256(external_identity),
            "identity": request_payload["execution_boundary"]["preparation"],
            "source_setup_timing": "OUTSIDE_MEASUREMENT_BOUNDARY",
            "started_monotonic_ns": 100_000_000,
            "finished_monotonic_ns": 200_000_000,
            "wall_time_ns": 100_000_000,
            "detector_slots_started": False,
            "status": "VERIFIED",
            "identity_probes": identity_probes,
        }
        _write_json(preparation_path, preparation)
        privilege = {
            "schema": PRIVILEGE_PROBE_SCHEMA,
            "request_sha256": canonical_sha256(request_payload),
            "host_identity_sha256": canonical_sha256(external_identity),
            "preparation_identity": request_payload["execution_boundary"]["preparation"],
            "command": request_payload["execution_boundary"]["timed_execution"]["probe_command"],
            "command_sha256": canonical_sha256(
                request_payload["execution_boundary"]["timed_execution"]["probe_command"]
            ),
            "started_monotonic_ns": 300_000_000,
            "finished_monotonic_ns": 310_000_000,
            "wall_time_ns": 10_000_000,
            "exit_code": 0,
            "stdout": _artifact(stdout_path, bundle),
            "stderr": _artifact(stderr_path, bundle),
            "observed_effective_uid": 0,
        }
        _write_json(privilege_path, privilege)
        entries.extend(
            [
                *identity_artifacts,
                _artifact(stdout_path, bundle),
                _artifact(stderr_path, bundle),
                _artifact(preparation_path, bundle),
                _artifact(privilege_path, bundle),
            ]
        )
        boundary_execution = {
            "preparation_probe": _artifact(preparation_path, bundle),
            "privilege_probe": _artifact(privilege_path, bundle),
        }
    manifest = {"schema": MANIFEST_SCHEMA, "entries": entries}
    manifest_path = bundle / "transport-manifest.json"
    _write_json(manifest_path, manifest)
    receipt = {
        "schema": RECEIPT_SCHEMA if version == 2 else RECEIPT_SCHEMA_V1,
        "request": _artifact(request_copy, bundle),
        "transport_manifest": {"path": manifest_path.name, "sha256": _sha(manifest_path)},
        "host": {
            "identity": external_identity,
            "identity_sha256": canonical_sha256(external_identity),
            "workspace_root": "/srv/flashpatch",
            **(
                {"tool_fingerprints": expected_tool_fingerprints}
                if version == 2
                else {"tools": legacy_tools}
            ),
        },
        "execution": {
            "request_sha256": canonical_sha256(request_payload),
            "protocol_sha256": canonical_sha256(protocol),
            "schedule_sha256": canonical_sha256(schedule),
            "input_sha256": input_sha,
            "conversion_receipt_sha256": conversion_sha,
            "source_population_sha256": request_payload["source_population_sha256"],
            **(
                {
                    "tool_fingerprints_sha256": canonical_sha256(
                        expected_tool_fingerprints
                    )
                }
                if version == 2
                else {}
            ),
            **boundary_execution,
            "source_manifests": source_manifest_refs,
            "slots": slots,
            "slots_sha256": canonical_sha256(slots),
        },
        "status": "WITNESSED",
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
    }
    receipt_path = bundle / "receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "request": trusted_request,
        "receipt": receipt_path,
        "receipt_payload": receipt,
        "protocol_sha256": canonical_sha256(protocol),
        "schedule_sha256": canonical_sha256(schedule),
        "input_sha256": input_sha,
        "origin": origin,
        "bundle": bundle,
        "expected_tool_fingerprints": expected_tool_fingerprints,
        "version": version,
    }


def _verify(case: dict[str, object]) -> dict[str, object]:
    return verify_external_host_witness(
        case["request"],
        case["receipt"],
        expected_protocol_sha256=case["protocol_sha256"],
        expected_schedule_sha256=case["schedule_sha256"],
        expected_input_sha256=case["input_sha256"],
        local_host_identity=case["origin"],
    )


def _rewrite_receipt(case: dict[str, object], mutate) -> None:
    receipt = copy.deepcopy(case["receipt_payload"])
    mutate(receipt)
    _write_json(case["receipt"], receipt)
    case["receipt_payload"] = receipt


def _reseal_transport_artifact(
    case: dict[str, object], relative: str
) -> dict[str, object]:
    path = case["bundle"] / relative
    replacement = _artifact(path, case["bundle"])
    manifest_path = case["bundle"] / "transport-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        index
        for index, entry in enumerate(manifest["entries"])
        if entry["path"] == relative
    ]
    assert len(matches) == 1
    manifest["entries"][matches[0]] = replacement
    _write_json(manifest_path, manifest)
    case["receipt_payload"]["transport_manifest"]["sha256"] = _sha(manifest_path)
    return replacement


def _seal_receipt(case: dict[str, object]) -> None:
    _write_json(case["receipt"], case["receipt_payload"])


def test_external_host_witness_reopens_every_portable_binding_but_remains_unscoreable(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    result = _verify(case)

    assert result["schema"] == VERIFICATION_SCHEMA
    assert result["status"] == "VERIFIED"
    assert result["witness_verified"] is True
    assert result["independent_host_identity_verified"] is True
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert len(result["verified_slots"]) == 6
    for slot in result["verified_slots"]:
        result_ref = slot["result"]
        result_path = Path(result_ref["path"])
        assert result_path.is_file()
        assert result_ref["sha256"] == _sha(result_path)
    assert "independent_gold_receipt_missing" in result["claim_blockers"]
    assert "fair_population_receipt_conditions_unproven" in result["claim_blockers"]
    assert not {"scores", "ranking", "winner"}.intersection(result)


def test_external_host_witness_v2_reopens_explicit_privilege_boundary(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)

    result = _verify(case)

    assert result["status"] == "VERIFIED"
    assert result["witness_verified"] is True
    preparation = result["host"]["execution_boundary"]["preparation"]
    assert preparation["mode"] == "UNPRIVILEGED_USER"
    assert preparation["uid"] == 1000
    assert preparation["gid"] == 1000
    assert preparation["username"] == "external-runner"
    assert preparation["source_setup_timing"] == "OUTSIDE_MEASUREMENT_BOUNDARY"
    assert [row["name"] for row in preparation["identity_probe_contracts"]] == [
        "uid",
        "gid",
        "username",
    ]
    assert result["scoreable"] is False


@pytest.mark.parametrize("field", ["preparation_probe", "privilege_probe"])
def test_external_host_witness_v2_boundary_rejects_missing_probe(
    tmp_path: Path, field: str
) -> None:
    case = _case(tmp_path, privileged=True)
    case["receipt_payload"]["execution"].pop(field)
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("execution fields are invalid" in failure for failure in result["failures"])


def test_external_host_witness_v2_boundary_rejects_slot_without_exact_sudo_prefix(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)
    request = json.loads(case["request"].read_text(encoding="utf-8"))
    request["fair_runtime"]["commands"][0]["command"] = request["fair_runtime"]["commands"][0]["command"][3:]
    _write_json(case["request"], request)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("omits the frozen privilege launcher" in failure for failure in result["failures"])


def test_external_host_witness_v2_boundary_rejects_sudo_tool_mismatch(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)
    request = json.loads(case["request"].read_text(encoding="utf-8"))
    sudo = next(row for row in request["expected_tool_fingerprints"] if row["name"] == "sudo")
    sudo["path"] = "/usr/local/bin/sudo"
    _write_json(case["request"], request)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("sudo fingerprint differs" in failure for failure in result["failures"])


def test_external_host_witness_v2_boundary_rejects_coherently_resealed_probe_tamper(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)
    receipt = case["receipt_payload"]
    relative = receipt["execution"]["privilege_probe"]["path"]
    probe_path = case["bundle"] / relative
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["observed_effective_uid"] = 1000
    _write_json(probe_path, probe)
    receipt["execution"]["privilege_probe"] = _reseal_transport_artifact(case, relative)
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("privilege probe drifted" in failure for failure in result["failures"])


@pytest.mark.parametrize("name", ["uid", "gid", "username"])
def test_external_host_witness_v2_boundary_rejects_missing_preparation_identity_probe(
    tmp_path: Path, name: str
) -> None:
    case = _case(tmp_path, privileged=True)
    receipt = case["receipt_payload"]
    relative = receipt["execution"]["preparation_probe"]["path"]
    preparation_path = case["bundle"] / relative
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["identity_probes"] = [
        row for row in preparation["identity_probes"] if row["name"] != name
    ]
    _write_json(preparation_path, preparation)
    receipt["execution"]["preparation_probe"] = _reseal_transport_artifact(
        case, relative
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("identity probes are incomplete" in failure for failure in result["failures"])


def test_external_host_witness_v2_boundary_rejects_resealed_preparation_identity_command_mismatch(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)
    receipt = case["receipt_payload"]
    relative = receipt["execution"]["preparation_probe"]["path"]
    preparation_path = case["bundle"] / relative
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["identity_probes"][0]["command"] = ["/usr/bin/id", "-g"]
    preparation["identity_probes"][0]["command_sha256"] = canonical_sha256(
        ["/usr/bin/id", "-g"]
    )
    _write_json(preparation_path, preparation)
    receipt["execution"]["preparation_probe"] = _reseal_transport_artifact(
        case, relative
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("identity probe drifted" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("name", "replacement"),
    [("uid", b"0\n"), ("gid", b"1001\n"), ("username", b"root\n")],
)
def test_external_host_witness_v2_boundary_rejects_resealed_preparation_identity_output_mismatch(
    tmp_path: Path, name: str, replacement: bytes
) -> None:
    case = _case(tmp_path, privileged=True)
    receipt = case["receipt_payload"]
    preparation_relative = receipt["execution"]["preparation_probe"]["path"]
    preparation_path = case["bundle"] / preparation_relative
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    probe = next(row for row in preparation["identity_probes"] if row["name"] == name)
    stdout_relative = probe["stdout"]["path"]
    (case["bundle"] / stdout_relative).write_bytes(replacement)
    probe["stdout"] = _reseal_transport_artifact(case, stdout_relative)
    _write_json(preparation_path, preparation)
    receipt["execution"]["preparation_probe"] = _reseal_transport_artifact(
        case, preparation_relative
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("identity probe output drifted" in failure for failure in result["failures"])


def test_external_host_witness_v2_boundary_rejects_preparation_timing_contamination(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, privileged=True)
    receipt = case["receipt_payload"]
    preparation_relative = receipt["execution"]["preparation_probe"]["path"]
    privilege_relative = receipt["execution"]["privilege_probe"]["path"]
    preparation_path = case["bundle"] / preparation_relative
    privilege_path = case["bundle"] / privilege_relative
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["finished_monotonic_ns"] = 1_050_000_000
    preparation["wall_time_ns"] = 950_000_000
    privilege = json.loads(privilege_path.read_text(encoding="utf-8"))
    privilege["started_monotonic_ns"] = 1_050_000_001
    privilege["finished_monotonic_ns"] = 1_060_000_000
    privilege["wall_time_ns"] = 9_999_999
    _write_json(preparation_path, preparation)
    _write_json(privilege_path, privilege)
    receipt["execution"]["preparation_probe"] = _reseal_transport_artifact(
        case, preparation_relative
    )
    receipt["execution"]["privilege_probe"] = _reseal_transport_artifact(
        case, privilege_relative
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("contaminated measured slots" in failure for failure in result["failures"])


def test_external_host_witness_rejects_same_host_identity(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        receipt["host"]["identity"] = case["origin"]
        receipt["host"]["identity_sha256"] = canonical_sha256(case["origin"])

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("not distinct" in failure for failure in result["failures"])


@pytest.mark.parametrize("binding", ["protocol_sha256", "schedule_sha256", "input_sha256"])
def test_external_host_witness_rejects_trusted_protocol_schedule_or_input_drift(
    tmp_path: Path,
    binding: str,
) -> None:
    case = _case(tmp_path)
    case[binding] = "f" * 64

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("drifted" in failure for failure in result["failures"])


def test_external_host_witness_rejects_source_binding_drift(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _rewrite_receipt(
        case,
        lambda receipt: receipt["execution"].__setitem__("source_population_sha256", "f" * 64),
    )

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("execution binding drifted" in failure for failure in result["failures"])


def test_external_host_witness_rejects_transported_source_manifest_tampering(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    source_ref = case["receipt_payload"]["execution"]["source_manifests"][0]["artifact"]
    (case["bundle"] / source_ref["path"]).write_bytes(b"different source manifest\n")

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("hash or size drifted" in failure for failure in result["failures"])


@pytest.mark.parametrize("field", ["sha256", "version_output"])
def test_external_host_witness_v2_rejects_direct_tool_hash_or_version_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        fingerprints = receipt["host"]["tool_fingerprints"]
        godot = next(row for row in fingerprints if row["name"] == "godot")
        godot[field] = "f" * 64 if field == "sha256" else "rewritten godot version\n"

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("pre-frozen request" in failure for failure in result["failures"])


@pytest.mark.parametrize("mutation", ["extra", "renamed", "reordered"])
def test_external_host_witness_v2_rejects_tool_population_or_order_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        fingerprints = receipt["host"]["tool_fingerprints"]
        if mutation == "extra":
            fingerprints.append(
                {
                    "name": "unexpected",
                    "path": "/usr/bin/unexpected",
                    "sha256": "f" * 64,
                    "version_output": "unexpected version\n",
                }
            )
        elif mutation == "renamed":
            fingerprints[0]["name"] = "renamed"
        else:
            fingerprints[0], fingerprints[1] = fingerprints[1], fingerprints[0]

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any(
        "population is incomplete" in failure
        or "names or order" in failure
        for failure in result["failures"]
    )


def test_external_host_witness_v2_rejects_coherently_rewritten_tools_and_probes(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    receipt = case["receipt_payload"]
    rewritten = copy.deepcopy(receipt["host"]["tool_fingerprints"])
    rewritten[2]["sha256"] = "f" * 64
    receipt["host"]["tool_fingerprints"] = rewritten
    receipt["execution"]["tool_fingerprints_sha256"] = canonical_sha256(rewritten)
    for slot in receipt["execution"]["slots"]:
        relative = slot["probes"][0]["path"]
        probe_path = case["bundle"] / relative
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe["tool_fingerprints"] = rewritten
        probe["tool_fingerprints_sha256"] = canonical_sha256(rewritten)
        _write_json(probe_path, probe)
        slot["probes"][0] = _reseal_transport_artifact(case, relative)
    receipt["execution"]["slots_sha256"] = canonical_sha256(
        receipt["execution"]["slots"]
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("pre-frozen request" in failure for failure in result["failures"])


def test_external_host_witness_v2_rejects_coherently_resealed_source_manifest_tamper(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    receipt = case["receipt_payload"]
    source_ref = receipt["execution"]["source_manifests"][0]["artifact"]
    source_path = case["bundle"] / source_ref["path"]
    source_path.write_bytes(
        b"rewritten manifest with matching apparent tool evidence\n"
    )
    receipt["execution"]["source_manifests"][0]["artifact"] = (
        _reseal_transport_artifact(case, source_ref["path"])
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("source manifest drifted" in failure for failure in result["failures"])


def test_external_host_witness_v2_rejects_source_manifest_only_tool_expectations(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    request = json.loads(case["request"].read_text(encoding="utf-8"))
    request.pop("expected_tool_fingerprints")
    _write_json(case["request"], request)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("request fields are invalid" in failure for failure in result["failures"])


def test_external_host_witness_v2_rejects_reordered_request_tool_contract(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    request = json.loads(case["request"].read_text(encoding="utf-8"))
    request["required_tools"].reverse()
    request["expected_tool_fingerprints"].reverse()
    _write_json(case["request"], request)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("required tool population is invalid" in failure for failure in result["failures"])


def test_external_host_witness_v2_rejects_resealed_probe_tool_tamper(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    receipt = case["receipt_payload"]
    slot = receipt["execution"]["slots"][0]
    relative = slot["probes"][0]["path"]
    probe_path = case["bundle"] / relative
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["tool_fingerprints"][0]["version_output"] = "probe-only rewrite\n"
    probe["tool_fingerprints_sha256"] = canonical_sha256(
        probe["tool_fingerprints"]
    )
    _write_json(probe_path, probe)
    slot["probes"][0] = _reseal_transport_artifact(case, relative)
    receipt["execution"]["slots_sha256"] = canonical_sha256(
        receipt["execution"]["slots"]
    )
    _seal_receipt(case)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("execution probe drifted" in failure for failure in result["failures"])


def test_external_host_witness_v1_remains_historical_and_not_scoreable(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, version=1)

    result = _verify(case)

    assert result["schema"] == VERIFICATION_SCHEMA_V1
    assert result["status"] == "VERIFIED"
    assert result["witness_verified"] is True
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["comparison_eligible"] is False
    assert not {"scores", "ranking", "winner"}.intersection(result)


@pytest.mark.parametrize(
    "mutation",
    ["slot", "retry", "budget", "timing", "result-ledger"],
)
def test_external_host_witness_rejects_slot_retry_budget_timing_and_result_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        slot = receipt["execution"]["slots"][0]
        if mutation == "slot":
            slot["slot"] = 99
        elif mutation == "retry":
            slot["retry_count"] = 1
        elif mutation == "budget":
            slot["timeout_seconds"] = 121
        elif mutation == "timing":
            slot["wall_time_ns"] += 1
        else:
            receipt["execution"]["slots_sha256"] = "f" * 64

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False


@pytest.mark.parametrize("artifact_kind", ["raw", "probe", "result"])
def test_external_host_witness_rejects_raw_probe_or_result_file_tampering(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    case = _case(tmp_path)
    slot = case["receipt_payload"]["execution"]["slots"][0]
    if artifact_kind == "raw":
        relative = slot["raw_outputs"][0]["path"]
    elif artifact_kind == "probe":
        relative = slot["probes"][0]["path"]
    else:
        relative = slot["result"]["path"]
    (case["bundle"] / relative).write_bytes(b"tampered\n")

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("hash or size drifted" in failure for failure in result["failures"])


def test_external_host_witness_rejects_transport_path_escape(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        receipt["execution"]["slots"][0]["raw_outputs"][0]["path"] = "../outside.json"

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("escapes bundle" in failure for failure in result["failures"])


def test_external_host_witness_rejects_environment_root_escape(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def mutate(receipt: dict[str, object]) -> None:
        receipt["execution"]["slots"][0]["environment_roots"]["checkout_root"] = "/tmp/checkout"

    _rewrite_receipt(case, mutate)

    result = _verify(case)

    assert result["witness_verified"] is False
    assert any("escapes the external workspace" in failure for failure in result["failures"])
