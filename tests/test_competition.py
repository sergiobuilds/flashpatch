from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flashpatch.competition import (
    CHECKPOINT_LEAVES,
    ContractError,
    EA_IRIS_RELEASE_ORACLE_ID,
    L7_DIRECT_CANDIDATE_TOOLS,
    INDEPENDENT_GOLD_NOT_VERIFIED,
    checkpoint_is_reusable,
    preflight,
    verify_chain,
    validate_independent_gold,
    verify_receipt,
    write_checkpoint,
)


ROOT = Path(__file__).parents[1]


def _independent_gold_receipt(tmp_path: Path) -> Path:
    import hashlib

    payloads = {
        "source_tree_sha256": ("source-tree.bin", b"pinned-source-tree"),
        "trace_sha256": ("trace.json", b'{"fixed_fps":60}'),
        "renderer_rgb_raw_sha256": ("renderer.rgb", b"renderer-rgb"),
        "timestamps_sha256": ("timestamps.bin", b"timestamps"),
        "rubric_sha256": ("rubric.txt", b"WCAG G19 rubric"),
        "blinding_commitment_sha256": ("blinding.txt", b"blind-before-tools"),
        "operator_separation_attestation_sha256": ("operator.txt", b"separated"),
    }
    artifacts = {}
    hashes = {}
    for field, (name, content) in payloads.items():
        target = tmp_path / name
        target.write_bytes(content)
        artifacts[field] = name
        hashes[field] = hashlib.sha256(content).hexdigest()
    execution = {
        "schema": "flashpatch-renderer-engine-receipt-v1",
        "controlled_mutation": False,
        "upstream": {
            "repository_url": "https://github.com/example/game",
            "source_revision": "a" * 40,
            "license": "MIT",
            "project_path": ".",
        },
        "factual_replay": {
            "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
            "timestamps_sha256": hashes["timestamps_sha256"],
            "frame_count": 3,
            "renderer_capture": {"trace_sha256": f"sha256:{hashes['trace_sha256']}"},
        },
    }
    execution_path = tmp_path / "renderer-execution.json"
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")
    artifacts["renderer_execution_receipt_sha256"] = execution_path.name
    hashes["renderer_execution_receipt_sha256"] = hashlib.sha256(execution_path.read_bytes()).hexdigest()
    submission_hash = hashlib.sha256(b"same-input-bundle").hexdigest()
    commitment_hashes = [hashlib.sha256(f"submission-{index}".encode()).hexdigest() for index in range(3)]
    receipt = {
        "schema": "flashpatch-l7-independent-gold-v1",
        "case_id": "natural-sparta-safe-001",
        "case_class": "natural_external",
        "claim_tier": "L9_ELIGIBLE",
        "controlled_mutation": False,
        "case_freeze": {
            "public_repository_url": "https://github.com/example/game",
            "source_revision": "a" * 40,
            "license": "MIT",
            "project_subpath": ".",
            "source_tree_sha256": hashes["source_tree_sha256"],
            "trace_sha256": hashes["trace_sha256"],
            "renderer_execution_receipt_sha256": hashes["renderer_execution_receipt_sha256"],
            "renderer_rgb_raw_sha256": hashes["renderer_rgb_raw_sha256"],
            "timestamps_sha256": hashes["timestamps_sha256"],
            "frame_count": 3,
            "color_space": "sRGB_BT709",
        },
        "standard_profile": {
            "id": "wcag22-g19-v1",
            "source_url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G19.html",
            "rubric_sha256": hashes["rubric_sha256"],
        },
        "gold_authority": {
            "kind": "independent_adjudication",
            "candidate_tools_excluded": [
                *L7_DIRECT_CANDIDATE_TOOLS,
                EA_IRIS_RELEASE_ORACLE_ID,
                "FFmpeg",
            ],
            "blinding_commitment_sha256": hashes["blinding_commitment_sha256"],
            "operator_separation_attestation_sha256": hashes["operator_separation_attestation_sha256"],
        },
        "adjudication": {
            "method": "three-independent-submissions-plus-resolution",
            "resolution": "unanimous",
            "result": "SAFE",
            "submissions": [
                {
                    "rater_pseudonym": f"R{index}",
                    "public_key_fingerprint": f"key-{index}",
                    "input_bundle_sha256": submission_hash,
                    "submission_commitment_sha256": commitment_hashes[index],
                    "decision": "SAFE",
                    "intervals": [],
                }
                for index in range(3)
            ],
        },
        "artifacts": artifacts,
    }
    path = tmp_path / "independent-gold.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_legacy_self_attested_gold_is_never_independent(tmp_path: Path) -> None:
    path = _independent_gold_receipt(tmp_path)
    assert validate_independent_gold(path) == INDEPENDENT_GOLD_NOT_VERIFIED

    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["comparator"] = EA_IRIS_RELEASE_ORACLE_ID
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_independent_gold(path) == INDEPENDENT_GOLD_NOT_VERIFIED

    receipt.pop("comparator")
    receipt["claim_tier"] = "CONTROLLED_ONLY"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_independent_gold(path) == INDEPENDENT_GOLD_NOT_VERIFIED


def test_independent_gold_rejects_controlled_execution_labeled_natural(tmp_path: Path) -> None:
    path = _independent_gold_receipt(tmp_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    execution_path = path.parent / receipt["artifacts"]["renderer_execution_receipt_sha256"]
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["controlled_mutation"] = True
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")
    receipt["case_freeze"]["renderer_execution_receipt_sha256"] = hashlib.sha256(execution_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_independent_gold(path) == INDEPENDENT_GOLD_NOT_VERIFIED


@pytest.mark.parametrize(
    ("args", "exit_code", "stdout", "stderr"),
    [
        (["validate-plan", "--plan", "specs/flashpatch-competition-evidence-v1.yaml"], 0, "VALID plan=flashpatch-competition-evidence-v1", ""),
        (["validate-plan", "--plan", "tests/fixtures/competition/plan-missing-freshness.yaml"], 1, "", "missing global_runtime.freshness_rule"),
        (["validate-case", "--case", "tests/fixtures/competition/public-godot-001-controlled.json"], 0, "VALID case_id=public-godot-001-controlled", ""),
        (["validate-case", "--case", "tests/fixtures/competition/case-safe-with-patch.json"], 1, "", "SAFE requires patch=null"),
        (["validate-provenance", "--manifest", "tests/fixtures/competition/provenance-valid.json"], 0, "VALID provenance", ""),
        (["validate-provenance", "--manifest", "tests/fixtures/competition/provenance-revision-mismatch.json"], 1, "", "source revision mismatch"),
        (["verify-receipt", "--receipt", "tests/fixtures/competition/receipt-pass-valid.json"], 0, "VALID receipt decision=PASS", ""),
        (["verify-receipt", "--receipt", "tests/fixtures/competition/receipt-frame-hash-mismatch.json"], 1, "", "frame hash mismatch"),
        (["prepare-league", "--run", "tests/fixtures/competition/league-valid"], 0, "SEALED lanes=3", ""),
        (["prepare-league", "--run", "tests/fixtures/competition/league-identity-leak"], 1, "", "identity leakage"),
        (["aggregate-league", "--run", "tests/fixtures/competition/league-valid", "--results", "tests/fixtures/competition/league-valid/results.json"], 0, "SEALED aggregate", ""),
        (["reveal-league", "--run", "tests/fixtures/competition/league-valid", "--aggregate", "tests/fixtures/competition/league-valid/aggregate.json"], 0, "REVEALED after seal", ""),
        (["claim-gate", "--manifest", "tests/fixtures/competition/claim-ineligible.json"], 1, "NOT_CLAIMABLE", ""),
        (["claim-gate", "--manifest", "tests/fixtures/competition/claim-unfrozen.json"], 1, "NOT_CLAIMABLE unfrozen input", ""),
    ],
)
def test_declared_matrix_row(args: list[str], exit_code: int, stdout: str, stderr: str) -> None:
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code
    assert result.stdout == (stdout + "\n" if stdout else "")
    assert result.stderr == (stderr + "\n" if stderr else "")


def test_claim_gate_rejects_a_forged_passing_scalar_summary(tmp_path: Path) -> None:
    """L9 must not mint an external-win claim from caller supplied numbers."""
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({
        "frozen": True,
        "distinct_public_repositories": 3,
        "independent_cases": 9,
        "repeats_per_case": 3,
        "inconclusive_cases": 0,
        "false_negative_delta": 0,
        "f1_mean_delta": 1.0,
        "all_direct_baselines_scoreable": True,
        "bootstrap": {
            "seed": 20260801,
            "resamples": 10000,
            "lower_confidence_bound": 0.1,
            "artifact_sha256": "a" * 64,
        },
    }), encoding="utf-8")
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", "claim-gate", "--manifest", str(forged)],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert result.stdout == "NOT_CLAIMABLE\n"


def test_run_matrix_executes_every_declared_row_with_exact_exit_and_output() -> None:
    """The spec-driven matrix is authoritative; this pytest only supplements it."""
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", "run-matrix",
         "--plan", "specs/flashpatch-competition-evidence-v1.yaml"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "MATRIX PASS rows=14" in result.stdout
    for validator in ("plan_schema", "case_schema", "provenance", "receipt_hash", "blind_league", "claim_gate"):
        assert f"ROW {validator}.pass" in result.stdout
        assert f"ROW {validator}.fail" in result.stdout
    assert "ROW blind_league.follow_up[1]" in result.stdout
    assert "ROW blind_league.follow_up[2]" in result.stdout


def test_run_matrix_fails_closed_on_declared_exit_mismatch(tmp_path: Path) -> None:
    plan_text = (ROOT / "specs/flashpatch-competition-evidence-v1.yaml").read_text(encoding="utf-8")
    broken = plan_text.replace(
        "validate-plan --plan specs/flashpatch-competition-evidence-v1.yaml, exit: 0",
        "validate-plan --plan specs/flashpatch-competition-evidence-v1.yaml, exit: 1",
        1,
    )
    broken_plan = tmp_path / "plan-exit-mismatch.yaml"
    broken_plan.write_text(broken, encoding="utf-8")
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", "run-matrix", "--plan", str(broken_plan)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "MATRIX FAIL row=plan_schema.pass exit=0 declared=1" in result.stderr


def test_run_matrix_rejects_untrusted_commands(tmp_path: Path) -> None:
    """Plan text stays data: only flashpatch.competition commands may execute."""
    plan_text = (ROOT / "specs/flashpatch-competition-evidence-v1.yaml").read_text(encoding="utf-8")
    hostile = plan_text.replace(
        "python -m flashpatch.competition validate-plan --plan specs/flashpatch-competition-evidence-v1.yaml",
        "sh -c evil",
        1,
    )
    hostile_plan = tmp_path / "plan-untrusted-command.yaml"
    hostile_plan.write_text(hostile, encoding="utf-8")
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", "run-matrix", "--plan", str(hostile_plan)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "untrusted command rejected" in result.stderr


def test_receipt_patch_decision_contract(tmp_path: Path) -> None:
    """SAFE means no patch, while PASS must name a concrete patch."""
    valid_pass = json.loads(
        (ROOT / "tests/fixtures/competition/receipt-pass-valid.json").read_text(encoding="utf-8")
    )

    safe = {**valid_pass, "decision": "SAFE", "patch": None, "failure_reason": None}
    safe_path = tmp_path / "safe.json"
    safe_path.write_text(json.dumps(safe), encoding="utf-8")
    assert verify_receipt(safe_path) == "VALID receipt decision=SAFE"

    invalid_safe = {**safe, "patch": {"parameter": "flash_intensity"}}
    invalid_safe_path = tmp_path / "invalid-safe.json"
    invalid_safe_path.write_text(json.dumps(invalid_safe), encoding="utf-8")
    with pytest.raises(ContractError, match="SAFE requires patch=null"):
        verify_receipt(invalid_safe_path)

    invalid_pass = {**valid_pass, "patch": None}
    invalid_pass_path = tmp_path / "invalid-pass.json"
    invalid_pass_path.write_text(json.dumps(invalid_pass), encoding="utf-8")
    with pytest.raises(ContractError, match="PASS requires patch object"):
        verify_receipt(invalid_pass_path)


def test_l6_placeholder_cannot_be_presented_as_renderer_evidence(tmp_path: Path) -> None:
    case_path = ROOT / "evidence/competition/cases/public-godot-001-controlled.json"
    with pytest.raises(ContractError, match="renderer artifacts unavailable"):
        verify_chain(case_path)

    case = json.loads(case_path.read_text(encoding="utf-8"))
    assert case["controlled_mutation"] is True
    assert case["source"]["repository_url"] == "https://github.com/godotengine/godot-demo-projects"
    assert case["source"]["source_revision"] == "52e3004"
    assert case["source"]["license"] == "MIT"
    assert case["source"]["project_path"] == "2d/pong"
    assert case["original_upstream"]["decision"] in {"SAFE", "INCONCLUSIVE"}
    assert case["original_upstream"]["upstream_defect"] is False

    copied_root = tmp_path / "competition"
    shutil.copytree(ROOT / "evidence/competition", copied_root)
    copied_case = copied_root / "cases/public-godot-001-controlled.json"

    def _rewrite(mutate) -> Path:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        mutate(payload)
        copied_case.write_text(json.dumps(payload), encoding="utf-8")
        return copied_case

    def _set_defect(payload: dict) -> None:
        payload["original_upstream"]["upstream_defect"] = True

    with pytest.raises(ContractError, match="must not be an upstream defect"):
        verify_chain(_rewrite(_set_defect))

    def _set_upstream_fail(payload: dict) -> None:
        payload["original_upstream"]["decision"] = "FAIL"

    with pytest.raises(ContractError, match="original upstream must be SAFE or INCONCLUSIVE"):
        verify_chain(_rewrite(_set_upstream_fail))

    def _drift_revision(payload: dict) -> None:
        payload["source"]["source_revision"] = "deadbee"

    with pytest.raises(ContractError, match="controlled source source_revision mismatch"):
        verify_chain(_rewrite(_drift_revision))

    def _drop_license(payload: dict) -> None:
        payload["source"]["license"] = "GPL-3.0"

    with pytest.raises(ContractError, match="controlled source license mismatch"):
        verify_chain(_rewrite(_drop_license))

    def _unlabel_mutation(payload: dict) -> None:
        payload["controlled_mutation"] = False

    with pytest.raises(ContractError, match="L6 requires controlled_mutation=true"):
        verify_chain(_rewrite(_unlabel_mutation))

    def _upstream_safe(payload: dict) -> None:
        payload["original_upstream"] = {
            "decision": "SAFE",
            "upstream_defect": False,
            "reason": None,
        }

    with pytest.raises(ContractError, match="renderer artifacts unavailable"):
        verify_chain(_rewrite(_upstream_safe))


def test_hash_bound_receipt_recomputes_declared_artifact_bytes(tmp_path: Path) -> None:
    receipt = json.loads((ROOT / "tests/fixtures/competition/receipt-pass-valid.json").read_text(encoding="utf-8"))
    frame = tmp_path / "frame.rgb"
    frame.write_bytes(b"factual-frame")
    import hashlib
    receipt["factual_frame_sha256"] = hashlib.sha256(frame.read_bytes()).hexdigest()
    receipt["artifacts"] = {"factual_frame_sha256": "frame.rgb"}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_receipt(path) == "VALID receipt decision=PASS"
    frame.write_bytes(b"tampered-frame")
    with pytest.raises(ContractError, match="frame hash mismatch"):
        verify_receipt(path)


def test_l7_missing_evidence_is_not_claimable() -> None:
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.competition", "compare-detectors", "--manifest", "evidence/competition/detector-comparison-manifest.json"],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert result.stdout == "NOT_CLAIMABLE\n"
    assert result.stderr == ""


def test_checkpoints_are_atomic_hash_bound_and_only_completed_runs_are_reusable(tmp_path: Path) -> None:
    """All L5 gate checkpoints persist exact hashes across a fresh read."""
    immutable_input = {"frozen": "input"}
    command = ["python", "-m", "flashpatch.competition", "gate"]
    environment = {"python": "3.11", "platform": "linux"}
    receipt = {"status": "PASS", "artifact": "receipt.json"}

    for leaf in CHECKPOINT_LEAVES:
        path = write_checkpoint(
            tmp_path, leaf, immutable_input=immutable_input, command=command,
            environment=environment, receipt=receipt,
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["leaf"] == leaf
        assert saved["status"] == "COMPLETED"
        assert all(saved[f"{name}_sha256"] for name in ("input", "command", "environment", "receipt"))
        assert checkpoint_is_reusable(
            tmp_path, leaf, immutable_input=immutable_input, command=command,
            environment=environment, receipt=receipt,
        )
        for changed_input, changed_command, changed_environment, changed_receipt in (
            ({"frozen": "changed"}, command, environment, receipt),
            (immutable_input, [*command, "--changed"], environment, receipt),
            (immutable_input, command, {**environment, "platform": "changed"}, receipt),
            (immutable_input, command, environment, {**receipt, "artifact": "changed.json"}),
        ):
            assert not checkpoint_is_reusable(
                tmp_path,
                leaf,
                immutable_input=changed_input,
                command=changed_command,
                environment=changed_environment,
                receipt=changed_receipt,
            )

    write_checkpoint(
        tmp_path, "L9", immutable_input=immutable_input, command=command, environment=environment,
        receipt={"status": "INCONCLUSIVE", "exclusion_reason": "timeout"}, status="INCONCLUSIVE",
        exclusion_reason="timeout",
    )
    assert not checkpoint_is_reusable(
        tmp_path, "L9", immutable_input=immutable_input, command=command, environment=environment,
        receipt={"status": "INCONCLUSIVE", "exclusion_reason": "timeout"},
    )
    assert not list(tmp_path.glob(".L9.json.*"))


def test_checkpoints_survive_interruption_and_resume_is_hash_verified(tmp_path: Path) -> None:
    """A torn write or an interrupted run never resumes; only identical hashes do."""
    immutable_input = {"frozen": "input"}
    command = ["python", "-m", "flashpatch.competition", "gate"]
    environment = {"python": "3.11", "platform": "linux"}
    receipt = {"status": "PASS", "artifact": "receipt.json"}

    for leaf in sorted(CHECKPOINT_LEAVES):
        # Simulate an interruption that tore the checkpoint file mid-publish.
        torn = tmp_path / ("preflight.json" if leaf == "P0" else f"{leaf}.json")
        torn.parent.mkdir(parents=True, exist_ok=True)
        torn.write_text('{"schema": "flashpatch-competition-che', encoding="utf-8")
        assert not checkpoint_is_reusable(
            tmp_path, leaf, immutable_input=immutable_input, command=command,
            environment=environment, receipt=receipt,
        )

        # A completed checkpoint written after the interruption survives a
        # fresh process read and is reused only under identical hashes.
        identity = {
            "input": f"frozen-input-{leaf}",
            "command": f"python -m flashpatch.competition gate --leaf {leaf}",
            "environment": "python-3.11-linux",
            "receipt": f"receipt-{leaf}.json",
        }
        write_checkpoint(
            tmp_path, leaf, immutable_input=identity["input"], command=identity["command"],
            environment=identity["environment"], receipt=identity["receipt"],
        )
        run_env = os.environ | {"PYTHONPATH": str(ROOT / "src")}
        base_args = [
            sys.executable, "-m", "flashpatch.competition", "resume",
            "--leaf", leaf, "--checkpoints", str(tmp_path),
        ]
        identical_args = [
            "--input", identity["input"], "--command", identity["command"],
            "--environment", identity["environment"], "--receipt", identity["receipt"],
        ]
        resumed = subprocess.run(
            [*base_args, *identical_args], cwd=ROOT, env=run_env,
            capture_output=True, text=True, check=False,
        )
        assert resumed.returncode == 0
        assert f"REUSED checkpoint leaf={leaf}" in resumed.stdout

        drifted_args = [
            "--input", identity["input"] + "-drift", "--command", identity["command"],
            "--environment", identity["environment"], "--receipt", identity["receipt"],
        ]
        mismatched = subprocess.run(
            [*base_args, *drifted_args], cwd=ROOT, env=run_env,
            capture_output=True, text=True, check=False,
        )
        assert mismatched.returncode == 1
        assert f"INCONCLUSIVE checkpoint leaf={leaf} not reusable" in mismatched.stdout


def test_preflight_writes_hash_bound_p0_checkpoint_and_rejects_stale_revision(tmp_path: Path) -> None:
    plan = ROOT / "specs/flashpatch-competition-evidence-v1.yaml"
    assert preflight(plan, "56c0ae6", tmp_path) == "VALID preflight"
    assert checkpoint_is_reusable(
        tmp_path, "P0", immutable_input=plan.read_text(encoding="utf-8"),
        command={"plan": str(plan), "expected_commit": "56c0ae6"},
        environment={"python": sys.version, "platform": sys.platform},
        receipt={"status": "PASS", "revision": "56c0ae6f16fce2ea4b2fd7827a942d2444aa1920"},
    )
    with pytest.raises(ContractError, match="exclusion_reason=stale_plan"):
        preflight(plan, "deadbeef", tmp_path)
    assert not checkpoint_is_reusable(
        tmp_path, "P0", immutable_input=plan.read_text(encoding="utf-8"),
        command={"plan": str(plan), "expected_commit": "deadbeef"},
        environment={"python": sys.version, "platform": sys.platform},
        receipt={"status": "INCONCLUSIVE", "exclusion_reason": "stale_plan", "revision": "56c0ae6f16fce2ea4b2fd7827a942d2444aa1920"},
    )
