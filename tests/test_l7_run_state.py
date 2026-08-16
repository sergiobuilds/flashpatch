from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashpatch.l7_run_state import (
    L7RunStateError,
    execution_contract,
    discard_incomplete_case,
    finalize_run,
    initialize_run,
    record_phase,
    record_completed_case,
)


def _contract(project: Path) -> dict[str, object]:
    return execution_contract(
        project,
        code_paths=["src/runner.py", "scripts/runner.py"],
        comparator_census_sha256="a" * 64,
        environment={"container": "sha256:test", "python": "3.11"},
    )


def _project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/runner.py").write_text("detector = 1\n", encoding="utf-8")
    (tmp_path / "scripts/runner.py").write_text("runner = 1\n", encoding="utf-8")
    return tmp_path


def test_document_edits_do_not_change_execution_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "docs").mkdir()
    document = project / "docs/MASTER-MAP.md"
    document.write_text("first narrative\n", encoding="utf-8")
    first = _contract(project)
    document.write_text("second narrative\n", encoding="utf-8")
    second = _contract(project)
    assert first["contract_sha256"] == second["contract_sha256"]


def test_runner_change_changes_execution_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = _contract(project)
    (project / "scripts/runner.py").write_text("runner = 2\n", encoding="utf-8")
    second = _contract(project)
    assert first["contract_sha256"] != second["contract_sha256"]


def test_run_requires_empty_root_and_exact_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    assert initialize_run(root, contract)["status"] == "RUNNING"
    with pytest.raises(L7RunStateError, match="empty"):
        initialize_run(root, contract)
    changed = dict(contract)
    changed["contract_sha256"] = "b" * 64
    with pytest.raises(L7RunStateError, match="digest mismatch"):
        record_completed_case(root, changed, case_id="case-a", receipt_path=root / "case-a.json")


def test_completed_case_is_receipt_bound_and_finalization_reopens_it(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    initialize_run(root, contract)
    receipt = root / "cases/case-a/complete-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    state = record_completed_case(root, contract, case_id="case-a", receipt_path=receipt)
    assert state["completed_cases"]["case-a"]["path"] == "cases/case-a/complete-receipt.json"
    assert finalize_run(root, contract, expected_case_ids=["case-a"])["status"] == "COMPLETE"


def test_partial_or_modified_case_cannot_finish(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    initialize_run(root, contract)
    receipt = root / "cases/case-a/complete-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("one", encoding="utf-8")
    record_completed_case(root, contract, case_id="case-a", receipt_path=receipt)
    with pytest.raises(L7RunStateError, match="all expected"):
        finalize_run(root, contract, expected_case_ids=["case-a", "case-b"])
    receipt.write_text("changed", encoding="utf-8")
    with pytest.raises(L7RunStateError, match="changed"):
        finalize_run(root, contract, expected_case_ids=["case-a"])


def test_phase_events_are_contract_bound_and_cannot_regress(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    initialize_run(root, contract)
    state = record_phase(root, contract, phase="prepare", status="STARTED", case_id="case-a")
    assert state["phases"]["case-a:prepare"] == "STARTED"
    record_phase(root, contract, phase="prepare", status="COMPLETE", case_id="case-a")
    with pytest.raises(L7RunStateError, match="cannot regress"):
        record_phase(root, contract, phase="prepare", status="FAILED", case_id="case-a")
    event = json.loads((root / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["status"] == "COMPLETE"


def test_phase_events_reject_a_different_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    initialize_run(root, contract)
    incompatible = dict(contract)
    incompatible["environment"] = {"python": "different"}
    incompatible["contract_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in incompatible.items() if key != "contract_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(L7RunStateError, match="execution contract"):
        record_phase(root, incompatible, phase="prepare", status="STARTED")


def test_initialize_allows_only_the_executor_guard_before_state(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    root.mkdir(parents=True)
    (root / ".executor.lock").touch()
    initialize_run(root, _contract(project))
    occupied = project / "artifacts/runs/run-b"
    occupied.mkdir()
    (occupied / "unexpected.txt").write_text("not a lock", encoding="utf-8")
    with pytest.raises(L7RunStateError, match="must be empty"):
        initialize_run(occupied, _contract(project))


def test_incomplete_case_can_be_discarded_but_checkpointed_case_cannot(tmp_path: Path) -> None:
    project = _project(tmp_path)
    root = project / "artifacts/runs/run-a"
    contract = _contract(project)
    initialize_run(root, contract)
    record_phase(root, contract, phase="execute", status="STARTED", case_id="case-a")
    state = discard_incomplete_case(root, contract, case_id="case-a")
    assert state["phases"] == {}
    event = json.loads((root / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["phase"] == "discard_incomplete_case"
    receipt = root / "cases/case-a/complete-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("complete", encoding="utf-8")
    record_completed_case(root, contract, case_id="case-a", receipt_path=receipt)
    with pytest.raises(L7RunStateError, match="checkpointed"):
        discard_incomplete_case(root, contract, case_id="case-a")
