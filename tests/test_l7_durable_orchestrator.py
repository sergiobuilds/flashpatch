from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from flashpatch.l7_run_state import execution_contract, initialize_run, record_phase

def _orchestrator():
    path = Path(__file__).resolve().parents[1] / "scripts/run_l7_durable.py"
    spec = importlib.util.spec_from_file_location("l7_durable_orchestrator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts/run_l7_score_pipeline.py"
    spec = importlib.util.spec_from_file_location("l7_score_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_contract_explicitly_binds_detector_and_threshold_helpers(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline()
    receipt = tmp_path / "census-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")

    contract = pipeline.durable_execution_contract(receipt)
    code_paths = {entry["path"] for entry in contract["code"]}

    assert "src/flashpatch/core.py" in code_paths
    assert "src/flashpatch/standards.py" in code_paths


def test_docker_executor_forwards_https_provenance(monkeypatch, tmp_path: Path) -> None:
    runner = _orchestrator()
    captured: dict[str, object] = {}

    def record(_log: Path, argv: list[str], *, env: dict[str, str]) -> None:
        captured["argv"] = argv

    monkeypatch.setattr(runner, "_run", record)
    case_root = tmp_path / "case"
    (case_root / "execution").mkdir(parents=True)
    runner._docker_execute(
        tmp_path / "log.jsonl",
        {
            "FLASHPATCH_CENSUS_ROOT": "/tmp/census",
            "FLASHPATCH_REMOTE_URL": "https://github.com/sergiobuilds/flashpatch.git",
            "FLASHPATCH_REMOTE_VERIFICATION_RECEIPT": "/tmp/census/flashpatch-remote-verification.json",
        },
        "image:test",
        case_root,
    )
    argv = captured["argv"]
    assert "FLASHPATCH_REMOTE_URL=https://github.com/sergiobuilds/flashpatch.git" in argv
    assert "FLASHPATCH_REMOTE_VERIFICATION_RECEIPT=/tmp/census/flashpatch-remote-verification.json" in argv
    assert "FLASHPATCH_GPU_ISOLATION=DOCKER_EMPTY_DEV" in argv
    assert "--privileged" not in argv
    assert "--security-opt=no-new-privileges" in argv
    project_mount = f"{runner.PROJECT_ROOT}:{runner.PROJECT_ROOT}:ro"
    assert project_mount in argv
    assert f"{case_root / 'execution'}:{case_root / 'execution'}" in argv
    assert f"FLASHPATCH_KAYA_REPLAY_SCRATCH={case_root / 'execution' / 'kaya-replay-scratch'}" in argv


def test_external_transport_bundle_uses_the_actual_readonly_checkout_path() -> None:
    runner = _orchestrator()
    source = (runner.PROJECT_ROOT / "scripts/run_l7_score_pipeline.py").read_text(encoding="utf-8")
    assert "workspace_root=str(PROJECT_ROOT)" in source


def test_assembly_phase_closes_before_finalizing_the_run(monkeypatch, tmp_path: Path) -> None:
    runner = _orchestrator()
    calls: list[tuple[str, list[str]]] = []

    def record_run(_log: Path, argv: list[str], *, env: dict[str, str]) -> None:
        calls.append(("run", argv))

    def record_pipeline(_log: Path, _env: dict[str, str], *args: str) -> None:
        calls.append(("pipeline", list(args)))

    monkeypatch.setattr(runner, "_run", record_run)
    monkeypatch.setattr(runner, "_pipeline", record_pipeline)

    bundle = tmp_path / "run" / "score-bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("verified bundle\n", encoding="utf-8")
    canonical_bundle = tmp_path / "artifacts" / "l7" / "score_bundle.json"
    current_validation = tmp_path / "artifacts" / "l7" / "current-validation.json"
    runner.PROJECT_ROOT = tmp_path
    (tmp_path / "artifacts" / "l7").mkdir(parents=True, exist_ok=True)
    census = tmp_path / "artifacts" / "l7" / "census-receipt.json"
    census.write_text("{}\n", encoding="utf-8")
    run_root = tmp_path / "artifacts" / "l7" / "runs" / "run-a"
    run_root.mkdir(parents=True)
    (run_root / "run-state.json").write_text(json.dumps({"status": "COMPLETE", "phases": {"run:assemble": "COMPLETE"}, "contract_sha256": "a" * 64}), encoding="utf-8")
    (run_root / "execution-contract.json").write_text(json.dumps({"contract_sha256": "a" * 64}), encoding="utf-8")

    runner._verify_and_finalize_assembly(
        tmp_path / "runner.log",
        {},
        run_root=run_root,
        census_receipt=census,
        bundle=bundle,
        canonical_bundle=canonical_bundle,
        current_validation=current_validation,
    )

    assert calls[0] == (
        "run",
        [
            runner.sys.executable,
            "-m",
            "flashpatch.l7_verify",
            "--bundle",
            str(bundle),
            "--bootstrap",
            "10000",
        ],
    )
    assert calls[1] == (
        "pipeline",
        [
            "mark-phase",
            "--run-root",
            str(run_root),
            "--census-receipt",
            str(census),
            "--phase",
            "assemble",
            "--status",
            "COMPLETE",
        ],
    )
    assert calls[2] == (
        "pipeline",
        [
            "finalize-run",
            "--run-root",
            str(run_root),
            "--census-receipt",
            str(census),
        ],
    )
    assert canonical_bundle.read_text(encoding="utf-8") == "verified bundle\n"
    pointer = json.loads(current_validation.read_text(encoding="utf-8"))
    assert pointer["run_root"] == "runs/run-a"
    assert pointer["score_bundle"]["sha256"]


def test_canonical_publication_rejects_a_nonfinal_run(tmp_path: Path) -> None:
    runner = _orchestrator()
    runner.PROJECT_ROOT = tmp_path
    l7 = tmp_path / "artifacts" / "l7"
    run_root = l7 / "runs" / "run-a"
    run_root.mkdir(parents=True)
    bundle = run_root / "score-bundle.json"
    bundle.write_text("bundle\n", encoding="utf-8")
    census = l7 / "census" / "census-receipt.json"
    census.parent.mkdir(parents=True)
    census.write_text("{}\n", encoding="utf-8")
    (run_root / "run-state.json").write_text(
        json.dumps({"status": "RUNNING", "phases": {"run:assemble": "COMPLETE"}, "contract_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (run_root / "execution-contract.json").write_text(
        json.dumps({"contract_sha256": "a" * 64}), encoding="utf-8"
    )
    import pytest

    with pytest.raises(RuntimeError, match="finalized run"):
        runner._publish_score_bundle(
            bundle,
            run_root=run_root,
            census_receipt=census,
            destination=l7 / "score_bundle.json",
            current_validation=l7 / "current-validation.json",
        )


def test_resume_rederives_contract_before_removing_an_incomplete_case(
    monkeypatch, tmp_path: Path
) -> None:
    """The destructive resume cleanup is unreachable until the contract gate passes."""
    runner = _orchestrator()
    run_root = tmp_path / "runs" / "resume-a"
    census_root = tmp_path / "censuses" / "resume-a"
    census_root.mkdir(parents=True)
    (census_root / "census-receipt.json").write_text("{}\n", encoding="utf-8")

    contract = execution_contract(
        Path(__file__).resolve().parents[1],
        code_paths=["src/flashpatch/core.py"],
        comparator_census_sha256="a" * 64,
        environment={"image": "sha256:test"},
    )
    initialize_run(run_root, contract)
    (run_root / "cases" / "case-a").mkdir(parents=True)
    (run_root / "cases" / "case-a" / "partial.txt").write_text(
        "incomplete", encoding="utf-8"
    )
    record_phase(run_root, contract, phase="prepare", status="STARTED", case_id="case-a")

    order: list[str] = []
    original_rmtree = runner.shutil.rmtree

    def assert_contract(_root: Path, _receipt: Path) -> None:
        order.append("contract")

    def record_rmtree(path: Path) -> None:
        order.append("rmtree")
        original_rmtree(path)

    def noop_pipeline(_log: Path, _env: dict[str, str], *_args: str) -> None:
        return None

    monkeypatch.setattr(runner, "_assert_resume_contract", assert_contract)
    monkeypatch.setattr(runner.shutil, "rmtree", record_rmtree)
    monkeypatch.setattr(runner, "_write_remote_verification_receipt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_image_digest", lambda *_args, **_kwargs: "sha256:test")
    monkeypatch.setattr(runner, "_pipeline", noop_pipeline)
    monkeypatch.setattr(runner, "_docker_external_profile", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_docker_execute", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_verify_and_finalize_assembly", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "case-a fixture-a\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_l7_durable.py",
            "--run-root",
            str(run_root),
            "--census-root",
            str(census_root),
            "--resume",
        ],
    )

    assert runner.main() == 0
    assert order[:2] == ["contract", "rmtree"]
    assert not (run_root / "cases" / "case-a" / "partial.txt").exists()
