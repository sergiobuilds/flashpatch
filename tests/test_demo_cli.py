from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path_factory.mktemp("installed-flashpatch")
    wheelhouse = root / "wheelhouse"
    prefix = root / "prefix"
    wheelhouse.mkdir()
    build_environment = os.environ.copy()
    build_environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=build_environment,
    )
    wheel = next(wheelhouse.glob("flashpatch-*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-installed",
            "--prefix",
            str(prefix),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=build_environment,
    )
    package = next(prefix.rglob("site-packages/flashpatch/__init__.py"))
    executable = prefix / "bin" / "flashpatch"
    assert executable.is_file()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package.parents[1])
    outside_checkout = root / "outside-checkout"
    outside_checkout.mkdir()
    return executable, environment, outside_checkout


def test_installed_safety_demo_is_user_readable_and_source_independent(
    installed_cli: tuple[Path, dict[str, str], Path],
) -> None:
    executable, environment, cwd = installed_cli
    output = cwd / "demo-output"

    completed = subprocess.run(
        [str(executable), "safety-demo", "--output", str(output)],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "FlashPatch safety demo" in completed.stdout
    assert "INPUT" in completed.stdout
    assert "LOCALIZE" in completed.stdout
    assert "PATCH" in completed.stdout
    assert "REVALIDATE" in completed.stdout
    assert "PASS=PASS SAFE=SAFE FAIL=FAIL INCONCLUSIVE=INCONCLUSIVE" in completed.stdout
    report = json.loads((output / "demo-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "flashpatch-safety-demo-v2"
    assert report["fixture"]["evidence_scope"] == "deterministic_contract_fixture"
    assert report["flow"]["hazard_localization"] == {
        "action_frame": 2,
        "observation_index": 2,
        "risk": 1.0,
        "threshold": 1.0,
    }
    assert report["flow"]["allowed_patch"]["changed_source_assignments"] == 1
    assert report["flow"]["same_trace_revalidation"] == {
        "factual_max_risk": 1.0,
        "gameplay_state_preserved": True,
        "patched_max_risk": 0.0,
        "timing_preserved": True,
    }
    assert report["results"] == {
        "fail": "FAIL",
        "inconclusive": "INCONCLUSIVE",
        "pass": "PASS",
        "safe": "SAFE",
    }
    reversals = report["failure_reversals"]
    assert reversals["schema"] == "flashpatch-failure-reversal-matrix-v1"
    assert {
        name: (case["expected_verdict"], case["actual_verdict"], case["reason"])
        for name, case in reversals["cases"].items()
    } == {
        "already_safe": ("SAFE", "SAFE", "no_hazard_in_declared_trace"),
        "gameplay_state_drift": (
            "FAIL",
            "FAIL",
            "patch_broke_declared_gameplay_invariants",
        ),
        "missing_renderer_timestamps": (
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            "frame artifact must contain valid frames and timestamps: "
            "renderer artifact must contain only frames.npy and timestamps.npy",
        ),
        "multiple_parameters_required": (
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            "multiple_parameters_required_no_single_patch_authorized",
        ),
        "residual_risk": (
            "FAIL",
            "FAIL",
            "hazard_persists_after_all_declared_candidates",
        ),
    }
    assert json.loads((output / "failure-matrix.json").read_text(encoding="utf-8")) == reversals
    for name in ("gameplay-drift", "missing-timestamps", "multi-parameter"):
        assert (output / f"{name}-receipt.json").is_file()
    for verdict in ("pass", "safe", "fail", "inconclusive"):
        receipt = json.loads((output / f"{verdict}-receipt.json").read_text(encoding="utf-8"))
        assert receipt["verdict"] == report["results"][verdict]
        assert receipt["demonstration"]["evidence_scope"] == "deterministic_contract_fixture"
        assert receipt["receipt_sha256"].startswith("sha256:")


def test_installed_safety_demo_json_output_matches_report(
    installed_cli: tuple[Path, dict[str, str], Path],
) -> None:
    executable, environment, cwd = installed_cli
    output = cwd / "json-demo-output"

    completed = subprocess.run(
        [str(executable), "safety-demo", "--json", "--output", str(output)],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == json.loads(
        (output / "demo-report.json").read_text(encoding="utf-8")
    )


def test_installed_compile_fails_closed_with_receipt(
    installed_cli: tuple[Path, dict[str, str], Path],
) -> None:
    executable, environment, cwd = installed_cli
    receipt_path = cwd / "compile" / "receipt.json"

    completed = subprocess.run(
        [
            str(executable),
            "compile",
            str(cwd / "missing-project"),
            str(cwd / "missing-contract.json"),
            "--workspace",
            str(cwd / "compile" / "workspace"),
            "--receipt",
            str(receipt_path),
        ],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["verdict"] == "INCONCLUSIVE"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["receipt_sha256"].startswith("sha256:")
