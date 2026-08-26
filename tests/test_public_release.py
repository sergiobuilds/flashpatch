from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
BOUNDARY_CHECK = ROOT / "scripts" / "check_public_release.py"
SBOM = ROOT / "sbom" / "flashpatch.cdx.json"
UNITY_PACKAGES_LOCK = "flashpatch/_unity_packages_lock_2022_3_8f1.json"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_built_wheel_runs_current_safety_demo_outside_source_checkout(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    installed = tmp_path / "installed"
    wheelhouse.mkdir()
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
    )
    wheel = next(wheelhouse.glob("flashpatch-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
        assert UNITY_PACKAGES_LOCK in names
        assert "License-Expression: Apache-2.0" in metadata
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = tmp_path / "safety-demo"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.cli",
            "safety-demo",
            "--json",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(installed)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["results"] == {
        "fail": "FAIL",
        "inconclusive": "INCONCLUSIVE",
        "pass": "PASS",
        "safe": "SAFE",
    }

    for command in (
        [
            "verify-engine-proof",
            str(tmp_path / "missing-receipt.json"),
            "--trust-policy",
            str(tmp_path / "missing-policy.json"),
            "--expected-trust-policy-sha256",
            "0" * 64,
        ],
        [
            "unity-renderer-run",
            "--factual-template",
            str(tmp_path / "missing-factual"),
            "--counterfactual-template",
            str(tmp_path / "missing-counterfactual"),
            "--factual-manifest",
            str(tmp_path / "missing-factual.json"),
            "--counterfactual-manifest",
            str(tmp_path / "missing-counterfactual.json"),
            "--runtime-output",
            str(tmp_path / "runtime"),
            "--entitlement",
            str(tmp_path / "missing-entitlement.xml"),
            "--vulkan-loader",
            str(tmp_path / "missing-vulkan.so"),
            "--gpu-index",
            "0",
            "--display",
            ":99",
        ],
    ):
        loaded = subprocess.run(
            [sys.executable, "-m", "flashpatch.cli", *command],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(installed)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert loaded.returncode == 2
        assert json.loads(loaded.stdout).get("verdict") == "INCONCLUSIVE"
        assert "_unity_packages_lock_2022_3_8f1.json" not in loaded.stderr


def test_checked_in_sbom_matches_declared_runtime_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sbom = json.loads(SBOM.read_text(encoding="utf-8"))
    declared = {requirement.split(">=")[0] for requirement in project["project"]["dependencies"]}

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["metadata"]["component"]["name"] == project["project"]["name"]
    assert sbom["metadata"]["component"]["version"] == project["project"]["version"]
    assert sbom["metadata"]["component"]["licenses"] == [
        {"license": {"id": "Apache-2.0"}}
    ]
    assert {component["name"] for component in sbom["components"]} == declared
    recorded_requirements = {
        property_["value"]
        for component in sbom["components"]
        for property_ in component["properties"]
        if property_["name"] == "flashpatch:declared-requirement"
    }
    assert recorded_requirements == set(project["project"]["dependencies"])


def test_public_boundary_checker_accepts_clean_projection_and_rejects_secrets(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _git("init", "-q", cwd=fixture)
    (fixture / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    fixture_sbom = fixture / "sbom" / "flashpatch.cdx.json"
    fixture_sbom.parent.mkdir()
    fixture_sbom.write_bytes(SBOM.read_bytes())
    (fixture / "README.md").write_text("Public FlashPatch projection.\n", encoding="utf-8")
    _git("add", ".", cwd=fixture)

    accepted = subprocess.run(
        [sys.executable, str(BOUNDARY_CHECK), "--root", str(fixture)],
        cwd=fixture,
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0
    assert "public release audit passed" in accepted.stdout

    (fixture / "README.md").write_text(
        "developer checkout: /" + "home/alice/projects/private-product\n"
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    evidence = fixture / "evidence"
    evidence.mkdir()
    (evidence / "internal.json").write_text("{}\n", encoding="utf-8")
    private = fixture / "private"
    private.mkdir()
    (private / "identity-map.json").write_text("{}\n", encoding="utf-8")
    plans = fixture / "docs" / "plans"
    plans.mkdir(parents=True)
    (plans / "internal.md").write_text("internal plan\n", encoding="utf-8")
    _git("add", ".", cwd=fixture)

    rejected = subprocess.run(
        [sys.executable, str(BOUNDARY_CHECK), "--root", str(fixture)],
        cwd=fixture,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert "forbidden tracked path: evidence/internal.json" in rejected.stdout
    assert "forbidden tracked path: private/identity-map.json" in rejected.stdout
    assert "forbidden tracked path: docs/plans/internal.md" in rejected.stdout
    assert "absolute user path: README.md" in rejected.stdout
    assert "private key material: README.md" in rejected.stdout


def test_public_boundary_checker_rejects_missing_sbom(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _git("init", "-q", cwd=fixture)
    (fixture / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    (fixture / "README.md").write_text("Public FlashPatch projection.\n", encoding="utf-8")
    _git("add", ".", cwd=fixture)

    rejected = subprocess.run(
        [sys.executable, str(BOUNDARY_CHECK), "--root", str(fixture)],
        cwd=fixture,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 1
    assert "missing release metadata: sbom/flashpatch.cdx.json" in rejected.stdout
