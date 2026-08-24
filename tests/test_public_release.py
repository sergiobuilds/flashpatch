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


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_built_wheel_runs_safety_demo_outside_source_checkout(tmp_path: Path) -> None:
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
        "pass": "PASS",
        "safe": "SAFE",
        "inconclusive": "INCONCLUSIVE",
    }


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


def test_public_boundary_checker_rejects_private_paths_and_secrets(tmp_path: Path) -> None:
    assert BOUNDARY_CHECK.is_file()
    subprocess.run(
        [sys.executable, str(BOUNDARY_CHECK), "--root", str(ROOT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _git("init", "-q", cwd=fixture)
    (fixture / "README.md").write_text(
        "developer checkout: /" + "home/alice/projects/private-product\n"
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    evidence = fixture / "evidence"
    evidence.mkdir()
    (evidence / "internal.json").write_text("{}\n", encoding="utf-8")
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
    assert "absolute user path: README.md" in rejected.stdout
    assert "private key material: README.md" in rejected.stdout
