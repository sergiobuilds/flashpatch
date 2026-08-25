from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
UNITY_PACKAGES_LOCK = "flashpatch/_unity_packages_lock_2022_3_8f1.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_bundle_is_licensed_reproducible_and_hash_verified(tmp_path: Path) -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1785417600"}
    first = tmp_path / "first"
    second = tmp_path / "second"
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_release.py"),
                "--output",
                str(output),
                "--source-commit",
                source_commit,
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )

    first_manifest = json.loads((first / "release-manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "release-manifest.json").read_text(encoding="utf-8"))
    sbom = json.loads((first / "flashpatch-0.1.0.cdx.json").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "FlashPatch" in notice
    source_manifest = json.loads((first / "source-manifest.json").read_text(encoding="utf-8"))

    assert first_manifest == second_manifest
    assert first_manifest["schema"] == "flashpatch-release-manifest-v2"
    assert first_manifest["source_git_sha"] == source_commit
    assert first_manifest["source_git_sha"] == source_manifest["source_git_sha"]
    assert len(first_manifest["artifacts"]) == 4
    for item in first_manifest["artifacts"]:
        path = first / item["file"]
        assert path.is_file()
        assert item["sha256"] == _hash(path)
    assert source_manifest["schema"] == "flashpatch-source-manifest-v1"
    source_paths = {row["path"] for row in source_manifest["files"]}
    assert {"LICENSE", "NOTICE", "README.md", "pyproject.toml"} <= source_paths
    assert any(path.startswith("src/flashpatch/") for path in source_paths)
    assert not any(path.startswith(("release/", "evidence/", "artifacts/")) for path in source_paths)

    source_archive = first / "flashpatch-0.1.0.tar.gz"
    with tarfile.open(source_archive, "r:gz") as archive:
        names = archive.getnames()
    assert "flashpatch-0.1.0/LICENSE" in names
    assert "flashpatch-0.1.0/NOTICE" in names
    assert not any("/release/" in name or "/evidence/" in name for name in names)

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["metadata"]["component"]["licenses"][0]["license"]["id"] == "Apache-2.0"
    components = {component["name"]: component for component in sbom["components"]}
    assert set(components) == {"av", "numpy", "opencv-python-headless"}
    for component in components.values():
        assert component["licenses"][0]["license"]["id"]
        assert component["externalReferences"][0]["type"] == "vcs"

    wheel = first / "flashpatch-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert any(name.endswith(".dist-info/licenses/NOTICE") for name in names)
    assert UNITY_PACKAGES_LOCK in names

    installed = tmp_path / "custom-installed"
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
    for command in (
        [
            "l10-verify",
            str(tmp_path / "missing-receipt.json"),
            "--trust-policy",
            str(tmp_path / "missing-policy.json"),
            "--expected-trust-policy-sha256",
            "0" * 64,
        ],
        [
            "l10-unity-run",
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
