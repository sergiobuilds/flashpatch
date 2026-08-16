from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_bundle_is_licensed_reproducible_and_hash_verified(tmp_path: Path) -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1785417600"}
    first = tmp_path / "first"
    second = tmp_path / "second"

    for output in (first, second):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_release.py"), "--output", str(output)],
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
    assert first_manifest == second_manifest
    assert first_manifest["git_sha"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert len(first_manifest["artifacts"]) == 3
    for item in first_manifest["artifacts"]:
        path = first / item["file"]
        assert path.is_file()
        assert item["sha256"] == _hash(path)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["metadata"]["component"]["licenses"][0]["license"]["id"] == "Apache-2.0"
    assert {component["name"] for component in sbom["components"]} == {
        "av",
        "numpy",
        "opencv-python-headless",
    }
