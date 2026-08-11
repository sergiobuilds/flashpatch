from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashpatch.unity_preflight import UnityPreflightError, verify_unity_source_preflight


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "UnityFixture"
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets").mkdir()
    version = project / "ProjectSettings" / "ProjectVersion.txt"
    scene = project / "Assets" / "Scene.unity"
    version.write_text("m_EditorVersion: 2022.3.8f1\n")
    scene.write_text("scene bytes\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema":"flashpatch-unity-source-preflight-v1","engine":"Unity","project_version":"2022.3.8f1","files":{"ProjectSettings/ProjectVersion.txt":_hash(version),"Assets/Scene.unity":_hash(scene)}}))
    return manifest, project


def test_unity_preflight_binds_declared_source_only(tmp_path: Path) -> None:
    manifest, project = _fixture(tmp_path)
    assert verify_unity_source_preflight(manifest, project)["engine"] == "Unity"


def test_unity_preflight_rejects_changed_source(tmp_path: Path) -> None:
    manifest, project = _fixture(tmp_path)
    (project / "Assets" / "Scene.unity").write_text("changed\n")
    with pytest.raises(UnityPreflightError, match="hash"):
        verify_unity_source_preflight(manifest, project)
