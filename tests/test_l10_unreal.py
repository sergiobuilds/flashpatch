from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashpatch.l10_unreal import UnrealHarnessError, install_unreal_harness


def _project(root: Path) -> Path:
    (root / "Content/Maps").mkdir(parents=True)
    (root / "Starter.uproject").write_text(json.dumps({"EngineAssociation": "5.6"}))
    (root / "Content/Maps/Blank.umap").write_bytes(b"map")
    return root


def test_installs_controlled_adapter_without_modifying_scene(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    scene = project / "Content/Maps/Blank.umap"
    before = scene.read_bytes()
    receipt = install_unreal_harness(project, tmp_path / "install.json")
    assert scene.read_bytes() == before
    assert receipt["source_files_modified"] == 0
    assert receipt["factual_flash_intensity"] == 1000000.0
    assert receipt["counterfactual_flash_intensity"] == 0.0
    adapter = project / receipt["adapter_path"]
    assert adapter.is_file()
    script = adapter.read_text()
    assert "FLASHPATCH_L10_START" in script
    assert "FLASHPATCH_L10_COMPLETE" in script
    assert "png_set_sha256" in script
    assert "execution-marker.json" in script


def test_rejects_wrong_engine_association(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "Starter.uproject").write_text(json.dumps({"EngineAssociation": "5.5"}))
    with pytest.raises(UnrealHarnessError, match="association"):
        install_unreal_harness(project, tmp_path / "install.json")


def test_rejects_stale_adapter(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "FlashPatchL10Capture.py").write_text("stale")
    with pytest.raises(UnrealHarnessError, match="already exists"):
        install_unreal_harness(project, tmp_path / "install.json")
