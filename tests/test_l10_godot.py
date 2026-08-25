from __future__ import annotations

from pathlib import Path

import pytest

from flashpatch.l10_godot import adapter_sha256, install_godot_l10_adapter


def _project(root: Path, source: str) -> Path:
    root.mkdir()
    (root / "main.gd").write_text(source, encoding="utf-8")
    return root


def test_installs_code_owned_marker_hook_once(tmp_path: Path) -> None:
    project = _project(
        tmp_path / "project",
        "extends Node\n\nfunc run():\n"
        "\t_write_result(_output_path, fps, gameplay_state, capture_directory)\n",
    )
    digest = install_godot_l10_adapter(project)
    installed = (project / "main.gd").read_text()

    assert digest == adapter_sha256()
    assert installed.count("_flashpatch_write_marker(capture_directory)") == 1
    assert '"nonce": _flashpatch_arg("--l10-nonce")' in installed
    assert 'print("FLASHPATCH_L10_COMPLETE " + encoded)' in installed


def test_rejects_missing_or_reused_adapter_insertion_point(tmp_path: Path) -> None:
    project = _project(tmp_path / "project", "extends Node\n")
    with pytest.raises(ValueError, match="insertion point"):
        install_godot_l10_adapter(project)
