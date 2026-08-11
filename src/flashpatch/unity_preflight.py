"""Fail-closed source preflight for a pinned Unity project fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class UnityPreflightError(ValueError):
    """A Unity source fixture cannot be bound to its declared manifest."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnityPreflightError("Unity preflight manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise UnityPreflightError("Unity preflight manifest must be an object")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise UnityPreflightError("Unity preflight path must be a non-empty relative path")
    candidate = (root / value).resolve()
    if root.resolve() not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
        raise UnityPreflightError("Unity preflight declared file is missing or unsafe")
    return candidate


def verify_unity_source_preflight(manifest_path: Path, project: Path) -> dict[str, object]:
    """Bind declared Unity source files without importing or running Unity."""
    manifest = _load(manifest_path)
    required = {"schema", "engine", "project_version", "files"}
    if set(manifest) != required or manifest["schema"] != "flashpatch-unity-source-preflight-v1":
        raise UnityPreflightError("Unity preflight manifest schema is invalid")
    if manifest["engine"] != "Unity" or not isinstance(manifest["project_version"], str):
        raise UnityPreflightError("Unity preflight manifest engine or version is invalid")
    declared = manifest["files"]
    if not isinstance(declared, dict) or not declared:
        raise UnityPreflightError("Unity preflight manifest must declare source files")
    root = project.resolve()
    if not root.is_dir() or root.is_symlink():
        raise UnityPreflightError("Unity project root is missing or unsafe")
    observed: dict[str, str] = {}
    for relative, expected in declared.items():
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise UnityPreflightError("Unity preflight file hash is invalid")
        file_path = _relative_file(root, relative)
        actual = _hash(file_path)
        if actual != expected:
            raise UnityPreflightError("Unity preflight file hash does not match")
        observed[relative] = actual
    return {"schema": "flashpatch-unity-source-preflight-receipt-v1", "engine": "Unity", "project_version": manifest["project_version"], "files": observed, "scope": "Pinned source files only; no Unity import, build, renderer capture, replay, or engine-support result was observed."}
