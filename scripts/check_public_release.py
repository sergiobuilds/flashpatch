#!/usr/bin/env python3
"""Fail when tracked public-release files expose private paths or secret material."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path


FORBIDDEN_PREFIXES = (
    ".closure/",
    ".closure-artifacts/",
    ".ouroboros/",
    "artifacts/",
    "evidence/",
    "keys/",
    "release/",
)
ABSOLUTE_USER_PATH = re.compile(
    r"(?<![\w.])/(?:home|Users)/[^/\s\"'`:,;)\]}=*?|<>]+/"
    r"|(?i:[A-Z]:\\Users\\[^\\\s\"'`:,;)\]}=*?|<>]+\\)"
)
PRIVATE_KEY_MATERIAL = re.compile(
    r"-----BEGIN " + r"(?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"
)
SECRET_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})"
)
def _tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted(
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def audit(root: Path) -> list[str]:
    violations: list[str] = []
    for relative in _tracked_files(root):
        if relative.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"forbidden tracked path: {relative}")
        path = root / relative
        if not path.is_file():
            violations.append(f"tracked entry is not a regular file: {relative}")
            continue
        content = _text(path)
        if content is None:
            continue
        if ABSOLUTE_USER_PATH.search(content):
            violations.append(f"absolute user path: {relative}")
        if PRIVATE_KEY_MATERIAL.search(content):
            violations.append(f"private key material: {relative}")
        if SECRET_TOKEN.search(content):
            violations.append(f"secret-like token: {relative}")
    project_path = root / "pyproject.toml"
    sbom_path = root / "sbom" / "flashpatch.cdx.json"
    if not project_path.is_file():
        violations.append("missing release metadata: pyproject.toml")
    if not sbom_path.is_file():
        violations.append("missing release metadata: sbom/flashpatch.cdx.json")
    if project_path.is_file() and sbom_path.is_file():
        project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
        sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
        declared = set(project["dependencies"])
        recorded = {
            property_["value"]
            for component in sbom.get("components", [])
            for property_ in component.get("properties", [])
            if property_.get("name") == "flashpatch:declared-requirement"
        }
        component = sbom.get("metadata", {}).get("component", {})
        if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
            violations.append("SBOM format must be CycloneDX 1.5")
        if component.get("name") != project["name"] or component.get("version") != project["version"]:
            violations.append("SBOM project identity does not match pyproject.toml")
        if component.get("licenses") != [{"license": {"id": "Apache-2.0"}}]:
            violations.append("SBOM project license must be Apache-2.0")
        if recorded != declared:
            violations.append("SBOM runtime requirements do not match pyproject.toml")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    root = parser.parse_args().root.resolve()
    violations = audit(root)
    if violations:
        for violation in violations:
            print(violation)
        print(f"public release audit failed: {len(violations)} violation(s)")
        return 1
    print(f"public release audit passed: {len(_tracked_files(root))} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
