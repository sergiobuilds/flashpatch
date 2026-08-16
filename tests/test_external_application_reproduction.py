from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "evidence" / "external-application-reproduction-2026-08-11.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_external_application_reproduction_evidence() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema"] == "flashpatch-external-application-reproduction-v1"
    assert payload["unit_id"] == "external-application-reproduction"
    assert payload["host"] == "campbell"
    assert payload["verdict"] == "PASS"
    source_commit = payload["source_commit"]
    assert len(source_commit) == 40
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    assert payload["full_test_suite"]["exit_code"] == 0
    assert "passed" in payload["full_test_suite"]["stdout_tail"]

    media = payload["actual_media"]
    source = ROOT / media["repository_path"]
    assert source.is_file()
    assert media["input_sha256"] == sha256(source)
    assert media["input_hazardous"] is True
    assert media["repair_status"] == "VERIFIED"
    assert media["independent_verification_passed"] is True
    assert media["repaired_sha256"]

    commands = [entry["command"] for entry in payload["commands"]]
    assert commands[0][:3] == ["git", "clone", "--no-local"]
    assert all(entry["exit_code"] == 0 for entry in payload["commands"])
