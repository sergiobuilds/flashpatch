from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashpatch import external_league


def test_checkout_provenance_uses_explicit_remote_not_stale_tracking_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def fake_run(command: list[str], **kwargs: object) -> Result:
        calls.append(command)
        if command[:2] == ["git", "ls-remote"]:
            assert command[3] == "https://github.com/sergiobuilds/flashpatch.git"
            return Result("a" * 40 + "\trefs/heads/master\n")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return Result("a" * 40 + "\n")
        if command[-2:] == ["rev-parse", "HEAD^{tree}"]:
            return Result("c" * 40 + "\n")
        if command[-2] == "rev-parse" and command[-1].startswith("HEAD:"):
            return Result("b" * 40 + "\n")
        if command[-3:] == ["remote", "get-url", "origin"]:
            return Result("git@github.com:sergiobuilds/flashpatch.git\n")
        if command[3:7] == ["status", "--porcelain", "--untracked-files=all", "--"]:
            return Result("")
        raise AssertionError(command)

    monkeypatch.setattr(external_league.subprocess, "run", fake_run)
    monkeypatch.setattr(external_league.Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setenv("FLASHPATCH_REMOTE_URL", "https://github.com/sergiobuilds/flashpatch.git")

    result = external_league._flashpatch_checkout_provenance()

    assert result["pushed"] is True
    assert result["revision"] == "a" * 40
    assert result["tree"] == "c" * 40
    assert result["execution_revision"] != result["revision"]
    assert result["remote_verification_url"] == "https://github.com/sergiobuilds/flashpatch"
    assert any(command[:2] == ["git", "ls-remote"] for command in calls)


def test_checkout_provenance_accepts_only_matching_coordinator_remote_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = tmp_path / "flashpatch-remote-verification.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "flashpatch-l7-remote-verification-receipt-v1",
                "remote_url": "https://github.com/sergiobuilds/flashpatch.git",
                "head": "a" * 40,
                "observed_remote_head": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FLASHPATCH_REMOTE_VERIFICATION_RECEIPT", str(receipt))
    assert external_league._verified_remote_receipt(
        tmp_path, "https://github.com/sergiobuilds/flashpatch.git", "a" * 40
    ) is True
    receipt.write_text(receipt.read_text(encoding="utf-8").replace("a" * 40, "b" * 40), encoding="utf-8")
    with pytest.raises(external_league.ExternalLeagueError, match="does not match"):
        external_league._verified_remote_receipt(
            tmp_path, "https://github.com/sergiobuilds/flashpatch.git", "a" * 40
        )
