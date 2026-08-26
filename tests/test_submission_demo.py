from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flashpatch.submission_demo import (
    SubmissionDemoError,
    format_submission_godot_demo,
    run_submission_godot_demo,
    verify_submission_godot_demo,
)


ROOT = Path(__file__).parents[1]
PROJECT = ROOT / "examples" / "godot" / "interaction-burst"
CONTRACT = PROJECT / "flashpatch.renderer.contract.json"
GOLDEN_DEMO = ROOT / "proof" / "godot-demo"


def _fake_pass_engine(tmp_path: Path) -> dict[str, object]:
    factual_frames = np.full((3, 8, 8, 3), 24, dtype=np.uint8)
    factual_frames[1:] = 255
    candidate_frames = np.full((3, 8, 8, 3), 24, dtype=np.uint8)
    factual_artifact = tmp_path / "private-run" / "factual.npz"
    candidate_artifact = tmp_path / "private-run" / "candidate.npz"
    diff = tmp_path / "private-run" / "patch.diff"
    factual_artifact.parent.mkdir(parents=True)
    np.savez_compressed(factual_artifact, frames=factual_frames)
    np.savez_compressed(candidate_artifact, frames=candidate_frames)
    diff.write_text(
        "--- a/main.gd\n+++ b/main.gd\n@@ -3 +3 @@\n-@export var burst_intensity := 1.0\n+@export var burst_intensity := 0.0\n",
        encoding="utf-8",
    )
    capture = {
        "godot_version": "4.7.1.stable.official.test",
        "renderer_configuration": {"display_driver": "x11", "rendering_driver": "opengl3"},
        "trace_sha256": "sha256:trace",
    }
    return {
        "verdict": "PASS",
        "reason": "same_trace_counterfactual_removed_declared_risk",
        "input_sha256": {
            "project.godot": "sha256:project",
            "source_snapshot": "sha256:source",
            "trace": "sha256:trace",
        },
        "risk_signal": {"threshold": 1.0},
        "factual_replay": {
            "frame_artifact": str(factual_artifact),
            "hazard_frame_indices": [1, 2],
            "hazardous": True,
            "max_risk": 5.0,
            "hazard_kinds": ["general_flash"],
            "renderer_capture": capture,
        },
        "attribution": {
            "frame_artifact": str(candidate_artifact),
            "diff": str(diff),
            "hazardous": False,
            "max_risk": 0.0,
            "hazard_kinds": [],
            "renderer_capture": capture,
            "runtime_attribution": {"factual_value": 1.0},
            "source_line": 3,
            "node": "/root/InteractionBurst",
            "parameter": "burst_intensity",
            "replacement": 0.0,
            "changed_source_assignments": 1,
            "causal_contribution": 5.0,
            "timing_preserved": True,
            "gameplay_state_preserved": True,
            "semantic_invariants_preserved": True,
            "gameplay_state_sha256": "sha256:state",
        },
    }


def test_godot_demo_exports_verifiable_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _fake_pass_engine(tmp_path)
    monkeypatch.setattr("flashpatch.submission_demo.compile_project", lambda *args, **kwargs: engine)
    output = tmp_path / "public-demo"

    receipt = run_submission_godot_demo(PROJECT, CONTRACT, output)
    verification = verify_submission_godot_demo(output / "receipt.json")

    assert receipt["verdict"] == "PASS"
    assert receipt["hazard"]["before"]["max_risk"] == 5.0
    assert receipt["hazard"]["after"]["max_risk"] == 0.0
    assert receipt["patch"]["changed_source_assignments"] == 1
    assert receipt["replay"]["gameplay_state_preserved"] is True
    assert receipt["source_isolation"]["original_unchanged"] is True
    assert verification["verified"] is True
    assert set(receipt["artifacts"]) == {
        "before.png",
        "after.png",
        "comparison.png",
        "patch.diff",
        "engine-receipt.json",
    }
    assert "/home/" not in (output / "engine-receipt.json").read_text(encoding="utf-8")
    rendered = format_submission_godot_demo(receipt, output)
    assert "actual Godot frames" in rendered
    assert "RESULT    PASS risk=5.0 -> 0.0" in rendered


def test_godot_demo_verifier_rejects_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _fake_pass_engine(tmp_path)
    monkeypatch.setattr("flashpatch.submission_demo.compile_project", lambda *args, **kwargs: engine)
    output = tmp_path / "public-demo"
    run_submission_godot_demo(PROJECT, CONTRACT, output)
    (output / "patch.diff").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(SubmissionDemoError, match="artifact hash mismatch"):
        verify_submission_godot_demo(output / "receipt.json")


def test_godot_demo_preserves_fail_closed_engine_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flashpatch.submission_demo.compile_project",
        lambda *args, **kwargs: {
            "verdict": "INCONCLUSIVE",
            "reason": "missing_renderer_timestamps",
        },
    )
    output = tmp_path / "inconclusive-demo"

    receipt = run_submission_godot_demo(PROJECT, CONTRACT, output)

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["reason"] == "missing_renderer_timestamps"
    assert set(receipt["artifacts"]) == {"engine-receipt.json"}
    with pytest.raises(SubmissionDemoError, match="receipt is not PASS"):
        verify_submission_godot_demo(output / "receipt.json")


def test_godot_demo_receipt_is_plain_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _fake_pass_engine(tmp_path)
    monkeypatch.setattr("flashpatch.submission_demo.compile_project", lambda *args, **kwargs: engine)
    output = tmp_path / "public-demo"
    run_submission_godot_demo(PROJECT, CONTRACT, output)

    assert json.loads((output / "receipt.json").read_text(encoding="utf-8"))["schema"].endswith("-v1")


def test_checked_in_godot_demo_is_self_verifying_and_public_safe() -> None:
    verification = verify_submission_godot_demo(GOLDEN_DEMO / "receipt.json")
    engine_receipt = (GOLDEN_DEMO / "engine-receipt.json").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert verification["verified"] is True
    assert verification["hazard_before"] == 5.0
    assert verification["hazard_after"] == 0.0
    assert "/home/" not in engine_receipt
    assert "proof/godot-demo/comparison.png" in readme
    assert "flashpatch verify-godot-demo proof/godot-demo/receipt.json" in readme
    assert "5.0 → 0.0" in readme
