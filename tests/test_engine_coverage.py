from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
COVERAGE = ROOT / "proof" / "engine-coverage.json"


def test_engine_coverage_keeps_capability_and_evidence_depth_separate() -> None:
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    engines = {row["engine"]: row for row in coverage["engines"]}

    assert coverage["schema"] == "flashpatch-engine-coverage-v1"
    assert set(engines) == {"Godot", "Unreal Engine", "Unity"}
    assert engines["Godot"]["verdict"] == "PASS"
    assert engines["Godot"]["source_bound_correction"] is True
    assert engines["Unreal Engine"]["verdict"] == "PASS"
    assert engines["Unreal Engine"]["evidence_class"] == "controlled_renderer"
    assert engines["Unreal Engine"]["source_bound_correction"] is False
    assert engines["Unity"]["verdict"] == "INCONCLUSIVE"
    assert engines["Unity"]["public_workflow"] == "manifest-bound source preflight before editor import"
    assert engines["Unity"]["source_bound_correction"] is False

    godot_receipt = ROOT / "proof" / engines["Godot"]["public_receipt"]
    actual = "sha256:" + hashlib.sha256(godot_receipt.read_bytes()).hexdigest()
    assert actual == engines["Godot"]["public_receipt_file_sha256"]


def test_readme_surfaces_all_three_engine_lanes_without_overclaiming() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Godot 4.7.1" in readme
    assert "Unreal Engine 5.6" in readme
    assert "Unity 2022.3.8f1" in readme
    assert "proof/engine-coverage.json" in readme
    assert "production source-repair adapter" in readme
