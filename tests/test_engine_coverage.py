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
    assert "상용 프로젝트용 소스 수선 어댑터로 검증된 것은 아닙니다" in readme


def test_readme_leads_with_public_comparison_and_video_production_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    comparison = readme.index("## 1 경쟁 제품과의 차이")
    result = readme.index("## 2 실제 검증 결과")
    assert comparison < result
    for product in (
        "Ubisoft Chroma",
        "EA IRIS",
        "FFmpeg photosensitivity",
        "Apple VideoFlashingReduction",
        "EPI-LENS",
    ):
        assert product in readme
    assert "## 3 게임과 영상 제작" in readme
    assert "flashpatch scan trailer.mp4" in readme
    assert "flashpatch repair trailer.mp4 trailer-safe.mp4" in readme
    assert "flashpatch verify trailer-safe.mp4" in readme
    assert "오디오·자막" in readme
