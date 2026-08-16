from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

import flashpatch.l7_verify as l7_verify
from flashpatch.l7_verify import (
    CORPUS_DISCOVERY_SCHEMA,
    CORPUS_DISCOVERY_SCHEMA_V2,
    CORPUS_DISCOVERY_SCHEMA_V3,
    CORPUS_DISCOVERY_SCHEMA_V4,
    CORPUS_DISCOVERY_SCHEMA_V5,
    L7VerificationFailure,
    assess_corpus_discovery_manifest,
    assess_natural_case_bundle,
    assess_native_main_natural_case_bundle,
    assess_native_main_qualification_summary,
    execute_v4_candidate_native_main_original,
    verify_corpus_discovery_manifest,
    verify_native_main_natural_case_bundle,
    verify_native_main_qualification_summary,
    verify_v4_candidate_source_preflight,
    verify_natural_case_bundle,
)


ROOT = Path(__file__).parents[1]
V4_DISCOVERY = ROOT / "evidence" / "l7" / "corpus-discovery" / "corpus-discovery-v4.json"
V5_DISCOVERY = ROOT / "evidence" / "l7" / "corpus-discovery" / "corpus-discovery-v5.json"
V4_OLD_FILM_PREFLIGHT = (
    ROOT / "evidence" / "l7" / "source-preflight" / "godotdemo-old-film-v4.json"
)
V5_OLD_FILM_PREFLIGHT = (
    ROOT / "evidence" / "l7" / "source-preflight" / "godotdemo-old-film-v5.json"
)
V4_DISCOVERY_SHA256 = "c2e79000212aaa09a0aee02c9089d52ccfc2f0eae893494b7a36993ebb7003ba"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "source-checkout"
    checkout.mkdir()
    (checkout / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (checkout / "project.godot").write_text("[application]\nconfig/name=\"Natural\"\n", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "tests@example.invalid"],
        ["git", "config", "user.name", "FlashPatch test"],
        ["git", "add", "LICENSE", "project.godot"],
        ["git", "commit", "-qm", "natural source"],
        ["git", "remote", "add", "origin", "https://github.com/example/natural-godot-game.git"],
    ):
        subprocess.run(command, cwd=checkout, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    return checkout, revision


def _source_tree_sha256(checkout: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(checkout.rglob("*")):
        relative = path.relative_to(checkout)
        if relative.parts[0] == ".git" or not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _bundle(tmp_path: Path) -> Path:
    rgb = tmp_path / "renderer.rgb"
    rgb.write_bytes(b"actual-rgb-renderer-bytes")
    timestamps = _write_json(tmp_path / "timestamps.json", [0.0, 1 / 60, 2 / 60])
    trace = _write_json(tmp_path / "trace.json", {"actions": [{"frame": 0, "action": "start"}]})
    checkout, revision = _checkout(tmp_path)
    repository = {
        "url": "https://github.com/example/natural-godot-game",
        "revision": revision,
        "license": "MIT",
        "project_subpath": "game",
    }
    source = _write_json(
        tmp_path / "source-provenance.json",
        {
            "repository_url": repository["url"],
            "revision": repository["revision"],
            "license": repository["license"],
            "project_subpath": repository["project_subpath"],
            "original_unmutated": True,
            "source_checkout_path": checkout.name,
            "source_tree_sha256": _source_tree_sha256(checkout),
            "license_path": f"{checkout.name}/LICENSE",
            "license_sha256": _sha256(checkout / "LICENSE"),
        },
    )
    receipt = _write_json(
        tmp_path / "renderer-execution.json",
        {
            "schema": "flashpatch-renderer-engine-receipt-v1",
            "controlled_mutation": False,
            "upstream": {
                "repository_url": repository["url"],
                "source_revision": repository["revision"],
                "license": repository["license"],
                "project_path": repository["project_subpath"],
            },
            "factual_replay": {
                "renderer_rgb_raw_sha256": _sha256(rgb),
                "timestamps_sha256": _sha256(timestamps),
                "frame_count": 3,
                "renderer_capture": {
                    "trace_sha256": f"sha256:{_sha256(trace)}",
                    "godot_version": "4.3-stable",
                },
            },
        },
    )
    _write_json(
        tmp_path / "natural-case.json",
        {
            "schema": "flashpatch-l7-natural-case-ledger-v1",
            "case_id": "natural-godot-001",
            "case_class": "natural_public_godot",
            "controlled_mutation": False,
            "repository": repository,
            "source_provenance": {"path": source.name, "sha256": _sha256(source)},
            "renderer": {
                "execution_receipt_path": receipt.name,
                "execution_receipt_sha256": _sha256(receipt),
                "rgb_path": rgb.name,
                "rgb_sha256": _sha256(rgb),
                "timestamps_path": timestamps.name,
                "timestamps_sha256": _sha256(timestamps),
                "frame_count": 3,
            },
            "trace": {"path": trace.name, "sha256": _sha256(trace)},
        },
    )
    return tmp_path


def _ledger(root: Path) -> dict[str, object]:
    return json.loads((root / "natural-case.json").read_text(encoding="utf-8"))


def _save_ledger(root: Path, ledger: dict[str, object]) -> None:
    _write_json(root / "natural-case.json", ledger)


def _native_main_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout, revision = _checkout(tmp_path)
    repository = {"url": "https://github.com/example/natural-godot-game", "revision": revision, "license": "MIT", "project_subpath": "."}
    source = _write_json(tmp_path / "source-provenance.json", {"repository_url": repository["url"], "revision": revision, "license": "MIT", "project_subpath": ".", "original_unmutated": True, "source_checkout_path": checkout.name, "source_tree_sha256": _source_tree_sha256(checkout), "license_path": f"{checkout.name}/LICENSE", "license_sha256": _sha256(checkout / "LICENSE")})
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    (qualification / "LICENSE").write_bytes((checkout / "LICENSE").read_bytes())
    (qualification / "project.godot").write_text("[application]\nconfig/name=\"Instrumented\"\n", encoding="utf-8")
    (qualification / ".flashpatch").mkdir()
    (qualification / ".flashpatch" / "upstream-project.godot").write_bytes((checkout / "project.godot").read_bytes())
    trace = _write_json(tmp_path / "trace.json", {"fixed_fps": 60, "capture_frames": 3, "warmup_frames": 0, "original_main_scene": "res://main.tscn", "actions": [], "pointer_events": [], "key_events": [], "scenario_readiness": {"required_node_paths": ["/root/Main"], "required_group_minimums": {}, "required_visible": [{"node_path": "/root/Main", "visible": True}], "required_option_selection": []}, "runtime_observations": [], "scene_transition": None, "ui_selection_observations": []})
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    timestamps = np.asarray([0.0, 0.016666, 0.033333], dtype=np.float64)
    artifact = tmp_path / "renderer-frames.npz"
    np.savez_compressed(artifact, frames=frames, timestamps=timestamps)
    replay = _write_json(tmp_path / "replay.json", {"qualification_only": True, "scoreable": False, "native_equivalence": "NOT_ESTABLISHED", "execution_mode": "instrumented_native_main_scene_capture", "frames_npz": artifact.name, "renderer_capture": {"artifact": artifact.name, "actual_capture_timestamps_us": [1, 2, 3]}})
    receipt = _write_json(tmp_path / "native-main-capture-receipt.json", {"schema": "flashpatch-godot-native-main-capture-v1", "decision": "SAFE_SCENARIO_READY", "controlled_mutation": False, "upstream_defect": None, "qualification_only": True, "scoreable": False, "execution_mode": "instrumented_native_main_scene_capture", "native_equivalence": "NOT_ESTABLISHED", "frame_artifact_sha256": _sha256(artifact), "replay_sha256": _sha256(replay), "trace_sha256": f"sha256:{_sha256(trace)}", "frame_count": 3, "presentation_timestamps_us": [0, 16666, 33333]})
    manifest = _write_json(tmp_path / "manifest.json", {"candidates": [{"candidate_id": "natural-native", "repository_url": repository["url"], "revision": revision, "license": "MIT", "trace_template_id": "trace-native"}], "trace_templates": [{"trace_template_id": "trace-native", "fixed_fps": 60, "capture_frames": 3, "original_main_scene": "res://main.tscn", "actions": [], "pointer_events": [], "key_events": [], "scenario_readiness": json.loads(trace.read_text())["scenario_readiness"], "ui_selection_observations": []}]})
    monkeypatch.setattr(l7_verify, "verify_corpus_discovery_manifest", lambda path: {"status": "NOT_SCOREABLE"})
    native = {"candidate_id": "natural-native", "manifest_path": manifest.name, "manifest_sha256": _sha256(manifest), "preflight_path": "preflight.json", "qualification_path": qualification.name, "frame_artifact_path": artifact.name, "frame_artifact_sha256": _sha256(artifact), "replay_path": replay.name, "replay_sha256": _sha256(replay), "execution_receipt_path": receipt.name, "execution_receipt_sha256": _sha256(receipt), "trace_path": trace.name, "trace_sha256": _sha256(trace), "rgb_bytes_sha256": hashlib.sha256(frames.tobytes()).hexdigest(), "timestamps_f64_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest()}
    preflight = _write_json(tmp_path / "preflight.json", {"candidate_id": "natural-native", "repository_url": repository["url"], "revision": revision, "license": "MIT", "project_subpath": ".", "manifest_sha256": native["manifest_sha256"], "renderer_executed": False, "scoreable": False})
    native["preflight_sha256"] = _sha256(preflight)
    _write_json(tmp_path / "native-main-natural-case.json", {"schema": "flashpatch-l7-native-main-natural-case-ledger-v1", "case_id": "native-main-001", "case_class": "natural_public_godot", "controlled_mutation": False, "repository": repository, "source_provenance": {"path": source.name, "sha256": _sha256(source)}, "native_main": native})
    return tmp_path


def test_native_main_npz_bundle_reopens_raw_artifact_but_stays_not_scoreable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = verify_native_main_natural_case_bundle(_native_main_bundle(tmp_path, monkeypatch))
    assert result["status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["reason"] == "qualification_only_native_equivalence_not_established"
    assert result["score_blockers"] == [
        "qualification_only_capture",
        "native_equivalence_not_established",
        "independent_gold_missing",
    ]
    assert result["native_equivalence"] == "NOT_ESTABLISHED"


def test_native_main_assessment_and_cli_never_promote_safe_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _native_main_bundle(tmp_path, monkeypatch)
    assert assess_native_main_natural_case_bundle(root)["status"] == "NOT_SCOREABLE"
    result = subprocess.run([sys.executable, "-m", "flashpatch.l7_verify", "--native-main-case-root", str(root)], cwd=ROOT, env=os.environ | {"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["scoreable"] is False


@pytest.mark.parametrize("field", ["frame_artifact_sha256", "replay_sha256", "trace_sha256"])
def test_native_main_npz_bundle_rejects_receipt_cross_binding_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    root = _native_main_bundle(tmp_path, monkeypatch)
    receipt_path = root / "native-main-capture-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = "0" * 64 if field != "trace_sha256" else "sha256:" + "0" * 64
    _write_json(receipt_path, receipt)
    ledger = json.loads((root / "native-main-natural-case.json").read_text())
    ledger["native_main"]["execution_receipt_sha256"] = _sha256(receipt_path)
    _write_json(root / "native-main-natural-case.json", ledger)
    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:native_main:receipt_contract_mismatch"):
        verify_native_main_natural_case_bundle(root)


def _native_main_summary_fixture(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    (project / "src" / "flashpatch").mkdir(parents=True)
    (project / "docs" / "MASTER-MAP.md").write_text("# map\n", encoding="utf-8")
    manifest = _write_json(
        project / "manifest.json",
        {"schema": CORPUS_DISCOVERY_SCHEMA_V5, "candidates": [], "trace_templates": []},
    )
    preflight = _write_json(
        project / "preflight-summary.json",
        {"schema": "flashpatch-l7-source-preflight-summary-v1", "status": "PRECHECKED_NOT_SCOREABLE"},
    )
    frame = project / "artifacts" / "case-a" / "renderer-frames.npz"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"npz-bytes")
    receipt = _write_json(
        project / "artifacts" / "case-a" / "native-main-capture-receipt.json",
        {
            "schema": "flashpatch-godot-native-main-capture-v1",
            "decision": "SAFE_SCENARIO_READY",
            "qualification_only": True,
            "scoreable": False,
            "native_equivalence": "NOT_ESTABLISHED",
            "frame_count": 120,
        },
    )
    summary = {
        "schema": "flashpatch-l7-native-main-qualification-summary-v1",
        "summary_id": "native-main-test",
        "discovery_manifest_path": "manifest.json",
        "discovery_manifest_sha256": _sha256(manifest),
        "source_preflight_summary_path": "preflight-summary.json",
        "source_preflight_summary_sha256": _sha256(preflight),
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "external_claim_authorized": False,
        "candidate_count": 2,
        "executed_complete_receipt_count": 1,
        "failed_count": 1,
        "safe_qualification_only_count": 1,
        "hazardous_attribution_pending_count": 0,
        "qualified_case_count": 0,
        "score_blockers": [
            "qualification_only_capture",
            "native_equivalence_not_established",
            "independent_gold_missing",
            "qualified_natural_case_count_below_nine",
            "same_condition_9_slot_execution_missing",
            "receipt_bound_score_bundle_missing",
        ],
        "results": [
            {
                "candidate_id": "case-a",
                "status": "SAFE_SCENARIO_READY_QUALIFICATION_ONLY",
                "decision": "SAFE_SCENARIO_READY",
                "scoreable": False,
                "external_claim_authorized": False,
                "receipt_created": True,
                "frame_count": 120,
                "raw_receipt_path": "artifacts/case-a/native-main-capture-receipt.json",
                "raw_receipt_sha256": _sha256(receipt),
                "frame_artifact_path": "artifacts/case-a/renderer-frames.npz",
                "frame_artifact_sha256": _sha256(frame),
            },
            {
                "candidate_id": "case-b",
                "status": "FAILED_TIMEOUT",
                "scoreable": False,
                "external_claim_authorized": False,
                "receipt_created": False,
                "failure": "timeout",
            },
        ],
    }
    summary_path = project / "evidence" / "l7" / "native-main" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    return _write_json(summary_path, summary)


def test_native_main_qualification_summary_reopens_receipts_but_stays_not_scoreable(tmp_path: Path) -> None:
    result = verify_native_main_qualification_summary(_native_main_summary_fixture(tmp_path))
    assert result["status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["candidate_count"] == 2
    assert result["executed_complete_receipt_count"] == 1
    assert result["failed_count"] == 1
    assert result["qualified_case_count"] == 0
    assert "independent_gold_missing" in result["score_blockers"]


def test_native_main_qualification_summary_rejects_scoreable_claim(tmp_path: Path) -> None:
    summary = _native_main_summary_fixture(tmp_path)
    payload = json.loads(summary.read_text())
    payload["scoreable"] = True
    _write_json(summary, payload)
    result = assess_native_main_qualification_summary(summary)
    assert result["status"] == "INCONCLUSIVE"
    assert result["scoreable"] is False
    assert result["diagnostic"] == "FAIL_CLOSED:native_main_summary:scoreability_claim_invalid"


def _discovery_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    qualified_root = tmp_path / "qualified-natural-case"
    qualified_root.mkdir()
    _bundle(qualified_root)
    qualified_ledger = _ledger(qualified_root)
    qualified_ledger["godot_version"] = "4.3-stable"
    _save_ledger(qualified_root, qualified_ledger)
    qualified_repository = qualified_ledger["repository"]
    assert isinstance(qualified_repository, dict)

    manifest: dict[str, object] = {
        "schema": CORPUS_DISCOVERY_SCHEMA,
        "discovery_id": "public-godot-census-2026-07",
        "discovery_cutoff_utc": "2026-07-01T00:00:00Z",
        "source_discovery_method": {
            "name": "github_repository_search",
            "source": "https://github.com/search",
            "query": "language:GDScript topic:godot sort:stars",
            "executed_at_utc": "2026-06-30T20:00:00Z",
        },
        "inclusion_rules": [
            {
                "id": "I-RUNTIME",
                "criterion": "Repository exposes a reproducible Godot runtime entrypoint",
            }
        ],
        "exclusion_rules": [
            {
                "id": "E-LICENSE",
                "criterion": "Repository lacks verifiable redistribution license evidence",
            }
        ],
        "candidates": [
            {
                "candidate_id": "natural-qualified",
                "discovered_at_utc": "2026-06-30T20:10:00Z",
                "repository_url": qualified_repository["url"],
                "revision": qualified_repository["revision"],
                "license": qualified_repository["license"],
                "godot_version": "4.3-stable",
                "case_class": "natural_public_godot",
                "controlled_mutation": False,
                "selection_outcome": "qualified",
                "selection_reason": "Runtime provenance and replay qualification completed",
                "rule_ids": ["I-RUNTIME"],
                "natural_bundle_path": qualified_root.name,
                "natural_bundle_ledger_sha256": _sha256(qualified_root / "natural-case.json"),
            },
            {
                "candidate_id": "natural-prospect",
                "discovered_at_utc": "2026-06-30T20:20:00Z",
                "repository_url": "https://github.com/example/second-godot-game",
                "revision": "b" * 40,
                "license": "Apache-2.0",
                "godot_version": "4.2.2-stable",
                "case_class": "natural_public_godot",
                "controlled_mutation": False,
                "selection_outcome": "prospect",
                "selection_reason": "Runtime provenance qualification remains pending",
                "rule_ids": ["I-RUNTIME"],
            },
            {
                "candidate_id": "natural-excluded",
                "discovered_at_utc": "2026-06-30T20:30:00Z",
                "repository_url": "https://github.com/example/third-godot-game",
                "revision": "c" * 40,
                "license": "UNKNOWN",
                "godot_version": "3.5.3-stable",
                "case_class": "natural_public_godot",
                "controlled_mutation": False,
                "selection_outcome": "excluded",
                "selection_reason": "Required redistribution license evidence is unavailable",
                "rule_ids": ["E-LICENSE"],
            },
        ],
    }
    return _write_json(tmp_path / "corpus-discovery.json", manifest), manifest


def test_valid_natural_receipt_bundle_is_qualified_but_not_scored(tmp_path: Path) -> None:
    result = verify_natural_case_bundle(_bundle(tmp_path))

    assert result["status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["reason"] == "independent_gold_missing"
    assert result["external_claim_authorized"] is False
    assert "winner" not in result


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda root, ledger: ledger.update({"controlled_mutation": True}),
            "FAIL_CLOSED:case_class:controlled_not_natural",
        ),
        (
            lambda root, ledger: ledger["repository"].update({"license": "Apache-2.0"}),  # type: ignore[index]
            "FAIL_CLOSED:provenance:source_provenance_mismatch",
        ),
        (
            lambda root, ledger: (root / "renderer.rgb").write_bytes(b"tampered-rgb"),
            "FAIL_CLOSED:renderer:rgb_hash_mismatch",
        ),
        (
            lambda root, ledger: (root / "timestamps.json").write_text("[0.0, 0.0, 0.03]\n", encoding="utf-8"),
            "FAIL_CLOSED:timestamps:timestamps_hash_mismatch",
        ),
        (
            lambda root, ledger: ledger["trace"].update({"path": "missing-trace.json"}),  # type: ignore[index]
            "FAIL_CLOSED:trace:trace_path_invalid",
        ),
        (
            lambda root, ledger: ledger["renderer"].update({"execution_receipt_path": "missing-receipt.json"}),  # type: ignore[index]
            "FAIL_CLOSED:renderer:execution_receipt_path_invalid",
        ),
    ],
)
def test_adversarial_natural_case_bundle_is_rejected(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    root = _bundle(tmp_path)
    ledger = _ledger(root)
    mutate(root, ledger)
    _save_ledger(root, ledger)

    with pytest.raises(L7VerificationFailure, match=diagnostic):
        verify_natural_case_bundle(root)
    assessed = assess_natural_case_bundle(root)
    assert assessed == {
        "schema": "flashpatch-l7-natural-case-assessment-v1",
        "status": "INCONCLUSIVE",
        "scoreable": False,
        "diagnostic": diagnostic,
        "external_claim_authorized": False,
    }


def test_receipt_cross_binding_rejects_rgb_timestamp_and_trace_claim_drift(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    receipt_path = root / "renderer-execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["factual_replay"]["renderer_rgb_raw_sha256"] = "c" * 64
    _write_json(receipt_path, receipt)
    ledger = _ledger(root)
    ledger["renderer"]["execution_receipt_sha256"] = _sha256(receipt_path)  # type: ignore[index]
    _save_ledger(root, ledger)
    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:renderer:receipt_artifact_mismatch"):
        verify_natural_case_bundle(root)


def test_controlled_renderer_receipt_cannot_be_disguised_by_a_natural_ledger(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    receipt_path = root / "renderer-execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["controlled_mutation"] = True
    _write_json(receipt_path, receipt)
    ledger = _ledger(root)
    ledger["renderer"]["execution_receipt_sha256"] = _sha256(receipt_path)  # type: ignore[index]
    _save_ledger(root, ledger)

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:case_class:controlled_not_natural"):
        verify_natural_case_bundle(root)


def test_source_checkout_is_remeasured_instead_of_trusting_provenance_json(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    checkout = root / "source-checkout"
    (checkout / "project.godot").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:provenance:source_checkout_mismatch"):
        verify_natural_case_bundle(root)


def test_timestamp_monotonicity_is_measured_not_trusted_from_a_flag(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    timestamps = _write_json(root / "timestamps.json", [0.0, 0.0, 2 / 60])
    ledger = _ledger(root)
    ledger["renderer"]["timestamps_sha256"] = _sha256(timestamps)  # type: ignore[index]
    receipt_path = root / "renderer-execution.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["factual_replay"]["timestamps_sha256"] = _sha256(timestamps)
    _write_json(receipt_path, receipt)
    ledger["renderer"]["execution_receipt_sha256"] = _sha256(receipt_path)  # type: ignore[index]
    _save_ledger(root, ledger)

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:timestamps:not_strictly_monotonic"):
        verify_natural_case_bundle(root)


def test_cli_never_emits_pass_or_score_for_a_g2_bundle(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "flashpatch.l7_verify", "--case-root", str(root)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "NOT_SCOREABLE"
    assert "PASS" not in result.stdout


def _candidate(manifest: dict[str, object], candidate_id: str) -> dict[str, object]:
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    return next(candidate for candidate in candidates if candidate["candidate_id"] == candidate_id)


def _append_repository_copy(
    manifest: dict[str, object],
    *,
    candidate_id: str,
    revision: str | None = None,
    repository_url: str | None = None,
) -> None:
    duplicate = copy.deepcopy(_candidate(manifest, "natural-prospect"))
    duplicate["candidate_id"] = candidate_id
    if revision is not None:
        duplicate["revision"] = revision
    if repository_url is not None:
        duplicate["repository_url"] = repository_url
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    candidates.append(duplicate)


def test_discovery_does_not_freeze_three_repositories_with_only_one_qualified_case(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _discovery_manifest(tmp_path)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["corpus_frozen"] is False
    assert result["reason"] == "minimum_nine_qualified_natural_cases_not_met"
    assert result["distinct_repository_count"] == 3
    assert result["qualified_case_count"] == 1
    assert result["outcome_counts"] == {"excluded": 1, "prospect": 1, "qualified": 1}
    assert result["qualified_case_receipts"][0]["status"] == "NOT_SCOREABLE"  # type: ignore[index]
    assert result["qualified_case_receipts"][0]["scoreable"] is False  # type: ignore[index]
    assert len(result["manifest_sha256"]) == 64
    assert result["external_claim_authorized"] is False
    assert not {"winner", "ranking", "scores"}.intersection(result)


def test_discovery_freezes_nine_distinct_cases_across_three_repositories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    template = copy.deepcopy(_candidate(manifest, "natural-qualified"))
    repositories = [copy.deepcopy(candidate) for candidate in candidates]
    for index in range(2, 10):
        candidate = copy.deepcopy(template)
        repository = repositories[(index - 1) % len(repositories)]
        candidate["candidate_id"] = f"natural-qualified-{index}"
        candidate["repository_url"] = repository["repository_url"]
        candidate["revision"] = repository["revision"]
        candidate["license"] = repository["license"]
        candidate["godot_version"] = repository["godot_version"]
        candidate["natural_bundle_path"] = f"qualified-natural-case-{index}"
        candidate["natural_bundle_ledger_sha256"] = f"{index:064x}"
        candidates.append(candidate)
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        l7_verify,
        "_verify_qualified_discovery_bundle",
        lambda root, candidate: {
            "schema": "flashpatch-l7-natural-case-assessment-v1",
            "case_id": candidate["candidate_id"],
            "status": "NOT_SCOREABLE",
            "scoreable": False,
            "reason": "independent_gold_missing",
            "external_claim_authorized": False,
        },
    )

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["corpus_frozen"] is True
    assert result["reason"] == "discovery_population_frozen_without_scoring_authority"
    assert result["distinct_repository_count"] == 3
    assert result["qualified_case_count"] == 9
    assert result["scoreable"] is False
    assert result["external_claim_authorized"] is False


def test_discovery_allows_distinct_candidates_from_same_normalized_immutable_repository(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _append_repository_copy(
        manifest,
        candidate_id="same-repository-new-case",
        repository_url="https://github.com/example/second-godot-game.git/",
    )
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["candidate_count"] == 4
    assert result["distinct_repository_count"] == 3
    assert result["qualified_case_count"] == 1


def test_v2_discovery_binds_every_candidate_to_its_own_source_method(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    manifest["schema"] = CORPUS_DISCOVERY_SCHEMA_V2
    method = manifest.pop("source_discovery_method")
    assert isinstance(method, dict)
    manifest["source_discovery_methods"] = [
        {"method_id": "github-census", **method},
        {
            "method_id": "gitlab-census",
            "name": "gitlab_repository_search",
            "source": "https://gitlab.com/explore/projects",
            "query": "topic:godot language:GDScript",
            "executed_at_utc": "2026-06-30T20:00:00Z",
        },
    ]
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        candidate["source_discovery_method_id"] = "github-census"
    candidates[-1]["repository_url"] = "https://gitlab.com/example/third-godot-game"
    candidates[-1]["source_discovery_method_id"] = "gitlab-census"
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["source_discovery_method_count"] == 2
    assert result["distinct_repository_count"] == 3


@pytest.mark.parametrize(
    ("method_id", "diagnostic"),
    [
        ("unknown-census", "FAIL_CLOSED:discovery_candidates:candidate_method_unknown:natural-prospect"),
        ("", "FAIL_CLOSED:discovery_candidates:candidate_method_unknown:natural-prospect"),
    ],
)
def test_v2_discovery_rejects_unknown_candidate_source_method(
    tmp_path: Path, method_id: str, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    manifest["schema"] = CORPUS_DISCOVERY_SCHEMA_V2
    method = manifest.pop("source_discovery_method")
    assert isinstance(method, dict)
    manifest["source_discovery_methods"] = [{"method_id": "github-census", **method}]
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        candidate["source_discovery_method_id"] = "github-census"
    _candidate(manifest, "natural-prospect")["source_discovery_method_id"] = method_id
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


def _upgrade_to_v3(manifest: dict[str, object]) -> None:
    manifest["schema"] = CORPUS_DISCOVERY_SCHEMA_V3
    method = manifest.pop("source_discovery_method")
    assert isinstance(method, dict)
    manifest["source_discovery_methods"] = [{"method_id": "github-census", **method}]
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        candidate["source_discovery_method_id"] = "github-census"
        candidate["trace_template_id"] = f"trace-{candidate['candidate_id']}"
    manifest["trace_budget"] = {
        "max_frames": 180,
        "max_wall_clock_seconds": 120,
        "repeat_count": 3,
    }
    manifest["trace_templates"] = [
        {
            "trace_template_id": candidate["trace_template_id"],
            "entrypoint": f"res://{candidate['candidate_id']}.tscn",
            "action_sequence": ["input:launch:tick=0"],
            "capture_frame_count": 120,
        }
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    manifest["selection_order"] = [candidate["candidate_id"] for candidate in candidates]
    manifest["stop_rule"] = {
        "maximum_selected_cases": len(candidates),
        "maximum_cases_per_repository": 3,
        "on_execution_failure": "retain_and_stop",
    }
    manifest["replacement_policy"] = {
        "replacement_allowed": False,
        "replacement_requires_new_manifest": True,
    }


def test_v3_discovery_freezes_trace_and_selection_contract(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v3(manifest)
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["candidate_count"] == 3
    assert result["source_discovery_method_count"] == 1


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda manifest: manifest.update({"selection_order": ["natural-prospect"]}),
            "FAIL_CLOSED:discovery_selection:order_not_exact_candidate_universe",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-prospect").update(
                {"trace_template_id": "unknown-trace"}
            ),
            "FAIL_CLOSED:discovery_trace:candidate_template_unknown:natural-prospect",
        ),
        (
            lambda manifest: manifest["replacement_policy"].update(  # type: ignore[index,union-attr]
                {"replacement_allowed": True}
            ),
            "FAIL_CLOSED:discovery_selection:replacement_policy_invalid",
        ),
    ],
)
def test_v3_discovery_rejects_postselection_seams(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v3(manifest)
    mutate(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


def _upgrade_to_v4(manifest: dict[str, object]) -> None:
    _upgrade_to_v3(manifest)
    manifest["schema"] = CORPUS_DISCOVERY_SCHEMA_V4
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        candidate["godot_version"] = "4.3-stable"
        candidate["license_path"] = "LICENSE"
        candidate["license_sha256"] = "d" * 64
    manifest["godot_binary"] = {
        "sha256": "e" * 64,
        "version": "4.7.stable.official.test",
    }
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    manifest["trace_templates"] = [
        {
            "trace_template_id": template["trace_template_id"],
            "runner_id": "flashpatch-native-main-godot4-v1",
            "project_subpath": ".",
            "original_main_scene": "res://main.tscn",
            "fixed_fps": 60,
            "capture_frames": 120,
            "actions": [],
            "launch_arguments": [],
            "pointer_events": [],
            "key_events": [],
            "scenario_readiness": {
                "required_node_paths": ["/root/Main"],
                "required_group_minimums": {},
                "required_visible": [{"node_path": "/root/Main", "visible": True}],
            },
            "runtime_observations": [],
        }
        for template in templates
        if isinstance(template, dict)
    ]


def _upgrade_to_v5(manifest: dict[str, object]) -> None:
    _upgrade_to_v4(manifest)
    manifest["schema"] = CORPUS_DISCOVERY_SCHEMA_V5
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        template["runner_id"] = "flashpatch-native-main-godot4-v2"
        template["scene_transition"] = None
        template["ui_selection_observations"] = []
        readiness = template["scenario_readiness"]
        assert isinstance(readiness, dict)
        readiness["required_option_selection"] = []


def test_v4_discovery_requires_runner_consumable_native_main_trace(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["candidate_count"] == 3


@pytest.mark.parametrize(
    ("mutate_budget", "diagnostic"),
    [
        (
            lambda budget: budget.pop("max_wall_clock_seconds"),
            "FAIL_CLOSED:discovery_trace:budget_field_missing:max_wall_clock_seconds",
        ),
        (
            lambda budget: budget.update({"max_wall_clock_seconds": 0}),
            "FAIL_CLOSED:discovery_trace:budget_invalid",
        ),
        (
            lambda budget: budget.update({"max_wall_clock_seconds": 121}),
            "FAIL_CLOSED:discovery_trace:budget_wall_clock_exceeds_runner_limit",
        ),
    ],
)
def test_v4_discovery_fails_closed_for_unusable_native_timeout_budget(
    tmp_path: Path, mutate_budget, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    budget = manifest["trace_budget"]
    assert isinstance(budget, dict)
    mutate_budget(budget)
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


def test_v4_native_original_execution_uses_exact_manifest_timeout_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    _write_json(manifest_path, manifest)
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "qualification"
    output = tmp_path / "replay.json"
    binary = tmp_path / "godot"
    qualification = object()
    observed: dict[str, object] = {}

    def fake_preflight(*args: object) -> dict[str, object]:
        observed["preflight"] = args
        return {"status": "PRECHECKED_NOT_SCOREABLE"}

    def fake_materialize(project: Path, staged: Path, **kwargs: object) -> object:
        observed["materialize"] = (project, staged, kwargs)
        return qualification

    def fake_classify(
        supplied: object,
        replay_output: Path,
        *,
        godot_binary: Path | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, object]:
        observed["classify"] = (supplied, replay_output, godot_binary, timeout_seconds)
        return {"schema": "flashpatch-godot-native-main-capture-v1"}

    monkeypatch.setattr(l7_verify, "verify_v4_candidate_source_preflight", fake_preflight)
    monkeypatch.setattr(l7_verify, "materialize_native_main_capture_qualification", fake_materialize)
    monkeypatch.setattr(l7_verify, "classify_native_main_capture_qualification", fake_classify)

    result = execute_v4_candidate_native_main_original(
        manifest_path,
        "natural-qualified",
        source,
        destination,
        output,
        binary,
    )

    template = manifest["trace_templates"][0]
    assert isinstance(template, dict)
    assert result["schema"] == "flashpatch-godot-native-main-capture-v1"
    assert observed["preflight"] == (manifest_path, "natural-qualified", source, binary)
    assert observed["materialize"] == (
        source / template["project_subpath"],
        destination,
        {
            "fixed_fps": template["fixed_fps"],
            "capture_frames": template["capture_frames"],
            "actions": template["actions"],
            "launch_arguments": template["launch_arguments"],
            "pointer_events": template["pointer_events"],
            "key_events": template["key_events"],
            "scenario_readiness": template["scenario_readiness"],
            "runtime_observations": template["runtime_observations"],
            "scene_transition": None,
        },
    )
    assert observed["classify"] == (qualification, output, binary, 120)


def test_v5_native_original_execution_passes_selection_contract_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v5(manifest)
    template = manifest["trace_templates"][0]
    readiness = template["scenario_readiness"]
    readiness["required_option_selection"] = [{
        "node_path": "/root/Main",
        "selected_index": 11,
        "selected_text": "FX: OldFilm",
    }]
    template["ui_selection_observations"] = ["/root/Main"]
    _write_json(manifest_path, manifest)
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "qualification"
    output = tmp_path / "replay.json"
    binary = tmp_path / "godot"
    qualification = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        l7_verify,
        "verify_v4_candidate_source_preflight",
        lambda *args: {"status": "PRECHECKED_NOT_SCOREABLE"},
    )

    def fake_materialize(project: Path, staged: Path, **kwargs: object) -> object:
        observed["materialize"] = (project, staged, kwargs)
        return qualification

    monkeypatch.setattr(
        l7_verify, "materialize_native_main_capture_qualification", fake_materialize
    )
    monkeypatch.setattr(
        l7_verify,
        "classify_native_main_capture_qualification",
        lambda *args, **kwargs: {"schema": "flashpatch-godot-native-main-capture-v1"},
    )

    execute_v4_candidate_native_main_original(
        manifest_path,
        "natural-qualified",
        source,
        destination,
        output,
        binary,
    )

    materialized = observed["materialize"]
    assert isinstance(materialized, tuple)
    options = materialized[2]
    assert options["scenario_readiness"] == readiness
    assert options["ui_selection_observations"] == ["/root/Main"]


def test_committed_v4_manifest_is_an_order_preserving_candidate_blind_v3_subset() -> None:
    v3 = json.loads(
        (ROOT / "evidence" / "l7" / "corpus-discovery" / "corpus-discovery-v3.json").read_text(
            encoding="utf-8"
        )
    )
    v4 = json.loads(V4_DISCOVERY.read_text(encoding="utf-8"))
    v3_by_id = {candidate["candidate_id"]: candidate for candidate in v3["candidates"]}
    v4_ids = v4["selection_order"]

    assert v4_ids == v3["selection_order"][:3]
    assert [candidate["candidate_id"] for candidate in v4["candidates"]] == v4_ids
    for candidate in v4["candidates"]:
        parent = v3_by_id[candidate["candidate_id"]]
        assert {
            key: candidate[key]
            for key in ("repository_url", "revision", "license", "godot_version")
        } == {
            key: parent[key]
            for key in ("repository_url", "revision", "license", "godot_version")
        }
    assert not {"detector", "gold", "scores", "ranking", "winner"}.intersection(v4)

    result = verify_corpus_discovery_manifest(V4_DISCOVERY)
    assert result["status"] == "NOT_SCOREABLE"
    assert result["candidate_count"] == 3
    assert result["qualified_case_count"] == 0
    assert result["external_claim_authorized"] is False


def test_committed_v5_manifest_replaces_v4_without_reordering_or_scoring() -> None:
    v4 = json.loads(V4_DISCOVERY.read_text(encoding="utf-8"))
    v5 = json.loads(V5_DISCOVERY.read_text(encoding="utf-8"))

    assert _sha256(V4_DISCOVERY) == V4_DISCOVERY_SHA256
    assert v5["selection_order"] == v4["selection_order"]
    assert v5["candidates"] == v4["candidates"]
    assert v5["stop_rule"] == v4["stop_rule"]
    assert v5["replacement_policy"] == v4["replacement_policy"]
    assert not {"detector", "gold", "scores", "ranking", "winner"}.intersection(v5)

    old_film = v5["trace_templates"][0]
    assert old_film["scenario_readiness"] == {
        "required_node_paths": [
            "/root/ScreenShaders/Effect",
            "/root/ScreenShaders/Effects/OldFilm",
        ],
        "required_group_minimums": {},
        "required_visible": [{
            "node_path": "/root/ScreenShaders/Effects/OldFilm",
            "visible": True,
        }],
        "required_option_selection": [{
            "node_path": "/root/ScreenShaders/Effect",
            "selected_index": 11,
            "selected_text": "FX: OldFilm",
        }],
    }
    assert old_film["ui_selection_observations"] == ["/root/ScreenShaders/Effect"]
    assert old_film["pointer_events"] == [{
        "frame": 10,
        "kind": "left_click",
        "x": 0.334375,
        "y": 0.034722222222222224,
    }]
    assert old_film["key_events"] == [
        *({"frame": frame, "key": "down"} for frame in range(12, 24)),
        {"frame": 25, "key": "enter"},
    ]

    result = verify_corpus_discovery_manifest(V5_DISCOVERY)
    assert result["status"] == "NOT_SCOREABLE"
    assert result["candidate_count"] == 3
    assert result["qualified_case_count"] == 0
    assert result["external_claim_authorized"] is False


def test_committed_old_film_preflight_is_manifest_bound_without_renderer_execution() -> None:
    receipt = json.loads(V4_OLD_FILM_PREFLIGHT.read_text(encoding="utf-8"))

    assert receipt["manifest_sha256"] == _sha256(V4_DISCOVERY)
    assert receipt["candidate_id"] == "godotdemo-old-film"
    assert receipt["status"] == "PRECHECKED_NOT_SCOREABLE"
    assert receipt["scoreable"] is False
    assert receipt["renderer_executed"] is False
    assert receipt["qualified_case_count"] == 0
    assert receipt["external_claim_authorized"] is False


def test_committed_v5_old_film_preflight_is_clean_pin_bound_without_renderer_execution() -> None:
    receipt = json.loads(V5_OLD_FILM_PREFLIGHT.read_text(encoding="utf-8"))

    assert receipt["manifest_sha256"] == _sha256(V5_DISCOVERY)
    assert receipt["candidate_id"] == "godotdemo-old-film"
    assert receipt["status"] == "PRECHECKED_NOT_SCOREABLE"
    assert receipt["scoreable"] is False
    assert receipt["renderer_executed"] is False
    assert receipt["qualified_case_count"] == 0
    assert receipt["external_claim_authorized"] is False


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda template: template.pop("key_events"),
            "FAIL_CLOSED:discovery_trace:template_0_field_missing:key_events",
        ),
        (
            lambda template: template.update({"undeclared_trace_input": []}),
            "FAIL_CLOSED:discovery_trace:template_0_field_forbidden:undeclared_trace_input",
        ),
    ],
)
def test_v4_discovery_rejects_missing_or_extra_trace_keys(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    template = manifest["trace_templates"][0]  # type: ignore[index]
    assert isinstance(template, dict)
    mutate(template)
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


@pytest.mark.parametrize(
    "key_events",
    [
        [{"frame": 2, "key": "left"}],
        [{"frame": 120, "key": "enter"}],
        [{"frame": 2, "key": "down", "pressed": True}],
    ],
)
def test_v4_discovery_rejects_invalid_native_main_key_events(
    tmp_path: Path, key_events: list[dict[str, object]]
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    manifest["trace_templates"][0]["key_events"] = key_events  # type: ignore[index]
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:discovery_trace:native_main_key_event"):
        verify_corpus_discovery_manifest(manifest_path)


def test_v4_empty_group_minimums_are_valid_only_inside_declared_readiness(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    _write_json(manifest_path, manifest)
    assert verify_corpus_discovery_manifest(manifest_path)["candidate_count"] == 3

    manifest["trace_templates"][0]["scenario_readiness"].pop("required_group_minimums")  # type: ignore[index]
    _write_json(manifest_path, manifest)
    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_trace:native_main_readiness_field_missing:required_group_minimums",
    ):
        verify_corpus_discovery_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda manifest: manifest["trace_templates"][0].update({"runner_id": "other"}),  # type: ignore[index,union-attr]
            "FAIL_CLOSED:discovery_trace:runner_unknown:trace-natural-qualified",
        ),
        (
            lambda manifest: manifest["trace_templates"][0].update({"actions": [{"frame": 120, "action": "jump", "pressed": True}]}),  # type: ignore[index,union-attr]
            "FAIL_CLOSED:discovery_trace:native_main_action_invalid:trace-natural-qualified:0",
        ),
        (
            lambda manifest: manifest["trace_templates"][0].update({"launch_arguments": ["--trace=override"]}),  # type: ignore[index,union-attr]
            "FAIL_CLOSED:discovery_trace:native_main_arguments_invalid:trace-natural-qualified",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-prospect").update({"godot_version": "3.5.3-stable"}),
            "FAIL_CLOSED:discovery_trace:candidate_runner_major_mismatch:natural-prospect",
        ),
    ],
)
def test_v4_discovery_rejects_runner_trace_seams(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    mutate(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


def test_v5_discovery_binds_native_main_transition_contract(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v5(manifest)
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        template["scene_transition"] = {
            "from_scene": "res://main.tscn", "to_scene": "res://game.tscn",
            "earliest_frame": 1, "latest_frame": 3,
        }
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["candidate_count"] == 3


def test_v5_discovery_rejects_transition_from_nonmain_scene(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v5(manifest)
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        template["scene_transition"] = {
            "from_scene": "res://other.tscn", "to_scene": "res://game.tscn",
            "earliest_frame": 1, "latest_frame": 3,
        }
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:discovery_trace:native_main_transition_invalid"):
        verify_corpus_discovery_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda template: template["scenario_readiness"].pop("required_option_selection"),
            "FAIL_CLOSED:discovery_trace:native_main_readiness_field_missing:required_option_selection",
        ),
        (
            lambda template: template["scenario_readiness"]["required_option_selection"][0].update(
                {"selected_label": "FX: OldFilm"}
            ),
            "FAIL_CLOSED:discovery_trace:native_main_selection_field_forbidden:selected_label",
        ),
        (
            lambda template: template.pop("ui_selection_observations"),
            "FAIL_CLOSED:discovery_trace:template_0_field_missing:ui_selection_observations",
        ),
        (
            lambda template: template.update({"selection_signal": "/root/ScreenShaders/Effect"}),
            "FAIL_CLOSED:discovery_trace:template_0_field_forbidden:selection_signal",
        ),
    ],
)
def test_v5_discovery_rejects_missing_or_extra_selection_fields(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    manifest = json.loads(V5_DISCOVERY.read_text(encoding="utf-8"))
    template = manifest["trace_templates"][0]
    mutate(template)
    manifest_path = _write_json(tmp_path / "v5.json", manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("selected_index", 10), ("selected_text", "FX: Sepia")],
)
def test_v5_discovery_rejects_wrong_old_film_selection_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = json.loads(V5_DISCOVERY.read_text(encoding="utf-8"))
    selection = manifest["trace_templates"][0]["scenario_readiness"][
        "required_option_selection"
    ][0]
    selection[field] = value
    manifest_path = _write_json(tmp_path / "v5.json", manifest)

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_trace:native_main_selection_contract_mismatch",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_v5_discovery_rejects_missing_selection_signal_observation(tmp_path: Path) -> None:
    manifest = json.loads(V5_DISCOVERY.read_text(encoding="utf-8"))
    manifest["trace_templates"][0]["ui_selection_observations"] = []
    manifest_path = _write_json(tmp_path / "v5.json", manifest)

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_trace:native_main_selection_signal_invalid",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_v5_discovery_rejects_unproved_post_selection_visibility(tmp_path: Path) -> None:
    manifest = json.loads(V5_DISCOVERY.read_text(encoding="utf-8"))
    manifest["trace_templates"][0]["scenario_readiness"]["required_visible"].append({
        "node_path": "/root/ScreenShaders",
        "visible": True,
    })
    manifest_path = _write_json(tmp_path / "v5.json", manifest)

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_trace:native_main_post_selection_visibility_mismatch",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_v4_source_preflight_binds_clean_pin_and_configured_main_scene(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "project.godot").write_text(
        'config_version=5\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8"
    )
    (source / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "FlashPatch Test"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture"],
        ["git", "remote", "add", "origin", "https://github.com/example/source"],
    ):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    candidate = _candidate(manifest, "natural-qualified")
    candidate["repository_url"] = "https://github.com/example/source"
    candidate["revision"] = revision
    candidate["license_sha256"] = _sha256(source / "LICENSE")
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\nprintf '%s\\n' '4.7.stable.official.test'\n", encoding="utf-8")
    godot.chmod(0o755)
    manifest["godot_binary"] = {"sha256": _sha256(godot), "version": "4.7.stable.official.test"}
    _write_json(manifest_path, manifest)

    result = verify_v4_candidate_source_preflight(
        manifest_path, "natural-qualified", source, godot
    )

    assert result["status"] == "PRECHECKED_NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["original_main_scene"] == "res://main.tscn"
    assert result["godot_binary_sha256"] == _sha256(godot)
    assert result["renderer_executed"] is False
    assert result["qualified_case_count"] == 0


def test_v4_source_preflight_rejects_manifest_main_scene_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "project.godot").write_text(
        'config_version=5\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8"
    )
    (source / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "FlashPatch Test"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture"],
        ["git", "remote", "add", "origin", "https://github.com/example/source"],
    ):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    candidate = _candidate(manifest, "natural-qualified")
    candidate["repository_url"] = "https://github.com/example/source"
    candidate["revision"] = revision
    candidate["license_sha256"] = _sha256(source / "LICENSE")
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\nprintf '%s\\n' '4.7.stable.official.test'\n", encoding="utf-8")
    godot.chmod(0o755)
    manifest["godot_binary"] = {"sha256": _sha256(godot), "version": "4.7.stable.official.test"}
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    templates[0]["original_main_scene"] = "res://other.tscn"
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match="FAIL_CLOSED:source_preflight:main_scene_mismatch"):
        verify_v4_candidate_source_preflight(manifest_path, "natural-qualified", source, godot)


@pytest.mark.parametrize("source_drift", ["revision", "origin", "dirty"])
def test_v4_source_preflight_rejects_stale_or_mismatched_source(
    tmp_path: Path, source_drift: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "project.godot").write_text(
        'config_version=5\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8"
    )
    (source / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "FlashPatch Test"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture"],
        ["git", "remote", "add", "origin", "https://github.com/example/source"],
    ):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v4(manifest)
    candidate = _candidate(manifest, "natural-qualified")
    candidate["repository_url"] = "https://github.com/example/source"
    candidate["revision"] = revision
    candidate["license_sha256"] = _sha256(source / "LICENSE")
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\nprintf '%s\\n' '4.7.stable.official.test'\n", encoding="utf-8")
    godot.chmod(0o755)
    manifest["godot_binary"] = {"sha256": _sha256(godot), "version": "4.7.stable.official.test"}
    _write_json(manifest_path, manifest)

    if source_drift == "revision":
        candidate["revision"] = "f" * 40
        _write_json(manifest_path, manifest)
    elif source_drift == "origin":
        subprocess.run(
            ["git", "remote", "set-url", "origin", "https://github.com/example/other"],
            cwd=source,
            check=True,
        )
    else:
        (source / "untracked.txt").write_text("drift\n", encoding="utf-8")

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:source_preflight:clean_pinned_source_mismatch",
    ):
        verify_v4_candidate_source_preflight(manifest_path, "natural-qualified", source, godot)


def test_v5_source_preflight_hashes_declared_transition_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "project.godot").write_text('config_version=5\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (source / "main.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    (source / "game.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    for command in (["git", "init"], ["git", "config", "user.email", "test@example.invalid"], ["git", "config", "user.name", "FlashPatch Test"], ["git", "add", "."], ["git", "commit", "-m", "fixture"], ["git", "remote", "add", "origin", "https://github.com/example/source"]):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    manifest_path, manifest = _discovery_manifest(tmp_path)
    _upgrade_to_v5(manifest)
    candidate = _candidate(manifest, "natural-qualified")
    candidate["repository_url"] = "https://github.com/example/source"
    candidate["revision"] = revision
    candidate["license_sha256"] = _sha256(source / "LICENSE")
    godot = tmp_path / "Godot"
    godot.write_text("#!/bin/sh\nprintf '%s\\n' '4.7.stable.official.test'\n", encoding="utf-8")
    godot.chmod(0o755)
    manifest["godot_binary"] = {"sha256": _sha256(godot), "version": "4.7.stable.official.test"}
    templates = manifest["trace_templates"]
    assert isinstance(templates, list)
    for template in templates:
        assert isinstance(template, dict)
        template["scene_transition"] = {"from_scene": "res://main.tscn", "to_scene": "res://game.tscn", "earliest_frame": 1, "latest_frame": 3}
    _write_json(manifest_path, manifest)

    result = verify_v4_candidate_source_preflight(
        manifest_path, "natural-qualified", source, godot
    )

    assert result["transition_target_sha256"].startswith("sha256:")


def test_discovery_rejects_duplicate_verified_case_id(tmp_path: Path) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    qualified_copy = copy.deepcopy(_candidate(manifest, "natural-qualified"))
    qualified_copy["candidate_id"] = "natural-qualified-copy"
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    candidates.append(qualified_copy)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_candidates:duplicate_case_id:natural-qualified-copy",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_discovery_with_fewer_than_three_repositories_cannot_freeze_corpus(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    manifest["candidates"] = candidates[:2]
    _write_json(manifest_path, manifest)

    result = verify_corpus_discovery_manifest(manifest_path)

    assert result["status"] == "NOT_SCOREABLE"
    assert result["corpus_frozen"] is False
    assert result["reason"] == "minimum_three_distinct_repositories_not_met"
    assert result["distinct_repository_count"] == 2
    assert result["qualified_case_count"] == 1


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda manifest: _append_repository_copy(
                manifest, candidate_id="natural-prospect"
            ),
            "FAIL_CLOSED:discovery_candidates:candidate_id_invalid_or_duplicate",
        ),
        (
            lambda manifest: _append_repository_copy(
                manifest,
                candidate_id="revision-drift-copy",
                revision="d" * 40,
            ),
            "FAIL_CLOSED:discovery_candidates:repository_revision_drift:revision-drift-copy",
        ),
        (
            lambda manifest: _append_repository_copy(
                manifest,
                candidate_id="repository-alias-copy",
                repository_url="https://github.com/example/second-godot-game.git/",
                revision="d" * 40,
            ),
            "FAIL_CLOSED:discovery_candidates:repository_revision_drift:repository-alias-copy",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-prospect").update(
                {"discovered_at_utc": "2026-07-01T00:00:01Z"}
            ),
            "FAIL_CLOSED:discovery_cutoff:candidate_after_cutoff:natural-prospect",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-prospect").update(
                {
                    "selection_outcome": "excluded",
                    "selection_reason": "Excluded because detector reported safe output",
                    "rule_ids": ["E-LICENSE"],
                }
            ),
            "FAIL_CLOSED:discovery_claim:forbidden_text:candidate.natural-prospect.selection_reason:detector",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-prospect").update(
                {"controlled_mutation": True}
            ),
            "FAIL_CLOSED:discovery_candidates:controlled_or_non_natural_case:natural-prospect",
        ),
        (
            lambda manifest: _candidate(manifest, "natural-excluded").pop("selection_reason"),
            "FAIL_CLOSED:discovery_candidates:candidate_2_field_missing:selection_reason",
        ),
        (
            lambda manifest: manifest.update({"scores": {"natural-qualified": 1.0}}),
            "FAIL_CLOSED:discovery_claim:forbidden_field:scores",
        ),
    ],
)
def test_discovery_adversaries_fail_closed(
    tmp_path: Path, mutate, diagnostic: str
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    mutate(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(L7VerificationFailure, match=re.escape(diagnostic)):
        verify_corpus_discovery_manifest(manifest_path)
    assessed = assess_corpus_discovery_manifest(manifest_path)
    assert assessed == {
        "schema": "flashpatch-l7-corpus-discovery-assessment-v1",
        "status": "INCONCLUSIVE",
        "scoreable": False,
        "corpus_frozen": False,
        "diagnostic": diagnostic,
        "external_claim_authorized": False,
    }


def test_qualified_discovery_candidate_must_repass_natural_bundle_verifier(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _discovery_manifest(tmp_path)
    (tmp_path / "qualified-natural-case" / "renderer.rgb").write_bytes(b"tampered")

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:renderer:rgb_hash_mismatch",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_qualified_candidate_godot_version_must_match_renderer_receipt(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _discovery_manifest(tmp_path)
    qualified = _candidate(manifest, "natural-qualified")
    qualified["godot_version"] = "4.4-stable"
    bundle_root = tmp_path / "qualified-natural-case"
    ledger = _ledger(bundle_root)
    ledger["godot_version"] = "4.4-stable"
    _save_ledger(bundle_root, ledger)
    qualified["natural_bundle_ledger_sha256"] = _sha256(bundle_root / "natural-case.json")
    _write_json(manifest_path, manifest)

    with pytest.raises(
        L7VerificationFailure,
        match="FAIL_CLOSED:discovery_bundle:candidate_godot_version_unbound",
    ):
        verify_corpus_discovery_manifest(manifest_path)


def test_discovery_cli_never_emits_pass_or_an_external_claim(tmp_path: Path) -> None:
    manifest_path, _ = _discovery_manifest(tmp_path)
    environment = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.l7_verify",
            "--discovery-manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "NOT_SCOREABLE"
    assert payload["external_claim_authorized"] is False
    assert "PASS" not in result.stdout
