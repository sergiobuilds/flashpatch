from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flashpatch.l6_authority import L6_PREFLIGHT_PINS
from flashpatch.l6_verify import (
    ADVERSARIAL_MANIFEST_SCHEMA,
    CANONICAL_ADVERSARIAL_MANIFEST,
    VerificationFailure,
    _EXPECTED_ADVERSARIAL,
    _adversarial_result,
    _canonical_json,
    _sha256_bytes,
    _tree_sha256,
    _verify_engine_contract_gates,
    main,
    materialize_adversarial_manifest,
)


def _timestamps() -> tuple[list[int], list[int]]:
    actual = list(range(1, L6_PREFLIGHT_PINS.capture_ticks + 1))
    presentation = [
        (index * 1_000_000) // L6_PREFLIGHT_PINS.fixed_fps
        for index in range(L6_PREFLIGHT_PINS.capture_ticks)
    ]
    return actual, presentation


def _events(*, value: float, script_hash: str) -> list[dict[str, object]]:
    actual, _ = _timestamps()
    return [
        {
            "actual_capture_timestamp_us": actual[index],
            "factual_value": value,
            "frame_index": index,
            "node_path": "/root/DemoInputRecorder/Battle/@Node2D@476",
            "normalized_node_identity": (
                "res://scenes/Battle.tscn::res://scripts/RoutShockwave.gd::0"
            ),
            "property": "flashpatch_intensity",
            "resource_path": "res://scenes/Battle.tscn",
            "resource_path_observation": "battle.scene_file_path",
            "resource_provenance": "packed_scene_state",
            "script_path": "res://scripts/RoutShockwave.gd",
            "script_path_observation": "node.get_script().resource_path",
            "script_sha256": script_hash,
            "source_line": 4,
            "source_line_observation": (
                "FileAccess.get_file_as_string(node.get_script().resource_path)"
            ),
            "spawned_ordinal": 0,
        }
        for index in range(L6_PREFLIGHT_PINS.capture_ticks)
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_json(payload))


def _valid_engine(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "run-001"
    root.mkdir()
    actual, presentation = _timestamps()
    factual_hash = "a" * 64
    candidate_hash = "b" * 64
    factual_replay = root / "factual-replay.json"
    candidate_replay = root / "candidate-replay.json"
    _write_json(
        factual_replay,
        {"runtime_events": _events(value=1.0, script_hash=factual_hash)},
    )
    _write_json(
        candidate_replay,
        {"runtime_events": _events(value=0.0, script_hash=candidate_hash)},
    )
    diff = root / "patch.diff"
    diff.write_text(
        "--- a/RoutShockwave.gd\n"
        "+++ b/RoutShockwave.gd\n"
        "-@export var flashpatch_intensity: float = 1.0\n"
        "+@export var flashpatch_intensity: float = 0.0\n",
        encoding="utf-8",
    )

    shared = {
        "action_acknowledgements": [{"frame": 0, "status": "APPLIED"}],
        "final_state_raw_sha256": "c" * 64,
        "final_state_sha256": "d" * 64,
        "renderer_capture": {
            "actual_capture_timestamps_us": actual,
            "presentation_timestamps_us": presentation,
        },
        "state_stream_record_count": 162,
        "state_stream_sha256": "e" * 64,
        "state_stream_tick_domain": [0, 161],
        "tick_domain": [0, 160],
        "timestamps_sha256": "f" * 64,
        "runtime_source_line": 4,
    }
    factual = {
        **copy.deepcopy(shared),
        "artifact": "factual-replay.json",
        "hazard_frame_indices": [1],
        "hazard_kinds": ["general_flash"],
        "hazardous": True,
        "max_risk": 1.0,
        "runtime_script_sha256": factual_hash,
    }
    candidate = {
        **copy.deepcopy(shared),
        "artifact": "candidate-replay.json",
        "changed_source_assignments": 1,
        "diff": "patch.diff",
        "hazard_frame_indices": [],
        "hazard_kinds": [],
        "hazardous": False,
        "max_risk": 0.0,
        "parameter": "flashpatch_intensity",
        "replacement": 0.0,
        "runtime_script_sha256": candidate_hash,
        "semantic_invariants_preserved": True,
        "source_line": 4,
        "status": "EVALUATED",
        "timing_preserved": True,
    }
    engine: dict[str, object] = {
        "attribution": copy.deepcopy(candidate),
        "candidates": [candidate],
        "contract_schema": "flashpatch-godot-safety-ci-v1",
        "controlled_mutation": True,
        "factual_replay": factual,
        "reason": "same_trace_counterfactual_removed_declared_risk",
        "schema": "flashpatch-renderer-engine-receipt-v1",
        "upstream": {
            "classification": "external_dynamic_effect_controlled_mutation",
            "license": "MIT",
            "project_path": ".",
            "repository_url": L6_PREFLIGHT_PINS.repository,
            "source_revision": L6_PREFLIGHT_PINS.revision,
            "upstream_defect": False,
        },
        "verdict": "PASS",
    }
    return root, engine


def _assert_failure(
    root: Path,
    engine: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(VerificationFailure) as captured:
        _verify_engine_contract_gates(root, engine)
    assert captured.value.diagnostic == expected


@pytest.mark.parametrize(
    "mutation",
    ["missing", "empty", "duplicate", "decreasing", "wrong_presentation"],
)
def test_n1_timestamp_subcases_fail_closed(tmp_path: Path, mutation: str) -> None:
    root, engine = _valid_engine(tmp_path)
    capture = engine["factual_replay"]["renderer_capture"]
    if mutation == "missing":
        capture.pop("actual_capture_timestamps_us")
    elif mutation == "empty":
        capture["actual_capture_timestamps_us"] = []
    elif mutation == "duplicate":
        capture["actual_capture_timestamps_us"][2] = capture[
            "actual_capture_timestamps_us"
        ][1]
    elif mutation == "decreasing":
        capture["actual_capture_timestamps_us"][2] = 1
    else:
        capture["presentation_timestamps_us"][1] += 1

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:renderer_capture:timestamps_invalid",
    )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("factual", "node_path", "/root"),
        ("factual", "normalized_node_identity", "wrong-node"),
        ("factual", "script_path", "res://wrong.gd"),
        ("factual", "resource_path", "res://wrong.tscn"),
        ("factual", "source_line", 3),
        ("factual", "property", "unused_intensity"),
        ("candidate", "normalized_node_identity", "candidate-mismatch"),
    ],
)
def test_n2_provenance_subcases_fail_closed(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
) -> None:
    root, engine = _valid_engine(tmp_path)
    replay_path = root / f"{target}-replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["runtime_events"][0][field] = value
    _write_json(replay_path, replay)

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:runtime_attribution:provenance_mismatch",
    )


@pytest.mark.parametrize("mutation", ["missing_artifact", "missing_events", "empty"])
def test_n3_unobserved_contributor_subcases_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    root, engine = _valid_engine(tmp_path)
    if mutation == "missing_artifact":
        engine["factual_replay"]["artifact"] = "missing.json"
    else:
        path = root / "factual-replay.json"
        payload = {} if mutation == "missing_events" else {"runtime_events": []}
        _write_json(path, payload)

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:runtime_attribution:contributor_not_observed",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_acknowledgements", [{"frame": 0, "status": "MISSING"}]),
        ("state_stream_sha256", "0" * 64),
        ("final_state_sha256", "1" * 64),
        ("tick_domain", [0, 159]),
        ("timestamps_sha256", "2" * 64),
    ],
)
def test_n4_preservation_subcases_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    root, engine = _valid_engine(tmp_path)
    engine["candidates"][0][field] = value
    engine["attribution"] = copy.deepcopy(engine["candidates"][0])

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:gameplay_preservation:invariant_mismatch",
    )


def test_n5_safe_factual_with_patch_fails_closed(tmp_path: Path) -> None:
    root, engine = _valid_engine(tmp_path)
    factual = engine["factual_replay"]
    factual["hazardous"] = False
    factual["max_risk"] = 0.0
    factual["hazard_frame_indices"] = []
    factual["hazard_kinds"] = []

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:patch_search:no_factual_hazard",
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("replacement", 1.0),
        ("parameter", "other"),
        ("changed_source_assignments", 2),
        ("hazardous", True),
        ("max_risk", 1.0),
        ("diff", "multi"),
        ("runtime_value", 1.0),
    ],
)
def test_n6_noncausal_patch_subcases_fail_closed(
    tmp_path: Path, mutation: str, value: object
) -> None:
    root, engine = _valid_engine(tmp_path)
    candidate = engine["candidates"][0]
    if mutation == "diff":
        (root / "patch.diff").write_text(
            "--- a/RoutShockwave.gd\n"
            "+++ b/RoutShockwave.gd\n"
            "-@export var flashpatch_intensity: float = 1.0\n"
            "+@export var flashpatch_intensity: float = 0.0\n"
            "+@export var other: float = 0.0\n",
            encoding="utf-8",
        )
    elif mutation == "runtime_value":
        path = root / "candidate-replay.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        replay["runtime_events"][0]["factual_value"] = value
        _write_json(path, replay)
    else:
        candidate[mutation] = value
        engine["attribution"] = copy.deepcopy(candidate)

    _assert_failure(
        root,
        engine,
        "FAIL_CLOSED:patch_validation:counterfactual_not_causal",
    )


@pytest.mark.parametrize("case_id", list(_EXPECTED_ADVERSARIAL))
def test_negative_cli_has_exact_stderr_and_never_writes_pass_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case_id: str,
) -> None:
    gate, reason = _EXPECTED_ADVERSARIAL[case_id]

    def rejected(_: Path) -> dict[str, object]:
        raise VerificationFailure(gate, reason)

    monkeypatch.setattr("flashpatch.l6_verify.verify_case", rejected)
    pass_receipt = tmp_path / "verification-pass.json"

    assert main(["--case", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"FAIL_CLOSED:{gate}:{reason}\n"
    assert not pass_receipt.exists()


def test_positive_cli_writes_only_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    result = {
        "case_sha256": "a" * 64,
        "schema": "flashpatch-l6-verification-v1",
        "verdict": "PASS",
    }
    monkeypatch.setattr("flashpatch.l6_verify.verify_case", lambda _: result)

    assert main(["--case", str(tmp_path)]) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == _canonical_json(result)
    assert captured.err == b""


def test_canonical_aggregate_replays_exactly_six_independent_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / CANONICAL_ADVERSARIAL_MANIFEST
    cases_parent = manifest_path.parent / "cases"
    cases_parent.mkdir(parents=True)
    records = []
    for case_id, (gate, reason) in _EXPECTED_ADVERSARIAL.items():
        fixture = cases_parent / case_id
        fixture.mkdir()
        _write_json(fixture / "attack.json", {"id": case_id})
        result = _adversarial_result(gate, reason)
        diagnostic = f"FAIL_CLOSED:{gate}:{reason}\n".encode()
        records.append(
            {
                "expected": {
                    "exit": 2,
                    "result_json": result,
                    "result_json_sha256": _sha256_bytes(_canonical_json(result)),
                    "stderr_sha256": _sha256_bytes(diagnostic),
                    "stdout_sha256": _sha256_bytes(b""),
                },
                "fixture": f"cases/{case_id}",
                "id": case_id,
                "pass_receipt": f"cases/{case_id}/verification-pass.json",
                "raw_input_sha256": _tree_sha256(fixture),
            }
        )
    manifest = {
        "cases": records,
        "schema": ADVERSARIAL_MANIFEST_SCHEMA,
        "source_positive_execution_receipt_sha256": "a" * 64,
    }
    _write_json(manifest_path, manifest)

    def reject_fixture(path: Path) -> dict[str, object]:
        gate, reason = _EXPECTED_ADVERSARIAL[path.name]
        raise VerificationFailure(gate, reason)

    monkeypatch.setattr("flashpatch.l6_verify.verify_case", reject_fixture)

    assert main(["--aggregate", str(CANONICAL_ADVERSARIAL_MANIFEST)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "PASS: verified 6 independent L6 adversarial cases\n"
    assert captured.err == ""
    assert not list(cases_parent.rglob("verification-pass.json"))


def test_aggregate_rejects_every_alternate_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    alternate = tmp_path / "alternate.json"
    _write_json(alternate, {})

    assert main(["--aggregate", str(alternate)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "FAIL_CLOSED:adversarial_aggregate:noncanonical_manifest_path\n"
    )


def test_materializer_copies_both_preflight_receipt_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    positive = tmp_path / "positive"
    (positive / "preflight").mkdir(parents=True)
    _write_json(positive / "execution-receipt.json", {"verdict": "PASS"})
    _write_json(positive / "execution-checkpoint.json", {"verdict": "PASS"})
    _write_json(positive / "preflight" / "preflight.json", {"verdict": "PASS"})
    _write_json(
        positive / "preflight" / "preflight-checkpoint.json",
        {"preflight_verdict": "PASS"},
    )
    actual, presentation = _timestamps()
    for index in range(1, 4):
        run = positive / f"run-{index:03d}"
        run.mkdir()
        events = _events(value=1.0, script_hash="a" * 64)
        _write_json(run / "factual.json", {"runtime_events": events})
        _write_json(run / "candidate.json", {"runtime_events": events})
        factual = {
            "artifact": "factual.json",
            "renderer_capture": {
                "actual_capture_timestamps_us": actual,
                "presentation_timestamps_us": presentation,
            },
        }
        candidate = {
            "artifact": "candidate.json",
            "action_acknowledgements": [{"frame": 0, "status": "APPLIED"}],
            "replacement": 0.0,
        }
        _write_json(
            run / "engine-receipt.json",
            {
                "attribution": candidate,
                "candidates": [candidate],
                "factual_replay": factual,
            },
        )

    monkeypatch.setattr(
        "flashpatch.l6_verify.verify_case",
        lambda _: {"case_sha256": "a" * 64},
    )
    manifest = materialize_adversarial_manifest(positive)

    assert manifest == tmp_path / CANONICAL_ADVERSARIAL_MANIFEST
    for case_id in _EXPECTED_ADVERSARIAL:
        copied = manifest.parent / "cases" / case_id / "preflight"
        assert (copied / "preflight.json").is_file()
        assert (copied / "preflight-checkpoint.json").is_file()
