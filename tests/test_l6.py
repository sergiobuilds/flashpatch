from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashpatch.l6_authority import L6_PREFLIGHT_PINS, L6PreflightPins
from flashpatch.l6_run import (
    _parser,
    _preservation_evidence,
    _renderer_color_space_gap,
    run_positive,
    run_preflight,
)


EXPECTED_CHECKS = {
    "upstream_path",
    "git_top_level",
    "git_origin",
    "git_revision",
    "git_clean_tree",
    "license_sha256",
    "required_project_inputs",
    "entry_scene",
    "symlink_free_inputs",
    "godot_canonical_path",
    "godot_regular_executable",
    "godot_sha256",
    "godot_version",
}


def _command(*arguments: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(arguments), cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _preflight_fixture(tmp_path: Path) -> tuple[Path, Path, L6PreflightPins]:
    upstream = tmp_path / "sparta"
    files = {
        "LICENSE": "MIT fixture license\n",
        "project.godot": '[application]\nrun/main_scene="res://scenes/MainMenu.tscn"\n',
        "tools/demo/DemoInputRecorder.tscn": "[gd_scene format=3]\n",
        "tools/demo/DemoInputRecorder.gd": "extends Node\n",
        "scripts/RoutShockwave.gd": "extends Node2D\n",
        "scenes/Battle.tscn": "[gd_scene format=3]\n",
    }
    for relative, content in files.items():
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _command("git", "init", "-q", cwd=upstream)
    _command("git", "config", "user.name", "L6 Test", cwd=upstream)
    _command("git", "config", "user.email", "l6@example.invalid", cwd=upstream)
    _command(
        "git",
        "remote",
        "add",
        "origin",
        "https://github.com/Lacaedemon/sparta",
        cwd=upstream,
    )
    _command("git", "add", ".", cwd=upstream)
    _command("git", "commit", "-qm", "fixture", cwd=upstream)
    revision = _command("git", "rev-parse", "HEAD", cwd=upstream)

    godot = tmp_path / "Godot_v4.7-stable_linux.x86_64"
    godot.write_text(
        "#!/bin/sh\nprintf '%s\\n' '4.7.stable.official.5b4e0cb0f'\n",
        encoding="utf-8",
    )
    godot.chmod(0o755)
    pins = L6PreflightPins(
        repository="https://github.com/Lacaedemon/sparta",
        revision=revision,
        license_sha256=hashlib.sha256(
            (upstream / "LICENSE").read_bytes()
        ).hexdigest(),
        local_checkout=str(upstream),
        entry_scene="res://tools/demo/DemoInputRecorder.tscn",
        required_inputs=(
            "project.godot",
            "tools/demo/DemoInputRecorder.tscn",
            "tools/demo/DemoInputRecorder.gd",
            "scripts/RoutShockwave.gd",
            "scenes/Battle.tscn",
        ),
        godot_binary=str(godot),
        godot_binary_sha256=hashlib.sha256(godot.read_bytes()).hexdigest(),
        godot_version="4.7.stable.official.5b4e0cb0f",
        xvfb_screen="1280x720x24",
        rendering_driver="opengl3",
        fixed_fps=60,
        capture_ticks=161,
        timeout_seconds=300,
    )
    return upstream, godot, pins


def test_preflight_measures_every_pin_before_allowing_replay(tmp_path: Path) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    run_root = tmp_path / "runs" / "execution-1"

    receipt = run_preflight(upstream, godot, run_root, pins=pins)

    assert receipt["preflight_verdict"] == "PASS"
    assert receipt["verdict"] == "PASS"
    assert receipt["upstream_product_verdict"] == "INCONCLUSIVE"
    assert receipt["controlled_mutation"] is False
    assert receipt["upstream_defect"] is False
    assert receipt["replay_allowed"] is True
    assert set(receipt["checks"]) == EXPECTED_CHECKS
    assert {item["status"] for item in receipt["checks"].values()} == {"PASS"}
    assert receipt["replay_contract"] == {
        "entry_scene": "res://tools/demo/DemoInputRecorder.tscn",
        "display": ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"],
        "godot_arguments": [
            "--rendering-driver",
            "opengl3",
            "--fixed-fps",
            "60",
        ],
        "capture_ticks": 161,
        "timeout_seconds": 300,
    }

    receipt_bytes = (run_root / "preflight.json").read_bytes()
    assert json.loads(receipt_bytes) == receipt
    assert receipt_bytes == (
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    checkpoint = json.loads(
        (run_root / "preflight-checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["preflight_receipt_sha256"] == hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    assert checkpoint["preflight_verdict"] == "PASS"


def test_preflight_accepts_equivalent_git_clone_suffix(tmp_path: Path) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    _command(
        "git",
        "remote",
        "set-url",
        "origin",
        "https://github.com/Lacaedemon/sparta.git",
        cwd=upstream,
    )

    receipt = run_preflight(upstream, godot, tmp_path / "runs" / "git-suffix", pins=pins)

    assert receipt["preflight_verdict"] == "PASS"
    assert receipt["checks"]["git_origin"]["status"] == "PASS"


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("upstream_path", "upstream_path"),
        ("origin", "git_origin"),
        ("revision", "git_revision"),
        ("clean", "git_clean_tree"),
        ("license", "license_sha256"),
        ("required_input", "required_project_inputs"),
        ("entry_scene", "entry_scene"),
        ("symlink", "symlink_free_inputs"),
        ("godot_path", "godot_canonical_path"),
        ("godot_executable", "godot_regular_executable"),
        ("godot_hash", "godot_sha256"),
        ("godot_version", "godot_version"),
    ],
)
def test_preflight_fails_closed_for_each_pinned_input(
    tmp_path: Path, case: str, failed_check: str
) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    if case == "upstream_path":
        pins = replace(pins, local_checkout=str(upstream / "other"))
    elif case == "origin":
        _command(
            "git",
            "remote",
            "set-url",
            "origin",
            "https://example.invalid/sparta",
            cwd=upstream,
        )
    elif case == "revision":
        pins = replace(pins, revision="0" * 40)
    elif case == "clean":
        (upstream / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    elif case == "license":
        pins = replace(pins, license_sha256="0" * 64)
    elif case == "required_input":
        (upstream / "scenes" / "Battle.tscn").unlink()
    elif case == "entry_scene":
        pins = replace(pins, entry_scene="res://tools/demo/Missing.tscn")
    elif case == "symlink":
        (upstream / "ignored-link").symlink_to(tmp_path / "outside")
    elif case == "godot_path":
        pins = replace(pins, godot_binary=str(tmp_path / "other-godot"))
    elif case == "godot_executable":
        godot.chmod(0o644)
    elif case == "godot_hash":
        pins = replace(pins, godot_binary_sha256="0" * 64)
    elif case == "godot_version":
        pins = replace(pins, godot_version="4.6.stable.invalid")

    receipt = run_preflight(
        upstream, godot, tmp_path / "runs" / case, pins=pins
    )

    assert receipt["preflight_verdict"] == "INCONCLUSIVE"
    assert receipt["upstream_product_verdict"] == "INCONCLUSIVE"
    assert receipt["replay_allowed"] is False
    assert receipt["checks"][failed_check]["status"] == "INCONCLUSIVE"
    assert (tmp_path / "runs" / case / "preflight.json").is_file()
    assert (tmp_path / "runs" / case / "preflight-checkpoint.json").is_file()


def test_preflight_does_not_execute_untrusted_godot_binary(tmp_path: Path) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    marker = tmp_path / "godot-was-executed"
    godot.write_text(
        f"#!/bin/sh\nprintf ran > {marker}\nprintf '%s\\n' '{pins.godot_version}'\n",
        encoding="utf-8",
    )
    godot.chmod(0o755)

    receipt = run_preflight(
        upstream,
        godot,
        tmp_path / "runs" / "bad-hash",
        pins=pins,
    )

    assert receipt["checks"]["godot_sha256"]["status"] == "INCONCLUSIVE"
    assert receipt["checks"]["godot_version"]["observed"]["returncode"] is None
    assert not marker.exists()


def test_preflight_never_overwrites_an_existing_run_root(tmp_path: Path) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    run_root = tmp_path / "runs" / "existing"
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_preflight(upstream, godot, run_root, pins=pins)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in run_root.iterdir()) == ["sentinel"]


def test_positive_cli_parses_only_an_explicit_execution_mode(tmp_path: Path) -> None:
    arguments = _parser().parse_args(
        [
            "--upstream",
            str(tmp_path / "sparta"),
            "--godot",
            str(tmp_path / "Godot"),
            "--runs",
            "3",
            "--run-root",
            str(tmp_path / "run"),
        ]
    )

    assert arguments.preflight is False
    assert arguments.runs == 3

    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--preflight",
                "--runs",
                "3",
                "--upstream",
                str(tmp_path / "sparta"),
                "--godot",
                str(tmp_path / "Godot"),
                "--run-root",
                str(tmp_path / "run"),
            ]
        )


def test_positive_execution_reuses_preflight_and_never_starts_on_pin_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    pins = replace(pins, revision="0" * 40)
    called = False

    def forbidden_execute(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("renderer must not start after preflight failure")

    monkeypatch.setattr(
        "flashpatch.public_godot.execute_controlled_sparta", forbidden_execute
    )

    receipt = run_positive(
        upstream,
        godot,
        tmp_path / "positive",
        pins=pins,
    )

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["phase"] == "PREFLIGHT_INCONCLUSIVE"
    assert receipt["upstream_product_verdict"] == "INCONCLUSIVE"
    assert called is False
    assert (tmp_path / "positive" / "preflight" / "preflight.json").is_file()
    assert (tmp_path / "positive" / "execution-receipt.json").is_file()


def test_positive_execution_never_overwrites_an_existing_root(tmp_path: Path) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)
    root = tmp_path / "positive"
    root.mkdir()
    sentinel = root / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_positive(upstream, godot, root, pins=pins)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in root.iterdir()) == ["sentinel"]


def test_positive_execution_rejects_legacy_pass_without_runtime_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, godot, pins = _preflight_fixture(tmp_path)

    def incomplete_execute(
        upstream_project: Path,
        output_root: Path,
        *,
        godot_binary: Path,
    ) -> object:
        assert upstream_project == upstream
        assert godot_binary == godot
        output_root.mkdir(parents=True)
        receipt_path = output_root / "engine-receipt.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": "flashpatch-renderer-engine-receipt-v1",
                    "verdict": "PASS",
                    "controlled_mutation": True,
                    "upstream": {"upstream_defect": False},
                    "factual_replay": {},
                    "candidates": [{}],
                    "attribution": {},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(receipt_path=receipt_path)

    monkeypatch.setattr(
        "flashpatch.public_godot.execute_controlled_sparta", incomplete_execute
    )

    receipt = run_positive(
        upstream,
        godot,
        tmp_path / "positive",
        pins=pins,
    )

    assert receipt["verdict"] == "INCONCLUSIVE"
    assert receipt["phase"] == "RUNTIME_EVIDENCE_INCONCLUSIVE"
    assert receipt["runs_completed"] == 1
    assert receipt["attempts"][0]["verdict"] == "INCONCLUSIVE"
    assert {item["gate"] for item in receipt["evidence_gaps"]} >= {
        "renderer_capture",
        "runtime_attribution",
        "patch_validation",
        "gameplay_preservation",
        "risk",
    }
    assert not list((tmp_path / "positive").rglob("*.npz"))


def test_l6_preservation_requires_actual_action_state_tick_and_timeline_equality(
    tmp_path: Path,
) -> None:
    stream_bytes = "".join(
        json.dumps({"tick": tick, "cheap": f"hash-{tick}"}) + "\n"
        for tick in range(162)
    ).encode("utf-8")
    final_payload = {"tick": 160, "units": [{"id": 1}]}
    final_bytes = (json.dumps(final_payload, indent=2) + "\n").encode("utf-8")
    canonical_final = (
        json.dumps(final_payload, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")

    def evidence(label: str) -> dict[str, object]:
        root = tmp_path / label
        root.mkdir()
        stream = root / "hash_stream.jsonl"
        final_state = root / "state_00160.json"
        stream.write_bytes(stream_bytes)
        final_state.write_bytes(final_bytes)
        return {
            "state_stream_sha256": hashlib.sha256(stream_bytes).hexdigest(),
            "state_stream_artifact": str(stream),
            "state_stream_tick_domain": [0, 161],
            "state_stream_record_count": 162,
            "final_state_sha256": hashlib.sha256(canonical_final).hexdigest(),
            "final_state_raw_sha256": hashlib.sha256(final_bytes).hexdigest(),
            "final_state_artifact": str(final_state),
            "tick_domain": [0, 160],
            "timestamps_sha256": "timeline-hash",
            "action_acknowledgements": [{"frame": 0, "status": "APPLIED"}],
        }

    factual = evidence("factual")
    candidate = evidence("candidate")
    gaps: list[dict[str, str]] = []

    preserved = _preservation_evidence(tmp_path, factual, candidate, gaps)

    assert preserved is not None
    assert preserved["state_stream_sha256"] == factual["state_stream_sha256"]
    assert preserved["final_state_sha256"] == factual["final_state_sha256"]
    assert preserved["presentation_timestamps_sha256"] == factual[
        "timestamps_sha256"
    ]
    assert preserved["factual"]["action_acknowledgements"] == [
        {"frame": 0, "status": "APPLIED"}
    ]
    assert preserved["factual"]["state_stream_artifact_sha256"] == factual[
        "state_stream_sha256"
    ]
    bound_records = {
        "factual": preserved["factual"],
        "candidate": preserved["candidate"],
    }
    assert preserved["preservation_evidence_sha256"] == hashlib.sha256(
        (
            json.dumps(
                bound_records,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert gaps == []

    for missing in (
        "action_acknowledgements",
        "tick_domain",
        "timestamps_sha256",
        "state_stream_sha256",
        "state_stream_artifact",
        "state_stream_tick_domain",
        "state_stream_record_count",
        "final_state_sha256",
        "final_state_raw_sha256",
        "final_state_artifact",
    ):
        incomplete = dict(factual)
        incomplete.pop(missing)
        missing_gaps: list[dict[str, str]] = []
        assert _preservation_evidence(
            tmp_path, incomplete, candidate, missing_gaps
        ) is None
        assert missing_gaps == [
            {
                "gate": "gameplay_preservation",
                "reason": "explicit_preservation_fields_missing",
            }
        ]

    candidate = dict(candidate)
    candidate["final_state_sha256"] = "different"
    candidate["action_acknowledgements"] = [{"frame": 0, "status": "MISSING"}]
    gaps = []
    _preservation_evidence(tmp_path, factual, candidate, gaps)
    assert {item["reason"] for item in gaps} >= {
        "candidate_action_not_applied",
        "final_state_sha256_mismatch",
        "candidate_final_state_canonical_hash_mismatch",
    }


def test_l6_color_space_diagnostic_does_not_promote_rgb8_sdr_observations() -> None:
    capture = {
        "color_space": "UNSPECIFIED",
        "color_space_provenance": {
            "status": "INSUFFICIENT",
            "runtime_observations": {
                "viewport_use_hdr_2d": False,
                "viewport_use_hdr_2d_api": "get_viewport().use_hdr_2d",
                "image_format": "Image.FORMAT_RGB8",
                "image_format_api": "Image.get_format()",
            },
            "missing_metadata": ["transfer_function", "color_primaries"],
        },
    }

    assert _renderer_color_space_gap("factual", capture) == (
        "factual_color_space_unproven"
    )
    capture["color_space"] = "sRGB/BT.709"
    assert _renderer_color_space_gap("factual", capture) == (
        "factual_color_space_direct_provenance_missing"
    )


def test_l6_color_space_requires_the_pinned_runtime_engine_contract() -> None:
    capture = {
        "godot_version": L6_PREFLIGHT_PINS.godot_version,
        "color_space": "sRGB",
        "color_space_provenance": {
            "status": "ENGINE_CONTRACT_DERIVED",
            "profile": {
                "encoding": "sRGB",
                "color_primaries": "BT.709",
                "white_point": "D65",
            },
            "engine_contract": {
                "godot_revision": "5b4e0cb0f",
                "renderer_documentation": "https://docs.godotengine.org/en/4.7/engine_details/architecture/internal_rendering_architecture.html",
                "compatibility_claim": "OpenGL uses Compatibility; Compatibility colors are stored in sRGB with no HDR support",
                "scope": "renderer frame encoding only; no physical display, ICC, or Xvfb colorimetry claim",
            },
            "runtime_observations": {
                "viewport_use_hdr_2d": False,
                "viewport_use_hdr_2d_api": "get_viewport().use_hdr_2d",
                "image_format": "Image.FORMAT_RGB8",
                "image_format_api": "Image.get_format()",
                "display_server": "X11",
                "display_server_api": "DisplayServer.get_name()",
                "rendering_method": "gl_compatibility",
                "rendering_method_api": "RenderingServer.get_current_rendering_method()",
                "rendering_driver": "opengl3",
                "rendering_driver_api": "RenderingServer.get_current_rendering_driver_name()",
                "hdr_output_supported": False,
                "hdr_output_requested": False,
                "hdr_output_enabled": False,
                "hdr_output_api": "DisplayServer.window_is_hdr_output_supported/requested/enabled()",
            },
        },
    }

    assert _renderer_color_space_gap("factual", capture) is None
    capture["color_space_provenance"]["runtime_observations"]["hdr_output_enabled"] = True
    assert _renderer_color_space_gap("factual", capture) == (
        "factual_color_space_engine_contract_invalid"
    )
