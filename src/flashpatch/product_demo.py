from __future__ import annotations

import json
from pathlib import Path
import re

from .safety_ci import compile_project, write_receipt


_PROJECT_GODOT = """[application]
config/name="FlashPatch Installed Contract Demo"
run/main_scene="res://main.tscn"
"""

_SCENE = """[gd_scene load_steps=2 format=3]

[ext_resource path="res://main.gd" type="Script" id="1"]

[node name="InteractionBurst" type="Node2D"]
script = ExtResource("1")
"""

_SCRIPT = """extends Node2D

@export var burst_intensity: float = 1.0
"""

_HAZARDOUS_TRACE = {
    "fixed_fps": 60,
    "actions": [
        {"frame": 0, "charge": True},
        {"frame": 1},
        {"frame": 2, "fire": True},
        {"frame": 3},
    ],
}

_SAFE_TRACE = {
    "fixed_fps": 60,
    "actions": [
        {"frame": 0, "charge": True},
        {"frame": 1},
    ],
}

_EXPORTED_INTENSITY = re.compile(
    r"^@export var burst_intensity:\s*float\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
    re.MULTILINE,
)

_DEMONSTRATION = {
    "fixture": "installed-interaction-burst",
    "adapter": "deterministic in-process contract adapter",
    "evidence_scope": "deterministic_contract_fixture",
    "limitation": (
        "Exercises the Safety CI contract and source patch path; it is not rendered-pixel, "
        "private, benchmark, comparison, or clinical evidence."
    ),
}


class DemoReplayRunner:
    """Small public adapter used only by the installed product demonstration."""

    def __init__(self, project: Path | str) -> None:
        self.project = Path(project)

    def replay(self, trace: Path | str, output: Path | str) -> dict[str, object]:
        source = (self.project / "main.gd").read_text(encoding="utf-8")
        match = _EXPORTED_INTENSITY.search(source)
        if match is None:
            raise RuntimeError("demo fixture lost its one exported burst_intensity assignment")
        intensity = float(match.group(1))
        trace_payload = json.loads(Path(trace).read_text(encoding="utf-8"))
        action_frames: list[int] = []
        observations: list[float] = []
        charged = False
        for action in trace_payload["actions"]:
            action_frames.append(action["frame"])
            if action.get("charge") is True:
                charged = True
            risk = 0.0
            if charged and action.get("fire") is True:
                risk = intensity
                charged = False
            observations.append(risk)
        gameplay_state = f"actions:{len(action_frames)}|charged:{int(charged)}"
        replay = {
            "fixed_fps": trace_payload["fixed_fps"],
            "action_frames": action_frames,
            "gameplay_state": gameplay_state,
            "observations": observations,
            "status": "REPLAYED",
        }
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return replay


def _write_project(
    project: Path,
    *,
    trace: dict[str, object],
    replacement: float,
) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.godot").write_text(_PROJECT_GODOT, encoding="utf-8")
    (project / "main.tscn").write_text(_SCENE, encoding="utf-8")
    (project / "main.gd").write_text(_SCRIPT, encoding="utf-8")
    (project / "trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema": "flashpatch-godot-safety-ci-v1",
        "trace": "trace.json",
        "scene": "main.tscn",
        "timing_field": "action_frames",
        "state_field": "gameplay_state",
        "risk_signal": {
            "kind": "replay_observations_v1",
            "field": "observations",
            "threshold": 1.0,
        },
        "patch_candidates": [
            {
                "source": "main.gd",
                "parameter": "burst_intensity",
                "parameter_kind": "intensity",
                "replacement": replacement,
            }
        ],
    }
    contract_path = project / "flashpatch.contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract_path


def _run_case(
    name: str,
    project: Path,
    contract: Path,
    output: Path,
) -> dict[str, object]:
    receipt = compile_project(
        project,
        contract,
        workspace=output / f"{name}-work",
        runner_factory=DemoReplayRunner,
    )
    receipt["demonstration"] = dict(_DEMONSTRATION)
    write_receipt(receipt, output / f"{name}-receipt.json")
    return receipt


def _receipt_reference(path: Path) -> dict[str, str]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "receipt_sha256": receipt["receipt_sha256"]}


def run_safety_demo(output: Path | str) -> dict[str, object]:
    """Exercise all four terminal contract states without checkout-only evidence."""

    output_path = Path(output).resolve()
    fixture_root = output_path / "fixture"
    pass_project = fixture_root / "pass-project"
    safe_project = fixture_root / "safe-project"
    fail_project = fixture_root / "fail-project"
    pass_contract = _write_project(
        pass_project,
        trace=_HAZARDOUS_TRACE,
        replacement=0.0,
    )
    safe_contract = _write_project(
        safe_project,
        trace=_SAFE_TRACE,
        replacement=0.0,
    )
    fail_contract = _write_project(
        fail_project,
        trace=_HAZARDOUS_TRACE,
        replacement=2.0,
    )

    passed = _run_case("pass", pass_project, pass_contract, output_path)
    safe = _run_case("safe", safe_project, safe_contract, output_path)
    failed = _run_case("fail", fail_project, fail_contract, output_path)
    inconclusive = compile_project(
        pass_project,
        pass_project / "missing-contract.json",
        workspace=output_path / "inconclusive-work",
        runner_factory=DemoReplayRunner,
    )
    inconclusive["demonstration"] = dict(_DEMONSTRATION)
    write_receipt(inconclusive, output_path / "inconclusive-receipt.json")

    results = {
        "pass": passed["verdict"],
        "safe": safe["verdict"],
        "fail": failed["verdict"],
        "inconclusive": inconclusive["verdict"],
    }
    expected = {
        "pass": "PASS",
        "safe": "SAFE",
        "fail": "FAIL",
        "inconclusive": "INCONCLUSIVE",
    }
    if results != expected:
        raise RuntimeError(f"safety demo terminal states are invalid: {results}")

    factual_path = Path(str(passed["factual_replay"]["artifact"]))
    factual = json.loads(factual_path.read_text(encoding="utf-8"))
    threshold = float(passed["risk_signal"]["threshold"])
    hazardous_indices = [
        index
        for index, risk in enumerate(factual["observations"])
        if float(risk) >= threshold
    ]
    if len(hazardous_indices) != 1:
        raise RuntimeError("demo fixture must localize exactly one hazardous observation")
    hazard_index = hazardous_indices[0]
    attribution = passed["attribution"]
    report: dict[str, object] = {
        "schema": "flashpatch-safety-demo-v2",
        "fixture": dict(_DEMONSTRATION),
        "flow": {
            "input": {
                "project": str(pass_project),
                "trace": str(pass_project / "trace.json"),
                "action_frames": factual["action_frames"],
                "risk_signal": "developer-declared deterministic replay signal",
            },
            "hazard_localization": {
                "observation_index": hazard_index,
                "action_frame": factual["action_frames"][hazard_index],
                "risk": float(factual["observations"][hazard_index]),
                "threshold": threshold,
            },
            "allowed_patch": {
                "source": "main.gd",
                "source_line": attribution["source_line"],
                "parameter": attribution["parameter"],
                "replacement": attribution["replacement"],
                "changed_source_assignments": attribution["changed_source_assignments"],
                "diff": attribution["diff"],
            },
            "same_trace_revalidation": {
                "factual_max_risk": float(passed["factual_replay"]["max_risk"]),
                "patched_max_risk": float(attribution["max_risk"]),
                "timing_preserved": attribution["timing_preserved"],
                "gameplay_state_preserved": attribution["gameplay_state_preserved"],
            },
        },
        "results": results,
        "receipts": {
            name: _receipt_reference(output_path / f"{name}-receipt.json")
            for name in expected
        },
        "output": str(output_path),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "demo-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def format_safety_demo(report: dict[str, object]) -> str:
    flow = report["flow"]
    input_flow = flow["input"]
    hazard = flow["hazard_localization"]
    patch = flow["allowed_patch"]
    revalidation = flow["same_trace_revalidation"]
    results = report["results"]
    return "\n".join(
        [
            "FlashPatch safety demo",
            "EVIDENCE   deterministic public contract fixture; not renderer or clinical evidence",
            f"INPUT      action frames {input_flow['action_frames']}",
            (
                "LOCALIZE   hazard at action frame "
                f"{hazard['action_frame']} (risk {hazard['risk']:g}, threshold {hazard['threshold']:g})"
            ),
            (
                f"PATCH      {patch['source']}:{patch['source_line']} {patch['parameter']} -> "
                f"{patch['replacement']:g}; {patch['changed_source_assignments']} allowlisted assignment"
            ),
            (
                "REVALIDATE same trace "
                f"{str(revalidation['timing_preserved']).lower()}, gameplay state "
                f"{str(revalidation['gameplay_state_preserved']).lower()}, risk "
                f"{revalidation['factual_max_risk']:g} -> {revalidation['patched_max_risk']:g}"
            ),
            (
                f"RECEIPTS   PASS={results['pass']} SAFE={results['safe']} "
                f"FAIL={results['fail']} INCONCLUSIVE={results['inconclusive']}"
            ),
            f"OUTPUT     {report['output']}",
        ]
    )
