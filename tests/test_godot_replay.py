from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

from flashpatch.godot import GodotNativeMainRendererReplayRunner, GodotRendererReplayRunner, GodotReplayRunner


FIXTURE = Path(__file__).parent / "fixtures" / "godot_replay"


def test_headless_replay_is_deterministic(tmp_path: Path) -> None:
    trace = {
        "fixed_fps": 60,
        "actions": [
            {"frame": 0, "move_x": 1},
            {"frame": 1, "move_x": 1},
            {"frame": 2, "move_x": -1},
            {"frame": 3, "move_x": 0},
        ],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    runner = GodotReplayRunner(project=FIXTURE)

    first = runner.replay(trace_path, tmp_path / "first.json")
    second = runner.replay(trace_path, tmp_path / "second.json")

    assert first == second
    assert first["fixed_fps"] == 60
    assert first["states"] == [
        {"frame": 0, "position_x": 2},
        {"frame": 1, "position_x": 4},
        {"frame": 2, "position_x": 2},
        {"frame": 3, "position_x": 2},
    ]
    assert first["status"] == "REPLAYED"


def test_renderer_replay_never_silently_falls_back_to_headless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"fixed_fps": 60, "actions": [{"frame": 0}]}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-headless renderer capture requires"):
        GodotRendererReplayRunner(project=FIXTURE).replay(trace, tmp_path / "renderer.json")


def test_runner_rejects_declared_godot_major_mismatch_before_replay(tmp_path: Path) -> None:
    project = tmp_path / "godot3-project"
    project.mkdir()
    (project / "project.godot").write_text("config_version=4\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="project declares Godot 3 .* runner binary is Godot 4"):
        GodotReplayRunner(project=project)


def test_native_main_import_removes_inherited_display_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["environment"] = kwargs.get("env")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr("flashpatch.godot.subprocess.run", fake_run)

    runner._prepare_project_import()

    assert observed["command"] == [str(runner.godot_binary), "--headless", "--path", str(runner.project), "--import"]
    assert isinstance(observed["environment"], dict)
    assert observed["environment"].get("DISPLAY") is None
    assert observed["environment"].get("WAYLAND_DISPLAY") is None


def test_native_main_import_revalidates_exact_godot_shutdown_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    calls: list[tuple[list[str], object]] = []
    shutdown_stdout = (
        'imports completed\nERROR: Parameter "singleton" is null.\n'
        "   at: is_cmdline_mode (editor/editor_node.cpp:6618)\n"
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs.get("env")))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, -signal.SIGABRT, shutdown_stdout)
        return subprocess.CompletedProcess(command, 0, "revalidated\n")

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.godot.subprocess.run", fake_run)

    runner._prepare_project_import()

    expected = [str(runner.godot_binary), "--headless", "--path", str(runner.project), "--import"]
    assert [command for command, _ in calls] == [expected, expected]
    assert calls[0][1] == calls[1][1]
    assert isinstance(calls[0][1], dict)
    assert calls[0][1].get("DISPLAY") is None


def test_native_main_import_does_not_retry_normal_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 2, "ordinary import failure\n")

    monkeypatch.setattr("flashpatch.godot.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="project import exited 2") as raised:
        runner._prepare_project_import()

    assert len(calls) == 1
    assert "ordinary import failure" in str(raised.value)


def test_native_main_import_requires_shutdown_abort_revalidation_to_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    calls: list[list[str]] = []
    results = iter([
        subprocess.CompletedProcess(
            [],
            -signal.SIGABRT,
            'ERROR: Parameter "singleton" is null.\n'
            "   at: is_cmdline_mode (editor/editor_node.cpp:6618)\n",
        ),
        subprocess.CompletedProcess([], 3, "second import still failed\n"),
    ])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        result = next(results)
        return subprocess.CompletedProcess(command, result.returncode, result.stdout)

    monkeypatch.setattr("flashpatch.godot.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="revalidation exited 3") as raised:
        runner._prepare_project_import()

    assert len(calls) == 2
    assert "Parameter \"singleton\" is null" in str(raised.value)
    assert "second import still failed" in str(raised.value)


def test_native_main_import_retry_does_not_mask_non_headless_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = tmp_path / "godot"
    runner.timeout_seconds = 7
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"fixed_fps": 60, "launch_arguments": []}), encoding="utf-8")
    output = tmp_path / "replay.json"
    calls: list[list[str]] = []
    results = iter([
        subprocess.CompletedProcess(
            [],
            -signal.SIGABRT,
            'ERROR: Parameter "singleton" is null.\n'
            "   at: is_cmdline_mode (editor/editor_node.cpp:6618)\n",
        ),
        subprocess.CompletedProcess([], 0, "revalidated\n"),
        subprocess.CompletedProcess([], 9, "native runtime failure\n"),
    ])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        result = next(results)
        return subprocess.CompletedProcess(command, result.returncode, result.stdout)

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr("flashpatch.godot.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="non-headless replay exited 9") as raised:
        runner.replay(trace, output)

    assert len(calls) == 3
    assert "native runtime failure" in str(raised.value)
    assert calls[2] == [
        str(runner.godot_binary), "--display-driver", "x11",
        "--fixed-fps", "60", "--disable-vsync",
        "--path", str(runner.project), "--",
        "--trace", str(trace.resolve()), "--output", str(output.resolve()), "--renderer-capture",
    ]
    assert "--headless" not in calls[2]


def test_native_main_runtime_timeout_reaps_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_file = tmp_path / "runtime.pid"
    godot = tmp_path / "slow-godot"
    godot.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    godot.chmod(0o755)
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"fixed_fps": 60, "launch_arguments": []}), encoding="utf-8")
    runner = object.__new__(GodotNativeMainRendererReplayRunner)
    runner.project = tmp_path
    runner.godot_binary = godot
    runner.timeout_seconds = 1
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(runner, "_prepare_project_import", lambda: None)

    with pytest.raises(subprocess.TimeoutExpired):
        runner.replay(trace, tmp_path / "replay.json")

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
