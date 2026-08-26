from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import socket

import pytest

from flashpatch.l10_unity import (
    UnityHarnessError,
    install_unity_harness,
    prepare_unity_linux_execution_inputs,
    unity_adapter_fingerprints,
    unity_linux_package_manifest_bytes,
    write_unity_project_manifest,
)
from flashpatch.l10_unity_runner import (
    UNITY_IMAGE,
    UNITY_RUNS,
    UNITY_VULKAN_LOADER_SHA256,
    UnityL10RunError,
    _docker_command,
    _copy_frozen_project,
    _gpu_identity,
    _gpu_lease,
    _install_entitlement,
    _run_one,
    _snapshot_loader,
    _verify_manifest,
    run_unity_l10_matrix,
)


def _project(root: Path) -> Path:
    scene = root / "Assets/Samples/SmokePortal/SmokePortal.unity"
    script = root / "Assets/Samples/SmokePortal/SmokePortal/LightFlicker.cs"
    scene.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text("m_IntensityJitterScale: 2000\n")
    script.write_text("m_Light.intensity = m_InitialIntensity + Noise.x * m_IntensityJitterScale;\n")
    return root


def test_installs_adapter_without_modifying_frozen_source(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    scene = project / "Assets/Samples/SmokePortal/SmokePortal.unity"
    before = scene.read_bytes()
    receipt = install_unity_harness(project, tmp_path / "install.json")
    assert scene.read_bytes() == before
    assert receipt["source_files_modified"] == 0
    adapter = project / receipt["adapter_path"]
    assert adapter.is_file()
    script = adapter.read_text()
    assert "execution-marker.json" in script
    assert "png_set_sha256" in script
    assert "flashpatch-l10-unity-execution-marker-v2" in script
    assert "deterministic-step-v2" in script
    assert "adapter_sha256" in script
    assert "runtime_events_sha256" in script
    assert "state_stream_sha256" in script
    assert "SystemInfo.graphicsDeviceType" in script
    assert "SystemInfo.graphicsDeviceName" in script
    assert "target.m_IntensityJitterScale = 0.0f" not in script
    assert 'mode == "factual" ? 2000.0f : 0.0f' in script
    assert "markerJson" in script
    assert "VerifyProjectInputs" in script
    assert "EditorApplication.Step()" in script
    assert "StableSeed" in script
    assert 'GetPrivate<Vector3>(flicker, "m_InitialPosition")' in script
    assert "flicker.transform.SetPositionAndRotation(initialPosition, initialRotation)" in script
    assert "effect.resetSeedOnPlay = false" in script
    assert "effect.Simulate(FrameStep, 1)" in script
    assert "CinemachineBrain.UpdateMethod.ManualUpdate" in script
    assert "CinemachineCore.UniformDeltaTimeOverride = FrameStep" in script
    assert "CinemachineCore.CurrentTimeOverride = frame * FrameStep" in script
    assert "rendered_value = light.intensity" in script
    assert "DirectorUpdateMode.Manual" in script
    assert "director.time = frame * FrameStep" in script
    assert "virtualCamera.PreviousStateIsValid = false" in script
    assert "HDAdditionalCameraData.AntialiasingMode.None" in script
    assert adapter.with_suffix(adapter.suffix + ".meta").is_file()
    assert receipt["adapter_sha256"] == unity_adapter_fingerprints()[
        "Assets/Editor/FlashPatchL10Capture.cs"
    ]
    assert receipt["adapter_meta_sha256"] == unity_adapter_fingerprints()[
        "Assets/Editor/FlashPatchL10Capture.cs.meta"
    ]


def test_rejects_candidate_switch_after_source_is_ambiguous(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    scene = project / "Assets/Samples/SmokePortal/SmokePortal.unity"
    scene.write_text(scene.read_text() * 2)
    with pytest.raises(UnityHarnessError, match="ambiguous"):
        install_unity_harness(project, tmp_path / "install.json")


def test_rejects_stale_adapter(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    destination = project / "Assets/Editor/FlashPatchL10Capture.cs"
    destination.parent.mkdir(parents=True)
    destination.write_text("stale")
    with pytest.raises(UnityHarnessError, match="already exists"):
        install_unity_harness(project, tmp_path / "install.json")


def test_installs_counterfactual_adapter_only_on_exact_source_patch(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    scene = project / "Assets/Samples/SmokePortal/SmokePortal.unity"
    scene.write_text("m_IntensityJitterScale: 0\n")
    receipt = install_unity_harness(
        project,
        tmp_path / "install.json",
        role="counterfactual",
    )
    assert receipt["role"] == "counterfactual"


def test_rejects_counterfactual_role_on_factual_source(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    with pytest.raises(UnityHarnessError, match="missing or ambiguous"):
        install_unity_harness(
            project,
            tmp_path / "install.json",
            role="counterfactual",
        )


def test_freezes_every_project_input_file(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    (project / "Packages/manifest.json").write_text("{}\n")
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/ProjectVersion.txt").write_text("2022.3.8f1\n")
    install_unity_harness(project, tmp_path / "install.json")
    result = write_unity_project_manifest(project, tmp_path / "inputs.json")
    value = json.loads((tmp_path / "inputs.json").read_text())
    assert result["file_count"] == len(value["files"])
    assert {row["path"] for row in value["files"]} == {
        path.relative_to(project).as_posix()
        for folder in ("Assets", "Packages", "ProjectSettings")
        for path in (project / folder).rglob("*")
        if path.is_file()
    }


def test_adapter_fingerprint_detects_post_install_mutation(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    receipt = install_unity_harness(project, tmp_path / "install.json")
    adapter = project / receipt["adapter_path"]
    adapter.write_text(adapter.read_text() + "// mutated\n")
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() != unity_adapter_fingerprints()[
        "Assets/Editor/FlashPatchL10Capture.cs"
    ]


def test_pins_linux_execution_inputs_only_in_isolated_copy(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    manifest = project / "Packages/manifest.json"
    manifest.write_text(
        '{\n  "dependencies": {\n    "com.unity.toolchain.win-x86_64-linux-x86_64": "2.0.4"\n  }\n}\n'
    )
    receipt = prepare_unity_linux_execution_inputs(project)
    value = json.loads(manifest.read_text())
    assert value["dependencies"]["com.unity.toolchain.linux-x86_64"] == "2.0.11"
    assert (project / "Packages/packages-lock.json").is_file()
    assert (project / "Assets/Editor.meta").is_file()
    assert receipt["source_files_modified"] == 0
    assert receipt["execution_copy_files_added"] == 2
    fingerprints = unity_adapter_fingerprints()
    assert hashlib.sha256((project / "Packages/packages-lock.json").read_bytes()).hexdigest() == fingerprints[
        "Packages/packages-lock.json"
    ]


def test_rejects_reusing_or_mutating_linux_execution_inputs(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    (project / "Packages/manifest.json").write_text(
        '{"dependencies":{"com.unity.toolchain.win-x86_64-linux-x86_64":"2.0.4"}}\n'
    )
    prepare_unity_linux_execution_inputs(project)
    with pytest.raises(UnityHarnessError, match="already exist"):
        prepare_unity_linux_execution_inputs(project)


def test_rejects_unity_execution_copy_with_generated_library(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    (project / "Packages/manifest.json").write_text(
        '{"dependencies":{"com.unity.toolchain.win-x86_64-linux-x86_64":"2.0.4"}}\n'
    )
    (project / "Library").mkdir()
    with pytest.raises(UnityHarnessError, match="stale generated state"):
        prepare_unity_linux_execution_inputs(project)


def test_linux_package_transform_rejects_preexisting_or_wrong_toolchain() -> None:
    with pytest.raises(UnityHarnessError, match="unexpected"):
        unity_linux_package_manifest_bytes(
            b'{"dependencies":{"com.unity.toolchain.linux-x86_64":"2.0.11"}}'
        )
    with pytest.raises(UnityHarnessError, match="toolchain changed"):
        unity_linux_package_manifest_bytes(
            b'{"dependencies":{"com.unity.toolchain.win-x86_64-linux-x86_64":"9"}}'
        )


def test_unity_matrix_has_baseline_and_three_fresh_pairs() -> None:
    assert UNITY_RUNS == (
        ("unity-factual-baseline", "factual"),
        ("unity-counterfactual-baseline", "counterfactual"),
        ("unity-repeat-1-factual", "factual"),
        ("unity-repeat-1-counterfactual", "counterfactual"),
        ("unity-repeat-2-factual", "factual"),
        ("unity-repeat-2-counterfactual", "counterfactual"),
        ("unity-repeat-3-factual", "factual"),
        ("unity-repeat-3-counterfactual", "counterfactual"),
    )


def test_unity_docker_command_pins_image_gpu_and_owned_inputs(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    runtime = tmp_path / "runtime"
    home = tmp_path / "home"
    client = tmp_path / "client"
    loader = tmp_path / "libvulkan.so.1"
    for directory in (runtime, home, client):
        directory.mkdir()
    loader.write_bytes(b"loader")
    socket_path = Path("/tmp/.X11-unix/X987")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    display_socket = socket.socket(socket.AF_UNIX)
    display_socket.bind(str(socket_path))
    try:
        command = _docker_command(
            name="unity-factual-baseline",
            lane="factual",
            gpu_index=1,
            display=":987",
            project=project,
            manifest=manifest,
            runtime=runtime,
            license_home=home,
            vulkan_loader=loader,
            gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        )
    finally:
        display_socket.close()
        socket_path.unlink()
    assert UNITY_IMAGE in command
    assert "device=1" in command
    assert f"{project}:/project" in command
    assert f"{home}:/home/unity" in command
    assert "flashpatch-unity" in command
    assert "02:42:ac:11:00:02" in command
    assert "campbell" not in command
    assert "--gpus" in command


def test_unity_gpu_gate_rejects_live_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter((
        "GPU-1, NVIDIA GeForce RTX 5070 Ti\n",
        "GPU Recovery Action : None\n",
        "GPU-1, 1234\n",
    ))

    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout

    monkeypatch.setattr(
        "flashpatch.l10_unity_runner.subprocess.run",
        lambda *args, **kwargs: Completed(next(outputs)),
    )
    with pytest.raises(UnityL10RunError, match="live worker"):
        _gpu_identity(1)


def test_unity_matrix_cli_fails_closed_before_creating_output(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    output = tmp_path / "runtime"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "flashpatch.cli",
            "unity-renderer-run",
            "--factual-template", str(missing),
            "--counterfactual-template", str(missing),
            "--factual-manifest", str(missing),
            "--counterfactual-manifest", str(missing),
            "--runtime-output", str(output),
            "--entitlement", str(missing),
            "--vulkan-loader", str(missing),
            "--gpu-index", "1",
            "--display", ":98",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["verdict"] == "INCONCLUSIVE"
    assert output.is_dir()
    assessment = json.loads((output / "execution-assessment.json").read_text())
    assert assessment["verdict"] == "INCONCLUSIVE"
    assert assessment["scoreable"] is False


@pytest.mark.parametrize("generated", ("Library", "Temp", "Logs"))
def test_unity_runner_rejects_generated_template_state(
    tmp_path: Path,
    generated: str,
) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    (project / "Packages/manifest.json").write_text("{}\n")
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/ProjectVersion.txt").write_text("2022.3.8f1\n")
    manifest = tmp_path / "manifest.json"
    write_unity_project_manifest(project, manifest)
    stale = project / generated / "ScriptAssemblies/Assembly-CSharp-Editor.dll"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"forged stale compiled adapter")
    with pytest.raises(UnityL10RunError, match="generated state"):
        _verify_manifest(project, manifest)


def test_unity_gpu_lease_rejects_second_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    with _gpu_lease(uuid):
        with pytest.raises(UnityL10RunError, match="leased by another"):
            with _gpu_lease(uuid):
                pytest.fail("second runner acquired one GPU lease")
    with _gpu_lease(uuid):
        pass


def test_unity_fresh_copy_excludes_inputs_outside_frozen_scope(tmp_path: Path) -> None:
    template = _project(tmp_path / "template")
    (template / "Packages").mkdir()
    (template / "Packages/manifest.json").write_text("{}\n")
    (template / "ProjectSettings").mkdir()
    (template / "ProjectSettings/ProjectVersion.txt").write_text("2022.3.8f1\n")
    stale = template / "UserSettings/EditorUserSettings.asset"
    stale.parent.mkdir()
    stale.write_text("unfrozen setting\n")
    (template / "Library").mkdir()
    destination = tmp_path / "fresh"
    _copy_frozen_project(template, destination)
    assert {path.name for path in destination.iterdir()} == {
        "Assets",
        "Packages",
        "ProjectSettings",
    }
    assert not (destination / "UserSettings").exists()
    assert not (destination / "Library").exists()


def test_unity_runner_rejects_unpinned_vulkan_loader_before_gpu_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual = _project(tmp_path / "factual")
    candidate = _project(tmp_path / "candidate")
    candidate_scene = candidate / "Assets/Samples/SmokePortal/SmokePortal.unity"
    candidate_scene.write_text("m_IntensityJitterScale: 0\n")
    for project in (factual, candidate):
        (project / "Packages").mkdir()
        (project / "Packages/manifest.json").write_text("{}\n")
        (project / "ProjectSettings").mkdir()
        (project / "ProjectSettings/ProjectVersion.txt").write_text("2022.3.8f1\n")
        install_unity_harness(
            project,
            tmp_path / f"{project.name}-install.json",
            role="factual" if project == factual else "counterfactual",
        )
    factual_manifest = tmp_path / "factual.json"
    candidate_manifest = tmp_path / "candidate.json"
    write_unity_project_manifest(factual, factual_manifest)
    write_unity_project_manifest(candidate, candidate_manifest)
    entitlement = tmp_path / "UnityEntitlementLicense.xml"
    entitlement.write_text("<Entitlement />\n")
    loader = tmp_path / "libvulkan.so.1"
    loader.write_bytes(b"forged loader")
    monkeypatch.setattr(
        "flashpatch.l10_unity_runner._gpu_uuid",
        lambda index: pytest.fail("GPU was queried before loader rejection"),
    )
    output = tmp_path / "runtime"
    with pytest.raises(UnityL10RunError, match="Vulkan loader"):
        run_unity_l10_matrix(
            factual,
            candidate,
            factual_manifest,
            candidate_manifest,
            output,
            entitlement,
            loader,
            gpu_index=1,
            display=":98",
        )
    assessment = json.loads((output / "execution-assessment.json").read_text())
    assert assessment["verdict"] == "INCONCLUSIVE"
    assert UNITY_VULKAN_LOADER_SHA256 not in hashlib.sha256(loader.read_bytes()).hexdigest()


def test_unity_entitlement_copy_excludes_login_startup_files(tmp_path: Path) -> None:
    source = tmp_path / "UnityEntitlementLicense.xml"
    source.write_text("<Entitlement />\n")
    hostile_profile = tmp_path / ".bash_profile"
    hostile_profile.write_text("echo PROFILE_EXECUTED\n")
    home = tmp_path / "home"
    home.mkdir()
    digest = _install_entitlement(source, home)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        home / ".config/unity3d/Unity/licenses/UnityEntitlementLicense.xml"
    ).read_bytes() == source.read_bytes()
    assert not (home / ".bash_profile").exists()
    assert not (home / ".bashrc").exists()


def test_unity_loader_is_snapshotted_once_into_runtime(tmp_path: Path) -> None:
    source = tmp_path / "source-loader"
    source.write_bytes(b"approved loader")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkey_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    from flashpatch import l10_unity_runner

    original = l10_unity_runner.UNITY_VULKAN_LOADER_SHA256
    l10_unity_runner.UNITY_VULKAN_LOADER_SHA256 = monkey_digest
    try:
        snapshot = _snapshot_loader(source, runtime)
    finally:
        l10_unity_runner.UNITY_VULKAN_LOADER_SHA256 = original
    source.write_bytes(b"mutated after snapshot")
    assert snapshot.read_bytes() == b"approved loader"
    assert snapshot.stat().st_mode & 0o777 == 0o400


def test_unity_timeout_cleanup_failure_quarantines_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    (project / "Packages").mkdir()
    (project / "Packages/manifest.json").write_text("{}\n")
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/ProjectVersion.txt").write_text("2022.3.8f1\n")
    manifest = tmp_path / "manifest.json"
    write_unity_project_manifest(project, manifest)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    loader = tmp_path / "libvulkan.so.1"
    loader.write_bytes(b"loader")
    monkeypatch.setattr(
        "flashpatch.l10_unity_runner.UNITY_VULKAN_LOADER_SHA256",
        hashlib.sha256(loader.read_bytes()).hexdigest(),
    )
    socket_path = Path("/tmp/.X11-unix/X986")
    display_socket = socket.socket(socket.AF_UNIX)
    display_socket.bind(str(socket_path))
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    class Completed:
        def __init__(self, returncode: int):
            self.returncode = returncode

    calls = iter(("run", "stop", "inspect"))

    def fake_run(*args, **kwargs):
        operation = next(calls)
        if operation == "run":
            raise subprocess.TimeoutExpired(args[0], 1)
        return Completed(125 if operation == "stop" else 0)

    monkeypatch.setattr("flashpatch.l10_unity_runner.subprocess.run", fake_run)
    try:
        with pytest.raises(UnityL10RunError, match="quarantined"):
            _run_one(
                name="unity-factual-baseline",
                lane="factual",
                template=project,
                manifest=manifest,
                runtime=runtime,
                license_home=home,
                vulkan_loader=loader,
                gpu_uuid=uuid,
                gpu_index=1,
                display=":986",
                timeout_seconds=1,
            )
    finally:
        display_socket.close()
        socket_path.unlink()
    with pytest.raises(UnityL10RunError, match="quarantined after failed cleanup"):
        with _gpu_lease(uuid):
            pytest.fail("quarantined GPU lease was acquired")
