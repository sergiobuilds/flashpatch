"""Run the frozen Unity L10 replay matrix without sharing generated project state."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .l10_capture import L10CaptureError, pack_engine_capture
from .l10_unity import (
    UnityHarnessError,
    unity_adapter_fingerprints,
    write_unity_project_manifest,
)


class UnityL10RunError(RuntimeError):
    """The Unity replay matrix could not produce promotable execution artifacts."""


UNITY_IMAGE = (
    "unityci/editor@sha256:"
    "966a619d057eefa1ca993f58ac461204e6f9f1b805f91c7be0f10ebf64a5b1df"
)
UNITY_IMAGE_DIGEST = "966a619d057eefa1ca993f58ac461204e6f9f1b805f91c7be0f10ebf64a5b1df"
UNITY_VULKAN_LOADER_SHA256 = (
    "f2c637267fc08e343f3d617252e1ddcaf035dc714728fbdc35ce9ebd8fc4a453"
)
UNITY_SCENE = Path("Assets/Samples/SmokePortal/SmokePortal.unity")
UNITY_RUNS = (
    ("unity-factual-baseline", "factual"),
    ("unity-counterfactual-baseline", "counterfactual"),
    ("unity-repeat-1-factual", "factual"),
    ("unity-repeat-1-counterfactual", "counterfactual"),
    ("unity-repeat-2-factual", "factual"),
    ("unity-repeat-2-counterfactual", "counterfactual"),
    ("unity-repeat-3-factual", "factual"),
    ("unity-repeat-3-counterfactual", "counterfactual"),
)
UNITY_EDITOR_COMMAND = (
    f"printf \"FLASHPATCH_L10_VULKAN_LOADER_VERIFIED {UNITY_VULKAN_LOADER_SHA256}\\n\" && "
    f"printf \"{UNITY_VULKAN_LOADER_SHA256}  /lib/x86_64-linux-gnu/libvulkan.so.1\\n\" | sha256sum -c - && "
    "/opt/unity/Editor/Unity -batchmode -force-vulkan -projectPath /project "
    "-executeMethod FlashPatchL10Capture.Run "
    "-flashpatchOutput \"$FLASHPATCH_OUTPUT\" "
    "-flashpatchMode \"$FLASHPATCH_MODE\" "
    "-flashpatchExpectedSceneSha256 \"$FLASHPATCH_SCENE_SHA256\" "
    "-flashpatchInputManifest \"$FLASHPATCH_INPUT_MANIFEST\" "
    "-flashpatchExpectedInputManifestSha256 \"$FLASHPATCH_INPUT_MANIFEST_SHA256\" "
    "-logFile -; status=$?; "
    "if [ $status -eq 0 ]; then printf \"FLASHPATCH_L10_COMPLETE \"; "
    "tr -d \"\\n\" < \"$FLASHPATCH_OUTPUT/execution-marker.json\"; printf \"\\n\"; fi; "
    "exit $status"
)
UNITY_RECEIPT_COMMAND = f"bash -c '{UNITY_EDITOR_COMMAND}'\n"
_GENERATED_PROJECT_STATE = ("Library", "Temp", "Logs")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _owned_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_dir():
        raise UnityL10RunError(f"{label} is missing or unsafe")
    return resolved


def _verify_manifest(project: Path, expected: Path) -> str:
    if any((project / name).exists() for name in _GENERATED_PROJECT_STATE):
        raise UnityL10RunError("Unity clean template contains generated state")
    if expected.is_symlink() or not expected.is_file():
        raise UnityL10RunError("Unity project manifest is missing or unsafe")
    with tempfile.TemporaryDirectory(prefix="unity-l10-manifest-") as temporary:
        observed = Path(temporary) / "manifest.json"
        write_unity_project_manifest(project, observed)
        if observed.read_bytes() != expected.read_bytes():
            raise UnityL10RunError("Unity clean template no longer matches its frozen manifest")
    return _sha256(expected)


def _copy_frozen_project(template: Path, destination: Path) -> None:
    destination.mkdir()
    for folder in ("Assets", "Packages", "ProjectSettings"):
        source = template / folder
        if source.is_symlink() or not source.is_dir():
            raise UnityL10RunError(f"Unity template input root is unsafe: {folder}")
        shutil.copytree(source, destination / folder)


def _install_entitlement(source: Path, home: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise UnityL10RunError("Unity entitlement is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > 1024 * 1024
            or metadata.st_mode & 0o022
        ):
            raise UnityL10RunError("Unity entitlement is missing or unsafe")
        raw = os.read(descriptor, metadata.st_size + 1)
        if len(raw) != metadata.st_size or b"Entitlement" not in raw:
            raise UnityL10RunError("Unity entitlement is unreadable")
    finally:
        os.close(descriptor)
    destination = home / ".config/unity3d/Unity/licenses/UnityEntitlementLicense.xml"
    destination.parent.mkdir(parents=True)
    (home / ".local/share/unity3d").mkdir(parents=True)
    (home / ".cache").mkdir()
    destination.write_bytes(raw)
    destination.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _snapshot_loader(source: Path, runtime: Path) -> Path:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise UnityL10RunError("Unity Vulkan loader is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > 16 * 1024 * 1024
            or metadata.st_mode & 0o022
        ):
            raise UnityL10RunError("Unity Vulkan loader is missing or unsafe")
        raw = os.read(descriptor, metadata.st_size + 1)
        if len(raw) != metadata.st_size:
            raise UnityL10RunError("Unity Vulkan loader snapshot is unstable")
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != UNITY_VULKAN_LOADER_SHA256:
        raise UnityL10RunError("Unity Vulkan loader is missing or unsafe")
    inputs = runtime / "_inputs"
    inputs.mkdir()
    destination = inputs / "libvulkan.so.1"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    output = os.open(destination, flags, 0o400)
    try:
        if os.write(output, raw) != len(raw):
            raise UnityL10RunError("Unity Vulkan loader snapshot write is incomplete")
        os.fsync(output)
    finally:
        os.close(output)
    if _sha256(destination) != UNITY_VULKAN_LOADER_SHA256:
        raise UnityL10RunError("Unity Vulkan loader snapshot hash mismatch")
    return destination


def _gpu_identity(index: int) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        uuid, name = (item.strip() for item in completed.stdout.strip().split(",", 1))
        recovery = subprocess.run(
            ["nvidia-smi", f"--id={index}", "-q"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise UnityL10RunError("Unity GPU identity is unavailable") from exc
    if "GPU Recovery Action" in recovery and "Reset" in recovery.split(
        "GPU Recovery Action", 1
    )[1].splitlines()[0]:
        raise UnityL10RunError("selected Unity GPU requires reset")
    if any(line.split(",", 1)[0].strip() == uuid for line in processes if line.strip()):
        raise UnityL10RunError("selected Unity GPU is already used by a live worker")
    approved = {
        "NVIDIA GeForce RTX 5070",
        "NVIDIA GeForce RTX 5070 Ti",
    }
    if name not in approved:
        raise UnityL10RunError("selected Unity GPU is not approved")
    return uuid, name


def _gpu_uuid(index: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnityL10RunError("Unity GPU UUID is unavailable") from exc
    uuid = completed.stdout.strip()
    if re.fullmatch(r"GPU-[0-9a-fA-F-]{32,}", uuid) is None:
        raise UnityL10RunError("Unity GPU UUID is invalid")
    return uuid


@contextmanager
def _gpu_lease(uuid: str):
    runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise UnityL10RunError("Unity GPU lease root is unavailable")
    lease_path = runtime_root / f"flashpatch-l10-unity-{uuid}.lock"
    quarantine = runtime_root / f"flashpatch-l10-unity-{uuid}.quarantine"
    if quarantine.exists():
        raise UnityL10RunError("selected Unity GPU is quarantined after failed cleanup")
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lease_path, flags, 0o600)
    except OSError as exc:
        raise UnityL10RunError("Unity GPU lease is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise UnityL10RunError("Unity GPU lease file is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnityL10RunError("selected Unity GPU is leased by another L10 runner") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\nuuid={uuid}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _quarantine_gpu(uuid: str, reason: str) -> None:
    runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    destination = runtime_root / f"flashpatch-l10-unity-{uuid}.quarantine"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        return
    except OSError as exc:
        raise UnityL10RunError("failed Unity GPU cleanup could not be quarantined") from exc
    try:
        os.write(descriptor, f"reason={reason}\n".encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _docker_command(
    *,
    name: str,
    lane: str,
    gpu_index: int,
    display: str,
    project: Path,
    manifest: Path,
    runtime: Path,
    license_home: Path,
    vulkan_loader: Path,
    gpu_uuid: str,
) -> list[str]:
    scene_sha256 = _sha256(project / UNITY_SCENE)
    manifest_sha256 = _sha256(manifest)
    display_number = display.removeprefix(":").split(".", 1)[0]
    socket = Path(f"/tmp/.X11-unix/X{display_number}")
    try:
        socket_mode = socket.stat().st_mode
    except OSError as exc:
        raise UnityL10RunError("Unity display socket is unavailable") from exc
    if not stat.S_ISSOCK(socket_mode):
        raise UnityL10RunError("Unity display socket is unavailable")
    return [
        "docker", "run", "--rm", "--name", f"flashpatch-l10-unity-v2-{name}",
        "--hostname", "flashpatch-unity", "--mac-address", "02:42:ac:11:00:02",
        "--user", f"{os.getuid()}:{os.getgid()}", "--gpus", f"device={gpu_index}",
        "-e", "HOME=/home/unity", "-e", "USER=unity", "-e", "LOGNAME=unity",
        "-e", "BASH_ENV=/dev/null", "-e", "ENV=/dev/null",
        "-e", f"DISPLAY={display}", "-e", "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json",
        "-e", f"FLASHPATCH_OUTPUT=/runtime/{name}", "-e", f"FLASHPATCH_MODE={lane}",
        "-e", f"FLASHPATCH_SCENE_SHA256={scene_sha256}",
        "-e", "FLASHPATCH_INPUT_MANIFEST=/input-manifest.json",
        "-e", f"FLASHPATCH_INPUT_MANIFEST_SHA256={manifest_sha256}",
        "-v", "/etc/machine-id:/etc/machine-id:ro",
        "-v", "/var/lib/dbus/machine-id:/var/lib/dbus/machine-id:ro",
        "-v", "/etc/passwd:/etc/passwd:ro", "-v", "/etc/group:/etc/group:ro",
        "-v", f"{license_home}:/home/unity",
        "-v", f"{vulkan_loader}:/lib/x86_64-linux-gnu/libvulkan.so.1:ro",
        "-v", f"{socket}:{socket}", "-v", f"{project}:/project",
        "-v", f"{manifest}:/input-manifest.json:ro", "-v", f"{runtime}:/runtime",
        "--entrypoint", "bash", UNITY_IMAGE, "-c", UNITY_EDITOR_COMMAND,
    ]


def _write_assessment(runtime: Path, verdict: str, reason: str, completed: list[str]) -> None:
    destination = runtime / "execution-assessment.json"
    temporary = runtime / ".execution-assessment.json.tmp"
    temporary.write_bytes(_canonical({
        "schema": "flashpatch-l10-unity-execution-assessment-v1",
        "verdict": verdict,
        "reason": reason,
        "completed_runs": completed,
        "required_runs": [name for name, _ in UNITY_RUNS],
        "scoreable": False,
    }))
    temporary.replace(destination)


def _run_one(
    *,
    name: str,
    lane: str,
    template: Path,
    manifest: Path,
    runtime: Path,
    license_home: Path,
    vulkan_loader: Path,
    gpu_uuid: str,
    gpu_index: int,
    display: str,
    timeout_seconds: int,
) -> None:
    if _sha256(vulkan_loader) != UNITY_VULKAN_LOADER_SHA256:
        raise UnityL10RunError("Unity Vulkan loader snapshot changed before execution")
    with tempfile.TemporaryDirectory(
        prefix=f"{name}-project-", dir=runtime
    ) as temporary:
        project = Path(temporary) / "project"
        _copy_frozen_project(template, project)
        if any((project / item).exists() for item in _GENERATED_PROJECT_STATE):
            raise UnityL10RunError("fresh Unity project copy contains generated state")
        _verify_manifest(project, manifest)
        log = runtime / f"{name}-console.log"
        command = _docker_command(
            name=name,
            lane=lane,
            gpu_index=gpu_index,
            display=display,
            project=project,
            manifest=manifest,
            runtime=runtime,
            license_home=license_home,
            vulkan_loader=vulkan_loader,
            gpu_uuid=gpu_uuid,
        )
        try:
            with log.open("wb") as stream:
                result = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            stopped = subprocess.run(
                ["docker", "stop", f"flashpatch-l10-unity-v2-{name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            remaining = subprocess.run(
                ["docker", "inspect", f"flashpatch-l10-unity-v2-{name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if stopped.returncode != 0 or remaining.returncode == 0:
                _quarantine_gpu(gpu_uuid, f"timeout_cleanup_failed:{name}")
                raise UnityL10RunError(
                    f"Unity run timed out and GPU was quarantined after cleanup failure: {name}"
                ) from exc
            raise UnityL10RunError(f"Unity run timed out: {name}") from exc
        (runtime / f"{name}-exit-status.txt").write_text(
            f"{result.returncode}\n", encoding="ascii"
        )
        if result.returncode != 0:
            raise UnityL10RunError(f"Unity run failed: {name}")
        raw = log.read_text(encoding="utf-8")
        if raw.count("FLASHPATCH_L10_COMPLETE ") != 1:
            raise UnityL10RunError(f"Unity run lacks one completion marker: {name}")
        if _sha256(vulkan_loader) != UNITY_VULKAN_LOADER_SHA256:
            raise UnityL10RunError("Unity Vulkan loader snapshot changed during execution")
        pack_engine_capture(runtime / name, runtime / f"{name}-pack")


def run_unity_l10_matrix(
    factual_template: Path,
    counterfactual_template: Path,
    factual_manifest: Path,
    counterfactual_manifest: Path,
    runtime_output: Path,
    entitlement: Path,
    vulkan_loader: Path,
    *,
    gpu_index: int,
    display: str,
    timeout_seconds: int = 1800,
) -> dict[str, object]:
    """Run eight sequential fresh Unity projects and seal every capture immediately."""
    try:
        runtime_output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise UnityL10RunError("Unity runtime output already exists") from exc
    completed: list[str] = []
    try:
        factual = _owned_directory(factual_template, "Unity factual template")
        counterfactual = _owned_directory(
            counterfactual_template, "Unity counterfactual template"
        )
        loader = _snapshot_loader(vulkan_loader, runtime_output)
        factual_manifest_sha256 = _verify_manifest(factual, factual_manifest)
        counterfactual_manifest_sha256 = _verify_manifest(
            counterfactual, counterfactual_manifest
        )
        if unity_adapter_fingerprints()["Assets/Editor/FlashPatchL10Capture.cs"] not in {
            row["sha256"] for row in json.loads(factual_manifest.read_text())["files"]
        }:
            raise UnityL10RunError("Unity deterministic adapter is not frozen in the manifest")
        gpu_uuid = _gpu_uuid(gpu_index)
        with _gpu_lease(gpu_uuid):
            observed_uuid, gpu_name = _gpu_identity(gpu_index)
            if observed_uuid != gpu_uuid:
                raise UnityL10RunError("selected Unity GPU identity changed after lease")
            shutil.copyfile(
                factual_manifest,
                runtime_output / "unity-smokeportal-isolated-project-inputs.json",
            )
            shutil.copyfile(
                counterfactual_manifest,
                runtime_output / "unity-smokeportal-candidate-project-inputs.json",
            )
            frozen_manifests = {
                "factual": runtime_output / "unity-smokeportal-isolated-project-inputs.json",
                "counterfactual": runtime_output / "unity-smokeportal-candidate-project-inputs.json",
            }
            if (
                _sha256(frozen_manifests["factual"]) != factual_manifest_sha256
                or _sha256(frozen_manifests["counterfactual"])
                != counterfactual_manifest_sha256
            ):
                raise UnityL10RunError("Unity frozen manifest changed while entering execution")
            with tempfile.TemporaryDirectory(prefix="unity-l10-license-home-") as home_temp:
                license_home = Path(home_temp) / "home"
                license_home.mkdir()
                entitlement_sha256 = _install_entitlement(entitlement, license_home)
                for name, lane in UNITY_RUNS:
                    current_uuid, current_name = _gpu_identity(gpu_index)
                    if (current_uuid, current_name) != (gpu_uuid, gpu_name):
                        raise UnityL10RunError("selected Unity GPU identity changed during execution")
                    template = factual if lane == "factual" else counterfactual
                    manifest = frozen_manifests[lane]
                    _run_one(
                        name=name,
                        lane=lane,
                        template=template,
                        manifest=manifest,
                        runtime=runtime_output,
                        license_home=license_home,
                        vulkan_loader=loader,
                        gpu_uuid=gpu_uuid,
                        gpu_index=gpu_index,
                        display=display,
                        timeout_seconds=timeout_seconds,
                    )
                    completed.append(name)
            if _sha256(loader) != UNITY_VULKAN_LOADER_SHA256:
                raise UnityL10RunError("Unity Vulkan loader snapshot changed before sealing")
        result = {
            "schema": "flashpatch-l10-unity-execution-matrix-v1",
            "image": UNITY_IMAGE,
            "image_digest": UNITY_IMAGE_DIGEST,
            "vulkan_loader_sha256": UNITY_VULKAN_LOADER_SHA256,
            "entitlement_sha256": entitlement_sha256,
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "gpu_name": gpu_name,
            "factual_manifest_sha256": factual_manifest_sha256,
            "counterfactual_manifest_sha256": counterfactual_manifest_sha256,
            "runs": completed,
            "verdict": "COMPLETE",
        }
        (runtime_output / "execution-matrix.json").write_bytes(_canonical(result))
        _write_assessment(runtime_output, "INCONCLUSIVE", "awaiting_receipt_verification", completed)
        return result
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        L10CaptureError,
        UnityHarnessError,
        UnityL10RunError,
    ) as exc:
        _write_assessment(runtime_output, "INCONCLUSIVE", str(exc), completed)
        raise UnityL10RunError(str(exc)) from exc
