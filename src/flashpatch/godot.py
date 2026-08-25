from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np


class GodotReplayRunner:
    def __init__(
        self,
        project: Path,
        *,
        godot_binary: Path | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.project = Path(project).resolve()
        self.godot_binary = self._resolve_binary(godot_binary)
        self.timeout_seconds = timeout_seconds
        self._require_declared_major_compatibility()

    @staticmethod
    def _resolve_binary(explicit: Path | None) -> Path:
        bundled = Path(__file__).resolve().parents[2] / ".tools"
        candidates = [
            explicit,
            Path(os.environ["GODOT_BINARY"]) if "GODOT_BINARY" in os.environ else None,
            bundled / "godot",
        ]
        candidates.extend(sorted(bundled.glob("Godot*_linux.x86_64")))
        for name in ("godot", "godot4"):
            located = shutil.which(name)
            if located:
                candidates.append(Path(located))
        for candidate in candidates:
            if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        raise FileNotFoundError(
            "Godot executable not found; set GODOT_BINARY or install it at .tools/godot"
        )

    def _require_declared_major_compatibility(self) -> None:
        """Reject an explicit Godot-3/4 project-versus-binary mismatch before replay."""
        project_file = self.project / "project.godot"
        try:
            project_text = project_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("Godot project.godot is unreadable") from exc
        declared = re.search(r"^config_version\s*=\s*(\d+)\s*$", project_text, re.MULTILINE)
        if declared is None:
            # Older or deliberately minimal fixtures cannot prove a major
            # requirement.  Keep their existing runner contract unchanged.
            return
        expected = {4: 3, 5: 4}.get(int(declared.group(1)))
        if expected is None:
            raise RuntimeError("project.godot config_version does not identify a supported Godot major")
        completed = subprocess.run(
            [str(self.godot_binary), "--version"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.timeout_seconds,
        )
        version = completed.stdout.strip().splitlines()
        actual_match = re.match(r"(\d+)\.", version[0]) if completed.returncode == 0 and version else None
        if actual_match is None:
            raise RuntimeError("Godot binary version is unavailable")
        actual = int(actual_match.group(1))
        if actual != expected:
            raise RuntimeError(
                f"project declares Godot {expected} (config_version={declared.group(1)}) but runner binary is Godot {actual}"
            )

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        trace = Path(trace).resolve()
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(self.godot_binary),
                "--headless",
                "--fixed-fps",
                str(self._fixed_fps(trace)),
                "--path",
                str(self.project),
                "--",
                "--trace",
                str(trace),
                "--output",
                str(output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Godot replay exited {completed.returncode}:\n{completed.stdout}"
            )
        if not output.is_file():
            raise RuntimeError(f"Godot replay did not create {output}:\n{completed.stdout}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("status") != "REPLAYED":
            raise RuntimeError(f"Godot replay returned an invalid result: {result!r}")
        return result

    @staticmethod
    def _fixed_fps(trace: Path) -> int:
        """Read the declared deterministic replay cadence, never a default."""
        try:
            payload = json.loads(trace.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("trace JSON with fixed_fps is required") from exc
        fixed_fps = payload.get("fixed_fps") if isinstance(payload, dict) else None
        if isinstance(fixed_fps, bool) or not isinstance(fixed_fps, int) or fixed_fps <= 0:
            raise RuntimeError("trace fixed_fps must be a positive integer")
        return fixed_fps


class GodotRendererReplayRunner(GodotReplayRunner):
    """Run a Godot replay through an actual X11/Wayland renderer and pack captures.

    The Godot-side adapter writes PNGs plus a manifest.  This runner is the
    only component which turns those renderer-owned files into the NPZ input
    accepted by FlashPatch's pixel detector.  It intentionally refuses to
    silently fall back to ``--headless``.
    """

    MAX_CAPTURE_RGB_BYTES = 256 * 1024 * 1024

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "non-headless renderer capture requires DISPLAY or WAYLAND_DISPLAY; "
                "headless replay is not a substitute"
            )
        trace = Path(trace).resolve()
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            self._renderer_command(trace, output),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Godot non-headless replay exited {completed.returncode}:\n{completed.stdout}"
            )
        if not output.is_file():
            raise RuntimeError(
                f"Godot non-headless replay did not create {output}:\n{completed.stdout}"
            )
        result = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("status") != "REPLAYED":
            raise RuntimeError(f"Godot renderer replay returned an invalid result: {result!r}")
        result["engine_execution_log"] = completed.stdout
        capture = result.get("renderer_capture")
        if not isinstance(capture, dict):
            raise RuntimeError("Godot renderer replay omitted renderer_capture manifest")
        raw_directory = capture.get("frame_directory")
        raw_timestamps = capture.get("timestamps_us")
        actual_timestamps = capture.get("actual_capture_timestamps_us")
        viewport = capture.get("viewport")
        color_space = capture.get("color_space")
        if not isinstance(raw_directory, str) or not raw_directory:
            raise RuntimeError("renderer_capture.frame_directory must be a non-empty string")
        if not isinstance(raw_timestamps, list) or not raw_timestamps:
            raise RuntimeError("renderer_capture.timestamps_us must be a non-empty list")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_timestamps
            )
            or any(
                right <= left
                for left, right in zip(raw_timestamps, raw_timestamps[1:])
            )
        ):
            raise RuntimeError(
                "renderer_capture.timestamps_us must be a strictly increasing provenance timeline"
            )
        if actual_timestamps is not None and (
            not isinstance(actual_timestamps, list)
            or len(actual_timestamps) != len(raw_timestamps)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in actual_timestamps)
            or any(right <= left for left, right in zip(actual_timestamps, actual_timestamps[1:]))
        ):
            raise RuntimeError("renderer_capture.actual_capture_timestamps_us must be a strictly increasing provenance timeline")
        if actual_timestamps is None:
            actual_timestamps = list(raw_timestamps)
        fixed_fps = self._fixed_fps(trace)
        raw_timestamps = [round(index * 1_000_000 / fixed_fps) for index in range(len(raw_timestamps))]
        if (
            not isinstance(viewport, list)
            or len(viewport) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in viewport)
        ):
            raise RuntimeError("renderer_capture.viewport must contain two positive integer dimensions")
        if color_space != "sRGB/BT.709":
            raise RuntimeError("renderer_capture.color_space must be sRGB/BT.709")
        frame_directory = (output.parent / raw_directory).resolve()
        if output.parent.resolve() not in frame_directory.parents or not frame_directory.is_dir():
            raise RuntimeError("renderer capture directory must stay inside replay output directory")
        frame_paths = sorted(frame_directory.glob("frame_*.png"))
        if len(frame_paths) != len(raw_timestamps):
            raise RuntimeError("renderer capture frame count does not match timestamps")
        first_bgr = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first_bgr is None:
            raise RuntimeError(f"renderer capture frame cannot be decoded: {frame_paths[0]}")
        height, width, channels = first_bgr.shape
        estimated_bytes = len(frame_paths) * height * width * channels
        if estimated_bytes > self.MAX_CAPTURE_RGB_BYTES:
            raise RuntimeError(
                "renderer capture exceeds the bounded RGB packing budget; "
                f"declared={estimated_bytes} max={self.MAX_CAPTURE_RGB_BYTES}"
            )
        frame_array = np.empty((len(frame_paths), height, width, channels), dtype=np.uint8)
        frame_array[0] = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        for index, frame_path in enumerate(frame_paths[1:], start=1):
            bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"renderer capture frame cannot be decoded: {frame_path}")
            if bgr.shape != (height, width, channels):
                raise RuntimeError("renderer capture frame dimensions differ within one replay")
            frame_array[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_array.dtype != np.uint8 or frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise RuntimeError("renderer capture must decode to uint8 RGB frames")
        timestamps_us = np.asarray(raw_timestamps, dtype=np.int64)
        if timestamps_us.shape != (len(frame_array),) or np.any(np.diff(timestamps_us) <= 0):
            raise RuntimeError("renderer capture timestamps_us must be strictly increasing")
        artifact = output.with_name("renderer-frames.npz")
        np.savez_compressed(artifact, frames=frame_array, timestamps=timestamps_us.astype(np.float64) / 1_000_000.0)
        result["frames_npz"] = artifact.name
        result["renderer_capture"] = {
            **capture,
            "actual_capture_timestamps_us": actual_timestamps,
            "timestamps_us": raw_timestamps,
            "frame_count": len(frame_array),
            "shape": list(frame_array.shape),
            "dtype": frame_array.dtype.str,
            "artifact": artifact.name,
            "trace_sha256": f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}",
            "godot_version": self._godot_version(),
            "renderer_configuration": self._renderer_configuration(),
        }
        output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return result

    def _renderer_command(self, trace: Path, output: Path) -> list[str]:
        return [
            str(self.godot_binary), "--display-driver", "x11", "--rendering-driver", "opengl3",
            "--fixed-fps", str(self._fixed_fps(trace)), "--disable-vsync", "--single-threaded-scene",
            "--resolution", "320x180", "--path", str(self.project), "--",
            "--trace", str(trace), "--output", str(output), "--renderer-capture",
        ]

    def _renderer_configuration(self) -> dict[str, str]:
        return {"display_driver": "x11", "rendering_driver": "opengl3"}

    def _godot_version(self) -> str:
        completed = subprocess.run(
            [str(self.godot_binary), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
        )
        version = completed.stdout.strip().splitlines()
        if completed.returncode != 0 or not version:
            raise RuntimeError("Godot renderer replay could not determine Godot version")
        return version[0]


class Godot3RendererReplayRunner(GodotRendererReplayRunner):
    """Godot-3 X11 capture using the same packed-RGB validation as Godot 4."""

    def _renderer_command(self, trace: Path, output: Path) -> list[str]:
        return [
            str(self.godot_binary), "--fixed-fps", str(self._fixed_fps(trace)),
            "--disable-vsync", "--path", str(self.project), "--",
            "--trace", str(trace), "--output", str(output), "--renderer-capture",
        ]


class GodotNativeMainRendererReplayRunner(GodotRendererReplayRunner):
    """Capture an instrumented native main scene without overriding project graphics.

    X11, fixed media time, and disabled V-Sync are required for reproducible
    non-headless capture.  The project's renderer driver, resolution, and
    scene-thread setting are deliberately left unchanged.
    """

    _RESERVED_USER_ARGUMENTS = {"--", "--trace", "--output", "--renderer-capture"}

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        self._prepare_project_import()
        return super().replay(trace, output)

    def _prepare_project_import(self) -> None:
        # Project imports produce the artifacts used by the later X11 run.
        # Keep that preparation genuinely headless and display-isolated. Godot
        # 4.7.1 can still abort during EditorNode shutdown after a completed
        # first import; only that exact abort is revalidated once below.
        import_environment = os.environ.copy()
        import_environment.pop("DISPLAY", None)
        import_environment.pop("WAYLAND_DISPLAY", None)
        command = [str(self.godot_binary), "--headless", "--path", str(self.project), "--import"]

        def run_import() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
                env=import_environment,
            )

        completed = run_import()
        known_shutdown_abort = (
            completed.returncode == -signal.SIGABRT
            and 'ERROR: Parameter "singleton" is null.' in completed.stdout
            and "at: is_cmdline_mode (editor/editor_node.cpp:" in completed.stdout
        )
        if known_shutdown_abort:
            revalidated = run_import()
            if revalidated.returncode != 0:
                raise RuntimeError(
                    "Godot native-main project import hit the known post-import shutdown abort, "
                    f"then revalidation exited {revalidated.returncode}:\n"
                    f"initial import stdout:\n{completed.stdout}\n"
                    f"revalidation stdout:\n{revalidated.stdout}"
                )
            return
        if completed.returncode != 0:
            raise RuntimeError(
                f"Godot native-main project import exited {completed.returncode}:\n{completed.stdout}"
            )

    @classmethod
    def _launch_arguments(cls, trace: Path) -> list[str]:
        try:
            payload = json.loads(trace.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("native-main trace JSON is unreadable") from exc
        arguments = payload.get("launch_arguments", []) if isinstance(payload, dict) else None
        if not isinstance(arguments, list):
            raise RuntimeError("native-main trace launch_arguments must be a list")
        validated: list[str] = []
        for argument in arguments:
            if not isinstance(argument, str) or not argument or "\x00" in argument:
                raise RuntimeError("native-main trace launch argument must be a non-empty safe string")
            if argument in cls._RESERVED_USER_ARGUMENTS or any(
                argument.startswith(f"{reserved}=") for reserved in cls._RESERVED_USER_ARGUMENTS - {"--"}
            ):
                raise RuntimeError("native-main trace launch argument attempts to override capture wiring")
            validated.append(argument)
        return validated

    def _renderer_command(self, trace: Path, output: Path) -> list[str]:
        return [
            str(self.godot_binary), "--display-driver", "x11",
            "--fixed-fps", str(self._fixed_fps(trace)), "--disable-vsync",
            "--path", str(self.project), "--",
            "--trace", str(trace), "--output", str(output), "--renderer-capture",
            *self._launch_arguments(trace),
        ]

    def _renderer_configuration(self) -> dict[str, str]:
        return {"display_driver": "x11", "rendering_driver": "project_default"}
