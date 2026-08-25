"""Fail-closed verifier for engine-neutral L10 execution evidence."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import cv2

from .core import analyze
from .l10_unity import unity_adapter_fingerprints
from .l10_unity_runner import UNITY_RECEIPT_COMMAND
from .l10_unity_runner import UNITY_VULKAN_LOADER_SHA256
from .renderer_artifact import renderer_rgb_sha256


class L10ReceiptError(ValueError):
    """An L10 receipt or one of its owned artifacts is not trustworthy."""


_TOP_LEVEL_KEYS = {
    "schema",
    "evidence_class",
    "engine",
    "engine_version",
    "source",
    "license",
    "scene",
    "trace",
    "runtime",
    "renderer",
    "factual",
    "attribution",
    "patch",
    "counterfactual",
    "preservation",
    "parity",
    "repeats",
    "verdict",
    "reason",
}
_ENGINES = {"Godot", "Unity", "Unreal"}
_VERDICTS = {"PASS", "SAFE", "INCONCLUSIVE", "UNSCOREABLE"}
_RUN_PROVENANCE_KEYS = {
    "run_id",
    "kind",
    "digest",
    "engine",
    "engine_version",
    "source_revision",
    "factual_scene_sha256",
    "counterfactual_scene_sha256",
    "exit_code",
    "factual_execution_log",
    "counterfactual_execution_log",
    "factual_execution_marker",
    "counterfactual_execution_marker",
    "factual_frames_sha256",
    "counterfactual_frames_sha256",
}
_PIXEL_RUN_PROVENANCE_KEYS = {"factual_pngs", "counterfactual_pngs"}
_UNITY_RUN_PROVENANCE_KEYS = {
    "factual_project_manifest",
    "counterfactual_project_manifest",
}
_TRUST_ENTRY_KEYS = {
    "receipt_sha256",
    "engine",
    "engine_version",
    "evidence_class",
    "repository",
    "revision",
    "tree_sha256",
    "scene_source_path",
    "scene_sha256",
    "runtime_kind",
    "runtime_digest",
}
_UNITY_TRUST_ENTRY_KEYS = {
    "factual_project_manifest_sha256",
    "counterfactual_project_manifest_sha256",
    "project_input_count",
}
_GODOT_TRUST_ENTRY_KEYS = {"adapter_sha256", "run_nonces"}
_UNITY_LEGACY_ADAPTER_SHA256_V1 = (
    "1b1b131793a1a802c1897dd23523324e21034a383ae881bc3719308c6fe85c03"
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise L10ReceiptError("duplicate JSON key")
        value[key] = item
    return value


def _load_json_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                L10ReceiptError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L10ReceiptError("receipt JSON is unreadable") from exc
    if not isinstance(value, dict):
        raise L10ReceiptError("receipt must be a JSON object")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise L10ReceiptError(f"{label} must be a non-empty string")
    return value


def _relative_name(value: object, label: str) -> PurePosixPath:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise L10ReceiptError(f"{label} must be a canonical relative POSIX path")
    return path


def _read_owned(root: Path, reference: object, label: str) -> tuple[bytes, str]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise L10ReceiptError(f"{label} artifact reference is invalid")
    relative = _relative_name(reference["path"], f"{label}.path")
    if not _hex64(reference["sha256"]):
        raise L10ReceiptError(f"{label}.sha256 is invalid")
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in relative.parts[:-1]:
            component = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(component.st_mode):
                raise L10ReceiptError(f"{label} artifact path contains a symlink")
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        metadata = os.fstat(file_fd)
        if not os.path.isfile(f"/proc/self/fd/{file_fd}") or metadata.st_nlink != 1:
            raise L10ReceiptError(f"{label} artifact is not an exclusively owned regular file")
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise L10ReceiptError(f"{label} artifact changed while being read")
        raw = b"".join(chunks)
    except OSError as exc:
        if exc.errno in {40, 62}:
            raise L10ReceiptError(f"{label} artifact path contains a symlink") from exc
        raise L10ReceiptError(f"{label} artifact is unreadable") from exc
    finally:
        if "file_fd" in locals():
            os.close(file_fd)
        os.close(directory_fd)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != reference["sha256"]:
        raise L10ReceiptError(f"{label} artifact hash mismatch")
    return raw, observed


def _read_top_level(path: Path, label: str) -> bytes:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise L10ReceiptError(f"{label} path contains a symlink")
    try:
        fd = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise L10ReceiptError(f"{label} must be an exclusively owned regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise L10ReceiptError(f"{label} changed while being read")
        return b"".join(chunks)
    except OSError as exc:
        raise L10ReceiptError(f"{label} is unreadable") from exc
    finally:
        if "fd" in locals():
            os.close(fd)


def _artifact(root: Path, value: object, label: str) -> str:
    return _read_owned(root, value, label)[1]


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise L10ReceiptError(f"{label} schema is invalid")
    return value


def _load_trust_policy(
    policy_path: Path,
    expected_sha256: str,
) -> dict[str, dict[str, Any]]:
    if not _hex64(expected_sha256):
        raise L10ReceiptError("expected trust-policy hash is invalid")
    raw = _read_top_level(policy_path, "trust-policy")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise L10ReceiptError("trust-policy hash mismatch")
    policy = _load_json_bytes(raw)
    if raw != _canonical_json(policy):
        raise L10ReceiptError("trust-policy bytes are not canonical JSON")
    if set(policy) != {"schema", "engines"} or policy["schema"] != "flashpatch-l10-trust-policy-v1":
        raise L10ReceiptError("trust-policy schema is invalid")
    entries = policy["engines"]
    if not isinstance(entries, list) or not entries:
        raise L10ReceiptError("trust-policy has no engine entries")
    result: dict[str, dict[str, Any]] = {}
    for value in entries:
        if not isinstance(value, dict):
            raise L10ReceiptError("trust-policy engine schema is invalid")
        expected_keys = _TRUST_ENTRY_KEYS | (
            _UNITY_TRUST_ENTRY_KEYS if value.get("engine") == "Unity" else set()
        ) | (_GODOT_TRUST_ENTRY_KEYS if value.get("engine") == "Godot" else set())
        entry = _exact_object(value, expected_keys, "trust-policy.engine")
        engine = entry["engine"]
        if engine not in _ENGINES or engine in result:
            raise L10ReceiptError("trust-policy engine identity is invalid or duplicated")
        if entry["evidence_class"] not in {"natural_project", "controlled_fixture"}:
            raise L10ReceiptError("trust-policy evidence class is invalid")
        for key in ("receipt_sha256", "tree_sha256", "scene_sha256", "runtime_digest"):
            if not _hex64(entry[key]):
                raise L10ReceiptError(f"trust-policy {key} is invalid")
        if not _git_revision(entry["revision"]):
            raise L10ReceiptError("trust-policy revision is invalid")
        _string(entry["engine_version"], "trust-policy.engine_version")
        repository = _string(entry["repository"], "trust-policy.repository")
        if not repository.startswith("https://github.com/"):
            raise L10ReceiptError("trust-policy repository is invalid")
        _relative_name(entry["scene_source_path"], "trust-policy.scene_source_path")
        if entry["runtime_kind"] not in {"container", "binary"}:
            raise L10ReceiptError("trust-policy runtime kind is invalid")
        if engine == "Unity":
            if (
                not _hex64(entry["factual_project_manifest_sha256"])
                or not _hex64(entry["counterfactual_project_manifest_sha256"])
                or not isinstance(entry["project_input_count"], int)
                or isinstance(entry["project_input_count"], bool)
                or entry["project_input_count"] <= 0
            ):
                raise L10ReceiptError("trust-policy Unity project input anchor is invalid")
        if engine == "Godot":
            nonces = entry["run_nonces"]
            if (
                not _hex64(entry["adapter_sha256"])
                or not isinstance(nonces, dict)
                or set(nonces) != {"baseline", "repeat-1", "repeat-2", "repeat-3"}
                or any(
                    not isinstance(pair, dict)
                    or set(pair) != {"factual", "counterfactual"}
                    or any(not _hex64(value) for value in pair.values())
                    for pair in nonces.values()
                )
                or len({value for pair in nonces.values() for value in pair.values()}) != 8
            ):
                raise L10ReceiptError("trust-policy Godot execution anchors are invalid")
        result[engine] = entry
    return result


def _validate_run_provenance(
    raw: bytes,
    *,
    root: Path,
    runtime: dict[str, Any],
    engine: str,
    engine_version: str,
    source_revision: str,
    factual_scene_sha256: str,
    counterfactual_scene_sha256: str,
    scene_source_path: str,
    label: str,
    factual_frames_sha256: str,
    counterfactual_frames_sha256: str,
    factual_events_sha256: str,
    counterfactual_events_sha256: str,
    factual_state_sha256: str,
    counterfactual_state_sha256: str,
    trusted_factual_manifest_sha256: str | None = None,
    trusted_counterfactual_manifest_sha256: str | None = None,
    trusted_project_input_count: int | None = None,
    trusted_adapter_sha256: str | None = None,
    trusted_run_nonces: dict[str, dict[str, str]] | None = None,
    trace_sha256: str | None = None,
) -> dict[str, Any]:
    value = _load_json_bytes(raw)
    expected_keys = _RUN_PROVENANCE_KEYS | _PIXEL_RUN_PROVENANCE_KEYS | (
        _UNITY_RUN_PROVENANCE_KEYS if engine == "Unity" else set()
    )
    if set(value) != expected_keys:
        raise L10ReceiptError(f"{label} schema is invalid")
    if (
        not _hex64(value["run_id"])
        or value["kind"] != runtime["kind"]
        or value["digest"] != runtime["digest"]
        or value["engine"] != engine
        or value["engine_version"] != engine_version
        or value["source_revision"] != source_revision
        or value["factual_scene_sha256"] != factual_scene_sha256
        or value["counterfactual_scene_sha256"] != counterfactual_scene_sha256
        or value["exit_code"] != 0
        or value["factual_frames_sha256"] != factual_frames_sha256
        or value["counterfactual_frames_sha256"] != counterfactual_frames_sha256
    ):
        raise L10ReceiptError(f"{label} does not bind a successful engine execution")
    def execution_log(
        reference: object,
        marker_reference: object,
        lane: str,
        frames_sha256: str,
        events_sha256: str,
        state_sha256: str,
    ) -> tuple[str, str, str | None, dict[str, str] | None, bool]:
        compressed, digest = _read_owned(root, reference, f"{label}.{lane}_execution_log")
        try:
            raw = gzip.decompress(compressed)
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise L10ReceiptError(f"{label} {lane} execution log is not valid gzip UTF-8") from exc
        marker = "FLASHPATCH_L10_PACKED "
        lines = [line[len(marker):] for line in text.splitlines() if line.startswith(marker)]
        if len(lines) != 1:
            raise L10ReceiptError(f"{label} {lane} execution log lacks one packed marker")
        packed = _load_json_bytes(lines[0].encode())
        expected = {
            "schema": "flashpatch-l10-packed-execution-v1",
            "command_sha256": runtime["command"]["sha256"],
            "engine": engine,
            "engine_version": engine_version,
            "frames_sha256": frames_sha256,
            "lane": lane,
            "execution_marker_sha256": packed.get("execution_marker_sha256"),
            "png_set_sha256": packed.get("png_set_sha256"),
            "raw_log_sha256": packed.get("raw_log_sha256"),
        }
        packed_hash_keys = [
            "execution_marker_sha256",
            "png_set_sha256",
            "raw_log_sha256",
        ]
        unity_semantic_binding = False
        unity_semantic_keys = {"runtime_events_sha256", "state_stream_sha256"}
        if engine == "Unity" and any(key in packed for key in unity_semantic_keys):
            expected.update({
                "runtime_events_sha256": events_sha256,
                "state_stream_sha256": state_sha256,
            })
            packed_hash_keys.extend(("runtime_events_sha256", "state_stream_sha256"))
            unity_semantic_binding = True
        if packed != expected or any(
            not _hex64(packed.get(key))
            for key in packed_hash_keys
        ):
            raise L10ReceiptError(f"{label} {lane} execution marker binding failed")
        raw_without_marker = raw[: raw.rfind(("\n" + marker).encode())]
        if hashlib.sha256(raw_without_marker).hexdigest() != packed["raw_log_sha256"]:
            raise L10ReceiptError(f"{label} {lane} raw execution log hash mismatch")
        required = (
            (
                f'engineversion="{engine_version}"',
                "LogInit: Display: Engine is initialized.",
                "-ExecutePythonScript=/project/FlashPatchL10Capture.py",
            )
            if engine == "Unreal"
            else (
                f"Unity Editor version:    {engine_version}",
                "FLASHPATCH_L10_COMPLETE ",
            )
            if engine == "Unity"
            else (
                f"Godot Engine v{engine_version}",
                "OpenGL API",
                "FLASHPATCH_L10_COMPLETE ",
            )
        )
        if any(item not in text for item in required):
            raise L10ReceiptError(f"{label} {lane} execution log semantics are incomplete")
        if engine == "Unity":
            argv = text.splitlines()
            expected_argv = ["-executeMethod", "FlashPatchL10Capture.Run"]
            if not any(
                argv[index : index + len(expected_argv)] == expected_argv
                for index in range(len(argv) - len(expected_argv) + 1)
            ):
                raise L10ReceiptError(f"{label} {lane} Unity command sequence is incomplete")
        if "Fatal error:" in text or "Assertion failed:" in text or "FLASHPATCH_L10_FAILURE" in text:
            raise L10ReceiptError(f"{label} {lane} execution log records a terminal failure")
        marker_raw, marker_digest = _read_owned(
            root, marker_reference, f"{label}.{lane}_execution_marker"
        )
        marker_value = _load_json_bytes(marker_raw)
        if engine != "Godot" and marker_raw != _canonical_json(marker_value):
            raise L10ReceiptError(f"{label} {lane} execution marker is not canonical")
        expected_marker: dict[str, object] = {
            "schema": f"flashpatch-l10-{engine.lower()}-execution-marker-v1",
            "engine": engine,
            "engine_version": engine_version,
            "frame_count": len(value[f"{lane}_pngs"]),
            "mode": lane,
            "png_set_sha256": packed["png_set_sha256"],
        }
        manifest_files: dict[str, str] | None = None
        if engine == "Godot":
            run_label = "baseline" if label == "runtime provenance" else label.split(" runtime provenance", 1)[0]
            expected_nonce = (
                trusted_run_nonces.get(run_label, {}).get(lane)
                if isinstance(trusted_run_nonces, dict)
                else None
            )
            expected_marker = {
                "adapter_sha256": trusted_adapter_sha256,
                "engine": "Godot",
                "engine_version": engine_version,
                "frame_count": len(value[f"{lane}_pngs"]),
                "lane": lane,
                "nonce": expected_nonce,
                "png_set_sha256": packed["png_set_sha256"],
                "scene_sha256": (
                    factual_scene_sha256 if lane == "factual" else counterfactual_scene_sha256
                ),
                "schema": "flashpatch-l10-godot-execution-marker-v2",
                "trace_sha256": trace_sha256,
            }
            if not _hex64(expected_nonce) or not _hex64(trace_sha256):
                raise L10ReceiptError(f"{label} {lane} Godot execution commitment is invalid")
        if engine == "Unity":
            manifest_raw, manifest_digest = _read_owned(
                root,
                value[f"{lane}_project_manifest"],
                f"{label}.{lane}_project_manifest",
            )
            manifest_value = _load_json_bytes(manifest_raw)
            if manifest_raw != _canonical_json(manifest_value) or set(manifest_value) != {"files", "schema"}:
                raise L10ReceiptError(f"{label} {lane} project manifest is not canonical")
            rows = manifest_value["files"]
            if manifest_value["schema"] != "flashpatch-l10-unity-project-inputs-v1" or not isinstance(rows, list):
                raise L10ReceiptError(f"{label} {lane} project manifest schema is invalid")
            manifest_files = {}
            for row in rows:
                entry = _exact_object(row, {"path", "sha256"}, f"{label}.{lane}_project_manifest.file")
                path = _relative_name(entry["path"], f"{label}.{lane}_project_manifest.path").as_posix()
                if path in manifest_files or not _hex64(entry["sha256"]):
                    raise L10ReceiptError(f"{label} {lane} project manifest file is invalid or duplicated")
                if path.split("/", 1)[0] not in {"Assets", "Packages", "ProjectSettings"}:
                    raise L10ReceiptError(f"{label} {lane} project manifest path is out of scope")
                manifest_files[path] = entry["sha256"]
            expected_manifest_digest = (
                trusted_factual_manifest_sha256
                if lane == "factual"
                else trusted_counterfactual_manifest_sha256
            )
            if (
                manifest_digest != expected_manifest_digest
                or len(manifest_files) != trusted_project_input_count
            ):
                raise L10ReceiptError(f"{label} {lane} project manifest does not match the external anchor")
            expected_marker.update({
                "project_manifest_sha256": manifest_digest,
                "scene_sha256": (
                    factual_scene_sha256 if lane == "factual" else counterfactual_scene_sha256
                ),
            })
            marker_schema = marker_value.get("schema")
            adapter_sha256 = manifest_files.get("Assets/Editor/FlashPatchL10Capture.cs")
            current_adapter_sha256 = unity_adapter_fingerprints()[
                "Assets/Editor/FlashPatchL10Capture.cs"
            ]
            if adapter_sha256 == current_adapter_sha256:
                if marker_schema != "flashpatch-l10-unity-execution-marker-v2":
                    raise L10ReceiptError(
                        f"{label} {lane} Unity execution marker schema does not match the adapter"
                    )
                if (
                    f"FLASHPATCH_L10_VULKAN_LOADER_VERIFIED {UNITY_VULKAN_LOADER_SHA256}"
                    not in text
                ):
                    raise L10ReceiptError(
                        f"{label} {lane} Unity Vulkan loader commitment is missing"
                    )
                if not unity_semantic_binding:
                    raise L10ReceiptError(
                        f"{label} {lane} Unity semantic artifact binding is missing"
                    )
                expected_marker.update({
                    "adapter_sha256": adapter_sha256,
                    "graphics_device_id": marker_value.get("graphics_device_id"),
                    "graphics_device_name": marker_value.get("graphics_device_name"),
                    "graphics_device_type": "Vulkan",
                    "graphics_device_vendor": "NVIDIA",
                    "replay_profile": "deterministic-step-v2",
                    "runtime_events_sha256": events_sha256,
                    "schema": marker_schema,
                    "state_stream_sha256": state_sha256,
                })
                allowed_devices = {
                    (0x2F04, "NVIDIA GeForce RTX 5070"),
                    (0x2C05, "NVIDIA GeForce RTX 5070 Ti"),
                }
                if (
                    marker_value.get("graphics_device_id"),
                    marker_value.get("graphics_device_name"),
                ) not in allowed_devices:
                    raise L10ReceiptError(f"{label} {lane} Unity graphics device is not approved")
            elif adapter_sha256 == _UNITY_LEGACY_ADAPTER_SHA256_V1:
                if marker_schema != "flashpatch-l10-unity-execution-marker-v1":
                    raise L10ReceiptError(
                        f"{label} {lane} Unity execution marker schema does not match the adapter"
                    )
                if unity_semantic_binding:
                    raise L10ReceiptError(
                        f"{label} {lane} legacy Unity execution has unexpected semantic binding"
                    )
            else:
                raise L10ReceiptError(f"{label} {lane} Unity execution adapter is not approved")
        if marker_value != expected_marker or marker_digest != packed["execution_marker_sha256"]:
            raise L10ReceiptError(f"{label} {lane} execution marker artifact binding failed")
        png_references = value[f"{lane}_pngs"]
        if not isinstance(png_references, list) or len(png_references) < 2:
            raise L10ReceiptError(f"{label} {lane} PNG evidence is incomplete")
        png_hashes: list[str] = []
        png_paths: list[str] = []
        png_rgb_frames: list[np.ndarray] = []
        for index, png_reference in enumerate(png_references):
            if not isinstance(png_reference, dict):
                raise L10ReceiptError(f"{label} {lane} PNG reference is invalid")
            relative = _relative_name(png_reference.get("path"), f"{label}.{lane}_pngs.path")
            if relative.name != f"{index:04d}.png":
                raise L10ReceiptError(f"{label} {lane} PNG sequence is noncanonical")
            png_paths.append(relative.as_posix())
            png_raw, png_hash = _read_owned(
                root, png_reference, f"{label}.{lane}_pngs[{index}]"
            )
            png_hashes.append(png_hash)
            decoded = cv2.imdecode(np.frombuffer(png_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
                raise L10ReceiptError(f"{label} {lane} PNG is not decodable RGB evidence")
            png_rgb_frames.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        if len(set(png_paths)) != len(png_references):
            raise L10ReceiptError(f"{label} {lane} PNG evidence is duplicated")
        if any(frame.shape != png_rgb_frames[0].shape for frame in png_rgb_frames[1:]):
            raise L10ReceiptError(f"{label} {lane} PNG dimensions are inconsistent")
        png_rgb_sha256 = renderer_rgb_sha256(np.stack(png_rgb_frames).astype(np.uint8, copy=False))
        observed_png_set = hashlib.sha256("\n".join(png_hashes).encode("ascii")).hexdigest()
        if observed_png_set != packed["png_set_sha256"]:
            raise L10ReceiptError(f"{label} {lane} PNG evidence hash set mismatch")
        completion_prefix = "FLASHPATCH_L10_COMPLETE "
        completion_lines = [
            line[len(completion_prefix):]
            for line in text.splitlines()
            if line.startswith(completion_prefix)
        ]
        if len(completion_lines) != 1 or _load_json_bytes(completion_lines[0].encode()) != marker_value:
            raise L10ReceiptError(f"{label} {lane} execution log lacks the engine completion marker")
        return (
            digest,
            marker_digest,
            png_rgb_sha256,
            manifest_files,
            marker_value.get("schema") == "flashpatch-l10-unity-execution-marker-v2",
        )

    factual_log, factual_marker, factual_png_rgb, factual_manifest, factual_deterministic = execution_log(
        value["factual_execution_log"],
        value["factual_execution_marker"],
        "factual",
        factual_frames_sha256,
        factual_events_sha256,
        factual_state_sha256,
    )
    counterfactual_log, counterfactual_marker, counterfactual_png_rgb, counterfactual_manifest, counterfactual_deterministic = execution_log(
        value["counterfactual_execution_log"],
        value["counterfactual_execution_marker"],
        "counterfactual",
        counterfactual_frames_sha256,
        counterfactual_events_sha256,
        counterfactual_state_sha256,
    )
    if factual_log == counterfactual_log:
        raise L10ReceiptError(f"{label} reuses one log for both executions")
    if engine == "Unity":
        if factual_manifest is None or counterfactual_manifest is None:
            raise L10ReceiptError(f"{label} Unity project manifests are missing")
        differing = {
            path
            for path in set(factual_manifest) | set(counterfactual_manifest)
            if factual_manifest.get(path) != counterfactual_manifest.get(path)
        }
        scene_path = scene_source_path
        if differing != {scene_path}:
            raise L10ReceiptError(f"{label} Unity project manifests differ outside the allowed scene")
        if (
            factual_manifest.get(scene_path) != factual_scene_sha256
            or counterfactual_manifest.get(scene_path) != counterfactual_scene_sha256
        ):
            raise L10ReceiptError(f"{label} Unity project manifest scene binding failed")
    expected_run_id = hashlib.sha256(
        _canonical_json({
            "counterfactual_execution_log_sha256": counterfactual_log,
            "counterfactual_execution_marker_sha256": counterfactual_marker,
            "counterfactual_frames_sha256": counterfactual_frames_sha256,
            "factual_execution_log_sha256": factual_log,
            "factual_execution_marker_sha256": factual_marker,
            "factual_frames_sha256": factual_frames_sha256,
        })
    ).hexdigest()
    if value["run_id"] != expected_run_id:
        raise L10ReceiptError(f"{label} run id is not derived from the execution artifacts")
    value = {
        **value,
        "factual_log_sha256": factual_log,
        "counterfactual_log_sha256": counterfactual_log,
        "factual_marker_sha256": factual_marker,
        "counterfactual_marker_sha256": counterfactual_marker,
        "factual_png_rgb_sha256": factual_png_rgb,
        "counterfactual_png_rgb_sha256": counterfactual_png_rgb,
        "factual_deterministic_profile": factual_deterministic,
        "counterfactual_deterministic_profile": counterfactual_deterministic,
    }
    return value


def _validate_capture(root: Path, value: object, label: str) -> dict[str, object]:
    capture = _exact_object(
        value,
        {"frames", "timestamps", "runtime_events", "state_stream", "detector"},
        label,
    )
    frames_raw, frames = _read_owned(root, capture["frames"], f"{label}.frames")
    timestamps_raw, timestamps = _read_owned(root, capture["timestamps"], f"{label}.timestamps")
    events_raw, events = _read_owned(root, capture["runtime_events"], f"{label}.runtime_events")
    state_raw, state = _read_owned(root, capture["state_stream"], f"{label}.state_stream")
    try:
        event_values = json.loads(events_raw)
        state_values = json.loads(state_raw)
    except json.JSONDecodeError as exc:
        raise L10ReceiptError(f"{label} runtime trace is unreadable") from exc
    if isinstance(event_values, dict) and set(event_values) == {"events"}:
        event_values = event_values["events"]
    if isinstance(state_values, dict) and set(state_values) == {"states"}:
        state_values = state_values["states"]
    if not isinstance(event_values, list) or not isinstance(state_values, list):
        raise L10ReceiptError(f"{label} runtime trace schema is invalid")
    detector = _exact_object(
        capture["detector"],
        {"result", "hazard_frames", "receipt"},
        f"{label}.detector",
    )
    if detector["result"] not in {"HAZARDOUS", "SAFE"}:
        raise L10ReceiptError(f"{label} detector result is invalid")
    if (
        not isinstance(detector["hazard_frames"], list)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in detector["hazard_frames"])
        or detector["hazard_frames"] != sorted(set(detector["hazard_frames"]))
    ):
        raise L10ReceiptError(f"{label} hazard frames are invalid")
    if (detector["result"] == "HAZARDOUS") != bool(detector["hazard_frames"]):
        raise L10ReceiptError(f"{label} detector result and frames disagree")
    _artifact(root, detector["receipt"], f"{label}.detector.receipt")
    try:
        timestamp_values = json.loads(timestamps_raw)
    except json.JSONDecodeError as exc:
        raise L10ReceiptError(f"{label} timestamps are unreadable") from exc
    if (
        not isinstance(timestamp_values, list)
        or len(timestamp_values) < 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in timestamp_values
        )
        or any(right <= left for left, right in zip(timestamp_values, timestamp_values[1:]))
    ):
        raise L10ReceiptError(f"{label} timestamps are not strictly monotonic")
    try:
        with np.load(io.BytesIO(frames_raw), allow_pickle=False) as payload:
            if set(payload.files) != {"frames", "timestamps"}:
                raise L10ReceiptError(f"{label} frame artifact schema is invalid")
            frame_values = payload["frames"]
            embedded_timestamps = payload["timestamps"]
            if (
                frame_values.dtype != np.uint8
                or frame_values.ndim != 4
                or frame_values.shape[-1] != 3
                or embedded_timestamps.dtype != np.float64
                or embedded_timestamps.ndim != 1
            ):
                raise L10ReceiptError(f"{label} frames are not a valid RGB artifact")
            if frame_values.shape[0] != len(timestamp_values):
                raise L10ReceiptError(f"{label} frame and timestamp counts differ")
            if not np.array_equal(embedded_timestamps, np.asarray(timestamp_values, dtype=np.float64)):
                raise L10ReceiptError(f"{label} embedded and external timestamps differ")
            rgb_hash = renderer_rgb_sha256(frame_values)
            semantic_timestamps_hash = hashlib.sha256(
                embedded_timestamps.tobytes(order="C")
            ).hexdigest()
            observed_analysis = analyze(frame_values, embedded_timestamps)
            observed_result = "HAZARDOUS" if observed_analysis.hazardous else "SAFE"
            observed_hazard_frames = sorted({
                index
                for window in observed_analysis.windows
                for index, timestamp in enumerate(embedded_timestamps)
                if window.start <= timestamp <= window.end
            })
            frame_count = int(frame_values.shape[0])
            unique_rgb_frames = len({frame.tobytes() for frame in frame_values})
            height = int(frame_values.shape[1])
            width = int(frame_values.shape[2])
    except (OSError, ValueError, EOFError) as exc:
        raise L10ReceiptError(f"{label} frames are not a valid RGB artifact") from exc
    if any(index >= frame_count for index in detector["hazard_frames"]):
        raise L10ReceiptError(f"{label} hazard frame is out of range")
    if (
        detector["result"] != observed_result
        or detector["hazard_frames"] != observed_hazard_frames
    ):
        raise L10ReceiptError(f"{label} detector claim does not match a fresh analysis")
    detector_raw, _ = _read_owned(root, detector["receipt"], f"{label}.detector.receipt")
    detector_receipt = _load_json_bytes(detector_raw)
    if (
        detector_receipt.get("result") != detector["result"]
        or detector_receipt.get("hazard_frames") != detector["hazard_frames"]
        or detector_receipt.get("frames_rgb_sha256") != rgb_hash
        or detector_receipt.get("timestamps_sha256") != semantic_timestamps_hash
    ):
        raise L10ReceiptError(f"{label} detector receipt is not bound to the capture")
    return {
        "artifact_sha256": frames,
        "rgb_sha256": rgb_hash,
        "timestamps_sha256": timestamps,
        "events_sha256": events,
        "state_sha256": state,
        "result": detector["result"],
        "frame_count": frame_count,
        "unique_rgb_frames": unique_rgb_frames,
        "width": width,
        "height": height,
        "event_values": event_values,
        "state_values": state_values,
    }


def verify_l10_receipt(
    receipt_path: Path,
    trust_policy_path: Path,
    expected_trust_policy_sha256: str,
) -> dict[str, object]:
    """Re-open one canonical L10 receipt and every receipt-owned artifact."""
    root = receipt_path.parent.resolve()
    receipt_raw = _read_top_level(receipt_path, "receipt")
    receipt = _load_json_bytes(receipt_raw)
    if receipt_raw != _canonical_json(receipt):
        raise L10ReceiptError("receipt bytes are not canonical JSON")
    if set(receipt) != _TOP_LEVEL_KEYS or receipt["schema"] != "flashpatch-l10-engine-evidence-v1":
        raise L10ReceiptError("receipt top-level schema is invalid")
    engine = receipt["engine"]
    if engine not in _ENGINES:
        raise L10ReceiptError("engine identity is invalid")
    _string(receipt["engine_version"], "engine_version")
    if receipt["evidence_class"] not in {"natural_project", "controlled_fixture"}:
        raise L10ReceiptError("evidence_class is invalid")
    trust_entries = _load_trust_policy(trust_policy_path, expected_trust_policy_sha256)
    if engine not in trust_entries:
        raise L10ReceiptError("engine is not present in the trusted policy")
    trusted = trust_entries[engine]

    source = _exact_object(receipt["source"], {"repository", "revision", "tree_sha256", "provenance"}, "source")
    repository = _string(source["repository"], "source.repository")
    if not repository.startswith("https://github.com/") or not _git_revision(source["revision"]) or not _hex64(source["tree_sha256"]):
        raise L10ReceiptError("source identity is invalid")
    provenance_raw, _ = _read_owned(root, source["provenance"], "source.provenance")
    provenance = _load_json_bytes(provenance_raw)
    if provenance != {
        "clean": True,
        "repository": repository,
        "revision": source["revision"],
        "tree_sha256": source["tree_sha256"],
    }:
        raise L10ReceiptError("source provenance does not bind the declared checkout")
    license_value = _exact_object(receipt["license"], {"spdx", "artifact"}, "license")
    _string(license_value["spdx"], "license.spdx")
    _artifact(root, license_value["artifact"], "license.artifact")
    scene = _exact_object(
        receipt["scene"],
        {"source_path", "artifact", "counterfactual_artifact"},
        "scene",
    )
    _relative_name(scene["source_path"], "scene.source_path")
    scene_hash = _artifact(root, scene["artifact"], "scene.artifact")
    counterfactual_scene_hash = _artifact(
        root, scene["counterfactual_artifact"], "scene.counterfactual_artifact"
    )
    if engine == "Unity" and receipt["evidence_class"] == "natural_project":
        factual_scene_raw, _ = _read_owned(root, scene["artifact"], "scene.artifact")
        candidate_scene_raw, _ = _read_owned(
            root, scene["counterfactual_artifact"], "scene.counterfactual_artifact"
        )
        old = b"  m_IntensityJitterScale: 2000\n"
        new = b"  m_IntensityJitterScale: 0\n"
        if factual_scene_raw.count(old) != 1 or candidate_scene_raw != factual_scene_raw.replace(old, new):
            raise L10ReceiptError("Unity counterfactual scene is not the exact one-parameter source patch")
    if engine == "Godot":
        factual_scene_raw, _ = _read_owned(root, scene["artifact"], "scene.artifact")
        candidate_scene_raw, _ = _read_owned(
            root, scene["counterfactual_artifact"], "scene.counterfactual_artifact"
        )
        old = b"@export var burst_intensity: float = 1.0\n"
        new = b"@export var burst_intensity: float = 0.0\n"
        if factual_scene_raw.count(old) != 1 or candidate_scene_raw != factual_scene_raw.replace(old, new):
            raise L10ReceiptError("Godot counterfactual source is not the exact one-parameter patch")
    trace = _exact_object(receipt["trace"], {"artifact", "commitment"}, "trace")
    trace_hash = _artifact(root, trace["artifact"], "trace.artifact")
    if trace["commitment"] != trace_hash:
        raise L10ReceiptError("trace commitment does not bind the trace artifact")

    runtime = _exact_object(
        receipt["runtime"],
        {"kind", "digest", "command", "engine_identity", "provenance"},
        "runtime",
    )
    if runtime["kind"] not in {"container", "binary"} or not _hex64(runtime["digest"]):
        raise L10ReceiptError("runtime identity is invalid")
    command_raw, _ = _read_owned(root, runtime["command"], "runtime.command")
    identity_raw, _ = _read_owned(root, runtime["engine_identity"], "runtime.engine_identity")
    identity = _load_json_bytes(identity_raw)
    if identity.get("engine") != engine or identity.get("version") != receipt["engine_version"]:
        raise L10ReceiptError("runtime engine identity does not match the receipt")
    runtime_raw, _ = _read_owned(root, runtime["provenance"], "runtime.provenance")
    runtime_provenance_raw = runtime_raw
    observed_trust = {
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "engine": engine,
        "engine_version": receipt["engine_version"],
        "evidence_class": receipt["evidence_class"],
        "repository": repository,
        "revision": source["revision"],
        "tree_sha256": source["tree_sha256"],
        "scene_source_path": scene["source_path"],
        "scene_sha256": scene_hash,
        "runtime_kind": runtime["kind"],
        "runtime_digest": runtime["digest"],
    }
    if engine == "Unity":
        observed_trust.update({
            "factual_project_manifest_sha256": trusted["factual_project_manifest_sha256"],
            "counterfactual_project_manifest_sha256": trusted["counterfactual_project_manifest_sha256"],
            "project_input_count": trusted["project_input_count"],
        })
    if engine == "Godot":
        observed_trust.update({
            "adapter_sha256": trusted["adapter_sha256"],
            "run_nonces": trusted["run_nonces"],
        })
    if observed_trust != trusted:
        raise L10ReceiptError("receipt provenance does not match the externally anchored trust policy")

    renderer = _exact_object(receipt["renderer"], {"backend", "color_space", "width", "height"}, "renderer")
    _string(renderer["backend"], "renderer.backend")
    _string(renderer["color_space"], "renderer.color_space")
    if any(not isinstance(renderer[key], int) or isinstance(renderer[key], bool) or renderer[key] <= 0 for key in ("width", "height")):
        raise L10ReceiptError("renderer dimensions are invalid")

    if (
        isinstance(receipt["factual"], dict)
        and isinstance(receipt["counterfactual"], dict)
        and receipt["factual"].get("frames") == receipt["counterfactual"].get("frames")
    ):
        raise L10ReceiptError("factual and counterfactual captures are identical")
    factual = _validate_capture(root, receipt["factual"], "factual")
    candidate = _validate_capture(root, receipt["counterfactual"], "counterfactual")
    runtime_provenance = _validate_run_provenance(
        runtime_provenance_raw,
        root=root,
        runtime=runtime,
        engine=engine,
        engine_version=receipt["engine_version"],
        source_revision=source["revision"],
        factual_scene_sha256=scene_hash,
        counterfactual_scene_sha256=counterfactual_scene_hash,
        scene_source_path=scene["source_path"],
        label="runtime provenance",
        factual_frames_sha256=factual["artifact_sha256"],
        counterfactual_frames_sha256=candidate["artifact_sha256"],
        factual_events_sha256=factual["events_sha256"],
        counterfactual_events_sha256=candidate["events_sha256"],
        factual_state_sha256=factual["state_sha256"],
        counterfactual_state_sha256=candidate["state_sha256"],
        trusted_factual_manifest_sha256=trusted.get("factual_project_manifest_sha256"),
        trusted_counterfactual_manifest_sha256=trusted.get("counterfactual_project_manifest_sha256"),
        trusted_project_input_count=trusted.get("project_input_count"),
        trusted_adapter_sha256=trusted.get("adapter_sha256"),
        trusted_run_nonces=trusted.get("run_nonces"),
        trace_sha256=trace_hash,
    )
    if (
        runtime_provenance["factual_deterministic_profile"]
        or runtime_provenance["counterfactual_deterministic_profile"]
    ) and command_raw != UNITY_RECEIPT_COMMAND.encode("utf-8"):
        raise L10ReceiptError("deterministic Unity command does not bind the execution closure")
    if (
        runtime_provenance["factual_png_rgb_sha256"] != factual["rgb_sha256"]
        or runtime_provenance["counterfactual_png_rgb_sha256"] != candidate["rgb_sha256"]
    ):
        raise L10ReceiptError("runtime provenance PNG RGB does not match the packed frames")
    for name, capture in (("factual", factual), ("counterfactual", candidate)):
        if runtime_provenance[f"{name}_deterministic_profile"]:
            rendered_values = [
                row.get("rendered_value")
                for row in capture["event_values"]
                if isinstance(row, dict)
            ]
            if (
                len(rendered_values) != capture["frame_count"]
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in rendered_values
                )
                or (
                    len(set(rendered_values)) != 1
                    if name == "counterfactual"
                    else len(set(rendered_values)) < 2
                )
                or capture["unique_rgb_frames"] < 2
            ):
                raise L10ReceiptError(
                    f"{name} deterministic Unity capture does not contain a dynamic rendered trace"
                )
    if factual["artifact_sha256"] == candidate["artifact_sha256"] or factual["rgb_sha256"] == candidate["rgb_sha256"]:
        raise L10ReceiptError("factual and counterfactual captures are identical")
    if (factual["width"], factual["height"]) != (renderer["width"], renderer["height"]):
        raise L10ReceiptError("factual capture dimensions do not match renderer provenance")
    if (candidate["width"], candidate["height"]) != (renderer["width"], renderer["height"]):
        raise L10ReceiptError("counterfactual capture dimensions do not match renderer provenance")
    attribution = _exact_object(
        receipt["attribution"],
        {"object_identity", "component", "property", "artifact"},
        "attribution",
    )
    for key in ("object_identity", "component", "property"):
        _string(attribution[key], f"attribution.{key}")
    attribution_raw, _ = _read_owned(root, attribution["artifact"], "attribution.artifact")
    attribution_receipt = _load_json_bytes(attribution_raw)
    expected_attribution_keys = {
        "object_identity",
        "component",
        "property",
        "factual_value",
        "counterfactual_value",
    }
    if set(attribution_receipt) != expected_attribution_keys:
        raise L10ReceiptError("attribution artifact schema is invalid")
    if any(
        attribution_receipt[key] != attribution[key]
        for key in ("object_identity", "component", "property")
    ):
        raise L10ReceiptError("attribution artifact does not bind the declared cause")
    patch = _exact_object(receipt["patch"], {"kind", "files_changed", "artifact"}, "patch")
    expected_patch = (
        ("controlled_runtime_parameter", 0)
        if receipt["evidence_class"] == "controlled_fixture" and engine == "Unreal"
        else ("minimal_source_parameter", 1)
    )
    if (patch["kind"], patch["files_changed"]) != expected_patch:
        raise L10ReceiptError("patch kind does not match the evidence class")
    patch_raw, _ = _read_owned(root, patch["artifact"], "patch.artifact")
    patch_receipt = _load_json_bytes(patch_raw)
    expected_patch_receipt = {
        "component": attribution["component"],
        "counterfactual_value": attribution_receipt["counterfactual_value"],
        "factual_value": attribution_receipt["factual_value"],
        "files_changed": patch["files_changed"],
        "kind": patch["kind"],
        "object_identity": attribution["object_identity"],
        "property": attribution["property"],
        "factual_scene_sha256": scene_hash,
        "counterfactual_scene_sha256": counterfactual_scene_hash,
    }
    if patch_receipt != expected_patch_receipt:
        raise L10ReceiptError("patch artifact does not bind the attributed parameter change")

    verdict = receipt["verdict"]
    if verdict not in _VERDICTS:
        raise L10ReceiptError("terminal verdict is invalid")
    reason = _string(receipt["reason"], "reason")
    if verdict not in {"PASS", "INCONCLUSIVE"}:
        raise L10ReceiptError("receipt terminal verdict is not supported by L10 verification")

    preservation = _exact_object(
        receipt["preservation"],
        {"action_sequence", "gameplay_state", "object_identity", "terminal_state", "timing", "visual_intent"},
        "preservation",
    )
    if any(not isinstance(preservation[key], bool) for key in preservation):
        raise L10ReceiptError("preservation invariant is not boolean")
    state_preserved = factual["state_values"] == candidate["state_values"]
    if (
        preservation["gameplay_state"] != state_preserved
        or preservation["terminal_state"] != state_preserved
    ):
        raise L10ReceiptError("preservation claim does not match gameplay state")
    measured_elsewhere = (
        "action_sequence",
        "object_identity",
        "timing",
        "visual_intent",
    )
    if any(preservation[key] is not True for key in measured_elsewhere):
        raise L10ReceiptError("preservation claim is not established by the artifact graph")
    if verdict == "PASS" and any(preservation[key] is not True for key in preservation):
        raise L10ReceiptError("PASS preservation invariant failed")
    def event_identity(values: object) -> list[tuple[object, object, object, object]]:
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            raise L10ReceiptError("runtime events are invalid")
        return [
            (item.get("frame"), item.get("object_identity"), item.get("component"), item.get("property"))
            for item in values
        ]
    if event_identity(factual["event_values"]) != event_identity(candidate["event_values"]):
        raise L10ReceiptError("action sequence or object identity changed")
    for name, capture in (("factual", factual), ("counterfactual", candidate)):
        rows = capture["event_values"]
        if not isinstance(rows, list) or len(rows) != capture["frame_count"]:
            raise L10ReceiptError(f"{name} runtime event count does not match frames")
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or row.get("frame") != index
                or row.get("object_identity") != attribution["object_identity"]
                or row.get("component") != attribution["component"]
                or row.get("property") != attribution["property"]
            ):
                raise L10ReceiptError(f"{name} runtime event does not bind the attributed value")
            observed_values = {row.get("value") for row in rows}
            expected_values = (
                {attribution_receipt["counterfactual_value"]}
                if name == "counterfactual"
                else (
                    {
                        attribution_receipt["factual_value"],
                        attribution_receipt["counterfactual_value"],
                    }
                    if patch["kind"] == "controlled_runtime_parameter"
                    else {attribution_receipt["factual_value"]}
                )
            )
        if observed_values != expected_values:
            raise L10ReceiptError(f"{name} runtime event values do not bind the causal parameter")
    parity = _exact_object(receipt["parity"], {"decoder", "timeline"}, "parity")
    if parity != {"decoder": True, "timeline": True} or factual["timestamps_sha256"] != candidate["timestamps_sha256"]:
        raise L10ReceiptError("factual and counterfactual decoder/timeline parity failed")

    repeats = receipt["repeats"]
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise L10ReceiptError("exactly three repeat receipts are required")
    seen: set[int] = set()
    seen_run_ids = {runtime_provenance["run_id"]}
    seen_execution_logs = {
        runtime_provenance["factual_log_sha256"],
        runtime_provenance["counterfactual_log_sha256"],
    }
    baseline_paths = {
        str(receipt["factual"]["frames"]["path"]),
        str(receipt["counterfactual"]["frames"]["path"]),
    }
    repeat_frame_paths: set[str] = set()
    repeat_reproducible = True
    strict_timestamp_integrity = True
    for item in repeats:
        row = _exact_object(
            item,
            {
                "ordinal",
                "artifact",
                "factual",
                "counterfactual",
                "runtime_provenance",
                "runtime_provenance_sha256",
            },
            "repeat",
        )
        if not isinstance(row["ordinal"], int) or isinstance(row["ordinal"], bool) or row["ordinal"] not in {1, 2, 3}:
            raise L10ReceiptError("repeat ordinal is invalid")
        seen.add(row["ordinal"])
        raw, _ = _read_owned(root, row["artifact"], f"repeat-{row['ordinal']}.artifact")
        repeat_receipt = _load_json_bytes(raw)
        expected_repeat = {
            "counterfactual": row["counterfactual"],
            "factual": row["factual"],
            "ordinal": row["ordinal"],
            "runtime_provenance": row["runtime_provenance"],
            "runtime_provenance_sha256": row["runtime_provenance_sha256"],
            "trace_commitment": trace_hash,
        }
        if repeat_receipt != expected_repeat:
            raise L10ReceiptError("repeat receipt content does not match the run")
        repeated_by_name: dict[str, dict[str, Any]] = {}
        for capture_name, baseline in (("factual", factual), ("counterfactual", candidate)):
            repeated_path = str(row[capture_name]["frames"]["path"])
            if repeated_path in baseline_paths or repeated_path in repeat_frame_paths:
                raise L10ReceiptError("repeat capture reuses an earlier frame artifact")
            repeat_frame_paths.add(repeated_path)
            repeated = _validate_capture(root, row[capture_name], f"repeat-{row['ordinal']}.{capture_name}")
            repeated_by_name[capture_name] = repeated
            for key in (
                "rgb_sha256",
                "timestamps_sha256",
                "events_sha256",
                "state_sha256",
                "result",
                "frame_count",
                "width",
                "height",
            ):
                if repeated[key] != baseline[key]:
                    repeat_reproducible = False
                    if key == "timestamps_sha256":
                        strict_timestamp_integrity = False
                    if verdict == "PASS":
                        raise L10ReceiptError("repeat capture is not reproducible")
        repeat_provenance_raw, repeat_provenance_sha256 = _read_owned(
            root,
            row["runtime_provenance"],
            f"repeat-{row['ordinal']}.runtime_provenance",
        )
        if row["runtime_provenance_sha256"] != repeat_provenance_sha256:
            raise L10ReceiptError("repeat runtime provenance binding failed")
        repeat_provenance = _validate_run_provenance(
            repeat_provenance_raw,
            root=root,
            runtime=runtime,
            engine=engine,
            engine_version=receipt["engine_version"],
            source_revision=source["revision"],
            factual_scene_sha256=scene_hash,
            counterfactual_scene_sha256=counterfactual_scene_hash,
            scene_source_path=scene["source_path"],
            label=f"repeat-{row['ordinal']} runtime provenance",
            factual_frames_sha256=repeated_by_name["factual"]["artifact_sha256"],
            counterfactual_frames_sha256=repeated_by_name["counterfactual"]["artifact_sha256"],
            factual_events_sha256=repeated_by_name["factual"]["events_sha256"],
            counterfactual_events_sha256=repeated_by_name["counterfactual"]["events_sha256"],
            factual_state_sha256=repeated_by_name["factual"]["state_sha256"],
            counterfactual_state_sha256=repeated_by_name["counterfactual"]["state_sha256"],
            trusted_factual_manifest_sha256=trusted.get("factual_project_manifest_sha256"),
            trusted_counterfactual_manifest_sha256=trusted.get("counterfactual_project_manifest_sha256"),
            trusted_project_input_count=trusted.get("project_input_count"),
            trusted_adapter_sha256=trusted.get("adapter_sha256"),
            trusted_run_nonces=trusted.get("run_nonces"),
            trace_sha256=trace_hash,
        )
        if (
            repeat_provenance["factual_png_rgb_sha256"]
            != repeated_by_name["factual"]["rgb_sha256"]
            or repeat_provenance["counterfactual_png_rgb_sha256"]
            != repeated_by_name["counterfactual"]["rgb_sha256"]
        ):
            raise L10ReceiptError("repeat runtime provenance PNG RGB does not match the packed frames")
        for name in ("factual", "counterfactual"):
            capture = repeated_by_name[name]
            if repeat_provenance[f"{name}_deterministic_profile"]:
                rendered_values = [
                    event.get("rendered_value")
                    for event in capture["event_values"]
                    if isinstance(event, dict)
                ]
                if (
                    len(rendered_values) != capture["frame_count"]
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        for value in rendered_values
                    )
                        or (
                            len(set(rendered_values)) != 1
                            if name == "counterfactual"
                            else len(set(rendered_values)) < 2
                        )
                        or capture["unique_rgb_frames"] < 2
                ):
                    raise L10ReceiptError(
                        f"repeat-{row['ordinal']} {name} deterministic Unity capture is not dynamic"
                    )
        if repeat_provenance["run_id"] in seen_run_ids:
            raise L10ReceiptError("repeat runtime provenance reuses an earlier run id")
        seen_run_ids.add(repeat_provenance["run_id"])
        repeat_logs = {
            repeat_provenance["factual_log_sha256"],
            repeat_provenance["counterfactual_log_sha256"],
        }
        if repeat_logs & seen_execution_logs:
            raise L10ReceiptError("repeat runtime provenance reuses an earlier execution log")
        seen_execution_logs |= repeat_logs
    if seen != {1, 2, 3}:
        raise L10ReceiptError("repeat ordinals are incomplete")

    hazard_removed = factual["result"] == "HAZARDOUS" and candidate["result"] == "SAFE"
    if verdict == "PASS" and not hazard_removed:
        raise L10ReceiptError("PASS requires factual hazard removal in the counterfactual")
    if (
        verdict == "INCONCLUSIVE"
        and hazard_removed
        and repeat_reproducible
        and state_preserved
    ):
        raise L10ReceiptError("INCONCLUSIVE contradicts completed repeatable L10 evidence")
    result: dict[str, object] = {
        "schema": "flashpatch-l10-verification-result-v1",
        "verified": verdict == "PASS",
        "artifact_graph_verified": True,
        "execution_origin_authenticated": False,
        "engine": engine,
        "engine_version": receipt["engine_version"],
        "evidence_class": receipt["evidence_class"],
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "trace_commitment": trace_hash,
        "repeats": 3,
        "repeat_reproducible": repeat_reproducible,
        "verdict": verdict,
        "reason": reason,
        "scoreable": False,
        "external_claim_authorized": False,
    }
    if engine == "Unity":
        gate_results = {
            "actual_final_frame_capture": True,
            "allowlisted_single_property_change": True,
            "editor_entitlement_bound": False,
            "editor_execution": True,
            "execution_origin_authenticated": False,
            "gameplay_state_preservation": state_preserved,
            "hazard_reduction_or_removal": hazard_removed,
            "independent_repeated_executions": True,
            "repeat_capture_reproducibility": repeat_reproducible,
            "same_trace": True,
            "source_revision_and_license": True,
            "strict_timestamp_integrity": strict_timestamp_integrity,
        }
        blocker_order = (
            (
                "EDITOR_ENTITLEMENT_NOT_BOUND",
                "editor_entitlement_bound",
                {
                    "observed": "the promoted receipt does not bind the execution-matrix entitlement commitment",
                    "required": "bind a non-secret entitlement commitment into the receipt and trust policy",
                },
            ),
            (
                "EXECUTION_ORIGIN_NOT_AUTHENTICATED",
                "execution_origin_authenticated",
                {
                    "observed": "engine logs and markers are self-authenticated artifacts",
                    "required": "bind an independently authenticated execution origin",
                },
            ),
            (
                "GAMEPLAY_STATE_NOT_PRESERVED",
                "gameplay_state_preservation",
                {
                    "observed": {
                        "factual_state_sha256": factual["state_sha256"],
                        "counterfactual_state_sha256": candidate["state_sha256"],
                    },
                    "required": "factual and counterfactual gameplay-state streams must be identical",
                },
            ),
            (
                "HAZARD_REDUCTION_NOT_ESTABLISHED",
                "hazard_reduction_or_removal",
                {
                    "observed": {
                        "factual_result": factual["result"],
                        "counterfactual_result": candidate["result"],
                    },
                    "required": "the factual trace must be HAZARDOUS and the counterfactual trace SAFE",
                },
            ),
            (
                "REPEAT_CAPTURE_NOT_REPRODUCIBLE",
                "repeat_capture_reproducibility",
                {
                    "observed": "one or more repeated RGB, timestamp, event, or state hashes differ from baseline",
                    "required": "all three fresh factual/counterfactual pairs must reproduce baseline semantic hashes",
                },
            ),
            (
                "STRICT_TIMESTAMP_INTEGRITY_FAILED",
                "strict_timestamp_integrity",
                {
                    "observed": "one or more repeated timestamp hashes differ from baseline",
                    "required": "factual, counterfactual, and all repeat timestamp hashes must match",
                },
            ),
        )
        unmet = [code for code, gate, _ in blocker_order if not gate_results[gate]]
        result.update({
            "gate_results": gate_results,
            "unmet_gates": unmet,
            "unmet_gate_details": {
                code: details
                for code, gate, details in blocker_order
                if not gate_results[gate]
            },
        })
    return result


def verify_l10_bundle(
    bundle_path: Path,
    trust_policy_path: Path,
    expected_trust_policy_sha256: str,
) -> dict[str, object]:
    """Reopen three engines while keeping natural and controlled evidence separate."""
    root = bundle_path.parent.resolve()
    bundle_raw = _read_top_level(bundle_path, "bundle")
    bundle = _load_json_bytes(bundle_raw)
    if bundle_raw != _canonical_json(bundle):
        raise L10ReceiptError("bundle bytes are not canonical JSON")
    if set(bundle) != {"schema", "receipts"} or bundle["schema"] != "flashpatch-l10-engine-neutral-bundle-v1":
        raise L10ReceiptError("bundle schema is invalid")
    rows = bundle["receipts"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise L10ReceiptError("bundle requires exactly three engine receipts")
    declared_classes: list[object] = []
    for row in rows:
        raw, _ = _read_owned(root, row, "bundle.receipt")
        declared_classes.append(_load_json_bytes(raw).get("evidence_class"))
    if "natural_project" not in declared_classes:
        raise L10ReceiptError("bundle requires at least one natural-project engine receipt")
    results: dict[str, dict[str, object]] = {}
    for row in rows:
        raw, observed = _read_owned(root, row, "bundle.receipt")
        relative = _relative_name(row["path"], "bundle.receipt.path")
        result = verify_l10_receipt(
            root / Path(*relative.parts),
            trust_policy_path,
            expected_trust_policy_sha256,
        )
        if result["receipt_sha256"] != observed:
            raise L10ReceiptError("bundle receipt hash binding failed")
        engine = str(result["engine"])
        if engine in results:
            raise L10ReceiptError("bundle contains a duplicate engine")
        results[engine] = result
        if hashlib.sha256(raw).hexdigest() != observed:
            raise L10ReceiptError("bundle receipt changed during verification")
    if set(results) != _ENGINES:
        raise L10ReceiptError("bundle does not cover Godot, Unity, and Unreal")
    by_class = {
        evidence_class: sorted(
            engine
            for engine, result in results.items()
            if result["evidence_class"] == evidence_class
        )
        for evidence_class in ("natural_project", "controlled_fixture")
    }
    if not by_class["natural_project"]:
        raise L10ReceiptError("bundle requires at least one natural-project engine receipt")
    all_passed = all(result["verified"] is True for result in results.values())
    natural_support_complete = (
        set(by_class["natural_project"]) == _ENGINES and all_passed
    )
    return {
        "schema": "flashpatch-l10-engine-neutral-verification-v1",
        "verified": natural_support_complete,
        "artifact_graph_verified": True,
        "engines": results,
        "evidence_classes": by_class,
        "engine_neutral": natural_support_complete,
        "execution_origin_authenticated": False,
        "external_claim_authorized": False,
        "scoreable": False,
        "verdict": "PASS" if natural_support_complete else "INCONCLUSIVE",
    }
