"""Code-owned Godot L10 execution-marker adapter for isolated fixture copies."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .godot import GodotRendererReplayRunner


_HOOK = r'''

func _flashpatch_sha256(path: String) -> String:
	var context := HashingContext.new()
	context.start(HashingContext.HASH_SHA256)
	context.update(FileAccess.get_file_as_bytes(path))
	return context.finish().hex_encode()


func _flashpatch_arg(name: String) -> String:
	var arguments := OS.get_cmdline_user_args()
	var index := arguments.find(name)
	return arguments[index + 1] if index >= 0 and index + 1 < arguments.size() else ""


func _flashpatch_write_marker(capture_directory: String) -> void:
	var hashes: Array[String] = []
	for index in range(12):
		hashes.append(_flashpatch_sha256(capture_directory.path_join("frame_%06d.png" % index)))
	var joined := "\n".join(hashes).to_utf8_buffer()
	var context := HashingContext.new()
	context.start(HashingContext.HASH_SHA256)
	context.update(joined)
	var marker := {
		"adapter_sha256": _flashpatch_arg("--l10-adapter-sha256"),
		"engine": "Godot",
		"engine_version": _flashpatch_arg("--l10-engine-version"),
		"frame_count": 12,
		"lane": _flashpatch_arg("--l10-lane"),
		"nonce": _flashpatch_arg("--l10-nonce"),
		"png_set_sha256": context.finish().hex_encode(),
		"scene_sha256": _flashpatch_arg("--l10-scene-sha256"),
		"schema": "flashpatch-l10-godot-execution-marker-v2",
		"trace_sha256": _flashpatch_arg("--l10-trace-sha256"),
	}
	var encoded := JSON.stringify(marker)
	var marker_path := _output_path.get_base_dir().path_join("execution-marker.json")
	var marker_file := FileAccess.open(marker_path, FileAccess.WRITE)
	marker_file.store_string(encoded + "\n")
	print("FLASHPATCH_L10_COMPLETE " + encoded)
'''


def adapter_sha256() -> str:
    return hashlib.sha256(_HOOK.encode()).hexdigest()


def install_godot_l10_adapter(project: Path) -> str:
    source = project / "main.gd"
    raw = source.read_text(encoding="utf-8")
    needle = "\t_write_result(_output_path, fps, gameplay_state, capture_directory)\n"
    replacement = needle + "\t_flashpatch_write_marker(capture_directory)\n"
    if raw.count(needle) != 1 or "func _flashpatch_write_marker" in raw:
        raise ValueError("Godot L10 adapter insertion point is invalid")
    source.write_text(raw.replace(needle, replacement) + _HOOK, encoding="utf-8")
    return adapter_sha256()


class GodotL10ReplayRunner(GodotRendererReplayRunner):
    """Run one isolated lane with an engine-emitted, prechallenged marker."""

    def __init__(self, project: Path, *, godot_binary: Path, nonce: str) -> None:
        source_text = (project / "main.gd").read_text(encoding="utf-8")
        if "@export var burst_intensity: float = 1.0\n" in source_text:
            self.lane = "factual"
        elif "@export var burst_intensity: float = 0.0\n" in source_text:
            self.lane = "counterfactual"
        else:
            raise ValueError("Godot L10 lane cannot be derived from the isolated source")
        if len(nonce) != 64 or any(character not in "0123456789abcdef" for character in nonce):
            raise ValueError("Godot L10 nonce is invalid")
        self.nonce = nonce
        self.source_sha256 = hashlib.sha256((project / "main.gd").read_bytes()).hexdigest()
        self.adapter_sha256 = install_godot_l10_adapter(project)
        super().__init__(project, godot_binary=godot_binary, timeout_seconds=30)

    def _renderer_command(self, trace: Path, output: Path) -> list[str]:
        return [
            *super()._renderer_command(trace, output),
            "--l10-lane", self.lane,
            "--l10-nonce", self.nonce,
            "--l10-adapter-sha256", self.adapter_sha256,
            "--l10-engine-version", self._godot_version(),
            "--l10-scene-sha256", self.source_sha256,
            "--l10-trace-sha256", hashlib.sha256(trace.read_bytes()).hexdigest(),
        ]
