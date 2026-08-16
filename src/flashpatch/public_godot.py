"""Controlled-mutation setup for public Godot projects.

This module never modifies the upstream checkout.  It creates a separately
labelled copy that contains the smallest possible FlashPatch probe: one
exported visual parameter, renderer-frame capture, and runtime provenance.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import signal
import shutil
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .external_league import pack_renderer_png_sequence
from .core import analyze
from .godot import (
    Godot3RendererReplayRunner,
    GodotNativeMainRendererReplayRunner,
    GodotRendererReplayRunner,
    GodotReplayRunner,
)
from .renderer_artifact import (
    RendererArtifactError,
    open_renderer_artifact,
    renderer_rgb_sha256,
    renderer_visual_change_ratio,
)

GODOT_DEMO_REPOSITORY = "https://github.com/godotengine/godot-demo-projects"
GODOT_DEMO_REVISION = "52e30044658448149b04e8f69b475eebbfbd8f6e"
GODOT_DEMO_LICENSE = "MIT"
PONG_PROJECT_PATH = "2d/pong"
SPARTA_REPOSITORY = "https://github.com/Lacaedemon/sparta"
SPARTA_REVISION = "06be859d9237192dca391a35bf3a267ff939ceae"
SPARTA_LICENSE = "MIT"


_CONTROLLED_NATIVE_SHADER_PROJECT = b'''config_version=5

[application]
run/main_scene="res://main.tscn"

[display]
window/size/viewport_width=320
window/size/viewport_height=180
window/size/window_width_override=320
window/size/window_height_override=180

[rendering]
renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"

[input]
advance={
"deadzone": 0.5,
"events": []
}
'''
_CONTROLLED_NATIVE_SHADER_SCENE = b'''[gd_scene load_steps=4 format=3]

[ext_resource type="Shader" path="res://effects/controlled_flash.gdshader" id="1_shader"]
[ext_resource type="Script" path="res://controlled_state_provider.gd" id="2_state"]

[sub_resource type="ShaderMaterial" id="ShaderMaterial_flash"]
shader = ExtResource("1_shader")
shader_parameter/flash_intensity = 1.0
shader_parameter/flash_phase = 0.0

[node name="Main" type="Node2D"]
script = ExtResource("2_state")

[node name="Flash" type="ColorRect" parent="."]
offset_right = 320.0
offset_bottom = 180.0
mouse_filter = 2
material = SubResource("ShaderMaterial_flash")
'''
_CONTROLLED_NATIVE_SHADER = b'''shader_type canvas_item;

uniform float flash_intensity = 1.0;
uniform float flash_phase = 0.0;

void fragment() {
    float level = flash_intensity * flash_phase;
    COLOR = vec4(level, level, level, 1.0);
}
'''
_CONTROLLED_NATIVE_STATE_PROVIDER = b'''extends Node2D

@export var phase: int = 0
@export var player_x: int = 0
@export var world_epoch: int = 0
@export var completed: bool = false
@export var score: int = 0

var _render_tick: int = 0


func _process(_delta: float) -> void:
    _render_tick += 1
    var material := $Flash.material as ShaderMaterial
    material.set_shader_parameter("flash_phase", float(_render_tick % 2))
    if Input.is_action_just_pressed("advance"):
        phase = 1
        player_x = 64
        world_epoch = 3
        completed = true
        score = 7
'''
_CONTROLLED_NATIVE_SHADER_FILES = {
    "project.godot": _CONTROLLED_NATIVE_SHADER_PROJECT,
    "main.tscn": _CONTROLLED_NATIVE_SHADER_SCENE,
    "effects/controlled_flash.gdshader": _CONTROLLED_NATIVE_SHADER,
    "controlled_state_provider.gd": _CONTROLLED_NATIVE_STATE_PROVIDER,
}


def _controlled_native_fixture_tree_sha256() -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(_CONTROLLED_NATIVE_SHADER_FILES.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _controlled_native_state_descriptor() -> dict[str, object]:
    return {
        "schema": "flashpatch-controlled-native-state-v1",
        "provider_script": "res://controlled_state_provider.gd",
        "provider_script_sha256": f"sha256:{hashlib.sha256(_CONTROLLED_NATIVE_STATE_PROVIDER).hexdigest()}",
        "node_path": "/root/Main",
        "properties": [
            {"name": "phase", "type": "int"},
            {"name": "player_x", "type": "int"},
            {"name": "world_epoch", "type": "int"},
            {"name": "completed", "type": "bool"},
            {"name": "score", "type": "int"},
        ],
        "terminal_completion": {"property": "completed", "equals": True},
        "terminal_state_properties": ["phase", "completed"],
        "player_world_properties": ["player_x", "world_epoch"],
        "score_property": "score",
        "action_transition": {
            "action": "advance",
            "pressed_frame": 1,
            "before_frame": 0,
            "after_frame": 1,
            "property": "phase",
            "before": 0,
            "after": 1,
        },
    }


PROBE_SCRIPT = """extends Node2D

@export var flash_intensity: float = 1.0

const CAPTURE_FRAMES := 121

var _trace: Dictionary
var _output_path := ""
var _timestamps_us: Array[int] = []
var _actual_capture_timestamps_us: Array[int] = []
var _runtime_events: Array[Dictionary] = []
var _gameplay_samples: Array[Dictionary] = []
var _capture_start_us := 0
var _overlay: ColorRect
var _capture_failed := false


func _ready() -> void:
	var paths := _paths()
	if paths.is_empty():
		get_tree().quit(2)
		return
	_trace = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if _trace.is_empty():
		push_error("trace JSON is required")
		get_tree().quit(2)
		return
	_output_path = paths[1]
	if not OS.get_cmdline_user_args().has("--renderer-capture"):
		push_error("controlled probe requires --renderer-capture")
		get_tree().quit(2)
		return
	_overlay = ColorRect.new()
	_overlay.position = Vector2.ZERO
	_overlay.size = Vector2(640, 400)
	_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_overlay.color = Color(1.0, 1.0, 1.0, 0.0)
	add_child(_overlay)
	_capture_start_us = Time.get_ticks_usec()
	_run_replay()


func _run_replay() -> void:
	var capture_directory := _output_path.get_base_dir().path_join("renderer-capture")
	DirAccess.make_dir_recursive_absolute(capture_directory)
	for frame_index in range(CAPTURE_FRAMES):
		_apply_actions(frame_index)
		var flash_on := flash_intensity > 0.0 and frame_index % 2 == 0
		_overlay.color = Color(1.0, 1.0, 1.0, flash_intensity if flash_on else 0.0)
		await RenderingServer.frame_post_draw
		if not _capture_frame(capture_directory, frame_index):
			get_tree().quit(2)
			return
		_record_gameplay_state(frame_index)
		_runtime_events.append({
			"capture_index": frame_index,
			"frame_index": frame_index,
			"presentation_timestamp_us": _timestamps_us.back(),
			"timestamp_us": _timestamps_us.back(),
			"actual_capture_timestamp_us": _actual_capture_timestamps_us.back(),
			"engine_process_frame": Engine.get_process_frames(),
			"node_path": str(get_path()),
			"overlay_node_path": str(_overlay.get_path()),
			"resource_path": "res://pong.tscn",
			"script_path": "res://flashpatch_probe.gd",
			"source_line": 3,
			"property": "flash_intensity",
			"factual_value": flash_intensity,
			"applied_overlay_alpha": _overlay.color.a,
			"event_kind": "render_property",
		})
	var actions: Array = _trace.get("actions", [])
	var action_frames: Array[int] = []
	for action in actions:
		action_frames.append(int(action["frame"]))
	_write_result(_output_path, action_frames, capture_directory)
	get_tree().quit(0)


func _apply_actions(frame_index: int) -> void:
	for action in _trace.get("actions", []):
		if int(action["frame"]) != frame_index:
			continue
		var name := str(action.get("action", ""))
		if name.is_empty():
			continue
		if bool(action.get("pressed", false)):
			Input.action_press(name)
		else:
			Input.action_release(name)


func _capture_frame(directory: String, index: int) -> bool:
	var image := get_viewport().get_texture().get_image()
	image.convert(Image.FORMAT_RGB8)
	var saved := image.save_png(directory.path_join("frame_%06d.png" % index))
	if saved != OK:
		push_error("renderer frame capture failed")
		_capture_failed = true
		return false
	var actual_capture_us := Time.get_ticks_usec() - _capture_start_us
	if actual_capture_us <= 0 or (not _actual_capture_timestamps_us.is_empty() and actual_capture_us <= _actual_capture_timestamps_us.back()):
		push_error("capture clock is not strictly increasing")
		_capture_failed = true
		return false
	_actual_capture_timestamps_us.append(actual_capture_us)
	_timestamps_us.append(int(index * 1000000 / int(_trace["fixed_fps"])))
	return true


func _paths() -> PackedStringArray:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--trace")
	var output_index := arguments.find("--output")
	if trace_index < 0 or output_index < 0 or trace_index + 1 >= arguments.size() or output_index + 1 >= arguments.size():
		push_error("--trace and --output are required")
		return PackedStringArray()
	return PackedStringArray([arguments[trace_index + 1], arguments[output_index + 1]])


func _record_gameplay_state(frame_index: int) -> void:
	var ball := get_node_or_null("Ball")
	var left := get_node_or_null("Left")
	var right := get_node_or_null("Right")
	_gameplay_samples.append({
		"capture_index": frame_index,
		"presentation_timestamp_us": _timestamps_us.back(),
		"engine_process_frame": Engine.get_process_frames(),
		"left_move_down_pressed": Input.is_action_pressed("left_move_down"),
		"left_move_up_pressed": Input.is_action_pressed("left_move_up"),
		"right_move_down_pressed": Input.is_action_pressed("right_move_down"),
		"right_move_up_pressed": Input.is_action_pressed("right_move_up"),
		"ball_position": str(ball.position) if ball != null else "missing",
		"ball_direction": str(ball.direction) if ball != null else "missing",
		"ball_speed": float(ball.get("_speed")) if ball != null else "missing",
		"left_position": str(left.position) if left != null else "missing",
		"right_position": str(right.position) if right != null else "missing",
	})


func _write_result(path: String, action_frames: Array[int], capture_directory: String) -> void:
	if _capture_failed or _timestamps_us.size() != CAPTURE_FRAMES or _actual_capture_timestamps_us.size() != CAPTURE_FRAMES:
		push_error("renderer capture did not produce the declared frame timeline")
		return
	var state_samples := _gameplay_samples
	var gameplay_state := JSON.stringify(state_samples)
	var viewport_size := get_viewport().get_texture().get_size()
	var output := FileAccess.open(path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"status": "REPLAYED",
		"action_frames": action_frames,
		"gameplay_state": gameplay_state,
		"gameplay_state_samples": state_samples,
		"runtime_events": _runtime_events,
		"semantic_invariants": {
			"terminal_completion": true,
			"terminal_state": gameplay_state,
			"player_world_digest": gameplay_state,
			"score": "score_not_applicable",
		},
		"renderer_capture": {
			"frame_directory": capture_directory.get_file(),
			"timestamps_us": _timestamps_us,
			"actual_capture_timestamps_us": _actual_capture_timestamps_us,
			"capture_clock": "Time.get_ticks_usec monotonic provenance; presentation timestamps are fixed-fps media time",
			"viewport": [int(viewport_size.x), int(viewport_size.y)],
			"color_space": "sRGB/BT.709",
		},
	}, "  ") + "\\n")
"""


QUALIFICATION_PROBE_SCRIPT = """extends Node

var _trace: Dictionary
var _output_path := ""
var _timestamps_us: Array[int] = []
var _actual_capture_timestamps_us: Array[int] = []

func _ready() -> void:
	var paths := _paths()
	if paths.is_empty():
		get_tree().quit(2)
		return
	_trace = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if _trace.is_empty() or not _trace.has("fixed_fps") or not _trace.has("capture_frames"):
		push_error("qualification probe requires fixed_fps and capture_frames")
		get_tree().quit(2)
		return
	_output_path = paths[1]
	var original := load(str(_trace.get("original_main_scene", ""))) as PackedScene
	if original == null:
		push_error("qualification probe cannot load original_main_scene")
		get_tree().quit(2)
		return
	add_child(original.instantiate())
	_capture()

func _capture() -> void:
	var capture_directory := _output_path.get_base_dir().path_join("renderer-capture")
	DirAccess.make_dir_recursive_absolute(capture_directory)
	var start_us := Time.get_ticks_usec()
	for tick in range(int(_trace["capture_frames"])):
		_apply_actions(tick)
		await RenderingServer.frame_post_draw
		var image := get_viewport().get_texture().get_image()
		image.convert(Image.FORMAT_RGB8)
		if image.save_png(capture_directory.path_join("frame_%06d.png" % tick)) != OK:
			push_error("qualification renderer frame capture failed")
			get_tree().quit(2)
			return
		var actual_us := Time.get_ticks_usec() - start_us
		if actual_us <= 0 or (not _actual_capture_timestamps_us.is_empty() and actual_us <= _actual_capture_timestamps_us.back()):
			push_error("qualification capture clock is invalid")
			get_tree().quit(2)
			return
		_actual_capture_timestamps_us.append(actual_us)
		_timestamps_us.append(int(tick * 1000000 / int(_trace["fixed_fps"])))
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"status": "REPLAYED",
		"action_frames": _action_frames(),
		"gameplay_state": "qualification-state-unavailable",
		"renderer_capture": {
			"frame_directory": capture_directory.get_file(),
			"timestamps_us": _timestamps_us,
			"actual_capture_timestamps_us": _actual_capture_timestamps_us,
			"viewport": [int(get_viewport().get_texture().get_size().x), int(get_viewport().get_texture().get_size().y)],
			"color_space": "sRGB/BT.709",
		},
	}) + "\\n")
	get_tree().quit(0)

func _apply_actions(tick: int) -> void:
	for action in _trace.get("actions", []):
		if int(action.get("frame", -1)) != tick:
			continue
		var name := str(action.get("action", ""))
		if not name.is_empty() and InputMap.has_action(name):
			if bool(action.get("pressed", false)): Input.action_press(name)
			else: Input.action_release(name)

func _action_frames() -> Array[int]:
	var frames: Array[int] = []
	for action in _trace.get("actions", []): frames.append(int(action["frame"]))
	return frames

func _paths() -> PackedStringArray:
	var args := OS.get_cmdline_user_args()
	var trace_index := args.find("--trace")
	var output_index := args.find("--output")
	if trace_index < 0 or output_index < 0 or trace_index + 1 >= args.size() or output_index + 1 >= args.size(): return PackedStringArray()
	return PackedStringArray([args[trace_index + 1], args[output_index + 1]])
"""


QUALIFICATION_PROBE_SCRIPT_GODOT3 = """extends Node

var trace_data = {}
var output_path = ""
var timestamps_us = []
var actual_timestamps_us = []

func _ready():
	var paths = _paths()
	if paths.empty():
		get_tree().quit(2)
		return
	var trace_file = File.new()
	if trace_file.open(paths[0], File.READ) != OK:
		get_tree().quit(2)
		return
	var parsed = JSON.parse(trace_file.get_as_text())
	trace_file.close()
	if parsed.error != OK or typeof(parsed.result) != TYPE_DICTIONARY:
		get_tree().quit(2)
		return
	trace_data = parsed.result
	if not trace_data.has("fixed_fps") or not trace_data.has("capture_frames"):
		get_tree().quit(2)
		return
	output_path = paths[1]
	var original = load(str(trace_data.get("original_main_scene", "")))
	if original == null:
		get_tree().quit(2)
		return
	add_child(original.instance())
	_capture()

func _capture():
	var directory = output_path.get_base_dir().plus_file("renderer-capture")
	var dir = Directory.new()
	dir.make_dir_recursive(directory)
	var start_us = OS.get_ticks_usec()
	for tick in range(int(trace_data["capture_frames"])):
		_apply_actions(tick)
		yield(VisualServer, "frame_post_draw")
		var image = get_viewport().get_texture().get_data()
		image.flip_y()
		image.convert(Image.FORMAT_RGB8)
		if image.save_png(directory.plus_file("frame_%06d.png" % tick)) != OK:
			get_tree().quit(2)
			return
		var actual_us = OS.get_ticks_usec() - start_us
		if actual_us <= 0 or (not actual_timestamps_us.empty() and actual_us <= actual_timestamps_us.back()):
			get_tree().quit(2)
			return
		actual_timestamps_us.append(actual_us)
		timestamps_us.append(int(tick * 1000000 / int(trace_data["fixed_fps"])))
	var output = File.new()
	if output.open(output_path, File.WRITE) != OK:
		get_tree().quit(2)
		return
	var viewport = get_viewport().get_texture().get_size()
	output.store_string(JSON.print({"status":"REPLAYED","action_frames":_action_frames(),"gameplay_state":"qualification-state-unavailable","renderer_capture":{"frame_directory":directory.get_file(),"timestamps_us":timestamps_us,"actual_capture_timestamps_us":actual_timestamps_us,"viewport":[int(viewport.x),int(viewport.y)],"color_space":"sRGB/BT.709"}}))
	output.close()
	get_tree().quit(0)

func _apply_actions(tick):
	for action in trace_data.get("actions", []):
		if int(action.get("frame", -1)) == tick:
			var name = str(action.get("action", ""))
			if not name.empty() and InputMap.has_action(name):
				if bool(action.get("pressed", false)): Input.action_press(name)
				else: Input.action_release(name)

func _action_frames():
	var frames = []
	for action in trace_data.get("actions", []): frames.append(int(action["frame"]))
	return frames

func _paths():
	var args = OS.get_cmdline_args()
	var trace_index = args.find("--trace")
	var output_index = args.find("--output")
	if trace_index < 0 or output_index < 0 or trace_index + 1 >= args.size() or output_index + 1 >= args.size(): return []
	return [args[trace_index + 1], args[output_index + 1]]
"""


NATIVE_MAIN_CAPTURE_SCRIPT_GODOT4 = """extends Node

var _trace: Dictionary
var _output_path := ""
var _warmup_frames := 0
var _timestamps_us: Array[int] = []
var _actual_capture_timestamps_us: Array[int] = []
var _action_acknowledgements: Array[Dictionary] = []
var _pointer_acknowledgements: Array[Dictionary] = []
var _key_acknowledgements: Array[Dictionary] = []
var _runtime_events: Array[Dictionary] = []
var _scene_observations: Array[Dictionary] = []
var _transition_observed_frame := -1
var _ui_rect_observations: Array[Dictionary] = []
var _ui_selection_observations: Array[Dictionary] = []
var _ui_selection_signal_events: Array[Dictionary] = []
var _state_observations: Array[Dictionary] = []
var _current_trace_tick := -1


func _ready() -> void:
	var paths := _paths()
	if paths.is_empty():
		_fail("--trace and --output are required")
		return
	_trace = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if (
		_trace.is_empty()
		or not _trace.has("fixed_fps")
		or not _trace.has("capture_frames")
		or not _trace.has("warmup_frames")
		or not _trace.has("original_main_scene")
	):
		_fail("native-main capture requires a complete trace")
		return
	_output_path = paths[1]
	if not OS.get_cmdline_user_args().has("--renderer-capture"):
		_fail("native-main capture requires --renderer-capture")
		return
	_warmup_frames = int(_trace["warmup_frames"])
	if _warmup_frames < 0:
		_fail("native-main capture warmup_frames is invalid")
		return
	await _capture_native_main()


func _capture_native_main() -> void:
	await get_tree().process_frame
	if not _record_ui_rect_observations():
		return
	var current_scene := get_tree().current_scene
	if current_scene == null:
		_fail("native-main capture current_scene is missing")
		return
	var expected_scene := str(_trace["original_main_scene"])
	var runtime_scene := str(current_scene.scene_file_path)
	if runtime_scene != expected_scene:
		_fail("native-main capture current_scene does not match original_main_scene")
		return
	var wrapper_ancestors: Array[String] = []
	var ancestor := current_scene.get_parent()
	while ancestor != null and ancestor != get_tree().root:
		wrapper_ancestors.append(str(ancestor.get_path()))
		ancestor = ancestor.get_parent()
	if not wrapper_ancestors.is_empty():
		_fail("native-main capture found a wrapper ancestor")
		return
	var runtime_native_main := {
		"current_scene_exists": true,
		"expected_scene_file_path": expected_scene,
		"runtime_scene_file_path": runtime_scene,
		"scene_file_path_match": true,
		"current_scene_node_path": str(current_scene.get_path()),
		"wrapper_ancestor_paths": wrapper_ancestors,
		"no_wrapper_ancestor": true,
		"verification_process_frame": Engine.get_process_frames(),
	}
	var capture_directory := _output_path.get_base_dir().path_join("renderer-capture")
	if DirAccess.make_dir_recursive_absolute(capture_directory) != OK:
		_fail("native-main capture directory could not be created")
		return
	var start_us := Time.get_ticks_usec()
	if not _install_ui_selection_signal_observers():
		return
	for tick in range(_warmup_frames + int(_trace["capture_frames"])):
		_current_trace_tick = tick
		_apply_actions(tick)
		_apply_pointer_events(tick)
		_apply_key_events(tick)
		# Some valid Godot projects enable low_processor_mode and do not emit
		# frame_post_draw while visually idle. Advance one process tick and force
		# the renderer to present the factual viewport for every declared sample.
		await get_tree().process_frame
		RenderingServer.force_draw()
		if tick < _warmup_frames:
			continue
		var capture_index := tick - _warmup_frames
		if not _record_scene_observation(capture_index):
			return
		var image := get_viewport().get_texture().get_image()
		image.convert(Image.FORMAT_RGB8)
		if image.save_png(capture_directory.path_join("frame_%06d.png" % capture_index)) != OK:
			_fail("native-main renderer frame capture failed")
			return
		var actual_us := Time.get_ticks_usec() - start_us
		if actual_us <= 0 or (not _actual_capture_timestamps_us.is_empty() and actual_us <= _actual_capture_timestamps_us.back()):
			_fail("native-main actual capture clock is not strictly increasing")
			return
		_actual_capture_timestamps_us.append(actual_us)
		_timestamps_us.append(int(tick * 1000000 / int(_trace["fixed_fps"])))
		if not _record_controlled_state_observation(capture_index):
			return
		if not _record_runtime_observations(capture_index):
			return
		if not _record_ui_selection_observations(capture_index):
			return
	if _action_acknowledgements.size() != _trace.get("actions", []).size():
		_fail("native-main capture did not acknowledge every trace action")
		return
	if _pointer_acknowledgements.size() != _trace.get("pointer_events", []).size():
		_fail("native-main capture did not acknowledge every pointer event")
		return
	if _key_acknowledgements.size() != _trace.get("key_events", []).size():
		_fail("native-main capture did not acknowledge every key event")
		return
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	if output == null:
		_fail("native-main capture output could not be opened")
		return
	var viewport_size := get_viewport().get_texture().get_size()
	output.store_string(JSON.stringify({
		"status": "REPLAYED",
		"qualification_only": true,
		"scoreable": false,
		"native_equivalence": "NOT_ESTABLISHED",
		"execution_mode": "instrumented_native_main_scene_capture",
		"action_frames": _action_frames(),
		"action_acknowledgements": _action_acknowledgements,
		"pointer_acknowledgements": _pointer_acknowledgements,
		"key_acknowledgements": _key_acknowledgements,
		"runtime_events": _runtime_events,
		"scene_observations": _scene_observations,
		"scene_transition_acknowledgement": _scene_transition_acknowledgement(),
		"ui_rect_observations": _ui_rect_observations,
		"ui_selection_observations": _ui_selection_observations,
		"ui_selection_signal_events": _ui_selection_signal_events,
		"scenario_readiness": _scenario_readiness(),
		"gameplay_state": "native-main-qualification-state-unavailable",
		"state_observation_descriptor": _trace.get("state_observation", null),
		"state_observations": _state_observations,
		"native_main": runtime_native_main,
		"renderer_capture": {
			"frame_directory": capture_directory.get_file(),
			"warmup_frames": _warmup_frames,
			"capture_trace_frame_indices": _capture_trace_frame_indices(),
			"timestamps_us": _timestamps_us,
			"actual_capture_timestamps_us": _actual_capture_timestamps_us,
			"capture_clock": "Time.get_ticks_usec monotonic provenance; presentation timestamps are absolute fixed-fps trace media time",
			"viewport": [int(viewport_size.x), int(viewport_size.y)],
			"color_space": "sRGB/BT.709",
		},
	}, "  ") + "\\n")
	get_tree().quit(0)


func _record_scene_observation(tick: int) -> bool:
	var scene := get_tree().current_scene
	if scene == null:
		_fail("native-main capture current_scene is missing")
		return false
	var ancestors: Array[String] = []
	var ancestor := scene.get_parent()
	while ancestor != null and ancestor != get_tree().root:
		ancestors.append(str(ancestor.get_path()))
		ancestor = ancestor.get_parent()
	if not ancestors.is_empty():
		_fail("native-main capture found a wrapper ancestor")
		return false
	var scene_path := str(scene.scene_file_path)
	_scene_observations.append({
		"frame_index": tick,
		"scene_file_path": scene_path,
		"current_scene_node_path": str(scene.get_path()),
		"current_scene_instance_id": scene.get_instance_id(),
		"wrapper_ancestor_paths": ancestors,
	})
	var declared: Variant = _trace.get("scene_transition", null)
	if declared is Dictionary and _transition_observed_frame < 0:
		var transition: Dictionary = declared as Dictionary
		if scene_path == str(transition.get("to_scene", "")):
			_transition_observed_frame = tick
	return true


func _scene_transition_acknowledgement() -> Dictionary:
	var declared: Variant = _trace.get("scene_transition", null)
	if not declared is Dictionary:
		return {}
	var transition: Dictionary = declared as Dictionary
	return {
		"from_scene": str(transition.get("from_scene", "")),
		"to_scene": str(transition.get("to_scene", "")),
		"observed_frame": _transition_observed_frame,
		"status": "APPLIED" if _transition_observed_frame >= 0 else "MISSING",
	}


func _record_ui_rect_observations() -> bool:
	for declared_path in _trace.get("ui_rect_observations", []):
		if not declared_path is String:
			_fail("native-main UI rect path is invalid")
			return false
		var control := get_node_or_null(NodePath(declared_path)) as Control
		if control == null:
			_fail("native-main UI rect control is missing")
			return false
		var rect := control.get_global_rect()
		_ui_rect_observations.append({
			"node_path": str(declared_path),
			"rect": [rect.position.x, rect.position.y, rect.size.x, rect.size.y],
			"visible": control.is_visible_in_tree(),
			"enabled": not control.disabled,
			"viewport": [get_viewport().get_visible_rect().size.x, get_viewport().get_visible_rect().size.y],
		})
	return true


func _apply_actions(tick: int) -> void:
	for action in _trace.get("actions", []):
		if int(action.get("frame", -1)) != tick:
			continue
		var name := str(action.get("action", ""))
		var pressed := bool(action.get("pressed", false))
		var acknowledgement := {
			"frame": tick,
			"action": name,
			"pressed": pressed,
			"status": "MISSING_INPUT_MAP_ACTION",
		}
		if not name.is_empty() and InputMap.has_action(name):
			if pressed:
				Input.action_press(name)
			else:
				Input.action_release(name)
			acknowledgement["status"] = "APPLIED"
		_action_acknowledgements.append(acknowledgement)


func _apply_pointer_events(tick: int) -> void:
	for event_data in _trace.get("pointer_events", []):
		if int(event_data.get("frame", -1)) != tick:
			continue
		var x: float = float(event_data.get("x", -1.0))
		var y: float = float(event_data.get("y", -1.0))
		var acknowledgement: Dictionary = {
			"frame": tick,
			"kind": str(event_data.get("kind", "")),
			"x": x,
			"y": y,
			"status": "INVALID_POINTER_EVENT",
		}
		if acknowledgement["kind"] == "left_click" and x >= 0.0 and x <= 1.0 and y >= 0.0 and y <= 1.0:
			var visible_size: Vector2 = get_viewport().get_visible_rect().size
			var position: Vector2 = Vector2(x * visible_size.x, y * visible_size.y)
			var press: InputEventMouseButton = InputEventMouseButton.new()
			press.position = position
			press.global_position = position
			press.button_index = MOUSE_BUTTON_LEFT
			press.pressed = true
			Input.parse_input_event(press)
			var release: InputEventMouseButton = InputEventMouseButton.new()
			release.position = position
			release.global_position = position
			release.button_index = MOUSE_BUTTON_LEFT
			release.pressed = false
			Input.parse_input_event(release)
			acknowledgement["status"] = "APPLIED"
		_pointer_acknowledgements.append(acknowledgement)


func _apply_key_events(tick: int) -> void:
	for event_data in _trace.get("key_events", []):
		if int(event_data.get("frame", -1)) != tick:
			continue
		var key_name := str(event_data.get("key", ""))
		var acknowledgement: Dictionary = {
			"frame": tick,
			"key": key_name,
			"status": "INVALID_KEY_EVENT",
		}
		var keycode := KEY_NONE
		if key_name == "down":
			keycode = KEY_DOWN
		elif key_name == "up":
			keycode = KEY_UP
		elif key_name == "enter":
			keycode = KEY_ENTER
		if keycode != KEY_NONE:
			var press := InputEventKey.new()
			press.keycode = keycode
			press.pressed = true
			Input.parse_input_event(press)
			var release := InputEventKey.new()
			release.keycode = keycode
			release.pressed = false
			Input.parse_input_event(release)
			acknowledgement["status"] = "APPLIED"
		_key_acknowledgements.append(acknowledgement)


func _record_ui_selection_observations(tick: int) -> bool:
	for declared_path in _trace.get("ui_selection_observations", []):
		if not declared_path is String:
			_fail("native-main UI selection path is invalid")
			return false
		var option_button := get_node_or_null(NodePath(declared_path)) as OptionButton
		if option_button == null:
			_fail("native-main UI selection OptionButton is missing")
			return false
		var selected := option_button.selected
		var item_count := option_button.item_count
		if selected < 0 or selected >= item_count:
			_fail("native-main UI selection index is invalid")
			return false
		var popup := option_button.get_popup()
		if popup == null:
			_fail("native-main UI selection PopupMenu is missing")
			return false
		_ui_selection_observations.append({
			"frame_index": tick,
			"node_path": str(option_button.get_path()),
			"selected_index": selected,
			"selected_text": option_button.get_item_text(selected),
			"popup_visible": popup.visible,
			"popup_focused_index": popup.get_focused_item(),
		})
	return true


func _install_ui_selection_signal_observers() -> bool:
	for declared_path in _trace.get("ui_selection_observations", []):
		var option_button := get_node_or_null(NodePath(str(declared_path))) as OptionButton
		if option_button == null:
			_fail("native-main UI selection OptionButton is missing")
			return false
		option_button.item_selected.connect(_record_ui_selection_signal.bind(str(option_button.get_path())))
	return true


func _record_ui_selection_signal(index: int, node_path: String) -> void:
	_ui_selection_signal_events.append({
		"trace_frame_index": _current_trace_tick,
		"node_path": node_path,
		"selected_index": index,
	})


func _scenario_readiness() -> Dictionary:
	var declared: Variant = _trace.get("scenario_readiness", null)
	if declared == null:
		return {"declared": false, "satisfied": false, "reason": "scenario_readiness_not_declared"}
	if not declared is Dictionary:
		return {"declared": true, "satisfied": false, "reason": "scenario_readiness_invalid"}
	var readiness: Dictionary = declared as Dictionary
	var paths: Array = readiness.get("required_node_paths", [])
	var groups: Dictionary = readiness.get("required_group_minimums", {})
	var visibility: Array = readiness.get("required_visible", [])
	var selections: Array = readiness.get("required_option_selection", [])
	var missing_paths: Array[String] = []
	for path_value in paths:
		if not path_value is String or get_node_or_null(NodePath(path_value)) == null:
			missing_paths.append(str(path_value))
	var observed_groups: Dictionary = {}
	var insufficient_groups: Dictionary = {}
	for group_name in groups:
		var observed: int = get_tree().get_nodes_in_group(StringName(group_name)).size()
		observed_groups[str(group_name)] = observed
		if observed < int(groups[group_name]):
			insufficient_groups[str(group_name)] = {"required": int(groups[group_name]), "observed": observed}
	var visibility_mismatches: Array[Dictionary] = []
	for requirement in visibility:
		if not requirement is Dictionary:
			visibility_mismatches.append({"node_path": str(requirement), "reason": "invalid_requirement"})
			continue
		var requirement_data: Dictionary = requirement as Dictionary
		var control: CanvasItem = get_node_or_null(NodePath(str(requirement_data.get("node_path", "")))) as CanvasItem
		if control == null or control.visible != bool(requirement_data.get("visible", false)):
			visibility_mismatches.append({"node_path": str(requirement_data.get("node_path", "")), "expected": bool(requirement_data.get("visible", false)), "observed": null if control == null else control.visible})
	var selection_mismatches: Array[Dictionary] = []
	for requirement in selections:
		if not requirement is Dictionary:
			selection_mismatches.append({"node_path": str(requirement), "reason": "invalid_requirement"})
			continue
		var selection_data: Dictionary = requirement as Dictionary
		var option_button := get_node_or_null(NodePath(str(selection_data.get("node_path", "")))) as OptionButton
		var expected_index := int(selection_data.get("selected_index", -1))
		var expected_text := str(selection_data.get("selected_text", ""))
		if option_button == null or option_button.selected != expected_index or option_button.get_item_text(option_button.selected) != expected_text:
			selection_mismatches.append({"node_path": str(selection_data.get("node_path", "")), "expected_index": expected_index, "expected_text": expected_text, "observed_index": null if option_button == null else option_button.selected, "observed_text": null if option_button == null else option_button.get_item_text(option_button.selected)})
	return {
		"declared": true,
		"satisfied": missing_paths.is_empty() and insufficient_groups.is_empty() and visibility_mismatches.is_empty() and selection_mismatches.is_empty(),
		"missing_node_paths": missing_paths,
		"observed_group_counts": observed_groups,
		"insufficient_groups": insufficient_groups,
		"visibility_mismatches": visibility_mismatches,
		"selection_mismatches": selection_mismatches,
	}


func _record_runtime_observations(tick: int) -> bool:
	var active_scene := get_tree().current_scene
	if active_scene == null:
		_fail("native-main runtime observation scene is missing")
		return false
	var active_scene_path := str(active_scene.scene_file_path)
	for declared_value in _trace.get("runtime_observations", []):
		if not declared_value is Dictionary:
			_fail("native-main runtime observation is invalid")
			return false
		var declared: Dictionary = declared_value as Dictionary
		var observation_kind := str(declared.get("kind", "render_property"))
		var scoped_scene := str(declared.get("scene_file_path", ""))
		if not scoped_scene.is_empty() and scoped_scene != active_scene_path:
			continue
		var node_path := str(declared.get("node_path", ""))
		var property_name := str(declared.get("property", ""))
		var node := get_node_or_null(NodePath(node_path))
		if node == null:
			_fail("native-main runtime observation node is missing")
			return false
		if observation_kind == "shader_parameter":
			var source_path := str(declared.get("source_path", ""))
			if active_scene_path != source_path:
				_fail("native-main shader observation source scene does not match")
				return false
			var canvas_item := node as CanvasItem
			if canvas_item == null or not canvas_item.material is ShaderMaterial:
				_fail("native-main shader observation material is not ShaderMaterial")
				return false
			var shader_material := canvas_item.material as ShaderMaterial
			var shader := shader_material.shader
			if shader == null or str(shader.resource_path) != str(declared.get("shader_path", "")):
				_fail("native-main shader observation shader does not match")
				return false
			var shader_value: Variant = shader_material.get_shader_parameter(property_name)
			if not shader_value is int and not shader_value is float:
				_fail("native-main shader observation parameter is not numeric")
				return false
			_runtime_events.append({
				"frame_index": tick,
				"node_path": node_path,
				"source_path": source_path,
				"resource_path": str(declared.get("resource_path", "")),
				"source_line": int(declared.get("source_line", 0)),
				"shader_path": str(shader.resource_path),
				"property": property_name,
				"factual_value": float(shader_value),
				"event_kind": "shader_parameter",
			})
			continue
		var script: Variant = node.get_script()
		if script == null or str(script.resource_path) != str(declared.get("script_path", "")):
			_fail("native-main runtime observation script does not match")
			return false
		var value: Variant = node.get(property_name)
		if not value is int and not value is float:
			_fail("native-main runtime observation property is not numeric")
			return false
		_runtime_events.append({
			"frame_index": tick,
			"node_path": node_path,
			"script_path": str(script.resource_path),
			"resource_path": str(declared.get("resource_path", "")),
			"source_line": int(declared.get("source_line", 0)),
			"property": property_name,
			"factual_value": float(value),
			"event_kind": "render_property",
		})
	return true


func _record_controlled_state_observation(tick: int) -> bool:
	var descriptor: Variant = _trace.get("state_observation", null)
	if descriptor == null:
		return true
	if not descriptor is Dictionary:
		_fail("native-main state observation descriptor is invalid")
		return false
	var declaration: Dictionary = descriptor as Dictionary
	var node := get_node_or_null(NodePath(str(declaration.get("node_path", ""))))
	if node == null:
		_fail("native-main controlled state provider node is missing")
		return false
	var script: Variant = node.get_script()
	if script == null or str(script.resource_path) != str(declaration.get("provider_script", "")):
		_fail("native-main controlled state provider script does not match")
		return false
	var values: Dictionary = {}
	for property_descriptor in declaration.get("properties", []):
		if not property_descriptor is Dictionary:
			_fail("native-main controlled state property descriptor is invalid")
			return false
		var property_name := str(property_descriptor.get("name", ""))
		var property_type := str(property_descriptor.get("type", ""))
		var value: Variant = node.get(property_name)
		if (
			property_type == "bool" and not value is bool
			or property_type == "int" and (not value is int or value is bool)
			or property_type == "float" and not value is float
			or property_type == "string" and not value is String
			or property_type not in ["bool", "int", "float", "string"]
		):
			_fail("native-main controlled state property type does not match")
			return false
		values[property_name] = value
	_state_observations.append({
		"frame_index": tick,
		"presentation_timestamp_us": _timestamps_us.back(),
		"values": values,
	})
	return true


func _action_frames() -> Array[int]:
	var frames: Array[int] = []
	for action in _trace.get("actions", []):
		frames.append(int(action["frame"]))
	return frames


func _capture_trace_frame_indices() -> Array[int]:
	var indices: Array[int] = []
	for capture_index in range(int(_trace["capture_frames"])):
		indices.append(_warmup_frames + capture_index)
	return indices


func _paths() -> PackedStringArray:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--trace")
	var output_index := arguments.find("--output")
	if trace_index < 0 or output_index < 0 or trace_index + 1 >= arguments.size() or output_index + 1 >= arguments.size():
		return PackedStringArray()
	return PackedStringArray([arguments[trace_index + 1], arguments[output_index + 1]])


func _fail(message: String) -> void:
	push_error(message)
	get_tree().quit(2)
"""


NATIVE_MAIN_UI_RECT_PREFLIGHT_SCRIPT_GODOT4 = """extends Node

var _trace: Dictionary
var _output_path := ""


func _ready() -> void:
	var paths := _paths()
	if paths.is_empty():
		_fail("UI rect preflight requires --ui-rect-trace and --ui-rect-output")
		return
	if OS.get_cmdline_user_args().has("--renderer-capture"):
		_fail("UI rect preflight rejects renderer capture")
		return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if not parsed is Dictionary:
		_fail("UI rect preflight trace must be an object")
		return
	_trace = parsed as Dictionary
	_output_path = paths[1]
	await get_tree().process_frame
	_probe()


func _probe() -> void:
	var current_scene := get_tree().current_scene
	if current_scene == null:
		_fail("UI rect preflight current scene is missing")
		return
	var expected_scene := str(_trace.get("original_main_scene", ""))
	if str(current_scene.scene_file_path) != expected_scene:
		_fail("UI rect preflight current scene does not match configured main scene")
		return
	var wrapper_ancestors: Array[String] = []
	var ancestor := current_scene.get_parent()
	while ancestor != null and ancestor != get_tree().root:
		wrapper_ancestors.append(str(ancestor.get_path()))
		ancestor = ancestor.get_parent()
	if not wrapper_ancestors.is_empty():
		_fail("UI rect preflight found a wrapper ancestor")
		return
	var viewport_size := get_viewport().get_visible_rect().size
	if viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		_fail("UI rect preflight viewport is invalid")
		return
	var observations: Array[Dictionary] = []
	for declared_path in _trace.get("control_paths", []):
		if not declared_path is String:
			_fail("UI rect preflight path is invalid")
			return
		var button := get_node_or_null(NodePath(declared_path)) as BaseButton
		if button == null:
			_fail("UI rect preflight declared path is not a BaseButton")
			return
		var rect := button.get_global_rect()
		if (
			not is_finite(rect.position.x)
			or not is_finite(rect.position.y)
			or not is_finite(rect.size.x)
			or not is_finite(rect.size.y)
			or rect.size.x <= 0.0
			or rect.size.y <= 0.0
		):
			_fail("UI rect preflight BaseButton rect is invalid")
			return
		observations.append({
			"node_path": str(button.get_path()),
			"rect": [rect.position.x, rect.position.y, rect.size.x, rect.size.y],
			"visible": button.is_visible_in_tree(),
			"enabled": not button.disabled,
			"viewport": [viewport_size.x, viewport_size.y],
		})
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	if output == null:
		_fail("UI rect preflight output could not be opened")
		return
	output.store_string(JSON.stringify({
		"schema": "flashpatch-native-main-ui-rect-preflight-v1",
		"status": "PROBED",
		"execution_mode": "metadata_only_native_main_ui_rect_preflight",
		"qualification_only": true,
		"scoreable": false,
		"renderer_png_capture": false,
		"detector_executed": false,
		"original_main_scene": expected_scene,
		"current_scene_node_path": str(current_scene.get_path()),
		"wrapper_ancestor_paths": wrapper_ancestors,
		"observations": observations,
	}, "  ") + "\\n")
	get_tree().quit(0)


func _paths() -> PackedStringArray:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--ui-rect-trace")
	var output_index := arguments.find("--ui-rect-output")
	if (
		trace_index < 0
		or output_index < 0
		or trace_index + 1 >= arguments.size()
		or output_index + 1 >= arguments.size()
	):
		return PackedStringArray()
	return PackedStringArray([arguments[trace_index + 1], arguments[output_index + 1]])


func _fail(message: String) -> void:
	push_error(message)
	get_tree().quit(2)
"""


NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4 = """extends Node

var _trace: Dictionary
var _output_path := ""


func _ready() -> void:
	var paths := _paths()
	if paths.is_empty():
		_fail("OptionButton popup preflight requires --option-button-popup-trace and --option-button-popup-output")
		return
	if OS.get_cmdline_user_args().has("--renderer-capture"):
		_fail("OptionButton popup preflight rejects renderer capture")
		return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(paths[0]))
	if not parsed is Dictionary:
		_fail("OptionButton popup preflight trace must be an object")
		return
	_trace = parsed as Dictionary
	_output_path = paths[1]
	await get_tree().process_frame
	await _probe()


func _probe() -> void:
	var current_scene := get_tree().current_scene
	if current_scene == null:
		_fail("OptionButton popup preflight current scene is missing")
		return
	var expected_scene := str(_trace.get("original_main_scene", ""))
	if str(current_scene.scene_file_path) != expected_scene:
		_fail("OptionButton popup preflight current scene does not match configured main scene")
		return
	var wrapper_ancestors: Array[String] = []
	var ancestor := current_scene.get_parent()
	while ancestor != null and ancestor != get_tree().root:
		wrapper_ancestors.append(str(ancestor.get_path()))
		ancestor = ancestor.get_parent()
	if not wrapper_ancestors.is_empty():
		_fail("OptionButton popup preflight found a wrapper ancestor")
		return
	var viewport_size := get_viewport().get_visible_rect().size
	if viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		_fail("OptionButton popup preflight viewport is invalid")
		return
	var declared_path := str(_trace.get("option_button_path", ""))
	var option_button := get_node_or_null(NodePath(declared_path)) as OptionButton
	if option_button == null:
		_fail("OptionButton popup preflight declared path is missing or not an OptionButton")
		return
	var button_rect := option_button.get_global_rect()
	if not _valid_rect(button_rect):
		_fail("OptionButton popup preflight OptionButton rect is invalid")
		return
	if not option_button.is_visible_in_tree() or option_button.disabled:
		_fail("OptionButton popup preflight OptionButton is inactive")
		return
	var activation: Dictionary = _trace.get("activation", {})
	var activation_kind := str(activation.get("kind", ""))
	var actual_activation: Dictionary = {"kind": activation_kind}
	if activation_kind == "input_event_left_click":
		var center := button_rect.get_center()
		var press := InputEventMouseButton.new()
		press.button_index = MOUSE_BUTTON_LEFT
		press.pressed = true
		press.position = center
		press.global_position = center
		Input.parse_input_event(press)
		var release := InputEventMouseButton.new()
		release.button_index = MOUSE_BUTTON_LEFT
		release.pressed = false
		release.position = center
		release.global_position = center
		Input.parse_input_event(release)
		actual_activation["button_center"] = [center.x, center.y]
	elif activation_kind == "explicit_popup_call":
		option_button.show_popup()
	else:
		_fail("OptionButton popup preflight activation is not declared")
		return
	await get_tree().process_frame
	var popup := option_button.get_popup()
	# PopupMenu inherits Window in Godot 4.  It is not a CanvasItem, so its
	# visibility and screen geometry must be read through the Window API.
	if popup == null or not popup.visible:
		_fail("OptionButton popup preflight popup is absent or inactive")
		return
	if popup.get_item_count() <= 0:
		_fail("OptionButton popup preflight popup has no items")
		return
	var button_observation := {
		"node_path": str(option_button.get_path()),
		"rect": [button_rect.position.x, button_rect.position.y, button_rect.size.x, button_rect.size.y],
		"visible": option_button.is_visible_in_tree(),
		"enabled": not option_button.disabled,
		"viewport": [viewport_size.x, viewport_size.y],
	}
	var items: Array[Dictionary] = []
	for index in range(popup.get_item_count()):
		items.append({
			"index": index,
			"text": popup.get_item_text(index),
			"enabled": not popup.is_item_disabled(index),
		})
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	if output == null:
		_fail("OptionButton popup preflight output could not be opened")
		return
	output.store_string(JSON.stringify({
		"schema": "flashpatch-native-main-option-button-popup-preflight-v2",
		"status": "PROBED",
		"execution_mode": "metadata_only_native_main_option_button_popup_preflight",
		"qualification_only": true,
		"scoreable": false,
		"renderer_png_capture": false,
		"detector_executed": false,
		"original_main_scene": expected_scene,
		"current_scene_node_path": str(current_scene.get_path()),
		"wrapper_ancestor_paths": wrapper_ancestors,
		"option_button": button_observation,
		"activation": actual_activation,
		"items": items,
	}, "  ") + "\\n")
	get_tree().quit(0)


func _valid_rect(rect: Rect2) -> bool:
	return (
		is_finite(rect.position.x)
		and is_finite(rect.position.y)
		and is_finite(rect.size.x)
		and is_finite(rect.size.y)
		and rect.size.x > 0.0
		and rect.size.y > 0.0
	)


func _paths() -> PackedStringArray:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--option-button-popup-trace")
	var output_index := arguments.find("--option-button-popup-output")
	if (
		trace_index < 0
		or output_index < 0
		or trace_index + 1 >= arguments.size()
		or output_index + 1 >= arguments.size()
	):
		return PackedStringArray()
	return PackedStringArray([arguments[trace_index + 1], arguments[output_index + 1]])


func _fail(message: String) -> void:
	push_error(message)
	get_tree().quit(2)
"""


@dataclass(frozen=True)
class ControlledPong:
    project: Path
    contract: Path
    trace: Path
    mutation_script: Path
    source: dict[str, str]


@dataclass(frozen=True)
class ControlledPongRun:
    controlled: ControlledPong
    receipt_path: Path
    receipt: dict[str, object]


@dataclass(frozen=True)
class ControlledSparta:
    project: Path
    contract: Path
    trace: Path
    mutation_script: Path
    source: dict[str, str]


@dataclass(frozen=True)
class ControlledSpartaRun:
    controlled: ControlledSparta
    receipt_path: Path
    receipt: dict[str, object]


@dataclass(frozen=True)
class CaptureOnlyQualification:
    project: Path
    trace: Path
    original_main_scene: str
    visual_candidates: tuple[dict[str, object], ...]
    source_provenance: dict[str, str] | None


@dataclass(frozen=True)
class NativeMainCaptureQualification:
    """An instrumented copy whose configured main scene remains the upstream scene.

    This is a structural precondition for a later native-main renderer lane,
    not proof that an autoload-instrumented run equals an uninstrumented run.
    """

    project: Path
    trace: Path
    original_main_scene: str
    native_main: dict[str, object]


@dataclass(frozen=True)
class NativeTscnTokenPatch:
    """The single declared ShaderMaterial numeric token a candidate may change.

    This is deliberately a source-location contract, not a patch payload.  The
    integrity receipt binds hashes and the fact of one numeric-token change but
    never serializes either numeric literal.
    """

    source_path: str
    parameter: str
    source_line: int


@dataclass(frozen=True)
class NativeMainUiRectPreflight:
    """A copied Godot-4 project wired for metadata-only BaseButton probing."""

    project: Path
    trace: Path
    original_main_scene: str
    source_binding: dict[str, str]


@dataclass(frozen=True)
class NativeMainOptionButtonPopupPreflight:
    """A copied Godot-4 project wired for OptionButton popup metadata only."""

    project: Path
    trace: Path
    original_main_scene: str
    source_binding: dict[str, str]


class NativeMainUiRectPreflightRunner(GodotReplayRunner):
    """Run one native main-scene metadata probe without capturing pixels."""

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("native-main UI rect preflight requires a usable display")
        trace = Path(trace).resolve()
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"native-main UI rect preflight output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        # The native process requires completed import artifacts.  Godot 4.7
        # can abort after headless import if a caller's X11 display leaks into
        # that process, so import with display variables removed and launch
        # the actual metadata probe only after it finishes successfully.
        import_environment = os.environ.copy()
        import_environment.pop("DISPLAY", None)
        import_environment.pop("WAYLAND_DISPLAY", None)
        prepared = subprocess.run(
            [str(self.godot_binary), "--headless", "--path", str(self.project), "--import"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
            env=import_environment,
        )
        if prepared.returncode != 0:
            raise RuntimeError(
                f"Godot UI rect preflight project import exited {prepared.returncode}:\n{prepared.stdout}"
            )
        display_arguments = ["--display-driver", "x11"] if os.environ.get("DISPLAY") else []
        completed = subprocess.run(
            [
                str(self.godot_binary),
                *display_arguments,
                "--path", str(self.project),
                "--",
                "--ui-rect-trace", str(trace),
                "--ui-rect-output", str(output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Godot UI rect preflight exited {completed.returncode}:\n{completed.stdout}"
            )
        if not output.is_file():
            raise RuntimeError("Godot UI rect preflight did not create its metadata output")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Godot UI rect preflight returned unreadable metadata") from exc
        if not isinstance(result, dict) or result.get("status") != "PROBED":
            raise RuntimeError("Godot UI rect preflight returned invalid metadata")
        return result


class NativeMainOptionButtonPopupPreflightRunner(GodotReplayRunner):
    """Run one native OptionButton popup metadata probe without pixels."""

    MAX_STDOUT_LOG_BYTES = 64 * 1024

    @classmethod
    def _write_stdout_log(cls, output: Path, stdout: str | bytes | None) -> Path:
        """Persist only a bounded diagnostic when the probe cannot produce metadata."""
        log_path = output.with_suffix(".stdout.log")
        stdout_bytes = (
            b"" if stdout is None
            else stdout.encode("utf-8", errors="replace") if isinstance(stdout, str)
            else stdout
        )
        truncation = b"\n[FlashPatch: stdout truncated]\n"
        if len(stdout_bytes) > cls.MAX_STDOUT_LOG_BYTES:
            stdout_bytes = stdout_bytes[: cls.MAX_STDOUT_LOG_BYTES - len(truncation)] + truncation
        log_path.write_bytes(stdout_bytes)
        return log_path

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("native-main OptionButton popup preflight requires a usable display")
        trace = Path(trace).resolve()
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"native-main OptionButton popup preflight output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        display_arguments = ["--display-driver", "x11"] if os.environ.get("DISPLAY") else []
        # PopupMenu is a Window. Godot 4.7 can complete a headless import and
        # then abort while closing the editor singleton. Import under the same
        # X11 renderer contract as the factual probe, rather than treating
        # that headless-editor shutdown bug as a project failure.
        try:
            prepared = subprocess.run(
                [str(self.godot_binary), *display_arguments, "--path", str(self.project), "--import"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            log_path = self._write_stdout_log(output, exc.stdout)
            raise RuntimeError(
                f"Godot OptionButton popup preflight project import timed out after "
                f"{self.timeout_seconds} seconds; see stdout log: {log_path}"
            ) from exc
        if prepared.returncode != 0:
            log_path = self._write_stdout_log(output, prepared.stdout)
            raise RuntimeError(
                f"Godot OptionButton popup preflight project import exited {prepared.returncode}; "
                f"see stdout log: {log_path}"
            )
        try:
            completed = subprocess.run(
                [
                    str(self.godot_binary),
                    *display_arguments,
                    "--path", str(self.project),
                    "--",
                    "--option-button-popup-trace", str(trace),
                    "--option-button-popup-output", str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            log_path = self._write_stdout_log(output, exc.stdout)
            raise RuntimeError(
                f"Godot OptionButton popup preflight timed out after {self.timeout_seconds} seconds; "
                f"see stdout log: {log_path}"
            ) from exc
        if completed.returncode != 0:
            log_path = self._write_stdout_log(output, completed.stdout)
            raise RuntimeError(
                f"Godot OptionButton popup preflight exited {completed.returncode}; "
                f"see stdout log: {log_path}"
            )
        if not output.is_file():
            log_path = self._write_stdout_log(output, completed.stdout)
            raise RuntimeError(
                "Godot OptionButton popup preflight did not create its metadata output; "
                f"see stdout log: {log_path}"
            )
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Godot OptionButton popup preflight returned unreadable metadata") from exc
        if not isinstance(result, dict) or result.get("status") != "PROBED":
            raise RuntimeError("Godot OptionButton popup preflight returned invalid metadata")
        return result


def _declared_godot_major(project_config: str) -> int:
    match = re.search(r"^config_version\s*=\s*(\d+)\s*$", project_config, re.MULTILINE)
    if match is None:
        raise ValueError("qualification requires project.godot config_version")
    return {4: 3, 5: 4}.get(int(match.group(1)), 0)


def _visual_export_candidates(project: Path, *, major: int) -> tuple[dict[str, object], ...]:
    """Inventory only pre-existing, project-local visual exported controls.

    This is deliberately a static *eligibility* inventory.  It is not runtime
    attribution and a listed control is never evidence that it caused a frame
    hazard.  A candidate location is always a project-relative POSIX path; a
    symlink that resolves outside ``project`` is excluded rather than letting a
    copied probe mint source provenance for an unrelated file.
    """
    taxonomy = re.compile(r"(?:intensity|brightness|flash|blink|frequency|duration|contrast|alpha|opacity|color|radius|area)", re.IGNORECASE)
    if major == 4:
        pattern = re.compile(
            r"^\s*@export\s+var\s+(?P<name>[A-Za-z_]\w*)(?:\s*:\s*(?P<type>[A-Za-z_]\w*))?",
            re.MULTILINE,
        )
    else:
        pattern = re.compile(
            r"^\s*export(?:\((?P<type>[A-Za-z_]\w*)[^)]*\))?\s+var\s+(?P<name>[A-Za-z_]\w*)",
            re.MULTILINE,
        )
    candidates: list[dict[str, object]] = []
    root = project.resolve()
    for source in sorted(project.rglob("*.gd")):
        try:
            source.resolve().relative_to(root)
        except ValueError:
            continue
        relative = source.relative_to(project).as_posix()
        for line, text in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            match = pattern.match(text)
            if match is not None and taxonomy.search(match.group("name")):
                candidates.append({
                    "source": relative,
                    "line": line,
                    "parameter": match.group("name"),
                    "declared_type": match.groupdict().get("type"),
                })
    return tuple(candidates)


def _git_stdout(project: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git failure"
        raise ValueError(f"clean pinned source verification failed: {detail}")
    return completed.stdout.strip()


def _normalize_repository_url(value: str) -> str:
    return value.rstrip("/").removesuffix(".git")


def _tracked_source_tree_sha256(project: Path) -> str:
    digest = hashlib.sha256()
    tracked = _git_stdout(project, "ls-files", "-z").split("\0")
    for relative in sorted(path for path in tracked if path):
        source = project / relative
        if source.is_symlink():
            content = os.fsencode(os.readlink(source))
        elif source.is_file():
            content = source.read_bytes()
        else:
            raise ValueError(f"clean pinned source verification found missing tracked file: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _native_source_tree_sha256(project: Path, *, original_project_config: bytes | None = None) -> str:
    """Hash executable project source while excluding generated and injected state."""
    root = project.resolve()
    digest = hashlib.sha256()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
        and ".godot" not in path.relative_to(root).parts
        and ".flashpatch" not in path.relative_to(root).parts
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"native-main source tree contains symlinked file: {relative}")
        content = original_project_config if relative == "project.godot" and original_project_config is not None else path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


_NATIVE_TREE_IGNORED_PARTS = frozenset({".git", ".godot"})
_NATIVE_TSCN_NUMBER = rb"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"


def _native_project_file_manifest(project: Path) -> dict[str, bytes]:
    """Return every regular, non-generated file and reject ambiguous topology."""
    root_argument = Path(project)
    if root_argument.is_symlink() or not root_argument.is_dir():
        raise RuntimeError("native-main project tree root is not a regular directory")
    root = root_argument.resolve()
    manifest: dict[str, bytes] = {}
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root)
        allowed_names: list[str] = []
        for name in sorted(names):
            child = directory_path / name
            relative = child.relative_to(root)
            if any(part in _NATIVE_TREE_IGNORED_PARTS for part in relative.parts):
                continue
            if child.is_symlink():
                raise RuntimeError(f"native-main project tree contains symlink: {relative.as_posix()}")
            if not child.is_dir():
                raise RuntimeError(f"native-main project tree contains non-directory entry: {relative.as_posix()}")
            allowed_names.append(name)
        names[:] = allowed_names
        if any(part in _NATIVE_TREE_IGNORED_PARTS for part in relative_directory.parts):
            continue
        for name in sorted(files):
            child = directory_path / name
            relative = child.relative_to(root)
            if any(part in _NATIVE_TREE_IGNORED_PARTS for part in relative.parts):
                continue
            if child.is_symlink():
                raise RuntimeError(f"native-main project tree contains symlink: {relative.as_posix()}")
            if not child.is_file():
                raise RuntimeError(f"native-main project tree contains non-regular file: {relative.as_posix()}")
            manifest[relative.as_posix()] = child.read_bytes()
    return manifest


def _native_project_manifest_sha256(manifest: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(manifest):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(manifest[relative])
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _native_tscn_patch_relative_path(patch: NativeTscnTokenPatch) -> str:
    if (
        not isinstance(patch.source_path, str)
        or not patch.source_path.startswith("res://")
        or not patch.source_path.endswith(".tscn")
        or not isinstance(patch.parameter, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", patch.parameter) is None
        or isinstance(patch.source_line, bool)
        or not isinstance(patch.source_line, int)
        or patch.source_line < 1
    ):
        raise RuntimeError("native-main candidate token declaration is invalid")
    relative = Path(patch.source_path.removeprefix("res://"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or ".flashpatch" in relative.parts:
        raise RuntimeError("native-main candidate token declaration escapes source tree")
    return relative.as_posix()


def _native_tscn_numeric_token(
    source: bytes,
    patch: NativeTscnTokenPatch,
    *,
    label: str,
) -> tuple[int, int, bytes]:
    """Locate exactly one finite numeric ShaderMaterial assignment by source line."""
    parameter = re.escape(patch.parameter.encode("ascii"))
    assignment = re.compile(rb"^[ \t]*shader_parameter/" + parameter + rb"[ \t]*=")
    numeric = re.compile(
        rb"^[ \t]*shader_parameter/" + parameter + rb"[ \t]*=[ \t]*(" + _NATIVE_TSCN_NUMBER + rb")[ \t]*$"
    )
    lines = source.splitlines(keepends=True)
    matches: list[tuple[int, int, bytes]] = []
    offset = 0
    for line_number, line in enumerate(lines, start=1):
        body = line[:-2] if line.endswith(b"\r\n") else line[:-1] if line.endswith(b"\n") else line
        if assignment.match(body) is not None:
            match = numeric.fullmatch(body)
            if match is None:
                raise RuntimeError(f"native-main {label} token is not one exact numeric assignment")
            token = match.group(1)
            try:
                value = float(token.decode("ascii"))
            except ValueError as exc:
                raise RuntimeError(f"native-main {label} token is not numeric") from exc
            if not math.isfinite(value):
                raise RuntimeError(f"native-main {label} token is not finite")
            matches.append((line_number, offset + match.start(1), token))
        offset += len(line)
    if len(matches) != 1 or matches[0][0] != patch.source_line:
        raise RuntimeError(f"native-main {label} token is missing, redundant, or at the wrong source line")
    return matches[0]


def _sealed_native_main_baseline(qualification: NativeMainCaptureQualification) -> tuple[dict[str, bytes], str]:
    """Validate the source-derived baseline before comparing a replay copy."""
    baseline = qualification.project
    original_config = baseline / ".flashpatch" / "upstream-project.godot"
    if original_config.is_symlink() or not original_config.is_file():
        raise RuntimeError("native-main baseline upstream-project.godot is missing")
    original_bytes = original_config.read_bytes()
    try:
        expected_config = _append_native_capture_autoload(original_bytes.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("native-main baseline upstream-project.godot is not UTF-8") from exc
    manifest = _native_project_file_manifest(baseline)
    if manifest.get("project.godot") != expected_config:
        raise RuntimeError("native-main baseline injected project.godot bytes changed")
    declared = qualification.native_main
    expected_source_tree = declared.get("upstream_source_tree_sha256")
    if (
        not isinstance(expected_source_tree, str)
        or declared.get("copied_source_tree_sha256") != expected_source_tree
        or _native_source_tree_sha256(baseline, original_project_config=original_bytes) != expected_source_tree
    ):
        raise RuntimeError("native-main baseline source tree bytes changed")
    sealed_hash = _native_project_manifest_sha256(manifest)
    if declared.get("sealed_project_tree_sha256") != sealed_hash:
        raise RuntimeError("native-main baseline sealed project tree bytes changed")
    return manifest, sealed_hash


def _verify_native_main_candidate_tree(
    qualification: NativeMainCaptureQualification,
    observed_project: Path,
    *,
    patch: NativeTscnTokenPatch | None = None,
) -> dict[str, object]:
    """Bind a factual copy or one declared ``.tscn`` numeric-token candidate.

    It is intentionally not connected to replay or patch selection yet.  Its
    only job is to make the project-copy boundary independently checkable.
    """
    baseline, baseline_hash = _sealed_native_main_baseline(qualification)
    observed = _native_project_file_manifest(Path(observed_project))
    for relative in sorted(set(observed) - set(baseline)):
        if not relative.endswith(".uid"):
            continue
        source_relative = relative.removesuffix(".uid")
        if (
            source_relative in baseline
            and Path(source_relative).suffix in {".gd", ".gdshader"}
            and re.fullmatch(rb"uid://[a-z0-9]+\n?", observed[relative]) is not None
        ):
            # Godot 4.4+ writes deterministic sidecar UIDs during import.  They
            # are generated metadata, not candidate source.  Only an adjacent,
            # previously sealed script/shader sidecar with the engine UID
            # grammar is ignored; every other extra file remains fatal.
            del observed[relative]
    if observed.get("project.godot") != baseline.get("project.godot"):
        raise RuntimeError("native-main observed project.godot does not match exact injected bytes")
    if set(observed) != set(baseline):
        raise RuntimeError("native-main observed project tree has missing or extra files")
    observed_hash = _native_project_manifest_sha256(observed)
    if patch is None:
        changed = [path for path in baseline if baseline[path] != observed[path]]
        if changed:
            raise RuntimeError("native-main factual project tree bytes changed")
        return {
            "schema": "flashpatch-native-main-project-tree-binding-v1",
            "mode": "FACTUAL_BYTE_EXACT",
            "baseline_project_tree_sha256": baseline_hash,
            "observed_project_tree_sha256": observed_hash,
            "file_count": len(baseline),
            "changed_file_count": 0,
        }
    relative = _native_tscn_patch_relative_path(patch)
    if relative not in baseline:
        raise RuntimeError("native-main candidate token source is missing from sealed project tree")
    changed = [path for path in baseline if baseline[path] != observed[path]]
    if changed != [relative]:
        raise RuntimeError("native-main candidate must change exactly one declared source file")
    baseline_line, baseline_start, baseline_token = _native_tscn_numeric_token(baseline[relative], patch, label="factual")
    candidate_line, candidate_start, candidate_token = _native_tscn_numeric_token(observed[relative], patch, label="candidate")
    if baseline_line != candidate_line or baseline_token == candidate_token:
        raise RuntimeError("native-main candidate token did not make one numeric change")
    if (
        baseline[relative][:baseline_start] != observed[relative][:candidate_start]
        or baseline[relative][baseline_start + len(baseline_token):] != observed[relative][candidate_start + len(candidate_token):]
    ):
        raise RuntimeError("native-main candidate changed bytes outside the declared numeric token")
    return {
        "schema": "flashpatch-native-main-project-tree-binding-v1",
        "mode": "CANDIDATE_ONE_TSCN_NUMERIC_TOKEN",
        "baseline_project_tree_sha256": baseline_hash,
        "observed_project_tree_sha256": observed_hash,
        "file_count": len(baseline),
        "changed_file_count": 1,
        "source": {
            "path": patch.source_path,
            "parameter": patch.parameter,
            "source_line": patch.source_line,
            "factual_file_sha256": f"sha256:{hashlib.sha256(baseline[relative]).hexdigest()}",
            "candidate_file_sha256": f"sha256:{hashlib.sha256(observed[relative]).hexdigest()}",
            "change_kind": "ONE_FINITE_NUMERIC_TOKEN",
        },
    }


def _clean_pinned_source_provenance(project: Path, *, repository: str, revision: str) -> dict[str, str]:
    """Return only machine-checked provenance for a clean pinned Git checkout."""
    if not repository or not revision:
        raise ValueError("clean pinned source verification requires repository and revision together")
    actual_repository = _git_stdout(project, "remote", "get-url", "origin")
    if _normalize_repository_url(actual_repository) != _normalize_repository_url(repository):
        raise ValueError("clean pinned source repository does not match expected repository")
    actual_revision = _git_stdout(project, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise ValueError("clean pinned source revision does not match expected revision")
    if _git_stdout(project, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("clean pinned source checkout is dirty")
    return {
        "status": "VERIFIED_CLEAN_PINNED_SOURCE",
        "repository": repository,
        "revision": actual_revision,
        "git_tree_object_id": _git_stdout(project, "rev-parse", "HEAD^{tree}"),
        "source_tree_sha256": _tracked_source_tree_sha256(project),
    }


def materialize_capture_only_qualification(
    upstream_project: Path,
    destination: Path,
    *,
    fixed_fps: int = 60,
    capture_frames: int = 121,
    actions: list[dict[str, object]] | None = None,
    repository: str | None = None,
    revision: str | None = None,
) -> CaptureOnlyQualification:
    """Stage a Godot-4 capture probe without altering upstream game source.

    This is a screen qualification lane only.  It deliberately emits neither
    runtime attribution nor gameplay invariants, so a hazardous result cannot
    be promoted to a source-patch PASS.
    """
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    if (repository is None) != (revision is None):
        raise ValueError("qualification source provenance requires repository and revision together")
    project_config = upstream_project / "project.godot"
    if not project_config.is_file():
        raise ValueError("upstream_project must contain project.godot")
    if destination.exists():
        raise FileExistsError(f"qualification destination already exists: {destination}")
    if isinstance(fixed_fps, bool) or not isinstance(fixed_fps, int) or fixed_fps <= 0:
        raise ValueError("qualification fixed_fps must be positive")
    if isinstance(capture_frames, bool) or not isinstance(capture_frames, int) or capture_frames < 2:
        raise ValueError("qualification capture_frames must be at least two")
    source_config = project_config.read_text(encoding="utf-8")
    source_provenance = (
        _clean_pinned_source_provenance(upstream_project, repository=repository, revision=revision)
        if repository is not None and revision is not None
        else None
    )
    major = _declared_godot_major(source_config)
    if major not in {3, 4}:
        raise ValueError("qualification supports declared Godot 3 or 4 projects only")
    marker = 'run/main_scene="'
    start = source_config.find(marker)
    if start < 0:
        raise ValueError("qualification requires project.godot run/main_scene")
    end = source_config.find('"', start + len(marker))
    if end < 0:
        raise ValueError("qualification project.godot run/main_scene is malformed")
    original_main_scene = source_config[start + len(marker):end]
    if not original_main_scene.startswith("res://"):
        original_main_scene = f"res://{original_main_scene}"
    shutil.copytree(upstream_project, destination, ignore=shutil.ignore_patterns(".git", ".godot", ".claude"))
    probe_dir = destination / ".flashpatch"
    probe_dir.mkdir()
    probe_script = QUALIFICATION_PROBE_SCRIPT_GODOT3 if major == 3 else QUALIFICATION_PROBE_SCRIPT
    (probe_dir / "qualification.gd").write_text(probe_script, encoding="utf-8")
    (probe_dir / "qualification.tscn").write_text(
        '[gd_scene load_steps=2 format=2]\n\n[ext_resource path="res://.flashpatch/qualification.gd" type="Script" id=1]\n\n[node name="FlashPatchQualification" type="Node"]\nscript = ExtResource( 1 )\n'
        if major == 3 else '[gd_scene load_steps=2 format=3]\n\n[ext_resource type="Script" path="res://.flashpatch/qualification.gd" id="1"]\n\n[node name="FlashPatchQualification" type="Node"]\nscript = ExtResource("1")\n',
        encoding="utf-8",
    )
    trace = probe_dir / "trace.json"
    trace.write_text(json.dumps({
        "fixed_fps": fixed_fps,
        "capture_frames": capture_frames,
        "original_main_scene": original_main_scene,
        "actions": actions or [],
    }, indent=2) + "\n", encoding="utf-8")
    copied_config = destination / "project.godot"
    copied_text = copied_config.read_text(encoding="utf-8")
    copied_config.write_text(
        copied_text[:start] + 'run/main_scene="res://.flashpatch/qualification.tscn"' + copied_text[end + 1:],
        encoding="utf-8",
    )
    return CaptureOnlyQualification(
        destination,
        trace,
        original_main_scene,
        _visual_export_candidates(upstream_project, major=major),
        source_provenance,
    )


def _append_native_capture_autoload(project_config: str) -> str:
    entry = 'FlashPatchNativeMainCapture="*res://.flashpatch/native_main_capture.gd"'
    if entry in project_config:
        raise ValueError("native-main capture autoload name already exists")
    section = re.search(r"^\[autoload\]\s*$", project_config, re.MULTILINE)
    if section is None:
        suffix = "" if project_config.endswith("\n") else "\n"
        return f"{project_config}{suffix}\n[autoload]\n{entry}\n"
    insert_at = project_config.find("\n", section.end())
    if insert_at < 0:
        insert_at = len(project_config)
        separator = "\n"
    else:
        insert_at += 1
        separator = ""
    return f"{project_config[:insert_at]}{entry}\n{separator}{project_config[insert_at:]}"


def _append_ui_rect_preflight_autoload(project_config: str) -> str:
    entry = 'FlashPatchUiRectPreflight="*res://.flashpatch/ui_rect_preflight.gd"'
    if "FlashPatchUiRectPreflight" in project_config:
        raise ValueError("native-main UI rect preflight autoload name already exists")
    section = re.search(r"^\[autoload\]\s*$", project_config, re.MULTILINE)
    if section is None:
        suffix = "" if project_config.endswith("\n") else "\n"
        return f"{project_config}{suffix}\n[autoload]\n{entry}\n"
    insert_at = project_config.find("\n", section.end())
    if insert_at < 0:
        insert_at = len(project_config)
        separator = "\n"
    else:
        insert_at += 1
        separator = ""
    return f"{project_config[:insert_at]}{entry}\n{separator}{project_config[insert_at:]}"


def _validated_ui_rect_control_paths(control_paths: list[str]) -> list[str]:
    if not isinstance(control_paths, list) or not control_paths:
        raise ValueError("native-main UI rect preflight requires declared Control paths")
    if len(set(control_paths)) != len(control_paths):
        raise ValueError("native-main UI rect preflight Control paths must be unique")
    for path in control_paths:
        if not isinstance(path, str) or not path.startswith("/root/") or "\x00" in path or ":" in path:
            raise ValueError("native-main UI rect preflight Control path must start with /root/")
        segments = path.split("/")[2:]
        if not segments or any(not segment or segment in {".", ".."} for segment in segments):
            raise ValueError("native-main UI rect preflight Control path is invalid")
    return list(control_paths)


def materialize_native_main_ui_rect_preflight(
    upstream_project: Path,
    destination: Path,
    *,
    control_paths: list[str],
) -> NativeMainUiRectPreflight:
    """Copy a Godot-4 project and add only a metadata-producing autoload probe."""
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    project_config = upstream_project / "project.godot"
    if project_config.is_symlink() or not project_config.is_file():
        raise ValueError("native-main UI rect preflight requires a regular project.godot")
    if destination.exists():
        raise FileExistsError(f"native-main UI rect preflight destination already exists: {destination}")
    declared_paths = _validated_ui_rect_control_paths(control_paths)
    source_config = project_config.read_text(encoding="utf-8")
    if _declared_godot_major(source_config) != 4:
        raise ValueError("native-main UI rect preflight supports declared Godot 4 projects only")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', source_config, re.MULTILINE)
    if match is None:
        raise ValueError("native-main UI rect preflight requires project.godot run/main_scene")
    original_main_scene = match.group("scene")
    if not original_main_scene.startswith("res://"):
        original_main_scene = f"res://{original_main_scene}"
    relative_scene = Path(original_main_scene.removeprefix("res://"))
    if relative_scene.is_absolute() or ".." in relative_scene.parts:
        raise ValueError("native-main UI rect preflight main scene escapes the project")
    source_scene = upstream_project / relative_scene
    if source_scene.is_symlink() or not source_scene.is_file():
        raise ValueError("native-main UI rect preflight configured main scene is missing")
    try:
        source_scene.resolve().relative_to(upstream_project)
    except ValueError as exc:
        raise ValueError("native-main UI rect preflight main scene escapes the project") from exc
    source_scene_bytes = source_scene.read_bytes()
    expected_config = _append_ui_rect_preflight_autoload(source_config)
    shutil.copytree(
        upstream_project,
        destination,
        ignore=shutil.ignore_patterns(".git", ".godot", ".claude"),
    )
    probe_dir = destination / ".flashpatch"
    probe_dir.mkdir()
    trace = probe_dir / "ui-rect-preflight.json"
    trace.write_text(
        json.dumps(
            {
                "schema": "flashpatch-native-main-ui-rect-preflight-trace-v1",
                "original_main_scene": original_main_scene,
                "control_paths": declared_paths,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    probe_script = probe_dir / "ui_rect_preflight.gd"
    probe_script.write_text(NATIVE_MAIN_UI_RECT_PREFLIGHT_SCRIPT_GODOT4, encoding="utf-8")
    copied_config = destination / "project.godot"
    copied_config.write_text(expected_config, encoding="utf-8")
    copied_scene = destination / relative_scene
    if copied_scene.is_symlink() or copied_scene.read_bytes() != source_scene_bytes:
        raise RuntimeError("native-main UI rect preflight did not preserve main scene bytes")
    binding = {
        "upstream_project_config_sha256": f"sha256:{hashlib.sha256(source_config.encode('utf-8')).hexdigest()}",
        "copied_project_config_sha256": f"sha256:{hashlib.sha256(expected_config.encode('utf-8')).hexdigest()}",
        "upstream_main_scene_sha256": f"sha256:{hashlib.sha256(source_scene_bytes).hexdigest()}",
        "copied_main_scene_sha256": f"sha256:{hashlib.sha256(copied_scene.read_bytes()).hexdigest()}",
        "injected_probe_script_sha256": f"sha256:{hashlib.sha256(probe_script.read_bytes()).hexdigest()}",
        "trace_sha256": f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}",
    }
    return NativeMainUiRectPreflight(destination, trace, original_main_scene, binding)


def _require_unchanged_ui_rect_preflight(preflight: NativeMainUiRectPreflight) -> dict[str, object]:
    try:
        trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native-main UI rect preflight trace is unreadable") from exc
    if (
        not isinstance(trace, dict)
        or set(trace) != {"schema", "original_main_scene", "control_paths"}
        or trace.get("schema") != "flashpatch-native-main-ui-rect-preflight-trace-v1"
        or trace.get("original_main_scene") != preflight.original_main_scene
    ):
        raise RuntimeError("native-main UI rect preflight trace contract is invalid")
    try:
        declared_paths = _validated_ui_rect_control_paths(trace.get("control_paths"))
    except ValueError as exc:
        raise RuntimeError("native-main UI rect preflight trace contract is invalid") from exc
    config_path = preflight.project / "project.godot"
    scene_path = preflight.project / preflight.original_main_scene.removeprefix("res://")
    probe_path = preflight.project / ".flashpatch" / "ui_rect_preflight.gd"
    if (
        config_path.is_symlink() or scene_path.is_symlink() or probe_path.is_symlink()
        or not config_path.is_file() or not scene_path.is_file() or not probe_path.is_file()
    ):
        raise RuntimeError("native-main UI rect preflight source binding is missing")
    config_bytes = config_path.read_bytes()
    scene_bytes = scene_path.read_bytes()
    if (
        f"sha256:{hashlib.sha256(config_bytes).hexdigest()}" != preflight.source_binding.get("copied_project_config_sha256")
        or f"sha256:{hashlib.sha256(scene_bytes).hexdigest()}" != preflight.source_binding.get("upstream_main_scene_sha256")
        or f"sha256:{hashlib.sha256(scene_bytes).hexdigest()}" != preflight.source_binding.get("copied_main_scene_sha256")
        or f"sha256:{hashlib.sha256(probe_path.read_bytes()).hexdigest()}" != preflight.source_binding.get("injected_probe_script_sha256")
        or f"sha256:{hashlib.sha256(preflight.trace.read_bytes()).hexdigest()}" != preflight.source_binding.get("trace_sha256")
    ):
        raise RuntimeError("native-main UI rect preflight source bytes changed")
    config = config_bytes.decode("utf-8")
    if _declared_godot_major(config) != 4:
        raise RuntimeError("native-main UI rect preflight no longer declares Godot 4")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', config, re.MULTILINE)
    configured_scene = None if match is None else match.group("scene")
    if configured_scene is not None and not configured_scene.startswith("res://"):
        configured_scene = f"res://{configured_scene}"
    if configured_scene != preflight.original_main_scene:
        raise RuntimeError("native-main UI rect preflight changed run/main_scene")
    return {**trace, "control_paths": declared_paths}


def execute_native_main_ui_rect_preflight(
    preflight: NativeMainUiRectPreflight,
    output: Path,
    *,
    godot_binary: Path | None = None,
) -> dict[str, object]:
    """Execute and validate metadata only; this path never invokes a detector."""
    trace = _require_unchanged_ui_rect_preflight(preflight)
    output = Path(output).resolve()
    runner = NativeMainUiRectPreflightRunner(preflight.project, godot_binary=godot_binary)
    result = runner.replay(preflight.trace, output)
    required_keys = {
        "schema", "status", "execution_mode", "qualification_only", "scoreable",
        "renderer_png_capture", "detector_executed", "original_main_scene",
        "current_scene_node_path", "wrapper_ancestor_paths", "observations",
    }
    if set(result) != required_keys or (
        result.get("schema") != "flashpatch-native-main-ui-rect-preflight-v1"
        or result.get("status") != "PROBED"
        or result.get("execution_mode") != "metadata_only_native_main_ui_rect_preflight"
        or result.get("qualification_only") is not True
        or result.get("scoreable") is not False
        or result.get("renderer_png_capture") is not False
        or result.get("detector_executed") is not False
        or result.get("original_main_scene") != preflight.original_main_scene
        or result.get("wrapper_ancestor_paths") != []
    ):
        raise RuntimeError("native-main UI rect preflight metadata contract is invalid")
    current_scene = result.get("current_scene_node_path")
    if not isinstance(current_scene, str) or not current_scene.startswith("/root/") or "/" in current_scene.removeprefix("/root/"):
        raise RuntimeError("native-main UI rect preflight current scene topology is invalid")
    observations = result.get("observations")
    paths = trace["control_paths"]
    if not isinstance(observations, list) or len(observations) != len(paths):
        raise RuntimeError("native-main UI rect preflight observations are incomplete")
    for declared_path, observation in zip(paths, observations):
        if not isinstance(observation, dict) or set(observation) != {
            "node_path", "rect", "visible", "enabled", "viewport"
        } or observation.get("node_path") != declared_path:
            raise RuntimeError("native-main UI rect preflight observation is invalid")
        rect, viewport = observation.get("rect"), observation.get("viewport")
        if (
            not isinstance(rect, list) or len(rect) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in rect)
            or float(rect[2]) <= 0.0 or float(rect[3]) <= 0.0
            or not isinstance(viewport, list) or len(viewport) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0 for value in viewport)
            or not isinstance(observation.get("visible"), bool)
            or not isinstance(observation.get("enabled"), bool)
        ):
            raise RuntimeError("native-main UI rect preflight observation is invalid")
    receipt = {
        **result,
        "trace_sha256": f"sha256:{hashlib.sha256(preflight.trace.read_bytes()).hexdigest()}",
        "source_binding": dict(preflight.source_binding),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _append_option_button_popup_preflight_autoload(project_config: str) -> str:
    entry = 'FlashPatchOptionButtonPopupPreflight="*res://.flashpatch/option_button_popup_preflight.gd"'
    if "FlashPatchOptionButtonPopupPreflight" in project_config:
        raise ValueError("native-main OptionButton popup preflight autoload name already exists")
    section = re.search(r"^\[autoload\]\s*$", project_config, re.MULTILINE)
    if section is None:
        suffix = "" if project_config.endswith("\n") else "\n"
        return f"{project_config}{suffix}\n[autoload]\n{entry}\n"
    insert_at = project_config.find("\n", section.end())
    if insert_at < 0:
        insert_at = len(project_config)
        separator = "\n"
    else:
        insert_at += 1
        separator = ""
    return f"{project_config[:insert_at]}{entry}\n{separator}{project_config[insert_at:]}"


def _validated_option_button_path(option_button_path: str) -> str:
    if not isinstance(option_button_path, str) or not option_button_path.startswith("/root/"):
        raise ValueError("native-main OptionButton popup preflight requires a declared /root/ OptionButton path")
    if "\x00" in option_button_path or ":" in option_button_path:
        raise ValueError("native-main OptionButton popup preflight OptionButton path is invalid")
    segments = option_button_path.split("/")[2:]
    if not segments or any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError("native-main OptionButton popup preflight OptionButton path is invalid")
    return option_button_path


def _validated_popup_activation(activation: dict[str, object]) -> dict[str, str]:
    if not isinstance(activation, dict) or set(activation) != {"kind"}:
        raise ValueError("native-main OptionButton popup preflight requires one declared activation kind")
    kind = activation.get("kind")
    if kind not in {"input_event_left_click", "explicit_popup_call"}:
        raise ValueError("native-main OptionButton popup preflight activation kind is invalid")
    return {"kind": kind}


def materialize_native_main_option_button_popup_preflight(
    upstream_project: Path,
    destination: Path,
    *,
    option_button_path: str,
    activation: dict[str, object] | None = None,
) -> NativeMainOptionButtonPopupPreflight:
    """Copy a Godot-4 project and inject one metadata-only popup autoload.

    The configured main scene and every copied source file stay unchanged.  The
    only copy-local changes are the autoload entry and the probe/trace files.
    """
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    project_config = upstream_project / "project.godot"
    if project_config.is_symlink() or not project_config.is_file():
        raise ValueError("native-main OptionButton popup preflight requires a regular project.godot")
    if destination.exists():
        raise FileExistsError(f"native-main OptionButton popup preflight destination already exists: {destination}")
    declared_path = _validated_option_button_path(option_button_path)
    declared_activation = _validated_popup_activation(activation or {"kind": "input_event_left_click"})
    source_config = project_config.read_text(encoding="utf-8")
    if _declared_godot_major(source_config) != 4:
        raise ValueError("native-main OptionButton popup preflight supports declared Godot 4 projects only")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', source_config, re.MULTILINE)
    if match is None:
        raise ValueError("native-main OptionButton popup preflight requires project.godot run/main_scene")
    original_main_scene = match.group("scene")
    if not original_main_scene.startswith("res://"):
        original_main_scene = f"res://{original_main_scene}"
    relative_scene = Path(original_main_scene.removeprefix("res://"))
    if relative_scene.is_absolute() or ".." in relative_scene.parts:
        raise ValueError("native-main OptionButton popup preflight main scene escapes the project")
    source_scene = upstream_project / relative_scene
    if source_scene.is_symlink() or not source_scene.is_file():
        raise ValueError("native-main OptionButton popup preflight configured main scene is missing")
    try:
        source_scene.resolve().relative_to(upstream_project)
    except ValueError as exc:
        raise ValueError("native-main OptionButton popup preflight main scene escapes the project") from exc
    source_scene_bytes = source_scene.read_bytes()
    expected_config = _append_option_button_popup_preflight_autoload(source_config)
    shutil.copytree(
        upstream_project,
        destination,
        ignore=shutil.ignore_patterns(".git", ".godot", ".claude"),
    )
    probe_dir = destination / ".flashpatch"
    probe_dir.mkdir()
    trace = probe_dir / "option-button-popup-preflight.json"
    trace.write_text(
        json.dumps(
            {
                "schema": "flashpatch-native-main-option-button-popup-preflight-trace-v1",
                "original_main_scene": original_main_scene,
                "option_button_path": declared_path,
                "activation": declared_activation,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    probe_script = probe_dir / "option_button_popup_preflight.gd"
    probe_script.write_text(NATIVE_MAIN_OPTION_BUTTON_POPUP_PREFLIGHT_SCRIPT_GODOT4, encoding="utf-8")
    copied_config = destination / "project.godot"
    copied_config.write_text(expected_config, encoding="utf-8")
    copied_scene = destination / relative_scene
    if copied_scene.is_symlink() or copied_scene.read_bytes() != source_scene_bytes:
        raise RuntimeError("native-main OptionButton popup preflight did not preserve main scene bytes")
    source_binding = {
        "upstream_project_config_sha256": f"sha256:{hashlib.sha256(source_config.encode('utf-8')).hexdigest()}",
        "copied_project_config_sha256": f"sha256:{hashlib.sha256(expected_config.encode('utf-8')).hexdigest()}",
        "upstream_main_scene_sha256": f"sha256:{hashlib.sha256(source_scene_bytes).hexdigest()}",
        "copied_main_scene_sha256": f"sha256:{hashlib.sha256(copied_scene.read_bytes()).hexdigest()}",
        "injected_probe_script_sha256": f"sha256:{hashlib.sha256(probe_script.read_bytes()).hexdigest()}",
        "trace_sha256": f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}",
    }
    return NativeMainOptionButtonPopupPreflight(destination, trace, original_main_scene, source_binding)


def _require_unchanged_option_button_popup_preflight(
    preflight: NativeMainOptionButtonPopupPreflight,
) -> dict[str, object]:
    try:
        trace = json.loads(preflight.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native-main OptionButton popup preflight trace is unreadable") from exc
    if (
        not isinstance(trace, dict)
        or set(trace) != {"schema", "original_main_scene", "option_button_path", "activation"}
        or trace.get("schema") != "flashpatch-native-main-option-button-popup-preflight-trace-v1"
        or trace.get("original_main_scene") != preflight.original_main_scene
    ):
        raise RuntimeError("native-main OptionButton popup preflight trace contract is invalid")
    try:
        declared_path = _validated_option_button_path(trace.get("option_button_path"))
        declared_activation = _validated_popup_activation(trace.get("activation"))
    except ValueError as exc:
        raise RuntimeError("native-main OptionButton popup preflight trace contract is invalid") from exc
    config_path = preflight.project / "project.godot"
    scene_path = preflight.project / preflight.original_main_scene.removeprefix("res://")
    probe_path = preflight.project / ".flashpatch" / "option_button_popup_preflight.gd"
    if (
        config_path.is_symlink() or scene_path.is_symlink() or probe_path.is_symlink()
        or not config_path.is_file() or not scene_path.is_file() or not probe_path.is_file()
    ):
        raise RuntimeError("native-main OptionButton popup preflight source binding is missing")
    config_bytes = config_path.read_bytes()
    scene_bytes = scene_path.read_bytes()
    if (
        f"sha256:{hashlib.sha256(config_bytes).hexdigest()}" != preflight.source_binding.get("copied_project_config_sha256")
        or f"sha256:{hashlib.sha256(scene_bytes).hexdigest()}" != preflight.source_binding.get("upstream_main_scene_sha256")
        or f"sha256:{hashlib.sha256(scene_bytes).hexdigest()}" != preflight.source_binding.get("copied_main_scene_sha256")
        or f"sha256:{hashlib.sha256(probe_path.read_bytes()).hexdigest()}" != preflight.source_binding.get("injected_probe_script_sha256")
        or f"sha256:{hashlib.sha256(preflight.trace.read_bytes()).hexdigest()}" != preflight.source_binding.get("trace_sha256")
    ):
        raise RuntimeError("native-main OptionButton popup preflight source bytes changed")
    config = config_bytes.decode("utf-8")
    if _declared_godot_major(config) != 4:
        raise RuntimeError("native-main OptionButton popup preflight no longer declares Godot 4")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', config, re.MULTILINE)
    configured_scene = None if match is None else match.group("scene")
    if configured_scene is not None and not configured_scene.startswith("res://"):
        configured_scene = f"res://{configured_scene}"
    if configured_scene != preflight.original_main_scene:
        raise RuntimeError("native-main OptionButton popup preflight changed run/main_scene")
    return {
        **trace,
        "option_button_path": declared_path,
        "activation": declared_activation,
    }


def _valid_popup_metadata_rect(value: object, *, viewport: list[object] | None = None) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
        return False
    if float(value[2]) <= 0.0 or float(value[3]) <= 0.0:
        return False
    if viewport is not None and (float(value[0]) < 0.0 or float(value[1]) < 0.0):
        return False
    return True


def _valid_popup_metadata_viewport(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0.0
            for item in value
        )
    )


def execute_native_main_option_button_popup_preflight(
    preflight: NativeMainOptionButtonPopupPreflight,
    output: Path,
    *,
    godot_binary: Path | None = None,
) -> dict[str, object]:
    """Execute an exact, non-scoreable OptionButton popup metadata receipt."""
    trace = _require_unchanged_option_button_popup_preflight(preflight)
    output = Path(output).resolve()
    runner = NativeMainOptionButtonPopupPreflightRunner(preflight.project, godot_binary=godot_binary)
    result = runner.replay(preflight.trace, output)
    required_keys = {
        "schema", "status", "execution_mode", "qualification_only", "scoreable",
        "renderer_png_capture", "detector_executed", "original_main_scene",
        "current_scene_node_path", "wrapper_ancestor_paths", "option_button",
        "activation", "items",
    }
    if set(result) != required_keys or (
        result.get("schema") != "flashpatch-native-main-option-button-popup-preflight-v2"
        or result.get("status") != "PROBED"
        or result.get("execution_mode") != "metadata_only_native_main_option_button_popup_preflight"
        or result.get("qualification_only") is not True
        or result.get("scoreable") is not False
        or result.get("renderer_png_capture") is not False
        or result.get("detector_executed") is not False
        or result.get("original_main_scene") != preflight.original_main_scene
        or result.get("wrapper_ancestor_paths") != []
    ):
        raise RuntimeError("native-main OptionButton popup preflight metadata contract is invalid")
    current_scene = result.get("current_scene_node_path")
    if not isinstance(current_scene, str) or not current_scene.startswith("/root/") or "/" in current_scene.removeprefix("/root/"):
        raise RuntimeError("native-main OptionButton popup preflight current scene topology is invalid")
    option_button = result.get("option_button")
    if not isinstance(option_button, dict) or set(option_button) != {
        "node_path", "rect", "visible", "enabled", "viewport"
    } or option_button.get("node_path") != trace["option_button_path"]:
        raise RuntimeError("native-main OptionButton popup preflight OptionButton observation is invalid")
    button_viewport = option_button.get("viewport")
    if (
        not _valid_popup_metadata_viewport(button_viewport)
        or not _valid_popup_metadata_rect(option_button.get("rect"), viewport=button_viewport)
        or option_button.get("visible") is not True
        or option_button.get("enabled") is not True
    ):
        raise RuntimeError("native-main OptionButton popup preflight OptionButton observation is invalid")
    activation = result.get("activation")
    expected_kind = trace["activation"]["kind"]
    if not isinstance(activation, dict):
        raise RuntimeError("native-main OptionButton popup preflight activation receipt is invalid")
    if expected_kind == "input_event_left_click":
        center = activation.get("button_center")
        if set(activation) != {"kind", "button_center"} or activation.get("kind") != expected_kind or not isinstance(center, list) or len(center) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in center
        ):
            raise RuntimeError("native-main OptionButton popup preflight activation receipt is invalid")
        rect = option_button["rect"]
        if not math.isclose(float(center[0]), float(rect[0]) + float(rect[2]) / 2.0) or not math.isclose(float(center[1]), float(rect[1]) + float(rect[3]) / 2.0):
            raise RuntimeError("native-main OptionButton popup preflight activation receipt is invalid")
    elif set(activation) != {"kind"} or activation.get("kind") != expected_kind:
        raise RuntimeError("native-main OptionButton popup preflight activation receipt is invalid")
    items = result.get("items")
    if not isinstance(items, list) or not items:
        raise RuntimeError("native-main OptionButton popup preflight popup items are absent")
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"index", "text", "enabled"}:
            raise RuntimeError("native-main OptionButton popup preflight popup item is invalid")
        if (
            item.get("index") != index
            or not isinstance(item.get("text"), str)
            or not isinstance(item.get("enabled"), bool)
        ):
            raise RuntimeError("native-main OptionButton popup preflight popup item is invalid")
    receipt = {
        **result,
        "trace_sha256": f"sha256:{hashlib.sha256(preflight.trace.read_bytes()).hexdigest()}",
        "source_binding": dict(preflight.source_binding),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def materialize_native_main_capture_qualification(
    upstream_project: Path,
    destination: Path,
    *,
    fixed_fps: int = 60,
    capture_frames: int = 121,
    warmup_frames: int = 0,
    actions: list[dict[str, object]] | None = None,
    launch_arguments: list[str] | None = None,
    pointer_events: list[dict[str, object]] | None = None,
    key_events: list[dict[str, object]] | None = None,
    scenario_readiness: dict[str, object] | None = None,
    runtime_observations: list[dict[str, object]] | None = None,
    scene_transition: dict[str, object] | None = None,
    ui_rect_observations: list[str] | None = None,
    ui_selection_observations: list[str] | None = None,
    _controlled_state_observation: dict[str, object] | None = None,
) -> NativeMainCaptureQualification:
    """Copy a Godot-4 project while retaining its configured main scene bytes.

    The generated autoload captures the renderer-owned viewport while the
    configured upstream main scene remains the SceneTree current scene.  This
    still does not establish behavioral equivalence with an uninstrumented
    upstream process.
    """
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    if _controlled_state_observation is not None:
        if _controlled_state_observation != _controlled_native_state_descriptor():
            raise ValueError("controlled native state observation descriptor is not the frozen fixture contract")
        observed_files: dict[str, bytes] = {}
        for path in sorted(upstream_project.rglob("*")):
            if path.is_symlink():
                raise ValueError("controlled native fixture rejects symlinks and external project linkage")
            if path.is_file():
                observed_files[path.relative_to(upstream_project).as_posix()] = path.read_bytes()
        if observed_files != _CONTROLLED_NATIVE_SHADER_FILES:
            raise ValueError("controlled native fixture tree does not match the frozen local fixture")
    project_config = upstream_project / "project.godot"
    if not project_config.is_file() or destination.exists():
        raise ValueError("native-main qualification requires a new destination and project.godot")
    if isinstance(fixed_fps, bool) or not isinstance(fixed_fps, int) or fixed_fps <= 0:
        raise ValueError("native-main qualification fixed_fps must be positive")
    if isinstance(capture_frames, bool) or not isinstance(capture_frames, int) or capture_frames < 2:
        raise ValueError("native-main qualification capture_frames must be at least two")
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int) or warmup_frames < 0:
        raise ValueError("native-main qualification warmup_frames must be a non-negative integer")
    replay_frames = warmup_frames + capture_frames
    trace_actions = actions or []
    if not isinstance(trace_actions, list) or any(not isinstance(action, dict) for action in trace_actions):
        raise ValueError("native-main qualification actions must be dictionaries")
    for action in trace_actions:
        frame = action.get("frame")
        name = action.get("action")
        pressed = action.get("pressed")
        if (
            isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= replay_frames
            or not isinstance(name, str) or not name
            or not isinstance(pressed, bool)
        ):
            raise ValueError("native-main qualification action is invalid or outside replay range")
    trace_launch_arguments = launch_arguments or []
    if not isinstance(trace_launch_arguments, list) or any(
        not isinstance(argument, str) or not argument or "\x00" in argument
        or argument in {"--", "--trace", "--output", "--renderer-capture"}
        or any(argument.startswith(f"{reserved}=") for reserved in {"--trace", "--output", "--renderer-capture"})
        for argument in trace_launch_arguments
    ):
        raise ValueError("native-main qualification launch arguments are invalid or override capture wiring")
    trace_pointer_events = pointer_events or []
    if not isinstance(trace_pointer_events, list):
        raise ValueError("native-main qualification pointer events must be a list")
    for event in trace_pointer_events:
        if (
            not isinstance(event, dict)
            or isinstance(event.get("frame"), bool) or not isinstance(event.get("frame"), int)
            or int(event["frame"]) < 0 or int(event["frame"]) >= replay_frames
            or event.get("kind") != "left_click"
            or isinstance(event.get("x"), bool) or not isinstance(event.get("x"), (int, float))
            or isinstance(event.get("y"), bool) or not isinstance(event.get("y"), (int, float))
            or not 0.0 <= float(event["x"]) <= 1.0 or not 0.0 <= float(event["y"]) <= 1.0
        ):
            raise ValueError("native-main qualification pointer event is invalid or outside replay range")
    trace_key_events = key_events or []
    if not isinstance(trace_key_events, list):
        raise ValueError("native-main qualification key events must be a list")
    for event in trace_key_events:
        if (
            not isinstance(event, dict)
            or isinstance(event.get("frame"), bool) or not isinstance(event.get("frame"), int)
            or int(event["frame"]) < 0 or int(event["frame"]) >= replay_frames
            or event.get("key") not in {"down", "up", "enter"}
        ):
            raise ValueError("native-main qualification key event is invalid or outside replay range")
    trace_runtime_observations = runtime_observations or []
    if not isinstance(trace_runtime_observations, list):
        raise ValueError("native-main qualification runtime observations must be a list")
    observation_identity: set[tuple[str, str]] = set()
    for observation in trace_runtime_observations:
        observation_kind = observation.get("kind", "render_property") if isinstance(observation, dict) else None
        if observation_kind == "shader_parameter":
            if (
                set(observation) - {"kind", "node_path", "property", "source_path", "resource_path", "source_line", "shader_path", "scene_file_path"}
                or not isinstance(observation.get("node_path"), str) or not observation["node_path"].startswith("/root/")
                or not isinstance(observation.get("property"), str) or not observation["property"]
                or not isinstance(observation.get("source_path"), str) or not observation["source_path"].startswith("res://") or not observation["source_path"].endswith(".tscn")
                or not isinstance(observation.get("resource_path"), str) or not observation["resource_path"].startswith("res://")
                or isinstance(observation.get("source_line"), bool) or not isinstance(observation.get("source_line"), int) or observation["source_line"] < 1
                or not isinstance(observation.get("shader_path"), str) or not observation["shader_path"].startswith("res://") or not observation["shader_path"].endswith(".gdshader")
                or observation.get("scene_file_path") is not None and (not isinstance(observation.get("scene_file_path"), str) or not observation["scene_file_path"].startswith("res://"))
            ):
                raise ValueError("native-main qualification shader runtime observation is invalid")
        elif observation_kind != "render_property" or (
            not isinstance(observation, dict)
            or not isinstance(observation.get("node_path"), str) or not observation["node_path"].startswith("/root/")
            or not isinstance(observation.get("property"), str) or not observation["property"]
            or not isinstance(observation.get("script_path"), str) or not observation["script_path"].startswith("res://")
            or not isinstance(observation.get("resource_path"), str) or not observation["resource_path"].startswith("res://")
            or isinstance(observation.get("source_line"), bool) or not isinstance(observation.get("source_line"), int) or observation["source_line"] < 1
            or observation.get("scene_file_path") is not None and (not isinstance(observation.get("scene_file_path"), str) or not observation["scene_file_path"].startswith("res://"))
        ):
            raise ValueError("native-main qualification runtime observation is invalid")
        identity = (observation["node_path"], observation["property"])
        if identity in observation_identity:
            raise ValueError("native-main qualification runtime observations must be unique")
        observation_identity.add(identity)
    trace_ui_rect_observations = ui_rect_observations or []
    if (
        not isinstance(trace_ui_rect_observations, list)
        or any(not isinstance(path, str) or not path.startswith("/root/") for path in trace_ui_rect_observations)
        or len(set(trace_ui_rect_observations)) != len(trace_ui_rect_observations)
    ):
        raise ValueError("native-main qualification UI rect observations are invalid")
    trace_ui_selection_observations = ui_selection_observations or []
    if (
        not isinstance(trace_ui_selection_observations, list)
        or any(not isinstance(path, str) or not path.startswith("/root/") for path in trace_ui_selection_observations)
        or len(set(trace_ui_selection_observations)) != len(trace_ui_selection_observations)
    ):
        raise ValueError("native-main qualification UI selection observations are invalid")
    if scenario_readiness is not None:
        if not isinstance(scenario_readiness, dict):
            raise ValueError("native-main qualification scenario readiness must be an object")
        required_paths = scenario_readiness.get("required_node_paths")
        required_groups = scenario_readiness.get("required_group_minimums")
        required_visibility = scenario_readiness.get("required_visible")
        required_selection = scenario_readiness.get("required_option_selection", [])
        if (
            not isinstance(required_paths, list) or not required_paths
            or any(not isinstance(path, str) or not path.startswith("/root/") for path in required_paths)
            or not isinstance(required_groups, dict)
            or any(not isinstance(group, str) or not group or isinstance(count, bool) or not isinstance(count, int) or count < 1 for group, count in required_groups.items())
            or not isinstance(required_visibility, list) or not required_visibility
            or any(not isinstance(item, dict) or not isinstance(item.get("node_path"), str) or not item["node_path"].startswith("/root/") or not isinstance(item.get("visible"), bool) for item in required_visibility)
            or not isinstance(required_selection, list)
            or any(not isinstance(item, dict) or set(item) != {"node_path", "selected_index", "selected_text"} or not isinstance(item.get("node_path"), str) or not item["node_path"].startswith("/root/") or isinstance(item.get("selected_index"), bool) or not isinstance(item.get("selected_index"), int) or item["selected_index"] < 0 or not isinstance(item.get("selected_text"), str) or not item["selected_text"] for item in required_selection)
        ):
            raise ValueError("native-main qualification scenario readiness is invalid")
    source_config = project_config.read_text(encoding="utf-8")
    if _declared_godot_major(source_config) != 4:
        raise ValueError("native-main qualification currently supports declared Godot 4 projects only")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', source_config, re.MULTILINE)
    if match is None:
        raise ValueError("native-main qualification requires project.godot run/main_scene")
    original_main_scene = match.group("scene")
    if not original_main_scene.startswith("res://"):
        original_main_scene = f"res://{original_main_scene}"
    source_scene = upstream_project / original_main_scene.removeprefix("res://")
    if not source_scene.is_file():
        raise ValueError("native-main qualification configured main scene is missing")
    transition_target: Path | None = None
    if scene_transition is not None:
        if not isinstance(scene_transition, dict) or set(scene_transition) != {
            "from_scene", "to_scene", "earliest_frame", "latest_frame"
        }:
            raise ValueError("native-main qualification scene transition is invalid")
        source, target = scene_transition.get("from_scene"), scene_transition.get("to_scene")
        earliest, latest = scene_transition.get("earliest_frame"), scene_transition.get("latest_frame")
        if (
            source != original_main_scene or not isinstance(target, str) or not target.startswith("res://")
            or target == source or ".." in Path(target.removeprefix("res://")).parts
            or not isinstance(earliest, int) or isinstance(earliest, bool)
            or not isinstance(latest, int) or isinstance(latest, bool)
            or earliest < 0 or earliest > latest or latest >= capture_frames
        ):
            raise ValueError("native-main qualification scene transition is invalid")
        transition_target = upstream_project / target.removeprefix("res://")
        if transition_target.is_symlink() or not transition_target.is_file():
            raise ValueError("native-main qualification transition target is missing")
    shutil.copytree(upstream_project, destination, ignore=shutil.ignore_patterns(".git", ".godot", ".claude"))
    probe_dir = destination / ".flashpatch"
    probe_dir.mkdir()
    trace = probe_dir / "native-main-trace.json"
    trace.write_text(json.dumps({
        "fixed_fps": fixed_fps,
        "capture_frames": capture_frames,
        "warmup_frames": warmup_frames,
        "original_main_scene": original_main_scene,
        "actions": trace_actions,
        "launch_arguments": trace_launch_arguments,
        "pointer_events": trace_pointer_events,
        "key_events": trace_key_events,
        "scenario_readiness": scenario_readiness,
        "runtime_observations": trace_runtime_observations,
        "scene_transition": scene_transition,
        "ui_rect_observations": trace_ui_rect_observations,
        "ui_selection_observations": trace_ui_selection_observations,
        "state_observation": _controlled_state_observation,
    }, indent=2) + "\n", encoding="utf-8")
    probe = probe_dir / "native_main_capture.gd"
    probe.write_text(NATIVE_MAIN_CAPTURE_SCRIPT_GODOT4, encoding="utf-8")
    original_config_copy = probe_dir / "upstream-project.godot"
    original_config_copy.write_text(source_config, encoding="utf-8")
    copied_config = destination / "project.godot"
    copied_config.write_text(_append_native_capture_autoload(source_config), encoding="utf-8")
    copied_scene = destination / original_main_scene.removeprefix("res://")
    copied_target = None if transition_target is None else destination / str(scene_transition["to_scene"]).removeprefix("res://")
    native_main = {
        "original_run_main_scene": original_main_scene,
        "copied_run_main_scene": original_main_scene,
        "run_main_scene_unchanged": True,
        "upstream_main_scene_sha256": f"sha256:{hashlib.sha256(source_scene.read_bytes()).hexdigest()}",
        "copied_main_scene_sha256": f"sha256:{hashlib.sha256(copied_scene.read_bytes()).hexdigest()}",
        "main_scene_bytes_match": source_scene.read_bytes() == copied_scene.read_bytes(),
        "upstream_source_tree_sha256": _native_source_tree_sha256(upstream_project),
        "copied_source_tree_sha256": _native_source_tree_sha256(destination, original_project_config=source_config.encode("utf-8")),
        "source_scenes": {
            "from_scene": {"upstream_sha256": f"sha256:{hashlib.sha256(source_scene.read_bytes()).hexdigest()}", "copied_sha256": f"sha256:{hashlib.sha256(copied_scene.read_bytes()).hexdigest()}"},
            "to_scene": None if transition_target is None else {"upstream_sha256": f"sha256:{hashlib.sha256(transition_target.read_bytes()).hexdigest()}", "copied_sha256": f"sha256:{hashlib.sha256(copied_target.read_bytes()).hexdigest()}"},
        },
        "native_equivalence": "NOT_ESTABLISHED",
        "controlled_mutation": _controlled_state_observation is not None,
        "upstream_defect": False if _controlled_state_observation is not None else None,
        "scoreable": False,
    }
    if _controlled_state_observation is not None:
        native_main["controlled_fixture"] = {
            "schema": "flashpatch-controlled-native-shader-fixture-v1",
            "fixture_tree_sha256": _controlled_native_fixture_tree_sha256(),
            "provider_script": _controlled_state_observation["provider_script"],
            "provider_script_sha256": _controlled_state_observation["provider_script_sha256"],
            "trace_sha256": f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}",
        }
    native_main["sealed_project_tree_sha256"] = _native_project_manifest_sha256(
        _native_project_file_manifest(destination)
    )
    return NativeMainCaptureQualification(destination, trace, original_main_scene, native_main)


def materialize_controlled_native_main_shader_qualification(
    destination: Path,
    *,
    fixed_fps: int = 60,
    capture_frames: int = 12,
) -> NativeMainCaptureQualification:
    """Materialize the one frozen local fixture authorized to emit a state oracle.

    The fixture is generated from exact in-module bytes and has no repository,
    revision, or external source identity.  Generic and pinned projects remain
    native-main capture-only and cannot supply semantic invariant claims.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("controlled native shader qualification requires a new destination")
    if fixed_fps != 60 or capture_frames != 12:
        raise ValueError("controlled native shader qualification uses the frozen 60 fps, 12-frame trace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".flashpatch-controlled-native-source-", dir=destination.parent
    ) as temporary:
        source = Path(temporary)
        for relative, payload in _CONTROLLED_NATIVE_SHADER_FILES.items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return materialize_native_main_capture_qualification(
            source,
            destination,
            fixed_fps=fixed_fps,
            capture_frames=capture_frames,
            actions=[
                {"frame": 1, "action": "advance", "pressed": True},
                {"frame": 2, "action": "advance", "pressed": False},
            ],
            scenario_readiness={
                "required_node_paths": ["/root/Main", "/root/Main/Flash"],
                "required_group_minimums": {},
                "required_visible": [{"node_path": "/root/Main/Flash", "visible": True}],
            },
            runtime_observations=[{
                "kind": "shader_parameter",
                "node_path": "/root/Main/Flash",
                "property": "flash_intensity",
                "source_path": "res://main.tscn",
                "resource_path": "res://main.tscn",
                "source_line": 8,
                "shader_path": "res://effects/controlled_flash.gdshader",
            }],
            _controlled_state_observation=_controlled_native_state_descriptor(),
        )


def _native_main_trace(qualification: NativeMainCaptureQualification) -> dict[str, object]:
    try:
        trace = json.loads(qualification.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native-main qualification trace is unreadable") from exc
    if not isinstance(trace, dict):
        raise RuntimeError("native-main qualification trace must be an object")
    fixed_fps = trace.get("fixed_fps")
    capture_frames = trace.get("capture_frames")
    warmup_frames = trace.get("warmup_frames")
    actions = trace.get("actions")
    launch_arguments = trace.get("launch_arguments")
    pointer_events = trace.get("pointer_events")
    key_events = trace.get("key_events")
    scenario_readiness = trace.get("scenario_readiness")
    runtime_observations = trace.get("runtime_observations")
    ui_selection_observations = trace.get("ui_selection_observations")
    state_observation = trace.get("state_observation")
    if (
        isinstance(fixed_fps, bool) or not isinstance(fixed_fps, int) or fixed_fps <= 0
        or isinstance(capture_frames, bool) or not isinstance(capture_frames, int) or capture_frames < 2
        or isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int) or warmup_frames < 0
        or trace.get("original_main_scene") != qualification.original_main_scene
        or not isinstance(actions, list)
        or any(
            not isinstance(action, dict)
            or isinstance(action.get("frame"), bool) or not isinstance(action.get("frame"), int)
            or int(action["frame"]) < 0 or int(action["frame"]) >= warmup_frames + capture_frames
            or not isinstance(action.get("action"), str) or not action["action"]
            or not isinstance(action.get("pressed"), bool)
            for action in actions
        )
        or not isinstance(launch_arguments, list)
        or any(not isinstance(argument, str) or not argument or "\x00" in argument for argument in launch_arguments)
        or not isinstance(pointer_events, list)
        or any(
            not isinstance(event, dict)
            or isinstance(event.get("frame"), bool) or not isinstance(event.get("frame"), int)
            or int(event["frame"]) < 0 or int(event["frame"]) >= warmup_frames + capture_frames
            or event.get("kind") != "left_click"
            or isinstance(event.get("x"), bool) or not isinstance(event.get("x"), (int, float))
            or isinstance(event.get("y"), bool) or not isinstance(event.get("y"), (int, float))
            or not 0.0 <= float(event["x"]) <= 1.0 or not 0.0 <= float(event["y"]) <= 1.0
            for event in pointer_events
        )
        or not isinstance(key_events, list)
        or any(
            not isinstance(event, dict)
            or isinstance(event.get("frame"), bool) or not isinstance(event.get("frame"), int)
            or int(event["frame"]) < 0 or int(event["frame"]) >= warmup_frames + capture_frames
            or event.get("key") not in {"down", "up", "enter"}
            for event in key_events
        )
        or scenario_readiness is not None and not isinstance(scenario_readiness, dict)
        or not isinstance(runtime_observations, list)
        or any(
            not isinstance(observation, dict)
            or (
                observation.get("kind", "render_property") == "shader_parameter"
                and (
                    set(observation) - {"kind", "node_path", "property", "source_path", "resource_path", "source_line", "shader_path", "scene_file_path"}
                    or not isinstance(observation.get("node_path"), str) or not observation["node_path"].startswith("/root/")
                    or not isinstance(observation.get("property"), str) or not observation["property"]
                    or not isinstance(observation.get("source_path"), str) or not observation["source_path"].startswith("res://") or not observation["source_path"].endswith(".tscn")
                    or not isinstance(observation.get("resource_path"), str) or not observation["resource_path"].startswith("res://")
                    or isinstance(observation.get("source_line"), bool) or not isinstance(observation.get("source_line"), int) or observation["source_line"] < 1
                    or not isinstance(observation.get("shader_path"), str) or not observation["shader_path"].startswith("res://") or not observation["shader_path"].endswith(".gdshader")
                    or observation.get("scene_file_path") is not None and (not isinstance(observation.get("scene_file_path"), str) or not observation["scene_file_path"].startswith("res://"))
                )
            )
            or (
                observation.get("kind", "render_property") != "shader_parameter"
                and (
                    observation.get("kind", "render_property") != "render_property"
                    or not isinstance(observation.get("node_path"), str) or not observation["node_path"].startswith("/root/")
                    or not isinstance(observation.get("property"), str) or not observation["property"]
                    or not isinstance(observation.get("script_path"), str) or not observation["script_path"].startswith("res://")
                    or not isinstance(observation.get("resource_path"), str) or not observation["resource_path"].startswith("res://")
                    or isinstance(observation.get("source_line"), bool) or not isinstance(observation.get("source_line"), int) or observation["source_line"] < 1
                    or observation.get("scene_file_path") is not None and (not isinstance(observation.get("scene_file_path"), str) or not observation["scene_file_path"].startswith("res://"))
                )
            )
            for observation in runtime_observations
        )
        or not isinstance(ui_selection_observations, list)
        or any(not isinstance(path, str) or not path.startswith("/root/") for path in ui_selection_observations)
        or len(set(ui_selection_observations)) != len(ui_selection_observations)
        or state_observation is not None and state_observation != _controlled_native_state_descriptor()
    ):
        raise RuntimeError("native-main qualification trace contract is invalid")
    return trace


def _verify_declared_scene_transition(trace: dict[str, object], replay: dict[str, object]) -> None:
    """Recompute a declared one-way scene transition from a frame-complete ledger."""
    declared = trace.get("scene_transition")
    if declared is None:
        return
    if not isinstance(declared, dict) or set(declared) != {
        "from_scene", "to_scene", "earliest_frame", "latest_frame"
    }:
        raise RuntimeError("native-main declared scene transition is invalid")
    source, target = declared.get("from_scene"), declared.get("to_scene")
    earliest, latest = declared.get("earliest_frame"), declared.get("latest_frame")
    frames = trace.get("capture_frames")
    if (
        source != trace.get("original_main_scene")
        or not isinstance(source, str) or not source.startswith("res://")
        or not isinstance(target, str) or not target.startswith("res://") or target == source
        or not isinstance(earliest, int) or isinstance(earliest, bool)
        or not isinstance(latest, int) or isinstance(latest, bool)
        or not isinstance(frames, int) or earliest < 0 or earliest > latest or latest >= frames
    ):
        raise RuntimeError("native-main declared scene transition is invalid")
    observations = replay.get("scene_observations")
    if not isinstance(observations, list) or len(observations) != frames:
        raise RuntimeError("native-main scene transition ledger is incomplete")
    observed_transition: int | None = None
    initial_instance: int | None = None
    target_instance: int | None = None
    for frame, observation in enumerate(observations):
        if not isinstance(observation, dict) or set(observation) != {
            "frame_index", "scene_file_path", "current_scene_node_path",
            "current_scene_instance_id", "wrapper_ancestor_paths",
        }:
            raise RuntimeError("native-main scene transition ledger entry is invalid")
        path, node, instance, ancestors = (
            observation.get("scene_file_path"), observation.get("current_scene_node_path"),
            observation.get("current_scene_instance_id"), observation.get("wrapper_ancestor_paths"),
        )
        if (
            observation.get("frame_index") != frame
            or not isinstance(path, str) or path not in {source, target}
            or not isinstance(node, str) or not node.startswith("/root/") or "/" in node.removeprefix("/root/")
            or not isinstance(instance, int) or isinstance(instance, bool) or instance <= 0
            or ancestors != []
        ):
            raise RuntimeError("native-main scene transition ledger entry is invalid")
        if frame == 0:
            if path != source:
                raise RuntimeError("native-main transition initial scene is invalid")
            initial_instance = instance
        if observed_transition is None:
            if path == target:
                if frame < earliest or frame > latest or instance == initial_instance:
                    raise RuntimeError("native-main scene transition is outside the declared window")
                observed_transition, target_instance = frame, instance
            elif frame > latest:
                raise RuntimeError("native-main declared scene transition did not occur")
        elif path != target or instance != target_instance:
            raise RuntimeError("native-main scene transition did not persist")
    if observed_transition is None:
        raise RuntimeError("native-main declared scene transition did not occur")
    acknowledgement = replay.get("scene_transition_acknowledgement")
    if acknowledgement != {
        "from_scene": source,
        "to_scene": target,
        "observed_frame": observed_transition,
        "status": "APPLIED",
    }:
        raise RuntimeError("native-main scene transition acknowledgement is invalid")


def _require_unchanged_native_main(qualification: NativeMainCaptureQualification) -> None:
    config_path = qualification.project / "project.godot"
    try:
        config = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("native-main qualification project.godot is unreadable") from exc
    if _declared_godot_major(config) != 4:
        raise RuntimeError("native-main qualification no longer declares Godot 4")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', config, re.MULTILINE)
    if match is None:
        raise RuntimeError("native-main qualification run/main_scene is missing")
    configured_scene = match.group("scene")
    if not configured_scene.startswith("res://"):
        configured_scene = f"res://{configured_scene}"
    if configured_scene != qualification.original_main_scene:
        raise RuntimeError("native-main qualification changed run/main_scene")
    declared = qualification.native_main
    if (
        declared.get("run_main_scene_unchanged") is not True
        or declared.get("main_scene_bytes_match") is not True
        or declared.get("native_equivalence") != "NOT_ESTABLISHED"
        or declared.get("scoreable") is not False
    ):
        raise RuntimeError("native-main qualification attempted to relax its structural or equivalence gate")
    original_config_copy = qualification.project / ".flashpatch" / "upstream-project.godot"
    if not original_config_copy.is_file() or original_config_copy.is_symlink():
        raise RuntimeError("native-main qualification source tree binding is missing")
    expected_tree = declared.get("upstream_source_tree_sha256")
    if (
        not isinstance(expected_tree, str)
        or declared.get("copied_source_tree_sha256") != expected_tree
        or _native_source_tree_sha256(qualification.project, original_project_config=original_config_copy.read_bytes()) != expected_tree
    ):
        raise RuntimeError("native-main qualification source tree bytes changed")
    scene = (qualification.project / configured_scene.removeprefix("res://")).resolve()
    project = qualification.project.resolve()
    if project not in scene.parents or not scene.is_file():
        raise RuntimeError("native-main qualification configured scene is missing or outside project")
    digest = f"sha256:{hashlib.sha256(scene.read_bytes()).hexdigest()}"
    if digest != declared.get("upstream_main_scene_sha256") or digest != declared.get("copied_main_scene_sha256"):
        raise RuntimeError("native-main qualification changed configured main scene bytes")
    trace = _native_main_trace(qualification)
    state_observation = trace.get("state_observation")
    controlled = declared.get("controlled_mutation") is True
    if state_observation is None:
        if controlled or declared.get("controlled_fixture") is not None:
            raise RuntimeError("native-main qualification has an unbound controlled fixture stamp")
    else:
        fixture = declared.get("controlled_fixture")
        if (
            not controlled
            or declared.get("upstream_defect") is not False
            or not isinstance(fixture, dict)
            or fixture.get("schema") != "flashpatch-controlled-native-shader-fixture-v1"
            or fixture.get("fixture_tree_sha256") != _controlled_native_fixture_tree_sha256()
            or fixture.get("provider_script") != state_observation.get("provider_script")
            or fixture.get("provider_script_sha256") != state_observation.get("provider_script_sha256")
            or fixture.get("trace_sha256") != f"sha256:{hashlib.sha256(qualification.trace.read_bytes()).hexdigest()}"
        ):
            raise RuntimeError("native-main controlled fixture or trace identity is invalid")
    transition = trace.get("scene_transition")
    source_scenes = declared.get("source_scenes")
    if not isinstance(source_scenes, dict):
        raise RuntimeError("native-main qualification source scene binding is missing")
    if transition is None:
        if source_scenes.get("to_scene") is not None:
            raise RuntimeError("native-main qualification has an undeclared transition target binding")
        return
    if not isinstance(transition, dict) or not isinstance(transition.get("to_scene"), str):
        raise RuntimeError("native-main qualification transition trace is invalid")
    target = qualification.project / transition["to_scene"].removeprefix("res://")
    target_binding = source_scenes.get("to_scene")
    if (
        target.is_symlink() or not target.is_file() or not isinstance(target_binding, dict)
        or f"sha256:{hashlib.sha256(target.read_bytes()).hexdigest()}" != target_binding.get("copied_sha256")
    ):
        raise RuntimeError("native-main qualification changed transition target scene bytes")


def _validate_controlled_native_state_oracle(
    qualification: NativeMainCaptureQualification,
    trace: dict[str, object],
    replay: dict[str, object],
    presentation_timestamps_us: list[int],
) -> dict[str, object] | None:
    descriptor = trace.get("state_observation")
    observed_descriptor = replay.get("state_observation_descriptor")
    observations = replay.get("state_observations")
    if descriptor is None:
        if observed_descriptor is not None or observations not in (None, []):
            raise RuntimeError("native-main replay emitted an undeclared state observation oracle")
        return None
    if descriptor != _controlled_native_state_descriptor() or observed_descriptor != descriptor:
        raise RuntimeError("native-main controlled state observation descriptor mismatches the sealed trace")
    fixture = qualification.native_main.get("controlled_fixture")
    if not isinstance(fixture, dict) or qualification.native_main.get("controlled_mutation") is not True:
        raise RuntimeError("native-main controlled state observation lacks fixture authority")
    if not isinstance(observations, list) or len(observations) != len(presentation_timestamps_us):
        raise RuntimeError("native-main controlled state stream is not frame complete")
    property_descriptors = descriptor["properties"]
    assert isinstance(property_descriptors, list)
    property_types = {item["name"]: item["type"] for item in property_descriptors}

    def valid_typed(value: object, expected: str) -> bool:
        if expected == "bool":
            return isinstance(value, bool)
        if expected == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "float":
            return isinstance(value, float) and math.isfinite(value)
        if expected == "string":
            return isinstance(value, str)
        return False

    canonical_stream: list[dict[str, object]] = []
    for frame_index, observation in enumerate(observations):
        if not isinstance(observation, dict) or set(observation) != {
            "frame_index", "presentation_timestamp_us", "values"
        }:
            raise RuntimeError("native-main controlled state stream entry is malformed")
        values = observation.get("values")
        if (
            observation.get("frame_index") != frame_index
            or observation.get("presentation_timestamp_us") != presentation_timestamps_us[frame_index]
            or not isinstance(values, dict)
            or set(values) != set(property_types)
            or any(not valid_typed(values[name], expected) for name, expected in property_types.items())
        ):
            raise RuntimeError("native-main controlled state stream domain or typed values are invalid")
        canonical_stream.append({
            "frame_index": frame_index,
            "presentation_timestamp_us": presentation_timestamps_us[frame_index],
            "values": {name: values[name] for name in sorted(property_types)},
        })
    transition = descriptor["action_transition"]
    assert isinstance(transition, dict)
    pressed = {
        "frame": transition["pressed_frame"],
        "action": transition["action"],
        "pressed": True,
    }
    if pressed not in trace["actions"]:
        raise RuntimeError("native-main controlled state transition is not bound to the trace action")
    before = canonical_stream[int(transition["before_frame"])]["values"]
    after = canonical_stream[int(transition["after_frame"])]["values"]
    assert isinstance(before, dict) and isinstance(after, dict)
    property_name = str(transition["property"])
    if before.get(property_name) != transition["before"] or after.get(property_name) != transition["after"]:
        raise RuntimeError("native-main controlled state stream omitted the required action transition")
    final_values = canonical_stream[-1]["values"]
    assert isinstance(final_values, dict)
    terminal_completion = descriptor["terminal_completion"]
    assert isinstance(terminal_completion, dict)
    if final_values.get(str(terminal_completion["property"])) != terminal_completion["equals"]:
        raise RuntimeError("native-main controlled state stream did not reach its terminal state")
    terminal_state = {
        name: final_values[name] for name in descriptor["terminal_state_properties"]
    }
    player_world_stream = [
        {
            "frame_index": item["frame_index"],
            "presentation_timestamp_us": item["presentation_timestamp_us"],
            "values": {
                name: item["values"][name] for name in descriptor["player_world_properties"]
            },
        }
        for item in canonical_stream
    ]
    score = final_values[str(descriptor["score_property"])]
    return {
        "schema": "flashpatch-controlled-native-preservation-oracle-v1",
        "controlled_fixture": fixture,
        "descriptor_sha256": _native_counterfactual_json_sha256(descriptor),
        "frame_domain_sha256": _native_counterfactual_json_sha256([
            [item["frame_index"], item["presentation_timestamp_us"]] for item in canonical_stream
        ]),
        "state_stream": canonical_stream,
        "state_stream_sha256": _native_counterfactual_json_sha256(canonical_stream),
        "terminal_completion": True,
        "terminal_state": terminal_state,
        "terminal_state_sha256": _native_counterfactual_json_sha256(terminal_state),
        "player_world_stream": player_world_stream,
        "player_world_digest": _native_counterfactual_json_sha256(player_world_stream),
        "score": score,
    }


def classify_native_main_capture_qualification(
    qualification: NativeMainCaptureQualification,
    output: Path,
    *,
    godot_binary: Path | None = None,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    """Capture the configured Godot-4 main scene without claiming equivalence."""
    output = Path(output).resolve()
    trace = _native_main_trace(qualification)
    _require_unchanged_native_main(qualification)
    runner = GodotNativeMainRendererReplayRunner(
        qualification.project,
        godot_binary=godot_binary,
        timeout_seconds=timeout_seconds,
    )
    replay = runner.replay(qualification.trace, output)
    _verify_declared_scene_transition(trace, replay)
    if (
        replay.get("qualification_only") is not True
        or replay.get("scoreable") is not False
        or replay.get("native_equivalence") != "NOT_ESTABLISHED"
        or replay.get("execution_mode") != "instrumented_native_main_scene_capture"
    ):
        raise RuntimeError("native-main replay attempted to claim scoreability or equivalence")
    runtime = replay.get("native_main")
    if not isinstance(runtime, dict):
        raise RuntimeError("native-main replay omitted runtime current_scene facts")
    expected_scene = qualification.original_main_scene
    node_path = runtime.get("current_scene_node_path")
    if (
        runtime.get("current_scene_exists") is not True
        or runtime.get("expected_scene_file_path") != expected_scene
        or runtime.get("runtime_scene_file_path") != expected_scene
        or runtime.get("scene_file_path_match") is not True
        or not isinstance(node_path, str) or not node_path.startswith("/root/")
        or node_path.removeprefix("/root/").find("/") >= 0
        or runtime.get("wrapper_ancestor_paths") != []
        or runtime.get("no_wrapper_ancestor") is not True
    ):
        raise RuntimeError("native-main replay current_scene or wrapper topology is invalid")
    renderer = replay.get("renderer_capture")
    if not isinstance(renderer, dict):
        raise RuntimeError("native-main replay omitted renderer_capture")
    presentation_us = renderer.get("timestamps_us")
    actual_us = renderer.get("actual_capture_timestamps_us")
    capture_frames = int(trace["capture_frames"])
    warmup_frames = int(trace["warmup_frames"])
    fixed_fps = int(trace["fixed_fps"])
    expected_capture_trace_frames = list(range(warmup_frames, warmup_frames + capture_frames))
    expected_presentation_us = [
        int(index * 1_000_000 / fixed_fps) for index in expected_capture_trace_frames
    ]
    if presentation_us != expected_presentation_us:
        raise RuntimeError("native-main replay presentation timestamps do not match absolute fixed media time")
    if renderer.get("warmup_frames") != warmup_frames:
        raise RuntimeError("native-main replay renderer warmup binding is invalid")
    if renderer.get("capture_trace_frame_indices") != expected_capture_trace_frames:
        raise RuntimeError("native-main replay capture trace frame binding is invalid")
    if (
        not isinstance(actual_us, list) or len(actual_us) != capture_frames
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in actual_us)
        or any(right <= left for left, right in zip(actual_us, actual_us[1:]))
    ):
        raise RuntimeError("native-main replay actual timestamps are missing or non-monotonic")
    raw_artifact = replay.get("frames_npz")
    if not isinstance(raw_artifact, str) or not raw_artifact:
        raise RuntimeError("native-main replay omitted packed RGB frame artifact")
    frame_path = (output.parent / raw_artifact).resolve()
    if output.parent not in frame_path.parents or not frame_path.is_file():
        raise RuntimeError("native-main packed RGB frame artifact is missing or outside output directory")
    expected_seconds = np.asarray(expected_presentation_us, dtype=np.float64) / 1_000_000.0
    try:
        with open_renderer_artifact(frame_path) as artifact:
            frames = artifact.frames
            timestamps = artifact.timestamps
            if (
                len(frames) != capture_frames
                or not np.array_equal(timestamps, expected_seconds)
            ):
                raise RuntimeError(
                    "native-main packed RGB frames or timestamps violate the trace contract"
                )
            result = analyze(frames, timestamps)
            frame_count = len(frames)
    except RendererArtifactError as exc:
        raise RuntimeError(
            "native-main packed RGB frames or timestamps are invalid"
        ) from exc
    actions = trace["actions"]
    acknowledgements = replay.get("action_acknowledgements")
    if not isinstance(acknowledgements, list) or len(acknowledgements) != len(actions):
        raise RuntimeError("native-main replay did not acknowledge every trace action")
    for action, acknowledgement in zip(actions, acknowledgements):
        if (
            not isinstance(acknowledgement, dict)
            or acknowledgement.get("frame") != action.get("frame")
            or acknowledgement.get("action") != action.get("action")
            or acknowledgement.get("pressed") != action.get("pressed")
            or acknowledgement.get("status") != "APPLIED"
        ):
            raise RuntimeError("native-main replay action was not applied")
    pointer_events = trace["pointer_events"]
    pointer_acknowledgements = replay.get("pointer_acknowledgements")
    if not isinstance(pointer_acknowledgements, list) or len(pointer_acknowledgements) != len(pointer_events):
        raise RuntimeError("native-main replay did not acknowledge every pointer event")
    for event, acknowledgement in zip(pointer_events, pointer_acknowledgements):
        observed_x = acknowledgement.get("x") if isinstance(acknowledgement, dict) else None
        observed_y = acknowledgement.get("y") if isinstance(acknowledgement, dict) else None
        expected_x, expected_y = event.get("x"), event.get("y")
        if (
            not isinstance(acknowledgement, dict)
            or acknowledgement.get("frame") != event.get("frame")
            or acknowledgement.get("kind") != event.get("kind")
            or isinstance(observed_x, bool) or not isinstance(observed_x, (int, float)) or not math.isfinite(float(observed_x))
            or isinstance(observed_y, bool) or not isinstance(observed_y, (int, float)) or not math.isfinite(float(observed_y))
            # GDScript JSON serialization may trim a final binary digit.  A
            # bounded tolerance accepts only that representation difference,
            # never a different normalized click coordinate.
            or not math.isclose(float(observed_x), float(expected_x), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(observed_y), float(expected_y), rel_tol=0.0, abs_tol=1e-12)
            or acknowledgement.get("status") != "APPLIED"
        ):
            raise RuntimeError("native-main replay pointer acknowledgement is invalid")
    key_events = trace["key_events"]
    key_acknowledgements = replay.get("key_acknowledgements")
    if not isinstance(key_acknowledgements, list) or len(key_acknowledgements) != len(key_events):
        raise RuntimeError("native-main replay did not acknowledge every key event")
    for event, acknowledgement in zip(key_events, key_acknowledgements):
        if (
            not isinstance(acknowledgement, dict)
            or acknowledgement.get("frame") != event.get("frame")
            or acknowledgement.get("key") != event.get("key")
            or acknowledgement.get("status") != "APPLIED"
        ):
            raise RuntimeError("native-main replay key acknowledgement is invalid")
    declared_observations = trace["runtime_observations"]
    runtime_events = replay.get("runtime_events")
    scene_ledger = replay.get("scene_observations")
    if trace.get("scene_transition") is not None:
        if not isinstance(scene_ledger, list) or len(scene_ledger) != capture_frames:
            raise RuntimeError("native-main replay scene ledger is missing for scoped observations")
    else:
        scene_ledger = [{"scene_file_path": qualification.original_main_scene} for _ in range(capture_frames)]
        if not isinstance(runtime_events, list):
            raise RuntimeError("native-main replay runtime observations are missing or incomplete")
    expected_runtime_events: set[tuple[object, ...]] = set()
    for observation in declared_observations:
        kind = observation.get("kind", "render_property")
        for tick in range(capture_frames):
            if observation.get("scene_file_path") is not None and scene_ledger[tick].get("scene_file_path") != observation.get("scene_file_path"):
                continue
            if kind == "shader_parameter":
                expected_runtime_events.add((
                    tick, observation["node_path"], observation["source_path"],
                    observation["resource_path"], observation["source_line"],
                    observation["shader_path"], observation["property"], kind,
                ))
            else:
                expected_runtime_events.add((
                    tick, observation["node_path"], observation["script_path"],
                    observation["resource_path"], observation["source_line"],
                    observation["property"], kind,
                ))
    observed_runtime_events: set[tuple[object, ...]] = set()
    for event in runtime_events:
        kind = event.get("event_kind") if isinstance(event, dict) else None
        if kind == "shader_parameter":
            if (
                not isinstance(event, dict)
                or set(event) != {"frame_index", "node_path", "source_path", "resource_path", "source_line", "shader_path", "property", "factual_value", "event_kind"}
                or isinstance(event.get("frame_index"), bool) or not isinstance(event.get("frame_index"), int)
                or not isinstance(event.get("node_path"), str)
                or not isinstance(event.get("source_path"), str) or not event["source_path"].startswith("res://") or not event["source_path"].endswith(".tscn")
                or not isinstance(event.get("resource_path"), str)
                or isinstance(event.get("source_line"), bool) or not isinstance(event.get("source_line"), int)
                or not isinstance(event.get("shader_path"), str) or not event["shader_path"].startswith("res://") or not event["shader_path"].endswith(".gdshader")
                or not isinstance(event.get("property"), str)
                or isinstance(event.get("factual_value"), bool) or not isinstance(event.get("factual_value"), (int, float)) or not math.isfinite(float(event["factual_value"]))
            ):
                raise RuntimeError("native-main replay shader runtime observation is invalid")
            observed_runtime_events.add((
                event["frame_index"], event["node_path"], event["source_path"], event["resource_path"],
                event["source_line"], event["shader_path"], event["property"], kind,
            ))
            continue
        if (
            not isinstance(event, dict)
            or set(event) != {"frame_index", "node_path", "script_path", "resource_path", "source_line", "property", "factual_value", "event_kind"}
            or isinstance(event.get("frame_index"), bool) or not isinstance(event.get("frame_index"), int)
            or not isinstance(event.get("node_path"), str)
            or not isinstance(event.get("script_path"), str)
            or not isinstance(event.get("resource_path"), str)
            or isinstance(event.get("source_line"), bool) or not isinstance(event.get("source_line"), int)
            or not isinstance(event.get("property"), str)
            or isinstance(event.get("factual_value"), bool) or not isinstance(event.get("factual_value"), (int, float)) or not math.isfinite(float(event["factual_value"]))
            or event.get("event_kind") != "render_property"
        ):
            raise RuntimeError("native-main replay runtime observation is invalid")
        observed_runtime_events.add(
            (event["frame_index"], event["node_path"], event["script_path"], event["resource_path"], event["source_line"], event["property"], kind)
        )
    if observed_runtime_events != expected_runtime_events:
        raise RuntimeError("native-main replay runtime observations do not match declared contributors")
    declared_ui_selection_observations = trace["ui_selection_observations"]
    observed_ui_selection_observations = replay.get("ui_selection_observations")
    expected_ui_selection_observations = len(declared_ui_selection_observations) * capture_frames
    if (
        not isinstance(observed_ui_selection_observations, list)
        or len(observed_ui_selection_observations) != expected_ui_selection_observations
    ):
        raise RuntimeError("native-main replay UI selection observations are missing or incomplete")
    for index, observation in enumerate(observed_ui_selection_observations):
        path = declared_ui_selection_observations[index % len(declared_ui_selection_observations)] if declared_ui_selection_observations else None
        expected_frame = index // len(declared_ui_selection_observations) if declared_ui_selection_observations else None
        if (
            not isinstance(observation, dict)
            or set(observation) != {"frame_index", "node_path", "selected_index", "selected_text", "popup_visible", "popup_focused_index"}
            or observation.get("frame_index") != expected_frame
            or observation.get("node_path") != path
            or isinstance(observation.get("selected_index"), bool) or not isinstance(observation.get("selected_index"), int) or observation["selected_index"] < 0
            or not isinstance(observation.get("selected_text"), str)
            or not isinstance(observation.get("popup_visible"), bool)
            or isinstance(observation.get("popup_focused_index"), bool) or not isinstance(observation.get("popup_focused_index"), int)
        ):
            raise RuntimeError("native-main replay UI selection observation is invalid")
    declared_readiness = trace["scenario_readiness"]
    observed_readiness = replay.get("scenario_readiness")
    if not isinstance(observed_readiness, dict):
        raise RuntimeError("native-main replay omitted scenario readiness facts")
    scenario_ready = False
    if declared_readiness is None:
        if observed_readiness.get("declared") is not False or observed_readiness.get("satisfied") is not False:
            raise RuntimeError("native-main replay attempted to claim undeclared scenario readiness")
    else:
        paths = declared_readiness.get("required_node_paths")
        groups = declared_readiness.get("required_group_minimums")
        visibility = declared_readiness.get("required_visible")
        selections = declared_readiness.get("required_option_selection", [])
        if (
            not isinstance(paths, list) or not paths
            or any(not isinstance(path, str) or not path.startswith("/root/") for path in paths)
            or not isinstance(groups, dict)
            or any(not isinstance(group, str) or not group or isinstance(count, bool) or not isinstance(count, int) or count < 1 for group, count in groups.items())
            or not isinstance(visibility, list) or not visibility
            or not isinstance(selections, list)
            or observed_readiness.get("declared") is not True
            or observed_readiness.get("satisfied") is not True
            or observed_readiness.get("missing_node_paths") != []
            or observed_readiness.get("insufficient_groups") != {}
            or observed_readiness.get("visibility_mismatches") != []
            or observed_readiness.get("selection_mismatches") != []
        ):
            raise RuntimeError("native-main replay scenario readiness is missing or unsatisfied")
        observed_groups = observed_readiness.get("observed_group_counts")
        if not isinstance(observed_groups, dict) or any(observed_groups.get(group, 0) < count for group, count in groups.items()):
            raise RuntimeError("native-main replay scenario group readiness is unsatisfied")
        signal_events = replay.get("ui_selection_signal_events")
        if not isinstance(signal_events, list):
            raise RuntimeError("native-main replay UI selection signal ledger is missing")
        for selection in selections:
            if not any(
                isinstance(event, dict)
                and event.get("node_path") == selection.get("node_path")
                and event.get("selected_index") == selection.get("selected_index")
                and isinstance(event.get("trace_frame_index"), int)
                and 0 <= event["trace_frame_index"] < warmup_frames + capture_frames
                for event in signal_events
            ):
                raise RuntimeError("native-main replay declared UI selection signal is missing")
        scenario_ready = True
    preservation_oracle = _validate_controlled_native_state_oracle(
        qualification, trace, replay, expected_presentation_us
    )
    hazardous = result.hazardous
    decision = (
        "HAZARDOUS_ATTRIBUTION_PENDING" if hazardous and scenario_ready
        else "SAFE_SCENARIO_READY" if scenario_ready
        else "NATIVE_MAIN_CAPTURE_ONLY"
    )
    reason = (
        "native_main_renderer_hazard_requires_runtime_attribution_and_preservation_oracles" if hazardous and scenario_ready
        else "no_hazard_in_declared_scenario_ready_trace" if scenario_ready
        else "scenario_readiness_not_declared"
    )
    receipt = {
        "schema": "flashpatch-godot-native-main-capture-v1",
        "decision": decision,
        "reason": reason,
        "controlled_mutation": qualification.native_main.get("controlled_mutation") is True,
        "upstream_defect": qualification.native_main.get("upstream_defect"),
        "qualification_only": True,
        "scoreable": False,
        "execution_mode": "instrumented_native_main_scene_capture",
        "native_equivalence": "NOT_ESTABLISHED",
        "warmup_frames": warmup_frames,
        "capture_trace_frame_indices": expected_capture_trace_frames,
        "presentation_timestamps_us": expected_presentation_us,
        "original_main_scene": expected_scene,
        "native_main": runtime,
        "action_acknowledgements": acknowledgements,
        "pointer_acknowledgements": pointer_acknowledgements,
        "key_acknowledgements": key_acknowledgements,
        "runtime_events": runtime_events,
        "ui_selection_observations": observed_ui_selection_observations,
        "scenario_readiness": observed_readiness,
        "preservation_oracle": preservation_oracle,
        "controlled_fixture": qualification.native_main.get("controlled_fixture"),
        "trace_sha256": f"sha256:{hashlib.sha256(qualification.trace.read_bytes()).hexdigest()}",
        "frame_artifact": str(frame_path),
        "frame_artifact_sha256": _sha256(frame_path),
        "frame_count": frame_count,
        "max_risk": result.max_flash_count if hazardous else 0.0,
        "hazard_frame_indices": np.flatnonzero(np.any(result.hazard_mask, axis=(1, 2))).tolist(),
        "replay_sha256": _sha256(output),
    }
    receipt_path = output.with_name("native-main-capture-receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path)}


def _native_counterfactual_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _native_shader_static_binding(
    qualification: NativeMainCaptureQualification,
    trace: dict[str, object],
    patch: NativeTscnTokenPatch,
) -> tuple[dict[str, object], dict[str, object], float]:
    """Bind the sole runtime observer to one statically wired ShaderMaterial."""
    observations = trace.get("runtime_observations")
    if not isinstance(observations, list) or len(observations) != 1:
        raise RuntimeError("native shader counterfactual requires exactly one runtime observer")
    observer = observations[0]
    if not isinstance(observer, dict) or observer.get("kind") != "shader_parameter":
        raise RuntimeError("native shader counterfactual observer is not a shader parameter")
    if (
        patch.source_path != qualification.original_main_scene
        or observer.get("source_path") != patch.source_path
        or observer.get("resource_path") != patch.source_path
        or observer.get("scene_file_path") not in {None, patch.source_path}
        or observer.get("property") != patch.parameter
        or observer.get("source_line") != patch.source_line
        or trace.get("scene_transition") is not None
    ):
        raise RuntimeError("native shader observer descriptor does not match the declared source token")

    relative = _native_tscn_patch_relative_path(patch)
    source = qualification.project / relative
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("native shader observer source scene is missing")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("native shader observer source scene is not UTF-8") from exc
    lines = text.splitlines()
    if patch.source_line > len(lines):
        raise RuntimeError("native shader observer source line is outside the scene")
    _, _, factual_token = _native_tscn_numeric_token(source.read_bytes(), patch, label="factual")
    factual_value = float(factual_token.decode("ascii"))

    section_start = patch.source_line - 1
    while section_start >= 0 and not lines[section_start].startswith("["):
        section_start -= 1
    material_header = re.fullmatch(
        r'\[sub_resource\b(?=[^\]]*\btype="ShaderMaterial")(?=[^\]]*\bid="(?P<id>[^"]+)")[^\]]*\]',
        lines[section_start] if section_start >= 0 else "",
    )
    if material_header is None:
        raise RuntimeError("native shader source token is not inside a ShaderMaterial sub-resource")
    section_end = section_start + 1
    while section_end < len(lines) and not lines[section_end].startswith("["):
        section_end += 1
    shader_bindings = [
        match.group("id")
        for line in lines[section_start + 1:section_end]
        if (match := re.fullmatch(r'[ \t]*shader[ \t]*=[ \t]*ExtResource\("(?P<id>[^"]+)"\)[ \t]*', line))
    ]
    if len(shader_bindings) != 1:
        raise RuntimeError("native ShaderMaterial does not bind exactly one external shader")
    shader_id = shader_bindings[0]
    shader_resources = [
        match.groupdict()
        for line in lines
        if (match := re.fullmatch(
            r'\[ext_resource\b(?=[^\]]*\btype="Shader")(?=[^\]]*\bpath="(?P<path>[^"]+)")(?=[^\]]*\bid="(?P<id>[^"]+)")[^\]]*\]',
            line,
        ))
        and match.group("id") == shader_id
    ]
    if len(shader_resources) != 1 or shader_resources[0]["path"] != observer.get("shader_path"):
        raise RuntimeError("native shader observer path does not match the static ShaderMaterial binding")
    shader_path = qualification.project / str(observer["shader_path"]).removeprefix("res://")
    if shader_path.is_symlink() or not shader_path.is_file():
        raise RuntimeError("native shader observer shader resource is missing")
    try:
        shader_text = shader_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("native shader observer shader resource is not UTF-8") from exc
    uniform = re.compile(rf"^[ \t]*uniform[ \t]+float[ \t]+{re.escape(patch.parameter)}(?:[ \t]*=[^;]+)?[ \t]*;[ \t]*$", re.MULTILINE)
    if len(uniform.findall(shader_text)) != 1:
        raise RuntimeError("native shader resource does not declare exactly one matching float uniform")

    root_name: str | None = None
    bound_nodes: list[str] = []
    index = 0
    material_id = material_header.group("id")
    while index < len(lines):
        header = re.fullmatch(r'\[node\b(?P<attributes>[^\]]*)\]', lines[index])
        if header is None:
            index += 1
            continue
        attributes = header.group("attributes")
        name_match = re.search(r'\bname="(?P<name>[^"]+)"', attributes)
        parent_match = re.search(r'\bparent="(?P<parent>[^"]+)"', attributes)
        if name_match is None:
            raise RuntimeError("native shader scene contains an unnamed node")
        name = name_match.group("name")
        if root_name is None:
            if parent_match is not None:
                raise RuntimeError("native shader scene root node has a parent")
            root_name = name
            node_path = f"/root/{name}"
        else:
            if parent_match is None:
                raise RuntimeError("native shader scene contains multiple root nodes")
            parent = parent_match.group("parent")
            parent_suffix = "" if parent == "." else f"/{parent}"
            node_path = f"/root/{root_name}{parent_suffix}/{name}"
        block_end = index + 1
        while block_end < len(lines) and not lines[block_end].startswith("["):
            block_end += 1
        bindings = [
            match.group("id")
            for line in lines[index + 1:block_end]
            if (match := re.fullmatch(r'[ \t]*material[ \t]*=[ \t]*SubResource\("(?P<id>[^"]+)"\)[ \t]*', line))
        ]
        if material_id in bindings:
            bound_nodes.append(node_path)
        index = block_end
    if root_name is None or bound_nodes != [observer.get("node_path")]:
        raise RuntimeError("native shader observer node does not uniquely bind the declared ShaderMaterial")
    return observer, {
        "source_path": patch.source_path,
        "source_line": patch.source_line,
        "parameter": patch.parameter,
        "node_path": observer["node_path"],
        "resource_path": observer["resource_path"],
        "shader_path": observer["shader_path"],
        "shader_sha256": _sha256(shader_path),
        "material_binding": "STATIC_SCENE_SHADER_MATERIAL",
    }, factual_value


def _native_observed_qualification(
    qualification: NativeMainCaptureQualification,
    project: Path,
) -> NativeMainCaptureQualification:
    """Describe an already tree-verified replay copy for the existing validator."""
    trace_relative = qualification.trace.resolve().relative_to(qualification.project.resolve())
    trace = project / trace_relative
    original_config = project / ".flashpatch" / "upstream-project.godot"
    source_tree = _native_source_tree_sha256(project, original_project_config=original_config.read_bytes())
    scene = project / qualification.original_main_scene.removeprefix("res://")
    scene_sha = f"sha256:{hashlib.sha256(scene.read_bytes()).hexdigest()}"
    declared = dict(qualification.native_main)
    declared.update({
        "upstream_source_tree_sha256": source_tree,
        "copied_source_tree_sha256": source_tree,
        "upstream_main_scene_sha256": scene_sha,
        "copied_main_scene_sha256": scene_sha,
    })
    source_scenes = dict(declared.get("source_scenes", {}))
    source_scenes["from_scene"] = {"upstream_sha256": scene_sha, "copied_sha256": scene_sha}
    declared["source_scenes"] = source_scenes
    return NativeMainCaptureQualification(project, trace, qualification.original_main_scene, declared)


def _native_counterfactual_preservation(receipt: dict[str, object]) -> dict[str, object] | None:
    oracle = receipt.get("preservation_oracle")
    if oracle is None:
        return None
    fixture = receipt.get("controlled_fixture")
    if (
        receipt.get("controlled_mutation") is not True
        or receipt.get("upstream_defect") is not False
        or not isinstance(fixture, dict)
        or not isinstance(oracle, dict)
        or oracle.get("schema") != "flashpatch-controlled-native-preservation-oracle-v1"
        or oracle.get("controlled_fixture") != fixture
        or not isinstance(oracle.get("state_stream"), list)
        or not oracle["state_stream"]
        or oracle.get("terminal_completion") is not True
    ):
        raise RuntimeError("native shader counterfactual controlled preservation oracle is invalid")
    return dict(oracle)


def execute_native_main_shader_counterfactual(
    qualification: NativeMainCaptureQualification,
    output_directory: Path,
    *,
    patch: NativeTscnTokenPatch,
    replacement: int | float,
    godot_binary: Path | None = None,
) -> dict[str, object]:
    """Run one non-scoreable native ShaderMaterial factual/candidate pair."""
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise FileExistsError(f"native shader counterfactual output already exists: {output_directory}")
    if qualification.project.resolve() in output_directory.parents:
        raise ValueError("native shader counterfactual output must stay outside the sealed qualification")
    if isinstance(replacement, bool) or not isinstance(replacement, (int, float)) or not math.isfinite(float(replacement)):
        raise ValueError("native shader counterfactual replacement must be a finite numeric scalar")
    trace = _native_main_trace(qualification)
    _require_unchanged_native_main(qualification)
    observer, static_binding, factual_value = _native_shader_static_binding(qualification, trace, patch)
    replacement_value = float(replacement)
    if replacement_value == factual_value:
        raise ValueError("native shader counterfactual replacement must differ from the factual token")

    output_directory.mkdir(parents=True)
    factual_project = output_directory / "factual-project"
    candidate_project = output_directory / "candidate-project"
    shutil.copytree(qualification.project, factual_project)
    shutil.copytree(qualification.project, candidate_project)
    relative = _native_tscn_patch_relative_path(patch)
    candidate_source = candidate_project / relative
    candidate_bytes = candidate_source.read_bytes()
    _, candidate_start, candidate_token = _native_tscn_numeric_token(candidate_bytes, patch, label="factual")
    replacement_token = repr(replacement_value).encode("ascii")
    candidate_source.write_bytes(
        candidate_bytes[:candidate_start]
        + replacement_token
        + candidate_bytes[candidate_start + len(candidate_token):]
    )
    factual_source = factual_project / relative
    source_diff_digest = hashlib.sha256()
    source_diff_digest.update(relative.encode("utf-8"))
    source_diff_digest.update(b"\0")
    source_diff_digest.update(factual_source.read_bytes())
    source_diff_digest.update(b"\0")
    source_diff_digest.update(candidate_source.read_bytes())
    source_diff_sha256 = f"sha256:{source_diff_digest.hexdigest()}"

    factual_tree_before = _verify_native_main_candidate_tree(qualification, factual_project)
    candidate_tree_before = _verify_native_main_candidate_tree(qualification, candidate_project, patch=patch)
    factual_run = output_directory / "factual"
    candidate_run = output_directory / "candidate"
    factual_run.mkdir()
    candidate_run.mkdir()
    factual_output = factual_run / "replay.json"
    candidate_output = candidate_run / "replay.json"
    factual_receipt = classify_native_main_capture_qualification(
        _native_observed_qualification(qualification, factual_project),
        factual_output,
        godot_binary=godot_binary,
    )
    candidate_receipt = classify_native_main_capture_qualification(
        _native_observed_qualification(qualification, candidate_project),
        candidate_output,
        godot_binary=godot_binary,
    )
    factual_tree_after = _verify_native_main_candidate_tree(qualification, factual_project)
    candidate_tree_after = _verify_native_main_candidate_tree(qualification, candidate_project, patch=patch)
    if factual_tree_before != factual_tree_after or candidate_tree_before != candidate_tree_after:
        raise RuntimeError("native shader counterfactual project tree changed during replay")

    try:
        factual_replay = json.loads(factual_output.read_text(encoding="utf-8"))
        candidate_replay = json.loads(candidate_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native shader counterfactual replay output is unreadable") from exc
    if not isinstance(factual_replay, dict) or not isinstance(candidate_replay, dict):
        raise RuntimeError("native shader counterfactual replay output must be an object")
    expected_action_frames = [action["frame"] for action in trace["actions"]]
    if (
        factual_replay.get("action_frames") != expected_action_frames
        or candidate_replay.get("action_frames") != expected_action_frames
    ):
        raise RuntimeError("native shader counterfactual action frames do not exactly match the trace")
    if (
        factual_receipt.get("trace_sha256") != candidate_receipt.get("trace_sha256")
        or factual_receipt.get("trace_sha256") != f"sha256:{_sha256(qualification.trace)}"
    ):
        raise RuntimeError("native shader counterfactual factual and candidate traces differ")
    factual_hazards = factual_receipt.get("hazard_frame_indices")
    candidate_hazards = candidate_receipt.get("hazard_frame_indices")
    if factual_receipt.get("decision") != "HAZARDOUS_ATTRIBUTION_PENDING" or not isinstance(factual_hazards, list) or not factual_hazards:
        raise RuntimeError("native shader counterfactual factual replay does not establish a scenario-ready hazard")
    if candidate_receipt.get("decision") != "SAFE_SCENARIO_READY" or candidate_hazards != []:
        raise RuntimeError("native shader counterfactual candidate retains renderer hazard")

    identity = {
        "node_path": observer["node_path"],
        "source_path": observer["source_path"],
        "resource_path": observer["resource_path"],
        "source_line": observer["source_line"],
        "shader_path": observer["shader_path"],
        "property": observer["property"],
        "event_kind": "shader_parameter",
    }
    capture_frames = int(trace["capture_frames"])
    factual_events = factual_receipt.get("runtime_events")
    candidate_events = candidate_receipt.get("runtime_events")
    if not isinstance(factual_events, list) or not isinstance(candidate_events, list):
        raise RuntimeError("native shader counterfactual runtime event ledger is missing")
    for label, events, expected_value in (
        ("factual", factual_events, factual_value),
        ("candidate", candidate_events, replacement_value),
    ):
        if len(events) != capture_frames:
            raise RuntimeError(f"native shader counterfactual {label} runtime event ledger is incomplete")
        frames = []
        for event in events:
            if (
                not isinstance(event, dict)
                or any(event.get(key) != value for key, value in identity.items())
                or isinstance(event.get("factual_value"), bool)
                or not isinstance(event.get("factual_value"), (int, float))
                or float(event["factual_value"]) != expected_value
                or isinstance(event.get("frame_index"), bool)
                or not isinstance(event.get("frame_index"), int)
            ):
                raise RuntimeError(f"native shader counterfactual {label} runtime value or identity does not match source")
            frames.append(event["frame_index"])
        if frames != list(range(capture_frames)):
            raise RuntimeError(f"native shader counterfactual {label} runtime event frames are incomplete")
    if not set(factual_hazards).intersection(event["frame_index"] for event in factual_events):
        raise RuntimeError("native shader factual runtime observation does not join an exact hazard frame")

    factual_artifact = Path(str(factual_receipt["frame_artifact"]))
    candidate_artifact = Path(str(candidate_receipt["frame_artifact"]))
    try:
        with open_renderer_artifact(factual_artifact) as factual_frames:
            with open_renderer_artifact(candidate_artifact) as candidate_frames:
                if not np.array_equal(factual_frames.timestamps, candidate_frames.timestamps):
                    raise RuntimeError("native shader counterfactual renderer timelines differ")
                factual_rgb_sha256 = renderer_rgb_sha256(factual_frames.frames)
                candidate_rgb_sha256 = renderer_rgb_sha256(candidate_frames.frames)
                visual_change_ratio = renderer_visual_change_ratio(factual_frames.frames, candidate_frames.frames)
    except RendererArtifactError as exc:
        raise RuntimeError("native shader counterfactual renderer artifact is invalid") from exc

    factual_preservation = _native_counterfactual_preservation(factual_receipt)
    candidate_preservation = _native_counterfactual_preservation(candidate_receipt)
    preservation_available = factual_preservation is not None and candidate_preservation is not None
    if preservation_available and factual_preservation != candidate_preservation:
        raise RuntimeError("native shader counterfactual gameplay state or semantic invariants changed")
    decision = "PASS" if preservation_available else "INCONCLUSIVE"
    reason = (
        "hazard_removed_with_equal_controlled_native_state_stream"
        if preservation_available
        else "native_state_or_semantic_preservation_oracle_missing"
    )
    receipt: dict[str, object] = {
        "schema": "flashpatch-native-main-shader-counterfactual-v1",
        "decision": decision,
        "reason": reason,
        "qualification_only": True,
        "scoreable": False,
        "native_equivalence": "NOT_ESTABLISHED",
        "execution_mode": "instrumented_native_main_shader_counterfactual",
        "trace_sha256": factual_receipt["trace_sha256"],
        "source_diff_sha256": source_diff_sha256,
        "static_binding": static_binding,
        "patch": {
            "source_path": patch.source_path,
            "parameter": patch.parameter,
            "source_line": patch.source_line,
            "change_kind": "ONE_FINITE_NUMERIC_TOKEN",
        },
        "factual": {
            "tree_binding_before": factual_tree_before,
            "tree_binding_after": factual_tree_after,
            "tree_receipt_sha256": _native_counterfactual_json_sha256(factual_tree_after),
            "trace_sha256": _sha256(factual_project / qualification.trace.relative_to(qualification.project)),
            "replay": str(factual_output),
            "replay_sha256": _sha256(factual_output),
            "runtime_events_sha256": _native_counterfactual_json_sha256(factual_events),
            "renderer_artifact": str(factual_artifact),
            "renderer_artifact_sha256": factual_receipt["frame_artifact_sha256"],
            "renderer_rgb_sha256": factual_rgb_sha256,
            "hazard_frame_indices": factual_hazards,
        },
        "candidate": {
            "tree_binding_before": candidate_tree_before,
            "tree_binding_after": candidate_tree_after,
            "tree_receipt_sha256": _native_counterfactual_json_sha256(candidate_tree_after),
            "trace_sha256": _sha256(candidate_project / qualification.trace.relative_to(qualification.project)),
            "replay": str(candidate_output),
            "replay_sha256": _sha256(candidate_output),
            "runtime_events_sha256": _native_counterfactual_json_sha256(candidate_events),
            "renderer_artifact": str(candidate_artifact),
            "renderer_artifact_sha256": candidate_receipt["frame_artifact_sha256"],
            "renderer_rgb_sha256": candidate_rgb_sha256,
            "hazard_frame_indices": candidate_hazards,
        },
        "renderer_comparison": {
            "factual_hazardous": True,
            "candidate_residual_hazard": False,
            "hazard_removed": True,
            "same_trace": True,
            "action_frames_exact": True,
            "runtime_identity_equal": True,
            "visual_change_ratio": visual_change_ratio,
        },
        "preservation": {
            "oracle_available": preservation_available,
            "gameplay_state_equal": preservation_available,
            "semantic_invariants_equal": preservation_available,
            "typed_full_state_stream_equal": preservation_available,
            "frame_domain_sha256": (
                factual_preservation["frame_domain_sha256"] if preservation_available else None
            ),
            "state_stream_sha256": (
                factual_preservation["state_stream_sha256"] if preservation_available else None
            ),
            "typed_full_state_stream": (
                factual_preservation["state_stream"] if preservation_available else None
            ),
            "terminal_completion": (
                factual_preservation["terminal_completion"] if preservation_available else None
            ),
            "terminal_state": (
                factual_preservation["terminal_state"] if preservation_available else None
            ),
            "player_world_digest": (
                factual_preservation["player_world_digest"] if preservation_available else None
            ),
            "score": factual_preservation["score"] if preservation_available else None,
        },
    }
    receipt_path = output_directory / "native-shader-counterfactual-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path)}


def classify_capture_only_qualification(
    qualification: CaptureOnlyQualification,
    output: Path,
    *,
    godot_binary: Path | None = None,
) -> dict[str, object]:
    """Capture actual Godot-4 frames and return a non-patch screen verdict."""
    config = (qualification.project / "project.godot").read_text(encoding="utf-8")
    runner_class = Godot3RendererReplayRunner if _declared_godot_major(config) == 3 else GodotRendererReplayRunner
    runner = runner_class(qualification.project, godot_binary=godot_binary)
    replay = runner.replay(qualification.trace, output)
    frame_path = output.parent / str(replay["frames_npz"])
    try:
        with open_renderer_artifact(frame_path) as artifact:
            result = analyze(artifact.frames, artifact.timestamps)
            frame_count = len(artifact.frames)
    except RendererArtifactError as exc:
        raise RuntimeError("capture-only packed RGB frame artifact is invalid") from exc
    hazardous = result.hazardous
    receipt = {
        "schema": "flashpatch-godot-capture-only-qualification-v1",
        "decision": "SAFE" if not hazardous else ("HAZARDOUS_ATTRIBUTION_PENDING" if qualification.visual_candidates else "HAZARDOUS_PATCH_INELIGIBLE"),
        "reason": "no_hazard_in_declared_trace" if not hazardous else ("renderer_hazard_has_declared_visual_candidates_but_requires_runtime_attribution_and_preservation_oracles" if qualification.visual_candidates else "renderer_hazard_has_no_declared_visual_patch_candidate"),
        "controlled_mutation": False,
        "qualification_only": True,
        "scoreable": False,
        "execution_mode": "instrumented_wrapper_capture_only",
        "original_main_scene": qualification.original_main_scene,
        "visual_candidates": list(qualification.visual_candidates),
        "source_provenance": qualification.source_provenance,
        "trace_sha256": f"sha256:{hashlib.sha256(qualification.trace.read_bytes()).hexdigest()}",
        "frame_artifact": str(frame_path),
        "frame_artifact_sha256": _sha256(frame_path),
        "frame_count": frame_count,
        "max_risk": result.max_flash_count if result.hazardous else 0.0,
        "hazard_frame_indices": np.flatnonzero(np.any(result.hazard_mask, axis=(1, 2))).tolist(),
    }
    receipt_path = output.with_name("qualification-receipt.json")
    receipt["replay_sha256"] = _sha256(output)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path)}


def execute_repeated_capture_only_qualification(
    upstream_project: Path,
    output_root: Path,
    *,
    godot_binary: Path | None = None,
    fixed_fps: int = 60,
    capture_frames: int = 121,
    actions: list[dict[str, object]] | None = None,
    repository: str | None = None,
    revision: str | None = None,
    repeats: int = 3,
) -> dict[str, object]:
    """Repeat a screen-only external qualification without minting a score."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"qualification repeat output already exists: {output_root}")
    if repeats != 3:
        raise ValueError("capture-only qualification requires exactly three repeats")
    output_root.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    for index in range(repeats):
        run_root = output_root / f"run-{index + 1:02d}"
        qualification = materialize_capture_only_qualification(
            upstream_project, run_root / "project", fixed_fps=fixed_fps,
            capture_frames=capture_frames, actions=actions, repository=repository,
            revision=revision,
        )
        try:
            receipt = classify_capture_only_qualification(
                qualification, run_root / "replay.json", godot_binary=godot_binary,
            )
            runs.append({
                "repeat": index + 1,
                "status": "PROCESS_VALID",
                "decision": receipt["decision"],
                "frame_artifact_sha256": receipt["frame_artifact_sha256"],
                "trace_sha256": receipt["trace_sha256"],
                "receipt": str(Path(str(receipt["receipt"])).relative_to(output_root)),
            })
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            runs.append({"repeat": index + 1, "status": "INCONCLUSIVE", "reason": str(exc)})
    valid = [run for run in runs if run["status"] == "PROCESS_VALID"]
    reproducible = (
        len(valid) == repeats
        and len({str(run["decision"]) for run in valid}) == 1
        and len({str(run["frame_artifact_sha256"]) for run in valid}) == 1
        and len({str(run["trace_sha256"]) for run in valid}) == 1
    )
    receipt = {
        "schema": "flashpatch-godot-capture-only-qualification-repeats-v1",
        "repeats_required": repeats,
        "runs": runs,
        "status": "PROCESS_REPRODUCIBLE" if reproducible else "INCONCLUSIVE",
        "scoreable": False,
        "scoreable_blockers": [
            "screen_qualification_only",
            "runtime_attribution_missing",
            "gameplay_preservation_oracle_missing",
            "independent_gold_missing",
            "comparator_repeats_missing",
        ],
    }
    receipt_path = output_root / "repeat-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path)}


class SpartaRendererRunner:
    """Adapter for Sparta's own deterministic demo recorder.

    It uses Sparta's non-headless PNG capture and state-hash stream, then
    returns the FlashPatch renderer replay shape.  The project copy owns the
    injected effect and runtime event log; the upstream checkout is never
    written by this runner.
    """

    def __init__(
        self,
        project: Path,
        *,
        godot_binary: Path,
        capture_ticks: int = 161,
        timeout_seconds: int = 300,
    ) -> None:
        self.project = Path(project).resolve()
        self.godot_binary = Path(godot_binary).resolve()
        self.capture_ticks = capture_ticks
        self.timeout_seconds = timeout_seconds

    def _run_process(self, command: list[str], *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        """Run an adapter command in its own process group and reap all children."""
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise RuntimeError("Sparta renderer adapter command timed out") from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, "")

    def replay(self, trace: Path, output: Path) -> dict[str, object]:
        trace = Path(trace).resolve()
        output = Path(output).resolve()
        if not self.godot_binary.is_file():
            raise RuntimeError("Sparta renderer adapter Godot executable is unavailable")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            trace_relative = trace.relative_to(self.project)
        except ValueError:
            staged_trace = self.project / ".flashpatch-runtime-trace.json"
            shutil.copy2(trace, staged_trace)
            trace = staged_trace
            trace_relative = trace.relative_to(self.project)
        frames = output.parent / "renderer-capture"
        state = output.parent / "state"
        events = output.parent / "runtime-events.jsonl"
        capture_metadata = output.parent / "capture-metadata.jsonl"
        action_events = output.parent / "action-events.jsonl"
        ticks = ",".join(str(index) for index in range(self.capture_ticks))
        environment = os.environ | {
            "SPARTA_DEMO_INPUT": f"res://{trace_relative.as_posix()}",
            "SPARTA_DEMO_FRAMES": ticks,
            "SPARTA_DEMO_FRAME_DIR": str(frames),
            "SPARTA_DEMO_STATE": str(self.capture_ticks - 1),
            "SPARTA_DEMO_STATE_DIR": str(state),
            "FLASHPATCH_RUNTIME_EVENTS": str(events),
            "FLASHPATCH_CAPTURE_METADATA": str(capture_metadata),
            "FLASHPATCH_ACTION_EVENTS": str(action_events),
            "FLASHPATCH_FIXED_FPS": "60",
        }
        prepare_command = [
            str(self.godot_binary), "--headless", "--import",
            "--path", str(self.project),
        ]
        try:
            prepared = self._run_process(prepare_command, environment=environment)
        except RuntimeError as exc:
            raise RuntimeError("Sparta renderer adapter preparation timed out") from exc
        (output.parent / "godot-prepare.stdout.log").write_text(prepared.stdout, encoding="utf-8")
        (output.parent / "godot-prepare.stderr.log").write_text(prepared.stderr, encoding="utf-8")
        if prepared.returncode != 0:
            raise RuntimeError(f"Sparta renderer adapter project preparation exited {prepared.returncode}")
        command = [
            "xvfb-run", "-a", "-s", "-screen 0 1280x720x24",
            str(self.godot_binary), "--rendering-driver", "opengl3", "--fixed-fps", "60",
            "--path", str(self.project), "res://tools/demo/DemoInputRecorder.tscn",
        ]
        try:
            completed = self._run_process(command, environment=environment)
        except RuntimeError as exc:
            raise RuntimeError("Sparta renderer adapter capture timed out") from exc
        (output.parent / "godot.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output.parent / "godot.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Sparta renderer adapter exited {completed.returncode}")
        packed = pack_renderer_png_sequence(frames, output.parent / "renderer-pack", fps=60)
        packed_artifact = Path(str(packed["artifact"]["path"]))
        source_artifact = output.parent / "renderer-pack" / packed_artifact
        frame_artifact = output.parent / "renderer-frames.npz"
        shutil.copy2(source_artifact, frame_artifact)
        state_stream = state / "hash_stream.jsonl"
        if (
            not state_stream.is_file()
            or not events.is_file()
            or not capture_metadata.is_file()
            or not action_events.is_file()
        ):
            raise RuntimeError(
                "Sparta renderer adapter omitted state, capture, action, or runtime event evidence"
            )
        runtime_events = self._read_jsonl(events, "runtime event")
        capture_events = self._read_jsonl(capture_metadata, "capture metadata")
        action_event_records = self._read_jsonl(action_events, "action event")
        expected_frames = list(range(self.capture_ticks))
        capture_frame_indices = [event.get("frame_index") for event in capture_events]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in capture_frame_indices
            )
            or capture_frame_indices != expected_frames
        ):
            raise RuntimeError("Sparta renderer capture metadata does not cover every frame exactly once")
        actual_capture_timestamps_us = [
            event.get("actual_capture_timestamp_us") for event in capture_events
        ]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in actual_capture_timestamps_us)
            or any(
                right <= left
                for left, right in zip(
                    actual_capture_timestamps_us, actual_capture_timestamps_us[1:]
                )
            )
        ):
            raise RuntimeError("Sparta renderer capture clock is missing or non-monotonic")
        viewport_values = {tuple(event.get("viewport", [])) for event in capture_events}
        if viewport_values != {(1280, 720)}:
            raise RuntimeError("Sparta renderer viewport provenance is missing or inconsistent")
        presentation_timestamps_us = [
            (index * 1_000_000) // 60 for index in expected_frames
        ]
        if [
            event.get("presentation_timestamp_us") for event in capture_events
        ] != presentation_timestamps_us:
            raise RuntimeError("Sparta renderer presentation timeline is missing or inconsistent")
        packed_source = packed.get("source_capture")
        packed_frames = packed_source.get("frames") if isinstance(packed_source, dict) else None
        if (
            not isinstance(packed_frames, list)
            or len(packed_frames) != self.capture_ticks
            or not all(isinstance(item, dict) for item in packed_frames)
        ):
            raise RuntimeError("Sparta renderer pack omitted source PNG provenance")
        if any(
            event.get("capture_kind") != "godot_viewport_rgb_png_sequence"
            or event.get("capture_api") != "get_viewport().get_texture().get_image()"
            or event.get("pixel_format") != "Image.FORMAT_RGB8"
            or event.get("pixel_format_id") != 4
            or event.get("source_pixel_format") != "Image.FORMAT_RGBA8"
            or not isinstance(event.get("source_pixel_format_id"), int)
            or event.get("source_pixel_format_api")
            != "Image.get_format() before convert(Image.FORMAT_RGB8)"
            or event.get("viewport_use_hdr_2d") is not False
            or event.get("png_file") != packed_frames[index].get("path")
            for index, event in enumerate(capture_events)
        ):
            raise RuntimeError(
                "Sparta renderer capture API, RGB format, or HDR-2D observation is malformed"
            )
        viewport_use_hdr_2d = capture_events[0]["viewport_use_hdr_2d"]
        image_format = capture_events[0]["pixel_format"]
        color_observation_fields = (
            "display_server",
            "display_server_api",
            "rendering_method",
            "rendering_method_api",
            "rendering_driver",
            "rendering_driver_api",
            "hdr_output_supported",
            "hdr_output_requested",
            "hdr_output_enabled",
            "hdr_output_api",
        )
        if any(
            any(field not in event for field in color_observation_fields)
            for event in capture_events
        ):
            raise RuntimeError("Sparta renderer color-pipeline observations are missing")
        color_observations = {
            field: capture_events[0][field] for field in color_observation_fields
        }
        if any(
            {field: event[field] for field in color_observation_fields}
            != color_observations
            for event in capture_events[1:]
        ):
            raise RuntimeError("Sparta renderer color-pipeline observations are inconsistent")
        if color_observations != {
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
        }:
            raise RuntimeError("Sparta renderer color-pipeline contract is not the pinned X11 Compatibility lane")

        source_path = self.project / "scripts" / "RoutShockwave.gd"
        source_hash = _sha256(source_path)
        source_line = self._exported_parameter_line(source_path, "flashpatch_intensity")
        runtime_frame_indices = [event.get("frame_index") for event in runtime_events]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in runtime_frame_indices
            )
            or runtime_frame_indices != expected_frames
        ):
            raise RuntimeError("Sparta runtime evidence does not cover every captured frame exactly once")
        capture_by_frame = {int(event["frame_index"]): event for event in capture_events}
        required_runtime_fields = {
            "actual_capture_timestamp_us",
            "frame_index",
            "node_path",
            "normalized_node_identity",
            "spawned_ordinal",
            "script_path",
            "script_path_observation",
            "script_sha256",
            "resource_path",
            "resource_path_observation",
            "resource_provenance",
            "source_line",
            "source_line_observation",
            "property",
            "factual_value",
            "event_kind",
        }
        for event in runtime_events:
            frame_index = event.get("frame_index")
            capture_event = capture_by_frame.get(frame_index) if isinstance(frame_index, int) else None
            if not required_runtime_fields.issubset(event):
                raise RuntimeError("Sparta runtime provenance fields are missing")
            if (
                capture_event is None
                or event.get("actual_capture_timestamp_us")
                != capture_event.get("actual_capture_timestamp_us")
            ):
                raise RuntimeError("Sparta runtime event is not bound to its captured frame timestamp")
            if (
                not isinstance(event.get("node_path"), str)
                or not str(event["node_path"]).startswith("/root/")
                or event.get("normalized_node_identity")
                != "res://scenes/Battle.tscn::res://scripts/RoutShockwave.gd::0"
                or isinstance(event.get("spawned_ordinal"), bool)
                or not isinstance(event.get("spawned_ordinal"), int)
                or event.get("spawned_ordinal") != 0
                or event.get("script_path") != "res://scripts/RoutShockwave.gd"
                or event.get("script_path_observation") != "node.get_script().resource_path"
                or event.get("script_sha256") != source_hash
                or event.get("resource_path") != "res://scenes/Battle.tscn"
                or event.get("resource_path_observation") != "battle.scene_file_path"
                or event.get("resource_provenance") != "packed_scene_state"
                or event.get("source_line") != source_line
                or event.get("source_line_observation")
                != "FileAccess.get_file_as_string(node.get_script().resource_path)"
                or event.get("property") != "flashpatch_intensity"
                or event.get("event_kind") != "render_property"
            ):
                raise RuntimeError("Sparta runtime provenance does not match the executed source")
        trace_payload = json.loads(trace.read_text(encoding="utf-8"))
        trace_actions = trace_payload.get("actions")
        if not isinstance(trace_actions, list) or not trace_actions:
            raise RuntimeError("Sparta renderer trace actions are missing")
        action_acknowledgements = self._action_acknowledgements(
            trace_actions,
            action_event_records,
            runtime_events,
        )
        action_frames = [item["frame"] for item in trace_actions]
        preservation = self._state_preservation_evidence(state, self.capture_ticks)
        state_digest = str(preservation["state_stream_sha256"])
        version = subprocess.run([str(self.godot_binary), "--version"], text=True, capture_output=True, check=False)
        if version.returncode != 0 or not version.stdout.strip():
            raise RuntimeError("Sparta renderer adapter could not identify Godot")
        result = {
            "status": "REPLAYED",
            "frames_npz": frame_artifact.name,
            "action_frames": action_frames,
            "action_acknowledgements": action_acknowledgements,
            "action_acknowledgement_evidence": action_event_records,
            "gameplay_state": state_digest,
            "semantic_invariants": {
                "terminal_completion": True,
                "terminal_state": preservation["final_state_sha256"],
                "player_world_digest": state_digest,
                "score": "score_not_applicable",
            },
            "state_stream_sha256": preservation["state_stream_sha256"],
            "state_stream_artifact": preservation["state_stream_artifact"],
            "state_stream_tick_domain": preservation["state_stream_tick_domain"],
            "state_stream_record_count": preservation["state_stream_record_count"],
            "final_state_sha256": preservation["final_state_sha256"],
            "final_state_raw_sha256": preservation["final_state_raw_sha256"],
            "final_state_artifact": preservation["final_state_artifact"],
            "tick_domain": [0, self.capture_ticks - 1],
            "runtime_events": runtime_events,
            "runtime_script_sha256": source_hash,
            "runtime_source_line": source_line,
            "renderer_capture": {
                "trace_sha256": f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}",
                "godot_version": version.stdout.strip().splitlines()[0],
                "renderer_configuration": {"display_driver": "x11", "rendering_driver": "opengl3"},
                "frame_count": self.capture_ticks,
                "presentation_timestamps_us": presentation_timestamps_us,
                "actual_capture_timestamps_us": actual_capture_timestamps_us,
                "capture_kind": "godot_viewport_rgb_png_sequence",
                "capture_api": "get_viewport().get_texture().get_image()",
                "viewport": [1280, 720],
                "pixel_format": "Image.FORMAT_RGB8",
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
                        "viewport_use_hdr_2d": viewport_use_hdr_2d,
                        "viewport_use_hdr_2d_api": "get_viewport().use_hdr_2d",
                        "image_format": image_format,
                        "image_format_api": "Image.get_format()",
                        **color_observations,
                    },
                },
                "packed_receipt_sha256": _sha256(output.parent / "renderer-pack" / "renderer-pack-receipt.json"),
            },
        }
        output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return result

    @staticmethod
    def _action_acknowledgements(
        trace_actions: list[object],
        action_events: list[dict[str, object]],
        runtime_events: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        expected: list[tuple[int, str]] = []
        for action in trace_actions:
            if (
                not isinstance(action, dict)
                or isinstance(action.get("frame"), bool)
                or not isinstance(action.get("frame"), int)
                or not isinstance(action.get("action"), str)
                or not action.get("action")
            ):
                raise RuntimeError("Sparta renderer trace action is malformed")
            expected.append((int(action["frame"]), str(action["action"])))
        observed = [
            (event.get("frame"), event.get("action")) for event in action_events
        ]
        if observed != expected or any(
            event.get("status") != "APPLIED" for event in action_events
        ):
            raise RuntimeError(
                "Sparta renderer did not acknowledge every trace action exactly once"
            )
        runtime_by_frame = {
            event.get("frame_index"): event
            for event in runtime_events
            if isinstance(event.get("frame_index"), int)
        }
        for event in action_events:
            runtime = runtime_by_frame.get(event.get("frame"))
            if (
                event.get("observation") != "runtime_node_present_at_capture"
                or runtime is None
                or event.get("node_path") != runtime.get("node_path")
                or event.get("property") != runtime.get("property")
                or event.get("factual_value") != runtime.get("factual_value")
            ):
                raise RuntimeError(
                    "Sparta renderer action acknowledgement lacks runtime evidence"
                )
        return [{"frame": frame, "status": "APPLIED"} for frame, _ in expected]

    @classmethod
    def _state_preservation_evidence(
        cls,
        state_directory: Path,
        capture_ticks: int,
    ) -> dict[str, object]:
        stream = state_directory / "hash_stream.jsonl"
        records = cls._read_jsonl(stream, "state hash")
        ticks = [record.get("tick") for record in records]
        if (
            any(isinstance(tick, bool) or not isinstance(tick, int) for tick in ticks)
            or ticks != list(range(len(ticks)))
            or ticks[:capture_ticks] != list(range(capture_ticks))
        ):
            raise RuntimeError(
                "Sparta state hash stream is not contiguous across the capture domain"
            )
        final_tick = capture_ticks - 1
        final_state = state_directory / f"state_{final_tick:05d}.json"
        try:
            payload = json.loads(final_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Sparta final state snapshot is missing or malformed"
            ) from exc
        if not isinstance(payload, dict) or payload.get("tick") != final_tick:
            raise RuntimeError(
                "Sparta final state snapshot does not match the capture terminal tick"
            )
        canonical = (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        return {
            "state_stream_sha256": _sha256(stream),
            "state_stream_artifact": str(stream),
            "state_stream_tick_domain": [ticks[0], ticks[-1]],
            "state_stream_record_count": len(records),
            "final_state_sha256": hashlib.sha256(canonical).hexdigest(),
            "final_state_raw_sha256": _sha256(final_state),
            "final_state_artifact": str(final_state),
        }

    @staticmethod
    def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
        try:
            parsed = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Sparta renderer {label} log is malformed") from exc
        if not parsed or not all(isinstance(item, dict) for item in parsed):
            raise RuntimeError(f"Sparta renderer emitted no valid {label} records")
        return parsed

    @staticmethod
    def _exported_parameter_line(source: Path, parameter: str) -> int:
        matches = [
            index
            for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1)
            if re.match(rf"^\s*@export\s+var\s+{re.escape(parameter)}\s*:", line)
        ]
        if len(matches) != 1:
            raise RuntimeError("Sparta renderer source has no unique exported parameter line")
        return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(output_root: Path, controlled: ControlledPong, receipt: dict[str, object]) -> list[Path]:
    """Select evidence outputs only, never every transient project-copy file."""
    selected = [controlled.mutation_script, controlled.contract, controlled.trace, controlled.project / "pong.tscn"]
    factual = receipt.get("factual_replay")
    if isinstance(factual, dict):
        for key in ("artifact", "frame_artifact"):
            value = factual.get(key)
            if isinstance(value, str):
                selected.append(Path(value))
    attribution = receipt.get("attribution")
    if isinstance(attribution, dict):
        for key in ("artifact", "frame_artifact", "diff", "source"):
            value = attribution.get(key)
            if isinstance(value, str):
                selected.append(Path(value))
    for directory in output_root.glob("workspace/**/renderer-capture"):
        selected.extend(sorted(directory.glob("frame_*.png")))
    output = []
    root = output_root.resolve()
    for path in selected:
        resolved = path.resolve()
        if resolved.is_file() and root in resolved.parents and resolved not in output:
            output.append(resolved)
    return output


def materialize_controlled_pong(upstream_project: Path, destination: Path) -> ControlledPong:
    """Copy a pinned upstream Pong project and add its explicit mutation only."""
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    if not (upstream_project / "project.godot").is_file() or not (upstream_project / "pong.tscn").is_file():
        raise ValueError("upstream_project must be the pinned Godot demo 2d/pong root")
    if destination.exists():
        raise FileExistsError(f"controlled destination already exists: {destination}")
    shutil.copytree(upstream_project, destination, ignore=shutil.ignore_patterns(".git", ".godot", ".claude"))
    script = destination / "flashpatch_probe.gd"
    script.write_text(PROBE_SCRIPT, encoding="utf-8")
    scene = destination / "pong.tscn"
    scene_text = scene.read_text(encoding="utf-8")
    if 'path="res://flashpatch_probe.gd"' in scene_text:
        raise ValueError("controlled source already has a FlashPatch probe")
    scene_text = scene_text.replace(
        '[ext_resource type="Script" path="res://logic/ceiling_floor.gd" id="8"]',
        '[ext_resource type="Script" path="res://logic/ceiling_floor.gd" id="8"]\n[ext_resource type="Script" path="res://flashpatch_probe.gd" id="9"]',
        1,
    )
    scene_text = scene_text.replace('[node name="Pong" type="Node2D"]', '[node name="Pong" type="Node2D"]\nscript = ExtResource("9")', 1)
    scene.write_text(scene_text, encoding="utf-8")
    trace = destination / "flashpatch-trace.json"
    trace.write_text(json.dumps({"fixed_fps": 60, "actions": [
        {"frame": 0, "action": "left_move_down", "pressed": True},
        {"frame": 12, "action": "left_move_down", "pressed": False},
    ]}, indent=2) + "\n", encoding="utf-8")
    contract = destination / "flashpatch.renderer.contract.json"
    contract.write_text(json.dumps({
        "schema": "flashpatch-godot-safety-ci-v1",
        "trace": trace.name,
        "scene": scene.name,
        "timing_field": "action_frames",
        "state_field": "gameplay_state",
        "risk_signal": {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0},
        "patch_candidates": [{"source": script.name, "parameter": "flash_intensity", "parameter_kind": "intensity", "replacement": 0.0}],
    }, indent=2) + "\n", encoding="utf-8")
    return ControlledPong(destination, contract, trace, script, {
        "repository_url": GODOT_DEMO_REPOSITORY,
        "source_revision": GODOT_DEMO_REVISION,
        "license": GODOT_DEMO_LICENSE,
        "project_path": PONG_PROJECT_PATH,
    })


def materialize_controlled_sparta(upstream_project: Path, destination: Path) -> ControlledSparta:
    """Create a labelled Sparta copy with a dynamic-effect-only mutation."""
    upstream_project = Path(upstream_project).resolve()
    destination = Path(destination).resolve()
    required = (
        upstream_project / "project.godot",
        upstream_project / "tools" / "demo" / "DemoInputRecorder.tscn",
        upstream_project / "tools" / "demo" / "DemoInputRecorder.gd",
        upstream_project / "scripts" / "RoutShockwave.gd",
        upstream_project / "scenes" / "Battle.tscn",
    )
    if not all(path.is_file() for path in required):
        raise ValueError("upstream_project must be the pinned Sparta project root")
    if destination.exists():
        raise FileExistsError(f"controlled destination already exists: {destination}")
    shutil.copytree(upstream_project, destination, ignore=shutil.ignore_patterns(".git", ".godot", ".claude"))
    shockwave = destination / "scripts" / "RoutShockwave.gd"
    source = shockwave.read_text(encoding="utf-8")
    if "@export var flashpatch_intensity" in source:
        raise ValueError("controlled Sparta source already has a FlashPatch parameter")
    source = source.replace(
        "extends TransientEffect\n",
        "extends TransientEffect\n\n@export var flashpatch_intensity: float = 1.0\n",
        1,
    )
    source = source.replace(
        "\t\tdraw_circle(Vector2.ZERO, disc_r, Color(_color, fade * 0.1))",
        "\t\tdraw_circle(Vector2.ZERO, disc_r, Color.WHITE * (fade * 0.55 * flashpatch_intensity * (1.0 if Engine.get_physics_frames() % 2 == 0 else 0.0)))",
        1,
    )
    source = source.replace(
        "\tdraw_arc(Vector2.ZERO, r, 0.0, TAU, 32, Color(_color, fade * 0.6), 2.0)",
        "\tdraw_arc(Vector2.ZERO, r, 0.0, TAU, 32, Color.WHITE * (fade * flashpatch_intensity * (1.0 if Engine.get_physics_frames() % 2 == 0 else 0.0)), 2.0)",
        1,
    )
    exported_line = next(
        (index for index, line in enumerate(source.splitlines(), start=1) if line.startswith("@export var flashpatch_intensity:")),
        None,
    )
    if exported_line is None:
        raise ValueError("controlled Sparta exported parameter source line is unavailable")
    shockwave.write_text(source, encoding="utf-8")
    recorder = destination / "tools" / "demo" / "DemoInputRecorder.gd"
    recorder_source = recorder.read_text(encoding="utf-8")
    if not recorder_source.startswith("extends Node\n"):
        raise ValueError("Sparta recorder must begin with its expected Node base")
    recorder_source = recorder_source.replace(
        "extends Node\n",
        "extends Node\n\nvar _flashpatch_capture_start_us: int = 0\n",
        1,
    )
    recorder_source = recorder_source.replace(
        "\t_cam = _battle.get_node(\"Camera2D\")\n\t_apply_camera(0)",
        "\t_cam = _battle.get_node(\"Camera2D\")\n\t_flashpatch_capture_start_us = Time.get_ticks_usec()\n\tRoutShockwave.spawn(_battle, Vector2(800.0, 480.0), 1400.0, Color.WHITE, 4.0)\n\t_apply_camera(0)",
        1,
    )
    capture_anchor = "\tvar err: int = img.save_png(path)"
    if capture_anchor not in recorder_source:
        raise ValueError("Sparta recorder PNG capture anchor is unavailable")
    recorder_source = recorder_source.replace(
        capture_anchor,
        "\tvar flashpatch_source_pixel_format := img.get_format()\n"
        "\timg.convert(Image.FORMAT_RGB8)\n" + capture_anchor,
        1,
    )
    success_anchor = (
        "\telse:\n"
        "\t\tprint(\"[demo-input] captured frame at tick %d -> %s (%dx%d)\" % "
    )
    if success_anchor not in recorder_source:
        raise ValueError("Sparta recorder successful PNG capture anchor is unavailable")
    recorder_source = recorder_source.replace(
        success_anchor,
        "\telse:\n\t\t_flashpatch_record_capture(tick, img, path, flashpatch_source_pixel_format)\n"
        "\t\tprint(\"[demo-input] captured frame at tick %d -> %s (%dx%d)\" % ",
        1,
    )
    marker = "\n\n## Set the camera to the track's framing for"
    instrumentation = """

func _flashpatch_append_jsonl(path: String, event: Dictionary) -> void:
\tif path == "":
\t\treturn
\tvar stream := FileAccess.open(path, FileAccess.READ_WRITE if FileAccess.file_exists(path) else FileAccess.WRITE_READ)
\tif stream == null:
\t\tpush_error("[flashpatch] runtime evidence stream could not be opened: %s" % path)
\t\treturn
\tstream.seek_end()
\tstream.store_string(JSON.stringify(event) + "\\n")


func _flashpatch_exported_line(source_text: String, property_name: String) -> int:
\tvar lines := source_text.split("\\n")
\tvar prefix := "@export var %s:" % property_name
\tfor index in range(lines.size()):
\t\tif str(lines[index]).strip_edges().begins_with(prefix):
\t\t\treturn index + 1
\treturn 0


func _flashpatch_record_capture(tick: int, image: Image, png_path: String, source_pixel_format: int) -> void:
\tvar actual_us := Time.get_ticks_usec() - _flashpatch_capture_start_us
\tvar fixed_fps := int(OS.get_environment("FLASHPATCH_FIXED_FPS"))
\tif fixed_fps <= 0:
\t\tpush_error("[flashpatch] fixed FPS evidence is unavailable")
\t\treturn
\tvar viewport_size := get_viewport().get_texture().get_size()
\t_flashpatch_append_jsonl(OS.get_environment("FLASHPATCH_CAPTURE_METADATA"), {
\t\t"frame_index": tick,
\t\t"presentation_timestamp_us": int(tick * 1000000 / fixed_fps),
\t\t"actual_capture_timestamp_us": actual_us,
\t\t"engine_process_frame": Engine.get_process_frames(),
\t\t"viewport": [int(viewport_size.x), int(viewport_size.y)],
\t\t"capture_kind": "godot_viewport_rgb_png_sequence",
\t\t"capture_api": "get_viewport().get_texture().get_image()",
\t\t"pixel_format": "Image.FORMAT_RGB8",
\t\t"pixel_format_id": image.get_format(),
\t\t"source_pixel_format": "Image.FORMAT_RGBA8" if source_pixel_format == Image.FORMAT_RGBA8 else "OTHER",
\t\t"source_pixel_format_id": source_pixel_format,
\t\t"source_pixel_format_api": "Image.get_format() before convert(Image.FORMAT_RGB8)",
\t\t"viewport_use_hdr_2d": get_viewport().use_hdr_2d,
\t\t"display_server": DisplayServer.get_name(),
\t\t"display_server_api": "DisplayServer.get_name()",
\t\t"rendering_method": RenderingServer.get_current_rendering_method(),
\t\t"rendering_method_api": "RenderingServer.get_current_rendering_method()",
\t\t"rendering_driver": RenderingServer.get_current_rendering_driver_name(),
\t\t"rendering_driver_api": "RenderingServer.get_current_rendering_driver_name()",
\t\t"hdr_output_supported": DisplayServer.window_is_hdr_output_supported(),
\t\t"hdr_output_requested": DisplayServer.window_is_hdr_output_requested(),
\t\t"hdr_output_enabled": DisplayServer.window_is_hdr_output_enabled(),
\t\t"hdr_output_api": "DisplayServer.window_is_hdr_output_supported/requested/enabled()",
\t\t"png_file": png_path.get_file(),
\t})
\tvar ordinal := 0
\tfor child in _battle.get_children():
\t\tif child is RoutShockwave:
\t\t\tvar script_resource: Script = child.get_script() as Script
\t\t\tvar script_path := script_resource.resource_path if script_resource != null else ""
\t\t\tvar source_text := FileAccess.get_file_as_string(script_path) if script_path != "" else ""
\t\t\tvar resource_path := _battle.scene_file_path
\t\t\t_flashpatch_append_jsonl(OS.get_environment("FLASHPATCH_RUNTIME_EVENTS"), {
\t\t\t\t"actual_capture_timestamp_us": actual_us,
\t\t\t\t"frame_index": tick,
\t\t\t\t"node_path": str(child.get_path()),
\t\t\t\t"normalized_node_identity": "%s::%s::%d" % [resource_path, script_path, ordinal],
\t\t\t\t"spawned_ordinal": ordinal,
\t\t\t\t"script_path": script_path,
\t\t\t\t"script_path_observation": "node.get_script().resource_path",
\t\t\t\t"script_sha256": source_text.sha256_text(),
\t\t\t\t"resource_path": resource_path,
\t\t\t\t"resource_path_observation": "battle.scene_file_path",
\t\t\t\t"resource_provenance": "packed_scene_state",
\t\t\t\t"source_line": _flashpatch_exported_line(source_text, "flashpatch_intensity"),
\t\t\t\t"source_line_observation": "FileAccess.get_file_as_string(node.get_script().resource_path)",
\t\t\t\t"property": "flashpatch_intensity",
\t\t\t\t"factual_value": child.get("flashpatch_intensity"),
\t\t\t\t"event_kind": "render_property",
\t\t\t})
\t\t\tif tick == 0:
\t\t\t\t_flashpatch_append_jsonl(OS.get_environment("FLASHPATCH_ACTION_EVENTS"), {
\t\t\t\t\t"frame": tick,
\t\t\t\t\t"action": "controlled_shockwave",
\t\t\t\t\t"status": "APPLIED",
\t\t\t\t\t"observation": "runtime_node_present_at_capture",
\t\t\t\t\t"node_path": str(child.get_path()),
\t\t\t\t\t"property": "flashpatch_intensity",
\t\t\t\t\t"factual_value": child.get("flashpatch_intensity"),
\t\t\t\t})
\t\t\tordinal += 1
"""
    if marker not in recorder_source:
        raise ValueError("Sparta recorder injection anchor is unavailable")
    recorder.write_text(recorder_source.replace(marker, instrumentation + marker, 1), encoding="utf-8")
    project_config = destination / "project.godot"
    project_config.write_text(
        project_config.read_text(encoding="utf-8").replace(
            'run/main_scene="res://scenes/MainMenu.tscn"',
            'run/main_scene="res://tools/demo/DemoInputRecorder.tscn"',
            1,
        ),
        encoding="utf-8",
    )
    trace = destination / "flashpatch-trace.json"
    trace.write_text(json.dumps({
        "fixed_fps": 60,
        "actions": [{"frame": 0, "action": "controlled_shockwave"}],
        "seed": "12345",
        "scenario": [
            {"team": 0, "type": "Infantry", "x": 800, "y": 430, "count": 60, "morale": 25.0},
            {"team": 1, "type": "Cavalry", "x": 740, "y": 560},
            {"team": 1, "type": "Cavalry", "x": 860, "y": 560},
        ],
        "camera": [{"tick": 0, "x": 800.0, "y": 480.0, "zoom": 1.3}],
        "steps": [],
    }, indent=2) + "\n", encoding="utf-8")
    contract = destination / "flashpatch.renderer.contract.json"
    contract.write_text(json.dumps({
        "schema": "flashpatch-godot-safety-ci-v1",
        "trace": trace.name,
        "scene": "tools/demo/DemoInputRecorder.tscn",
        "timing_field": "action_frames",
        "state_field": "gameplay_state",
        "risk_signal": {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0},
        "patch_candidates": [{
            "source": "scripts/RoutShockwave.gd",
            "parameter": "flashpatch_intensity",
            "parameter_kind": "intensity",
            "replacement": 0.0,
            "runtime_binding": "dynamic",
            "runtime_resource": "scenes/Battle.tscn",
        }],
    }, indent=2) + "\n", encoding="utf-8")
    return ControlledSparta(destination, contract, trace, shockwave, {
        "repository_url": SPARTA_REPOSITORY,
        "source_revision": SPARTA_REVISION,
        "license": SPARTA_LICENSE,
        "project_path": ".",
    })


def execute_controlled_sparta(
    upstream_project: Path,
    output_root: Path,
    *,
    godot_binary: Path,
    runner_factory: Callable[[Path], object] | None = None,
) -> ControlledSpartaRun:
    """Persist a labelled external controlled-mutation engine receipt."""
    from .safety_ci import compile_project, write_receipt

    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"controlled Sparta output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    controlled = materialize_controlled_sparta(upstream_project, output_root / "controlled-project")
    factory = runner_factory or (
        lambda project: SpartaRendererRunner(project, godot_binary=godot_binary)
    )
    receipt = compile_project(
        controlled.project,
        controlled.contract,
        workspace=output_root / "workspace",
        runner_factory=factory,
        checkpoint_path=output_root / "engine-checkpoint.json",
    )
    receipt["controlled_mutation"] = True
    receipt["upstream"] = {
        **controlled.source,
        "classification": "external_dynamic_effect_controlled_mutation",
        "upstream_defect": False,
    }
    receipt_path = output_root / "engine-receipt.json"
    write_receipt(receipt, receipt_path)
    return ControlledSpartaRun(controlled, receipt_path, receipt)


def execute_controlled_pong(
    upstream_project: Path,
    output_root: Path,
    *,
    runner_factory: Callable[..., object] | None = None,
) -> ControlledPongRun:
    """Run the labelled mutation and persist the engine's real receipt.

    The caller owns the display boundary.  In CI or a Linux worker this is
    deliberately invoked under Xvfb; without a renderer, the engine receipt is
    INCONCLUSIVE rather than a headless substitute.
    """
    from .safety_ci import compile_project, write_receipt

    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"controlled output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    controlled = materialize_controlled_pong(upstream_project, output_root / "controlled-project")
    workspace = output_root / "workspace"
    kwargs: dict[str, object] = {"workspace": workspace}
    if runner_factory is not None:
        kwargs["runner_factory"] = runner_factory
    receipt = compile_project(controlled.project, controlled.contract, **kwargs)
    receipt["controlled_mutation"] = True
    receipt["upstream"] = {
        **controlled.source,
        "classification": "upstream_adapter_absent",
        "upstream_defect": False,
    }
    artifacts = _artifact_paths(output_root, controlled, receipt)
    manifest = {
        "schema": "flashpatch-controlled-godot-artifact-manifest-v1",
        "controlled_mutation": True,
        "artifacts": [
            {"path": path.relative_to(output_root).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    manifest_path = output_root / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["artifact_manifest"] = {
        "path": manifest_path.name,
        "sha256": _sha256(manifest_path),
        "count": len(manifest["artifacts"]),
    }
    receipt_path = output_root / "engine-receipt.json"
    write_receipt(receipt, receipt_path)
    return ControlledPongRun(controlled, receipt_path, receipt)
