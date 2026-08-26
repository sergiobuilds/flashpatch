extends Node2D

@export var burst_intensity: float = 1.0

const VIEWPORT_SIZE := Vector2(320, 180)

var _trace: Dictionary
var _output_path := ""
var _action_frames: Array[int] = []
var _observations: Array[float] = []
var _runtime_events: Array[Dictionary] = []
var _timestamps_us: Array[int] = []
var _charged := false
var _capture_start_us := 0
var _flash_on := false


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
		_run_headless_replay()
		return
	_capture_start_us = Time.get_ticks_usec()
	_run_replay()


func _run_headless_replay() -> void:
	for action in _trace["actions"]:
		_action_frames.append(int(action["frame"]))
		if bool(action.get("charge", false)):
			_charged = true
		var intensity := 0.0
		if _charged and bool(action.get("fire", false)):
			intensity = burst_intensity
			_charged = false
		_observations.append(intensity)
	var gameplay_state := "%d:%s" % [_action_frames.size(), str(_charged)]
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"fixed_fps": int(_trace["fixed_fps"]),
		"action_frames": _action_frames,
		"gameplay_state": gameplay_state,
		"observations": _observations,
		"status": "REPLAYED",
	}, "  ") + "\n")
	get_tree().quit(0)


func _run_replay() -> void:
	var actions: Array = _trace["actions"]
	var fps := int(_trace["fixed_fps"])
	var capture_directory := _output_path.get_base_dir().path_join("renderer-capture")
	DirAccess.make_dir_recursive_absolute(capture_directory)
	for index in range(actions.size() + 8):
		var action: Dictionary = actions[index] if index < actions.size() else {"frame": index}
		var frame := int(action["frame"])
		var intensity := 0.0
		if index < actions.size():
			_action_frames.append(frame)
			if bool(action.get("charge", false)):
				_charged = true
			if _charged and bool(action.get("fire", false)):
				intensity = burst_intensity
				_charged = false
			_observations.append(intensity)
		_flash_on = intensity > 0.0 or (index >= actions.size() and index < actions.size() + 8 and index % 2 == 0)
		if burst_intensity <= 0.0:
			_flash_on = false
		queue_redraw()
		await RenderingServer.frame_post_draw
		_capture_frame(capture_directory, index)
		_runtime_events.append({
			"frame_index": index,
			"timestamp_us": _timestamps_us.back(),
			"node_path": str(get_path()),
			"resource_path": "res://main.tscn",
			"script_path": "res://main.gd",
			"source_line": 3,
			"property": "burst_intensity",
			"factual_value": burst_intensity,
			"event_kind": "render_property",
		})
	var gameplay_state := "%d:%s" % [_action_frames.size(), str(_charged)]
	_write_result(_output_path, fps, gameplay_state, capture_directory)
	get_tree().quit(0)


func _draw() -> void:
	var color := Color.WHITE if _flash_on else Color.BLACK
	draw_rect(Rect2(Vector2.ZERO, VIEWPORT_SIZE), color)


func _capture_frame(directory: String, index: int) -> void:
	var image := get_viewport().get_texture().get_image()
	image.convert(Image.FORMAT_RGB8)
	var saved := image.save_png(directory.path_join("frame_%06d.png" % index))
	if saved != OK:
		push_error("renderer frame capture failed")
		return
	var timestamp_us := Time.get_ticks_usec() - _capture_start_us
	if not _timestamps_us.is_empty():
		timestamp_us = max(timestamp_us, _timestamps_us.back() + 1)
	_timestamps_us.append(timestamp_us)


func _paths() -> PackedStringArray:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--trace")
	var output_index := arguments.find("--output")
	if trace_index < 0 or output_index < 0:
		push_error("--trace and --output are required")
		return PackedStringArray()
	return PackedStringArray([arguments[trace_index + 1], arguments[output_index + 1]])


func _write_result(path: String, fixed_fps: int, gameplay_state: String, capture_directory: String) -> void:
	var output := FileAccess.open(path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"fixed_fps": fixed_fps,
		"action_frames": _action_frames,
		"gameplay_state": gameplay_state,
		"observations": _observations,
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
			"viewport": [320, 180],
			"color_space": "sRGB/BT.709",
		},
		"status": "REPLAYED",
	}, "  ") + "\n")
