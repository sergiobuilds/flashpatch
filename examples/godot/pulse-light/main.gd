extends Node2D

@export var flash_interval_frames: int = 1

const VIEWPORT_SIZE := Vector2(320, 180)

var _trace: Dictionary
var _output_path := ""
var _action_frames: Array[int] = []
var _observations: Array[float] = []
var _runtime_events: Array[Dictionary] = []
var _timestamps_us: Array[int] = []
var _capture_start_us := 0
var _pulse_on := false


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
	_run_renderer_replay()


func _run_headless_replay() -> void:
	for action in _trace["actions"]:
		var frame := int(action["frame"])
		_action_frames.append(frame)
		_observations.append(float((frame / max(flash_interval_frames, 1)) % 2))
	var gameplay_state := "%d:stable" % _action_frames.size()
	var output := FileAccess.open(_output_path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"fixed_fps": int(_trace["fixed_fps"]),
		"action_frames": _action_frames,
		"gameplay_state": gameplay_state,
		"observations": _observations,
		"status": "REPLAYED",
	}, "  ") + "\n")
	get_tree().quit(0)


func _run_renderer_replay() -> void:
	var actions: Array = _trace["actions"]
	var capture_directory := _output_path.get_base_dir().path_join("renderer-capture")
	DirAccess.make_dir_recursive_absolute(capture_directory)
	for index in range(actions.size() + 8):
		if index < actions.size():
			_action_frames.append(int(actions[index]["frame"]))
			_observations.append(float((index / max(flash_interval_frames, 1)) % 2))
		_pulse_on = (index / max(flash_interval_frames, 1)) % 2 == 1
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
			"property": "flash_interval_frames",
			"factual_value": flash_interval_frames,
			"event_kind": "render_property",
		})
	var gameplay_state := "%d:stable" % _action_frames.size()
	_write_result(_output_path, gameplay_state, capture_directory)
	get_tree().quit(0)


func _draw() -> void:
	var color := Color.WHITE if _pulse_on else Color.BLACK
	draw_rect(Rect2(Vector2.ZERO, VIEWPORT_SIZE), color)


func _capture_frame(directory: String, index: int) -> void:
	var image := get_viewport().get_texture().get_image()
	image.convert(Image.FORMAT_RGB8)
	if image.save_png(directory.path_join("frame_%06d.png" % index)) != OK:
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


func _write_result(path: String, gameplay_state: String, capture_directory: String) -> void:
	var output := FileAccess.open(path, FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"fixed_fps": int(_trace["fixed_fps"]),
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
