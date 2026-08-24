extends Node

@export var burst_intensity: float = 1.0


func _ready() -> void:
	var arguments := OS.get_cmdline_user_args()
	var trace_index := arguments.find("--trace")
	var output_index := arguments.find("--output")
	if trace_index < 0 or output_index < 0:
		get_tree().quit(2)
		return
	var trace: Dictionary = JSON.parse_string(
		FileAccess.get_file_as_string(arguments[trace_index + 1])
	)
	var action_frames: Array[int] = []
	var observations: Array[float] = []
	var charged := false
	for action in trace["actions"]:
		action_frames.append(int(action["frame"]))
		if bool(action.get("charge", false)):
			charged = true
		var intensity := 0.0
		if charged and bool(action.get("fire", false)):
			intensity = burst_intensity
			charged = false
		observations.append(intensity)
	var output := FileAccess.open(arguments[output_index + 1], FileAccess.WRITE)
	output.store_string(JSON.stringify({
		"fixed_fps": int(trace["fixed_fps"]),
		"action_frames": action_frames,
		"gameplay_state": "%d:%s" % [action_frames.size(), str(charged)],
		"observations": observations,
		"status": "REPLAYED",
	}, "  ") + "\n")
	get_tree().quit(0)
