extends Node


func _ready() -> void:
    var arguments := OS.get_cmdline_user_args()
    var trace_path := _argument(arguments, "--trace")
    var output_path := _argument(arguments, "--output")
    if trace_path.is_empty() or output_path.is_empty():
        push_error("--trace and --output are required")
        get_tree().quit(2)
        return

    var source := FileAccess.open(trace_path, FileAccess.READ)
    if source == null:
        push_error("unable to read trace")
        get_tree().quit(3)
        return
    var trace = JSON.parse_string(source.get_as_text())
    if not trace is Dictionary or not trace.has("actions") or not trace.has("fixed_fps"):
        push_error("invalid action trace")
        get_tree().quit(4)
        return

    var position_x := 0.0
    var states: Array[Dictionary] = []
    for action in trace["actions"]:
        position_x += float(action.get("move_x", 0.0)) * 2.0
        states.append({
            "frame": int(action["frame"]),
            "position_x": position_x,
        })

    var result := {
        "fixed_fps": int(trace["fixed_fps"]),
        "states": states,
        "status": "REPLAYED",
    }
    var output := FileAccess.open(output_path, FileAccess.WRITE)
    if output == null:
        push_error("unable to write replay result")
        get_tree().quit(5)
        return
    output.store_string(JSON.stringify(result, "  ") + "\n")
    output.close()
    get_tree().quit(0)


func _argument(arguments: PackedStringArray, name: String) -> String:
    var index := arguments.find(name)
    if index < 0 or index + 1 >= arguments.size():
        return ""
    return arguments[index + 1]
