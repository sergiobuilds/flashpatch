"""Prepare an isolated Unreal 5.6 controlled renderer adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class UnrealHarnessError(ValueError):
    """The Unreal fixture or adapter cannot be prepared safely."""


_CAPTURE_SCRIPT = r'''import hashlib
import json
import os
import unreal

WIDTH = 640
HEIGHT = 360
FRAME_COUNT = 18
FRAME_STEP = 1.0 / 30.0
BASE_INTENSITY = 0.0
FLASH_INTENSITY = 1000000.0

output = os.path.abspath(os.environ["FLASHPATCH_OUTPUT"])
mode = os.environ["FLASHPATCH_MODE"]
if os.path.exists(output):
    raise RuntimeError("output already exists")
os.makedirs(output)
if mode == "counterfactual":
    FLASH_INTENSITY = 0.0
elif mode != "factual":
    raise RuntimeError("invalid capture mode")
engine_version = unreal.SystemLibrary.get_engine_version()
unreal.log("FLASHPATCH_L10_START " + json.dumps({"engine": "Unreal", "engine_version": engine_version, "frame_count": FRAME_COUNT, "mode": mode}, separators=(",", ":"), sort_keys=True))

editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = editor.get_editor_world()
cube = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
cube.set_actor_label("FlashPatchTarget")
cube_component = cube.get_component_by_class(unreal.StaticMeshComponent)
cube_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube.Cube"))
cube.set_actor_scale3d(unreal.Vector(4.0, 4.0, 4.0))
light = actors.spawn_actor_from_class(unreal.PointLight, unreal.Vector(-400, 0, 100))
light.set_actor_label("FlashPatchPointLight")
light_component = light.get_component_by_class(unreal.PointLightComponent)
light_component.set_attenuation_radius(5000.0)
capture = actors.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(-600, 0, 100), unreal.Rotator(0, 0, 0))
capture.set_actor_label("FlashPatchCapture")
capture_component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
target = unreal.RenderingLibrary.create_render_target2d(
    world,
    WIDTH,
    HEIGHT,
    unreal.TextureRenderTargetFormat.RTF_RGBA8,
)
capture_component.texture_target = target
capture_component.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR

events = []
states = []
timestamps = []
for frame in range(FRAME_COUNT):
    value = BASE_INTENSITY + (FLASH_INTENSITY if frame % 2 else 0.0)
    light_component.set_intensity(value)
    capture_component.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, output, "%04d.png" % frame)
    events.append({"frame": frame, "object_identity": "FlashPatchPointLight", "component": "PointLightComponent", "property": "intensity", "value": value})
    states.append({"frame": frame, "target_object": "FlashPatchTarget", "target_location": [0.0, 0.0, 0.0], "capture_location": [-600.0, 0.0, 100.0], "terminal_state": "captured"})
    timestamps.append(frame * FRAME_STEP)

with open(os.path.join(output, "runtime-events.json"), "w", encoding="utf-8") as stream:
    json.dump(events, stream, indent=2, sort_keys=True)
    stream.write("\n")
with open(os.path.join(output, "state-stream.json"), "w", encoding="utf-8") as stream:
    json.dump(states, stream, indent=2, sort_keys=True)
    stream.write("\n")
with open(os.path.join(output, "timestamps.txt"), "w", encoding="utf-8") as stream:
    stream.write("\n".join(format(value, ".17g") for value in timestamps) + "\n")
with open(os.path.join(output, "engine-identity.json"), "w", encoding="utf-8") as stream:
    json.dump({"engine": "Unreal", "version": engine_version}, stream, separators=(",", ":"))
    stream.write("\n")
png_hashes = []
for frame in range(FRAME_COUNT):
    with open(os.path.join(output, "%04d.png" % frame), "rb") as stream:
        png_hashes.append(hashlib.sha256(stream.read()).hexdigest())
png_set_sha256 = hashlib.sha256("\n".join(png_hashes).encode("ascii")).hexdigest()
completion = {"schema": "flashpatch-l10-unreal-execution-marker-v1", "engine": "Unreal", "engine_version": engine_version, "frame_count": FRAME_COUNT, "mode": mode, "png_set_sha256": png_set_sha256}
with open(os.path.join(output, "execution-marker.json"), "w", encoding="utf-8") as stream:
    json.dump(completion, stream, indent=2, sort_keys=True)
    stream.write("\n")
unreal.log("FLASHPATCH_L10_COMPLETE " + json.dumps(completion, separators=(",", ":"), sort_keys=True))
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_unreal_harness(project: Path, output: Path) -> dict[str, object]:
    """Install an adapter script without rewriting the pinned map or game source."""
    root = project.resolve()
    uproject = root / "Starter.uproject"
    scene = root / "Content/Maps/Blank.umap"
    if any(path.is_symlink() or not path.is_file() for path in (uproject, scene)):
        raise UnrealHarnessError("pinned Unreal fixture is missing or unsafe")
    try:
        project_value = json.loads(uproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnrealHarnessError("pinned Unreal project descriptor is unreadable") from exc
    if project_value.get("EngineAssociation") != "5.6":
        raise UnrealHarnessError("pinned Unreal engine association changed")
    destination = root / "FlashPatchL10Capture.py"
    if destination.exists():
        raise UnrealHarnessError("Unreal harness destination already exists")
    destination.write_text(_CAPTURE_SCRIPT, encoding="utf-8")
    receipt = {
        "schema": "flashpatch-l10-unreal-harness-install-v1",
        "adapter_path": destination.name,
        "adapter_sha256": _sha256(destination),
        "scene_path": str(scene.relative_to(root)),
        "scene_sha256": _sha256(scene),
        "uproject_path": uproject.name,
        "uproject_sha256": _sha256(uproject),
        "source_files_modified": 0,
        "adapter_files_added": 1,
        "factual_flash_intensity": 1000000.0,
        "counterfactual_flash_intensity": 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
