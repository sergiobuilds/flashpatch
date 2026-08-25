"""Prepare the repository-owned Unity Editor capture harness for L10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class UnityHarnessError(ValueError):
    """The pinned Unity fixture cannot accept the L10 harness safely."""


_EDITOR_SCRIPT = r'''using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using Cinemachine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.SceneManagement;
using UnityEngine.VFX;

public static class FlashPatchL10Capture
{
    const string ScenePath = "Assets/Samples/SmokePortal/SmokePortal.unity";
    const int Width = 640;
    const int Height = 360;
    const int FrameCount = 121;
    const float FrameStep = 1.0f / 30.0f;

    [Serializable] class RuntimeEvent
    {
        public int frame;
        public string object_identity;
        public string component;
        public string property;
        public float value;
        public float rendered_value;
    }

    [Serializable] class RuntimeEvents { public RuntimeEvent[] events; }

    [Serializable] class StateRow
    {
        public int frame;
        public float time;
        public string camera;
        public Vector3 camera_position;
        public Quaternion camera_rotation;
        public string target_object;
        public bool target_active;
    }

    [Serializable] class StateStream { public StateRow[] states; }

    [Serializable] class InputRow { public string path; public string sha256; }
    [Serializable] class InputManifest { public InputRow[] files; public string schema; }

    [Serializable] class ExecutionMarker
    {
        public string engine;
        public string engine_version;
        public int frame_count;
        public string mode;
        public string png_set_sha256;
        public string project_manifest_sha256;
        public string scene_sha256;
        public string schema;
    }

    static string output;
    static Camera camera;
    static CinemachineBrain brain;
    static LightFlicker target;
    static VisualEffect[] effects;
    static PlayableDirector[] directors;
    static RenderTexture texture;
    static Texture2D readable;
    static RenderTexture previous;
    static List<RuntimeEvent> events;
    static List<StateRow> states;
    static int frame;
    static bool waitingForStep;
    static string mode;
    static string adapterSha256;
    static string projectManifestSha256;
    static string sceneSha256;

    static string HierarchyPath(Transform value)
    {
        var rows = new List<string>();
        while (value != null)
        {
            rows.Add(value.GetSiblingIndex().ToString("D4", CultureInfo.InvariantCulture) + ":" + value.name);
            value = value.parent;
        }
        rows.Reverse();
        return string.Join("/", rows);
    }

    static uint StableSeed(string domain, Transform value)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(domain + "\n" + HierarchyPath(value));
        using (var sha = System.Security.Cryptography.SHA256.Create())
        {
            var digest = sha.ComputeHash(bytes);
            return BitConverter.ToUInt32(digest, 0);
        }
    }

    static T GetPrivate<T>(object value, string name)
    {
        var field = value.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field == null) throw new InvalidOperationException("deterministic field missing: " + name);
        return (T)field.GetValue(value);
    }

    static void SetPrivate<T>(object value, string name, T observed)
    {
        var field = value.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field == null) throw new InvalidOperationException("deterministic field missing: " + name);
        field.SetValue(value, observed);
    }

    static void ConfigureDeterministicReplay()
    {
        var flickers = UnityEngine.Object.FindObjectsOfType<LightFlicker>(true)
            .OrderBy(x => HierarchyPath(x.transform), StringComparer.Ordinal).ToArray();
        foreach (var flicker in flickers)
        {
            var seed = StableSeed("LightFlicker", flicker.transform);
            var initialPosition = GetPrivate<Vector3>(flicker, "m_InitialPosition");
            var initialRotation = GetPrivate<Quaternion>(flicker, "m_InitialRotation");
            var initialIntensity = GetPrivate<float>(flicker, "m_InitialIntensity");
            flicker.transform.SetPositionAndRotation(initialPosition, initialRotation);
            flicker.GetComponent<Light>().intensity = initialIntensity;
            SetPrivate(flicker, "m_XSeed", (seed & 0xff) / 255.0f * 248.0f);
            SetPrivate(flicker, "m_YSeed", ((seed >> 8) & 0xff) / 255.0f * 248.0f);
            SetPrivate(flicker, "m_ZSeed", ((seed >> 16) & 0xff) / 255.0f * 248.0f);
        }
        effects = UnityEngine.Object.FindObjectsOfType<VisualEffect>(true)
            .OrderBy(x => HierarchyPath(x.transform), StringComparer.Ordinal).ToArray();
        foreach (var effect in effects)
        {
            effect.resetSeedOnPlay = false;
            effect.startSeed = StableSeed("VisualEffect", effect.transform);
            effect.pause = true;
            effect.Reinit();
        }
        directors = UnityEngine.Object.FindObjectsOfType<PlayableDirector>(true)
            .OrderBy(x => HierarchyPath(x.transform), StringComparer.Ordinal).ToArray();
        foreach (var director in directors)
        {
            director.timeUpdateMode = DirectorUpdateMode.Manual;
            director.time = 0.0;
            director.Evaluate();
        }
        brain = camera.GetComponent<CinemachineBrain>();
        if (brain == null) throw new InvalidOperationException("CinemachineBrain missing");
        var hdCamera = camera.GetComponent<HDAdditionalCameraData>();
        if (hdCamera == null) throw new InvalidOperationException("HDAdditionalCameraData missing");
        hdCamera.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
        brain.m_UpdateMethod = CinemachineBrain.UpdateMethod.ManualUpdate;
        brain.m_BlendUpdateMethod = CinemachineBrain.BrainUpdateMethod.LateUpdate;
        CinemachineCore.UniformDeltaTimeOverride = FrameStep;
        CinemachineCore.CurrentTimeOverride = 0.0f;
        foreach (var virtualCamera in UnityEngine.Object.FindObjectsOfType<CinemachineVirtualCameraBase>(true))
            virtualCamera.PreviousStateIsValid = false;
    }

    static string Sha256(byte[] value)
    {
        using (var sha = System.Security.Cryptography.SHA256.Create())
            return BitConverter.ToString(sha.ComputeHash(value)).Replace("-", "").ToLowerInvariant();
    }

    static string Arg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++) if (args[i] == name) return args[i + 1];
        throw new InvalidOperationException("missing argument " + name);
    }

    static void WriteJson(string path, string value)
    {
        File.WriteAllText(path, value.Replace("\\/", "/") + "\n");
    }

    static void VerifyProjectInputs(string projectRoot)
    {
        var manifestPath = Path.GetFullPath(Arg("-flashpatchInputManifest"));
        projectManifestSha256 = Sha256(File.ReadAllBytes(manifestPath));
        if (projectManifestSha256 != Arg("-flashpatchExpectedInputManifestSha256"))
            throw new InvalidOperationException("input manifest hash mismatch");
        var manifest = JsonUtility.FromJson<InputManifest>(File.ReadAllText(manifestPath));
        if (manifest == null || manifest.schema != "flashpatch-l10-unity-project-inputs-v1" || manifest.files == null)
            throw new InvalidOperationException("input manifest schema mismatch");
        var declared = new HashSet<string>(manifest.files.Select(x => x.path), StringComparer.Ordinal);
        if (declared.Count != manifest.files.Length) throw new InvalidOperationException("duplicate input path");
        foreach (var row in manifest.files)
        {
            var absolute = Path.GetFullPath(Path.Combine(projectRoot, row.path));
            if (!absolute.StartsWith(projectRoot + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                throw new InvalidOperationException("input path escapes project");
            if (!File.Exists(absolute) || Sha256(File.ReadAllBytes(absolute)) != row.sha256)
                throw new InvalidOperationException("project input hash mismatch: " + row.path);
        }
        var observed = new HashSet<string>(new[] { "Assets", "Packages", "ProjectSettings" }
            .SelectMany(folder => Directory.GetFiles(Path.Combine(projectRoot, folder), "*", SearchOption.AllDirectories))
            .Select(path => path.Substring(projectRoot.Length + 1).Replace('\\', '/')), StringComparer.Ordinal);
        if (!observed.SetEquals(declared)) throw new InvalidOperationException("project input file set mismatch");
    }

    public static void Run()
    {
        output = Path.GetFullPath(Arg("-flashpatchOutput"));
        mode = Arg("-flashpatchMode");
        var expected = Arg("-flashpatchExpectedSceneSha256");
        if (Directory.Exists(output)) throw new InvalidOperationException("output already exists");
        Directory.CreateDirectory(output);
        var projectRoot = Path.GetFullPath(Arg("-projectPath")).TrimEnd(Path.DirectorySeparatorChar);
        VerifyProjectInputs(projectRoot);
        adapterSha256 = Sha256(File.ReadAllBytes(Path.Combine(projectRoot, "Assets/Editor/FlashPatchL10Capture.cs")));
        var sceneAbsolute = Path.Combine(projectRoot, ScenePath);
        using (var sha = System.Security.Cryptography.SHA256.Create())
        using (var stream = File.OpenRead(sceneAbsolute))
        {
            sceneSha256 = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            if (sceneSha256 != expected) throw new InvalidOperationException("scene hash mismatch");
        }

        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        if (mode != "factual" && mode != "counterfactual") throw new InvalidOperationException("invalid capture mode");
        var expectedValue = mode == "factual" ? 2000.0f : 0.0f;
        target = scene.GetRootGameObjects().SelectMany(x => x.GetComponentsInChildren<LightFlicker>(true))
            .Single(x => x.gameObject.name == "Spot Light" && Mathf.Approximately(x.m_IntensityJitterScale, expectedValue));

        camera = Camera.main ?? UnityEngine.Object.FindObjectOfType<Camera>();
        if (camera == null) throw new InvalidOperationException("camera missing");
        texture = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        readable = new Texture2D(Width, Height, TextureFormat.RGB24, false, false);
        previous = RenderTexture.active;
        events = new List<RuntimeEvent>();
        states = new List<StateRow>();
        frame = 0;
        waitingForStep = false;
        Time.captureFramerate = 30;
        camera.targetTexture = texture;
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
        EditorApplication.isPlaying = true;
    }

    static void OnPlayModeChanged(PlayModeStateChange state)
    {
        if (state != PlayModeStateChange.EnteredPlayMode) return;
        target = UnityEngine.Object.FindObjectsOfType<LightFlicker>(true)
            .Single(x => x.gameObject.name == "Spot Light" && Mathf.Approximately(x.m_IntensityJitterScale, mode == "factual" ? 2000.0f : 0.0f));
        camera = Camera.main ?? UnityEngine.Object.FindObjectOfType<Camera>();
        camera.targetTexture = texture;
        ConfigureDeterministicReplay();
        EditorApplication.isPaused = true;
        EditorApplication.update += PumpReplay;
    }

    static void PumpReplay()
    {
        try
        {
            if (frame >= FrameCount) { Finish(); return; }
            if (frame > 0 && !waitingForStep)
            {
                waitingForStep = true;
                EditorApplication.Step();
                return;
            }
            if (waitingForStep)
            {
                waitingForStep = false;
                foreach (var effect in effects) effect.Simulate(FrameStep, 1);
            }
            foreach (var director in directors)
            {
                director.time = frame * FrameStep;
                director.Evaluate();
            }
            CinemachineCore.CurrentTimeOverride = frame * FrameStep;
            brain.ManualUpdate();
            camera.Render();
            RenderTexture.active = texture;
            readable.ReadPixels(new Rect(0, 0, Width, Height), 0, 0, false);
            readable.Apply(false, false);
            File.WriteAllBytes(Path.Combine(output, frame.ToString("D4") + ".png"), readable.EncodeToPNG());
            var light = target.GetComponent<Light>();
            events.Add(new RuntimeEvent { frame = frame, object_identity = target.gameObject.name,
                component = typeof(LightFlicker).FullName, property = "m_IntensityJitterScale",
                value = target.m_IntensityJitterScale, rendered_value = light.intensity });
            states.Add(new StateRow { frame = frame, time = frame * FrameStep, camera = camera.name,
                camera_position = camera.transform.position, camera_rotation = camera.transform.rotation,
                target_object = target.gameObject.name, target_active = target.gameObject.activeInHierarchy });
            frame++;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            Cleanup();
            EditorApplication.Exit(1);
        }
    }

    static void Finish()
    {
        var runtimeEventsPath = Path.Combine(output, "runtime-events.json");
        var stateStreamPath = Path.Combine(output, "state-stream.json");
        WriteJson(runtimeEventsPath, JsonUtility.ToJson(new RuntimeEvents { events = events.ToArray() }, true));
        WriteJson(stateStreamPath, JsonUtility.ToJson(new StateStream { states = states.ToArray() }, true));
        WriteJson(Path.Combine(output, "engine-identity.json"), "{\"engine\":\"Unity\",\"version\":\"" + Application.unityVersion + "\"}");
        File.WriteAllLines(Path.Combine(output, "timestamps.txt"), states.Select(x => x.time.ToString("R", CultureInfo.InvariantCulture)));
        var pngHashes = Enumerable.Range(0, FrameCount)
            .Select(x => Sha256(File.ReadAllBytes(Path.Combine(output, x.ToString("D4") + ".png"))));
        var pngSet = Sha256(System.Text.Encoding.ASCII.GetBytes(string.Join("\n", pngHashes)));
        var markerJson = "{\n" +
            "  \"adapter_sha256\": \"" + adapterSha256 + "\",\n" +
            "  \"engine\": \"Unity\",\n" +
            "  \"engine_version\": \"" + Application.unityVersion + "\",\n" +
            "  \"frame_count\": " + FrameCount + ",\n" +
            "  \"graphics_device_id\": " + SystemInfo.graphicsDeviceID + ",\n" +
            "  \"graphics_device_name\": \"" + SystemInfo.graphicsDeviceName + "\",\n" +
            "  \"graphics_device_type\": \"" + SystemInfo.graphicsDeviceType + "\",\n" +
            "  \"graphics_device_vendor\": \"" + SystemInfo.graphicsDeviceVendor + "\",\n" +
            "  \"mode\": \"" + mode + "\",\n" +
            "  \"png_set_sha256\": \"" + pngSet + "\",\n" +
            "  \"project_manifest_sha256\": \"" + projectManifestSha256 + "\",\n" +
            "  \"replay_profile\": \"deterministic-step-v2\",\n" +
            "  \"runtime_events_sha256\": \"" + Sha256(File.ReadAllBytes(runtimeEventsPath)) + "\",\n" +
            "  \"scene_sha256\": \"" + sceneSha256 + "\",\n" +
            "  \"schema\": \"flashpatch-l10-unity-execution-marker-v2\",\n" +
            "  \"state_stream_sha256\": \"" + Sha256(File.ReadAllBytes(stateStreamPath)) + "\"\n" +
            "}";
        WriteJson(Path.Combine(output, "execution-marker.json"), markerJson);
        Cleanup();
        EditorApplication.Exit(0);
    }

    static void Cleanup()
    {
        EditorApplication.update -= PumpReplay;
        EditorApplication.playModeStateChanged -= OnPlayModeChanged;
        CinemachineCore.UniformDeltaTimeOverride = -1.0f;
        CinemachineCore.CurrentTimeOverride = -1.0f;
        if (camera != null) camera.targetTexture = null;
        RenderTexture.active = previous;
        if (readable != null) UnityEngine.Object.DestroyImmediate(readable);
        if (texture != null) UnityEngine.Object.DestroyImmediate(texture);
        if (EditorApplication.isPlaying) EditorApplication.isPlaying = false;
    }
}
'''

_EDITOR_META = (
    "fileFormatVersion: 2\n"
    "guid: f1a5c4f9dff44ddda4552e5345a10c10\n"
    "MonoImporter:\n"
    "  externalObjects: {}\n"
    "  serializedVersion: 2\n"
    "  defaultReferences: []\n"
    "  executionOrder: 0\n"
    "  icon: {instanceID: 0}\n"
    "  userData: \n"
    "  assetBundleName: \n"
    "  assetBundleVariant: \n"
)

_EDITOR_FOLDER_META = (
    "fileFormatVersion: 2\n"
    "guid: 32498e40995c5b4199bcdbe112854fb9\n"
    "folderAsset: yes\n"
    "DefaultImporter:\n"
    "  externalObjects: {}\n"
    "  userData: \n"
    "  assetBundleName: \n"
    "  assetBundleVariant: \n"
)

_LINUX_TOOLCHAIN_PACKAGE = "com.unity.toolchain.linux-x86_64"
_LINUX_TOOLCHAIN_VERSION = "2.0.11"
_PACKAGES_LOCK = Path(__file__).with_name("_unity_packages_lock_2022_3_8f1.json").read_bytes()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unity_adapter_fingerprints() -> dict[str, str]:
    """Return the code-owned adapter bytes approved for natural evidence."""
    return {
        "Assets/Editor/FlashPatchL10Capture.cs": hashlib.sha256(
            _EDITOR_SCRIPT.encode()
        ).hexdigest(),
        "Assets/Editor/FlashPatchL10Capture.cs.meta": hashlib.sha256(
            _EDITOR_META.encode()
        ).hexdigest(),
        "Assets/Editor.meta": hashlib.sha256(_EDITOR_FOLDER_META.encode()).hexdigest(),
        "Packages/packages-lock.json": hashlib.sha256(_PACKAGES_LOCK).hexdigest(),
    }


def unity_linux_package_manifest_bytes(source: bytes) -> bytes:
    """Return the one approved package-manifest transform for the Linux image."""
    try:
        value = json.loads(source)
        dependencies = value["dependencies"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise UnityHarnessError("pinned Unity package manifest is invalid") from exc
    if not isinstance(dependencies, dict) or _LINUX_TOOLCHAIN_PACKAGE in dependencies:
        raise UnityHarnessError("pinned Unity package manifest is unexpected")
    if dependencies.get("com.unity.toolchain.win-x86_64-linux-x86_64") != "2.0.4":
        raise UnityHarnessError("pinned Unity package toolchain changed")
    dependencies[_LINUX_TOOLCHAIN_PACKAGE] = _LINUX_TOOLCHAIN_VERSION
    return (json.dumps(value, indent=2) + "\n").encode()


def prepare_unity_linux_execution_inputs(project: Path) -> dict[str, object]:
    """Pin image-required Linux inputs in an isolated project copy."""
    root = project.resolve()
    if any((root / name).exists() for name in ("Library", "Temp", "Logs")):
        raise UnityHarnessError("Unity execution copy contains stale generated state")
    manifest = root / "Packages/manifest.json"
    editor_meta = root / "Assets/Editor.meta"
    packages_lock = root / "Packages/packages-lock.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise UnityHarnessError("pinned Unity package manifest is missing or unsafe")
    if editor_meta.exists() or editor_meta.is_symlink() or packages_lock.exists() or packages_lock.is_symlink():
        raise UnityHarnessError("Unity Linux execution inputs already exist")
    manifest_bytes = unity_linux_package_manifest_bytes(manifest.read_bytes())
    manifest.write_bytes(manifest_bytes)
    editor_meta.write_text(_EDITOR_FOLDER_META, encoding="utf-8")
    packages_lock.write_bytes(_PACKAGES_LOCK)
    return {
        "schema": "flashpatch-l10-unity-linux-inputs-v1",
        "package_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "packages_lock_sha256": hashlib.sha256(_PACKAGES_LOCK).hexdigest(),
        "editor_folder_meta_sha256": hashlib.sha256(_EDITOR_FOLDER_META.encode()).hexdigest(),
        "linux_toolchain": f"{_LINUX_TOOLCHAIN_PACKAGE}@{_LINUX_TOOLCHAIN_VERSION}",
        "source_files_modified": 0,
        "execution_copy_files_modified": 1,
        "execution_copy_files_added": 2,
    }


def write_unity_project_manifest(project: Path, output: Path) -> dict[str, object]:
    """Freeze every Unity-authored input file and reject symlinks or aliases."""
    root = project.resolve()
    rows = []
    for folder in ("Assets", "Packages", "ProjectSettings"):
        base = root / folder
        if base.is_symlink() or not base.is_dir():
            raise UnityHarnessError(f"Unity project input root is unsafe: {folder}")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise UnityHarnessError("Unity project input contains a symlink")
            if path.is_file():
                rows.append({
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                })
    value = {"files": rows, "schema": "flashpatch-l10-unity-project-inputs-v1"}
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise UnityHarnessError("Unity input manifest output already exists")
    output.write_bytes(raw)
    return {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest(), "file_count": len(rows)}


def install_unity_harness(
    project: Path,
    output: Path,
    *,
    role: str = "factual",
) -> dict[str, object]:
    """Install only the adapter script after checking the frozen SmokePortal cause."""
    root = project.resolve()
    scene = root / "Assets/Samples/SmokePortal/SmokePortal.unity"
    script = root / "Assets/Samples/SmokePortal/SmokePortal/LightFlicker.cs"
    if not scene.is_file() or not script.is_file() or scene.is_symlink() or script.is_symlink():
        raise UnityHarnessError("pinned SmokePortal source is missing or unsafe")
    if role not in {"factual", "counterfactual"}:
        raise UnityHarnessError("Unity harness role is invalid")
    expected_scene_value = "2000" if role == "factual" else "0"
    scene_text = scene.read_text(encoding="utf-8")
    if scene_text.count(f"m_IntensityJitterScale: {expected_scene_value}") != 1:
        raise UnityHarnessError("pinned Unity causal property is missing or ambiguous")
    if "m_Light.intensity = m_InitialIntensity + Noise.x * m_IntensityJitterScale;" not in script.read_text(encoding="utf-8"):
        raise UnityHarnessError("pinned Unity runtime attribution source is missing")
    destination = root / "Assets/Editor/FlashPatchL10Capture.cs"
    if destination.exists():
        raise UnityHarnessError("Unity harness destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_EDITOR_SCRIPT, encoding="utf-8")
    meta = destination.with_suffix(destination.suffix + ".meta")
    meta.write_text(_EDITOR_META, encoding="utf-8")
    receipt = {
        "schema": "flashpatch-l10-unity-harness-install-v1",
        "adapter_path": str(destination.relative_to(root)),
        "adapter_sha256": _sha256(destination),
        "scene_path": str(scene.relative_to(root)),
        "scene_sha256": _sha256(scene),
        "attribution_source_path": str(script.relative_to(root)),
        "attribution_source_sha256": _sha256(script),
        "source_files_modified": 0,
        "adapter_files_added": 1,
        "adapter_meta_sha256": _sha256(meta),
        "factual_value": 2000.0,
        "counterfactual_value": 0.0,
        "role": role,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return receipt
