# FlashPatch

**Fail-closed visual QA for game development.** FlashPatch is built for game builds, play captures, cinematics, UI, and VFX. Its verified source-bound path uses a Godot replay adapter: rendered frames are linked to one permitted source edit, then replayed with the same trace to verify risk reduction without gameplay drift.

## What FlashPatch does

1. Replays a declared Godot action trace in an isolated project copy.
2. Captures renderer-owned RGB frames when the project supplies a non-headless replay adapter.
3. Detects visual-risk intervals and binds them to timestamped runtime node, property, script, and source-line evidence.
4. Tests one allowlisted exported source parameter edit at a time.
5. Accepts a patch only when the copied project lowers risk and preserves the declared gameplay invariants.

Every terminal result is a hash-bound receipt: `PASS`, `SAFE`, `FAIL`, or `INCONCLUSIVE`.

## Safety behavior

FlashPatch does not guess a replay contract, invent a runtime cause, or silently substitute a numeric smoke signal for pixel evidence. Missing renderer evidence, malformed timestamps, ambiguous source binding, multi-parameter edits, or gameplay drift produce `INCONCLUSIVE`.

The current implementation provides a source-bound Godot path. Unity and Unreal adapters are not included.

## Quick start

Requirements: Python 3.11 or later. A renderer-backed run also needs the Godot version declared by the project and a display-capable replay adapter.

```bash
python -m pip install -e ".[dev]"

export GODOT_BINARY=/path/to/Godot
flashpatch safety-demo --output artifacts/safety-demo
```

The demo writes one receipt for each terminal state. Without the declared Godot binary, renderer-backed cases close as `INCONCLUSIVE`.

To run a project that implements the FlashPatch replay contract:

```bash
flashpatch compile <project> <contract-or-trace> \
  --workspace artifacts/flashpatch-ci \
  --receipt artifacts/flashpatch-ci/receipt.json
```

## Replay contract

`flashpatch compile` accepts `flashpatch-godot-safety-ci-v1`. A project declares its trace, scene, gameplay preservation fields, and the one-parameter patch candidates that FlashPatch may test.

```json
{
  "schema": "flashpatch-godot-safety-ci-v1",
  "trace": "trace.json",
  "scene": "main.tscn",
  "timing_field": "action_frames",
  "state_field": "gameplay_state",
  "risk_signal": {"kind": "replay_observations_v1", "field": "observations", "threshold": 1.0},
  "patch_candidates": [
    {"source": "main.gd", "parameter": "burst_intensity", "replacement": 0.0}
  ]
}
```

A renderer-backed adapter may declare `frame_npz_v1` and emit uint8 RGB frames plus strictly increasing timestamps. FlashPatch hashes that artifact before analysis. A headless numeric signal remains a smoke regression and is never pixel evidence.

## Unity source preflight

For a pinned Unity fixture, FlashPatch can bind declared project files before any editor import:

```bash
flashpatch unity-preflight unity-source-manifest.json /path/to/UnityProject
```

This checks only manifest-bound source bytes. It does not import Unity, build a player, capture frames, or establish Unity compatibility.

## Capture intake

An externally produced renderer capture can be analyzed without claiming a source or engine integration:

```bash
flashpatch renderer-intake capture.npz --receipt artifacts/capture-intake.json
```

The capture must satisfy `frame_npz_v1`. Its receipt records only the supplied RGB frames, timestamps, and detector result. Engine identity, source causality, replay preservation, display output, and repair effectiveness require their own bound evidence.

## Development

```bash
python -m pytest -q \
  tests/test_standards.py tests/test_localization.py tests/test_repair.py \
  tests/test_media.py tests/test_renderer_artifact.py \
  tests/test_product_surface.py::test_installed_cli_and_ci_contract_are_executable
```

CI runs that portable regression subset on supported Python versions. A separate Ubuntu job downloads the pinned Godot 4.7.1 release, verifies its SHA-256, and runs the renderer-backed safety demo.

## License

[Apache License 2.0](LICENSE).
