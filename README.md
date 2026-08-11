# FlashPatch

FlashPatch is a fail-closed visual QA tool for game projects. It analyzes rendered video for temporal visual-risk patterns, produces localized evidence, and can validate a narrowly scoped source-level repair against the same declared replay trace.

## Product scope

The public implementation supports the following path.

1. Read an RGB video or replay artifact with timestamp validation.
2. Detect general flash, red flash, and dynamic regular-pattern risk.
3. Localize hazardous frames and pixels.
4. Test a declared one-parameter source edit in an isolated Godot project copy.
5. Emit `PASS`, `SAFE`, `FAIL`, or `INCONCLUSIVE` receipts.

The current implementation provides a Godot adapter. Unity and Unreal adapters are not included.

## Safety behavior

Missing renderer evidence, malformed timestamps, ambiguous source binding, multi-parameter edits, or replay preservation failures result in `INCONCLUSIVE`. A headless numeric replay is a regression signal and is not treated as pixel evidence.

## Quick start

Python 3.11 or later is required. A renderer-backed run also needs the Godot version declared by the target project and a display-capable replay adapter.

```bash
python -m pip install -e ".[dev]"
flashpatch demo --output artifacts/demo
```

The demo writes a synthetic hazardous clip, its repaired output, and a receipt.

To run a project that implements the replay contract:

```bash
flashpatch compile <project> <contract-or-trace> \
  --workspace artifacts/flashpatch-ci \
  --receipt artifacts/flashpatch-ci/receipt.json
```

## Replay contract

`flashpatch compile` accepts `flashpatch-godot-safety-ci-v1`. A project declares its action trace, main scene, gameplay-preservation fields, and allowed patch candidates.

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

Renderer-backed adapters may instead emit `frame_npz_v1` with uint8 RGB frames and strictly increasing timestamps. FlashPatch binds that artifact to its analysis before accepting a repair.

## Development

```bash
python -m pytest -q \
  tests/test_standards.py tests/test_localization.py tests/test_repair.py \
  tests/test_media.py tests/test_renderer_artifact.py \
  tests/test_product_surface.py::test_installed_cli_and_ci_contract_are_executable
```

Godot-backed compilation needs a matching Godot binary and is run in an environment that supplies one.

## License

[Apache License 2.0](LICENSE).
