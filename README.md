# FlashPatch

**FlashPatch finds a dangerous flash in a running Godot scene, traces it to one source parameter, tests an allowed edit, and replays the same input before it reports `PASS`.**

![Actual Godot renderer capture before and after the verified patch](benchmarks/godot-demo/comparison.png)

| Actual Godot 4.7.1 run | Result |
|---|---:|
| Hazard verdict | **FAIL → PASS** |
| Risk before → after | **5.0 → 0.0** |
| Source edit | `main.gd:3`, `burst_intensity 1.0 → 0.0` |
| Source assignments changed | **1** |
| Same action trace | **Yes** |
| Timing, gameplay state, semantic invariants preserved | **Yes** |
| Original project modified | **No** |

Inspect the [patch](benchmarks/godot-demo/patch.diff), [public receipt](benchmarks/godot-demo/receipt.json), or verify every artifact hash locally:

```bash
flashpatch verify-godot-demo benchmarks/godot-demo/receipt.json
```

Contents: [한국어 요약](#1-한국어-요약) · [Product comparison](#2-product-comparison) · [Run the Godot proof](#3-run-the-godot-proof) · [Verdicts](#4-verdicts) · [How it works](#5-how-it-works) · [Evidence](#6-measured-evidence) · [Scope](#7-engine-scope-and-limitations) · [Development](#9-development-and-release-checks)

## 1 한국어 요약

기존 번쩍임 검사 도구는 대개 위험한 영상 구간을 알려 주는 데서 끝납니다. 개발자는 다시 게임으로 돌아가 원인 코드를 찾고, 값을 고친 뒤, 그 수정이 플레이를 망가뜨리지 않았는지 별도로 확인해야 합니다.

FlashPatch의 검증된 Godot 경로는 이 과정을 하나로 묶습니다.

1. 선언된 조작을 실제 Godot 장면에서 재생하고 렌더러가 그린 RGB 프레임을 검사합니다.
2. 위험 순간의 런타임 노드·속성·스크립트·소스 줄을 연결합니다.
3. 프로젝트가 미리 허용한 소스 변수 하나만 격리된 복사본에서 바꿉니다.
4. 같은 조작을 다시 재생해 위험 제거와 게임 상태 보존을 함께 확인합니다.
5. 입력, 수정, 결과물의 해시가 결박된 영수증을 남깁니다.

따라서 FlashPatch가 더 나은 지점은 단순히 “번쩍임을 더 잘 찾는다”가 아닙니다. **화면의 문제를 실제 게임 소스 수정과 동일 플레이 회귀 검증까지 끝내는 범위**가 더 넓습니다. 증거가 없거나 원인이 여러 값에 걸치면 추측하지 않습니다.

## 2 Product comparison

The tools below solve related but different parts of visual accessibility QA. A check means the capability is present in the tool's public product path; a dash means it is not the stated public workflow. This is a capability comparison, not a claim that FlashPatch has the best detector.

| Tool | Primary job | Detect flashing | Modify output | Link rendered event to game source | Replay same game input | Check gameplay state | Hash-bound verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **FlashPatch** | **Godot source repair with replay validation** | ✓ | **One allowed source parameter** | **✓** | **✓** | **✓** | **✓** |
| [Ubisoft Chroma](https://github.com/ubisoft/Chroma) | Real-time color-blindness simulation | Different target | — | — | — | — | — |
| [EA IRIS](https://github.com/electronicarts/IRIS) | Photosensitivity analysis of video | ✓ | — | — | — | — | — |
| [TooFlashy](https://github.com/hashb/TooFlashy) / [EPI-LENS](https://github.com/Pi-0r-Tau/EPI-LENS) | Video detection and reporting | ✓ | — | — | — | — | — |
| [FFmpeg photosensitivity filter](https://ffmpeg.org/ffmpeg-filters.html#photosensitivity) | Video filtering | ✓ | Video | — | — | — | — |
| [Kaya PSE detection/correction](https://github.com/samfatu/pse-detection-correction) | Video detection and correction | ✓ | Video | — | — | — | — |
| [Apple VideoFlashingReduction](https://github.com/apple/VideoFlashingReduction) | Playback-time flashing reduction | ✓ | Displayed video | — | — | — | — |
| [Unflash](https://github.com/Kelardry/unflash-video) | Video detection, correction, and recheck | ✓ | Video | — | — | — | Video record |

Ubisoft Chroma is an important accessibility tool, but it targets color-vision simulation rather than photosensitive-seizure risk. The direct difference is the workflow endpoint: Chroma helps a developer see a visual accessibility issue; FlashPatch's Godot path tests a source-level correction and proves the same gameplay still completes.

## 3 Run the Godot proof

Requirements: Python 3.11 or later and Godot 4. On headless Linux, `xvfb-run` provides the display used by the real renderer.

```bash
git clone https://github.com/sergiobuilds/flashpatch-public.git
cd flashpatch-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .

export GODOT_BINARY=/path/to/godot
xvfb-run -a flashpatch godot-demo
flashpatch verify-godot-demo artifacts/godot-demo/receipt.json
```

On a desktop session, run `flashpatch godot-demo` without `xvfb-run`. The command uses the included `interaction-burst` Godot project and performs the full render → localize → patch → replay → verify loop. It writes `before.png`, `after.png`, `comparison.png`, `patch.diff`, the detailed engine receipt, and the independently checkable public receipt.

For a fast engine-free tour of all four terminal states:

```bash
flashpatch safety-demo --output artifacts/safety-demo
```

`safety-demo` is a deterministic contract fixture for learning the verdict flow. It also writes `failure-matrix.json` and separate receipts for residual risk, gameplay-state drift, multi-parameter causality, missing renderer timestamps, and an already-safe scene. It is not renderer evidence. The checked-in images and Godot receipt above come from actual rendered frames.

## 4 Verdicts

| Receipt | Meaning |
|---|---|
| `PASS` | One declared source edit removed the measured risk and preserved timing, gameplay state, and semantic invariants. |
| `SAFE` | The actual replay was already below the declared threshold. |
| `FAIL` | Risk remained after every allowed single edit, or an edit removed risk but broke a declared gameplay invariant. |
| `INCONCLUSIVE` | Required evidence was missing, malformed, ambiguous, or would require an unauthorized multi-parameter edit. |

Fail-closed behavior is deliberate. Missing frames or timestamps, ambiguous source binding, and multi-parameter causality never become a guessed `PASS`.

| Reversal fixture | Required result |
|---|---|
| Risk remains after the allowed edit | `FAIL` |
| Risk disappears but gameplay state changes | `FAIL` |
| Two parameters are required | `INCONCLUSIVE` |
| Renderer timestamps are missing | `INCONCLUSIVE` |
| Scene starts below threshold | `SAFE` |

## 5 How it works

```text
action trace + Godot project
            │
            ▼
  isolated factual replay ──► renderer RGB frames ──► hazard interval
            │                                           │
            │                                           ▼
            └──────── runtime event ──► node / property / script / line
                                                        │
                                                        ▼
                                      one allowlisted source edit
                                                        │
                                                        ▼
                                        same-trace counterfactual replay
                                                        │
                                                        ▼
                                  risk removed + game preserved + receipt
```

The original project is hashed before and after the run. Edits occur only inside a temporary sealed copy. A candidate is accepted only if the detector passes on renderer-owned pixels and the declared replay invariants match the factual run.

The contract is explicit about what FlashPatch may change:

```json
{
  "schema": "flashpatch-godot-safety-ci-v1",
  "trace": "trace.json",
  "scene": "main.tscn",
  "timing_field": "action_frames",
  "state_field": "gameplay_state",
  "risk_signal": {"kind": "frame_npz_v1", "field": "frames_npz", "threshold": 1.0},
  "patch_candidates": [
    {"source": "main.gd", "parameter": "burst_intensity", "replacement": 0.0}
  ]
}
```

Two redistributable Godot fixtures, traces, contracts, and ground truth live in [`benchmarks/aigame-psebench/corpus/`](benchmarks/aigame-psebench/corpus/).

## 6 Measured evidence

### 6.1 Source-bound Godot result

The hero result is a real X11/OpenGL Godot 4.7.1 capture of 12 frames. FlashPatch localized `/root/InteractionBurst` to `main.gd:3`, changed one declared assignment in the workspace copy, reduced maximum measured risk from 5.0 to 0.0, and preserved the same trace, timing, gameplay state, and semantic invariants. The [engine receipt](benchmarks/godot-demo/engine-receipt.json) retains the detailed run evidence.

### 6.2 Same-input video baselines

The checked-in direct-baseline result uses three sealed cases and keeps detection and repair comparisons separate.

| Measurement | FlashPatch | Comparator | Scope |
|---|---:|---:|---|
| Detection accuracy | 3/3 | EA IRIS 3/3; EPI-LENS 1/3 | Three sealed video cases |
| Residual hazard after repair | 0% | FFmpeg 0% | Same three cases |
| Fraction of pixels changed | **25.00%** | FFmpeg 45.83% | Same three cases |
| Pixels changed outside the gold hazard region | 0% | FFmpeg 0% | Same three cases |
| Mean structural similarity | **0.7871** | FFmpeg 0.6523 | Same three cases |

Raw case-level values, fixed comparator revisions, input manifest hash, and reproduction metadata are in [`benchmarks/direct-baseline/results.json`](benchmarks/direct-baseline/results.json). Three cases are useful regression evidence, not a universal accuracy ranking.

## 7 Engine scope and limitations

The verified source-bound path is Godot. FlashPatch does not claim production Unity or Unreal repair support.

The Unity command is source preflight only. It binds manifest-declared files before editor import; it does not build, render, or repair a Unity project.

```bash
flashpatch unity-preflight unity-source-manifest.json /path/to/UnityProject
```

External renderer captures can be analyzed without claiming engine identity or source causality:

```bash
flashpatch renderer-intake capture.npz --receipt artifacts/capture-intake.json
```

The current detector is standards-oriented engineering evidence, not a clinical certification. Detection quality is not claimed to exceed every specialist detector. The product advantage demonstrated here is the source-bound repair and same-game replay contract.

## 8 Standards and research record

Thresholds are pinned to primary standards and tracked with claim status:

- [`docs/research/2026-WCAG22-THRESHOLDS.md`](docs/research/2026-WCAG22-THRESHOLDS.md) documents the general-flash threshold, area fraction, and time-window boundary.
- [`docs/research/sources.json`](docs/research/sources.json) records primary sources, retrieval metadata, and hashes.
- [`docs/research/claims.json`](docs/research/claims.json) separates confirmed claims from open gaps.
- [`docs/research/standards-boundary-vectors.json`](docs/research/standards-boundary-vectors.json) contains detector boundary vectors.

## 9 Development and release checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py
```

CI runs the portable regression subset on Python 3.11 and 3.12 across Linux, Windows, and macOS, plus the complete public suite on Linux. Package tests build a wheel, install it outside the checkout, and exercise the installed CLI. The release checker validates the tracked public boundary and the CycloneDX source SBOM.

Security reports follow [`SECURITY.md`](SECURITY.md). Contribution setup and review expectations are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 10 License and supply chain

FlashPatch is licensed under [Apache License 2.0](LICENSE). Third-party attribution is in [NOTICE](NOTICE). The checked-in CycloneDX 1.5 source SBOM is [`sbom/flashpatch.cdx.json`](sbom/flashpatch.cdx.json).

## 12 Change history

- 2026-08-26: added the one-command actual-Godot proof, independently verifiable image and receipt bundle, fail-closed reversal tests, capability comparison, and measured evidence table.
- 2026-08-25: synchronized the public source package, added the installed four-state fixture, and hardened private-path and SBOM checks.
- 2026-08-24: documented anonymous installation and separated engine-free demonstration from renderer evidence.
