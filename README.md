# FlashPatch

FlashPatch turns a rendered visual-safety failure into the smallest permitted game-source correction and verifies the same gameplay again.

Contents: [Result](#1-result) · [Product](#2-product) · [Godot proof](#3-godot-proof) · [Verdicts](#4-verdicts) · [Engine coverage](#5-engine-coverage) · [Architecture](#6-architecture) · [Standards](#7-standards) · [Development](#8-development) · [License](#9-license) · [Change history](#10-change-history)

![Actual Godot renderer capture before and after the verified patch](proof/godot-demo/comparison.png)

## 1 Result

The checked-in proof comes from an actual Godot 4.7.1 X11/OpenGL render.

| Measured outcome | Result |
|---|---:|
| Visual-safety verdict | **FAIL → PASS** |
| Maximum measured risk | **5.0 → 0.0** |
| Source correction | `main.gd:3`, `burst_intensity 1.0 → 0.0` |
| Source assignments changed | **1** |
| Same action trace replayed | **Yes** |
| Timing preserved | **Yes** |
| Gameplay state preserved | **Yes** |
| Semantic invariants preserved | **Yes** |
| Original project modified | **No** |

Inspect the [one-line patch](proof/godot-demo/patch.diff), read the [public receipt](proof/godot-demo/receipt.json), or verify every referenced artifact locally.

```bash
flashpatch verify-godot-demo proof/godot-demo/receipt.json
```

### 1.1 한국어 요약

FlashPatch는 게임 화면의 번쩍임 위험을 찾은 뒤 개발자에게 목록만 넘기지 않습니다. 실제 렌더링 결과를 원인 소스 줄에 연결하고, 프로젝트가 허용한 값 하나를 격리된 복사본에서 수정한 다음, 같은 조작을 다시 재생합니다. 위험이 사라지고 게임 상태가 그대로일 때만 `PASS` 영수증을 남깁니다.

현재 공개 저장소에서 전체 과정이 검증된 엔진은 Godot입니다. Unreal Engine 5.6은 실제 렌더러를 사용한 통제된 검증에서 `PASS`를 기록했습니다. Unity는 소스 파일 결박과 실행 전 검사를 공개하며, 실제 프로젝트 렌더러 검증은 `INCONCLUSIVE` 상태입니다. 세 엔진을 같은 수준으로 포장하지 않고 확인된 범위를 그대로 표시합니다.

## 2 Product

FlashPatch completes one source-bound correction loop.

| Stage | FlashPatch output |
|---|---|
| Capture | Renderer-owned RGB frames and timestamps |
| Detect | Hazard intervals, affected area, and flash type |
| Attribute | Runtime node, property, script, and source line |
| Correct | One allowlisted source assignment in an isolated copy |
| Replay | The same declared action trace |
| Preserve | Timing, gameplay state, and semantic invariants |
| Prove | Hash-bound artifacts and a terminal receipt |

Missing renderer evidence, ambiguous attribution, unauthorized edits, and multi-parameter causality remain `INCONCLUSIVE`. A visually safer result that changes declared gameplay state is `FAIL`.

## 3 Godot proof

Requirements are Python 3.11 or later and Godot 4. Headless Linux also needs `xvfb-run` for the real renderer.

```bash
git clone https://github.com/sergiobuilds/flashpatch.git
cd flashpatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .

export GODOT_BINARY=/path/to/godot
xvfb-run -a flashpatch godot-demo
flashpatch verify-godot-demo artifacts/godot-demo/receipt.json
```

On a desktop session, run `flashpatch godot-demo` without `xvfb-run`. The command uses the included [interaction-burst example](examples/godot/interaction-burst/) and performs render, attribution, correction, replay, and verification. It writes before and after images, a visual comparison, the source diff, the engine receipt, and a portable verification receipt.

For a fast engine-free tour of all terminal states, run:

```bash
flashpatch safety-demo --output artifacts/safety-demo
```

The safety demo is a deterministic contract example. The checked-in Godot proof above is the renderer-backed result.

## 4 Verdicts

| Verdict | Meaning |
|---|---|
| `PASS` | One declared source edit removed the measured risk and preserved the replay contract. |
| `SAFE` | The original replay was already below the declared threshold. |
| `FAIL` | Risk remained, or the correction broke a declared invariant. |
| `INCONCLUSIVE` | Required evidence was missing, ambiguous, malformed, or outside the permitted correction scope. |

Five reversal examples keep the terminal states honest.

| Reversal | Required result |
|---|---:|
| Risk remains after the allowed edit | `FAIL` |
| Risk disappears but gameplay state changes | `FAIL` |
| Two parameters are required | `INCONCLUSIVE` |
| Renderer timestamps are missing | `INCONCLUSIVE` |
| Scene starts below threshold | `SAFE` |

## 5 Engine coverage

Each engine is labeled by the strongest evidence currently available.

| Engine | Verified path | Evidence level | Result |
|---|---|---|---:|
| **Godot 4.7.1** | Actual renderer → source line → one source edit → same-trace replay | Public end-to-end | **`PASS`** |
| **Unreal Engine 5.6** | Actual renderer → controlled runtime-parameter correction → preservation checks | Controlled renderer | **`PASS`** |
| **Unity 2022.3.8f1** | Manifest-bound source preflight plus natural-project renderer run | Source preflight; renderer evidence incomplete | **`INCONCLUSIVE`** |

The Unreal result preserves the action sequence, timing, object identity, terminal state, visual intent, and gameplay state. It is controlled runtime-parameter evidence, not a production source-repair adapter.

The Unity public command binds declared source files before editor import. Its natural-project renderer run did not establish hazard removal, reproducible capture artifacts, and gameplay-state preservation together.

```bash
flashpatch unity-preflight unity-source-manifest.json /path/to/UnityProject
```

External renderer captures can also be inspected without asserting engine identity or source causality.

```bash
flashpatch renderer-intake capture.npz --receipt artifacts/capture-intake.json
```

The machine-readable engine matrix and receipt hashes are in [proof/engine-coverage.json](proof/engine-coverage.json).

## 6 Architecture

```text
action trace + game project
            │
            ▼
 isolated original replay ──► renderer RGB frames ──► hazard interval
            │                                            │
            │                                            ▼
            └──────── runtime event ──► node / property / script / line
                                                         │
                                                         ▼
                                       one allowlisted source edit
                                                         │
                                                         ▼
                                             same-trace replay
                                                         │
                                                         ▼
                                  risk removed + game preserved + receipt
```

The original project is hashed before and after the run. All edits occur inside an isolated copy. A candidate is accepted only when the renderer pixels pass and the declared replay invariants match the original run.

The project contract defines the permitted intervention.

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

Two redistributable Godot projects with traces and contracts are available under [examples/godot](examples/godot/).

## 7 Standards

The detector boundary is documented in [WCAG 2.2 thresholds](docs/research/2026-WCAG22-THRESHOLDS.md). Machine-readable edge cases are in [standards-boundary-vectors.json](docs/research/standards-boundary-vectors.json).

## 8 Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py
```

CI runs the portable regression subset on Python 3.11 and 3.12 across Linux, Windows, and macOS. The complete public suite runs on Linux. Package tests build and install the wheel outside the checkout before exercising the CLI. The release checker validates tracked files and the CycloneDX source SBOM.

Security reports follow [SECURITY.md](SECURITY.md). Contribution setup is in [CONTRIBUTING.md](CONTRIBUTING.md).

## 9 License

FlashPatch is licensed under [Apache License 2.0](LICENSE). Third-party attribution is in [NOTICE](NOTICE). The checked-in CycloneDX 1.5 source SBOM is [sbom/flashpatch.cdx.json](sbom/flashpatch.cdx.json).

## 10 Change history

- 2026-08-26: prepared the public `flashpatch` repository with a one-command Godot proof, independently verifiable artifacts, explicit Godot, Unreal, and Unity evidence levels, and a product-only release surface.
- 2026-08-25: added installed-package checks, release-boundary validation, and supply-chain metadata.
- 2026-08-24: separated the engine-free contract demo from renderer-backed evidence.
