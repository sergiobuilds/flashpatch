# FlashPatch

**FlashPatch는 게임의 위험한 번쩍임을 찾고, 원인 코드 한 곳을 고친 뒤, 수정 전과 같은 입력·상태로 장면을 다시 실행해 게임이 망가지지 않았는지 확인합니다.**

_Fail-closed visual QA for game development: one source edit, the same gameplay replayed, and a hash-bound verification record._

Contents: [비교](#1-기존-도구보다-무엇을-더-합니까) · [한국어 요약](#2-한국어-요약) · [Problem](#3-the-problem) · [Quickstart](#4-anonymous-clone-and-30-second-demo) · [Verdicts](#5-what-a-verdict-means) · [Replay contract](#7-the-replay-contract) · [Engine scope](#8-engine-scope) · [Development](#10-development-and-public-release-checks) · [License and SBOM](#11-license-and-sbom)

## 1 기존 도구보다 무엇을 더 합니까

폭발 효과가 너무 강하게 번쩍인다고 가정해 보겠습니다.

1. **찾습니다.** 실제 게임 화면에서 위험한 순간을 잡습니다.
2. **원인을 고칩니다.** 그 순간에 실행된 효과와 소스 변수를 연결하고, 허용된 값 하나만 바꿔 시험합니다.
3. **게임을 다시 확인합니다.** 수정 전과 같은 조작과 상태로 장면을 재실행합니다. 번쩍임은 줄고 플레이 결과는 같을 때만 `PASS`를 남깁니다.

차이는 1번이 아니라 **2번과 3번까지 한 흐름으로 끝낸다**는 점입니다. 증거가 부족하거나 게임 상태가 달라지면 성공으로 꾸미지 않고 `INCONCLUSIVE`로 종료합니다.

### 1.1 사용하고 나면 무엇이 남습니까

| 현재 공개 도구 | 도구가 주는 결과 | 개발자가 이어서 해야 하는 일 |
|---|---|---|
| [Ubisoft Chroma](https://github.com/ubisoft/Chroma) | 플레이 화면의 색각 이상 시뮬레이션 | 문제를 판단하고 디자인·소스를 고친 뒤 다시 시험 |
| [EA IRIS](https://github.com/electronicarts/IRIS), [TooFlashy](https://github.com/hashb/TooFlashy), [Q6](https://github.com/qwertey6/Q6), [EPI-LENS](https://github.com/Pi-0r-Tau/EPI-LENS) | 위험이 발생한 시각 또는 분석 보고서 | 원인 코드를 찾고 수정한 뒤 같은 장면을 다시 시험 |
| [FFmpeg](https://ffmpeg.org/ffmpeg-filters.html#photosensitivity), [Apple VideoFlashingReduction](https://github.com/apple/VideoFlashingReduction), [Kaya](https://github.com/samfatu/pse-detection-correction) | 번쩍임을 줄인 영상 | 실제 게임 소스의 원인과 플레이 부작용을 별도로 확인 |
| [Unflash](https://github.com/Kelardry/unflash-video) | 수정하고 다시 검사한 영상 | 실제 게임 소스 수정과 게임 상태 보존을 별도로 확인 |
| **FlashPatch** | **원인 소스 변수 한 곳의 수정안, 같은 입력·상태의 재실행 결과, 해시로 묶인 검증 기록** | **검증 기록을 검토해 실제 작업 브랜치에 반영. 증거가 부족하면 `INCONCLUSIVE`의 원인을 해결** |

Ubisoft Chroma는 색각 이상, FlashPatch는 광과민성 번쩍임을 다루므로 탐지 성능을 직접 겨루는 경쟁품은 아닙니다. 다만 접근성 QA의 **완료 지점**은 비교할 수 있습니다. Chroma가 문제를 실시간으로 보여 주어 사람이 수정하도록 돕는다면, FlashPatch의 검증된 Godot 경로는 원인 소스 수정 시험과 동일 플레이 재검증까지 이어집니다.

이 표는 2026년 8월 26일에 공식 공개 설명과 고정 revision을 다시 확인한 결과입니다. 공개 검색에서 발견하지 못한 프로젝트까지 없다고 주장하지 않으며, 서로 다른 제품군의 빈 기능을 전체 품질 점수로 사용하지 않습니다.

### 1.2 수치로 확인한 결과

| 질문 | 결과 | 해석 |
|---|---|---|
| 탐지만 더 정확합니까 | FlashPatch 7/9, Kaya 8/9, TooFlashy 8/9 | **아닙니다.** 탐지 전용 수치에서는 FlashPatch가 앞서지 않습니다. |
| 위험을 없앴습니까 | FlashPatch 0%, FFmpeg 0% 잔존 위험 | 봉인 영상 사례 3건에서는 둘 다 위험을 제거했습니다. |
| 화면을 얼마나 덜 바꿨습니까 | FlashPatch 25.00%, FFmpeg 45.83% 변경 픽셀 | 같은 사례에서 FlashPatch가 바꾼 픽셀이 상대적으로 45.5% 적었습니다. |
| 게임 수정까지 검증했습니까 | **FlashPatch만 전체 흐름 확인** | 공개 비교군 중 `화면 → 원인 소스 → 수정 하나 → 같은 플레이 → 상태 보존`을 controlled Godot 사례에서 확인했습니다. |

영상 보조 탐지 결과는 FlashPatch 3/3, EA IRIS 3/3, EPI-LENS 1/3이었습니다. 원시 측정값과 실행 조건은 [`benchmarks/direct-baseline/results.json`](benchmarks/direct-baseline/results.json)에 고정돼 있습니다. 현재 source-bound 검증 범위는 Godot이며, Unity와 Unreal 전체 지원을 뜻하지 않습니다.

## 2 한국어 요약

게임의 번쩍임은 출시 후에 발견하면 늦습니다. 스토어 심사, 등급 분류, 접근성 요구를 다시 통과해야 하고, 무엇보다 실제 이용자에게 발작을 유발할 수 있습니다.

FlashPatch는 출시 전 최종 화면에서 그 위험을 찾아내고, **찾았다는 사실을 재현 가능한 증거로 남깁니다.** 프레임을 눈으로 보고 판단하는 것이 아니라, 선언된 조작 기록을 그대로 재생해 렌더러가 실제로 그린 픽셀을 캡처하고, 위험 구간을 소스 코드의 한 줄·한 파라미터까지 연결합니다.

핵심은 **모르는 것을 모른다고 말하는 것**입니다. 렌더러 증거가 없거나, 타임스탬프가 깨졌거나, 원인이 여러 파라미터에 걸쳐 있으면 추측하지 않고 `INCONCLUSIVE`로 닫습니다. 통과·안전·실패·판정불가 네 가지 결론만 존재하며, 각각이 입력 바이트 해시와 함께 영수증으로 남습니다.

익명 clone과 비편집 설치 상태에서 30초 데모를 직접 확인할 수 있습니다. 이 데모에는 Godot, GPU, 디스플레이가 필요하지 않습니다.

```bash
git clone https://github.com/sergiobuilds/flashpatch-public.git
cd flashpatch-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
flashpatch safety-demo
```

`safety-demo`는 설치된 패키지만으로 `PASS`, `SAFE`, `FAIL`, `INCONCLUSIVE` 네 결론을 생성합니다. 이 deterministic contract fixture는 제품 흐름을 설명하는 데모이며 renderer evidence는 아닙니다.

판정 기준은 임의로 정하지 않았습니다. WCAG 2.2의 일반 번쩍임 임계값과 ITU-R BT.1702 권고를 원문 출처와 함께 [`docs/research/`](docs/research/)에 고정해 두었고, 각 주장의 상태는 [`claims.json`](docs/research/claims.json)에서 확인하실 수 있습니다.

---

## 3 The problem

A single unlucky flash sequence can trigger a seizure. Studios usually catch this at the very end, by eye, on someone's monitor — after the build is locked. When it is caught, the next question has no good answer: *which line of our code caused it, and did fixing it break the game?*

FlashPatch answers both, and refuses to guess when it cannot.

## 4 Anonymous clone and 30-second demo

Python 3.11 or later. No engine, no GPU, no display required.

```bash
git clone https://github.com/sergiobuilds/flashpatch-public.git
cd flashpatch-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
flashpatch safety-demo --json --output artifacts/safety-demo
```

`safety-demo` runs one deterministic contract fixture through localization, an allowlisted one-parameter patch, and same-trace revalidation. It writes a report and one hash-bound receipt for each terminal state.

```json
{
  "results": {
    "fail": "FAIL",
    "inconclusive": "INCONCLUSIVE",
    "pass": "PASS",
    "safe": "SAFE"
  },
  "schema": "flashpatch-safety-demo-v2"
}
```

The source-bound Godot path is separate from this installed demonstration and uses the replay contract below.

## 5 What a verdict means

| Receipt | Meaning |
|---|---|
| `PASS` | A candidate one-parameter edit lowered the risk and preserved the declared gameplay invariants. |
| `SAFE` | The project was already below the risk threshold under its declared trace. |
| `FAIL` | Risk remained, or the edit broke a declared invariant. |
| `INCONCLUSIVE` | The evidence required for a verdict was missing or ambiguous. |

`INCONCLUSIVE` is a feature. Missing renderer evidence, malformed timestamps, ambiguous source binding, multi-parameter edits, and gameplay drift all close as `INCONCLUSIVE` rather than as a guess. A headless numeric signal is a smoke regression and never stands in for pixel evidence.

## 6 How a verdict is produced

1. Replay a declared action trace in an isolated copy of the project.
2. Capture renderer-owned RGB frames when the project supplies a non-headless replay adapter.
3. Detect visual-risk intervals and bind them to timestamped runtime node, property, script, and source-line evidence.
4. Test exactly one allowlisted exported source parameter edit at a time.
5. Accept the patch only when the copied project lowers risk *and* preserves the declared gameplay invariants.

Every step hashes what it read and what it produced, so a receipt can be re-verified later against the same bytes.

## 7 The replay contract

`flashpatch compile` accepts `flashpatch-godot-safety-ci-v1`. A project declares its trace, scene, gameplay preservation fields, and the one-parameter patch candidates FlashPatch is permitted to test.

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

Working examples live in [`benchmarks/aigame-psebench/corpus/`](benchmarks/aigame-psebench/corpus/) — two small Godot projects, one with an interaction-triggered burst and one with a periodic pulse, each with its trace, contract, and ground truth.

```bash
export GODOT_BINARY=/path/to/Godot
flashpatch compile <project> <contract-or-trace> \
  --workspace artifacts/flashpatch-ci \
  --receipt artifacts/flashpatch-ci/receipt.json
```

A renderer-backed adapter declares `frame_npz_v1` and emits uint8 RGB frames with strictly increasing timestamps. FlashPatch hashes that artifact before analysis.

## 8 Engine scope

The verified source-bound path is Godot. A Unity source preflight binds manifest-declared project files before any editor import:

```bash
flashpatch unity-preflight unity-source-manifest.json /path/to/UnityProject
```

It checks manifest-bound source bytes only — it does not import Unity, build a player, or capture frames.

An externally produced capture can be analyzed on its own, without any engine or source claim:

```bash
flashpatch renderer-intake capture.npz --receipt artifacts/capture-intake.json
```

That receipt records the supplied frames, timestamps, and detector result. Engine identity, source causality, replay preservation, and repair effectiveness each require their own bound evidence.

## 9 Thresholds and sources

Detection thresholds are pinned to primary standards rather than chosen by hand:

- [`docs/research/2026-WCAG22-THRESHOLDS.md`](docs/research/2026-WCAG22-THRESHOLDS.md) — general flash threshold, area fraction, window boundary
- [`docs/research/sources.json`](docs/research/sources.json) — every primary source with its retrieval metadata and hash
- [`docs/research/claims.json`](docs/research/claims.json) — each claim, its evidence, and the gaps that remain open
- [`docs/research/standards-boundary-vectors.json`](docs/research/standards-boundary-vectors.json) — boundary vectors the detector is tested against

## 10 Development and public release checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py
```

A clean anonymous clone completes the public test suite without failures. Tests bound to non-public evaluation inputs, runner scripts, release bundles, or the project map are explicitly skipped by [`tests/conftest.py`](tests/conftest.py). Missing public files are not part of that allowlist and remain failures.

CI runs the portable regression subset on Python 3.11 and 3.12 across Linux, Windows, and macOS. That subset invokes the public-release checker for the SBOM and tracked source boundary. A second job runs the complete public suite. The package tests build a wheel, install it outside the checkout, and run the four-state demo from the installed package.

## 11 License and SBOM

[Apache License 2.0](LICENSE). Third-party attribution is recorded in [NOTICE](NOTICE). The checked-in CycloneDX 1.5 source SBOM is [`sbom/flashpatch.cdx.json`](sbom/flashpatch.cdx.json); CI verifies that its direct components match `pyproject.toml`.

## 12 Change history

- 2026-08-26: replaced the capability grid with a blind-reviewed, task-oriented comparison and added the current Ubisoft Chroma boundary.
- 2026-08-25: synchronized the source package with the final candidate, added the four-state installed demo, and hardened the public boundary against private paths and a missing SBOM.
- 2026-08-24: documented anonymous non-editable installation, separated the engine-free and Godot-backed demos, and added public-boundary and SBOM verification.
