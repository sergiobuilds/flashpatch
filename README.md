# FlashPatch

**Fail-closed visual QA for game development.** FlashPatch finds photosensitive-seizure risk and other time-based visual defects in a game before release, and proves each verdict with a hash-bound receipt.

Contents: [한국어 요약](#1-한국어-요약) · [Problem](#2-the-problem) · [Quickstart](#3-anonymous-clone-and-30-second-demo) · [Verdicts](#4-what-a-verdict-means) · [Replay contract](#6-the-replay-contract) · [Engine scope](#7-engine-scope) · [Development](#9-development-and-public-release-checks) · [License and SBOM](#10-license-and-sbom)

## 1 한국어 요약

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

## 2 The problem

A single unlucky flash sequence can trigger a seizure. Studios usually catch this at the very end, by eye, on someone's monitor — after the build is locked. When it is caught, the next question has no good answer: *which line of our code caused it, and did fixing it break the game?*

FlashPatch answers both, and refuses to guess when it cannot.

## 3 Anonymous clone and 30-second demo

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

## 4 What a verdict means

| Receipt | Meaning |
|---|---|
| `PASS` | A candidate one-parameter edit lowered the risk and preserved the declared gameplay invariants. |
| `SAFE` | The project was already below the risk threshold under its declared trace. |
| `FAIL` | Risk remained, or the edit broke a declared invariant. |
| `INCONCLUSIVE` | The evidence required for a verdict was missing or ambiguous. |

`INCONCLUSIVE` is a feature. Missing renderer evidence, malformed timestamps, ambiguous source binding, multi-parameter edits, and gameplay drift all close as `INCONCLUSIVE` rather than as a guess. A headless numeric signal is a smoke regression and never stands in for pixel evidence.

## 5 How a verdict is produced

1. Replay a declared action trace in an isolated copy of the project.
2. Capture renderer-owned RGB frames when the project supplies a non-headless replay adapter.
3. Detect visual-risk intervals and bind them to timestamped runtime node, property, script, and source-line evidence.
4. Test exactly one allowlisted exported source parameter edit at a time.
5. Accept the patch only when the copied project lowers risk *and* preserves the declared gameplay invariants.

Every step hashes what it read and what it produced, so a receipt can be re-verified later against the same bytes.

## 6 The replay contract

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

## 7 Engine scope

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

## 8 Thresholds and sources

Detection thresholds are pinned to primary standards rather than chosen by hand:

- [`docs/research/2026-WCAG22-THRESHOLDS.md`](docs/research/2026-WCAG22-THRESHOLDS.md) — general flash threshold, area fraction, window boundary
- [`docs/research/sources.json`](docs/research/sources.json) — every primary source with its retrieval metadata and hash
- [`docs/research/claims.json`](docs/research/claims.json) — each claim, its evidence, and the gaps that remain open
- [`docs/research/standards-boundary-vectors.json`](docs/research/standards-boundary-vectors.json) — boundary vectors the detector is tested against

## 9 Development and public release checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/check_public_release.py
```

A clean anonymous clone completes the public test suite without failures. Tests bound to non-public evaluation inputs, runner scripts, release bundles, or the project map are explicitly skipped by [`tests/conftest.py`](tests/conftest.py). Missing public files are not part of that allowlist and remain failures.

CI runs the portable regression subset on Python 3.11 and 3.12 across Linux, Windows, and macOS. That subset invokes the public-release checker for the SBOM and tracked source boundary. A second job runs the complete public suite. The package tests build a wheel, install it outside the checkout, and run the four-state demo from the installed package.

## 10 License and SBOM

[Apache License 2.0](LICENSE). Third-party attribution is recorded in [NOTICE](NOTICE). The checked-in CycloneDX 1.5 source SBOM is [`sbom/flashpatch.cdx.json`](sbom/flashpatch.cdx.json); CI verifies that its direct components match `pyproject.toml`.

## 11 Change history

- 2026-08-25: synchronized the source package with the final candidate, added the four-state installed demo, and hardened the public boundary against private paths and a missing SBOM.
- 2026-08-24: documented anonymous non-editable installation, separated the engine-free and Godot-backed demos, and added public-boundary and SBOM verification.
