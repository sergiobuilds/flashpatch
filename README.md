# FlashPatch

**Fail-closed visual QA for game development.** FlashPatch finds photosensitive-seizure risk and other time-based visual defects in a game before release, and proves each verdict with a hash-bound receipt.

---

## 한국어 요약

게임의 번쩍임은 출시 후에 발견하면 늦습니다. 스토어 심사, 등급 분류, 접근성 요구를 다시 통과해야 하고, 무엇보다 실제 이용자에게 발작을 유발할 수 있습니다.

FlashPatch는 출시 전 최종 화면에서 그 위험을 찾아내고, **찾았다는 사실을 재현 가능한 증거로 남깁니다.** 프레임을 눈으로 보고 판단하는 것이 아니라, 선언된 조작 기록을 그대로 재생해 렌더러가 실제로 그린 픽셀을 캡처하고, 위험 구간을 소스 코드의 한 줄·한 파라미터까지 연결합니다.

핵심은 **모르는 것을 모른다고 말하는 것**입니다. 렌더러 증거가 없거나, 타임스탬프가 깨졌거나, 원인이 여러 파라미터에 걸쳐 있으면 추측하지 않고 `INCONCLUSIVE`로 닫습니다. 통과·안전·실패·판정불가 네 가지 결론만 존재하며, 각각이 입력 바이트 해시와 함께 영수증으로 남습니다.

30초 만에 직접 확인하실 수 있습니다. Godot 설치도 필요 없습니다.

```bash
python -m pip install -e ".[dev]"
flashpatch demo          # 위험 클립을 탐지하고 복구한 뒤 영수증 출력
flashpatch safety-demo --output artifacts/safety-demo   # PASS·SAFE·INCONCLUSIVE 세 영수증
```

판정 기준은 임의로 정하지 않았습니다. WCAG 2.2의 일반 번쩍임 임계값과 ITU-R BT.1702 권고를 원문 출처와 함께 [`docs/research/`](docs/research/)에 고정해 두었고, 각 주장의 상태는 [`claims.json`](docs/research/claims.json)에서 확인하실 수 있습니다.

---

## The problem

A single unlucky flash sequence can trigger a seizure. Studios usually catch this at the very end, by eye, on someone's monitor — after the build is locked. When it is caught, the next question has no good answer: *which line of our code caused it, and did fixing it break the game?*

FlashPatch answers both, and refuses to guess when it cannot.

## Try it in 30 seconds

Python 3.11 or later. No engine, no GPU, no display required.

```bash
python -m pip install -e ".[dev]"

flashpatch demo
```

`demo` runs a deterministic hazardous clip through detection and repair, then prints a receipt: the SHA-256 of every input and output artifact, the detected flash count and area fraction, the thresholds used, and an independent re-check of the repaired output.

```json
{
  "detected_area_fraction": 0.5,
  "detected_max_flash_count": 5.5,
  "input_hazardous": true,
  "independent_repaired_passed": true,
  "profile": "wcag22-general-flash-bootstrap",
  "status": "VERIFIED"
}
```

To see all four terminal states at once:

```bash
flashpatch safety-demo --output artifacts/safety-demo
# {"results": {"pass": "PASS", "safe": "SAFE", "inconclusive": "INCONCLUSIVE"}}
```

## What a verdict means

| Receipt | Meaning |
|---|---|
| `PASS` | A candidate one-parameter edit lowered the risk and preserved the declared gameplay invariants. |
| `SAFE` | The project was already below the risk threshold under its declared trace. |
| `FAIL` | Risk remained, or the edit broke a declared invariant. |
| `INCONCLUSIVE` | The evidence required for a verdict was missing or ambiguous. |

`INCONCLUSIVE` is a feature. Missing renderer evidence, malformed timestamps, ambiguous source binding, multi-parameter edits, and gameplay drift all close as `INCONCLUSIVE` rather than as a guess. A headless numeric signal is a smoke regression and never stands in for pixel evidence.

## How a verdict is produced

1. Replay a declared action trace in an isolated copy of the project.
2. Capture renderer-owned RGB frames when the project supplies a non-headless replay adapter.
3. Detect visual-risk intervals and bind them to timestamped runtime node, property, script, and source-line evidence.
4. Test exactly one allowlisted exported source parameter edit at a time.
5. Accept the patch only when the copied project lowers risk *and* preserves the declared gameplay invariants.

Every step hashes what it read and what it produced, so a receipt can be re-verified later against the same bytes.

## The replay contract

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

## Engine scope

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

## Thresholds and sources

Detection thresholds are pinned to primary standards rather than chosen by hand:

- [`docs/research/2026-WCAG22-THRESHOLDS.md`](docs/research/2026-WCAG22-THRESHOLDS.md) — general flash threshold, area fraction, window boundary
- [`docs/research/sources.json`](docs/research/sources.json) — every primary source with its retrieval metadata and hash
- [`docs/research/claims.json`](docs/research/claims.json) — each claim, its evidence, and the gaps that remain open
- [`docs/research/standards-boundary-vectors.json`](docs/research/standards-boundary-vectors.json) — boundary vectors the detector is tested against

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

A clean anonymous clone reports **818 passed, 86 skipped**. The skips are tests bound to inputs that live in the private development repository — internal evaluation records, release bundles, and the project map. [`tests/conftest.py`](tests/conftest.py) lists exactly which paths and which tests those are.

CI runs the portable regression subset on supported Python versions, plus an Ubuntu job that downloads the pinned Godot 4.7.1 release, verifies its SHA-256, and runs the renderer-backed safety demo.

## License

[Apache License 2.0](LICENSE). Third-party attribution is recorded in [NOTICE](NOTICE).
