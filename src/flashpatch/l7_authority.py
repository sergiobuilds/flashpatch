from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPECTED_PROJECT_ROOT = "."
EXPECTED_MASTER_MAP_VERSION = "2026-08-11_v160"
EXPECTED_PLAN_PATH = "docs/plans/2026-08-02-l7-direct-detector-league.md"
EXPECTED_PLAN_COMMIT = "b930148f088899f3fbf6bc87b1ebbd393ad3c071"
EXPECTED_SEED_PATH = "specs/L7-direct-detector-league-execution.seed.yaml"
EXPECTED_WRITEBACK = (
    "own_leaf_status",
    "status_section",
    "history",
    "evidence_links",
)
EXPECTED_OUTPUTS = (
    "src/flashpatch/competition.py",
    "src/flashpatch/l7_authority.py",
    "src/flashpatch/l7_executor.py",
    "src/flashpatch/l7_external_host_cli.py",
    "src/flashpatch/l7_score.py",
    "src/flashpatch/external_league.py",
    "src/flashpatch/l7_verify.py",
    "tests/test_l7_executor.py",
    "tests/test_l7_executor_integration.py",
    "tests/test_l7_handoff.py",
    "tests/test_l7_score.py",
    "tests/test_l7_plan.py",
    "tests/test_external_league.py",
    "tests/test_l7.py",
    "tests/test_l7_gold_attestation.py",
    "tests/fixtures/l7_direct_detector_league",
    "evidence/l7",
    "artifacts/l7",
)
EXPECTED_GOAL = (
    "Execute only MASTER-MAP leaf L7 using "
    "docs/plans/2026-08-02-l7-direct-detector-league.md. Establish a scoreable "
    "direct detector league from natural public Godot cases, frozen independent "
    "gold, same-input parity, equal-budget repeats, receipt-derived statistics, "
    "and D1-D7 fail-closed checks without treating L6 controlled evidence as "
    "natural, mixing detector and mitigation lanes, modifying the Charter or "
    "siblings, or authorizing any external claim."
)
EXPECTED_BOUNDARIES = {
    "l6_controlled_evidence": "controlled_only_never_natural_or_score_denominator",
    "league_separation": "detector_and_mitigation_never_share_ranking",
    "external_claim_authority": "none",
    "sibling_authority": "none",
}
EXPECTED_DIRECT_DETECTOR_POPULATION = (
    "FlashPatch",
    "KAYA_PSE_DETECTION_CORRECTION_0776EA3",
    "TooFlashy",
)
EXPECTED_CONFORMANCE_ORACLE = "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU"
EXPECTED_SEED_PARTICIPANT_LINES = (
    "- FlashPatch, KAYA_PSE_DETECTION_CORRECTION_0776EA3, and TooFlashy 8274e1ea form the fixed direct detector population. Kaya remains an unscored participant until natural corpus, independent gold, same-input parity, and fair three-repeat receipts all verify. EPI-LENS remains UNSCORABLE until a full same-input application receipt exists.",
    "- KAYA_SOURCE_DIRECT_INPUT_PROTOTYPE_0776EA3E_UNSCORED remains the historical prototype receipt identity only. It cannot enter the detector population, score denominator, ranking, or winner surface.",
    "- EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU is a frozen non-scoring conformance oracle and never a direct detector participant, score input, ranking entry, or winner candidate.",
    "- EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC is an excluded semantic-mismatch baseline with SEMANTIC_MISMATCH_NOT_VERIFIED and NOT_SCOREABLE status. Its release output, archive-closed build, or terminal-only agreement cannot authorize participation.",
    "- G1-G3 freeze FlashPatch, KAYA_PSE_DETECTION_CORRECTION_0776EA3, and TooFlashy as the direct-detector population; require Kaya's verified participant conformance receipt; keep both EA IRIS identities outside the population; preregister at least three public repositories and nine natural cases; and freeze signed blinded candidate-independent gold without admitting L6 controlled evidence or SAFE process controls to the score denominator.",
    "- G4-G5 require Kaya's exact native/direct raw arrays, interval tuples, copied-base replay, tool-specific decode parity, fair end-to-end runtime, and exactly three same-budget ordered normalized repeats from hash-identical RGB and timestamp evidence, preserving every disagreement, failure, retry, timeout, parser conflict, child receipt hash mismatch, and external slot join mismatch.",
)

_PROHIBITED_TOP_LEVEL_KEYS = frozenset(
    {
        "project_charter",
        "project_seed",
        "work_tree",
        "strategy",
        "competitors",
        "competitor_map",
        "siblings",
        "roadmap",
    }
)


class L7AuthorityError(ValueError):
    """Raised when the L7 execution Seed exceeds its authority."""


@dataclass(frozen=True)
class L7AuthorityBinding:
    leaf_id: str
    master_map_version: str
    plan_path: str
    plan_commit: str
    writeback: tuple[str, ...]
    outputs: tuple[str, ...]


def _fail(reason: str) -> None:
    raise L7AuthorityError(reason)


def _single_scalar(text: str, key: str, *, top_level: bool = True) -> str:
    indent = "" if top_level else r"[ \t]+"
    matches = re.findall(rf"(?m)^{indent}{re.escape(key)}:[ \t]*(.*)$", text)
    if len(matches) != 1 or not matches[0].strip():
        location = "top-level " if top_level else "nested "
        _fail(f"{key} must occur exactly once as a non-empty {location}scalar")
    return matches[0].strip()


def _inline_mapping(text: str, key: str) -> dict[str, str]:
    raw = _single_scalar(text, key, top_level=False)
    if not (raw.startswith("{") and raw.endswith("}")):
        _fail(f"{key} must be an inline mapping")
    result: dict[str, str] = {}
    for item in raw[1:-1].split(","):
        if ":" not in item:
            _fail(f"{key} contains a malformed mapping item")
        item_key, value = item.split(":", 1)
        item_key = item_key.strip()
        if not item_key or item_key in result:
            _fail(f"{key} contains an empty or duplicate mapping key")
        result[item_key] = value.strip()
    return result


def _inline_sequence(text: str, key: str) -> tuple[str, ...]:
    raw = _single_scalar(text, key)
    if not (raw.startswith("[") and raw.endswith("]")):
        _fail(f"{key} must be an inline sequence")
    return tuple(item.strip() for item in raw[1:-1].split(",") if item.strip())


def _block_lines(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line == f"{key}:"]
    if len(starts) != 1:
        _fail(f"{key} must occur exactly once as a top-level block")
    block: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line and not line[0].isspace():
            break
        block.append(line)
    return block


def _folded_scalar(text: str, key: str) -> str:
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(rf"{re.escape(key)}:[ \t]*>-", line)
    ]
    if len(starts) != 1:
        _fail(f"{key} must occur exactly once as a folded top-level scalar")
    value: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line and not line[0].isspace():
            break
        if line.strip():
            value.append(line.strip())
    return " ".join(value)


def _frontmatter(text: str, *, label: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        _fail(f"{label} frontmatter is missing")
    parts = text.split("---\n", 2)
    if len(parts) != 3:
        _fail(f"{label} frontmatter is malformed")
    values: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _plan_authority(plan_text: str) -> dict[str, str]:
    pairs = re.findall(r"(?m)^\| `([^`]+)` \| `([^`]+)` \|$", plan_text)
    if len(pairs) != len(dict(pairs)):
        _fail("binding plan authority table contains duplicate keys")
    return dict(pairs)


def _assert_plan_commit(root: Path, local_bytes: bytes) -> None:
    committed_plan = subprocess.run(
        ["git", "show", f"{EXPECTED_PLAN_COMMIT}:{EXPECTED_PLAN_PATH}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if committed_plan.returncode != 0:
        _fail(
            f"binding plan commit does not contain {EXPECTED_PLAN_PATH}: "
            f"{EXPECTED_PLAN_COMMIT}"
        )
    if committed_plan.stdout != local_bytes:
        _fail(f"binding plan differs from its pinned commit: {EXPECTED_PLAN_COMMIT}")
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PLAN_COMMIT, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if is_ancestor.returncode != 0:
        _fail(
            f"binding plan commit is not an ancestor of HEAD: {EXPECTED_PLAN_COMMIT}"
        )


def _assert_plan_contract(plan_text: str) -> None:
    frontmatter = _frontmatter(plan_text, label="binding plan")
    if frontmatter != {
        "doc_kind": "project-material",
        "status": "working",
        "version": "2026-08-11_v81",
        "canonical_path": "self",
    }:
        _fail("binding plan frontmatter changed")

    required_boundaries = (
        "L6 Sparta controlled mutation은 엔진 proof이며 L7 natural case, independent gold, external win denominator가 아닙니다.",
        "FFmpeg `vf_photosensitivity`는 화면을 바꾸는 mitigation 기준선입니다. detector league에 넣지 않고 별도 repair league로 분리합니다.",
        "L7 완료는 L8 blind league, L9 external claim 또는 대회 우승을 뜻하지 않습니다.",
        "L8 handoff는 detector와 mitigation 결과를 합친 combined winner를 만들 수 없습니다.",
        "현재 detector handoff generator는 scoreable L7 receipt에서 direct-detector identity-free card 하나만 생성합니다.",
        "각 detector는 `detector_cell_N`으로 redacted되며, identity mapping, ranking, winner, combined winner, external claim surface는 출력하지 않습니다.",
        "candidate output을 보기 전에 repository pool, inclusion·exclusion rule, trace budget, selection order, stop rule을 discovery manifest로 동결합니다.",
        "primary lane은 사례별 `SAFE`·`HAZARDOUS` exact match입니다. detector league의 순위 판정은 이 case-level endpoint만 사용합니다.",
        "검증 가능한 독립 서명과 blind seal이 없으면 gold를 승인하지 않습니다.",
        "packet verifier는 blinded input, frame copy, timestamp projection, corpus commitment, rubric, adjudicator instructions, expected return contract, trust policy template hash를 다시 열어 `INDEPENDENT_GOLD_MISSING`으로 닫습니다.",
        "expected return contract는 per-case returned gold skeleton이 signer identity·role·timestamp·signature field, blind mapping hash, case freeze hash, trust policy hash를 가져야 한다고 명시합니다.",
        "return verifier는 blind intake receipt, adjudicator packet manifest, returned package를 함께 열어 blinded case id, packet manifest file hash, packet root hash, corpus commitment, expected return contract, per-case independent-gold file, trust policy file, candidate-start receipt hash를 대조합니다.",
        "expected return contract는 returned manifest가 adjudicator packet manifest file hash와 packet root hash를 echo해야 한다고 명시합니다.",
        "expected return contract와 return verifier는 returned package가 adjudicator packet manifest file hash와 packet root hash를 echo하지 않으면 fail-closed로 거절합니다.",
        "return verifier는 per-case gold skeleton의 signer identity·role·timestamp·signature field, blind mapping hash, case freeze hash, trust policy hash가 없으면 fail-closed로 거절합니다.",
        "return verifier는 adjudicator와 timestamp witness signer가 trust policy roster에 없거나 FlashPatch·Kaya·TooFlashy·EA IRIS·FFmpeg·operator identity와 겹치면 fail-closed로 거절합니다.",
        "trust policy가 externally approved registry에 pinned되지 않았으면 returned package가 구조적으로 맞아도 `INDEPENDENT_GOLD_UNVERIFIED`, `scoreable=false`입니다.",
        "candidate-start gate는 blind intake, adjudicator packet manifest, returned gold package를 먼저 다시 열어 gold return 상태를 확인합니다.",
        "returned gold가 `INDEPENDENT_GOLD_VERIFIED`가 아니면 start receipt가 있어도 `CANDIDATE_START_BLOCKED`, `scoreable=false`입니다.",
        "gate receipt는 adjudicator packet manifest file hash, packet root hash, FlashPatch, Kaya, TooFlashy 고정 population과 9 slot, detector당 3 repeat를 요구합니다.",
        "executor는 caller-supplied candidate-start gate assessment JSON을 신뢰하지 않습니다.",
        "executor는 blind intake receipt, returned gold root, adjudicator packet manifest, candidate-start receipt 원본에서 candidate-start gate를 재계산합니다.",
        "재계산된 gate가 verified `CANDIDATE_START_WITNESS_VERIFIED`가 아니면 frozen v2 request 실행을 시작하지 않습니다.",
        "executor output directory는 실행 전에 비어 있어야 하며, stale `slot-N.json` artifact가 있으면 command 실행 전에 request를 거절합니다.",
        "executor는 candidate-start gate를 원본 입력에서 재계산한 결과가 `CANDIDATE_START_WITNESS_VERIFIED`가 아닐 때 request를 실행하지 않습니다.",
        "runtime은 process start부터 input validation·decode, detector, parse, receipt finalize까지 같은 end-to-end 경계로 측정합니다.",
        "cold 또는 warm cache policy, seeded balanced run order, concurrency limit, process isolation, timed region 밖의 설치·준비 단계를 실행 전에 고정합니다.",
        "세 반복을 best-of 없이 ordered normalized observation으로 만듭니다.",
        "3개 미만 canonicalized public repository identity에서 온 score bundle은 case 수가 9개 이상이어도 L7 `SCOREABLE`이 아니며, `minimum_three_public_repositories_not_met` blocker를 남깁니다.",
        "같은 repository URL의 여러 revision과 `.git` suffix, trailing slash, 대소문자만 다른 URL 변형은 여러 repository로 세지 않습니다.",
        "independent gold의 timestamp나 interval schema가 누락·비수치·역전·zero-length이면 `FAIL_CLOSED:gold:independent_gold_not_verified`입니다.",
        "JSON `NaN`·`Infinity`, non-finite gold timestamp, bool형 또는 중복·비정렬 interval index는 score 계산 입력이 아니며 fail-closed입니다.",
        "fair-runtime child receipt의 `wall_time_ns`는 score verifier가 다시 열어 detector별 runtime summary로 계산해야 하며, 누락·비양수·시작/종료 시간 불일치·top-level/nested ledger 불일치는 fail-closed입니다.",
        "gold case id는 non-empty string이어야 하며 숫자, null, 빈 문자열은 score 입력이 아닙니다.",
        "independent gold projection schema는 `flashpatch-l7-independent-gold-projection-v1`이어야 합니다.",
        "independent gold projection의 receipt와 trust policy path·SHA-256은 score bundle에서 재개봉한 원본 파일과 일치해야 합니다.",
        "score receipt는 fixed direct detector population과 `detector_population_sha256`을 top-level과 각 case row에 남겨야 하며, EA IRIS release oracle이나 mitigation identity가 detector population에 섞이면 fail-closed입니다.",
        "score receipt는 comparator별 false negative count, false positive count, disagreement case list, secondary case diagnostics, onset error seconds를 frozen gold, parity, fair-runtime observation에서 다시 계산해야 합니다.",
        "independent gold의 `timestamps_seconds`는 strictly increasing이어야 하며, 중복·역전 timestamp는 `FAIL_CLOSED:gold:independent_gold_not_verified`입니다.",
        "repository URL은 query·fragment·credential 없는 public HTTP(S) repository URL이어야 하며, repository URL 또는 revision이 비어 있거나 문자열이 아니면 score bundle은 fail-closed입니다.",
        "그 전에는 standard-derived oracle을 새로 채택하는 방향 전환도 하지 않고, L7을 `NOT_SCOREABLE`로 유지합니다.",
        "L8 전에는 우위 또는 winner 표현을 하지 않습니다.",
        "`EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU`는 official Ubuntu release `1.1.0`의 frozen conformance oracle입니다.",
        "release 결과로 대체하거나 timing 차이를 정규화하지 않으며 `SEMANTIC_MISMATCH_NOT_VERIFIED`·`NOT_SCOREABLE`로 유지합니다.",
        "Kaya detector는 pinned Python 3.10 native/direct exact fixture, independent copied interpreter-base replay, import-closure adversarial audit을 통과했습니다.",
        "이 사실은 comparator participant 자격만 뜻하며, natural corpus·independent gold·fair repeat·score receipt 전에는 `NOT_SCOREABLE`입니다.",
    )
    if any(boundary not in plan_text for boundary in required_boundaries):
        _fail("binding plan no longer preserves an L7 evidence or claim boundary")

    required_gates = tuple(f"### 4.{index + 1} G{index} " for index in range(8))
    if any(gate not in plan_text for gate in required_gates):
        _fail("binding plan must retain every G0-G7 gate")
    required_adversarial = tuple(f"| D{index} |" for index in range(1, 8))
    if any(case not in plan_text for case in required_adversarial):
        _fail("binding plan must retain every D1-D7 adversarial case")


def validate_l7_authority(
    seed_path: Path, *, repository_root: Path | None = None
) -> L7AuthorityBinding:
    """Validate the execution Seed against only MASTER-MAP leaf L7."""

    seed_path = seed_path.resolve()
    root = (repository_root or seed_path.parents[1]).resolve()
    try:
        relative_seed = seed_path.relative_to(root).as_posix()
    except ValueError:
        _fail("Seed must be inside the repository")
    if relative_seed != EXPECTED_SEED_PATH:
        _fail(f"unexpected L7 Seed path: {relative_seed}")

    seed_text = seed_path.read_text(encoding="utf-8")
    top_level_keys = {
        match.group(1)
        for match in re.finditer(r"(?m)^([a-z][a-z0-9_]*):(?:[ \t]|$)", seed_text)
    }
    prohibited = sorted(top_level_keys & _PROHIBITED_TOP_LEVEL_KEYS)
    if prohibited:
        _fail(f"Seed contains map-owned top-level keys: {','.join(prohibited)}")

    expected_scalars = {
        "authority_role": "execution-seed",
        "authority_project_root": EXPECTED_PROJECT_ROOT,
        "authority_master_map_version": EXPECTED_MASTER_MAP_VERSION,
        "authority_leaf_id": "L7",
    }
    for key, expected in expected_scalars.items():
        observed = _single_scalar(seed_text, key)
        if observed != expected:
            _fail(f"{key} must be {expected}, observed {observed}")

    writeback = _inline_sequence(seed_text, "authority_writeback")
    if writeback != EXPECTED_WRITEBACK:
        _fail("L7 writeback may target only its own status, history, and evidence links")
    if _folded_scalar(seed_text, "goal") != EXPECTED_GOAL:
        _fail("L7 goal changed or crossed a sibling, Charter, or external-claim boundary")

    participant_lines = tuple(
        line.strip()
        for line in seed_text.splitlines()
        if line.strip().startswith("- ")
        and ("iris" in line.lower() or "kaya" in line.lower())
    )
    if participant_lines != EXPECTED_SEED_PARTICIPANT_LINES:
        _fail("L7 Kaya participation or EA IRIS exclusion contract changed or became incomplete")

    observed_boundaries = {
        key: _single_scalar(seed_text, key, top_level=False)
        for key in EXPECTED_BOUNDARIES
    }
    if observed_boundaries != EXPECTED_BOUNDARIES:
        _fail("L7 evidence, league, sibling, or external-claim boundary changed")

    outputs = tuple(
        line.removeprefix("  - ").strip()
        for line in _block_lines(seed_text, "outputs")
        if line.startswith("  - ")
    )
    if outputs != EXPECTED_OUTPUTS:
        _fail("L7 output scope changed")

    plan_binding = _inline_mapping(seed_text, "binding_plan")
    if plan_binding != {"path": EXPECTED_PLAN_PATH, "commit": EXPECTED_PLAN_COMMIT}:
        _fail("L7 binding plan path or commit changed")
    map_binding = _inline_mapping(seed_text, "master_map")
    if map_binding != {
        "path": "docs/MASTER-MAP.md",
        "version": EXPECTED_MASTER_MAP_VERSION,
    }:
        _fail("L7 MASTER-MAP binding changed")

    plan_path = root / EXPECTED_PLAN_PATH
    map_path = root / "docs" / "MASTER-MAP.md"
    if not plan_path.is_file() or not map_path.is_file():
        _fail("bound plan or MASTER-MAP is missing")

    plan_bytes = plan_path.read_bytes()
    map_bytes = map_path.read_bytes()
    _assert_plan_commit(root, plan_bytes)
    try:
        plan_text = plan_bytes.decode("utf-8")
        map_text = map_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("bound plan or MASTER-MAP is not strict UTF-8")
    _assert_plan_contract(plan_text)
    expected_plan_authority = {
        "authority_role": "execution-seed",
        "authority_project_root": EXPECTED_PROJECT_ROOT,
        "authority_master_map_version": EXPECTED_MASTER_MAP_VERSION,
        "authority_leaf_id": "L7",
        "authority_writeback": ",".join(EXPECTED_WRITEBACK),
    }
    if _plan_authority(plan_text) != expected_plan_authority:
        _fail("binding plan authority table does not exactly match the L7 Seed")

    frontmatter = _frontmatter(map_text, label="MASTER-MAP")
    if frontmatter.get("authority_model") != "v2":
        _fail("MASTER-MAP does not opt in to authority model v2")
    if frontmatter.get("version") != EXPECTED_MASTER_MAP_VERSION:
        _fail("MASTER-MAP version differs from the L7 binding")

    work_tree_match = re.search(
        r"(?ms)^## 4 Work Tree\n(?P<body>.*?)(?=^## 5 )", map_text
    )
    if work_tree_match is None:
        _fail("MASTER-MAP Work Tree section is missing")
    work_tree = work_tree_match.group("body")
    leaf_ids = re.findall(r"\| (L\d+) \|", work_tree)
    if leaf_ids.count("L7") != 1:
        _fail("MASTER-MAP must contain exactly one L7 leaf")
    if not {"L6", "L8", "L9"}.issubset(leaf_ids):
        _fail("MASTER-MAP sibling boundary is incomplete")
    l7_rows = [line for line in work_tree.splitlines() if "| L7 |" in line]
    if len(l7_rows) != 1 or EXPECTED_PLAN_PATH not in l7_rows[0]:
        _fail("MASTER-MAP L7 row is not bound to the expected plan")

    return L7AuthorityBinding(
        leaf_id="L7",
        master_map_version=EXPECTED_MASTER_MAP_VERSION,
        plan_path=EXPECTED_PLAN_PATH,
        plan_commit=EXPECTED_PLAN_COMMIT,
        writeback=writeback,
        outputs=outputs,
    )
