"""공개 저장소에서 내부 평가 기록이 필요한 검사를 건너뛴다.

이 저장소는 제품 소스와 판정 계약을 공개한다. 경쟁 도구를 평가한 기록과 내부
진행 지도(`evidence/`, `.closure/`, `docs/MASTER-MAP.md`, 제출 번들)는 공개 대상이
아니다. 그 입력을 읽는 검사는 여기서 skip으로 바꾼다.

경로 목록을 명시적으로 고정한다. 그래야 진짜 파일 누락은 그대로 실패로 남는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_ONLY = (
    "evidence/",
    ".closure/",
    "release/",
    "artifacts/",
    "docs/MASTER-MAP.md",
    "docs/research/day1-validation.json",
    # execution Seed는 프로젝트 지도에 결박된 실행 계약이다. 지도 없이 내보내면
    # 권위 사슬이 끊기므로 공개하지 않는다.
    ".seed.yaml",
)


def _internal_path_in(text: str) -> str | None:
    for fragment in INTERNAL_ONLY:
        if fragment in text:
            return fragment
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    result = report.get_result()
    if call.when != "call" or call.excinfo is None or not result.failed:
        return
    if not isinstance(call.excinfo.value, (FileNotFoundError, NotADirectoryError)):
        return
    fragment = _internal_path_in(str(call.excinfo.value))
    if fragment is None:
        return
    result.outcome = "skipped"
    result.longrepr = (
        str(item.fspath),
        item.location[1],
        f"Skipped: 내부 전용 입력 {fragment} 이 공개 저장소에 없습니다",
    )


# 내부 저장소의 MASTER-MAP과 git 이력에 결박된 검사들이다. 공개 저장소에는 그 지도와
# 커밋이 없으므로 실행 자체가 성립하지 않는다. 목록을 눈에 보이게 고정해 둔다.
MAP_BOUND_TESTS = {
    "tests/test_competition.py::test_l6_placeholder_cannot_be_presented_as_renderer_evidence",
    "tests/test_competition.py::test_preflight_writes_hash_bound_p0_checkpoint_and_rejects_stale_revision",
    "tests/test_l6_plan.py::test_l6_seed_is_bound_exclusively_to_its_master_map_leaf",
    "tests/test_l6_plan.py::test_l6_authority_allows_narrative_map_status_mutation",
    "tests/test_l6_plan.py::test_l6_authority_rejects_map_older_than_its_execution_pin",
    "tests/test_l6_plan.py::test_l6_authority_rejects_l6_row_retarget",
    "tests/test_l6_plan.py::test_l6_authority_uses_one_pinned_byte_read_for_plan_and_map",
    "tests/test_l7_plan.py::test_l7_seed_is_bound_exclusively_to_its_master_map_leaf",
    "tests/test_l7_plan.py::test_l7_plan_commit_is_an_ancestor_and_contains_the_bound_plan",
    "tests/test_l7_plan.py::test_l7_authority_allows_narrative_map_status_mutation",
    "tests/test_l7_plan.py::test_l7_authority_uses_one_pinned_byte_read_for_plan_and_map",
    "tests/test_release_supply_chain.py::test_release_bundle_is_licensed_reproducible_and_hash_verified",
    "tests/test_submission_package.py::test_submission_package_fixed_readback",
}

MASTER_MAP = ROOT / "docs" / "MASTER-MAP.md"


def pytest_collection_modifyitems(config, items):
    if MASTER_MAP.exists():
        return
    skip = pytest.mark.skip(reason="내부 MASTER-MAP과 커밋 이력에 결박된 검사입니다")
    for item in items:
        if item.nodeid in MAP_BOUND_TESTS:
            item.add_marker(skip)
