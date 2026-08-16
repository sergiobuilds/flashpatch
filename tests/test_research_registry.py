import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "docs" / "research"
SOURCES_PATH = RESEARCH_DIR / "sources.json"
CLAIMS_PATH = RESEARCH_DIR / "claims.json"
REQUIREMENTS_PATH = RESEARCH_DIR / "2026-CONTEST-REQUIREMENTS.md"
VALIDATION_PATH = RESEARCH_DIR / "day1-validation.json"
MASTER_MAP_PATH = ROOT / "docs" / "MASTER-MAP.md"
CHRONICLE_PATH = ROOT / "docs" / "CHRONICLE.md"
PLAN_PATH = ROOT / ".hermes" / "plans" / "2026-07-26_125730-flashpatch-95-day-grand-prize-campaign.md"

REQUIRED_SOURCE_FIELDS = {
    "id",
    "title",
    "url",
    "source_type",
    "published_or_version",
    "retrieved_at",
    "sha256",
    "status",
    "notes",
}
REQUIRED_CLAIM_FIELDS = {
    "id",
    "statement",
    "status",
    "source_ids",
    "locator",
    "implementation_consequence",
}
REQUIRED_CLAIM_IDS = {
    "direct-code-osi-license",
    "external-material-provenance",
    "submit-complete-source",
    "publish-public-repository",
    "winner-five-year-public-retention",
    "evaluation-repository-snapshot",
    "development-outside-evaluation",
    "embedded-ai-open-weight-minimum",
    "closed-api-model-restriction",
    "coding-assistant-permitted",
    "understanding-may-affect-score",
    "government-support-history-disclosure",
    "government-support-history-verification",
    "report-deadline",
    "report-original-and-pdf",
    "report-body-format",
    "report-public-repository-url",
    "report-youtube-demo-url",
    "sbom-required-unlimited",
    "sbom-required-fields",
    "ai-model-attachment-when-applicable",
    "coding-assistant-disclosure",
    "orientation-date-and-access",
    "orientation-evaluation-session",
}
EXPECTED_HASHES = {
    "contest-operating-rules-2026": "5c129ed9f389ecc04b6f7ba8b97f719a313efaf32aea9178e635500023ae1da1",
    "result-report-template-2026": "9a5d2968d48ff8a8fd85ce991dc72dc2b0818d7e8c06ebb871cc97ce5cc62d95",
    "orientation-program-image-2026": "808e6b3c6830a6b05431795dfe1fa4ea325cbee81ac5feaf450985b0153a77d7",
    "itu-r-bt1702-3": "9d1167de5553a82c633020a44b5f729118e90ff6536a3b519f1813e235437bf2",
    "ffmpeg-photosensitivity-601d9ee": "4be61b30d16672cc87dc1f09399c40dea9fd9a5042a201aefe5f4639d6ed9c7b",
}
EXPECTED_URLS = {
    "contest-operating-rules-2026": "https://api.osscontest.kr/static/uploads/b3b4491a-3bbe-454e-a1d8-6ed475b01b14.pdf",
    "result-report-template-2026": "https://api.osscontest.kr/static/uploads/46414fba-c473-4dae-b595-7214d635b494.zip",
    "orientation-program-image-2026": "https://api.osscontest.kr/static/uploads/8d98fbbb-d256-4fa1-9521-1ff689f0c885.png",
    "orientation-notice-2026": "https://osscontest.kr/notice/31",
    "itu-r-bt1702-3": "https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf",
    "ffmpeg-photosensitivity-601d9ee": "https://github.com/FFmpeg/FFmpeg/blob/601d9ee881fbd9d9ff44466c561c480ff244eb9f/libavfilter/vf_photosensitivity.c",
}
DAY1_MANAGED_PATHS = (
    ".hermes/plans/2026-07-26_125730-flashpatch-95-day-grand-prize-campaign.md",
    "docs/CHRONICLE.md",
    "docs/MASTER-MAP.md",
    "docs/research/2026-CONTEST-REQUIREMENTS.md",
    "docs/research/claims.json",
    "docs/research/day1-validation.json",
    "docs/research/sources.json",
    "tests/test_research_registry.py",
)
PROHIBITED_EXPRESSIONS = (
    "원하시면",
    "필요하시면",
    "다음 단계는",
    "이렇게 하면 좋겠습니다",
    "—",
)
EXPECTED_FRONTMATTER = {
    "doc_kind": "project-material",
    "status": "canonical",
    "version": "2026-07-26_v1",
    "canonical_path": "self",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = read_text(path)
    assert text.startswith("---\n")
    block = text.split("---\n", 2)[1]
    return dict(line.split(": ", 1) for line in block.splitlines() if line)


def test_source_registry_has_required_schema_and_pinned_official_artifacts():
    registry = load_json(SOURCES_PATH)

    assert isinstance(registry["version"], str) and registry["version"]
    assert isinstance(registry["retrieved_at"], str) and registry["retrieved_at"]
    assert len(registry["sources"]) >= 5

    source_ids = [source["id"] for source in registry["sources"]]
    assert len(source_ids) == len(set(source_ids))
    for source in registry["sources"]:
        assert REQUIRED_SOURCE_FIELDS <= source.keys()
        assert source["url"].startswith("https://")
        assert source["status"] in {"available", "not_found"}
        assert source["sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", source["sha256"])

    sources_by_id = {source["id"]: source for source in registry["sources"]}
    for source_id, expected_url in EXPECTED_URLS.items():
        assert sources_by_id[source_id]["url"] == expected_url
    assert set(EXPECTED_HASHES) < set(EXPECTED_URLS)
    for source_id, expected_hash in EXPECTED_HASHES.items():
        assert sources_by_id[source_id]["sha256"] == expected_hash

    validation = load_json(VALIDATION_PATH)
    assert set(validation) == {
        "version",
        "task",
        "red",
        "green",
        "full_suite",
        "quality_checks",
    }
    assert validation["version"] == "2026-07-26_v1"
    assert validation["task"] == "Day 1 official requirements registry"
    assert set(validation["red"]) == {
        "command",
        "expected_reason",
        "failed_test_ids",
        "passing_test_ids",
        "exit_code",
        "started_at",
        "source_log",
        "source_log_note",
        "result",
        "observed_summary",
    }
    assert validation["red"]["result"] == "failed"
    assert validation["red"]["observed_summary"] == "2 failed, 1 passed"
    assert validation["red"]["failed_test_ids"] == [
        "test_source_registry_has_required_schema_and_pinned_official_artifacts",
        "test_claim_ledger_enforces_evidence_rules_and_required_gaps",
    ]
    assert validation["red"]["passing_test_ids"] == [
        "test_raw_official_artifacts_are_not_committed"
    ]
    assert validation["red"]["exit_code"] == 1
    assert validation["red"]["started_at"] == "2026-07-26T13:19:26+09:00"
    assert Path(validation["red"]["source_log"]).is_absolute()
    assert not Path(validation["red"]["source_log"]).is_relative_to(ROOT)
    assert "저장소 밖" in validation["red"]["source_log_note"]
    assert set(validation["green"]) == {
        "focused_command",
        "result",
        "observed_summary",
    }
    assert validation["green"]["result"] == "passed"
    assert validation["green"]["observed_summary"] == "3 passed"
    assert set(validation["full_suite"]) == {"command", "result", "observed_summary"}
    assert validation["full_suite"]["result"] == "passed"
    assert validation["full_suite"]["observed_summary"] == "13 passed"
    assert isinstance(validation["quality_checks"], list)
    checks_by_name = {check["name"]: check for check in validation["quality_checks"]}
    assert checks_by_name["staged-index-bytes"]["command"] == (
        "FLASH_PATCH_VERIFY_INDEX=1 python -m pytest tests/test_research_registry.py -q"
    )
    json_command = checks_by_name["json-parse"]["command"]
    assert json_command.startswith("python -c ")
    assert "docs/research" in json_command and "json.loads" in json_command
    scan_command = checks_by_name["prohibited-expression-scan"]["command"]
    assert scan_command.startswith("python -c ")
    for expression in PROHIBITED_EXPRESSIONS:
        assert expression in scan_command
    for path in (CLAIMS_PATH, REQUIREMENTS_PATH, PLAN_PATH):
        assert path.relative_to(ROOT).as_posix() in scan_command


def test_claim_ledger_enforces_evidence_rules_and_required_gaps():
    ledger = load_json(CLAIMS_PATH)
    registry = load_json(SOURCES_PATH)

    assert isinstance(ledger["version"], str) and ledger["version"]
    assert len(ledger["claims"]) >= len(REQUIRED_CLAIM_IDS)

    known_source_ids = {source["id"] for source in registry["sources"]}
    claim_ids = [claim["id"] for claim in ledger["claims"]]
    assert len(claim_ids) == len(set(claim_ids))
    assert REQUIRED_CLAIM_IDS <= set(claim_ids)

    for claim in ledger["claims"]:
        assert REQUIRED_CLAIM_FIELDS <= claim.keys()
        assert claim["status"] in {"confirmed", "unconfirmed"}
        assert set(claim["source_ids"]) <= known_source_ids
        if claim["status"] == "confirmed":
            assert claim["source_ids"]
        else:
            assert isinstance(claim.get("what_would_confirm"), str)
            assert claim["what_would_confirm"].strip()

    claims_by_id = {claim["id"]: claim for claim in ledger["claims"]}
    assert claims_by_id["detailed-scoring-table"]["status"] == "unconfirmed"
    assert claims_by_id["orientation-recording-and-materials"]["status"] == "unconfirmed"

    retention = claims_by_id["winner-five-year-public-retention"]
    assert "우수팀 및 수상팀" in retention["statement"]
    assert "우수팀 및 수상팀" in retention["implementation_consequence"]

    closed_api = claims_by_id["closed-api-model-restriction"]
    for exception in ("MCP", "AI 연동 생태계 software", "테스트 목적 API 호출"):
        assert exception in closed_api["statement"]
        assert exception in closed_api["implementation_consequence"]
    assert "제품 핵심" in closed_api["implementation_consequence"]

    disclosure = claims_by_id["government-support-history-disclosure"]
    assert set(disclosure["source_ids"]) == {
        "contest-operating-rules-2026",
        "result-report-template-2026",
    }
    for status in ("현재 참여 중", "결과 대기 중", "이미 수혜"):
        assert status in disclosure["statement"]
    assert "단순 신청" not in disclosure["statement"]
    assert "단순 신청 전체를 공식 기재 의무로 확대하지 않는다" in disclosure[
        "implementation_consequence"
    ]

    assert parse_frontmatter(REQUIREMENTS_PATH) == EXPECTED_FRONTMATTER
    requirements = read_text(REQUIREMENTS_PATH)
    assert "docs/research/day1-validation.json" in requirements
    assert "우수팀 및 수상팀" in requirements
    for exception in ("MCP", "AI 연동 생태계 software", "테스트 목적 API 호출"):
        assert exception in requirements
    for status in ("현재 참여 중", "결과 대기 중", "이미 수혜"):
        assert status in requirements

    master_map = read_text(MASTER_MAP_PATH)
    for pointer in (
        "docs/research/2026-CONTEST-REQUIREMENTS.md",
        "docs/research/sources.json",
        "docs/research/claims.json",
        "docs/research/day1-validation.json",
    ):
        assert pointer in master_map
    assert "Private" in master_map and "Public" in master_map

    chronicle = read_text(CHRONICLE_PATH)
    assert "Day 1" in chronicle and "source registry" in chronicle
    assert "Private" in chronicle and "Public" in chronicle

    plan = read_text(PLAN_PATH)
    assert "우수팀 및 수상팀" in plan
    assert "Day 1에 공식 원문을 찾고, 없으면 unconfirmed gap으로 등록한다" in plan
    assert "## 16. 심사 대리 gate" in plan
    assert "심사 대리 scorecard" not in plan
    assert "85점" not in plan
    assert (
        "AI 모델 활용 명세서를 작성하는 경우 4번 항목에 상용 AI 보조도구 활용 여부와 범위를 기재"
        in plan
    )
    assert "보고서의 보조도구 활용 범위에 기재" not in plan


def test_raw_official_artifacts_are_not_committed():
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")

    prohibited_paths = [
        path
        for path in tracked
        if path
        and (
            path.startswith("docs/research/raw/")
            or (
                path.startswith("docs/research/")
                and Path(path).suffix.lower() in {".pdf", ".zip"}
            )
        )
    ]
    prohibited_hashes = []
    official_hashes = set(EXPECTED_HASHES.values())
    for path in filter(None, tracked):
        blob = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() in official_hashes:
            prohibited_hashes.append(path)

    assert prohibited_paths == []
    assert prohibited_hashes == []


def test_canonical_documents_match_fail_closed_godot_safety_ci():
    readme = read_text(ROOT / "README.md")
    master_map = read_text(MASTER_MAP_PATH)
    current_master_map = master_map.split("## 10 이력", 1)[0]
    campaign = load_json(
        ROOT
        / ".closure"
        / "archive"
        / "flashpatch-ai-game-neurosafety-2026.completed.json"
    )

    assert "research-only through 2026-07-29 17:11 KST" not in readme
    assert "3일 자료조사 전용" not in current_master_map
    assert "2026-07-29 17:11 KST까지" not in current_master_map
    assert campaign["status"] == "completed"
    assert campaign["objective"].startswith("2026 공개SW 개발자대회 일반부 1위를 겨냥해 AI 생성 게임")
    for document in (readme, current_master_map):
        assert "Godot" in document
        assert "Safety CI" in document
        assert "INCONCLUSIVE" in document
        assert "source parameter" in document
    assert "전체 캠페인의 한 경로" not in readme
    assert "전체 캠페인의 한 경로" not in current_master_map
    for unit_id in campaign["scope_ids"]:
        assert f"`{unit_id}`" in master_map


def test_day1_managed_worktree_bytes_match_index_when_requested():
    if os.environ.get("FLASH_PATCH_VERIFY_INDEX") != "1":
        return

    mismatches = []
    for relative_path in DAY1_MANAGED_PATHS:
        index_bytes = subprocess.run(
            ["git", "show", f":{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if (ROOT / relative_path).read_bytes() != index_bytes:
            mismatches.append(relative_path)
    assert mismatches == []


def test_wcag22_threshold_research_records_pinned_primary_sources_and_vectors():
    registry = load_json(SOURCES_PATH)
    sources_by_id = {source["id"]: source for source in registry["sources"]}
    expected_sources = {
        "wcag22-recommendation": {
            "url": "https://www.w3.org/TR/WCAG22/#three-flashes-or-below-threshold",
            "published_or_version": "WCAG 2.2, W3C Recommendation 2023-10-05",
            "retrieved_at": "2026-07-26",
            "sha256": "6e3c5fe397257cae509a2fb4752b73062cf8cbeb92c2cec618989b17e4cf7057",
        },
        "wcag22-understanding-sc231": {
            "url": "https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html",
            "published_or_version": "Updated 2025-09-17",
            "retrieved_at": "2026-07-26",
            "sha256": "3107de0b82b0d56ca44a6c3a2fa0ed1390e22e8cddd5059544ca7fcf495ec709",
        },
        "wcag22-technique-g15": {
            "url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G15",
            "published_or_version": "Updated 2025-07-15",
            "retrieved_at": "2026-07-26",
            "sha256": "dd42a517982c73dff31ae5a66d53da0e49b41418005f74cf16ca906ab097cef9",
        },
        "wcag22-technique-g19": {
            "url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G19",
            "published_or_version": "Updated 2025-07-15",
            "retrieved_at": "2026-07-26",
            "sha256": "64c6c7c72009fcf9984c411a4c407f04dfee23699a7d2ff9691166b94166399f",
        },
        "wcag22-technique-g176": {
            "url": "https://www.w3.org/WAI/WCAG22/Techniques/general/G176",
            "published_or_version": "Updated 2025-07-15",
            "retrieved_at": "2026-07-26",
            "sha256": "9f982583616b6baeed01c1de815cf72db91edc02ab017099fdd0bf3cd86ce99b",
        },
    }
    for source_id, expected in expected_sources.items():
        source = sources_by_id[source_id]
        assert source["status"] == "available"
        assert {field: source[field] for field in expected} == expected
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        assert "license" in source["notes"].lower()

    claims_by_id = {claim["id"]: claim for claim in load_json(CLAIMS_PATH)["claims"]}
    expected_claim_evidence = {
        "wcag22-general-flash-threshold": (
            {"wcag22-recommendation"},
            "#dfn-general-flash-and-red-flash-thresholds",
        ),
        "wcag22-red-flash-threshold": (
            {"wcag22-understanding-sc231"},
            "Field working definition for saturated red",
        ),
        "wcag22-threshold-vectors": (
            {"wcag22-recommendation", "wcag22-technique-g19", "wcag22-technique-g176"},
            "#three-flashes-or-below-threshold; G19 Tests Procedure; G176 Formula 1 and Tests Procedure",
        ),
    }
    for claim_id, (source_ids, locator) in expected_claim_evidence.items():
        claim = claims_by_id[claim_id]
        assert claim["status"] == "confirmed"
        assert set(claim["source_ids"]) == source_ids
        assert claim["locator"] == locator

    thresholds = read_text(RESEARCH_DIR / "2026-WCAG22-THRESHOLDS.md")
    expected_vectors = {
        "general-delta-below": ("|ΔL| = 0.099", "general transition 아님"),
        "general-delta-at": ("|ΔL| = 0.10", "general transition"),
        "general-dark-at": ("L = 0.80", "general transition 아님"),
        "red-ratio-at": ("R/(R+G+B)=0.8", "red 후보"),
        "red-distance-at": ("Δu′v′=0.2", "red 후보 아님"),
        "area-21823": ("21,823 pixels", "G176 간이 경로 통과"),
        "area-21824": ("21,824 pixels", "G176 간이 경로에 넣지 않음"),
        "frequency-three-closed-same-state": ("closed 1-second window", "통과"),
        "frequency-three-closed-different-state": ("closed 1-second window", "실패"),
        "frequency-four": ("4 flash", "빈도 경로 실패"),
    }
    for vector_id, expectations in expected_vectors.items():
        row = next(line for line in thresholds.splitlines() if f"| {vector_id} |" in line)
        for expectation in expectations:
            assert expectation in row
    assert "not a complete WCAG 2.2 conformance implementation" in thresholds


def test_itu_bt1702_and_ffmpeg_baselines_are_pinned_to_primary_sources():
    registry = load_json(SOURCES_PATH)
    sources_by_id = {source["id"]: source for source in registry["sources"]}
    expected_sources = {
        "itu-r-bt1702-3": {
            "url": "https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf",
            "published_or_version": "Recommendation ITU-R BT.1702-3 (11/2023)",
            "sha256": "9d1167de5553a82c633020a44b5f729118e90ff6536a3b519f1813e235437bf2",
        },
        "ffmpeg-photosensitivity-601d9ee": {
            "url": "https://github.com/FFmpeg/FFmpeg/blob/601d9ee881fbd9d9ff44466c561c480ff244eb9f/libavfilter/vf_photosensitivity.c",
            "published_or_version": "FFmpeg master commit 601d9ee881fbd9d9ff44466c561c480ff244eb9f",
            "sha256": "4be61b30d16672cc87dc1f09399c40dea9fd9a5042a201aefe5f4639d6ed9c7b",
        },
    }
    for source_id, expected in expected_sources.items():
        source = sources_by_id[source_id]
        assert source["status"] == "available"
        assert {field: source[field] for field in expected} == expected
        assert source["retrieved_at"] == "2026-07-26"
        assert "license" in source["notes"].lower() or "all rights reserved" in source[
            "notes"
        ].lower()

    claims_by_id = {claim["id"]: claim for claim in load_json(CLAIMS_PATH)["claims"]}
    expected_claims = {
        "itu-bt1702-flash-gate": {
            "source_ids": {"itu-r-bt1702-3"},
            "locator": "Annex 1 Guideline 1, printed pages 3 to 4 (physical PDF pages 5 to 6)",
            "fragments": (
                "160 cd/m² 미만",
                "20 cd/m² 이상",
                "160 cd/m² 이상",
                "1/17 초과",
                "saturated red",
                "25%를 초과",
                "3회를 초과",
            ),
        },
        "itu-bt1702-pattern-gate": {
            "source_ids": {"itu-r-bt1702-3"},
            "locator": "Attachment 1 to Annex 1, printed page 6 (physical PDF page 8)",
            "fragments": ("5쌍을 초과", "40% 초과", "25% 초과", "부드럽게 흐르는"),
        },
        "itu-bt1702-michelson-boundary-conflict": {
            "source_ids": {"itu-r-bt1702-3"},
            "locator": "Annex 1 Guideline 1, printed page 3 (physical PDF page 5); Annex 2 Note 2, printed page 8 (physical PDF page 10)",
            "fragments": ("160.4", "161.7", "160.7", "1/17 초과", "1/17 이상"),
        },
        "ffmpeg-photosensitivity-global-repair": {
            "source_ids": {"ffmpeg-photosensitivity-601d9ee"},
            "locator": "GRID_SIZE and convert_frame_partial lines 29 to 136; get_badness lines 178 to 192; filter_frame lines 205 to 299",
            "fragments": ("8 × 8", "threshold 이상", "전체 output frame", "복제"),
        },
    }
    for claim_id, expected in expected_claims.items():
        claim = claims_by_id[claim_id]
        assert claim["status"] == "confirmed"
        assert set(claim["source_ids"]) == expected["source_ids"]
        assert claim["locator"] == expected["locator"]
        for fragment in expected["fragments"]:
            assert fragment in claim["statement"]
        assert claim["implementation_consequence"].strip()
