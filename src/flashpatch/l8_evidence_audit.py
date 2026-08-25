"""Reopen and seal deadline-critical L8 competition evidence.

This module does not score candidates.  It checks official-source hashes,
recomputes roster coverage, classifies already sealed blind finals, and extracts
only claims that are present in frozen 2026 official sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
OPERATING_RULES_SHA256 = "5c129ed9f389ecc04b6f7ba8b97f719a313efaf32aea9178e635500023ae1da1"
REPORT_TEMPLATE_SHA256 = "9a5d2968d48ff8a8fd85ce991dc72dc2b0818d7e8c06ebb871cc97ce5cc62d95"
ROSTER = Path("evidence/competition/contest-award-roster-2026-08-11.json")
CENSUS = Path("evidence/competition/contest-coverage-census-2026-08-11.json")
SYMMETRIC = Path("evidence/competition/contest-symmetric-audit-2026-08-11.json")
L8_BASE = Path("evidence/competition/l8-contest-source-locked-2026-08-11")
RESULTS = {
    "source_locked_final_v1": L8_BASE / "final-run-20260811-source-locked-v1",
    "source_locked_final_v2": L8_BASE / "final-run-20260811-source-locked-v2",
    "parity_reconstructed_final_v2": L8_BASE
    / "parity-reconstructed-final-run-20260811-v2",
}


class L8EvidenceAuditError(ValueError):
    """Raised when evidence cannot be reopened exactly."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L8EvidenceAuditError(f"cannot read JSON evidence: {path}") from exc


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise L8EvidenceAuditError(f"{label} must be a lowercase SHA-256")
    return value


def audit_roster(roster: Any, *, source_hashes: dict[str, str]) -> dict[str, Any]:
    if not isinstance(roster, dict) or roster.get("schema") != "flashpatch-contest-award-roster-v1":
        raise L8EvidenceAuditError("official award roster schema is invalid")
    recipients = roster.get("award_recipients")
    if not isinstance(recipients, list) or not recipients:
        raise L8EvidenceAuditError("official award roster is empty")
    expected_hashes = {
        "2024": roster.get("official_sources", {}).get("2024_ebook_config_sha256"),
        "2025": roster.get("official_sources", {}).get("2025_ebook_config_sha256"),
    }
    for year in ("2024", "2025"):
        observed = _require_sha256(source_hashes.get(year), f"{year} official roster source hash")
        expected = _require_sha256(expected_hashes[year], f"stored {year} official roster source hash")
        if observed != expected:
            raise L8EvidenceAuditError(f"{year} official roster source hash mismatch")
    ids = [item.get("id") for item in recipients if isinstance(item, dict)]
    if len(ids) != len(recipients) or any(not isinstance(item, str) or not item for item in ids):
        raise L8EvidenceAuditError("official roster contains an invalid recipient id")
    if len(set(ids)) != len(ids):
        raise L8EvidenceAuditError("official roster contains duplicate recipient ids")
    year_counts = Counter(str(item.get("year")) for item in recipients)
    status_counts = Counter(item.get("source_status") for item in recipients)
    computed = {
        "recipient_count": len(recipients),
        "source_access_count": status_counts["SOURCE_ACCESS"],
        "unscorable_org_only_count": status_counts["UNSCORABLE_ORG_ONLY_SOURCE"],
    }
    if computed != roster.get("summary"):
        raise L8EvidenceAuditError("official roster summary does not match recipient rows")
    if year_counts != Counter({"2024": 19, "2025": 22}):
        raise L8EvidenceAuditError("official roster year coverage is not 19 plus 22")
    if set(status_counts) != {"SOURCE_ACCESS", "UNSCORABLE_ORG_ONLY_SOURCE"}:
        raise L8EvidenceAuditError("official roster contains an unknown source status")
    return {
        "status": "VERIFIED",
        "recipient_count": len(recipients),
        "year_counts": {"2024": 19, "2025": 22},
        "source_access_count": status_counts["SOURCE_ACCESS"],
        "unscorable_org_only_count": status_counts["UNSCORABLE_ORG_ONLY_SOURCE"],
        "official_source_hashes": {"2024": source_hashes["2024"], "2025": source_hashes["2025"]},
    }


def audit_final_result(
    aggregate: Any, revealed: Any, *, expected_names: set[str]
) -> dict[str, Any]:
    if not isinstance(aggregate, dict) or not isinstance(revealed, dict):
        raise L8EvidenceAuditError("blind final artifacts must be JSON objects")
    if aggregate.get("state") != "SEALED" or revealed.get("state") != "REVEALED_AFTER_SEAL":
        raise L8EvidenceAuditError("blind final is not sealed before reveal")
    aggregate_candidates = aggregate.get("candidates")
    revealed_candidates = revealed.get("candidates")
    if not isinstance(aggregate_candidates, list) or not isinstance(revealed_candidates, list):
        raise L8EvidenceAuditError("blind final candidate rows are missing")
    by_blind = {
        item.get("blind_id"): item
        for item in aggregate_candidates
        if isinstance(item, dict) and isinstance(item.get("blind_id"), str)
    }
    names: dict[str, str] = {}
    scores: dict[str, float] = {}
    for item in revealed_candidates:
        if not isinstance(item, dict) or item.get("blind_id") not in by_blind:
            raise L8EvidenceAuditError("revealed candidate does not match aggregate")
        identity = item.get("identity")
        name = identity.get("name") if isinstance(identity, dict) else None
        if not isinstance(name, str) or not name:
            raise L8EvidenceAuditError("revealed candidate identity is invalid")
        blind_id = item["blind_id"]
        if item.get("corrected_total_median") != by_blind[blind_id].get("corrected_total_median"):
            raise L8EvidenceAuditError("revealed score does not match aggregate")
        names[blind_id] = name
        scores[name] = float(item["corrected_total_median"])
    if set(names.values()) != expected_names:
        raise L8EvidenceAuditError("blind final candidate coverage changed")
    if len(scores) != len(expected_names):
        raise L8EvidenceAuditError("blind final contains duplicate candidate names")
    aggregate_warnings = aggregate.get("warnings")
    revealed_warnings = revealed.get("warnings")
    if not isinstance(aggregate_warnings, list) or aggregate_warnings != revealed_warnings:
        raise L8EvidenceAuditError("aggregate and reveal warnings differ")
    warning_codes = sorted(
        {
            item.get("code")
            for item in aggregate_warnings
            if isinstance(item, dict) and isinstance(item.get("code"), str)
        }
    )
    if len(warning_codes) != len(aggregate_warnings):
        raise L8EvidenceAuditError("blind final warning schema is invalid or duplicated")
    orientation = aggregate.get("orientation")
    if warning_codes:
        return {
            "status": "INVALID_FOR_FINAL_DECISION",
            "winner": None,
            "scores": dict(sorted(scores.items())),
            "orientation": orientation,
            "orientation_warnings": warning_codes,
            "external_claim_authorized": False,
        }
    if not isinstance(orientation, dict) or orientation.get("measured") is not True:
        raise L8EvidenceAuditError("warning-free final lacks measured orientation")
    if orientation.get("decisive_count") != 2 or orientation.get("left_position_win_rate") != 0.5:
        raise L8EvidenceAuditError("warning-free final is not balanced A/B and B/A")
    winner = revealed.get("winner")
    winner_identity = winner.get("identity") if isinstance(winner, dict) else None
    winner_name = winner_identity.get("name") if isinstance(winner_identity, dict) else None
    if winner_name not in expected_names or winner.get("blind_id") != aggregate.get("winner_blind_id"):
        raise L8EvidenceAuditError("blind final winner does not match sealed aggregate")
    return {
        "status": "BOUNDED_INTERNAL_RESULT",
        "winner": winner_name,
        "scores": dict(sorted(scores.items())),
        "orientation": {
            "measured": True,
            "decisive_count": 2,
            "left_position_win_rate": 0.5,
        },
        "orientation_warnings": [],
        "external_claim_authorized": False,
    }


def parse_2026_notice(text: str, *, source_sha256: str) -> dict[str, Any]:
    _require_sha256(source_sha256, "2026 public notice source hash")
    if not isinstance(text, str) or not text.strip():
        raise L8EvidenceAuditError("2026 public notice text is empty")
    normalized = re.sub(r"\s+", " ", text)
    if re.search(r"1차 평가\s*\(서면,\s*30점\)", normalized) is None or re.search(
        r"2차 평가\s*\(발표,\s*70점\)", normalized
    ) is None:
        raise L8EvidenceAuditError("2026 public notice does not bind the 30/70 evaluation weights")
    required = ("8.27", "개발보고서", "시연영상", "3분 이내", "소스코드")
    if any(token not in normalized for token in required):
        raise L8EvidenceAuditError("2026 public notice is missing a required submission artifact")
    warnings: list[str] = []
    deadline_time = "18:00" if "18:00" in normalized else None
    if deadline_time is None:
        warnings.append("DEADLINE_TIME_NOT_PRESENT_IN_PUBLIC_NOTICE")
    warnings.append("ORIENTATION_DETAIL_NOT_PRESENT_IN_PUBLIC_NOTICE")
    return {
        "status": "VERIFIED_PUBLIC_NOTICE",
        "source_sha256": source_sha256,
        "evaluation_weights": {"first_written": 30, "second_presentation": 70},
        "submission": {
            "date": "2026-08-27",
            "deadline_time": deadline_time,
            "artifacts": ["development_report", "demo_video_max_3_minutes", "source_code"],
        },
        "orientation_public_notice": {
            "date": "2026-07-23",
            "evaluation_criteria_announced": True,
            "detailed_material_reopened": False,
        },
        "warnings": warnings,
    }


def parse_2026_operating_rules(text: str, *, source_sha256: str) -> dict[str, Any]:
    _require_sha256(source_sha256, "2026 operating-rules source hash")
    if not isinstance(text, str) or not text.strip():
        raise L8EvidenceAuditError("2026 operating rules text is empty")
    compact = re.sub(r"\s+", "", text)
    required = {
        "osi_license_for_authored_source": ("직접작성한소스코드", "OSI인증"),
        "complete_source_submitted_and_verifiable": (
            "전체소스코드를포함하여제출",
            "심사및검증이가능한상태",
        ),
        "public_repository_required": ("공개저장소에게시",),
        "award_repository_public_years": ("수상일로부터5년간", "공개(Public)상태"),
        "commercial_ai_coding_assistance_allowed": (
            "코드작성및디버깅보조용도로상용AI서비스",
            "허용한다",
        ),
        "ai_code_understanding_may_affect_score": (
            "AI가작성한코드의동작원리에대한이해도가부족",
            "평가시감점될수있다",
        ),
        "detailed_scoring_delegated_to_notice_or_judging_plan": (
            "세부적인평가방식",
            "배점기준",
            "모집공고또는심사계획에따른다",
        ),
    }
    missing = [label for label, tokens in required.items() if any(token not in compact for token in tokens)]
    if missing:
        raise L8EvidenceAuditError("2026 operating rules are missing obligations: " + ", ".join(missing))
    return {
        "status": "VERIFIED_OPERATING_RULES",
        "source_sha256": source_sha256,
        "obligations": {
            "osi_license_for_authored_source": True,
            "complete_source_submitted_and_verifiable": True,
            "public_repository_required": True,
            "award_repository_public_years": 5,
            "commercial_ai_coding_assistance_allowed": True,
            "ai_code_understanding_may_affect_score": True,
            "detailed_scoring_delegated_to_notice_or_judging_plan": True,
        },
    }


def parse_2026_report_template(text: str, *, source_sha256: str) -> dict[str, Any]:
    _require_sha256(source_sha256, "2026 result-report source hash")
    if not isinstance(text, str) or not text.strip():
        raise L8EvidenceAuditError("2026 result-report template text is empty")
    compact = re.sub(r"\s+", "", text)
    required = (
        "2026.8.27.",
        "18:00",
        "총2개파일필수제출",
        "HWP(HWPX)또는DOC(DOCX)",
        "PDF",
        "유튜브업로드후URL기재",
        "별도파일제출불가",
        "공개저장소(Repository)URL기재",
    )
    if any(token not in compact for token in required):
        raise L8EvidenceAuditError("2026 result-report template is missing a deadline or delivery rule")
    return {
        "status": "VERIFIED_RESULT_REPORT_TEMPLATE",
        "source_sha256": source_sha256,
        "deadline": "2026-08-27T18:00:00+09:00",
        "report_files": ["editable_report", "pdf_report"],
        "demo_video": "youtube_url_only_no_video_file_upload",
        "public_repository_url_required": True,
    }


def _repo_inputs(repo_root: Path) -> dict[str, Any]:
    roster = _load(repo_root / ROSTER)
    census = _load(repo_root / CENSUS)
    symmetric = _load(repo_root / SYMMETRIC)
    if census.get("award_recipient_source_audit", {}).get("official_roster_count") != 41:
        raise L8EvidenceAuditError("coverage census does not bind all 41 official recipients")
    if census.get("award_recipient_audit_status") != "OFFICIAL_2024_2025_ROSTER_SYMMETRIC_AUDIT_COMPLETE":
        raise L8EvidenceAuditError("coverage census does not close the symmetric roster audit")
    current_entries = census.get("current_2026_public_entries")
    if not isinstance(current_entries, dict) or current_entries.get("candidate_count") != 0:
        raise L8EvidenceAuditError("stored 2026 official-registry census changed")
    candidates = symmetric.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 7:
        raise L8EvidenceAuditError("current-source symmetric audit must contain seven candidates")
    roster_ids = [item.get("roster_id") for item in candidates if isinstance(item, dict)]
    if len(set(roster_ids)) != 7 or "2025-02-ttakkeun-hot-updater" not in roster_ids:
        raise L8EvidenceAuditError("current-source symmetric audit coverage is invalid")
    return {"roster": roster, "census": census, "symmetric": symmetric}


def _build_receipt(
    *, repo_root: Path, roster_2024_index: Path, roster_2025_index: Path, notice_pdf: Path, notice_text: Path
) -> dict[str, Any]:
    repo = _repo_inputs(repo_root)
    source_hashes = {"2024": _file_hash(roster_2024_index), "2025": _file_hash(roster_2025_index)}
    roster_audit = audit_roster(repo["roster"], source_hashes=source_hashes)
    notice_pdf_bytes = notice_pdf.read_bytes()
    if not notice_pdf_bytes.startswith(b"%PDF-"):
        raise L8EvidenceAuditError("2026 public notice snapshot is not a PDF")
    notice_text_value = notice_text.read_text(encoding="utf-8")
    rules = parse_2026_notice(notice_text_value, source_sha256=_sha256_bytes(notice_pdf_bytes))
    results: dict[str, Any] = {}
    commitments: dict[str, str] = {
        ROSTER.as_posix(): _file_hash(repo_root / ROSTER),
        CENSUS.as_posix(): _file_hash(repo_root / CENSUS),
        SYMMETRIC.as_posix(): _file_hash(repo_root / SYMMETRIC),
    }
    for label, directory in RESULTS.items():
        aggregate_path = repo_root / directory / "aggregate.json"
        revealed_path = repo_root / directory / "revealed.json"
        results[label] = audit_final_result(
            _load(aggregate_path), _load(revealed_path), expected_names={"Hot Updater", "FlashPatch"}
        )
        commitments[directory.joinpath("aggregate.json").as_posix()] = _file_hash(aggregate_path)
        commitments[directory.joinpath("revealed.json").as_posix()] = _file_hash(revealed_path)
    return {
        "schema": "flashpatch-l8-deadline-evidence-audit-v1",
        "state": "VERIFIED_WITH_BOUNDED_CLAIMS",
        "roster": roster_audit,
        "current_source_audit": {
            "status": "VERIFIED_STORED_CUTOFF",
            "as_of": repo["census"].get("as_of"),
            "symmetric_candidate_count": 7,
            "official_2026_registry_candidate_count_at_cutoff": 0,
            "external_claim_authorized": False,
        },
        "blind_results": results,
        "rules_2026": rules,
        "source_snapshots": {
            "roster_2024_index": {"path": "sources/2024-book_config.js", "sha256": source_hashes["2024"]},
            "roster_2025_index": {"path": "sources/2025-book_config.js", "sha256": source_hashes["2025"]},
            "official_notice_pdf": {"path": "sources/2026-public-notice.pdf", "sha256": _file_hash(notice_pdf)},
            "official_notice_text": {"path": "sources/2026-public-notice.txt", "sha256": _file_hash(notice_text)},
        },
        "repository_input_sha256": dict(sorted(commitments.items())),
        "claim_boundary": {
            "external_judges": False,
            "contest_superiority": False,
            "independent_human_judgment": False,
            "public_notice_deadline_time_verified": rules["submission"]["deadline_time"] is not None,
            "orientation_detail_reopened": False,
        },
    }


def create_audit(
    *, repo_root: Path, roster_2024_index: Path, roster_2025_index: Path, notice_pdf: Path,
    notice_text: Path, out: Path
) -> dict[str, Any]:
    if out.exists():
        raise L8EvidenceAuditError("audit output already exists")
    receipt = _build_receipt(
        repo_root=repo_root,
        roster_2024_index=roster_2024_index,
        roster_2025_index=roster_2025_index,
        notice_pdf=notice_pdf,
        notice_text=notice_text,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        sources = temporary / "sources"
        sources.mkdir()
        shutil.copyfile(roster_2024_index, sources / "2024-book_config.js")
        shutil.copyfile(roster_2025_index, sources / "2025-book_config.js")
        shutil.copyfile(notice_pdf, sources / "2026-public-notice.pdf")
        shutil.copyfile(notice_text, sources / "2026-public-notice.txt")
        (temporary / "receipt.json").write_bytes(_canonical(receipt))
        seal = {
            "schema": "flashpatch-l8-deadline-evidence-audit-seal-v1",
            "receipt_sha256": _file_hash(temporary / "receipt.json"),
        }
        (temporary / "receipt.seal.json").write_bytes(_canonical(seal))
        temporary.rename(out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def verify_audit(
    *, repo_root: Path, audit_dir: Path, raw_source_dir: Path | None = None
) -> dict[str, Any]:
    receipt_path = audit_dir / "receipt.json"
    receipt = _load(receipt_path)
    seal = _load(audit_dir / "receipt.seal.json")
    if seal != {
        "schema": "flashpatch-l8-deadline-evidence-audit-seal-v1",
        "receipt_sha256": _file_hash(receipt_path),
    }:
        raise L8EvidenceAuditError("deadline audit receipt seal mismatch")
    raw_sources = raw_source_dir or audit_dir / "sources"
    expected = _build_receipt(
        repo_root=repo_root,
        roster_2024_index=audit_dir / "sources/2024-book_config.js",
        roster_2025_index=audit_dir / "sources/2025-book_config.js",
        notice_pdf=raw_sources / "2026-public-notice.pdf",
        notice_text=audit_dir / "sources/2026-public-notice.txt",
    )
    if receipt != expected:
        raise L8EvidenceAuditError("deadline audit receipt does not match reopened evidence")
    return receipt


def _build_rules_supplement(
    *, operating_rules_pdf: Path, operating_rules_text: Path, report_template_zip: Path,
    report_template_text: Path
) -> dict[str, Any]:
    rules_bytes = operating_rules_pdf.read_bytes()
    report_bytes = report_template_zip.read_bytes()
    if not rules_bytes.startswith(b"%PDF-"):
        raise L8EvidenceAuditError("operating-rules snapshot is not a PDF")
    if not report_bytes.startswith(b"PK"):
        raise L8EvidenceAuditError("result-report snapshot is not a ZIP")
    rules_hash = _sha256_bytes(rules_bytes)
    report_hash = _sha256_bytes(report_bytes)
    if rules_hash != OPERATING_RULES_SHA256:
        raise L8EvidenceAuditError("official operating-rules source hash mismatch")
    if report_hash != REPORT_TEMPLATE_SHA256:
        raise L8EvidenceAuditError("official result-report source hash mismatch")
    return {
        "schema": "flashpatch-l8-official-rules-supplement-v1",
        "state": "VERIFIED_OFFICIAL_RULES_AND_DEADLINE",
        "operating_rules": parse_2026_operating_rules(
            operating_rules_text.read_text(encoding="utf-8"), source_sha256=rules_hash
        ),
        "result_report_template": parse_2026_report_template(
            report_template_text.read_text(encoding="utf-8"), source_sha256=report_hash
        ),
        "source_snapshots": {
            "operating_rules_pdf": {
                "path": "sources/2026-operating-rules.pdf",
                "sha256": rules_hash,
            },
            "operating_rules_text": {
                "path": "sources/2026-operating-rules.txt",
                "sha256": _file_hash(operating_rules_text),
            },
            "result_report_zip": {
                "path": "sources/2026-result-report-template.zip",
                "sha256": report_hash,
            },
            "result_report_text": {
                "path": "sources/2026-result-report-template.txt",
                "sha256": _file_hash(report_template_text),
            },
        },
        "claim_boundary": {
            "detailed_axis_weights_reopened": False,
            "orientation_materials_reopened": False,
            "deadline_reopened": True,
        },
    }


def create_rules_supplement(
    *, audit_dir: Path, operating_rules_pdf: Path, operating_rules_text: Path,
    report_template_zip: Path, report_template_text: Path
) -> dict[str, Any]:
    if not audit_dir.is_dir() or not (audit_dir / "receipt.json").is_file():
        raise L8EvidenceAuditError("base deadline audit is missing")
    destination = audit_dir / "rules-supplement.json"
    seal_path = audit_dir / "rules-supplement.seal.json"
    if destination.exists() or seal_path.exists():
        raise L8EvidenceAuditError("rules supplement already exists")
    supplement = _build_rules_supplement(
        operating_rules_pdf=operating_rules_pdf,
        operating_rules_text=operating_rules_text,
        report_template_zip=report_template_zip,
        report_template_text=report_template_text,
    )
    sources = audit_dir / "sources"
    copies = {
        sources / "2026-operating-rules.pdf": operating_rules_pdf,
        sources / "2026-operating-rules.txt": operating_rules_text,
        sources / "2026-result-report-template.zip": report_template_zip,
        sources / "2026-result-report-template.txt": report_template_text,
    }
    if any(path.exists() for path in copies):
        raise L8EvidenceAuditError("rules supplement source snapshot already exists")
    for target, source in copies.items():
        shutil.copyfile(source, target)
    destination.write_bytes(_canonical(supplement))
    seal_path.write_bytes(
        _canonical(
            {
                "schema": "flashpatch-l8-official-rules-supplement-seal-v1",
                "supplement_sha256": _file_hash(destination),
            }
        )
    )
    return supplement


def verify_rules_supplement(
    *, audit_dir: Path, raw_source_dir: Path | None = None
) -> dict[str, Any]:
    supplement_path = audit_dir / "rules-supplement.json"
    supplement = _load(supplement_path)
    seal = _load(audit_dir / "rules-supplement.seal.json")
    if seal != {
        "schema": "flashpatch-l8-official-rules-supplement-seal-v1",
        "supplement_sha256": _file_hash(supplement_path),
    }:
        raise L8EvidenceAuditError("rules supplement seal mismatch")
    raw_sources = raw_source_dir or audit_dir / "sources"
    expected = _build_rules_supplement(
        operating_rules_pdf=raw_sources / "2026-operating-rules.pdf",
        operating_rules_text=audit_dir / "sources/2026-operating-rules.txt",
        report_template_zip=raw_sources / "2026-result-report-template.zip",
        report_template_text=audit_dir / "sources/2026-result-report-template.txt",
    )
    if supplement != expected:
        raise L8EvidenceAuditError("rules supplement does not match reopened sources")
    return supplement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashpatch.l8_evidence_audit")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repo-root", type=Path, default=Path.cwd())
    create.add_argument("--roster-2024-index", type=Path, required=True)
    create.add_argument("--roster-2025-index", type=Path, required=True)
    create.add_argument("--official-notice-pdf", type=Path, required=True)
    create.add_argument("--official-notice-text", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--audit-dir", type=Path, required=True)
    verify.add_argument("--raw-source-dir", type=Path)
    supplement = commands.add_parser("supplement-rules")
    supplement.add_argument("--audit-dir", type=Path, required=True)
    supplement.add_argument("--operating-rules-pdf", type=Path, required=True)
    supplement.add_argument("--operating-rules-text", type=Path, required=True)
    supplement.add_argument("--result-report-zip", type=Path, required=True)
    supplement.add_argument("--result-report-text", type=Path, required=True)
    verify_supplement = commands.add_parser("verify-rules")
    verify_supplement.add_argument("--audit-dir", type=Path, required=True)
    verify_supplement.add_argument("--raw-source-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            receipt = create_audit(
                repo_root=args.repo_root,
                roster_2024_index=args.roster_2024_index,
                roster_2025_index=args.roster_2025_index,
                notice_pdf=args.official_notice_pdf,
                notice_text=args.official_notice_text,
                out=args.out,
            )
        elif args.command == "verify":
            receipt = verify_audit(
                repo_root=args.repo_root,
                audit_dir=args.audit_dir,
                raw_source_dir=args.raw_source_dir,
            )
        elif args.command == "supplement-rules":
            supplement_receipt = create_rules_supplement(
                audit_dir=args.audit_dir,
                operating_rules_pdf=args.operating_rules_pdf,
                operating_rules_text=args.operating_rules_text,
                report_template_zip=args.result_report_zip,
                report_template_text=args.result_report_text,
            )
            print(supplement_receipt["state"])
            return 0
        else:
            supplement_receipt = verify_rules_supplement(
                audit_dir=args.audit_dir,
                raw_source_dir=args.raw_source_dir,
            )
            print(supplement_receipt["state"])
            return 0
    except (L8EvidenceAuditError, OSError, UnicodeDecodeError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1
    print(
        f"{receipt['state']} roster={receipt['roster']['recipient_count']} "
        f"source_locked={receipt['blind_results']['source_locked_final_v2']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
