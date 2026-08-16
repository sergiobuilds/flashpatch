"""Fail-closed L7 natural-case ledger verification.

This is deliberately a *qualification* verifier, not a scoring path.  A case
can enter neither a detector denominator nor a blind league from this module:
even a fully coherent natural renderer bundle returns ``NOT_SCOREABLE`` until
independent gold exists in the separately-authorized G3 lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .public_godot import (
    classify_native_main_capture_qualification,
    materialize_native_main_capture_qualification,
)
from .renderer_artifact import RendererArtifactError, open_renderer_artifact, renderer_rgb_sha256


NATURAL_CASE_SCHEMA = "flashpatch-l7-natural-case-ledger-v1"
NATIVE_MAIN_NATURAL_CASE_SCHEMA = "flashpatch-l7-native-main-natural-case-ledger-v1"
NATIVE_MAIN_QUALIFICATION_SUMMARY_SCHEMA = "flashpatch-l7-native-main-qualification-summary-v1"
ASSESSMENT_SCHEMA = "flashpatch-l7-natural-case-assessment-v1"
CORPUS_DISCOVERY_SCHEMA = "flashpatch-l7-corpus-discovery-v1"
CORPUS_DISCOVERY_SCHEMA_V2 = "flashpatch-l7-corpus-discovery-v2"
CORPUS_DISCOVERY_SCHEMA_V3 = "flashpatch-l7-corpus-discovery-v3"
CORPUS_DISCOVERY_SCHEMA_V4 = "flashpatch-l7-corpus-discovery-v4"
CORPUS_DISCOVERY_SCHEMA_V5 = "flashpatch-l7-corpus-discovery-v5"
CORPUS_DISCOVERY_ASSESSMENT_SCHEMA = "flashpatch-l7-corpus-discovery-assessment-v1"
L7_NATIVE_MAX_WALL_CLOCK_SECONDS = 120
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_GODOT_VERSION = re.compile(r"^[34]\.\d+(?:[.-][0-9A-Za-z]+){0,6}$")
_DISCOVERY_OUTCOMES = frozenset({"prospect", "qualified", "excluded"})
_FORBIDDEN_DISCOVERY_KEYS = re.compile(
    r"(?:^|_)(?:hazard|risk_label|detector|detection|gold|ground_truth|"
    r"score|scores|scoreable|ranking|rank|winner|winning)(?:_|$)",
    re.IGNORECASE,
)
_FORBIDDEN_DISCOVERY_WORDS = frozenset(
    {
        "detected",
        "detection",
        "detector",
        "f1",
        "flash",
        "flashes",
        "flashing",
        "gold",
        "hazard",
        "hazardous",
        "negative",
        "outperform",
        "outperforms",
        "photosensitive",
        "photosensitivity",
        "positive",
        "precision",
        "rank",
        "ranked",
        "ranking",
        "recall",
        "risk",
        "safe",
        "seizure",
        "score",
        "scored",
        "scores",
        "scoreable",
        "superior",
        "unsafe",
        "victory",
        "wins",
        "winner",
        "winning",
    }
)


class L7VerificationFailure(ValueError):
    """A natural-case bundle cannot be used as L7 evidence."""

    def __init__(self, gate: str, reason: str) -> None:
        super().__init__(f"FAIL_CLOSED:{gate}:{reason}")
        self.gate = gate
        self.reason = reason

    @property
    def diagnostic(self) -> str:
        return f"FAIL_CLOSED:{self.gate}:{self.reason}"


def _fail(gate: str, reason: str) -> NoReturn:
    raise L7VerificationFailure(gate, reason)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _case_root(path: Path | str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        _fail("bundle", "case_root_invalid")
    try:
        root = supplied.resolve(strict=True)
    except OSError:
        _fail("bundle", "case_root_missing")
    if not root.is_dir():
        _fail("bundle", "case_root_invalid")
    return root


def _owned_file(root: Path, value: object, *, gate: str, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        _fail(gate, f"{field}_path_invalid")
    candidate = root / value
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _fail(gate, f"{field}_path_invalid")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            _fail(gate, f"{field}_path_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail(gate, f"{field}_path_invalid")
    if not resolved.is_file():
        _fail(gate, f"{field}_missing")
    return resolved


def _owned_directory(root: Path, value: object, *, gate: str, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        _fail(gate, f"{field}_path_invalid")
    candidate = root / value
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _fail(gate, f"{field}_path_invalid")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            _fail(gate, f"{field}_path_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        _fail(gate, f"{field}_path_invalid")
    if not resolved.is_dir():
        _fail(gate, f"{field}_missing")
    return resolved


def _read_json(path: Path, *, gate: str, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(gate, reason)
    if not isinstance(value, dict):
        _fail(gate, reason)
    return value


def _read_json_without_duplicate_keys(path: Path, *, gate: str, reason: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(gate, "duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)
    except L7VerificationFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(gate, reason)
    if not isinstance(value, dict):
        _fail(gate, reason)
    return value


def _require_hash_bound_file(
    root: Path, value: object, expected: object, *, gate: str, field: str
) -> Path:
    if not _is_hex64(expected):
        _fail(gate, f"{field}_hash_invalid")
    path = _owned_file(root, value, gate=gate, field=field)
    if _sha256_file(path) != expected:
        _fail(gate, f"{field}_hash_mismatch")
    return path


def _natural_case_identity(ledger: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    case_id = ledger.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        _fail("case", "case_id_invalid")
    if ledger.get("case_class") != "natural_public_godot":
        _fail("case_class", "not_natural_public_godot")
    if ledger.get("controlled_mutation") is not False:
        _fail("case_class", "controlled_not_natural")
    repository = ledger.get("repository")
    if not isinstance(repository, dict):
        _fail("provenance", "repository_missing")
    url = repository.get("url")
    revision = repository.get("revision")
    license_name = repository.get("license")
    project_subpath = repository.get("project_subpath")
    if (
        not isinstance(url, str)
        or not url.startswith("https://")
        or not _GIT_REVISION.fullmatch(str(revision))
        or not isinstance(license_name, str)
        or not license_name
        or not isinstance(project_subpath, str)
        or not project_subpath
        or Path(project_subpath).is_absolute()
        or ".." in Path(project_subpath).parts
    ):
        _fail("provenance", "repository_identity_invalid")
    return case_id, repository


def _normalized_repository_url(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    parsed = urlsplit(normalized)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.rstrip("/"), "", ""))


def _is_canonical_https_repository_url(value: object) -> bool:
    if not isinstance(value, str) or value != value.strip():
        return False
    parsed = urlsplit(value)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path.strip("/")
    )


def _source_tree_sha256(checkout: Path) -> str:
    """Hash every regular tracked source file with its project-relative name."""

    digest = hashlib.sha256()
    for path in sorted(checkout.rglob("*")):
        relative = path.relative_to(checkout)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            _fail("provenance", "source_checkout_symlink")
        if not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_source_provenance(root: Path, ledger: dict[str, Any], repository: dict[str, Any]) -> None:
    source = ledger.get("source_provenance")
    if not isinstance(source, dict):
        _fail("provenance", "source_provenance_missing")
    source_path = _require_hash_bound_file(
        root, source.get("path"), source.get("sha256"), gate="provenance", field="source_provenance"
    )
    evidence = _read_json(source_path, gate="provenance", reason="source_provenance_invalid")
    required = {
        "repository_url": repository["url"],
        "revision": repository["revision"],
        "license": repository["license"],
        "project_subpath": repository["project_subpath"],
        "original_unmutated": True,
    }
    if any(evidence.get(field) != expected for field, expected in required.items()):
        _fail("provenance", "source_provenance_mismatch")
    checkout = _owned_directory(
        root, evidence.get("source_checkout_path"), gate="provenance", field="source_checkout"
    )
    results = [
        subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, check=False)
        for args in (
            ("rev-parse", "HEAD"),
            ("remote", "get-url", "origin"),
            ("status", "--porcelain", "--untracked-files=no"),
        )
    ]
    if any(result.returncode != 0 for result in results):
        _fail("provenance", "source_checkout_unverifiable")
    head, origin, status = (result.stdout.strip() for result in results)
    if (
        head != repository["revision"]
        or _normalized_repository_url(origin) != _normalized_repository_url(repository["url"])
        or status
    ):
        _fail("provenance", "source_checkout_mismatch")
    license_path = _owned_file(root, evidence.get("license_path"), gate="provenance", field="license")
    if license_path.parent != checkout:
        _fail("provenance", "license_not_in_source_checkout")
    if _sha256_file(license_path) != evidence.get("license_sha256"):
        _fail("provenance", "license_hash_mismatch")
    if _source_tree_sha256(checkout) != evidence.get("source_tree_sha256"):
        _fail("provenance", "source_tree_hash_mismatch")


def _verify_timestamps(path: Path, *, expected_count: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("timestamps", "invalid")
    if not isinstance(payload, list) or len(payload) != expected_count:
        _fail("timestamps", "count_mismatch")
    values: list[float] = []
    for item in payload:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            _fail("timestamps", "invalid")
        values.append(float(item))
    if any(right <= left for left, right in zip(values, values[1:])):
        _fail("timestamps", "not_strictly_monotonic")


def _verify_renderer_receipt(
    root: Path, ledger: dict[str, Any], repository: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    renderer = ledger.get("renderer")
    if not isinstance(renderer, dict):
        _fail("renderer", "missing")
    frame_count = renderer.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        _fail("renderer", "frame_count_invalid")
    rgb_path = _require_hash_bound_file(
        root, renderer.get("rgb_path"), renderer.get("rgb_sha256"), gate="renderer", field="rgb"
    )
    if rgb_path.stat().st_size == 0:
        _fail("renderer", "rgb_empty")
    timestamps_path = _require_hash_bound_file(
        root,
        renderer.get("timestamps_path"),
        renderer.get("timestamps_sha256"),
        gate="timestamps",
        field="timestamps",
    )
    _verify_timestamps(timestamps_path, expected_count=frame_count)
    receipt_path = _require_hash_bound_file(
        root,
        renderer.get("execution_receipt_path"),
        renderer.get("execution_receipt_sha256"),
        gate="renderer",
        field="execution_receipt",
    )
    receipt = _read_json(receipt_path, gate="renderer", reason="execution_receipt_invalid")
    if receipt.get("schema") != "flashpatch-renderer-engine-receipt-v1":
        _fail("renderer", "execution_receipt_schema_invalid")
    if receipt.get("controlled_mutation") is not False:
        _fail("case_class", "controlled_not_natural")
    upstream = receipt.get("upstream")
    if not isinstance(upstream, dict) or any(
        upstream.get(field) != expected
        for field, expected in {
            "repository_url": repository["url"],
            "source_revision": repository["revision"],
            "license": repository["license"],
            "project_path": repository["project_subpath"],
        }.items()
    ):
        _fail("provenance", "renderer_receipt_mismatch")
    factual = receipt.get("factual_replay")
    if not isinstance(factual, dict):
        _fail("renderer", "factual_replay_missing")
    expected_values = {
        "renderer_rgb_raw_sha256": renderer["rgb_sha256"],
        "timestamps_sha256": renderer["timestamps_sha256"],
        "frame_count": frame_count,
    }
    if any(factual.get(field) != expected for field, expected in expected_values.items()):
        _fail("renderer", "receipt_artifact_mismatch")
    return receipt_path, receipt, renderer


def _verify_trace(root: Path, ledger: dict[str, Any], receipt: dict[str, Any]) -> None:
    trace = ledger.get("trace")
    if not isinstance(trace, dict):
        _fail("trace", "missing")
    path = _require_hash_bound_file(root, trace.get("path"), trace.get("sha256"), gate="trace", field="trace")
    if path.stat().st_size == 0:
        _fail("trace", "empty")
    capture = receipt.get("factual_replay", {}).get("renderer_capture")
    if not isinstance(capture, dict) or capture.get("trace_sha256") != f"sha256:{trace['sha256']}":
        _fail("trace", "renderer_receipt_mismatch")


def verify_natural_case_bundle(case_root: Path | str) -> dict[str, object]:
    """Reopen one natural renderer bundle and qualify it only up to G2.

    A successful G2 receipt intentionally stops at ``NOT_SCOREABLE``.  G3 gold
    and all comparator evidence are outside this verifier's authority.
    """

    root = _case_root(case_root)
    ledger_path = root / "natural-case.json"
    if ledger_path.is_symlink() or not ledger_path.is_file():
        _fail("bundle", "natural_case_ledger_missing")
    ledger = _read_json(ledger_path, gate="bundle", reason="natural_case_ledger_invalid")
    if ledger.get("schema") != NATURAL_CASE_SCHEMA:
        _fail("bundle", "natural_case_ledger_schema_invalid")
    case_id, repository = _natural_case_identity(ledger)
    _verify_source_provenance(root, ledger, repository)
    receipt_path, receipt, renderer = _verify_renderer_receipt(root, ledger, repository)
    _verify_trace(root, ledger, receipt)
    return {
        "schema": ASSESSMENT_SCHEMA,
        "case_id": case_id,
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "reason": "independent_gold_missing",
        "case_class": "natural_public_godot",
        "ledger_sha256": _sha256_file(ledger_path),
        "renderer_execution_receipt_sha256": _sha256_file(receipt_path),
        "renderer_rgb_sha256": renderer["rgb_sha256"],
        "timestamps_sha256": renderer["timestamps_sha256"],
        "external_claim_authorized": False,
    }


def _native_main_qualification_tree_sha256(root: Path) -> str:
    """Hash a qualification copy as its upstream project, not its probe output."""
    replacement = root / ".flashpatch" / "upstream-project.godot"
    if replacement.is_symlink() or not replacement.is_file():
        _fail("qualification", "upstream_project_config_missing")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in {".flashpatch", ".godot"} or path.is_symlink():
            continue
        if path.name.endswith((".import", ".uid")):
            continue
        if not path.is_file():
            continue
        content = replacement.read_bytes() if relative.as_posix() == "project.godot" else path.read_bytes()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _native_main_project_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", ".godot"} or path.is_symlink() or path.name.endswith((".import", ".uid")):
            continue
        if not path.is_file():
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_native_main_natural_case_bundle(case_root: Path | str) -> dict[str, object]:
    """Reopen native-main qualification evidence without converting its NPZ artifact.

    This is deliberately a G2 qualification-only lane.  It never emits a
    detector label, score, patch, or external claim.
    """
    root = _case_root(case_root)
    ledger_path = root / "native-main-natural-case.json"
    ledger = _read_json_without_duplicate_keys(
        ledger_path, gate="bundle", reason="native_main_ledger_invalid"
    )
    if ledger.get("schema") != NATIVE_MAIN_NATURAL_CASE_SCHEMA:
        _fail("bundle", "native_main_ledger_schema_invalid")
    case_id, repository = _natural_case_identity(ledger)
    _verify_source_provenance(root, ledger, repository)
    source = ledger["source_provenance"]
    assert isinstance(source, dict)
    checkout = _owned_directory(root, _read_json(_owned_file(root, source.get("path"), gate="provenance", field="source_provenance"), gate="provenance", reason="source_provenance_invalid").get("source_checkout_path"), gate="provenance", field="source_checkout")
    project = checkout / str(repository["project_subpath"])
    if not project.is_dir() or project.is_symlink():
        _fail("qualification", "pinned_project_missing")
    native = ledger.get("native_main")
    if not isinstance(native, dict):
        _fail("native_main", "missing")
    candidate_id = native.get("candidate_id")
    manifest_path = _require_hash_bound_file(root, native.get("manifest_path"), native.get("manifest_sha256"), gate="native_main", field="manifest")
    verify_corpus_discovery_manifest(manifest_path)
    manifest = _read_json_without_duplicate_keys(manifest_path, gate="native_main", reason="manifest_invalid")
    candidates = manifest.get("candidates")
    templates = manifest.get("trace_templates")
    if not isinstance(candidate_id, str) or not isinstance(candidates, list) or not isinstance(templates, list):
        _fail("native_main", "manifest_binding_invalid")
    candidate = next((item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == candidate_id), None)
    if candidate is None or any(candidate.get(key) != expected for key, expected in {
        "repository_url": repository["url"], "revision": repository["revision"], "license": repository["license"],
    }.items()):
        _fail("native_main", "candidate_provenance_mismatch")
    template = next((item for item in templates if isinstance(item, dict) and item.get("trace_template_id") == candidate.get("trace_template_id")), None)
    if template is None:
        _fail("native_main", "candidate_template_missing")
    preflight_path = _require_hash_bound_file(root, native.get("preflight_path"), native.get("preflight_sha256"), gate="native_main", field="preflight")
    preflight = _read_json_without_duplicate_keys(preflight_path, gate="native_main", reason="preflight_invalid")
    if any(preflight.get(key) != expected for key, expected in {
        "candidate_id": candidate_id, "repository_url": repository["url"], "revision": repository["revision"],
        "license": repository["license"], "project_subpath": repository["project_subpath"],
        "manifest_sha256": native.get("manifest_sha256"), "renderer_executed": False, "scoreable": False,
    }.items()):
        _fail("native_main", "preflight_mismatch")
    qualification = _owned_directory(root, native.get("qualification_path"), gate="qualification", field="qualification")
    if _native_main_qualification_tree_sha256(qualification) != _native_main_project_tree_sha256(project):
        _fail("qualification", "source_tree_mismatch")
    frame_path = _require_hash_bound_file(root, native.get("frame_artifact_path"), native.get("frame_artifact_sha256"), gate="renderer", field="frame_artifact")
    replay_path = _require_hash_bound_file(root, native.get("replay_path"), native.get("replay_sha256"), gate="native_main", field="replay")
    receipt_path = _require_hash_bound_file(root, native.get("execution_receipt_path"), native.get("execution_receipt_sha256"), gate="native_main", field="execution_receipt")
    trace_path = _require_hash_bound_file(root, native.get("trace_path"), native.get("trace_sha256"), gate="trace", field="trace")
    replay = _read_json_without_duplicate_keys(replay_path, gate="native_main", reason="replay_invalid")
    receipt = _read_json_without_duplicate_keys(receipt_path, gate="native_main", reason="execution_receipt_invalid")
    trace = _read_json_without_duplicate_keys(trace_path, gate="trace", reason="invalid")
    if any(receipt.get(key) != expected for key, expected in {
        "schema": "flashpatch-godot-native-main-capture-v1", "qualification_only": True,
        "scoreable": False, "native_equivalence": "NOT_ESTABLISHED", "controlled_mutation": False,
        "upstream_defect": None, "execution_mode": "instrumented_native_main_scene_capture",
        "frame_artifact_sha256": native["frame_artifact_sha256"], "replay_sha256": native["replay_sha256"],
        "trace_sha256": f"sha256:{native['trace_sha256']}",
    }.items()):
        _fail("native_main", "receipt_contract_mismatch")
    if receipt.get("decision") not in {"SAFE_SCENARIO_READY", "HAZARDOUS_ATTRIBUTION_PENDING"}:
        _fail("native_main", "receipt_decision_invalid")
    if any(replay.get(key) != expected for key, expected in {
        "qualification_only": True, "scoreable": False, "native_equivalence": "NOT_ESTABLISHED",
        "execution_mode": "instrumented_native_main_scene_capture", "frames_npz": frame_path.name,
    }.items()):
        _fail("native_main", "replay_contract_mismatch")
    if trace.get("fixed_fps") != template.get("fixed_fps") or trace.get("capture_frames") != template.get("capture_frames") or trace.get("original_main_scene") != template.get("original_main_scene") or trace.get("actions") != template.get("actions") or trace.get("pointer_events") != template.get("pointer_events") or trace.get("key_events") != template.get("key_events") or trace.get("scenario_readiness") != template.get("scenario_readiness") or trace.get("ui_selection_observations") != template.get("ui_selection_observations"):
        _fail("trace", "manifest_template_mismatch")
    capture = replay.get("renderer_capture")
    if not isinstance(capture, dict) or capture.get("artifact") != frame_path.name:
        _fail("renderer", "replay_capture_mismatch")
    presentation = receipt.get("presentation_timestamps_us")
    actual = capture.get("actual_capture_timestamps_us")
    if not isinstance(presentation, list) or not isinstance(actual, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in presentation + actual) or any(right <= left for left, right in zip(actual, actual[1:])):
        _fail("timestamps", "native_main_invalid")
    try:
        with open_renderer_artifact(frame_path) as artifact:
            if len(artifact.frames) != receipt.get("frame_count") or len(artifact.frames) != len(presentation) or not np.array_equal(artifact.timestamps, np.asarray(presentation, dtype=np.float64) / 1_000_000.0):
                _fail("renderer", "native_main_artifact_mismatch")
            rgb_sha256 = renderer_rgb_sha256(artifact.frames)
            timestamps_sha256 = hashlib.sha256(np.ascontiguousarray(artifact.timestamps).tobytes()).hexdigest()
    except RendererArtifactError:
        _fail("renderer", "native_main_artifact_invalid")
    declared_rgb = native.get("rgb_bytes_sha256")
    declared_timestamps = native.get("timestamps_f64_sha256")
    if not _is_hex64(declared_rgb) or not _is_hex64(declared_timestamps) or rgb_sha256 != declared_rgb or timestamps_sha256 != declared_timestamps:
        _fail("renderer", "native_main_array_hash_mismatch")
    return {
        "schema": ASSESSMENT_SCHEMA,
        "case_id": case_id,
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "reason": "qualification_only_native_equivalence_not_established",
        "score_blockers": [
            "qualification_only_capture",
            "native_equivalence_not_established",
            "independent_gold_missing",
        ],
        "case_class": "natural_public_godot",
        "renderer_execution_receipt_sha256": _sha256_file(receipt_path),
        "renderer_rgb_sha256": rgb_sha256,
        "timestamps_sha256": timestamps_sha256,
        "native_equivalence": "NOT_ESTABLISHED",
        "external_claim_authorized": False,
    }


def _project_root_for_summary(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "docs" / "MASTER-MAP.md").is_file() and (parent / "src" / "flashpatch").is_dir():
            return parent
    _fail("native_main_summary", "project_root_unavailable")


def _summary_bound_file(project_root: Path, value: object, expected_sha256: object, *, field: str) -> Path:
    if not isinstance(value, str) or value.startswith("/") or "\x00" in value:
        _fail("native_main_summary", f"{field}_path_invalid")
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError:
        _fail("native_main_summary", f"{field}_path_escapes_project")
    if path.is_symlink() or not path.is_file():
        _fail("native_main_summary", f"{field}_missing")
    if not _is_hex64(expected_sha256) or _sha256_file(path) != expected_sha256:
        _fail("native_main_summary", f"{field}_sha256_mismatch")
    return path


def verify_native_main_qualification_summary(summary_path: Path | str) -> dict[str, object]:
    """Reopen a batch native-main qualification summary without creating score authority."""

    summary_file = Path(summary_path).resolve(strict=True)
    project_root = _project_root_for_summary(summary_file)
    summary = _read_json_without_duplicate_keys(
        summary_file, gate="native_main_summary", reason="invalid_json"
    )
    if summary.get("schema") != NATIVE_MAIN_QUALIFICATION_SUMMARY_SCHEMA:
        _fail("native_main_summary", "schema_invalid")
    _summary_bound_file(
        project_root,
        summary.get("discovery_manifest_path"),
        summary.get("discovery_manifest_sha256"),
        field="discovery_manifest",
    )
    _summary_bound_file(
        project_root,
        summary.get("source_preflight_summary_path"),
        summary.get("source_preflight_summary_sha256"),
        field="source_preflight_summary",
    )
    if (
        summary.get("status") != "NOT_SCOREABLE"
        or summary.get("scoreable") is not False
        or summary.get("external_claim_authorized") is not False
        or summary.get("qualified_case_count") != 0
    ):
        _fail("native_main_summary", "scoreability_claim_invalid")
    blockers = summary.get("score_blockers")
    required_blockers = {
        "qualification_only_capture",
        "native_equivalence_not_established",
        "independent_gold_missing",
        "qualified_natural_case_count_below_nine",
        "same_condition_9_slot_execution_missing",
        "receipt_bound_score_bundle_missing",
    }
    if not isinstance(blockers, list) or not required_blockers.issubset(set(blockers)):
        _fail("native_main_summary", "score_blockers_incomplete")
    results = summary.get("results")
    candidate_count = summary.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or not isinstance(results, list)
        or len(results) != candidate_count
    ):
        _fail("native_main_summary", "candidate_count_mismatch")
    complete_count = 0
    failed_count = 0
    hazardous_count = 0
    safe_count = 0
    seen: set[str] = set()
    for row in results:
        if not isinstance(row, dict):
            _fail("native_main_summary", "result_row_invalid")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            _fail("native_main_summary", "candidate_id_invalid")
        seen.add(candidate_id)
        if row.get("scoreable") is not False or row.get("external_claim_authorized") is not False:
            _fail("native_main_summary", "row_scoreability_claim_invalid")
        status = row.get("status")
        if row.get("receipt_created") is True:
            receipt = _summary_bound_file(
                project_root,
                row.get("raw_receipt_path"),
                row.get("raw_receipt_sha256"),
                field=f"{candidate_id}_raw_receipt",
            )
            _summary_bound_file(
                project_root,
                row.get("frame_artifact_path"),
                row.get("frame_artifact_sha256"),
                field=f"{candidate_id}_frame_artifact",
            )
            payload = _read_json_without_duplicate_keys(
                receipt, gate="native_main_summary", reason="raw_receipt_invalid"
            )
            decision = payload.get("decision")
            if (
                payload.get("schema") != "flashpatch-godot-native-main-capture-v1"
                or payload.get("qualification_only") is not True
                or payload.get("scoreable") is not False
                or payload.get("native_equivalence") != "NOT_ESTABLISHED"
                or decision not in {"SAFE_SCENARIO_READY", "HAZARDOUS_ATTRIBUTION_PENDING"}
                or row.get("decision") != decision
                or row.get("frame_count") != payload.get("frame_count")
            ):
                _fail("native_main_summary", "raw_receipt_contract_invalid")
            complete_count += 1
            if decision == "SAFE_SCENARIO_READY":
                safe_count += 1
            if decision == "HAZARDOUS_ATTRIBUTION_PENDING":
                hazardous_count += 1
        else:
            if not isinstance(status, str) or not status.startswith(("FAILED_", "NOT_EXECUTED")):
                _fail("native_main_summary", "failed_row_status_invalid")
            failed_count += 1
    if (
        summary.get("executed_complete_receipt_count") != complete_count
        or summary.get("failed_count") != failed_count
        or summary.get("safe_qualification_only_count") != safe_count
        or summary.get("hazardous_attribution_pending_count") != hazardous_count
    ):
        _fail("native_main_summary", "aggregate_counts_mismatch")
    return {
        "schema": "flashpatch-l7-native-main-qualification-summary-assessment-v1",
        "summary_id": summary.get("summary_id"),
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "candidate_count": candidate_count,
        "executed_complete_receipt_count": complete_count,
        "failed_count": failed_count,
        "safe_qualification_only_count": safe_count,
        "hazardous_attribution_pending_count": hazardous_count,
        "qualified_case_count": 0,
        "score_blockers": blockers,
        "external_claim_authorized": False,
    }


def _copy_tree_without_links(source: Path, destination: Path, *, gate: str) -> None:
    if source.is_symlink() or not source.is_dir() or destination.exists():
        _fail(gate, "copy_root_invalid")
    for path in source.rglob("*"):
        if path.is_symlink():
            _fail(gate, "copy_source_symlink")
    try:
        shutil.copytree(source, destination, symlinks=True)
    except OSError:
        _fail(gate, "copy_failed")


def package_native_main_natural_case_bundle(
    destination: Path | str,
    *,
    source_checkout: Path | str,
    qualification_root: Path | str,
    raw_output: Path | str,
    manifest_path: Path | str,
    preflight_path: Path | str,
    candidate_id: str,
) -> Path:
    """Seal one already-executed native-main run without rewriting its artifacts."""
    destination = Path(destination)
    supplied_source = Path(source_checkout)
    if supplied_source.is_symlink():
        _fail("package", "source_symlink")
    source = supplied_source.resolve(strict=True)
    qualification = Path(qualification_root).resolve(strict=True)
    output = Path(raw_output).resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)
    preflight = Path(preflight_path).resolve(strict=True)
    if destination.exists() or not isinstance(candidate_id, str) or not candidate_id:
        _fail("package", "destination_or_candidate_invalid")
    manifest_data = _read_json_without_duplicate_keys(manifest, gate="package", reason="manifest_invalid")
    verify_corpus_discovery_manifest(manifest)
    candidates = manifest_data.get("candidates")
    candidate = next((item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == candidate_id), None) if isinstance(candidates, list) else None
    if candidate is None:
        _fail("package", "candidate_unknown")
    for item in (source, qualification, output, manifest, preflight):
        if item.is_symlink():
            _fail("package", "source_symlink")
    required_output = {"renderer-frames.npz", "replay.json", "native-main-capture-receipt.json"}
    entries = {path.name: path for path in output.iterdir()} if output.is_dir() else {}
    capture_dir = entries.get("renderer-capture")
    if set(entries) != required_output | {"renderer-capture"} or any(entries[name].is_symlink() or not entries[name].is_file() for name in required_output) or capture_dir is None or capture_dir.is_symlink() or not capture_dir.is_dir() or any(path.is_symlink() or not path.is_file() for path in capture_dir.iterdir()):
        _fail("package", "raw_output_shape_invalid")
    checks = [("rev-parse", "HEAD"), ("remote", "get-url", "origin"), ("status", "--porcelain=v1", "--untracked-files=all")]
    observed = [subprocess.run(["git", "-C", str(source), *check], capture_output=True, text=True, check=False) for check in checks]
    if any(item.returncode != 0 for item in observed) or observed[2].stdout.strip():
        _fail("package", "source_checkout_not_clean")
    trace = qualification / ".flashpatch" / "native-main-trace.json"
    if not trace.is_file() or trace.is_symlink():
        _fail("package", "trace_missing")
    license_path = source / str(_read_json_without_duplicate_keys(preflight, gate="package", reason="preflight_invalid").get("license_path", ""))
    if not license_path.is_file() or license_path.is_symlink():
        _fail("package", "license_missing")
    if source in destination.parents or qualification in destination.parents or output in destination.parents:
        _fail("package", "destination_nested_in_input")
    destination.mkdir(parents=True)
    _copy_tree_without_links(source, destination / "source-checkout", gate="package")
    _copy_tree_without_links(qualification, destination / "qualification", gate="package")
    (destination / "output").mkdir()
    for name in sorted(required_output):
        shutil.copy2(output / name, destination / "output" / name, follow_symlinks=False)
    shutil.copy2(trace, destination / "trace.json", follow_symlinks=False)
    shutil.copy2(manifest, destination / "manifest.json", follow_symlinks=False)
    shutil.copy2(preflight, destination / "preflight.json", follow_symlinks=False)
    templates = manifest_data.get("trace_templates")
    template = next((item for item in templates if isinstance(item, dict) and item.get("trace_template_id") == candidate.get("trace_template_id")), None) if isinstance(templates, list) else None
    if not isinstance(template, dict) or not isinstance(template.get("project_subpath"), str):
        _fail("package", "candidate_template_missing")
    repository = {"url": candidate["repository_url"], "revision": candidate["revision"], "license": candidate["license"], "project_subpath": template["project_subpath"]}
    source_provenance = {"repository_url": repository["url"], "revision": repository["revision"], "license": repository["license"], "project_subpath": repository["project_subpath"], "original_unmutated": True, "source_checkout_path": "source-checkout", "source_tree_sha256": _source_tree_sha256(destination / "source-checkout"), "license_path": f"source-checkout/{license_path.relative_to(source)}", "license_sha256": _sha256_file(destination / "source-checkout" / license_path.relative_to(source))}
    (destination / "source-provenance.json").write_text(json.dumps(source_provenance, sort_keys=True) + "\n", encoding="utf-8")
    artifact = destination / "output" / "renderer-frames.npz"
    with open_renderer_artifact(artifact) as loaded:
        rgb_hash = renderer_rgb_sha256(loaded.frames)
        timestamp_hash = hashlib.sha256(np.ascontiguousarray(loaded.timestamps).tobytes()).hexdigest()
    native = {"candidate_id": candidate_id, "manifest_path": "manifest.json", "manifest_sha256": _sha256_file(destination / "manifest.json"), "preflight_path": "preflight.json", "preflight_sha256": _sha256_file(destination / "preflight.json"), "qualification_path": "qualification", "frame_artifact_path": "output/renderer-frames.npz", "frame_artifact_sha256": _sha256_file(artifact), "replay_path": "output/replay.json", "replay_sha256": _sha256_file(destination / "output/replay.json"), "execution_receipt_path": "output/native-main-capture-receipt.json", "execution_receipt_sha256": _sha256_file(destination / "output/native-main-capture-receipt.json"), "trace_path": "trace.json", "trace_sha256": _sha256_file(destination / "trace.json"), "rgb_bytes_sha256": rgb_hash, "timestamps_f64_sha256": timestamp_hash}
    ledger = {"schema": NATIVE_MAIN_NATURAL_CASE_SCHEMA, "case_id": f"{candidate_id}-native-main", "case_class": "natural_public_godot", "controlled_mutation": False, "repository": repository, "source_provenance": {"path": "source-provenance.json", "sha256": _sha256_file(destination / "source-provenance.json")}, "native_main": native}
    (destination / "native-main-natural-case.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_native_main_natural_case_bundle(destination)
    return destination


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    gate: str,
    label: str,
) -> None:
    missing = required.difference(value)
    if missing:
        _fail(gate, f"{label}_field_missing:{sorted(missing)[0]}")
    unexpected = set(value).difference(required | optional)
    if unexpected:
        _fail(gate, f"{label}_field_forbidden:{sorted(unexpected)[0]}")


def _reject_discovery_claim_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _FORBIDDEN_DISCOVERY_KEYS.search(key):
                _fail("discovery_claim", f"forbidden_field:{key}")
            _reject_discovery_claim_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_discovery_claim_keys(child)


def _reject_discovery_claim_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(" ".join(value.split())) < 12:
        _fail("discovery", f"{field}_imprecise")
    normalized = " ".join(value.split())
    words = set(re.findall(r"[a-z]+", normalized.lower()))
    forbidden = sorted(words & _FORBIDDEN_DISCOVERY_WORDS)
    if forbidden or "ground truth" in normalized.lower():
        token = forbidden[0] if forbidden else "ground_truth"
        _fail("discovery_claim", f"forbidden_text:{field}:{token}")
    return normalized


def _parse_discovery_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        _fail("discovery_cutoff", f"{field}_invalid")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        _fail("discovery_cutoff", f"{field}_invalid")


def _discovery_rule_ids(
    manifest: dict[str, Any],
) -> tuple[set[str], set[str]]:
    all_ids: set[str] = set()
    groups: list[set[str]] = []
    for field in ("inclusion_rules", "exclusion_rules"):
        rules = manifest.get(field)
        if not isinstance(rules, list) or not rules:
            _fail("discovery_rules", f"{field}_invalid")
        group: set[str] = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                _fail("discovery_rules", f"{field}_entry_invalid:{index}")
            _require_exact_keys(
                rule,
                required=frozenset({"id", "criterion"}),
                gate="discovery_rules",
                label=f"{field}_entry",
            )
            rule_id = rule.get("id")
            if (
                not isinstance(rule_id, str)
                or re.fullmatch(r"[A-Z][A-Z0-9_-]{1,31}", rule_id) is None
                or rule_id in all_ids
            ):
                _fail("discovery_rules", f"{field}_id_invalid_or_duplicate")
            _reject_discovery_claim_text(rule.get("criterion"), field=f"{field}.{rule_id}")
            group.add(rule_id)
            all_ids.add(rule_id)
        groups.append(group)
    return groups[0], groups[1]


def _verify_qualified_discovery_bundle(
    root: Path, candidate: dict[str, Any]
) -> dict[str, object]:
    bundle_root = _owned_directory(
        root,
        candidate.get("natural_bundle_path"),
        gate="discovery_bundle",
        field="natural_bundle",
    )
    ledger_path = bundle_root / "natural-case.json"
    expected_ledger_hash = candidate.get("natural_bundle_ledger_sha256")
    if not _is_hex64(expected_ledger_hash):
        _fail("discovery_bundle", "natural_bundle_ledger_hash_invalid")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        _fail("discovery_bundle", "natural_bundle_ledger_missing")
    if _sha256_file(ledger_path) != expected_ledger_hash:
        _fail("discovery_bundle", "natural_bundle_ledger_hash_mismatch")
    ledger = _read_json(ledger_path, gate="discovery_bundle", reason="natural_bundle_ledger_invalid")
    repository = ledger.get("repository")
    if not isinstance(repository, dict):
        _fail("discovery_bundle", "natural_bundle_repository_missing")
    expected_identity = {
        "revision": candidate["revision"],
        "license": candidate["license"],
    }
    if (
        _normalized_repository_url(str(repository.get("url")))
        != _normalized_repository_url(candidate["repository_url"])
        or any(repository.get(field) != expected for field, expected in expected_identity.items())
        or ledger.get("godot_version") != candidate["godot_version"]
    ):
        _fail("discovery_bundle", "candidate_bundle_identity_mismatch")
    assessment = verify_natural_case_bundle(bundle_root)
    if assessment.get("status") != "NOT_SCOREABLE" or assessment.get("scoreable") is not False:
        _fail("discovery_bundle", "natural_bundle_cross_gate_invalid")
    renderer = ledger.get("renderer")
    if not isinstance(renderer, dict):
        _fail("discovery_bundle", "natural_bundle_renderer_missing")
    receipt_path = _require_hash_bound_file(
        bundle_root,
        renderer.get("execution_receipt_path"),
        renderer.get("execution_receipt_sha256"),
        gate="discovery_bundle",
        field="natural_bundle_execution_receipt",
    )
    execution_receipt = _read_json(
        receipt_path,
        gate="discovery_bundle",
        reason="natural_bundle_execution_receipt_invalid",
    )
    capture = execution_receipt.get("factual_replay", {}).get("renderer_capture")
    if not isinstance(capture, dict) or capture.get("godot_version") != candidate["godot_version"]:
        _fail("discovery_bundle", "candidate_godot_version_unbound")
    return {
        "candidate_id": candidate["candidate_id"],
        "case_id": assessment["case_id"],
        "ledger_sha256": assessment["ledger_sha256"],
        "status": "NOT_SCOREABLE",
        "scoreable": False,
    }


def verify_corpus_discovery_manifest(manifest_path: Path | str) -> dict[str, object]:
    """Freeze a pre-benchmark candidate universe without authorizing any score.

    Discovery is deliberately independent of hazard labels and candidate detector
    outputs.  Qualified entries are re-opened through the G2 natural bundle gate,
    but even a frozen universe remains ``NOT_SCOREABLE``.
    """

    supplied = Path(manifest_path)
    if supplied.is_symlink():
        _fail("discovery", "manifest_path_invalid")
    try:
        path = supplied.resolve(strict=True)
    except OSError:
        _fail("discovery", "manifest_missing")
    if not path.is_file():
        _fail("discovery", "manifest_path_invalid")
    root = path.parent
    manifest = _read_json_without_duplicate_keys(
        path, gate="discovery", reason="manifest_invalid"
    )
    _reject_discovery_claim_keys(manifest)
    schema = manifest.get("schema")
    common_fields = frozenset(
        {
            "schema",
            "discovery_id",
            "discovery_cutoff_utc",
            "inclusion_rules",
            "exclusion_rules",
            "candidates",
        }
    )
    if schema == CORPUS_DISCOVERY_SCHEMA:
        _require_exact_keys(
            manifest,
            required=common_fields | frozenset({"source_discovery_method"}),
            gate="discovery",
            label="manifest",
        )
    elif schema == CORPUS_DISCOVERY_SCHEMA_V2:
        _require_exact_keys(
            manifest,
            required=common_fields | frozenset({"source_discovery_methods"}),
            gate="discovery",
            label="manifest",
        )
    elif schema in {CORPUS_DISCOVERY_SCHEMA_V3, CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        schema_specific_fields = (
            frozenset({"godot_binary"})
            if schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}
            else frozenset()
        )
        _require_exact_keys(
            manifest,
            required=(
                common_fields
                | frozenset(
                    {
                        "source_discovery_methods",
                        "trace_budget",
                        "trace_templates",
                        "selection_order",
                        "stop_rule",
                        "replacement_policy",
                    }
                )
                | schema_specific_fields
            ),
            gate="discovery",
            label="manifest",
        )
    else:
        _fail("discovery", "manifest_schema_invalid")
    discovery_id = manifest.get("discovery_id")
    if not isinstance(discovery_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", discovery_id) is None:
        _fail("discovery", "discovery_id_invalid")
    cutoff = _parse_discovery_timestamp(
        manifest.get("discovery_cutoff_utc"), field="discovery_cutoff_utc"
    )
    if cutoff > datetime.now(timezone.utc):
        _fail("discovery_cutoff", "cutoff_in_future")

    method_times: dict[str, datetime] = {}
    if schema == CORPUS_DISCOVERY_SCHEMA:
        methods: list[object] = [manifest.get("source_discovery_method")]
        method_field_prefix = "source_discovery_method"
    else:
        methods_value = manifest.get("source_discovery_methods")
        if not isinstance(methods_value, list) or not methods_value:
            _fail("discovery_method", "methods_missing_or_empty")
        methods = methods_value
        method_field_prefix = "source_discovery_methods"

    for index, method in enumerate(methods):
        if not isinstance(method, dict):
            _fail("discovery_method", f"method_invalid:{index}")
        required = frozenset({"name", "source", "query", "executed_at_utc"})
        if schema in {CORPUS_DISCOVERY_SCHEMA_V2, CORPUS_DISCOVERY_SCHEMA_V3, CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
            required |= frozenset({"method_id"})
        _require_exact_keys(
            method,
            required=required,
            gate="discovery_method",
            label=f"method_{index}",
        )
        method_id = "legacy" if schema == CORPUS_DISCOVERY_SCHEMA else method.get("method_id")
        if (
            not isinstance(method_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", method_id) is None
            or method_id in method_times
        ):
            _fail("discovery_method", f"method_id_invalid_or_duplicate:{index}")
        _reject_discovery_claim_text(method.get("name"), field=f"{method_field_prefix}.{method_id}.name")
        source = method.get("source")
        if not _is_canonical_https_repository_url(source):
            _fail("discovery_method", f"source_invalid:{method_id}")
        _reject_discovery_claim_text(method.get("query"), field=f"{method_field_prefix}.{method_id}.query")
        method_time = _parse_discovery_timestamp(
            method.get("executed_at_utc"), field=f"method_{method_id}_executed_at_utc"
        )
        if method_time > cutoff:
            _fail("discovery_cutoff", f"method_after_cutoff:{method_id}")
        method_times[method_id] = method_time

    inclusion_ids, exclusion_ids = _discovery_rule_ids(manifest)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        _fail("discovery_candidates", "candidate_universe_empty")

    candidate_ids: set[str] = set()
    repositories: dict[str, str] = {}
    qualified_case_ids: set[str] = set()
    outcome_counts = {outcome: 0 for outcome in sorted(_DISCOVERY_OUTCOMES)}
    qualified_receipts: list[dict[str, object]] = []
    candidate_required = frozenset(
        {
            "candidate_id",
            "discovered_at_utc",
            "repository_url",
            "revision",
            "license",
            "godot_version",
            "case_class",
            "controlled_mutation",
            "selection_outcome",
            "selection_reason",
            "rule_ids",
        }
    )
    if schema in {CORPUS_DISCOVERY_SCHEMA_V2, CORPUS_DISCOVERY_SCHEMA_V3, CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        candidate_required |= frozenset({"source_discovery_method_id"})
    if schema in {CORPUS_DISCOVERY_SCHEMA_V3, CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        candidate_required |= frozenset({"trace_template_id"})
    if schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        candidate_required |= frozenset({"license_path", "license_sha256"})
    bundle_fields = frozenset({"natural_bundle_path", "natural_bundle_ledger_sha256"})
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            _fail("discovery_candidates", f"candidate_invalid:{index}")
        outcome = candidate.get("selection_outcome")
        optional = bundle_fields if outcome == "qualified" else frozenset()
        _require_exact_keys(
            candidate,
            required=candidate_required | (bundle_fields if outcome == "qualified" else frozenset()),
            optional=optional,
            gate="discovery_candidates",
            label=f"candidate_{index}",
        )
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", candidate_id) is None
            or candidate_id in candidate_ids
        ):
            _fail("discovery_candidates", "candidate_id_invalid_or_duplicate")
        candidate_ids.add(candidate_id)
        discovered_at = _parse_discovery_timestamp(
            candidate.get("discovered_at_utc"), field=f"candidate_{candidate_id}_discovered_at_utc"
        )
        method_id = "legacy" if schema == CORPUS_DISCOVERY_SCHEMA else candidate.get("source_discovery_method_id")
        if not isinstance(method_id, str) or method_id not in method_times:
            _fail("discovery_candidates", f"candidate_method_unknown:{candidate_id}")
        if discovered_at < method_times[method_id]:
            _fail("discovery_cutoff", f"candidate_before_method:{candidate_id}")
        if discovered_at > cutoff:
            _fail("discovery_cutoff", f"candidate_after_cutoff:{candidate_id}")
        repository_url = candidate.get("repository_url")
        revision = candidate.get("revision")
        if not _is_canonical_https_repository_url(repository_url):
            _fail("discovery_candidates", f"repository_url_invalid:{candidate_id}")
        if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
            _fail("discovery_candidates", f"revision_invalid:{candidate_id}")
        normalized_url = _normalized_repository_url(repository_url)
        prior_revision = repositories.get(normalized_url)
        if prior_revision is not None and prior_revision != revision:
            _fail("discovery_candidates", f"repository_revision_drift:{candidate_id}")
        repositories[normalized_url] = revision
        license_name = candidate.get("license")
        if not isinstance(license_name, str) or not license_name.strip():
            _fail("discovery_candidates", f"license_invalid:{candidate_id}")
        if schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
            license_path = candidate.get("license_path")
            if (
                not isinstance(license_path, str)
                or not license_path
                or Path(license_path).is_absolute()
                or ".." in Path(license_path).parts
                or not _is_hex64(candidate.get("license_sha256"))
            ):
                _fail("discovery_candidates", f"license_evidence_invalid:{candidate_id}")
        godot_version = candidate.get("godot_version")
        if not isinstance(godot_version, str) or _GODOT_VERSION.fullmatch(godot_version) is None:
            _fail("discovery_candidates", f"godot_version_invalid:{candidate_id}")
        if candidate.get("case_class") != "natural_public_godot" or candidate.get("controlled_mutation") is not False:
            _fail("discovery_candidates", f"controlled_or_non_natural_case:{candidate_id}")
        if outcome not in _DISCOVERY_OUTCOMES:
            _fail("discovery_candidates", f"selection_outcome_invalid:{candidate_id}")
        outcome_counts[outcome] += 1
        _reject_discovery_claim_text(
            candidate.get("selection_reason"), field=f"candidate.{candidate_id}.selection_reason"
        )
        rule_ids = candidate.get("rule_ids")
        if (
            not isinstance(rule_ids, list)
            or not rule_ids
            or any(not isinstance(rule_id, str) for rule_id in rule_ids)
            or len(set(rule_ids)) != len(rule_ids)
        ):
            _fail("discovery_rules", f"candidate_rule_ids_invalid:{candidate_id}")
        referenced = set(rule_ids)
        if not referenced <= inclusion_ids | exclusion_ids:
            _fail("discovery_rules", f"candidate_rule_id_unknown:{candidate_id}")
        if outcome == "excluded" and not referenced & exclusion_ids:
            _fail("discovery_rules", f"exclusion_reason_missing:{candidate_id}")
        if outcome != "excluded" and not referenced & inclusion_ids:
            _fail("discovery_rules", f"inclusion_reason_missing:{candidate_id}")
        if outcome == "qualified":
            receipt = _verify_qualified_discovery_bundle(root, candidate)
            case_id = receipt.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                _fail("discovery_candidates", f"case_id_invalid:{candidate_id}")
            if case_id in qualified_case_ids:
                _fail("discovery_candidates", f"duplicate_case_id:{candidate_id}")
            qualified_case_ids.add(case_id)
            qualified_receipts.append(receipt)

    if schema in {CORPUS_DISCOVERY_SCHEMA_V3, CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        _verify_frozen_discovery_contract(
            manifest,
            schema=schema,
            candidate_ids=candidate_ids,
            repositories=repositories,
        )

    repository_count = len(repositories)
    qualified_case_count = len(qualified_receipts)
    corpus_frozen = repository_count >= 3 and qualified_case_count >= 9
    if repository_count < 3:
        reason = "minimum_three_distinct_repositories_not_met"
    elif qualified_case_count < 9:
        reason = "minimum_nine_qualified_natural_cases_not_met"
    else:
        reason = "discovery_population_frozen_without_scoring_authority"
    return {
        "schema": CORPUS_DISCOVERY_ASSESSMENT_SCHEMA,
        "discovery_id": discovery_id,
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "corpus_frozen": corpus_frozen,
        "reason": reason,
        "discovery_cutoff_utc": manifest["discovery_cutoff_utc"],
        "candidate_count": len(candidates),
        "source_discovery_method_count": len(method_times),
        "distinct_repository_count": repository_count,
        "qualified_case_count": qualified_case_count,
        "outcome_counts": outcome_counts,
        "qualified_case_receipts": qualified_receipts,
        "manifest_sha256": _sha256_file(path),
        "external_claim_authorized": False,
    }


def _verify_frozen_discovery_contract(
    manifest: dict[str, object],
    *,
    schema: str,
    candidate_ids: set[str],
    repositories: dict[str, str],
) -> None:
    """Verify the pre-execution choices that prevent outcome-driven replacement."""

    if schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        binary = manifest.get("godot_binary")
        if not isinstance(binary, dict):
            _fail("discovery_trace", "godot_binary_missing")
        _require_exact_keys(
            binary,
            required=frozenset({"sha256", "version"}),
            gate="discovery_trace",
            label="godot_binary",
        )
        if not _is_hex64(binary.get("sha256")) or not isinstance(binary.get("version"), str):
            _fail("discovery_trace", "godot_binary_invalid")
        if not str(binary["version"]).startswith("4."):
            _fail("discovery_trace", "godot_binary_major_mismatch")

    budget = manifest.get("trace_budget")
    if not isinstance(budget, dict):
        _fail("discovery_trace", "budget_missing")
    _require_exact_keys(
        budget,
        required=frozenset({"max_frames", "max_wall_clock_seconds", "repeat_count"}),
        gate="discovery_trace",
        label="budget",
    )
    if (
        not isinstance(budget.get("max_frames"), int)
        or isinstance(budget.get("max_frames"), bool)
        or budget["max_frames"] <= 0
        or not isinstance(budget.get("max_wall_clock_seconds"), int)
        or isinstance(budget.get("max_wall_clock_seconds"), bool)
        or budget["max_wall_clock_seconds"] <= 0
        or budget.get("repeat_count") != 3
    ):
        _fail("discovery_trace", "budget_invalid")
    if (
        schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}
        and budget["max_wall_clock_seconds"] > L7_NATIVE_MAX_WALL_CLOCK_SECONDS
    ):
        _fail("discovery_trace", "budget_wall_clock_exceeds_runner_limit")

    templates = manifest.get("trace_templates")
    if not isinstance(templates, list) or not templates:
        _fail("discovery_trace", "templates_missing_or_empty")
    template_ids: set[str] = set()
    for index, template in enumerate(templates):
        if not isinstance(template, dict):
            _fail("discovery_trace", f"template_invalid:{index}")
        required = (
            frozenset({"trace_template_id", "entrypoint", "action_sequence", "capture_frame_count"})
            if schema == CORPUS_DISCOVERY_SCHEMA_V3
            else frozenset(
                {
                    "trace_template_id", "runner_id", "project_subpath", "original_main_scene",
                    "fixed_fps", "capture_frames", "actions", "launch_arguments",
                    "pointer_events", "key_events", "scenario_readiness", "runtime_observations",
                } | (
                    frozenset({"scene_transition", "ui_selection_observations"})
                    if schema == CORPUS_DISCOVERY_SCHEMA_V5
                    else frozenset()
                )
            )
        )
        _require_exact_keys(template, required=required, gate="discovery_trace", label=f"template_{index}")
        template_id = template.get("trace_template_id")
        if (
            not isinstance(template_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", template_id) is None
            or template_id in template_ids
        ):
            _fail("discovery_trace", f"template_id_invalid_or_duplicate:{index}")
        if schema == CORPUS_DISCOVERY_SCHEMA_V3:
            entrypoint = template.get("entrypoint")
            actions = template.get("action_sequence")
            frame_count = template.get("capture_frame_count")
            if (
                not isinstance(entrypoint, str)
                or not entrypoint.strip()
                or not isinstance(actions, list)
                or any(not isinstance(action, str) or not action.strip() for action in actions)
                or not isinstance(frame_count, int)
                or isinstance(frame_count, bool)
                or frame_count <= 0
                or frame_count > budget["max_frames"]
            ):
                _fail("discovery_trace", f"template_invalid:{template_id}")
            _reject_discovery_claim_text(entrypoint, field=f"trace_templates.{template_id}.entrypoint")
            for action_index, action in enumerate(actions):
                _reject_discovery_claim_text(
                    action, field=f"trace_templates.{template_id}.action_sequence.{action_index}"
                )
        else:
            _verify_v4_native_main_trace(template, template_id=template_id, budget=budget, transition_required=schema == CORPUS_DISCOVERY_SCHEMA_V5)
        template_ids.add(template_id)

    candidates = manifest["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        trace_template_id = candidate.get("trace_template_id")
        if not isinstance(trace_template_id, str) or trace_template_id not in template_ids:
            _fail("discovery_trace", f"candidate_template_unknown:{candidate.get('candidate_id')}")
        if schema in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5} and not str(candidate["godot_version"]).startswith("4."):
            _fail("discovery_trace", f"candidate_runner_major_mismatch:{candidate.get('candidate_id')}")

    selection_order = manifest.get("selection_order")
    if (
        not isinstance(selection_order, list)
        or any(not isinstance(candidate_id, str) for candidate_id in selection_order)
        or len(selection_order) != len(set(selection_order))
        or set(selection_order) != candidate_ids
    ):
        _fail("discovery_selection", "order_not_exact_candidate_universe")

    stop_rule = manifest.get("stop_rule")
    if not isinstance(stop_rule, dict):
        _fail("discovery_selection", "stop_rule_missing")
    _require_exact_keys(
        stop_rule,
        required=frozenset(
            {"maximum_selected_cases", "maximum_cases_per_repository", "on_execution_failure"}
        ),
        gate="discovery_selection",
        label="stop_rule",
    )
    maximum_cases = stop_rule.get("maximum_selected_cases")
    maximum_per_repository = stop_rule.get("maximum_cases_per_repository")
    if (
        not isinstance(maximum_cases, int)
        or isinstance(maximum_cases, bool)
        or maximum_cases != len(candidate_ids)
        or not isinstance(maximum_per_repository, int)
        or isinstance(maximum_per_repository, bool)
        or maximum_per_repository <= 0
        or stop_rule.get("on_execution_failure") != "retain_and_stop"
    ):
        _fail("discovery_selection", "stop_rule_invalid")
    per_repository: dict[str, int] = {repository: 0 for repository in repositories}
    for candidate in candidates:
        assert isinstance(candidate, dict)
        repository = _normalized_repository_url(candidate["repository_url"])
        per_repository[repository] += 1
    if any(count > maximum_per_repository for count in per_repository.values()):
        _fail("discovery_selection", "repository_cap_exceeded")

    replacement = manifest.get("replacement_policy")
    if not isinstance(replacement, dict):
        _fail("discovery_selection", "replacement_policy_missing")
    _require_exact_keys(
        replacement,
        required=frozenset({"replacement_allowed", "replacement_requires_new_manifest"}),
        gate="discovery_selection",
        label="replacement_policy",
    )
    if replacement != {"replacement_allowed": False, "replacement_requires_new_manifest": True}:
        _fail("discovery_selection", "replacement_policy_invalid")


def _verify_v4_native_main_trace(
    template: dict[str, object], *, template_id: str, budget: dict[str, object], transition_required: bool = False
) -> None:
    """Reject any trace that cannot be handed verbatim to the native-main runner."""

    if template.get("runner_id") != ("flashpatch-native-main-godot4-v2" if transition_required else "flashpatch-native-main-godot4-v1"):
        _fail("discovery_trace", f"runner_unknown:{template_id}")
    project_subpath = template.get("project_subpath")
    main_scene = template.get("original_main_scene")
    fixed_fps = template.get("fixed_fps")
    capture_frames = template.get("capture_frames")
    if (
        not isinstance(project_subpath, str)
        or not project_subpath
        or Path(project_subpath).is_absolute()
        or ".." in Path(project_subpath).parts
        or not isinstance(main_scene, str)
        or not main_scene.startswith("res://")
        or not isinstance(fixed_fps, int)
        or isinstance(fixed_fps, bool)
        or fixed_fps <= 0
        or not isinstance(capture_frames, int)
        or isinstance(capture_frames, bool)
        or capture_frames < 2
        or capture_frames > budget["max_frames"]
    ):
        _fail("discovery_trace", f"native_main_scalar_invalid:{template_id}")

    actions = template.get("actions")
    if not isinstance(actions, list):
        _fail("discovery_trace", f"native_main_actions_invalid:{template_id}")
    action_identity: set[tuple[int, str]] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            _fail("discovery_trace", f"native_main_action_invalid:{template_id}:{index}")
        _require_exact_keys(action, required=frozenset({"frame", "action", "pressed"}), gate="discovery_trace", label="native_main_action")
        frame, name, pressed = action.get("frame"), action.get("action"), action.get("pressed")
        if (
            not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or frame >= capture_frames
            or not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or not isinstance(pressed, bool) or (frame, name) in action_identity
        ):
            _fail("discovery_trace", f"native_main_action_invalid:{template_id}:{index}")
        action_identity.add((frame, name))

    arguments = template.get("launch_arguments")
    if not isinstance(arguments, list):
        _fail("discovery_trace", f"native_main_arguments_invalid:{template_id}")
    reserved = {"--", "--trace", "--output", "--renderer-capture"}
    if any(
        not isinstance(argument, str) or not argument or "\x00" in argument
        or argument in reserved or argument.startswith(("--trace=", "--output=", "--renderer-capture="))
        for argument in arguments
    ):
        _fail("discovery_trace", f"native_main_arguments_invalid:{template_id}")

    pointers = template.get("pointer_events")
    if not isinstance(pointers, list):
        _fail("discovery_trace", f"native_main_pointers_invalid:{template_id}")
    for index, event in enumerate(pointers):
        if not isinstance(event, dict):
            _fail("discovery_trace", f"native_main_pointer_invalid:{template_id}:{index}")
        _require_exact_keys(event, required=frozenset({"frame", "kind", "x", "y"}), gate="discovery_trace", label="native_main_pointer")
        frame, x, y = event.get("frame"), event.get("x"), event.get("y")
        if (
            not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or frame >= capture_frames
            or event.get("kind") != "left_click"
            or isinstance(x, bool) or not isinstance(x, (int, float)) or not 0.0 <= float(x) <= 1.0
            or isinstance(y, bool) or not isinstance(y, (int, float)) or not 0.0 <= float(y) <= 1.0
        ):
            _fail("discovery_trace", f"native_main_pointer_invalid:{template_id}:{index}")

    key_events = template.get("key_events")
    if not isinstance(key_events, list):
        _fail("discovery_trace", f"native_main_key_events_invalid:{template_id}")
    for index, event in enumerate(key_events):
        if not isinstance(event, dict):
            _fail("discovery_trace", f"native_main_key_event_invalid:{template_id}:{index}")
        _require_exact_keys(
            event,
            required=frozenset({"frame", "key"}),
            gate="discovery_trace",
            label="native_main_key_event",
        )
        frame = event.get("frame")
        if (
            not isinstance(frame, int)
            or isinstance(frame, bool)
            or frame < 0
            or frame >= capture_frames
            or event.get("key") not in {"down", "up", "enter"}
        ):
            _fail("discovery_trace", f"native_main_key_event_invalid:{template_id}:{index}")

    readiness = template.get("scenario_readiness")
    if not isinstance(readiness, dict):
        _fail("discovery_trace", f"native_main_readiness_invalid:{template_id}")
    readiness_fields = {
        "required_node_paths", "required_group_minimums", "required_visible"
    }
    if transition_required:
        readiness_fields.add("required_option_selection")
    _require_exact_keys(
        readiness,
        required=frozenset(readiness_fields),
        gate="discovery_trace",
        label="native_main_readiness",
    )
    paths = readiness.get("required_node_paths")
    groups = readiness.get("required_group_minimums")
    visibility = readiness.get("required_visible")
    if (
        not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path.startswith("/root/") for path in paths)
        or not isinstance(groups, dict) or any(not isinstance(group, str) or not group or not isinstance(count, int) or isinstance(count, bool) or count < 1 for group, count in groups.items())
        or not isinstance(visibility, list) or not visibility
    ):
        _fail("discovery_trace", f"native_main_readiness_invalid:{template_id}")
    for index, item in enumerate(visibility):
        if not isinstance(item, dict):
            _fail("discovery_trace", f"native_main_visibility_invalid:{template_id}:{index}")
        _require_exact_keys(item, required=frozenset({"node_path", "visible"}), gate="discovery_trace", label="native_main_visibility")
        if not isinstance(item.get("node_path"), str) or not item["node_path"].startswith("/root/") or not isinstance(item.get("visible"), bool):
            _fail("discovery_trace", f"native_main_visibility_invalid:{template_id}:{index}")

    if transition_required:
        selections = readiness.get("required_option_selection")
        if not isinstance(selections, list):
            _fail("discovery_trace", f"native_main_selection_invalid:{template_id}")
        selection_paths: set[str] = set()
        for index, selection in enumerate(selections):
            if not isinstance(selection, dict):
                _fail("discovery_trace", f"native_main_selection_invalid:{template_id}:{index}")
            _require_exact_keys(
                selection,
                required=frozenset({"node_path", "selected_index", "selected_text"}),
                gate="discovery_trace",
                label="native_main_selection",
            )
            node_path = selection.get("node_path")
            selected_index = selection.get("selected_index")
            selected_text = selection.get("selected_text")
            if (
                not isinstance(node_path, str)
                or not node_path.startswith("/root/")
                or node_path not in paths
                or node_path in selection_paths
                or isinstance(selected_index, bool)
                or not isinstance(selected_index, int)
                or selected_index < 0
                or not isinstance(selected_text, str)
                or not selected_text
            ):
                _fail("discovery_trace", f"native_main_selection_invalid:{template_id}:{index}")
            selection_paths.add(node_path)

        selection_observations = template.get("ui_selection_observations")
        if (
            not isinstance(selection_observations, list)
            or any(
                not isinstance(path, str) or not path.startswith("/root/")
                for path in selection_observations
            )
            or len(set(selection_observations)) != len(selection_observations)
            or set(selection_observations) != selection_paths
        ):
            _fail("discovery_trace", f"native_main_selection_signal_invalid:{template_id}")

    observations = template.get("runtime_observations")
    if not isinstance(observations, list):
        _fail("discovery_trace", f"native_main_observations_invalid:{template_id}")
    identities: set[tuple[str, str]] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            _fail("discovery_trace", f"native_main_observation_invalid:{template_id}:{index}")
        _require_exact_keys(observation, required=frozenset({"node_path", "property", "script_path", "resource_path", "source_line"}), gate="discovery_trace", label="native_main_observation")
        node, prop = observation.get("node_path"), observation.get("property")
        if (
            not isinstance(node, str) or not node.startswith("/root/")
            or not isinstance(prop, str) or not prop
            or not isinstance(observation.get("script_path"), str) or not observation["script_path"].startswith("res://")
            or not isinstance(observation.get("resource_path"), str) or not observation["resource_path"].startswith("res://")
            or not isinstance(observation.get("source_line"), int) or isinstance(observation.get("source_line"), bool) or observation["source_line"] < 1
            or (node, prop) in identities
        ):
            _fail("discovery_trace", f"native_main_observation_invalid:{template_id}:{index}")
        identities.add((node, prop))
    if transition_required and template.get("scene_transition") is not None:
        transition = template["scene_transition"]
        if not isinstance(transition, dict) or set(transition) != {"from_scene", "to_scene", "earliest_frame", "latest_frame"}:
            _fail("discovery_trace", f"native_main_transition_invalid:{template_id}")
        source, target = transition.get("from_scene"), transition.get("to_scene")
        earliest, latest = transition.get("earliest_frame"), transition.get("latest_frame")
        if source != main_scene or not isinstance(target, str) or not target.startswith("res://") or target == source or not isinstance(earliest, int) or not isinstance(latest, int) or isinstance(earliest, bool) or isinstance(latest, bool) or earliest < 0 or earliest > latest or latest >= capture_frames:
            _fail("discovery_trace", f"native_main_transition_invalid:{template_id}")
    if transition_required and template_id == "trace-godotdemo-old-film-native-main":
        expected_selection = [{
            "node_path": "/root/ScreenShaders/Effect",
            "selected_index": 11,
            "selected_text": "FX: OldFilm",
        }]
        if readiness.get("required_option_selection") != expected_selection:
            _fail("discovery_trace", f"native_main_selection_contract_mismatch:{template_id}")
        if template.get("ui_selection_observations") != ["/root/ScreenShaders/Effect"]:
            _fail("discovery_trace", f"native_main_selection_signal_invalid:{template_id}")
        expected_pointer = [{
            "frame": 10,
            "kind": "left_click",
            "x": 0.334375,
            "y": 0.034722222222222224,
        }]
        expected_keys = [
            *({"frame": frame, "key": "down"} for frame in range(12, 24)),
            {"frame": 25, "key": "enter"},
        ]
        if pointers != expected_pointer or key_events != expected_keys:
            _fail("discovery_trace", f"native_main_selection_timing_mismatch:{template_id}")
        if visibility != [{
            "node_path": "/root/ScreenShaders/Effects/OldFilm",
            "visible": True,
        }]:
            _fail("discovery_trace", f"native_main_post_selection_visibility_mismatch:{template_id}")


def assess_corpus_discovery_manifest(manifest_path: Path | str) -> dict[str, object]:
    try:
        return verify_corpus_discovery_manifest(manifest_path)
    except L7VerificationFailure as exc:
        return {
            "schema": CORPUS_DISCOVERY_ASSESSMENT_SCHEMA,
            "status": "INCONCLUSIVE",
            "scoreable": False,
            "corpus_frozen": False,
            "diagnostic": exc.diagnostic,
            "external_claim_authorized": False,
        }


def verify_v4_candidate_source_preflight(
    manifest_path: Path | str,
    candidate_id: str,
    source_root: Path | str,
    godot_binary: Path | str,
) -> dict[str, object]:
    """Bind a v4 trace to the actual clean pinned Godot project before capture.

    This deliberately stops before invoking Godot or any detector.  It prevents
    a manifest from naming an arbitrary scene while the runner launches a
    different configured project main scene.
    """

    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = _read_json_without_duplicate_keys(
        manifest_file, gate="source_preflight", reason="manifest_invalid"
    )
    if manifest.get("schema") not in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        _fail("source_preflight", "v4_or_v5_manifest_required")
    candidates = manifest.get("candidates")
    templates = manifest.get("trace_templates")
    if not isinstance(candidates, list) or not isinstance(templates, list):
        _fail("source_preflight", "manifest_shape_invalid")
    candidate = next(
        (item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == candidate_id),
        None,
    )
    if candidate is None:
        _fail("source_preflight", "candidate_unknown")
    template_id = candidate.get("trace_template_id")
    template = next(
        (item for item in templates if isinstance(item, dict) and item.get("trace_template_id") == template_id),
        None,
    )
    if template is None:
        _fail("source_preflight", "candidate_template_unknown")
    root = Path(source_root)
    if root.is_symlink():
        _fail("source_preflight", "source_root_invalid")
    try:
        root = root.resolve(strict=True)
    except OSError:
        _fail("source_preflight", "source_root_invalid")
    if not root.is_dir():
        _fail("source_preflight", "source_root_invalid")

    supplied_binary = Path(godot_binary)
    if supplied_binary.is_symlink():
        _fail("source_preflight", "godot_binary_invalid")
    try:
        binary_path = supplied_binary.resolve(strict=True)
    except OSError:
        _fail("source_preflight", "godot_binary_invalid")
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        _fail("source_preflight", "godot_binary_invalid")
    binary_contract = manifest.get("godot_binary")
    if not isinstance(binary_contract, dict):
        _fail("source_preflight", "godot_binary_contract_missing")
    observed_binary_sha256 = _sha256_file(binary_path)
    if observed_binary_sha256 != binary_contract.get("sha256"):
        _fail("source_preflight", "godot_binary_hash_mismatch")
    try:
        version_result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        _fail("source_preflight", "godot_binary_version_unavailable")
    observed_binary_version = version_result.stdout.strip()
    if version_result.returncode != 0 or observed_binary_version != binary_contract.get("version"):
        _fail("source_preflight", "godot_binary_version_mismatch")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            _fail("source_preflight", "git_identity_unavailable")
        return completed.stdout.strip()

    expected_url, expected_revision = candidate.get("repository_url"), candidate.get("revision")
    if (
        not isinstance(expected_url, str)
        or not isinstance(expected_revision, str)
        or _normalized_repository_url(git("remote", "get-url", "origin")) != _normalized_repository_url(expected_url)
        or git("rev-parse", "HEAD") != expected_revision
        or git("status", "--porcelain=v1", "--untracked-files=all")
    ):
        _fail("source_preflight", "clean_pinned_source_mismatch")
    license_path_value = candidate.get("license_path")
    if not isinstance(license_path_value, str):
        _fail("source_preflight", "license_evidence_invalid")
    license_file = root / license_path_value
    try:
        license_file.relative_to(root)
    except ValueError:
        _fail("source_preflight", "license_evidence_invalid")
    if license_file.is_symlink() or not license_file.is_file():
        _fail("source_preflight", "license_evidence_invalid")
    if _sha256_file(license_file) != candidate.get("license_sha256"):
        _fail("source_preflight", "license_evidence_mismatch")
    project_subpath = template.get("project_subpath")
    if not isinstance(project_subpath, str):
        _fail("source_preflight", "project_subpath_invalid")
    project = root / project_subpath
    if project.is_symlink() or not project.is_dir():
        _fail("source_preflight", "project_subpath_invalid")
    config = project / "project.godot"
    if config.is_symlink() or not config.is_file():
        _fail("source_preflight", "project_config_missing")
    text = config.read_text(encoding="utf-8", errors="strict")
    if re.search(r"^config_version\s*=\s*5\s*$", text, re.MULTILINE) is None:
        _fail("source_preflight", "godot4_project_required")
    match = re.search(r'^run/main_scene="(?P<scene>[^"]+)"\s*$', text, re.MULTILINE)
    expected_scene = template.get("original_main_scene")
    if match is None or not isinstance(expected_scene, str):
        _fail("source_preflight", "main_scene_missing")
    observed_scene = match.group("scene")
    if not observed_scene.startswith("res://"):
        observed_scene = f"res://{observed_scene}"
    if observed_scene != expected_scene:
        _fail("source_preflight", "main_scene_mismatch")
    scene = project / observed_scene.removeprefix("res://")
    if scene.is_symlink() or not scene.is_file():
        _fail("source_preflight", "main_scene_missing")
    transition_target_sha256: str | None = None
    if manifest.get("schema") == CORPUS_DISCOVERY_SCHEMA_V5:
        transition = template.get("scene_transition")
        if transition is not None:
            if not isinstance(transition, dict) or not isinstance(transition.get("to_scene"), str):
                _fail("source_preflight", "transition_invalid")
            target = project / transition["to_scene"].removeprefix("res://")
            if target.is_symlink() or not target.is_file():
                _fail("source_preflight", "transition_target_missing")
            transition_target_sha256 = f"sha256:{_sha256_file(target)}"
    return {
        "schema": "flashpatch-l7-v4-source-preflight-v1",
        "status": "PRECHECKED_NOT_SCOREABLE",
        "scoreable": False,
        "candidate_id": candidate_id,
        "manifest_sha256": _sha256_file(manifest_file),
        "repository_url": expected_url,
        "revision": expected_revision,
        "license": candidate.get("license"),
        "license_path": license_path_value,
        "license_sha256": candidate.get("license_sha256"),
        "project_subpath": project_subpath,
        "original_main_scene": observed_scene,
        "project_config_sha256": _sha256_file(config),
        "main_scene_sha256": _sha256_file(scene),
        "godot_binary_path": str(binary_path),
        "godot_binary_sha256": observed_binary_sha256,
        "godot_version": observed_binary_version,
        "transition_target_sha256": transition_target_sha256,
        "renderer_executed": False,
        "qualified_case_count": 0,
        "external_claim_authorized": False,
    }


def assess_v4_candidate_source_preflight(
    manifest_path: Path | str,
    candidate_id: str,
    source_root: Path | str,
    godot_binary: Path | str,
) -> dict[str, object]:
    try:
        return verify_v4_candidate_source_preflight(
            manifest_path, candidate_id, source_root, godot_binary
        )
    except (L7VerificationFailure, OSError, UnicodeError) as exc:
        diagnostic = exc.diagnostic if isinstance(exc, L7VerificationFailure) else "FAIL_CLOSED:source_preflight:source_unreadable"
        return {
            "schema": "flashpatch-l7-v4-source-preflight-v1",
            "status": "INCONCLUSIVE",
            "scoreable": False,
            "diagnostic": diagnostic,
            "external_claim_authorized": False,
        }


def execute_v4_candidate_native_main_original(
    manifest_path: Path | str,
    candidate_id: str,
    source_root: Path | str,
    qualification_destination: Path | str,
    replay_output: Path | str,
    godot_binary: Path | str,
) -> dict[str, object]:
    """Execute one frozen v4/v5 native original with its exact wall-clock budget.

    This is the only L7 integration point that constructs the native-main
    qualification and runner together.  Generic and L6 callers retain their
    existing timeout defaults; L7 cannot supply an independent override.
    """

    supplied_manifest = Path(manifest_path)
    verify_corpus_discovery_manifest(supplied_manifest)
    manifest_file = supplied_manifest.resolve(strict=True)
    manifest = _read_json_without_duplicate_keys(
        manifest_file, gate="native_execution", reason="manifest_invalid"
    )
    if manifest.get("schema") not in {CORPUS_DISCOVERY_SCHEMA_V4, CORPUS_DISCOVERY_SCHEMA_V5}:
        _fail("native_execution", "v4_or_v5_manifest_required")
    candidates = manifest.get("candidates")
    templates = manifest.get("trace_templates")
    budget = manifest.get("trace_budget")
    if not isinstance(candidates, list) or not isinstance(templates, list) or not isinstance(budget, dict):
        _fail("native_execution", "manifest_shape_invalid")
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if candidate is None:
        _fail("native_execution", "candidate_unknown")
    template_id = candidate.get("trace_template_id")
    template = next(
        (
            item
            for item in templates
            if isinstance(item, dict) and item.get("trace_template_id") == template_id
        ),
        None,
    )
    if template is None:
        _fail("native_execution", "candidate_template_unknown")
    timeout_seconds = budget.get("max_wall_clock_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or timeout_seconds > L7_NATIVE_MAX_WALL_CLOCK_SECONDS
    ):
        _fail("native_execution", "wall_clock_budget_invalid")

    source = Path(source_root).resolve()
    binary = Path(godot_binary).resolve()
    destination = Path(qualification_destination).resolve()
    output = Path(replay_output).resolve()
    verify_v4_candidate_source_preflight(
        supplied_manifest, candidate_id, source, binary
    )
    project_subpath = template["project_subpath"]
    assert isinstance(project_subpath, str)
    materialize_options = {
        "fixed_fps": template["fixed_fps"],
        "capture_frames": template["capture_frames"],
        "actions": template["actions"],
        "launch_arguments": template["launch_arguments"],
        "pointer_events": template["pointer_events"],
        "key_events": template["key_events"],
        "scenario_readiness": template["scenario_readiness"],
        "runtime_observations": template["runtime_observations"],
        "scene_transition": template.get("scene_transition"),
    }
    if manifest.get("schema") == CORPUS_DISCOVERY_SCHEMA_V5:
        materialize_options["ui_selection_observations"] = template[
            "ui_selection_observations"
        ]
    qualification = materialize_native_main_capture_qualification(
        source / project_subpath,
        destination,
        **materialize_options,
    )
    return classify_native_main_capture_qualification(
        qualification,
        output,
        godot_binary=binary,
        timeout_seconds=timeout_seconds,
    )


def assess_natural_case_bundle(case_root: Path | str) -> dict[str, object]:
    """Return the only two G2 terminal outputs without turning evidence gaps into PASS."""

    try:
        return verify_natural_case_bundle(case_root)
    except L7VerificationFailure as exc:
        return {
            "schema": ASSESSMENT_SCHEMA,
            "status": "INCONCLUSIVE",
            "scoreable": False,
            "diagnostic": exc.diagnostic,
            "external_claim_authorized": False,
    }


def assess_native_main_natural_case_bundle(case_root: Path | str) -> dict[str, object]:
    try:
        return verify_native_main_natural_case_bundle(case_root)
    except L7VerificationFailure as exc:
        return {
            "schema": ASSESSMENT_SCHEMA,
            "status": "INCONCLUSIVE",
            "scoreable": False,
            "diagnostic": exc.diagnostic,
            "external_claim_authorized": False,
        }


def assess_native_main_qualification_summary(summary_path: Path | str) -> dict[str, object]:
    try:
        return verify_native_main_qualification_summary(summary_path)
    except L7VerificationFailure as exc:
        return {
            "schema": "flashpatch-l7-native-main-qualification-summary-assessment-v1",
            "status": "INCONCLUSIVE",
            "scoreable": False,
            "diagnostic": exc.diagnostic,
            "external_claim_authorized": False,
        }


def _score_bundle_status(
    bundle_path: Path,
    *,
    bootstrap_iterations: int,
) -> tuple[int, dict[str, object], str]:
    if bootstrap_iterations != 10_000:
        return (
            2,
            {
                "status": "NOT_SCOREABLE",
                "reason": "BOOTSTRAP_ITERATION_COUNT_INVALID",
                "l8_handoff_eligible": False,
            },
            "bootstrap must be exactly 10000",
        )
    if not bundle_path.is_file():
        return (
            2,
            {
                "status": "NOT_SCOREABLE",
                "reason": "RECEIPT_BOUND_SCORE_BUNDLE_MISSING",
                "l8_handoff_eligible": False,
            },
            f"score bundle missing: {bundle_path}",
        )
    from .l7_score import (
        BOOTSTRAP_ITERATIONS,
        GOLD_FAILURE,
        L7ScoreError,
        MINIMUM_SCOREABLE_NATURAL_CASES,
        verify_score_bundle,
    )

    try:
        result = verify_score_bundle(bundle_path)
    except L7ScoreError as exc:
        diagnostic = str(exc)
        reason = (
            "INDEPENDENT_GOLD_MISSING"
            if diagnostic == GOLD_FAILURE
            else "RECEIPT_BOUND_SCORE_BUNDLE_INVALID"
        )
        return (
            2,
            {
                "status": "NOT_SCOREABLE",
                "reason": reason,
                "l8_handoff_eligible": False,
            },
            diagnostic,
        )
    if result.get("scoreable") is not True or result.get("claim_status") != "SCOREABLE":
        blockers = result.get("scoreability_blockers")
        reason = (
            str(blockers[0]).upper()
            if isinstance(blockers, list) and blockers
            else "SCORE_BUNDLE_NOT_SCOREABLE"
        )
        return (
            2,
            {
                "status": "NOT_SCOREABLE",
                "reason": reason,
                "l8_handoff_eligible": False,
            },
            reason,
        )
    case_count = result.get("case_count")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < MINIMUM_SCOREABLE_NATURAL_CASES
    ):
        return (
            2,
            {
                "status": "NOT_SCOREABLE",
                "reason": "MINIMUM_NINE_QUALIFIED_NATURAL_CASES_NOT_MET",
                "l8_handoff_eligible": False,
            },
            f"case_count must be at least {MINIMUM_SCOREABLE_NATURAL_CASES}",
        )
    return (
        0,
        {
            "status": "SCOREABLE",
            "cases": case_count,
            "slots": 9,
            "bootstrap": BOOTSTRAP_ITERATIONS,
            "l8_handoff_eligible": True,
        },
        "",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify L7 G2 discovery or natural-case evidence")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--case-root", type=Path)
    target.add_argument("--native-main-case-root", type=Path)
    target.add_argument("--native-main-qualification-summary", type=Path)
    target.add_argument("--discovery-manifest", type=Path)
    target.add_argument("--bundle", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--godot-binary", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.bundle is not None:
        if any(
            value is not None
            for value in (
                args.candidate_id,
                args.source_root,
                args.godot_binary,
                args.preflight_output,
            )
        ):
            parser.error("--bundle cannot be combined with source preflight arguments")
        code, payload, diagnostic = _score_bundle_status(
            args.bundle,
            bootstrap_iterations=args.bootstrap,
        )
        print(json.dumps(payload, separators=(",", ":"), sort_keys=False))
        if diagnostic:
            print(diagnostic, file=sys.stderr)
        return code
    if args.bootstrap != 10_000:
        parser.error("--bootstrap is only valid with --bundle")
    preflight_values = (args.candidate_id, args.source_root, args.godot_binary)
    if any(value is not None for value in preflight_values) and not all(
        value is not None for value in preflight_values
    ):
        parser.error("--candidate-id, --source-root, and --godot-binary must be supplied together")
    if args.preflight_output is not None and args.source_root is None:
        parser.error("--preflight-output requires source preflight")
    if args.source_root is not None:
        if args.discovery_manifest is None:
            parser.error("source preflight requires --discovery-manifest")
        result = assess_v4_candidate_source_preflight(
            args.discovery_manifest, args.candidate_id, args.source_root, args.godot_binary
        )
    elif args.discovery_manifest is not None:
        result = assess_corpus_discovery_manifest(args.discovery_manifest)
    elif args.native_main_case_root is not None:
        result = assess_native_main_natural_case_bundle(args.native_main_case_root)
    elif args.native_main_qualification_summary is not None:
        result = assess_native_main_qualification_summary(args.native_main_qualification_summary)
    elif args.source_root is None:
        result = assess_natural_case_bundle(args.case_root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.preflight_output is not None:
        try:
            with args.preflight_output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        except FileExistsError:
            parser.error("--preflight-output must not already exist")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
