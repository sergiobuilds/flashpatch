"""Receipt-bound L7 statistics without rank, winner, or release authority."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .competition import (
    ContractError,
    _approved_registry_snapshot_sha256,
    project_independent_gold,
    verify_native_main_independent_gold_return,
)
from .external_league import (
    DECODER_TIMELINE_PARITY_SCHEMA,
    DIRECT_DETECTOR_POPULATION,
    FAIR_RUNTIME_BUNDLE_SCHEMA,
    TOOFLASHY_PARITY_ADAPTER_SCHEMA,
    verify_decoder_timeline_parity,
    verify_fair_runtime_receipts,
)
from .l7_verify import (
    L7VerificationFailure,
    verify_native_main_qualification_summary,
    verify_natural_case_bundle,
)
from .renderer_artifact import (
    RendererArtifactError,
    open_renderer_artifact,
    renderer_rgb_sha256,
)


SCORE_BUNDLE_SCHEMA = "flashpatch-l7-receipt-bound-score-bundle-v1"
SCORE_RECEIPT_SCHEMA = "flashpatch-l7-receipt-bound-statistics-v1"
L8_DETECTOR_HANDOFF_SCHEMA = "flashpatch-l8-identity-free-detector-handoff-v1"
BOOTSTRAP_ITERATIONS = 10_000
MINIMUM_SCOREABLE_NATURAL_CASES = 9
MINIMUM_SCOREABLE_PUBLIC_REPOSITORIES = 3
PRIMARY_BOOTSTRAP_SEED = 0x46504C375052494D
SECONDARY_BOOTSTRAP_SEED = 0x46504C375345434F
CAPABILITY_FAILURE = "FAIL_CLOSED:league:capability_or_endpoint_mismatch"
GOLD_FAILURE = "FAIL_CLOSED:gold:independent_gold_not_verified"
DIRECT_DETECTOR_POPULATION_SHA256 = hashlib.sha256(
    json.dumps(
        list(DIRECT_DETECTOR_POPULATION),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class L7ScoreError(ValueError):
    """A referenced receipt could not authorize a metric input."""


def _fail(message: str = CAPABILITY_FAILURE) -> None:
    raise L7ScoreError(message)


def _load_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                _fail()
            value[key] = child
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda constant: (_ for _ in ()).throw(L7ScoreError()),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, L7ScoreError) as exc:
        raise L7ScoreError(CAPABILITY_FAILURE) from exc
    if not isinstance(value, dict):
        _fail()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_path(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            _fail()


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, *, failure: str = CAPABILITY_FAILURE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(failure)
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(failure)
    return parsed


def _frame_index_list(value: object, *, failure: str = CAPABILITY_FAILURE) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in value
    ):
        _fail(failure)
    if value != sorted(set(value)):
        _fail(failure)
    return list(value)


def _runtime_wall_time_ns(receipt: Mapping[str, object]) -> int:
    candidates: list[int] = []
    for timing in (
        receipt,
        receipt.get("fair_runtime"),
        receipt.get("process"),
    ):
        if not isinstance(timing, Mapping) or "wall_time_ns" not in timing:
            continue
        wall_time = timing.get("wall_time_ns")
        if isinstance(wall_time, bool) or not isinstance(wall_time, int) or wall_time <= 0:
            _fail()
        started = timing.get("started_monotonic_ns")
        finished = timing.get("finished_monotonic_ns")
        if started is not None or finished is not None:
            if (
                isinstance(started, bool)
                or not isinstance(started, int)
                or isinstance(finished, bool)
                or not isinstance(finished, int)
                or finished <= started
                or finished - started != wall_time
            ):
                _fail()
        candidates.append(wall_time)
    if not candidates or any(value != candidates[0] for value in candidates[1:]):
        _fail()
    return candidates[0]


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _artifact(reference: object, *, directory: bool = False) -> Path:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        _fail()
    path_value = reference.get("path")
    expected = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        _fail()
    supplied = Path(path_value)
    _reject_symlink_path(supplied)
    path = supplied.resolve()
    target = path / "natural-case.json" if directory else path
    target_exists = path.is_dir() if directory else path.is_file()
    if not target_exists:
        _fail()
    try:
        if _sha256_file(target) != expected:
            _fail()
    except OSError as exc:
        raise L7ScoreError(CAPABILITY_FAILURE) from exc
    return path


def _canonical_public_repository_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        _fail()
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        _fail()
    host = parsed.hostname.lower() if parsed.hostname else ""
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not host or len(path_parts) < 2:
        _fail()
    owner = path_parts[0].lower()
    repository = path_parts[1]
    if repository.endswith(".git"):
        repository = repository[:-4]
    repository = repository.lower()
    if not owner or not repository:
        _fail()
    return f"{host}/{owner}/{repository}"


def _natural_projection(reference: object) -> dict[str, object]:
    if (
        isinstance(reference, Mapping)
        and set(reference) == {
            "native_main_summary",
            "blind_case_id",
            "frame_artifact_sha256",
        }
    ):
        return _native_main_natural_projection(reference)
    root = _artifact(reference, directory=True)
    try:
        assessment = verify_natural_case_bundle(root)
    except L7VerificationFailure as exc:
        raise L7ScoreError(CAPABILITY_FAILURE) from exc
    if (
        assessment.get("status") != "NOT_SCOREABLE"
        or assessment.get("scoreable") is not False
        or assessment.get("reason") != "independent_gold_missing"
        or assessment.get("case_class") != "natural_public_godot"
    ):
        _fail()
    ledger = _load_json(root / "natural-case.json")
    if (
        assessment.get("ledger_sha256") != _sha256_file(root / "natural-case.json")
        or ledger.get("controlled_mutation") is not False
    ):
        _fail()
    source_ref = ledger.get("source_provenance")
    repository = ledger.get("repository")
    renderer = ledger.get("renderer")
    trace = ledger.get("trace")
    if not all(isinstance(value, Mapping) for value in (source_ref, repository, renderer, trace)):
        _fail()
    source_path = root / str(source_ref.get("path", ""))
    source = _load_json(source_path)
    if source_ref.get("sha256") != _sha256_file(source_path):
        _fail()
    return {
        "assessment": assessment,
        "ledger": ledger,
        "repository": dict(repository),
        "source": source,
        "renderer": dict(renderer),
        "trace": dict(trace),
    }


def _project_root_for_reference(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "docs" / "MASTER-MAP.md").is_file() and (parent / "src" / "flashpatch").is_dir():
            return parent
    _fail()


def _native_summary_bound_file(
    project_root: Path,
    value: object,
    expected_sha256: object,
) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/") or "\x00" in value:
        _fail()
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError:
        _fail()
    if path.is_symlink() or not path.is_file() or expected_sha256 != _sha256_file(path):
        _fail()
    return path


def _native_main_natural_projection(reference: Mapping[str, object]) -> dict[str, object]:
    summary_path = _artifact(reference["native_main_summary"])
    blind_case_id = reference.get("blind_case_id")
    frame_artifact_sha256 = reference.get("frame_artifact_sha256")
    if (
        not isinstance(blind_case_id, str)
        or not blind_case_id
        or not _is_hex64(frame_artifact_sha256)
    ):
        _fail()
    try:
        assessment = verify_native_main_qualification_summary(summary_path)
    except L7VerificationFailure as exc:
        raise L7ScoreError(CAPABILITY_FAILURE) from exc
    if (
        assessment.get("status") != "NOT_SCOREABLE"
        or assessment.get("scoreable") is not False
        or assessment.get("qualified_case_count") != 0
        or assessment.get("external_claim_authorized") is not False
    ):
        _fail()
    summary = _load_json(summary_path)
    results = summary.get("results")
    if not isinstance(results, list):
        _fail()
    matches = [
        row
        for row in results
        if (
            isinstance(row, Mapping)
            and row.get("receipt_created") is True
            and row.get("frame_artifact_sha256") == frame_artifact_sha256
        )
    ]
    if len(matches) != 1:
        _fail()
    row = matches[0]
    project_root = _project_root_for_reference(summary_path)
    frame_path = _native_summary_bound_file(
        project_root,
        row.get("frame_artifact_path"),
        row.get("frame_artifact_sha256"),
    )
    receipt_path = _native_summary_bound_file(
        project_root,
        row.get("raw_receipt_path"),
        row.get("raw_receipt_sha256"),
    )
    try:
        with open_renderer_artifact(frame_path) as artifact:
            rgb_sha256 = renderer_rgb_sha256(artifact.frames)
            timestamps_sha256 = hashlib.sha256(
                artifact.timestamps.copy(order="C").tobytes()
            ).hexdigest()
            frame_count = len(artifact.frames)
    except RendererArtifactError as exc:
        raise L7ScoreError(CAPABILITY_FAILURE) from exc
    if (
        row.get("frame_count") != frame_count
        or row.get("repository_url") is None
        or row.get("revision") is None
    ):
        _fail()
    repository = {
        "url": row.get("repository_url"),
        "revision": row.get("revision"),
        "license": row.get("license"),
        "project_subpath": ".",
    }
    return {
        "assessment": {
            "case_id": blind_case_id,
            "ledger_sha256": _sha256_file(receipt_path),
            "renderer_execution_receipt_sha256": _sha256_file(receipt_path),
            "renderer_rgb_sha256": rgb_sha256,
            "timestamps_sha256": timestamps_sha256,
        },
        "ledger": {
            "controlled_mutation": False,
            "native_main_summary_sha256": _sha256_file(receipt_path),
        },
        "repository": repository,
        "source": {
            "projection_schema": "flashpatch-l7-native-main-natural-projection-v1",
            "source_summary_sha256": _sha256_file(summary_path),
        },
        "renderer": {
            "frame_artifact_sha256": frame_artifact_sha256,
            "frame_count": frame_count,
        },
        "trace": {},
    }


def _gold_projection(reference: object) -> dict[str, Any]:
    if (
        isinstance(reference, Mapping)
        and set(reference) == {"receipt", "trust_policy", "native_main_gold_return"}
    ):
        return _native_main_gold_projection(reference)
    if not isinstance(reference, Mapping) or set(reference) != {"receipt", "trust_policy"}:
        _fail(GOLD_FAILURE)
    try:
        receipt = _artifact(reference.get("receipt"))
        policy = _artifact(reference.get("trust_policy"))
        projected = project_independent_gold(receipt, trust_policy_path=policy)
    except (ContractError, L7ScoreError, OSError, ValueError) as exc:
        raise L7ScoreError(GOLD_FAILURE) from exc
    projected_receipt = projected.get("receipt")
    projected_policy = projected.get("trust_policy")
    if (
        projected.get("schema") != "flashpatch-l7-independent-gold-projection-v1"
        or projected.get("gold_verified") is not True
        or projected.get("case_class") != "natural_external"
        or projected.get("controlled_mutation") is not False
        or projected.get("league_score_authorized") is not False
        or not isinstance(projected_receipt, Mapping)
        or not isinstance(projected_policy, Mapping)
        or set(projected_receipt) != {"path", "sha256"}
        or set(projected_policy) != {
            "path",
            "sha256",
            "policy_id",
            "registry_snapshot_sha256",
        }
        or Path(str(projected_receipt.get("path", ""))).resolve() != receipt
        or Path(str(projected_policy.get("path", ""))).resolve() != policy
        or projected_receipt.get("sha256") != _sha256_file(receipt)
        or projected_policy.get("sha256") != _sha256_file(policy)
        or not isinstance(projected_policy.get("policy_id"), str)
        or not projected_policy.get("policy_id")
        or not _is_hex64(projected_policy.get("registry_snapshot_sha256"))
    ):
        _fail(GOLD_FAILURE)
    return projected


def _resolve_under_directory(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/") or "\x00" in value:
        _fail(GOLD_FAILURE)
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        _fail(GOLD_FAILURE)
    if path.is_symlink() or not path.is_file():
        _fail(GOLD_FAILURE)
    return path


def _native_main_gold_projection(reference: Mapping[str, object]) -> dict[str, Any]:
    receipt = _artifact(reference["receipt"])
    policy = _artifact(reference["trust_policy"])
    native_ref = reference.get("native_main_gold_return")
    if not isinstance(native_ref, Mapping) or set(native_ref) != {
        "intake_receipt",
        "packet_manifest",
        "return_manifest",
    }:
        _fail(GOLD_FAILURE)
    intake = _artifact(native_ref["intake_receipt"])
    packet = _artifact(native_ref["packet_manifest"])
    return_manifest = _artifact(native_ref["return_manifest"])
    return_root = return_manifest.parent
    if return_root.is_symlink() or not return_root.is_dir():
        _fail(GOLD_FAILURE)
    try:
        assessment = verify_native_main_independent_gold_return(
            intake_receipt=intake,
            return_root=return_root,
            packet_manifest=packet,
        )
    except (ContractError, OSError, ValueError) as exc:
        raise L7ScoreError(GOLD_FAILURE) from exc
    if (
        assessment.get("status") != "INDEPENDENT_GOLD_VERIFIED"
        or assessment.get("scoreable") is not False
        or assessment.get("external_claim_authorized") is not False
    ):
        _fail(GOLD_FAILURE)
    try:
        manifest = _load_json(return_manifest)
        gold = _load_json(receipt)
        trust_policy = _load_json(policy)
        intake_payload = _load_json(intake)
    except L7ScoreError as exc:
        raise L7ScoreError(GOLD_FAILURE) from exc
    blind_case_id = gold.get("case_id")
    entries = manifest.get("gold_receipts")
    if not isinstance(blind_case_id, str) or not blind_case_id or not isinstance(entries, list):
        _fail(GOLD_FAILURE)
    matching_entries = [
        row
        for row in entries
        if isinstance(row, Mapping) and row.get("blind_case_id") == blind_case_id
    ]
    if len(matching_entries) != 1:
        _fail(GOLD_FAILURE)
    entry = matching_entries[0]
    if (
        _resolve_under_directory(return_root, entry.get("independent_gold_path")) != receipt
        or entry.get("independent_gold_sha256") != _sha256_file(receipt)
        or _resolve_under_directory(return_root, entry.get("trust_policy_path")) != policy
        or entry.get("trust_policy_sha256") != _sha256_file(policy)
    ):
        _fail(GOLD_FAILURE)
    blinded_rows = intake_payload.get("blinded_inputs")
    if not isinstance(blinded_rows, list):
        _fail(GOLD_FAILURE)
    matching_inputs = [
        row
        for row in blinded_rows
        if isinstance(row, Mapping) and row.get("blind_case_id") == blind_case_id
    ]
    if len(matching_inputs) != 1:
        _fail(GOLD_FAILURE)
    blinded_input = _load_json(
        _resolve_under_directory(intake.parent, matching_inputs[0].get("path"))
    )
    timestamps_path = _resolve_under_directory(
        intake.parent / "adjudicator-packet",
        blinded_input.get("timestamps_path"),
    )
    timestamps_payload = _load_json(timestamps_path)
    timestamps = timestamps_payload.get("timestamps_seconds")
    freeze = gold.get("case_freeze")
    adjudication = gold.get("adjudication")
    policy_id = trust_policy.get("policy_id")
    if (
        gold.get("schema") != "flashpatch-l7-independent-gold-v2"
        or gold.get("case_class") != "natural_external"
        or gold.get("controlled_mutation") is not False
        or not isinstance(freeze, Mapping)
        or not isinstance(adjudication, Mapping)
        or adjudication.get("result") not in {"SAFE", "HAZARDOUS"}
        or not isinstance(adjudication.get("intervals"), list)
        or not isinstance(timestamps, list)
        or not isinstance(policy_id, str)
        or not policy_id
    ):
        _fail(GOLD_FAILURE)
    return {
        "schema": "flashpatch-l7-independent-gold-projection-v1",
        "gold_verified": True,
        "case_id": blind_case_id,
        "case_class": "natural_external",
        "claim_tier": gold.get("claim_tier"),
        "controlled_mutation": False,
        "case_freeze": dict(freeze),
        "decision": adjudication["result"],
        "intervals": list(adjudication["intervals"]),
        "timestamps_seconds": list(timestamps),
        "receipt": {
            "path": str(receipt),
            "sha256": _sha256_file(receipt),
        },
        "trust_policy": {
            "path": str(policy),
            "sha256": _sha256_file(policy),
            "policy_id": policy_id,
            "registry_snapshot_sha256": _approved_registry_snapshot_sha256(),
        },
        "league_score_authorized": False,
    }


def _match_natural_and_gold(
    natural: Mapping[str, object], gold: Mapping[str, object]
) -> None:
    assessment = natural["assessment"]
    repository = natural["repository"]
    source = natural["source"]
    renderer = natural["renderer"]
    trace = natural["trace"]
    freeze = gold.get("case_freeze")
    if not all(
        isinstance(value, Mapping)
        for value in (assessment, repository, source, renderer, trace, freeze)
    ):
        _fail()
    if source.get("projection_schema") == "flashpatch-l7-native-main-natural-projection-v1":
        expected_native = {
            "source_summary_sha256": source.get("source_summary_sha256"),
            "frame_artifact_sha256": renderer.get("frame_artifact_sha256"),
            "renderer_rgb_raw_sha256": assessment.get("renderer_rgb_sha256"),
            "timestamps_f64_sha256": assessment.get("timestamps_sha256"),
            "frame_count": renderer.get("frame_count"),
        }
        if (
            assessment.get("case_id") != gold.get("case_id")
            or any(freeze.get(field) != value for field, value in expected_native.items())
        ):
            _fail()
        return
    expected = {
        "public_repository_url": repository.get("url"),
        "source_revision": repository.get("revision"),
        "license": repository.get("license"),
        "project_subpath": repository.get("project_subpath"),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "trace_sha256": trace.get("sha256"),
        "renderer_execution_receipt_sha256": assessment.get(
            "renderer_execution_receipt_sha256"
        ),
        "renderer_rgb_raw_sha256": assessment.get("renderer_rgb_sha256"),
        "timestamps_sha256": assessment.get("timestamps_sha256"),
        "frame_count": renderer.get("frame_count"),
    }
    if (
        assessment.get("case_id") != gold.get("case_id")
        or any(freeze.get(field) != value for field, value in expected.items())
    ):
        _fail()


def _infer_tooflashy_adapter(stored: Mapping[str, object]) -> Path | None:
    rows = stored.get("comparators")
    if not isinstance(rows, list):
        _fail()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("comparator") != "TooFlashy":
            continue
        decoder = row.get("decoder_timeline")
        artifacts = decoder.get("native_artifacts") if isinstance(decoder, Mapping) else None
        if not isinstance(artifacts, list):
            _fail()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            supplied_path = Path(str(artifact.get("path", "")))
            _reject_symlink_path(supplied_path)
            path = supplied_path.resolve()
            if not path.is_file() or artifact.get("sha256") != _sha256_file(path):
                _fail()
            try:
                payload = _load_json(path)
            except L7ScoreError:
                continue
            if payload.get("schema") == TOOFLASHY_PARITY_ADAPTER_SCHEMA:
                return path
    return None


def _reopen_tool_parity(reference: object) -> dict[str, Any]:
    path = _artifact(reference)
    stored = _load_json(path)
    if stored.get("schema") != DECODER_TIMELINE_PARITY_SCHEMA:
        _fail()
    rows = stored.get("comparators")
    contract = stored.get("canonical_contract")
    if not isinstance(rows, list) or not isinstance(contract, Mapping):
        _fail()
    run_receipts: list[Path] = []
    for row in rows:
        if not isinstance(row, Mapping):
            _fail()
        run_receipts.append(_artifact(row.get("run_receipt")))
    supplied_canonical_video = Path(str(contract.get("canonical_video", {}).get("path", "")))
    supplied_conversion = Path(str(contract.get("conversion_receipt", {}).get("path", "")))
    _reject_symlink_path(supplied_canonical_video)
    _reject_symlink_path(supplied_conversion)
    canonical_video = supplied_canonical_video.resolve()
    conversion = supplied_conversion.resolve()
    adapter = _infer_tooflashy_adapter(stored)
    recomputed = verify_decoder_timeline_parity(
        run_receipts,
        canonical_video,
        conversion,
        tooflashy_adapter_receipt=adapter,
    )
    if stored != recomputed or recomputed.get("failures") != []:
        _fail()
    reopened_rows = recomputed.get("comparators")
    if (
        recomputed.get("detector_population") != list(DIRECT_DETECTOR_POPULATION)
        or not isinstance(reopened_rows, list)
        or len(reopened_rows) != len(DIRECT_DETECTOR_POPULATION)
    ):
        _fail()
    observations: dict[str, dict[str, object]] = {}
    for row in reopened_rows:
        primary = row.get("primary_case_level_endpoint")
        secondary = row.get("secondary_interval_endpoint")
        comparator = row.get("comparator")
        decoder = row.get("decoder_timeline")
        if (
            comparator not in DIRECT_DETECTOR_POPULATION
            or row.get("status") != "VERIFIED"
            or not isinstance(primary, Mapping)
            or primary.get("status") != "VERIFIED"
            or primary.get("prediction") not in {"SAFE", "HAZARDOUS"}
            or not isinstance(secondary, Mapping)
            or not isinstance(decoder, Mapping)
            or decoder.get("parity_status") != "VERIFIED"
        ):
            _fail()
        indices = (
            list(secondary["hazard_frame_indices"])
            if secondary.get("status") == "VERIFIED"
            and isinstance(secondary.get("hazard_frame_indices"), list)
            else None
        )
        if indices is not None:
            indices = _frame_index_list(indices)
        observations[str(comparator)] = {
            "prediction": primary["prediction"],
            "hazard_frame_indices": indices,
        }
    return {
        "receipt": {"path": str(path), "sha256": _sha256_file(path)},
        "contract": dict(contract),
        "observations": observations,
    }


def _reopen_fair_runtime(reference: object) -> dict[str, Any]:
    path = _artifact(reference)
    stored = _load_json(path)
    if stored.get("schema") != FAIR_RUNTIME_BUNDLE_SCHEMA:
        _fail()
    receipt_rows = stored.get("receipts")
    schedule = stored.get("schedule")
    witness = stored.get("external_host_witness")
    if not isinstance(receipt_rows, list) or not isinstance(schedule, Mapping) or not isinstance(witness, Mapping):
        _fail()
    repeat_paths = []
    for row in receipt_rows:
        if not isinstance(row, Mapping):
            _fail()
        supplied_repeat_path = Path(str(row.get("receipt", "")))
        _reject_symlink_path(supplied_repeat_path)
        repeat_path = supplied_repeat_path.resolve()
        if (
            not repeat_path.is_file()
            or row.get("sha256") != _sha256_file(repeat_path)
        ):
            _fail()
        repeat_paths.append(repeat_path)
    supplied_schedule_path = Path(str(schedule.get("path", "")))
    _reject_symlink_path(supplied_schedule_path)
    schedule_path = supplied_schedule_path.resolve()
    if not schedule_path.is_file() or schedule.get("artifact_sha256") != _sha256_file(schedule_path):
        _fail()
    schedule_payload = _load_json(schedule_path)
    input_sha256 = schedule_payload.get("input_sha256")
    if not isinstance(input_sha256, str):
        _fail()
    supplied_request_path = Path(str(witness.get("request", "")))
    supplied_witness_path = Path(str(witness.get("receipt", "")))
    _reject_symlink_path(supplied_request_path)
    _reject_symlink_path(supplied_witness_path)
    request_path = supplied_request_path.resolve()
    witness_path = supplied_witness_path.resolve()
    recomputed = verify_fair_runtime_receipts(
        repeat_paths,
        schedule_receipt=schedule_path,
        external_host_witness={"request": request_path, "receipt": witness_path},
    )
    if stored != recomputed:
        _fail()
    if (
        recomputed.get("receipts_verified") is not True
        or recomputed.get("fair_runtime_verified") is not True
        or recomputed.get("independent_execution_witness_verified") is not True
        or recomputed.get("failures") != []
    ):
        _fail()
    join_rows = recomputed.get("external_slot_child_joins")
    if not isinstance(join_rows, list) or len(join_rows) != 9:
        _fail()
    joined_children: set[tuple[str, int, str]] = set()
    joined_slots: set[int] = set()
    for row in join_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "slot", "comparator", "repeat_ordinal",
            "external_result_sha256", "child_receipt_sha256",
        }:
            _fail()
        slot = row.get("slot")
        comparator = row.get("comparator")
        repeat_ordinal = row.get("repeat_ordinal")
        child_sha256 = row.get("child_receipt_sha256")
        if (
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < 1
            or slot > 9
            or slot in joined_slots
            or comparator not in DIRECT_DETECTOR_POPULATION
            or isinstance(repeat_ordinal, bool)
            or not isinstance(repeat_ordinal, int)
            or repeat_ordinal not in {1, 2, 3}
            or not _is_hex64(row.get("external_result_sha256"))
            or not _is_hex64(child_sha256)
        ):
            _fail()
        joined_slots.add(slot)
        joined_children.add((str(comparator), repeat_ordinal, str(child_sha256)))
    if joined_slots != set(range(1, 10)) or len(joined_children) != 9:
        _fail()
    observations: dict[str, dict[str, object]] = {}
    runtime_wall_times: dict[str, list[int]] = {}
    aggregate_children: set[tuple[str, int, str]] = set()
    for repeat_path in repeat_paths:
        aggregate = _load_json(repeat_path)
        comparator = aggregate.get("comparator")
        runs = aggregate.get("runs")
        if comparator not in DIRECT_DETECTOR_POPULATION or not isinstance(runs, list) or len(runs) != 3:
            _fail()
        repeated: list[dict[str, object]] = []
        runtimes: list[int] = []
        repeat_ordinals: set[int] = set()
        for run in runs:
            if not isinstance(run, Mapping):
                _fail()
            repeat_ordinal = run.get("repeat")
            if (
                isinstance(repeat_ordinal, bool)
                or not isinstance(repeat_ordinal, int)
                or repeat_ordinal not in {1, 2, 3}
                or repeat_ordinal in repeat_ordinals
            ):
                _fail()
            repeat_ordinals.add(repeat_ordinal)
            supplied_child_path = Path(str(run.get("receipt", "")))
            _reject_symlink_path(supplied_child_path)
            child_path = supplied_child_path.resolve()
            child_sha256 = _sha256_file(child_path) if child_path.is_file() else None
            if (
                not child_path.is_file()
                or run.get("receipt_sha256") != child_sha256
            ):
                _fail()
            aggregate_children.add((str(comparator), repeat_ordinal, str(child_sha256)))
            child = _load_json(child_path)
            observation = child.get("observation")
            if not isinstance(observation, Mapping):
                observation = child.get("parsed_observation")
            if not isinstance(observation, Mapping) or observation.get("prediction") not in {"SAFE", "HAZARDOUS"}:
                _fail()
            runtime_ns = _runtime_wall_time_ns(child)
            runtimes.append(runtime_ns)
            indices = observation.get("hazard_frame_indices")
            if indices is not None and (
                not isinstance(indices, list)
                or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
                or indices != sorted(set(indices))
            ):
                _fail()
            repeated.append(
                {
                    "prediction": observation["prediction"],
                    "hazard_frame_indices": list(indices) if isinstance(indices, list) else None,
                }
            )
        if any(value != repeated[0] for value in repeated[1:]):
            _fail("FAIL_CLOSED:parser:terminal_or_repeat_conflict")
        observations[str(comparator)] = repeated[0]
        runtime_wall_times[str(comparator)] = runtimes
    if aggregate_children != joined_children:
        _fail()
    if (
        set(observations) != set(DIRECT_DETECTOR_POPULATION)
        or set(runtime_wall_times) != set(DIRECT_DETECTOR_POPULATION)
        or any(len(values) != 3 for values in runtime_wall_times.values())
    ):
        _fail()
    return {
        "receipt": {"path": str(path), "sha256": _sha256_file(path)},
        "input_sha256": input_sha256,
        "external_slot_child_joins": [dict(row) for row in join_rows],
        "observations": observations,
        "runtime_wall_time_ns": runtime_wall_times,
    }


def _match_case_inputs(
    gold: Mapping[str, object],
    parity: Mapping[str, object],
    fair: Mapping[str, object],
) -> None:
    freeze = gold.get("case_freeze")
    contract = parity.get("contract")
    timestamps = gold.get("timestamps_seconds")
    if not isinstance(freeze, Mapping) or not isinstance(contract, Mapping) or not isinstance(timestamps, list):
        _fail()
    video = contract.get("canonical_video")
    renderer = contract.get("renderer_source")
    frame_map = contract.get("frame_map")
    if not isinstance(video, Mapping) or not isinstance(renderer, Mapping) or not isinstance(frame_map, list):
        _fail()
    expected_timestamps_us = [
        round(_finite_float(value, failure=GOLD_FAILURE) * 1_000_000)
        for value in timestamps
    ]
    observed_timestamps_us = []
    for row in frame_map:
        if not isinstance(row, Mapping):
            _fail()
        timestamp_us = row.get("renderer_timestamp_us")
        if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int):
            _fail()
        observed_timestamps_us.append(timestamp_us)
    if (
        renderer.get("rgb_sha256") != freeze.get("renderer_rgb_raw_sha256")
        or contract.get("frame_count") != freeze.get("frame_count")
        or observed_timestamps_us != expected_timestamps_us
        or fair.get("input_sha256") != video.get("sha256")
    ):
        _fail()


def _gold_frame_indices(gold: Mapping[str, object]) -> list[int]:
    timestamps = gold.get("timestamps_seconds")
    intervals = gold.get("intervals")
    if not isinstance(timestamps, list) or not isinstance(intervals, list):
        _fail(GOLD_FAILURE)
    parsed_timestamps: list[float] = []
    for timestamp in timestamps:
        parsed_timestamps.append(_finite_float(timestamp, failure=GOLD_FAILURE))
    if any(
        previous >= current
        for previous, current in zip(parsed_timestamps, parsed_timestamps[1:])
    ):
        _fail(GOLD_FAILURE)
    parsed_intervals: list[tuple[float, float]] = []
    for interval in intervals:
        if not isinstance(interval, Mapping):
            _fail(GOLD_FAILURE)
        start = interval.get("start_seconds")
        end = interval.get("end_seconds")
        parsed_start = _finite_float(start, failure=GOLD_FAILURE)
        parsed_end = _finite_float(end, failure=GOLD_FAILURE)
        if not parsed_start < parsed_end:
            _fail(GOLD_FAILURE)
        parsed_intervals.append((parsed_start, parsed_end))
    indices: list[int] = []
    for index, timestamp in enumerate(parsed_timestamps):
        if any(start <= timestamp < end for start, end in parsed_intervals):
            indices.append(index)
    return indices


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _secondary_stat(cases: Sequence[tuple[set[int], set[int]]]) -> dict[str, object]:
    tp = sum(len(predicted & gold) for predicted, gold in cases)
    fp = sum(len(predicted - gold) for predicted, gold in cases)
    fn = sum(len(gold - predicted) for predicted, gold in cases)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    onset_errors = [
        abs(min(predicted) - min(gold))
        for predicted, gold in cases
        if predicted and gold
    ]
    return {
        "eligible_case_count": len(cases),
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "false_negative_frames": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_onset_error_frames": (
            sum(onset_errors) / len(onset_errors) if onset_errors else None
        ),
        "onset_comparable_case_count": len(onset_errors),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _seed(base: int, comparator: str) -> int:
    return base ^ int(hashlib.sha256(comparator.encode("utf-8")).hexdigest()[:16], 16)


def _bootstrap_primary(values: Sequence[bool], comparator: str) -> dict[str, object]:
    rng = random.Random(_seed(PRIMARY_BOOTSTRAP_SEED, comparator))
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(BOOTSTRAP_ITERATIONS)
    ]
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": _seed(PRIMARY_BOOTSTRAP_SEED, comparator),
        "metric": "case_level_exact_agreement",
        "percentile_95_interval": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    }


def _bootstrap_secondary(
    values: Sequence[tuple[set[int], set[int]]], comparator: str
) -> dict[str, object]:
    rng = random.Random(_seed(SECONDARY_BOOTSTRAP_SEED, comparator))
    metrics = {"precision": [], "recall": [], "f1": [], "mean_absolute_onset_error_frames": []}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[rng.randrange(len(values))] for _ in values]
        result = _secondary_stat(sample)
        for field in ("precision", "recall", "f1"):
            metrics[field].append(float(result[field]))
        onset = result["mean_absolute_onset_error_frames"]
        if isinstance(onset, (int, float)):
            metrics["mean_absolute_onset_error_frames"].append(float(onset))
    intervals = {
        field: (
            [_percentile(samples, 0.025), _percentile(samples, 0.975)]
            if samples
            else None
        )
        for field, samples in metrics.items()
    }
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": _seed(SECONDARY_BOOTSTRAP_SEED, comparator),
        "lane": "secondary_interval_only",
        "percentile_95_intervals": intervals,
    }


def _runtime_stat(values: Sequence[int]) -> dict[str, object]:
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        )
    ):
        _fail()
    return {
        "repeat_count": len(values),
        "total_wall_time_ns": sum(values),
        "mean_wall_time_ns": sum(values) / len(values),
        "min_wall_time_ns": min(values),
        "max_wall_time_ns": max(values),
    }


def verify_score_bundle(
    bundle_path: Path | str,
    *,
    destination: Path | str | None = None,
) -> dict[str, object]:
    """Recompute separated statistics from receipts; never emit a ranking."""
    supplied_bundle_path = Path(bundle_path)
    _reject_symlink_path(supplied_bundle_path)
    path = supplied_bundle_path.resolve()
    bundle = _load_json(path)
    if set(bundle) != {"schema", "cases"} or bundle.get("schema") != SCORE_BUNDLE_SCHEMA:
        _fail()
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail()
    primary: dict[str, list[bool]] = {name: [] for name in DIRECT_DETECTOR_POPULATION}
    primary_disagreements: dict[str, list[dict[str, object]]] = {
        name: [] for name in DIRECT_DETECTOR_POPULATION
    }
    secondary: dict[str, list[tuple[set[int], set[int]]]] = {
        name: [] for name in DIRECT_DETECTOR_POPULATION
    }
    secondary_case_diagnostics: dict[str, list[dict[str, object]]] = {
        name: [] for name in DIRECT_DETECTOR_POPULATION
    }
    runtime_wall_times: dict[str, list[int]] = {
        name: [] for name in DIRECT_DETECTOR_POPULATION
    }
    case_receipts: list[dict[str, object]] = []
    case_ids: set[str] = set()
    natural_ledger_hashes: set[str] = set()
    renderer_input_hashes: set[tuple[str, str]] = set()
    score_artifact_hashes: set[tuple[str, str]] = set()
    repository_urls: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {
            "natural_case", "independent_gold", "tool_parity", "fair_runtime"
        }:
            _fail()
        natural = _natural_projection(case["natural_case"])
        gold = _gold_projection(case["independent_gold"])
        if (
            not isinstance(natural.get("ledger"), Mapping)
            or natural["ledger"].get("controlled_mutation") is not False
            or gold.get("controlled_mutation") is not False
        ):
            _fail()
        _match_natural_and_gold(natural, gold)
        parity = _reopen_tool_parity(case["tool_parity"])
        fair = _reopen_fair_runtime(case["fair_runtime"])
        _match_case_inputs(gold, parity, fair)
        if (
            not isinstance(parity.get("observations"), Mapping)
            or not isinstance(fair.get("observations"), Mapping)
            or not isinstance(fair.get("runtime_wall_time_ns"), Mapping)
            or set(parity["observations"]) != set(DIRECT_DETECTOR_POPULATION)
            or set(fair["observations"]) != set(DIRECT_DETECTOR_POPULATION)
            or set(fair["runtime_wall_time_ns"]) != set(DIRECT_DETECTOR_POPULATION)
        ):
            _fail()
        case_id_value = gold.get("case_id")
        if not isinstance(case_id_value, str) or not case_id_value:
            _fail()
        case_id = case_id_value
        if case_id in case_ids:
            _fail()
        case_ids.add(case_id)
        repository = natural.get("repository")
        if not isinstance(repository, Mapping):
            _fail()
        repository_revision = repository.get("revision")
        if (
            not isinstance(repository_revision, str)
            or not repository_revision
        ):
            _fail()
        repository_urls.add(_canonical_public_repository_url(repository.get("url")))
        gold_prediction = gold.get("decision")
        if gold_prediction not in {"SAFE", "HAZARDOUS"}:
            _fail(GOLD_FAILURE)
        gold_indices = set(_gold_frame_indices(gold))
        gold_timestamps = [
            _finite_float(value, failure=GOLD_FAILURE)
            for value in gold["timestamps_seconds"]
        ]
        for comparator in DIRECT_DETECTOR_POPULATION:
            parity_observation = parity["observations"].get(comparator)
            fair_observation = fair["observations"].get(comparator)
            if parity_observation != fair_observation or not isinstance(parity_observation, Mapping):
                _fail()
            runtimes = fair["runtime_wall_time_ns"].get(comparator)
            if (
                not isinstance(runtimes, list)
                or len(runtimes) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in runtimes
                )
            ):
                _fail()
            runtime_wall_times[comparator].extend(runtimes)
            predicted = parity_observation.get("prediction")
            if predicted not in {"SAFE", "HAZARDOUS"}:
                _fail()
            exact = predicted == gold_prediction
            primary[comparator].append(exact)
            if not exact:
                error_type = (
                    "false_negative"
                    if gold_prediction == "HAZARDOUS" and predicted == "SAFE"
                    else "false_positive"
                    if gold_prediction == "SAFE" and predicted == "HAZARDOUS"
                    else "label_mismatch"
                )
                primary_disagreements[comparator].append(
                    {
                        "case_id": case_id,
                        "gold_decision": gold_prediction,
                        "predicted_decision": predicted,
                        "error_type": error_type,
                    }
                )
            indices = parity_observation.get("hazard_frame_indices")
            if indices is not None:
                parsed_indices = _frame_index_list(indices)
                if any(index < 0 or index >= len(gold["timestamps_seconds"]) for index in parsed_indices):
                    _fail()
                predicted_indices = set(parsed_indices)
                secondary[comparator].append((predicted_indices, gold_indices))
                onset_error_seconds = (
                    abs(
                        gold_timestamps[min(predicted_indices)]
                        - gold_timestamps[min(gold_indices)]
                    )
                    if predicted_indices and gold_indices
                    else None
                )
                secondary_case_diagnostics[comparator].append(
                    {
                        "case_id": case_id,
                        "predicted_hazard_frame_indices": parsed_indices,
                        "gold_hazard_frame_indices": sorted(gold_indices),
                        "false_positive_frames": sorted(predicted_indices - gold_indices),
                        "false_negative_frames": sorted(gold_indices - predicted_indices),
                        "onset_error_seconds": onset_error_seconds,
                    }
                )
        natural_assessment = natural.get("assessment")
        parity_contract = parity.get("contract")
        canonical_video = (
            parity_contract.get("canonical_video")
            if isinstance(parity_contract, Mapping)
            else None
        )
        fair_joins = fair.get("external_slot_child_joins")
        if (
            not isinstance(natural_assessment, Mapping)
            or not isinstance(canonical_video, Mapping)
            or not isinstance(fair_joins, list)
            or not _is_hex64(natural_assessment.get("ledger_sha256"))
            or not _is_hex64(natural_assessment.get("renderer_rgb_sha256"))
            or not _is_hex64(natural_assessment.get("timestamps_sha256"))
            or not _is_hex64(canonical_video.get("sha256"))
            or not _is_hex64(fair.get("input_sha256"))
        ):
            _fail()
        natural_ledger_sha256 = str(natural_assessment["ledger_sha256"])
        renderer_input_key = (
            str(natural_assessment["renderer_rgb_sha256"]),
            str(natural_assessment["timestamps_sha256"]),
        )
        score_hash_rows = (
            ("independent_gold", str(gold["receipt"]["sha256"])),
            ("tool_parity", str(parity["receipt"]["sha256"])),
            ("fair_runtime", str(fair["receipt"]["sha256"])),
        )
        if natural_ledger_sha256 in natural_ledger_hashes or renderer_input_key in renderer_input_hashes:
            _fail()
        for row in score_hash_rows:
            if row in score_artifact_hashes:
                _fail()
            score_artifact_hashes.add(row)
        natural_ledger_hashes.add(natural_ledger_sha256)
        renderer_input_hashes.add(renderer_input_key)
        case_receipts.append(
            {
                "case_id": case_id,
                "repository_url": repository["url"],
                "repository_revision": repository_revision,
                "detector_population": list(DIRECT_DETECTOR_POPULATION),
                "detector_population_sha256": DIRECT_DETECTOR_POPULATION_SHA256,
                "natural_case_ledger_sha256": natural_ledger_sha256,
                "renderer_rgb_sha256": natural_assessment["renderer_rgb_sha256"],
                "timestamps_sha256": natural_assessment["timestamps_sha256"],
                "canonical_video_sha256": canonical_video["sha256"],
                "fair_runtime_input_sha256": fair["input_sha256"],
                "fair_runtime_external_slot_child_joins_sha256": _canonical_json_sha256(fair_joins),
                "fair_runtime_external_slot_child_join_count": len(fair_joins),
                "independent_gold_receipt_sha256": gold["receipt"]["sha256"],
                "tool_parity_receipt_sha256": parity["receipt"]["sha256"],
                "fair_runtime_receipt_sha256": fair["receipt"]["sha256"],
            }
        )
    primary_statistics = {
        comparator: {
            "case_count": len(values),
            "exact_matches": sum(values),
            "exact_agreement": sum(values) / len(values),
            "false_negative_count": sum(
                1
                for row in primary_disagreements[comparator]
                if row["error_type"] == "false_negative"
            ),
            "false_positive_count": sum(
                1
                for row in primary_disagreements[comparator]
                if row["error_type"] == "false_positive"
            ),
            "disagreement_case_count": len(primary_disagreements[comparator]),
            "disagreement_cases": primary_disagreements[comparator],
            "bootstrap": _bootstrap_primary(values, comparator),
        }
        for comparator, values in primary.items()
    }
    secondary_statistics = {
        comparator: (
            {
                **_secondary_stat(values),
                "bootstrap": _bootstrap_secondary(values, comparator),
            }
            if values
            else {
                "eligible_case_count": 0,
                "status": "NOT_AVAILABLE_NATIVE_TOOL_HAS_NO_INTERVAL_ENDPOINT",
                "bootstrap": None,
            }
        )
        for comparator, values in secondary.items()
    }
    runtime_statistics = {
        comparator: {
            "case_count": len(primary[comparator]),
            **_runtime_stat(values),
        }
        for comparator, values in runtime_wall_times.items()
    }
    scoreability_blockers = []
    if len(cases) < MINIMUM_SCOREABLE_NATURAL_CASES:
        scoreability_blockers.append("minimum_nine_qualified_natural_cases_not_met")
    if len(repository_urls) < MINIMUM_SCOREABLE_PUBLIC_REPOSITORIES:
        scoreability_blockers.append("minimum_three_public_repositories_not_met")
    scoreable = not scoreability_blockers
    result: dict[str, object] = {
        "schema": SCORE_RECEIPT_SCHEMA,
        "status": "RECEIPT_BOUND_STATISTICS_VERIFIED",
        "case_count": len(cases),
        "minimum_scoreable_natural_cases": MINIMUM_SCOREABLE_NATURAL_CASES,
        "minimum_scoreable_public_repositories": MINIMUM_SCOREABLE_PUBLIC_REPOSITORIES,
        "public_repository_count": len(repository_urls),
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "detector_population_sha256": DIRECT_DETECTOR_POPULATION_SHA256,
        "case_receipts": case_receipts,
        "primary_case_level": primary_statistics,
        "secondary_interval": secondary_statistics,
        "secondary_case_diagnostics": secondary_case_diagnostics,
        "runtime": runtime_statistics,
        "bootstrap_lanes_separate": True,
        "claim_status": "SCOREABLE" if scoreable else "NOT_SCOREABLE",
        "scoreable": scoreable,
        "scoreability_blockers": scoreability_blockers,
        "external_claim_authorized": False,
    }
    if destination is not None:
        output = Path(destination).resolve()
        if output.exists():
            raise FileExistsError(f"L7 score receipt already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {**result, "receipt": str(output)}
    return result

def generate_l8_handoff(bundle_path: Path | str, output_dir: Path | str) -> list[Path]:
    """Emit separate identity-free detector and mitigation L8 handoffs when scoreable.

    It never emits a combined winner or grants external superiority.
    """
    supplied_path = Path(bundle_path)
    _reject_symlink_path(supplied_path)
    path = supplied_path.resolve()
    bundle = _load_json(path)
    if (
        bundle.get("schema") != SCORE_RECEIPT_SCHEMA
        or {"rank", "ranking", "winner", "winning", "external_winner"}.intersection(bundle)
    ):
        _fail()
    if not bundle.get("scoreable"):
        from .competition import ContractError
        raise ContractError("NO_HANDOFF: L7 is not scoreable")
    if (
        bundle.get("status") != "RECEIPT_BOUND_STATISTICS_VERIFIED"
        or bundle.get("claim_status") != "SCOREABLE"
        or bundle.get("external_claim_authorized") is not False
        or bundle.get("bootstrap_lanes_separate") is not True
        or bundle.get("detector_population") != list(DIRECT_DETECTOR_POPULATION)
        or bundle.get("detector_population_sha256") != DIRECT_DETECTOR_POPULATION_SHA256
    ):
        _fail()
    case_count = bundle.get("case_count")
    public_repository_count = bundle.get("public_repository_count")
    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < MINIMUM_SCOREABLE_NATURAL_CASES
        or isinstance(public_repository_count, bool)
        or not isinstance(public_repository_count, int)
        or public_repository_count < MINIMUM_SCOREABLE_PUBLIC_REPOSITORIES
    ):
        _fail()
    case_receipts = bundle.get("case_receipts")
    primary = bundle.get("primary_case_level")
    secondary = bundle.get("secondary_interval")
    secondary_diagnostics = bundle.get("secondary_case_diagnostics")
    runtime = bundle.get("runtime")
    if (
        not isinstance(case_receipts, list)
        or len(case_receipts) != case_count
        or not isinstance(primary, Mapping)
        or not isinstance(secondary, Mapping)
        or not isinstance(secondary_diagnostics, Mapping)
        or not isinstance(runtime, Mapping)
        or set(primary) != set(DIRECT_DETECTOR_POPULATION)
        or set(secondary) != set(DIRECT_DETECTOR_POPULATION)
        or set(secondary_diagnostics) != set(DIRECT_DETECTOR_POPULATION)
        or set(runtime) != set(DIRECT_DETECTOR_POPULATION)
    ):
        _fail()

    output_root = Path(output_dir)
    _reject_symlink_path(output_root)
    output = output_root.resolve() / "l8-direct-detector-handoff.json"
    if output.exists():
        raise FileExistsError(f"L8 handoff already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_receipt_sha256 = _sha256_file(path)
    cells = []
    for ordinal, comparator in enumerate(DIRECT_DETECTOR_POPULATION, start=1):
        cell_id = f"detector_cell_{ordinal}"
        cells.append(
            {
                "cell_id": cell_id,
                "cell_ordinal": ordinal,
                "identity_redacted": True,
                "identity_commitment_sha256": _canonical_json_sha256(
                    {
                        "cell_id": cell_id,
                        "cell_ordinal": ordinal,
                        "detector_population_sha256": DIRECT_DETECTOR_POPULATION_SHA256,
                        "source_score_receipt_sha256": source_receipt_sha256,
                    }
                ),
                "primary_case_level": primary[comparator],
                "secondary_interval": secondary[comparator],
                "secondary_case_diagnostics": secondary_diagnostics[comparator],
                "runtime": runtime[comparator],
            }
        )
    handoff = {
        "schema": L8_DETECTOR_HANDOFF_SCHEMA,
        "source_score_receipt_sha256": source_receipt_sha256,
        "lane": "direct_detector",
        "identity_free": True,
        "detector_identity_mapping_included": False,
        "case_count": case_count,
        "public_repository_count": public_repository_count,
        "primary_endpoint": "case_level_exact_agreement",
        "secondary_endpoint": "diagnostic_interval_only",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "external_claim_authorized": False,
        "combined_winner_authorized": False,
        "cells": cells,
    }
    output.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [output]
