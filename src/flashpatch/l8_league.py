"""Deterministic, fail-closed L8 sealed artifact-league engine.

The engine prepares scorer-readable lane packets without identities, validates a
complete four-axis result set, corrects lane severity from common anchors, seals
the aggregate, and permits identity reveal only from that sealed aggregate.
It deliberately performs no scoring and calls no external provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from statistics import median
from typing import Any


CARD_FIELDS = (
    "problem",
    "user",
    "experience",
    "mechanism",
    "required_technology",
    "business_case",
    "build_scope",
    "proof_plan",
    "observed_evidence",
    "claims",
    "inferences",
    "missing_evidence",
)
NARRATIVE_FIELDS = CARD_FIELDS[:8]
EVIDENCE_FIELDS = CARD_FIELDS[8:]
LANES = ("lane-1", "lane-2", "lane-3")
ANCHOR_LEVELS = ("low", "mid", "high")
REPLICAS = 3
SHA256 = re.compile(r"^[0-9a-f]{64}$")
BLIND_ID = re.compile(r"^BLIND-[0-9A-F]{16}$")
PREPARED_SCHEMA = "sealed-artifact-league-freeze-v1"
PACKET_INDEX_SCHEMA = "sealed-artifact-league-packet-index-v1"
PACKET_SCHEMA = "sealed-artifact-league-cell-v1"
AGGREGATE_SCHEMA = "sealed-artifact-league-aggregate-v1"
REVEAL_SCHEMA = "sealed-artifact-league-reveal-v1"


class L8LeagueError(ValueError):
    """An L8 league input or artifact violated the sealed contract."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise L8LeagueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, L8LeagueError):
            raise
        raise L8LeagueError(f"invalid JSON: {path}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise L8LeagueError(f"{context} must have exact keys: {sorted(expected)}")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise L8LeagueError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise L8LeagueError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise L8LeagueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise L8LeagueError(f"{context} must be between {minimum:g} and {maximum:g}")
    return result


def _strings(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise L8LeagueError(f"{context} must be an array of strings")
    return value


def _validate_card(card: Any, context: str) -> dict[str, Any]:
    card = _exact_keys(card, set(CARD_FIELDS), context)
    for field in NARRATIVE_FIELDS:
        _nonempty_string(card[field], f"{context}.{field}")
    for field in EVIDENCE_FIELDS:
        _strings(card[field], f"{context}.{field}")
    return card


def _card_characters(card: dict[str, Any]) -> int:
    return sum(len(card[field]) for field in NARRATIVE_FIELDS) + sum(
        len(item) for field in EVIDENCE_FIELDS for item in card[field]
    )


def _validate_parity_card(card: dict[str, Any], parity: dict[str, Any], context: str) -> None:
    for field in NARRATIVE_FIELDS:
        length = len(card[field])
        if not parity["field_min_characters"] <= length <= parity["field_max_characters"]:
            raise L8LeagueError(f"{context}.{field} violates field information parity")
    total = _card_characters(card)
    if not parity["min_total_characters"] <= total <= parity["max_total_characters"]:
        raise L8LeagueError(f"{context} violates total information parity")


def _validate_preregistration(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "decision",
        "population_label",
        "evidence_cutoff",
        "information_parity",
        "scorer_model_identifiers",
        "disagreement_rule",
        "tie_rule",
        "audit_promotion_count",
        "pairwise_plan",
        "source_locked_sensitivity_plan",
        "parity_reconstructed_sensitivity_plan",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise L8LeagueError("preregistration is missing required frozen fields")
    if value.get("evaluation_class", "ARTIFACT") != "ARTIFACT":
        raise L8LeagueError("L8 engine requires evaluation_class=ARTIFACT")
    for field in ("decision", "population_label", "evidence_cutoff"):
        _nonempty_string(value[field], f"preregistration.{field}")
    models = _strings(value["scorer_model_identifiers"], "scorer_model_identifiers")
    if not models or any(not model.strip() for model in models) or len(models) != len(set(models)):
        raise L8LeagueError("scorer_model_identifiers must be unique and non-empty")
    _integer(value["audit_promotion_count"], "audit_promotion_count", minimum=1)
    parity = _exact_keys(
        value["information_parity"],
        {
            "min_total_characters",
            "max_total_characters",
            "field_min_characters",
            "field_max_characters",
            "narrative_fields",
        },
        "information_parity",
    )
    for field in (
        "min_total_characters",
        "max_total_characters",
        "field_min_characters",
        "field_max_characters",
    ):
        _integer(parity[field], f"information_parity.{field}", minimum=1)
    if parity["min_total_characters"] > parity["max_total_characters"]:
        raise L8LeagueError("invalid total information parity bounds")
    if parity["field_min_characters"] > parity["field_max_characters"]:
        raise L8LeagueError("invalid field information parity bounds")
    if parity["narrative_fields"] != list(NARRATIVE_FIELDS):
        raise L8LeagueError("information parity narrative fields do not match card schema")
    rules = _exact_keys(
        value["disagreement_rule"],
        {"total_raw_range", "axis_raw_range", "pairwise_position_win_rate"},
        "disagreement_rule",
    )
    _number(rules["total_raw_range"], "total_raw_range", minimum=0, maximum=100)
    _number(rules["axis_raw_range"], "axis_raw_range", minimum=0, maximum=25)
    _number(
        rules["pairwise_position_win_rate"],
        "pairwise_position_win_rate",
        minimum=0.5,
        maximum=1,
    )
    for field in (
        "tie_rule",
        "pairwise_plan",
        "source_locked_sensitivity_plan",
        "parity_reconstructed_sensitivity_plan",
    ):
        if value[field] in (None, "", [], {}):
            raise L8LeagueError(f"preregistration.{field} must be frozen")
    pairwise_contract = value.get("pairwise_contract", "none")
    if pairwise_contract not in {"none", "two_candidate_reversed"}:
        raise L8LeagueError("preregistration.pairwise_contract is invalid")
    return value, parity


def _validate_rubric(value: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict) or set(value) != {"axes", "hard_gates"}:
        raise L8LeagueError("rubric must contain only axes and hard_gates")
    axes = value["axes"]
    if not isinstance(axes, list) or len(axes) != 4:
        raise L8LeagueError("rubric must define exactly four scored axes")
    axis_ids: list[str] = []
    for index, axis in enumerate(axes):
        axis = _exact_keys(
            axis, {"id", "label", "min_score", "max_score", "anchors"}, f"axis[{index}]"
        )
        axis_id = _nonempty_string(axis["id"], f"axis[{index}].id")
        if axis_id in axis_ids:
            raise L8LeagueError("rubric axis ids must be unique")
        axis_ids.append(axis_id)
        if axis["min_score"] != 0 or axis["max_score"] != 25:
            raise L8LeagueError("every rubric axis must span 0 through 25")
        _nonempty_string(axis["label"], f"axis[{index}].label")
        anchors = axis["anchors"]
        if not isinstance(anchors, dict) or not {"0", "25"}.issubset(anchors):
            raise L8LeagueError("every rubric axis requires 0 and 25 scoring anchors")
        _nonempty_string(anchors["0"], f"axis[{index}].anchors.0")
        _nonempty_string(anchors["25"], f"axis[{index}].anchors.25")
    if not isinstance(value["hard_gates"], list):
        raise L8LeagueError("rubric.hard_gates must be an array")
    return value, axis_ids


def _identity_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.casefold()] if len(value.strip()) >= 3 else []
    if isinstance(value, dict):
        return [item for child in value.values() for item in _identity_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _identity_strings(child)]
    return []


def _reject_identity_leak(value: Any, identities: list[Any], context: str) -> None:
    public = _canonical(value).decode("utf-8").casefold()
    for identity in identities:
        for secret in _identity_strings(identity):
            if secret in public:
                raise L8LeagueError(f"identity leakage in {context}")


def _validate_inputs(
    candidates: Any,
    anchors: Any,
    rubric: Any,
    preregistration: Any,
    reconstruction: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str], dict[str, Any]]:
    preregistration, parity = _validate_preregistration(preregistration)
    rubric, axis_ids = _validate_rubric(rubric)
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise L8LeagueError("candidates must be an array with at least two entries")
    cards: list[dict[str, Any]] = []
    identities: list[Any] = []
    card_hashes: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate = _exact_keys(candidate, {"identity", "card"}, f"candidate[{index}]")
        identity = _exact_keys(
            candidate["identity"], {"name", "url"}, f"candidate[{index}].identity"
        )
        _nonempty_string(identity["name"], f"candidate[{index}].identity.name")
        _nonempty_string(identity["url"], f"candidate[{index}].identity.url")
        card = _validate_card(candidate["card"], f"candidate[{index}].card")
        _validate_parity_card(card, parity, f"candidate[{index}].card")
        _reject_identity_leak(card, [candidate["identity"]], f"candidate[{index}].card")
        identities.append(candidate["identity"])
        cards.append(card)
        card_hashes.append(_hash(card))
    if len(card_hashes) != len(set(card_hashes)):
        raise L8LeagueError("duplicate candidate card in canonical set")
    if len({_hash(identity) for identity in identities}) != len(identities):
        raise L8LeagueError("candidate identities must be unique")

    if not isinstance(reconstruction, list) or len(reconstruction) != len(candidates):
        raise L8LeagueError("reconstruction must contain one entry per candidate")
    budgets: set[int] = set()
    for index, entry in enumerate(reconstruction):
        entry = _exact_keys(
            entry,
            {
                "candidate_index",
                "status",
                "source_tier",
                "research_budget_units",
                "assumptions",
                "card_sha256",
            },
            f"reconstruction[{index}]",
        )
        if entry["candidate_index"] != index or entry["status"] != "SCOREABLE":
            raise L8LeagueError("reconstruction accepts SCOREABLE entries in source order only")
        _nonempty_string(entry["source_tier"], f"reconstruction[{index}].source_tier")
        budgets.add(_integer(entry["research_budget_units"], "research_budget_units", minimum=1))
        _strings(entry["assumptions"], f"reconstruction[{index}].assumptions")
        if entry["card_sha256"] != card_hashes[index]:
            raise L8LeagueError(f"reconstruction[{index}] card hash mismatch")
    if len(budgets) != 1:
        raise L8LeagueError("research budget parity violated")

    if not isinstance(anchors, list) or len(anchors) != 3:
        raise L8LeagueError("exactly three low/mid/high anchors are required")
    normalized_anchors: list[dict[str, Any]] = []
    seen_levels: set[str] = set()
    for index, anchor in enumerate(anchors):
        anchor = _exact_keys(anchor, {"level", "expected_total", "card"}, f"anchor[{index}]")
        level = "mid" if anchor["level"] == "middle" else anchor["level"]
        if level not in ANCHOR_LEVELS or level in seen_levels:
            raise L8LeagueError("anchors must contain unique low, mid, and high levels")
        seen_levels.add(level)
        expected = _number(anchor["expected_total"], "anchor.expected_total", minimum=0, maximum=100)
        card = _validate_card(anchor["card"], f"anchor[{index}].card")
        _validate_parity_card(card, parity, f"anchor[{index}].card")
        normalized_anchors.append({"level": level, "expected_total": expected, "card": card})
    normalized_anchors.sort(key=lambda item: ANCHOR_LEVELS.index(item["level"]))
    totals = [item["expected_total"] for item in normalized_anchors]
    if not totals[0] < totals[1] < totals[2]:
        raise L8LeagueError("low/mid/high anchor totals must be strictly increasing")
    return cards, normalized_anchors, rubric, axis_ids, preregistration


def _blind_ids(cards: list[dict[str, Any]], identities: list[Any], seed: int) -> list[str]:
    result: list[str] = []
    for index, (card, identity) in enumerate(zip(cards, identities, strict=True)):
        digest = hashlib.sha256(
            _canonical({"seed": seed, "index": index, "card": card, "identity": identity})
        ).hexdigest()[:16].upper()
        result.append(f"BLIND-{digest}")
    if len(result) != len(set(result)):
        raise L8LeagueError("blind id collision")
    return result


def _stable_order(values: list[Any], *, seed: int, label: str) -> list[Any]:
    """Order JSON values by a SHA-256 key, independent of Python RNG versions."""
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            _canonical({"seed": seed, "label": label, "value": value})
        ).digest(),
    )


def prepare_artifact_league(
    *,
    candidates_path: Path,
    anchors_path: Path,
    rubric_path: Path,
    preregistration_path: Path,
    reconstruction_path: Path,
    out: Path,
    seed: int,
    replicas: int = REPLICAS,
) -> dict[str, Any]:
    """Validate and atomically prepare a sealed three-lane artifact league."""
    if replicas != REPLICAS:
        raise L8LeagueError("L8 requires exactly three replicas across three lanes")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise L8LeagueError("seed must be an integer")
    if out.exists():
        raise L8LeagueError("league output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = _load(candidates_path)
    anchors_input = _load(anchors_path)
    rubric_input = _load(rubric_path)
    preregistration_input = _load(preregistration_path)
    reconstruction = _load(reconstruction_path)
    cards, anchors, rubric, axis_ids, preregistration = _validate_inputs(
        candidates, anchors_input, rubric_input, preregistration_input, reconstruction
    )
    pairwise_contract = preregistration.get("pairwise_contract", "none")
    if pairwise_contract == "two_candidate_reversed" and len(cards) != 2:
        raise L8LeagueError("two_candidate_reversed requires exactly two candidates")
    identities = [candidate["identity"] for candidate in candidates]
    blind_ids = _blind_ids(cards, identities, seed)
    shuffled = _stable_order(list(blind_ids), seed=seed, label="candidate-order")
    positions = {blind_id: index for index, blind_id in enumerate(shuffled)}
    cards_by_id = dict(zip(blind_ids, cards, strict=True))
    assignments: list[dict[str, Any]] = []
    serial = 1
    for replica in range(REPLICAS):
        orientation = "FORWARD" if replica % 2 == 0 else "REVERSE"
        ordered = shuffled if orientation == "FORWARD" else list(reversed(shuffled))
        for blind_id in ordered:
            lane = LANES[(positions[blind_id] + replica) % len(LANES)]
            assignments.append(
                {
                    "assignment_id": f"A-{serial:06d}",
                    "kind": "candidate",
                    "blind_id": blind_id,
                    "lane": lane,
                    "replica": replica + 1,
                    "orientation": orientation,
                }
            )
            serial += 1
    for lane in LANES:
        for anchor in anchors:
            assignments.append(
                {
                    "assignment_id": f"H-{lane[-1]}-{anchor['level'].upper()}",
                    "kind": "anchor",
                    "blind_id": f"ANCHOR-{anchor['level'].upper()}",
                    "lane": lane,
                    "replica": None,
                    "orientation": "ANCHOR",
                    "expected_total": anchor["expected_total"],
                }
            )

    engine_hash = _file_hash(Path(__file__))
    private_mapping = {
        "schema": "flashpatch-l8-private-identity-mapping-v1",
        "identities": dict(zip(blind_ids, identities, strict=True)),
    }
    private_reconstruction = {
        "schema": "flashpatch-l8-private-reconstruction-v1",
        "entries": reconstruction,
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        packet_entries: list[dict[str, Any]] = []
        anchor_cards = {f"ANCHOR-{item['level'].upper()}": item["card"] for item in anchors}
        for lane in LANES:
            lane_assignments = [item for item in assignments if item["lane"] == lane]
            packet_items: list[dict[str, Any]] = []
            for assignment in lane_assignments:
                card = (
                    cards_by_id[assignment["blind_id"]]
                    if assignment["kind"] == "candidate"
                    else anchor_cards[assignment["blind_id"]]
                )
                packet_items.append({**assignment, "card": card})
            packet_items = _stable_order(packet_items, seed=seed, label=f"packet:{lane}")
            packet = {
                "schema": PACKET_SCHEMA,
                "lane": lane,
                "rubric": rubric,
                "axis_ids": axis_ids,
                "items": packet_items,
            }
            if pairwise_contract == "two_candidate_reversed":
                left, right = sorted(blind_ids)
                packet["pairwise_requirements"] = (
                    [{"pair_id": "PAIRWISE-AB", "left_blind_id": left, "right_blind_id": right}]
                    if lane == "lane-1"
                    else ([{"pair_id": "PAIRWISE-BA", "left_blind_id": right, "right_blind_id": left}]
                          if lane == "lane-2" else [])
                )
            _reject_identity_leak(packet, identities, f"{lane} packet")
            packet_path = temporary / "cells" / f"{lane}.json"
            _write(packet_path, packet)
            packet_entries.append(
                {"lane": lane, "path": f"cells/{lane}.json", "sha256": _file_hash(packet_path)}
            )
        packet_index = {"schema": PACKET_INDEX_SCHEMA, "packets": packet_entries}
        _reject_identity_leak(packet_index, identities, "packet index")
        _write(temporary / "cell-packets.json", packet_index)
        _write(temporary / "private" / "identity-mapping.json", private_mapping)
        _write(temporary / "private" / "reconstruction-manifest.json", private_reconstruction)
        manifest = {
            "schema": PREPARED_SCHEMA,
            "evaluation_class": "ARTIFACT",
            "state": "FROZEN",
            "seed": seed,
            "replicas": REPLICAS,
            "lanes": list(LANES),
            "axis_ids": axis_ids,
            "candidate_count": len(cards),
            "anchor_levels": list(ANCHOR_LEVELS),
            "research_budget_units": reconstruction[0]["research_budget_units"],
            "assignments": assignments,
            "input_sha256": {
                "candidates": _hash(candidates),
                "anchors": _hash(anchors_input),
                "rubric": _hash(rubric_input),
                "preregistration": _hash(preregistration_input),
            },
            "private_commitments": {
                "identity_mapping_sha256": _file_hash(
                    temporary / "private" / "identity-mapping.json"
                ),
                "reconstruction_manifest_sha256": _file_hash(
                    temporary / "private" / "reconstruction-manifest.json"
                ),
            },
            "packet_index_sha256": _file_hash(temporary / "cell-packets.json"),
            "engine_sha256": engine_hash,
            "aggregate_policy": {
                "disagreement_rule": preregistration["disagreement_rule"],
                "tie_rule": preregistration["tie_rule"],
                "pairwise_plan": preregistration["pairwise_plan"],
                "pairwise_contract": pairwise_contract,
                "evidence_cutoff": preregistration["evidence_cutoff"],
            },
        }
        _reject_identity_leak(manifest, identities, "freeze manifest")
        _write(temporary / "freeze-manifest.json", manifest)
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _open_prepared(run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load(run / "freeze-manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != PREPARED_SCHEMA:
        raise L8LeagueError("invalid L8 freeze manifest")
    if manifest.get("state") != "FROZEN" or manifest.get("lanes") != list(LANES):
        raise L8LeagueError("league is not frozen with exactly three lanes")
    if manifest.get("replicas") != REPLICAS or manifest.get("anchor_levels") != list(ANCHOR_LEVELS):
        raise L8LeagueError("league replica or anchor contract mismatch")
    if manifest.get("engine_sha256") != _file_hash(Path(__file__)):
        raise L8LeagueError("L8 engine changed after league freeze")
    if not isinstance(manifest.get("input_sha256"), dict) or any(
        not isinstance(value, str) or SHA256.fullmatch(value) is None
        for value in manifest["input_sha256"].values()
    ):
        raise L8LeagueError("freeze input commitments are invalid")
    axis_ids = manifest.get("axis_ids")
    if (
        not isinstance(axis_ids, list)
        or len(axis_ids) != 4
        or len(set(axis_ids)) != 4
        or any(not isinstance(axis, str) or not axis for axis in axis_ids)
    ):
        raise L8LeagueError("freeze manifest must bind exactly four axes")
    assignments = manifest.get("assignments")
    candidate_count = manifest.get("candidate_count")
    if (
        not isinstance(assignments, list)
        or isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 2
        or len(assignments) != candidate_count * REPLICAS + len(LANES) * len(ANCHOR_LEVELS)
    ):
        raise L8LeagueError("freeze assignment plan is incomplete")
    assignment_ids: set[str] = set()
    candidate_assignments: dict[str, list[dict[str, Any]]] = {}
    anchor_cells: set[tuple[str, str]] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise L8LeagueError("freeze assignment is invalid")
        common = {"assignment_id", "kind", "blind_id", "lane", "replica", "orientation"}
        if assignment.get("kind") == "candidate":
            if set(assignment) != common or BLIND_ID.fullmatch(str(assignment.get("blind_id"))) is None:
                raise L8LeagueError("candidate assignment schema is invalid")
            replica = assignment.get("replica")
            if replica not in {1, 2, 3} or assignment.get("orientation") != (
                "FORWARD" if replica % 2 else "REVERSE"
            ):
                raise L8LeagueError("candidate replica orientation is invalid")
            candidate_assignments.setdefault(assignment["blind_id"], []).append(assignment)
        elif assignment.get("kind") == "anchor":
            if set(assignment) != common | {"expected_total"}:
                raise L8LeagueError("anchor assignment schema is invalid")
            level = str(assignment.get("blind_id", "")).removeprefix("ANCHOR-").lower()
            if (
                level not in ANCHOR_LEVELS
                or assignment.get("replica") is not None
                or assignment.get("orientation") != "ANCHOR"
            ):
                raise L8LeagueError("anchor assignment is invalid")
            _number(
                assignment.get("expected_total"),
                "anchor assignment expected_total",
                minimum=0,
                maximum=100,
            )
            anchor_cells.add((assignment.get("lane"), level))
        else:
            raise L8LeagueError("unknown assignment kind")
        assignment_id = assignment.get("assignment_id")
        if (
            not isinstance(assignment_id, str)
            or not assignment_id
            or assignment_id in assignment_ids
            or assignment.get("lane") not in LANES
        ):
            raise L8LeagueError("assignment identity or lane is invalid")
        assignment_ids.add(assignment_id)
    if len(candidate_assignments) != candidate_count or any(
        {item["lane"] for item in entries} != set(LANES)
        or {item["replica"] for item in entries} != {1, 2, 3}
        for entries in candidate_assignments.values()
    ):
        raise L8LeagueError("candidate lane assignment is not balanced")
    if anchor_cells != {(lane, level) for lane in LANES for level in ANCHOR_LEVELS}:
        raise L8LeagueError("every lane must contain the same low/mid/high anchors")
    packet_index_path = run / "cell-packets.json"
    packet_index = _load(packet_index_path)
    if _file_hash(packet_index_path) != manifest.get("packet_index_sha256"):
        raise L8LeagueError("cell packet index changed after freeze")
    if not isinstance(packet_index, dict) or packet_index.get("schema") != PACKET_INDEX_SCHEMA:
        raise L8LeagueError("invalid cell packet index")
    packets = packet_index.get("packets")
    if not isinstance(packets, list) or [item.get("lane") for item in packets] != list(LANES):
        raise L8LeagueError("cell packet lanes are incomplete or unbalanced")
    for item in packets:
        if set(item) != {"lane", "path", "sha256"}:
            raise L8LeagueError("invalid cell packet index entry")
        path = run / item["path"]
        if not path.is_file() or path.is_symlink() or _file_hash(path) != item["sha256"]:
            raise L8LeagueError("cell packet changed after freeze")
        packet = _load(path)
        if packet.get("schema") != PACKET_SCHEMA or packet.get("lane") != item["lane"]:
            raise L8LeagueError("cell packet binding mismatch")
        packet_items = packet.get("items")
        expected_assignments = {
            assignment["assignment_id"]: assignment
            for assignment in assignments
            if assignment["lane"] == item["lane"]
        }
        if not isinstance(packet_items, list) or len(packet_items) != candidate_count + 3:
            raise L8LeagueError("cell packet assignment coverage is invalid")
        observed_ids: set[str] = set()
        for packet_item in packet_items:
            if not isinstance(packet_item, dict) or "card" not in packet_item:
                raise L8LeagueError("cell packet item schema is invalid")
            assignment_id = packet_item.get("assignment_id")
            if assignment_id in observed_ids or assignment_id not in expected_assignments:
                raise L8LeagueError("cell packet has duplicate or unknown assignment")
            observed_ids.add(assignment_id)
            if {key: value for key, value in packet_item.items() if key != "card"} != (
                expected_assignments[assignment_id]
            ):
                raise L8LeagueError("cell packet assignment changed after freeze")
            _validate_card(packet_item["card"], f"cell packet {assignment_id} card")
        if observed_ids != set(expected_assignments):
            raise L8LeagueError("cell packet is missing a frozen assignment")
    mapping_path = run / "private" / "identity-mapping.json"
    mapping = _load(mapping_path)
    if _file_hash(mapping_path) != manifest.get("private_commitments", {}).get(
        "identity_mapping_sha256"
    ):
        raise L8LeagueError("private identity mapping changed after freeze")
    identities = mapping.get("identities") if isinstance(mapping, dict) else None
    if not isinstance(identities, dict) or len(identities) != manifest.get("candidate_count"):
        raise L8LeagueError("private identity mapping is invalid")
    if set(identities) != set(candidate_assignments) or any(
        not isinstance(identity, dict) for identity in identities.values()
    ):
        raise L8LeagueError("private identity mapping does not match frozen candidates")
    _reject_identity_leak(manifest, list(identities.values()), "freeze manifest")
    for item in packets:
        _reject_identity_leak(
            _load(run / item["path"]), list(identities.values()), f"{item['lane']} packet"
        )
    reconstruction_path = run / "private" / "reconstruction-manifest.json"
    if _file_hash(reconstruction_path) != manifest.get("private_commitments", {}).get(
        "reconstruction_manifest_sha256"
    ):
        raise L8LeagueError("private reconstruction manifest changed after freeze")
    return manifest, packet_index, mapping


def _result_document(value: Any) -> tuple[list[Any], list[Any]]:
    if isinstance(value, list):
        return value, []
    if isinstance(value, dict) and set(value).issubset({"results", "pairwise_results"}):
        if set(value) == {"pairwise_results"}:
            raise L8LeagueError("results document is missing results")
        results = value.get("results")
        pairwise = value.get("pairwise_results", [])
        if isinstance(results, list) and isinstance(pairwise, list):
            return results, pairwise
    raise L8LeagueError("results must be an array or a results/pairwise_results object")


def _validate_results(
    manifest: dict[str, Any], value: Any
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results, pairwise = _result_document(value)
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise L8LeagueError("freeze manifest assignments are invalid")
    by_assignment = {item["assignment_id"]: item for item in assignments}
    if len(results) != len(by_assignment):
        raise L8LeagueError("results must cover every planned assignment exactly once")
    axis_ids = manifest.get("axis_ids")
    seen: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(results):
        result = _exact_keys(
            result,
            {"assignment_id", "blind_id", "scores", "reason", "evidence_gaps"},
            f"result[{index}]",
        )
        assignment_id = result["assignment_id"]
        if assignment_id not in by_assignment or assignment_id in seen:
            raise L8LeagueError("missing, duplicate, or unknown result assignment")
        assignment = by_assignment[assignment_id]
        if result["blind_id"] != assignment["blind_id"]:
            raise L8LeagueError("result blind identity does not match frozen assignment")
        if not isinstance(result["scores"], dict) or set(result["scores"]) != set(axis_ids):
            raise L8LeagueError("every result must cover the exact four axes in frozen order")
        for axis_id, score in result["scores"].items():
            _number(score, f"result[{index}].scores.{axis_id}", minimum=0, maximum=25)
        _nonempty_string(result["reason"], f"result[{index}].reason")
        _strings(result["evidence_gaps"], f"result[{index}].evidence_gaps")
        seen[assignment_id] = result
    candidate_ids = {
        item["blind_id"] for item in assignments if item.get("kind") == "candidate"
    }
    validated_pairwise: list[dict[str, Any]] = []
    seen_pairwise_ids: set[str] = set()
    for index, result in enumerate(pairwise):
        result = _exact_keys(
            result,
            {"pair_id", "left_blind_id", "right_blind_id", "winner_blind_id"},
            f"pairwise_result[{index}]",
        )
        pair_id = _nonempty_string(result["pair_id"], f"pairwise_result[{index}].pair_id")
        left, right, winner = (
            result["left_blind_id"],
            result["right_blind_id"],
            result["winner_blind_id"],
        )
        if pair_id in seen_pairwise_ids or left == right or left not in candidate_ids or right not in candidate_ids:
            raise L8LeagueError("pairwise result has invalid or duplicate identities")
        if winner is not None and winner not in {left, right}:
            raise L8LeagueError("pairwise winner must be left, right, or null")
        seen_pairwise_ids.add(pair_id)
        validated_pairwise.append(result)
    contract = manifest.get("aggregate_policy", {}).get("pairwise_contract")
    if contract == "two_candidate_reversed":
        left, right = sorted(candidate_ids)
        expected = {
            ("PAIRWISE-AB", left, right),
            ("PAIRWISE-BA", right, left),
        }
        observed = {
            (item["pair_id"], item["left_blind_id"], item["right_blind_id"])
            for item in validated_pairwise
        }
        if observed != expected:
            raise L8LeagueError("pairwise results must contain frozen A/B and B/A orientations")
    return seen, validated_pairwise


def _orientation_warnings(
    pairwise: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not pairwise:
        return ([{"code": "ORIENTATION_NOT_MEASURED"}], {"measured": False})
    decisive = [item for item in pairwise if item["winner_blind_id"] is not None]
    left_wins = sum(item["winner_blind_id"] == item["left_blind_id"] for item in decisive)
    left_rate = left_wins / len(decisive) if decisive else 0.0
    warnings: list[dict[str, Any]] = []
    if decisive and max(left_rate, 1 - left_rate) > threshold:
        warnings.append(
            {
                "code": "PAIRWISE_ORIENTATION_BIAS",
                "left_position_win_rate": left_rate,
                "threshold": threshold,
            }
        )
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for item in pairwise:
        unordered = tuple(sorted((item["left_blind_id"], item["right_blind_id"])))
        grouped.setdefault(unordered, set()).add((item["left_blind_id"], item["right_blind_id"]))
    incomplete = [list(pair) for pair, orientations in grouped.items() if len(orientations) != 2]
    if incomplete:
        warnings.append({"code": "PAIRWISE_ORIENTATION_INCOMPLETE", "pairs": incomplete})
    return warnings, {
        "measured": True,
        "decisive_count": len(decisive),
        "left_position_win_rate": left_rate if decisive else None,
    }


def aggregate_artifact_league(
    *, run: Path, results_path: Path, out: Path | None = None
) -> dict[str, Any]:
    """Validate complete results, apply anchor correction, and atomically seal."""
    manifest, packet_index, mapping = _open_prepared(run)
    destination = out or run / "aggregate.json"
    if destination.exists():
        raise L8LeagueError("aggregate output already exists")
    result_value = _load(results_path)
    results, pairwise = _validate_results(manifest, result_value)
    assignments = {item["assignment_id"]: item for item in manifest["assignments"]}
    axis_ids = manifest["axis_ids"]
    lane_errors: dict[str, list[float]] = {lane: [] for lane in LANES}
    for assignment_id, result in results.items():
        assignment = assignments[assignment_id]
        if assignment["kind"] == "anchor":
            observed = sum(float(result["scores"][axis]) for axis in axis_ids)
            lane_errors[assignment["lane"]].append(assignment["expected_total"] - observed)
    if any(len(errors) != 3 for errors in lane_errors.values()):
        raise L8LeagueError("every lane requires low/mid/high anchor results")
    corrections = {lane: float(median(errors)) for lane, errors in lane_errors.items()}
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for assignment_id, result in results.items():
        assignment = assignments[assignment_id]
        if assignment["kind"] == "candidate":
            candidates.setdefault(assignment["blind_id"], []).append((assignment, result))
    if any(len(entries) != REPLICAS for entries in candidates.values()):
        raise L8LeagueError("every candidate requires exactly three replica results")
    aggregate_policy = manifest.get("aggregate_policy")
    if not isinstance(aggregate_policy, dict):
        raise L8LeagueError("freeze aggregate policy is missing")
    disagreement_rule = aggregate_policy.get("disagreement_rule")
    if not isinstance(disagreement_rule, dict):
        raise L8LeagueError("freeze disagreement policy is invalid")
    warnings: list[dict[str, Any]] = []
    aggregate_candidates: list[dict[str, Any]] = []
    for blind_id in sorted(candidates):
        entries = candidates[blind_id]
        raw_totals = [sum(float(result["scores"][axis]) for axis in axis_ids) for _, result in entries]
        corrected_totals = [
            min(100.0, max(0.0, total + corrections[assignment["lane"]]))
            for (assignment, _), total in zip(entries, raw_totals, strict=True)
        ]
        raw_axis_medians = {
            axis: float(median(float(result["scores"][axis]) for _, result in entries))
            for axis in axis_ids
        }
        corrected_axis_medians = {
            axis: float(
                median(
                    min(
                        25.0,
                        max(
                            0.0,
                            float(result["scores"][axis])
                            + corrections[assignment["lane"]] / len(axis_ids),
                        ),
                    )
                    for assignment, result in entries
                )
            )
            for axis in axis_ids
        }
        total_range = max(raw_totals) - min(raw_totals)
        axis_ranges = {
            axis: max(float(result["scores"][axis]) for _, result in entries)
            - min(float(result["scores"][axis]) for _, result in entries)
            for axis in axis_ids
        }
        if total_range >= disagreement_rule["total_raw_range"]:
            warnings.append(
                {"code": "TOTAL_DISAGREEMENT_ADJUDICATION_REQUIRED", "blind_id": blind_id}
            )
        for axis, axis_range in axis_ranges.items():
            if axis_range >= disagreement_rule["axis_raw_range"]:
                warnings.append(
                    {
                        "code": "AXIS_DISAGREEMENT_ADJUDICATION_REQUIRED",
                        "blind_id": blind_id,
                        "axis_id": axis,
                    }
                )
        aggregate_candidates.append(
            {
                "blind_id": blind_id,
                "raw_axis_medians": raw_axis_medians,
                "corrected_axis_medians": corrected_axis_medians,
                "raw_total_range": total_range,
                "raw_axis_ranges": axis_ranges,
                "corrected_total_median": float(median(corrected_totals)),
                "evidence_gaps": sorted(
                    {gap for _, result in entries for gap in result["evidence_gaps"]}
                ),
            }
        )
    aggregate_candidates.sort(key=lambda item: (-item["corrected_total_median"], item["blind_id"]))
    rank = 0
    previous_score: float | None = None
    for position, candidate in enumerate(aggregate_candidates, start=1):
        if previous_score is None or candidate["corrected_total_median"] != previous_score:
            rank = position
            previous_score = candidate["corrected_total_median"]
        candidate["rank"] = rank
    orientation_warnings, orientation = _orientation_warnings(
        pairwise, float(disagreement_rule["pairwise_position_win_rate"])
    )
    warnings.extend(orientation_warnings)
    unique_winner = (
        aggregate_candidates[0]["blind_id"]
        if len(aggregate_candidates) == 1
        or aggregate_candidates[0]["corrected_total_median"]
        > aggregate_candidates[1]["corrected_total_median"]
        else None
    )
    normalized_results = {
        "results": [results[item["assignment_id"]] for item in manifest["assignments"]],
        "pairwise_results": sorted(pairwise, key=lambda item: item["pair_id"]),
    }
    sealed_results_path = run / "sealed-results.json"
    if sealed_results_path.exists():
        raise L8LeagueError("sealed results already exist")
    _write(sealed_results_path, normalized_results)
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "state": "SEALED",
        "evaluation_class": "ARTIFACT",
            "population_commitment_sha256": manifest["input_sha256"]["preregistration"],
            "evidence_cutoff": aggregate_policy["evidence_cutoff"],
        "lane_anchor_corrections": corrections,
        "candidates": aggregate_candidates,
        "winner_blind_id": unique_winner,
        "warnings": warnings,
        "orientation": orientation,
        "seal": {
            "freeze_manifest_sha256": _file_hash(run / "freeze-manifest.json"),
            "packet_index_sha256": _hash(packet_index),
            "sealed_results_sha256": _file_hash(sealed_results_path),
            "engine_sha256": manifest["engine_sha256"],
        },
    }
    _reject_identity_leak(aggregate, list(mapping["identities"].values()), "aggregate")
    _write(destination, aggregate)
    _write(
        destination.with_name(destination.name + ".seal.json"),
        {
            "schema": "flashpatch-l8-artifact-league-aggregate-seal-v1",
            "aggregate_sha256": _file_hash(destination),
            "freeze_manifest_sha256": aggregate["seal"]["freeze_manifest_sha256"],
            "sealed_results_sha256": aggregate["seal"]["sealed_results_sha256"],
        },
    )
    return aggregate


def reveal_artifact_league(
    *, run: Path, aggregate_path: Path, out: Path | None = None
) -> dict[str, Any]:
    """Reveal identities only after verifying the frozen run and sealed results."""
    manifest, packet_index, mapping = _open_prepared(run)
    aggregate = _load(aggregate_path)
    if not isinstance(aggregate, dict) or aggregate.get("schema") != AGGREGATE_SCHEMA:
        raise L8LeagueError("invalid aggregate for reveal")
    if aggregate.get("state") != "SEALED":
        raise L8LeagueError("aggregate is not sealed")
    aggregate_seal_path = aggregate_path.with_name(aggregate_path.name + ".seal.json")
    aggregate_seal = _load(aggregate_seal_path)
    if aggregate_seal != {
        "schema": "flashpatch-l8-artifact-league-aggregate-seal-v1",
        "aggregate_sha256": _file_hash(aggregate_path),
        "freeze_manifest_sha256": _file_hash(run / "freeze-manifest.json"),
        "sealed_results_sha256": _file_hash(run / "sealed-results.json"),
    }:
        raise L8LeagueError("aggregate bytes do not match aggregate seal")
    seal = aggregate.get("seal")
    if not isinstance(seal, dict) or seal != {
        "freeze_manifest_sha256": _file_hash(run / "freeze-manifest.json"),
        "packet_index_sha256": _hash(packet_index),
        "sealed_results_sha256": _file_hash(run / "sealed-results.json"),
        "engine_sha256": manifest["engine_sha256"],
    }:
        raise L8LeagueError("aggregate seal does not match frozen league")
    identities = mapping["identities"]
    revealed_candidates: list[dict[str, Any]] = []
    for candidate in aggregate.get("candidates", []):
        blind_id = candidate.get("blind_id")
        if blind_id not in identities:
            raise L8LeagueError("aggregate contains unknown blind identity")
        revealed_candidates.append({**candidate, "identity": identities[blind_id]})
    winner_blind_id = aggregate.get("winner_blind_id")
    revealed = {
        "schema": REVEAL_SCHEMA,
        "state": "REVEALED_AFTER_SEAL",
        "aggregate_sha256": _file_hash(aggregate_path),
        "population_commitment_sha256": aggregate["population_commitment_sha256"],
        "evidence_cutoff": aggregate["evidence_cutoff"],
        "winner": (
            None
            if winner_blind_id is None
            else {"blind_id": winner_blind_id, "identity": identities[winner_blind_id]}
        ),
        "candidates": revealed_candidates,
        "warnings": aggregate["warnings"],
    }
    destination = out or run / "revealed.json"
    if destination.exists():
        raise L8LeagueError("reveal output already exists")
    _write(destination, revealed)
    return revealed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flashpatch.l8_league")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--anchors", type=Path, required=True)
    prepare.add_argument("--rubric", type=Path, required=True)
    prepare.add_argument("--preregistration", type=Path, required=True)
    prepare.add_argument("--reconstruction", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--replicas", type=int, default=REPLICAS)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run", type=Path, required=True)
    aggregate.add_argument("--results", type=Path, required=True)
    aggregate.add_argument("--out", type=Path)
    reveal = commands.add_parser("reveal")
    reveal.add_argument("--run", type=Path, required=True)
    reveal.add_argument("--aggregate", type=Path, required=True)
    reveal.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_artifact_league(
                candidates_path=args.candidates,
                anchors_path=args.anchors,
                rubric_path=args.rubric,
                preregistration_path=args.preregistration,
                reconstruction_path=args.reconstruction,
                out=args.out,
                seed=args.seed,
                replicas=args.replicas,
            )
            output = f"SEALED artifact-league lanes=3 candidates={result['candidate_count']}"
        elif args.command == "aggregate":
            result = aggregate_artifact_league(
                run=args.run, results_path=args.results, out=args.out
            )
            output = f"SEALED aggregate candidates={len(result['candidates'])} warnings={len(result['warnings'])}"
        else:
            result = reveal_artifact_league(
                run=args.run, aggregate_path=args.aggregate, out=args.out
            )
            output = f"REVEALED after seal candidates={len(result['candidates'])}"
    except L8LeagueError as exc:
        print(str(exc), file=os.sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
