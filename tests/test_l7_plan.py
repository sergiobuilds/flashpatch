from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flashpatch.l7_authority import (
    EXPECTED_BOUNDARIES,
    EXPECTED_CONFORMANCE_ORACLE,
    EXPECTED_DIRECT_DETECTOR_POPULATION,
    EXPECTED_SEED_PARTICIPANT_LINES,
    EXPECTED_MASTER_MAP_VERSION,
    EXPECTED_OUTPUTS,
    EXPECTED_PLAN_COMMIT,
    EXPECTED_PLAN_PATH,
    EXPECTED_WRITEBACK,
    L7AuthorityError,
    validate_l7_authority,
)


ROOT = Path(__file__).parents[1]
SEED = ROOT / "specs" / "L7-direct-detector-league-execution.seed.yaml"


def test_l7_seed_is_bound_exclusively_to_its_master_map_leaf() -> None:
    binding = validate_l7_authority(SEED, repository_root=ROOT)

    assert binding.leaf_id == "L7"
    assert binding.master_map_version == EXPECTED_MASTER_MAP_VERSION
    assert binding.plan_path == EXPECTED_PLAN_PATH
    assert binding.plan_commit == EXPECTED_PLAN_COMMIT
    assert binding.writeback == EXPECTED_WRITEBACK
    assert binding.outputs == EXPECTED_OUTPUTS


def test_l7_seed_does_not_retain_superseded_authority_labels() -> None:
    seed_text = SEED.read_text(encoding="utf-8")

    assert "MASTER-MAP v160 and plan v81" in seed_text
    assert "master-map-v160-game-visual-qa-unity-preflight-l8-regression" in seed_text
    assert "MASTER-MAP v90" not in seed_text
    assert "plan v13" not in seed_text
    assert "master-map-v90-authority-rebind" not in seed_text


def test_l7_plan_commit_is_an_ancestor_and_contains_the_bound_plan() -> None:
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PLAN_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    committed = subprocess.run(
        ["git", "show", f"{EXPECTED_PLAN_COMMIT}:{EXPECTED_PLAN_PATH}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    assert committed == (ROOT / EXPECTED_PLAN_PATH).read_bytes()


def test_l7_seed_freezes_kaya_participant_and_excludes_both_iris_identities() -> None:
    seed_text = SEED.read_text(encoding="utf-8")

    assert EXPECTED_DIRECT_DETECTOR_POPULATION == (
        "FlashPatch",
        "KAYA_PSE_DETECTION_CORRECTION_0776EA3",
        "TooFlashy",
    )
    assert EXPECTED_CONFORMANCE_ORACLE == "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU"
    assert "FlashPatch, EA IRIS 1.1.0, and TooFlashy" not in seed_text
    assert "never a direct detector participant, score input, ranking entry, or winner candidate" in seed_text
    assert "historical prototype receipt identity only" in seed_text
    assert "excluded semantic-mismatch baseline" in seed_text


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("authority_leaf_id: L7", "authority_leaf_id: L6"),
        ("authority_leaf_id: L7", "authority_leaf_id: L8"),
        ("authority_leaf_id: L7", "authority_leaf_id: L9"),
        (
            "authority_writeback: [own_leaf_status, status_section, history, evidence_links]",
            "authority_writeback: [own_leaf_status, status_section, history, "
            "chronicle_proposal, evidence_links]",
        ),
        (
            "modifying the Charter or siblings, or authorizing any",
            "modifying the Charter and siblings while authorizing every",
        ),
        (
            "outputs:\n  - src/flashpatch/competition.py",
            "outputs:\n  - src/flashpatch/l6_run.py\n"
            "  - src/flashpatch/competition.py",
        ),
        (
                "binding_plan: {path: docs/plans/2026-08-02-l7-direct-detector-league.md, commit: b930148f088899f3fbf6bc87b1ebbd393ad3c071}",
            "binding_plan: {path: docs/plans/2026-08-02-l7-direct-detector-league.md, commit: deadbee}",
        ),
        (
                "authority_master_map_version: 2026-08-11_v160",
            "authority_master_map_version: 2026-08-02_v81",
        ),
    ],
)
def test_l7_authority_rejects_sibling_charter_output_and_stale_scope(
    tmp_path: Path, old: str, new: str
) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    assert old in seed_text
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(seed_text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(L7AuthorityError):
        validate_l7_authority(hostile_seed, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("key", "expected", "hostile"),
    [
        (
            "l6_controlled_evidence",
            "controlled_only_never_natural_or_score_denominator",
            "natural_external_score_input",
        ),
        (
            "league_separation",
            "detector_and_mitigation_never_share_ranking",
            "combined_detector_mitigation_ranking",
        ),
        ("external_claim_authority", "none", "CLAIM_ELIGIBLE"),
        ("sibling_authority", "none", "L8_and_L9"),
    ],
)
def test_l7_authority_rejects_evidence_lane_and_claim_escalation(
    tmp_path: Path, key: str, expected: str, hostile: str
) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    boundary = f"    {key}: {expected}"
    assert EXPECTED_BOUNDARIES[key] == expected
    assert boundary in seed_text
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(
        seed_text.replace(boundary, f"    {key}: {hostile}", 1),
        encoding="utf-8",
    )

    with pytest.raises(L7AuthorityError, match="boundary"):
        validate_l7_authority(hostile_seed, repository_root=tmp_path)


def test_l7_authority_rejects_map_owned_scope(tmp_path: Path) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(
        seed_text + "\nproject_charter:\n  goal: replace the product boundary\n",
        encoding="utf-8",
    )

    with pytest.raises(L7AuthorityError, match="map-owned top-level keys"):
        validate_l7_authority(hostile_seed, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "FlashPatch, KAYA_PSE_DETECTION_CORRECTION_0776EA3, and TooFlashy 8274e1ea form the fixed direct detector population.",
            "FlashPatch, EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU, and TooFlashy 8274e1ea form the fixed direct detector population.",
        ),
        (
            "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU is a frozen non-scoring conformance oracle and never a direct detector participant, score input, ranking entry, or winner candidate.",
            "EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU is a direct detector participant and score input.",
        ),
        (
            "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC is an excluded semantic-mismatch baseline with SEMANTIC_MISMATCH_NOT_VERIFIED and NOT_SCOREABLE status. Its release output, archive-closed build, or terminal-only agreement cannot authorize participation.",
            "EA_IRIS_SOURCE_FRAME_ADAPTER_D96978AC is VERIFIED without build or conformance.",
        ),
    ],
)
def test_l7_authority_rejects_iris_identity_conflation(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    assert old in seed_text
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(seed_text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(L7AuthorityError, match="Kaya participation or EA IRIS exclusion"):
        validate_l7_authority(hostile_seed, repository_root=tmp_path)


def test_l7_authority_rejects_additive_iris_identity_contradiction(
    tmp_path: Path,
) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    insertion = (
        "  - EA_IRIS_RELEASE_1_1_0_FD3E09E_UBUNTU is also a direct detector, "
        "score input, ranking entry, and winner candidate.\n"
    )
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(
        seed_text.replace("ontology_schema:\n", insertion + "ontology_schema:\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(L7AuthorityError, match="Kaya participation or EA IRIS exclusion"):
        validate_l7_authority(hostile_seed, repository_root=tmp_path)


def test_l7_authority_allows_narrative_map_status_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_path = (ROOT / "docs" / "MASTER-MAP.md").resolve()
    original_read_bytes = Path.read_bytes
    map_reads = 0

    def hostile_read_bytes(path: Path) -> bytes:
        nonlocal map_reads
        data = original_read_bytes(path)
        if path.resolve() == map_path:
            map_reads += 1
            if map_reads == 1:
                return data.replace(b"L7 | in progress", b"L7 | completed", 1)
        return data

    monkeypatch.setattr(Path, "read_bytes", hostile_read_bytes)
    binding = validate_l7_authority(SEED, repository_root=ROOT)
    assert binding.leaf_id == "L7"
    assert map_reads == 1


def test_l7_seed_binds_map_identity_without_narrative_byte_pin() -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    observed = tuple(
        line.strip()
        for line in seed_text.splitlines()
        if line.strip().startswith("- ")
        and ("iris" in line.lower() or "kaya" in line.lower())
    )
    assert observed == EXPECTED_SEED_PARTICIPANT_LINES
    assert "master_map: {path: docs/MASTER-MAP.md, version: 2026-08-11_v160}" in seed_text
    assert "master_map: {path: docs/MASTER-MAP.md, version: 2026-08-11_v160, commit:" not in seed_text
    assert "detector_cell_N" in seed_text
    assert "no identity mapping is included" in seed_text
    assert "duplicate raw artifacts as denominator expansion" in seed_text
    assert "not non-empty strings" in seed_text
    assert "registry_snapshot_sha256 is not lowercase hex64" in seed_text
    assert "schema is not flashpatch-l7-independent-gold-projection-v1" in seed_text
    assert "receipt or trust policy path and SHA-256" in seed_text


def test_l7_authority_uses_one_pinned_byte_read_for_plan_and_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        (ROOT / "docs" / "MASTER-MAP.md").resolve(),
        (ROOT / "docs" / "plans" / "2026-08-02-l7-direct-detector-league.md").resolve(),
    }
    original_read_text = Path.read_text
    protected_text_reads: list[Path] = []

    def hostile_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            protected_text_reads.append(path.resolve())
            return original_read_text(path, *args, **kwargs).replace("NOT_SCOREABLE", "SCOREABLE")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", hostile_read_text)
    binding = validate_l7_authority(SEED, repository_root=ROOT)
    assert binding.leaf_id == "L7"
    assert protected_text_reads == []
