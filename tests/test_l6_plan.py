from __future__ import annotations

import re
from pathlib import Path

import pytest

from flashpatch.l6_authority import (
    EXPECTED_MASTER_MAP_VERSION,
    EXPECTED_PLAN_COMMIT,
    EXPECTED_PLAN_PATH,
    EXPECTED_WRITEBACK,
    L6_PREFLIGHT_PINS,
    L6AuthorityError,
    validate_l6_authority,
)


ROOT = Path(__file__).parents[1]
SEED = ROOT / "specs" / "L6-sparta-controlled-chain-execution.seed.yaml"


def test_l6_seed_is_bound_exclusively_to_its_master_map_leaf() -> None:
    binding = validate_l6_authority(SEED, repository_root=ROOT)

    assert binding.leaf_id == "L6"
    assert binding.master_map_version == EXPECTED_MASTER_MAP_VERSION
    assert binding.plan_path == EXPECTED_PLAN_PATH
    assert binding.plan_commit == EXPECTED_PLAN_COMMIT
    assert binding.writeback == EXPECTED_WRITEBACK


def test_l6_preflight_pins_match_the_bound_plan() -> None:
    assert L6_PREFLIGHT_PINS.repository == "https://github.com/Lacaedemon/sparta"
    assert L6_PREFLIGHT_PINS.revision == "06be859d9237192dca391a35bf3a267ff939ceae"
    assert L6_PREFLIGHT_PINS.license_sha256 == (
        "8ecf69be8e2b9f65ad65220847f9fea7387961d31ecbe042982764a2996144a5"
    )
    assert L6_PREFLIGHT_PINS.local_checkout == "/tmp/flashpatch-sparta"
    assert L6_PREFLIGHT_PINS.entry_scene == (
        "res://tools/demo/DemoInputRecorder.tscn"
    )
    assert L6_PREFLIGHT_PINS.required_inputs == (
        "project.godot",
        "tools/demo/DemoInputRecorder.tscn",
        "tools/demo/DemoInputRecorder.gd",
        "scripts/RoutShockwave.gd",
        "scenes/Battle.tscn",
    )
    assert L6_PREFLIGHT_PINS.godot_binary == (
        "/tmp/flashpatch-godot-4.7/Godot_v4.7-stable_linux.x86_64"
    )
    assert L6_PREFLIGHT_PINS.godot_binary_sha256 == (
        "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75"
    )
    assert L6_PREFLIGHT_PINS.godot_version == "4.7.stable.official.5b4e0cb0f"
    assert L6_PREFLIGHT_PINS.xvfb_screen == "1280x720x24"
    assert L6_PREFLIGHT_PINS.rendering_driver == "opengl3"
    assert L6_PREFLIGHT_PINS.fixed_fps == 60
    assert L6_PREFLIGHT_PINS.capture_ticks == 161
    assert L6_PREFLIGHT_PINS.timeout_seconds == 300


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("authority_leaf_id: L6", "authority_leaf_id: L7"),
        ("authority_leaf_id: L6", "authority_leaf_id: L8"),
        ("authority_leaf_id: L6", "authority_leaf_id: L9"),
        (
            "authority_writeback: [own_leaf_status, status_section, history, evidence_links]",
            "authority_writeback: [own_leaf_status, status_section, history, "
            "chronicle_proposal, evidence_links]",
        ),
        (
            "without altering L7-L9, the Charter, or any",
            "while altering L7-L9, the Charter, and every",
        ),
        (
            "outputs:\n  - src/flashpatch/public_godot.py",
            "outputs:\n  - src/flashpatch/external_league.py\n"
            "  - src/flashpatch/public_godot.py",
        ),
    ],
)
def test_l6_authority_rejects_sibling_charter_and_claim_scope(
    tmp_path: Path, old: str, new: str
) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    assert old in seed_text
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(seed_text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(L6AuthorityError):
        validate_l6_authority(hostile_seed, repository_root=tmp_path)


def test_l6_authority_rejects_map_owned_scope(tmp_path: Path) -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    hostile_seed = tmp_path / "specs" / SEED.name
    hostile_seed.parent.mkdir()
    hostile_seed.write_text(
        seed_text + "\nproject_charter:\n  goal: replace the product boundary\n",
        encoding="utf-8",
    )

    with pytest.raises(L6AuthorityError, match="map-owned top-level keys"):
        validate_l6_authority(hostile_seed, repository_root=tmp_path)


def test_l6_authority_allows_narrative_map_status_mutation(
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
                return data.replace(b"L6 | completed", b"L6 | in progress", 1)
        return data

    monkeypatch.setattr(Path, "read_bytes", hostile_read_bytes)
    binding = validate_l6_authority(SEED, repository_root=ROOT)
    assert binding.leaf_id == "L6"
    assert map_reads == 1


def test_l6_authority_rejects_map_older_than_its_execution_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_path = (ROOT / "docs" / "MASTER-MAP.md").resolve()
    original_read_bytes = Path.read_bytes

    def hostile_read_bytes(path: Path) -> bytes:
        data = original_read_bytes(path)
        if path.resolve() == map_path:
            hostile = re.sub(
                rb"(?m)^version: \S+$",
                b"version: 2026-08-09_v143",
                data,
                count=1,
            )
            assert hostile != data
            return hostile
        return data

    monkeypatch.setattr(Path, "read_bytes", hostile_read_bytes)
    with pytest.raises(L6AuthorityError, match="predates the L6 binding"):
        validate_l6_authority(SEED, repository_root=ROOT)


def test_l6_authority_rejects_l6_row_retarget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_path = (ROOT / "docs" / "MASTER-MAP.md").resolve()
    original_read_bytes = Path.read_bytes

    def hostile_read_bytes(path: Path) -> bytes:
        data = original_read_bytes(path)
        if path.resolve() == map_path:
            return data.replace(
                b"| 4.5.1 public Godot controlled chain | L6 | completed |",
                b"| 4.5.1 public Godot controlled chain | L7 | completed |",
                1,
            )
        return data

    monkeypatch.setattr(Path, "read_bytes", hostile_read_bytes)
    with pytest.raises(L6AuthorityError, match="exactly one L6 leaf"):
        validate_l6_authority(SEED, repository_root=ROOT)


def test_l6_seed_binds_map_identity_without_narrative_byte_pin() -> None:
    seed_text = SEED.read_text(encoding="utf-8")
    assert "master_map: {path: docs/MASTER-MAP.md, version: 2026-08-10_v144}" in seed_text
    assert "master_map: {path: docs/MASTER-MAP.md, version: 2026-08-10_v144, commit:" not in seed_text


def test_l6_authority_uses_one_pinned_byte_read_for_plan_and_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {
        (ROOT / "docs" / "MASTER-MAP.md").resolve(),
        (ROOT / "docs" / "plans" / "2026-08-02-l6-sparta-controlled-chain.md").resolve(),
    }
    original_read_text = Path.read_text
    protected_text_reads: list[Path] = []

    def hostile_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            protected_text_reads.append(path.resolve())
            return original_read_text(path, *args, **kwargs).replace("completed", "in progress")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", hostile_read_text)
    binding = validate_l6_authority(SEED, repository_root=ROOT)
    assert binding.leaf_id == "L6"
    assert protected_text_reads == []
