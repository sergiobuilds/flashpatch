from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPECTED_PROJECT_ROOT = "."
EXPECTED_MASTER_MAP_VERSION = "2026-08-25_v163"
EXPECTED_PLAN_PATH = "docs/plans/2026-08-02-l6-sparta-controlled-chain.md"
EXPECTED_PLAN_COMMIT = "bf36a6250bb42258da299c31eb0c8d0bf562eb5f"
EXPECTED_SEED_PATH = "specs/L6-sparta-controlled-chain-execution.seed.yaml"
EXPECTED_WRITEBACK = (
    "own_leaf_status",
    "status_section",
    "history",
    "evidence_links",
)
EXPECTED_OUTPUTS = (
    "src/flashpatch/public_godot.py",
    "src/flashpatch/l6_authority.py",
    "src/flashpatch/l6_run.py",
    "src/flashpatch/safety_ci.py",
    "src/flashpatch/l6_verify.py",
    "tests/test_public_godot.py",
    "tests/test_safety_ci.py",
    "tests/test_l6.py",
    "tests/fixtures/l6_sparta",
    "artifacts/l6",
)
EXPECTED_GOAL = (
    "Implement only MASTER-MAP leaf L6 using "
    "docs/plans/2026-08-02-l6-sparta-controlled-chain.md. Prove the controlled "
    "Sparta renderer to runtime contributor to one source parameter patch to "
    "same-trace preservation chain without altering L7-L9, the Charter, or any "
    "external claim boundary."
)

SPARTA_REPOSITORY = "https://github.com/Lacaedemon/sparta"
SPARTA_REVISION = "06be859d9237192dca391a35bf3a267ff939ceae"
SPARTA_LICENSE_SHA256 = (
    "8ecf69be8e2b9f65ad65220847f9fea7387961d31ecbe042982764a2996144a5"
)
SPARTA_LOCAL_CHECKOUT = "/tmp/flashpatch-sparta"
SPARTA_ENTRY_SCENE = "res://tools/demo/DemoInputRecorder.tscn"
SPARTA_REQUIRED_INPUTS = (
    "project.godot",
    "tools/demo/DemoInputRecorder.tscn",
    "tools/demo/DemoInputRecorder.gd",
    "scripts/RoutShockwave.gd",
    "scenes/Battle.tscn",
)
GODOT_BINARY = "/tmp/flashpatch-godot-4.7/Godot_v4.7-stable_linux.x86_64"
GODOT_BINARY_SHA256 = (
    "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75"
)
GODOT_VERSION = "4.7.stable.official.5b4e0cb0f"
GODOT_XVFB_SCREEN = "1280x720x24"
GODOT_RENDERING_DRIVER = "opengl3"
GODOT_FIXED_FPS = 60
GODOT_CAPTURE_TICKS = 161
GODOT_TIMEOUT_SECONDS = 300

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


class L6AuthorityError(ValueError):
    """Raised when the L6 execution Seed exceeds its authority."""


@dataclass(frozen=True)
class L6AuthorityBinding:
    leaf_id: str
    master_map_version: str
    plan_path: str
    plan_commit: str
    writeback: tuple[str, ...]


@dataclass(frozen=True)
class L6PreflightPins:
    repository: str
    revision: str
    license_sha256: str
    local_checkout: str
    entry_scene: str
    required_inputs: tuple[str, ...]
    godot_binary: str
    godot_binary_sha256: str
    godot_version: str
    xvfb_screen: str
    rendering_driver: str
    fixed_fps: int
    capture_ticks: int
    timeout_seconds: int


L6_PREFLIGHT_PINS = L6PreflightPins(
    repository=SPARTA_REPOSITORY,
    revision=SPARTA_REVISION,
    license_sha256=SPARTA_LICENSE_SHA256,
    local_checkout=SPARTA_LOCAL_CHECKOUT,
    entry_scene=SPARTA_ENTRY_SCENE,
    required_inputs=SPARTA_REQUIRED_INPUTS,
    godot_binary=GODOT_BINARY,
    godot_binary_sha256=GODOT_BINARY_SHA256,
    godot_version=GODOT_VERSION,
    xvfb_screen=GODOT_XVFB_SCREEN,
    rendering_driver=GODOT_RENDERING_DRIVER,
    fixed_fps=GODOT_FIXED_FPS,
    capture_ticks=GODOT_CAPTURE_TICKS,
    timeout_seconds=GODOT_TIMEOUT_SECONDS,
)


def _fail(reason: str) -> None:
    raise L6AuthorityError(reason)


def _master_map_version_key(version: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})_v(\d+)", version)
    if match is None:
        _fail("MASTER-MAP version is malformed")
    return tuple(int(part) for part in match.groups())


def _single_scalar(text: str, key: str) -> str:
    matches = re.findall(rf"(?m)^{re.escape(key)}:[ \t]*(.*)$", text)
    if len(matches) != 1 or not matches[0].strip():
        _fail(f"{key} must occur exactly once as a non-empty top-level scalar")
    return matches[0].strip()


def _inline_mapping(text: str, key: str) -> dict[str, str]:
    matches = re.findall(rf"(?m)^[ \t]+{re.escape(key)}:[ \t]*(.*)$", text)
    if len(matches) != 1 or not matches[0].strip():
        _fail(f"{key} must occur exactly once as a non-empty nested mapping")
    raw = matches[0].strip()
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


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        _fail("MASTER-MAP frontmatter is missing")
    parts = text.split("---\n", 2)
    if len(parts) != 3:
        _fail("MASTER-MAP frontmatter is malformed")
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


def _assert_plan_commit(
    root: Path,
    relative_path: str,
    commit: str,
    local_bytes: bytes,
) -> None:
    committed_plan = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root,
        capture_output=True,
    )
    if committed_plan.returncode != 0:
        _fail(f"binding plan commit does not contain {relative_path}: {commit}")
    if committed_plan.stdout != local_bytes:
        _fail(f"binding plan differs from its pinned commit: {commit}")
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if is_ancestor.returncode != 0:
        _fail(f"binding plan commit is not an ancestor of HEAD: {commit}")


def validate_l6_authority(
    seed_path: Path, *, repository_root: Path | None = None
) -> L6AuthorityBinding:
    """Validate the L6 Seed against its binding plan and MASTER-MAP leaf."""

    seed_path = seed_path.resolve()
    root = (repository_root or seed_path.parents[1]).resolve()
    try:
        relative_seed = seed_path.relative_to(root).as_posix()
    except ValueError:
        _fail("Seed must be inside the repository")
    if relative_seed != EXPECTED_SEED_PATH:
        _fail(f"unexpected L6 Seed path: {relative_seed}")

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
        "authority_leaf_id": "L6",
    }
    for key, expected in expected_scalars.items():
        observed = _single_scalar(seed_text, key)
        if observed != expected:
            _fail(f"{key} must be {expected}, observed {observed}")

    writeback = _inline_sequence(seed_text, "authority_writeback")
    if writeback != EXPECTED_WRITEBACK:
        _fail(
            "L6 writeback may target only its status, status section, history, "
            "and evidence links"
        )
    if _folded_scalar(seed_text, "goal") != EXPECTED_GOAL:
        _fail("L6 goal changed or crossed a sibling, Charter, or external-claim boundary")

    outputs = tuple(
        line.removeprefix("  - ").strip()
        for line in _block_lines(seed_text, "outputs")
        if line.startswith("  - ")
    )
    if outputs != EXPECTED_OUTPUTS:
        _fail("L6 output scope changed")

    plan_binding = _inline_mapping(seed_text, "binding_plan")
    if plan_binding != {"path": EXPECTED_PLAN_PATH, "commit": EXPECTED_PLAN_COMMIT}:
        _fail("L6 binding plan path or commit changed")
    map_binding = _inline_mapping(seed_text, "master_map")
    if map_binding != {
        "path": "docs/MASTER-MAP.md",
        "version": EXPECTED_MASTER_MAP_VERSION,
    }:
        _fail("L6 MASTER-MAP binding changed")

    plan_path = root / EXPECTED_PLAN_PATH
    map_path = root / "docs" / "MASTER-MAP.md"
    if not plan_path.is_file() or not map_path.is_file():
        _fail("bound plan or MASTER-MAP is missing")

    plan_bytes = plan_path.read_bytes()
    map_bytes = map_path.read_bytes()
    _assert_plan_commit(root, EXPECTED_PLAN_PATH, EXPECTED_PLAN_COMMIT, plan_bytes)
    try:
        plan_text = plan_bytes.decode("utf-8")
        map_text = map_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("bound plan or MASTER-MAP is not strict UTF-8")
    expected_plan_authority = {
        "authority_role": "execution-seed",
        "authority_project_root": EXPECTED_PROJECT_ROOT,
        "authority_master_map_version": EXPECTED_MASTER_MAP_VERSION,
        "authority_leaf_id": "L6",
        "authority_writeback": ",".join(EXPECTED_WRITEBACK),
    }
    if _plan_authority(plan_text) != expected_plan_authority:
        _fail("binding plan authority table does not exactly match the L6 Seed")

    required_boundaries = (
        "L7 비교, L8 블라인드 리그, L9 외부 우위 주장은 이 계획의 산출물이 아닙니다.",
        "controlled mutation은 upstream defect나 자연 발생 결함으로 표현하지 않습니다.",
    )
    if any(boundary not in plan_text for boundary in required_boundaries):
        _fail("binding plan no longer preserves the sibling or external-claim boundary")

    frontmatter = _frontmatter(map_text)
    if frontmatter.get("authority_model") != "v2":
        _fail("MASTER-MAP does not opt in to authority model v2")
    current_map_version = frontmatter.get("version", "")
    if _master_map_version_key(current_map_version) < _master_map_version_key(
        EXPECTED_MASTER_MAP_VERSION
    ):
        _fail("MASTER-MAP version predates the L6 binding")

    work_tree_match = re.search(
        r"(?ms)^## 4 Work Tree\n(?P<body>.*?)(?=^## 5 )", map_text
    )
    if work_tree_match is None:
        _fail("MASTER-MAP Work Tree section is missing")
    work_tree = work_tree_match.group("body")
    leaf_ids = re.findall(r"\| (L\d+) \|", work_tree)
    if leaf_ids.count("L6") != 1:
        _fail("MASTER-MAP must contain exactly one L6 leaf")
    if not {"L7", "L8", "L9"}.issubset(leaf_ids):
        _fail("MASTER-MAP sibling boundary is incomplete")
    return L6AuthorityBinding(
        leaf_id="L6",
        master_map_version=EXPECTED_MASTER_MAP_VERSION,
        plan_path=EXPECTED_PLAN_PATH,
        plan_commit=EXPECTED_PLAN_COMMIT,
        writeback=writeback,
    )
