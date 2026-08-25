from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SETUP_UV = (
    "uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
)
BUBBLEWRAP_PIN = "bubblewrap=0.9.0-1ubuntu0.1"
UV_DEPENDENT_COMMANDS = (
    'GODOT_BINARY="$PWD/.tools/godot" python -m pytest -q -rs',
)
EXPECTED_SHARED_WORKFLOW_SHA256 = (
    "c5dec25e696d43213fbf213644d264aab48f9e83307dd6fa0351c1f1cb56af5c"
)


def test_ci_installs_exact_uv_before_uv_dependent_commands() -> None:
    workflow_bytes = WORKFLOW.read_bytes()
    workflow = workflow_bytes.decode("utf-8")

    assert hashlib.sha256(workflow_bytes).hexdigest() == EXPECTED_SHARED_WORKFLOW_SHA256
    assert workflow.count(SETUP_UV) == 1
    assert workflow.count('version: "0.12.0"') == 1
    setup_offset = workflow.index(SETUP_UV)
    for command in UV_DEPENDENT_COMMANDS:
        assert workflow.count(command) == 1
        assert setup_offset < workflow.index(command)


def test_ci_installs_exact_bubblewrap_before_full_regression() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    full_regression = UV_DEPENDENT_COMMANDS[0]

    assert workflow.count(BUBBLEWRAP_PIN) == 1
    assert workflow.index(BUBBLEWRAP_PIN) < workflow.index(full_regression)
