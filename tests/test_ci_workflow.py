from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SETUP_UV = (
    "uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
)
UV_DEPENDENT_COMMANDS = (
    'GODOT_BINARY="$PWD/.tools/godot" python -m pytest -q -rs',
)
EXPECTED_SHARED_WORKFLOW_SHA256 = (
    "c5207689b9eaedd27e53c02a84753964ea449330f9dee749bbae3b44158e8b99"
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
