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
    "23faafebe24a6ac5ed972ab8719322cb5afc01b40f9a2ae1bb8ecb39c2929e05"
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

    assert 'bubblewrap_version="$(apt-cache policy bubblewrap | awk' in workflow
    assert 'ffmpeg_version="$(apt-cache policy ffmpeg | awk' in workflow
    assert '"bubblewrap=$bubblewrap_version"' in workflow
    assert '"ffmpeg=$ffmpeg_version"' in workflow
    assert workflow.index("bubblewrap_version=") < workflow.index(full_regression)
