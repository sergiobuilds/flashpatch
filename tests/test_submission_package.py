from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_submission_package_fixed_readback() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_submission.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "benchmark claims bound" in result.stdout
    assert "180.000s demo" in result.stdout
