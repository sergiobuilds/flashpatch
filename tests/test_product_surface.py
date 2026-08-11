from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from flashpatch import repair_video, scan_video, verify_video
from flashpatch.media import VideoMetadata, write_video
from flashpatch.web import create_server

ROOT = Path(__file__).parents[1]


def _hazard_video(path: Path) -> VideoMetadata:
    frames = np.zeros((12, 8, 10, 3), dtype=np.uint8)
    frames[1::2] = 255
    timestamps = np.linspace(0.0, 1.0, len(frames))
    metadata = VideoMetadata(color_range=1, colorspace=1, color_primaries=1, color_trc=1)
    write_video(path, frames, timestamps, metadata=metadata)
    return metadata


def test_public_file_api_closes_scan_repair_verify_loop(tmp_path: Path) -> None:
    source = tmp_path / "hazard.mp4"
    repaired = tmp_path / "repaired.mp4"
    receipt = tmp_path / "receipt.json"
    metadata = _hazard_video(source)

    scan = scan_video(source)
    repair = repair_video(source, repaired, receipt=receipt)
    verification = verify_video(repaired)

    assert scan["hazardous"] is True
    assert repair["status"] == "VERIFIED"
    assert verification["passed"] is True
    assert repair["color_metadata_preserved"] is True
    assert json.loads(receipt.read_text(encoding="utf-8")) == repair
    assert metadata.color_primaries == 1


def test_installed_cli_and_ci_contract_are_executable() -> None:
    executable = Path(sys.executable).with_name("flashpatch")
    assert executable.is_file()
    completed = subprocess.run(
        [str(executable), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "compile" in completed.stdout
    assert "scan" in completed.stdout
    assert "repair" in completed.stdout
    assert "web" in completed.stdout


def test_safety_demo_writes_all_terminal_receipts(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flashpatch.cli", "safety-demo", "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert json.loads(completed.stdout)["results"] == {
        "pass": "PASS",
        "safe": "SAFE",
        "inconclusive": "INCONCLUSIVE",
    }
    assert json.loads((tmp_path / "pass-receipt.json").read_text())["verdict"] == "PASS"
    assert json.loads((tmp_path / "safe-receipt.json").read_text())["verdict"] == "SAFE"
    assert json.loads((tmp_path / "inconclusive-receipt.json").read_text())["verdict"] == "INCONCLUSIVE"


def test_browser_judge_flow_uploads_repairs_and_downloads(tmp_path: Path) -> None:
    source = tmp_path / "hazard.mp4"
    _hazard_video(source)
    server = create_server("127.0.0.1", 0, workspace=tmp_path / "web")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            page = response.read().decode()
        request = Request(
            base + "/api/repair",
            data=source.read_bytes(),
            headers={"Content-Type": "video/mp4", "X-FlashPatch-Filename": "hazard.mp4"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        with urlopen(base + payload["download_url"], timeout=5) as response:
            repaired_bytes = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "Temporal visual-risk scan" in page
    assert "Upload video" in page
    assert payload["scan"]["hazardous"] is True
    assert payload["repair"]["status"] == "VERIFIED"
    assert payload["verification"]["passed"] is True
    assert repaired_bytes[:8].endswith(b"ftyp")
