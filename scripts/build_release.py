from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import UTC
from pathlib import Path

ROOT = Path(__file__).parents[1]
VERSION = "0.1.0"
SOURCE_PATHS = ("LICENSE", "NOTICE", "README.md", "pyproject.toml", "src")
DEPENDENCIES = (
    {
        "name": "av",
        "requirement": ">=18",
        "license": "BSD-3-Clause",
        "repository": "https://github.com/PyAV-Org/PyAV",
        "license_note": "Binary distributions may carry separately licensed FFmpeg components.",
    },
    {
        "name": "numpy",
        "requirement": ">=1.26",
        "license": "BSD-3-Clause",
        "repository": "https://github.com/numpy/numpy",
        "license_note": "NumPy distributions include additional vendored-component licenses.",
    },
    {
        "name": "opencv-python-headless",
        "requirement": ">=4.10",
        "license": "MIT",
        "repository": "https://github.com/opencv/opencv-python",
        "license_note": "The wheel also carries OpenCV and third-party license notices.",
    },
)
PACKAGE_DATA = {"src/flashpatch/_unity_packages_lock_2022_3_8f1.json"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _git_text(*args: str) -> str:
    return _git_bytes(*args).decode("utf-8").strip()


def _resolve_source_sha(source_commit: str) -> str:
    source_sha = _git_text("rev-parse", "--verify", f"{source_commit}^{{commit}}")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError(f"source commit did not resolve to a full Git SHA: {source_commit}")
    return source_sha


def _zip_info(name: str, epoch: int) -> zipfile.ZipInfo:
    from datetime import datetime

    timestamp = max(epoch, 315532800)
    date_time = datetime.fromtimestamp(timestamp, UTC).timetuple()[:6]
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _build_wheel(output: Path, source_sha: str, epoch: int) -> Path:
    wheel = output / f"flashpatch-{VERSION}-py3-none-any.whl"
    dist_info = f"flashpatch-{VERSION}.dist-info"
    files: dict[str, bytes] = {}
    names = _git_text(
        "ls-tree", "-r", "--name-only", source_sha, "--", "src/flashpatch"
    ).splitlines()
    for relative in names:
        if relative.endswith(".py") or relative in PACKAGE_DATA:
            files[relative.removeprefix("src/")] = _git_bytes(
                "show", f"{source_sha}:{relative}"
            )
    files[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.4\n"
        "Name: flashpatch\n"
        f"Version: {VERSION}\n"
        "Summary: Fail-closed visual QA for game-development evidence\n"
        "License-Expression: Apache-2.0\n"
        "Requires-Python: >=3.11\n"
        + "".join(
            f"Requires-Dist: {item['name']}{item['requirement']}\n"
            for item in DEPENDENCIES
        )
        + "\n"
    ).encode()
    files[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: flashpatch-build-release\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n\n"
    )
    files[f"{dist_info}/entry_points.txt"] = (
        b"[console_scripts]\nflashpatch = flashpatch.cli:main\n"
    )
    files[f"{dist_info}/licenses/LICENSE"] = _git_bytes(
        "show", f"{source_sha}:LICENSE"
    )
    files[f"{dist_info}/licenses/NOTICE"] = _git_bytes(
        "show", f"{source_sha}:NOTICE"
    )
    rows: list[list[str]] = []
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in sorted(files.items()):
            archive.writestr(_zip_info(name, epoch), data)
            rows.append([name, _record_hash(data), str(len(data))])
        record_name = f"{dist_info}/RECORD"
        rows.append([record_name, "", ""])
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        archive.writestr(_zip_info(record_name, epoch), stream.getvalue().encode())
    return wheel


def _build_source_archive(output: Path, source_sha: str, epoch: int) -> Path:
    destination = output / f"flashpatch-{VERSION}.tar.gz"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        tar_path = Path(temporary) / f"flashpatch-{VERSION}.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--prefix=flashpatch-{VERSION}/",
                "-o",
                str(tar_path),
                source_sha,
                "--",
                *SOURCE_PATHS,
            ],
            cwd=ROOT,
            check=True,
        )
        with (
            tar_path.open("rb") as source,
            destination.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
        ):
            while block := source.read(1024 * 1024):
                compressed.write(block)
    return destination


def _write_source_manifest(output: Path, source_sha: str) -> Path:
    names = _git_text(
        "ls-tree", "-r", "--name-only", source_sha, "--", *SOURCE_PATHS
    ).splitlines()
    files = []
    for name in names:
        data = _git_bytes("show", f"{source_sha}:{name}")
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload = {
        "schema": "flashpatch-source-manifest-v1",
        "source_git_sha": source_sha,
        "scope": list(SOURCE_PATHS),
        "files": files,
    }
    path = output / "source-manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_sbom(output: Path, source_sha: str) -> Path:
    path = output / f"flashpatch-{VERSION}.cdx.json"
    root_ref = f"pkg:pypi/flashpatch@{VERSION}"
    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{source_sha[:8]}-{source_sha[8:12]}-4{source_sha[13:16]}-8{source_sha[17:20]}-{source_sha[20:32]}",
        "version": 1,
        "metadata": {
            "component": {
                "bom-ref": root_ref,
                "type": "application",
                "name": "flashpatch",
                "version": VERSION,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "properties": [
                    {"name": "flashpatch:source-git-sha", "value": source_sha}
                ],
            }
        },
        "components": [
            {
                "bom-ref": f"pkg:pypi/{item['name']}?requirement={item['requirement']}",
                "type": "library",
                "name": item["name"],
                "licenses": [{"license": {"id": item["license"]}}],
                "externalReferences": [
                    {"type": "vcs", "url": item["repository"]}
                ],
                "properties": [
                    {
                        "name": "flashpatch:version-requirement",
                        "value": item["requirement"],
                    },
                    {
                        "name": "flashpatch:license-note",
                        "value": item["license_note"],
                    },
                ],
            }
            for item in DEPENDENCIES
        ],
        "dependencies": [
            {
                "ref": root_ref,
                "dependsOn": [
                    f"pkg:pypi/{item['name']}?requirement={item['requirement']}"
                    for item in DEPENDENCIES
                ],
            }
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def build_release(output: Path, source_commit: str = "HEAD") -> Path:
    output.mkdir(parents=True, exist_ok=True)
    source_sha = _resolve_source_sha(source_commit)
    epoch = int(
        os.environ.get("SOURCE_DATE_EPOCH")
        or _git_text("show", "-s", "--format=%ct", source_sha)
    )
    artifacts = [
        _build_source_archive(output, source_sha, epoch),
        _build_wheel(output, source_sha, epoch),
        _write_sbom(output, source_sha),
        _write_source_manifest(output, source_sha),
    ]
    manifest = {
        "schema": "flashpatch-release-manifest-v2",
        "version": VERSION,
        "source_git_sha": source_sha,
        "source_date_epoch": epoch,
        "artifacts": [
            {
                "file": artifact.name,
                "sha256": _sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
            for artifact in sorted(artifacts)
        ],
    }
    path = output / "release-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        default="HEAD",
        help="exact Git commit used for every source-bound release artifact",
    )
    args = parser.parse_args()
    print(build_release(args.output, args.source_commit))


if __name__ == "__main__":
    main()
