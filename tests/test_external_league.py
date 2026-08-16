from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flashpatch.external_league as external_league
import flashpatch.l7_verify as l7_verify
from flashpatch.external_league import (
    COMPARATOR_CENSUS_SCHEMA,
    CONFORMANCE_ORACLE_POPULATION,
    ComparatorSpec,
    DIRECT_DETECTOR_POPULATION,
    EXCLUDED_SEMANTIC_MISMATCH_POPULATION,
    EA_IRIS_LEGACY_JSON_ID,
    EA_IRIS_RELEASE_ORACLE_ID,
    EA_IRIS_SOURCE_ADAPTER_ID,
    KAYA_DIRECT_PARTICIPANT_ID,
    KAYA_PARTICIPANT_CONFORMANCE_SCHEMA,
    KAYA_PROTOTYPE_ID,
    ExternalLeagueError,
    FairRuntimeProtocol,
    IrisReleaseSpec,
    aggregate_detection_cases,
    capture_fair_runtime_protocol,
    execute_comparator,
    execute_flashpatch_detector,
    execute_iris_release,
    execute_repeated_comparator,
    execute_repeated_flashpatch_detector,
    execute_repeated_iris_release,
    freeze_fair_runtime_schedule,
    freeze_fair_runtime_protocol,
    materialize_cfr_ffv1,
    materialize_native_main_comparator_input,
    pack_renderer_png_sequence,
    parse_iris_json,
    parse_iris_release_csv,
    parse_tooflashy_json,
    verify_tooflashy_oldfilm_repeats,
    validate_comparator_census,
    verify_decoder_timeline_parity,
    verify_fair_runtime_receipts,
    write_fair_runtime_schedule,
    write_comparator_census_receipt,
    write_scheduled_runtime_repeat_receipt,
)
from flashpatch.l7_external_host import (
    VERIFICATION_SCHEMA_V1 as EXTERNAL_HOST_VERIFICATION_SCHEMA_V1,
    VERIFICATION_SCHEMA_V2 as EXTERNAL_HOST_VERIFICATION_SCHEMA_V2,
)

_REAL_VERIFY_TOOFLASHY_DISTRIBUTION = external_league._verify_tooflashy_distribution


def _require_bubblewrap_namespaces() -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable on this host")
    probe = subprocess.run(
        [bwrap, "--bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("bubblewrap user namespaces are unavailable on this host")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _stable_flashpatch_checkout_for_census_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=checkout, text=True).strip()
    monkeypatch.setattr(
        external_league,
        "_flashpatch_checkout_provenance",
        lambda: {
            "revision": revision,
            "tree": tree,
            "origin": "https://github.com/sergiobuilds/flashpatch",
            "clean": True,
            "pushed": True,
        },
    )
    monkeypatch.setattr(
        external_league,
        "_verify_upstream_checkout",
        lambda entry: {
            "path": entry["source_checkout"],
            "revision": entry["revision"],
            "tree": "b" * 40,
            "origin": entry["repository_url"],
            "clean": True,
        },
    )
    monkeypatch.setattr(external_league, "_verify_iris_release_artifacts", lambda entry: None)
    monkeypatch.setattr(external_league, "_verify_tooflashy_distribution", lambda entry, command, environment: None)
    monkeypatch.setattr(
        external_league,
        "verify_kaya_participant_conformance_receipt",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )


def _frames(path: Path, *, cfr: bool = True) -> Path:
    frames = np.arange(8 * 4 * 6 * 3, dtype=np.uint8).reshape(8, 4, 6, 3)
    timestamps = np.arange(8, dtype=np.float64) / 60.0
    if not cfr:
        timestamps[3] += 0.001
    np.savez_compressed(path, frames=frames, timestamps=timestamps)
    return path


def _fair_runtime_protocol(
    lock_path: Path,
    *,
    timeout_seconds: int = 120,
) -> FairRuntimeProtocol:
    available = sorted(os.sched_getaffinity(0))
    return capture_fair_runtime_protocol(
        concurrency_lock_path=lock_path,
        timeout_seconds=timeout_seconds,
        cpu_affinity=available[: min(2, len(available))],
        thread_limit=1,
    )


def _valid_comparator_census(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    flashpatch_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    ffmpeg_binary_name = shutil.which("ffmpeg")
    if ffmpeg_binary_name is None:
        pytest.skip("L7 comparator census requires the pinned FFmpeg build")
    ffmpeg_binary = Path(ffmpeg_binary_name).resolve()
    ffmpeg_configuration = subprocess.run(
        [str(ffmpeg_binary), "-version"],
        capture_output=True,
        check=True,
    ).stdout
    ffmpeg_package = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n", "ffmpeg"],
        capture_output=True,
        check=True,
    ).stdout
    ffmpeg_distribution, ffmpeg_revision, _ = ffmpeg_package.decode().strip().split("\t")
    ffmpeg_license = Path("/usr/share/doc/ffmpeg/copyright")
    if not ffmpeg_license.is_file():
        pytest.skip("L7 comparator census requires FFmpeg package copyright evidence")
    identities = [
        {
            "name": "FlashPatch",
            "repository_url": "https://github.com/sergiobuilds/flashpatch",
            "revision": flashpatch_revision,
            "source_checkout": str(Path(__file__).resolve().parents[1]),
            "license": "Apache-2.0",
            "distribution": "pushed-source-checkout",
            "distribution_revision": flashpatch_revision,
            "distribution_source_revision": flashpatch_revision,
            "release_asset_sha256": None,
            "capability": "detector",
            "lane": "direct-detector",
            "execution_status": "RUNNABLE",
            "unscorable_reason": None,
        },
        {
            "name": EA_IRIS_RELEASE_ORACLE_ID,
            "repository_url": "https://github.com/electronicarts/IRIS",
            "revision": "fd3e09e4e6fce30a5141ad6eca94a4ff61096e05",
            "source_checkout": None,
            "license": "BSD-3-Clause",
            "distribution": "official-ubuntu-example-app-1.1.0",
            "distribution_revision": "1.1.0",
            "distribution_source_revision": "fd3e09e4e6fce30a5141ad6eca94a4ff61096e05",
            "release_asset_sha256": "440eb0cb814a03a4eff7c8c4f499492b669a33cf2ba4f23843b479365eeedaeb",
            "capability": "conformance-oracle",
            "lane": "conformance-oracle",
            "execution_status": "RUNNABLE",
            "unscorable_reason": None,
        },
        {
            "name": EA_IRIS_SOURCE_ADAPTER_ID,
            "repository_url": "https://github.com/electronicarts/IRIS",
            "revision": "d96978ac1107f3463b77f69a9c1b1ec5d45291a0",
            "source_checkout": None,
            "license": "BSD-3-Clause",
            "distribution": "source-revision-only",
            "distribution_revision": "d96978ac1107f3463b77f69a9c1b1ec5d45291a0",
            "distribution_source_revision": "d96978ac1107f3463b77f69a9c1b1ec5d45291a0",
            "release_asset_sha256": None,
            "capability": "excluded-baseline",
            "lane": "excluded-semantic-mismatch",
            "execution_status": "UNSCORABLE",
            "unscorable_reason": "semantic_conformance_mismatch_excluded",
        },
        {
            "name": KAYA_DIRECT_PARTICIPANT_ID,
            "repository_url": "https://github.com/samfatu/pse-detection-correction",
            "revision": "0776ea3e6949a62d5becb8027a2765961b515793",
            "source_checkout": None,
            "license": "BSD-3-Clause",
            "distribution": "pinned-source-and-python-3.10-conformance",
            "distribution_revision": "0776ea3e6949a62d5becb8027a2765961b515793",
            "distribution_source_revision": "0776ea3e6949a62d5becb8027a2765961b515793",
            "release_asset_sha256": None,
            "capability": "detector",
            "lane": "direct-detector",
            "execution_status": "UNSCORABLE",
            "unscorable_reason": "natural_corpus_gold_parity_and_fair_repeats_missing",
        },
        {
            "name": "TooFlashy",
            "repository_url": "https://github.com/hashb/TooFlashy",
            "revision": "8274e1ea09bd6099d384056f0fcb6fbc32cf0e3f",
            "source_checkout": None,
            "license": "Apache-2.0",
            "distribution": "fixed-source-checkout",
            "distribution_revision": "8274e1ea09bd6099d384056f0fcb6fbc32cf0e3f",
            "distribution_source_revision": "8274e1ea09bd6099d384056f0fcb6fbc32cf0e3f",
            "release_asset_sha256": None,
            "capability": "detector",
            "lane": "direct-detector",
            "execution_status": "FIXED",
            "unscorable_reason": None,
        },
        {
            "name": "EPI-LENS",
            "repository_url": "https://github.com/Pi-0r-Tau/EPI-LENS",
            "revision": "a7c5ab95278e9c590324d6cb95b5f90982561f13",
            "source_checkout": None,
            "license": "MIT",
            "distribution": "fixed-browser-extension-source",
            "distribution_revision": "a7c5ab95278e9c590324d6cb95b5f90982561f13",
            "distribution_source_revision": "a7c5ab95278e9c590324d6cb95b5f90982561f13",
            "release_asset_sha256": None,
            "capability": "detector",
            "lane": "reserve-detector",
            "execution_status": "UNSCORABLE",
            "unscorable_reason": "full_same_input_application_runner_missing",
        },
        {
            "name": "FFmpeg vf_photosensitivity",
            "repository_url": "https://github.com/FFmpeg/FFmpeg",
            "revision": "601d9ee881fbd9d9ff44466c561c480ff244eb9f",
            "source_checkout": None,
            "license": "GPL-3.0-or-later",
            "distribution": ffmpeg_distribution,
            "distribution_revision": ffmpeg_revision,
            "distribution_source_revision": ffmpeg_revision,
            "release_asset_sha256": None,
            "capability": "mitigation",
            "lane": "mitigation",
            "execution_status": "FIXED",
            "unscorable_reason": None,
        },
    ]
    command_arguments = {
        "FlashPatch": ["scan", "{input}"],
        EA_IRIS_RELEASE_ORACLE_ID: ["-j", "{input}"],
        EA_IRIS_SOURCE_ADAPTER_ID: [],
        KAYA_DIRECT_PARTICIPANT_ID: [],
        "TooFlashy": ["{input}", "--json"],
        "EPI-LENS": [],
        "FFmpeg vf_photosensitivity": ["-i", "{input}", "-vf", "photosensitivity", "{output}"],
    }
    for identity in identities:
        artifact_dir = tmp_path / str(identity["name"]).lower().replace(" ", "-")
        artifact_dir.mkdir()
        if identity["name"] in {
            EA_IRIS_RELEASE_ORACLE_ID,
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
            "TooFlashy",
            "EPI-LENS",
        }:
            identity["source_checkout"] = str(artifact_dir)
        license_artifact = artifact_dir / "LICENSE"
        binary_artifact = artifact_dir / "binary"
        configuration_artifact = artifact_dir / "configuration.txt"
        environment_artifact = artifact_dir / "environment.json"
        command_artifact = artifact_dir / "command.json"
        if identity["name"] == "FFmpeg vf_photosensitivity":
            shutil.copy2(ffmpeg_license, license_artifact)
            shutil.copy2(ffmpeg_binary, binary_artifact)
            configuration_artifact.write_bytes(ffmpeg_configuration)
        elif identity["name"] == "TooFlashy":
            license_artifact.write_text("Apache License\nVersion 2.0\n")
            binary_artifact.write_text(
                "#!/usr/bin/python3\n"
                "import json, sys\n"
                "print(json.dumps({'path': sys.argv[1], 'passes': True, 'fps': 60.0, "
                "'frame_count': 8, 'event_count': 0, 'failures': []}, sort_keys=True))\n"
            )
            binary_artifact.chmod(0o755)
            configuration_artifact.write_text(json.dumps({"name": identity["name"], "configuration": "fixed"}) + "\n")
        else:
            license_text = {
                "Apache-2.0": "Apache License\nVersion 2.0\n",
                "BSD-3-Clause": "Redistribution and use in source and binary forms are permitted.\n",
                "MIT": "Permission is hereby granted, free of charge.\n",
            }[str(identity["license"])]
            license_artifact.write_text(license_text)
            if identity["name"] not in {
                EA_IRIS_SOURCE_ADAPTER_ID,
                KAYA_DIRECT_PARTICIPANT_ID,
            }:
                binary_artifact.write_text("#!/bin/sh\nprintf 'result\\n'\n")
                binary_artifact.chmod(0o755)
                configuration_artifact.write_text(json.dumps({"name": identity["name"], "configuration": "fixed"}) + "\n")
        if identity["name"] in {
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
        }:
            for field in (
                "binary_artifact",
                "binary_sha256",
                "configuration_artifact",
                "configuration_sha256",
                "environment_artifact",
                "environment_sha256",
                "command_artifact",
                "command_sha256",
            ):
                identity[field] = None
        elif identity["name"] == "TooFlashy":
            environment = {
                "PATH": "/usr/bin:/bin",
                "UV_PROJECT": str(Path(str(identity["source_checkout"])).resolve()),
            }
        else:
            environment = {"name": str(identity["name"]), "environment": "fixed-test-fixture"}
        if identity["name"] not in {
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
        }:
            environment_artifact.write_text(json.dumps(environment, sort_keys=True) + "\n")
            command_template = [str(binary_artifact.resolve()), *command_arguments[str(identity["name"])]]
            command_artifact.write_text(json.dumps(command_template, separators=(",", ":")) + "\n")
        artifacts = [("license_artifact", "license_sha256", license_artifact)]
        if identity["name"] not in {
            EA_IRIS_SOURCE_ADAPTER_ID,
            KAYA_DIRECT_PARTICIPANT_ID,
        }:
            artifacts.extend([
                ("binary_artifact", "binary_sha256", binary_artifact),
                ("configuration_artifact", "configuration_sha256", configuration_artifact),
                ("environment_artifact", "environment_sha256", environment_artifact),
                ("command_artifact", "command_sha256", command_artifact),
            ])
        for artifact_field, hash_field, artifact in artifacts:
            identity[artifact_field] = str(artifact.relative_to(tmp_path))
            identity[hash_field] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        identity["participant_conformance_artifact"] = None
        identity["participant_conformance_sha256"] = None
        if identity["name"] == KAYA_DIRECT_PARTICIPANT_ID:
            participant_receipt = artifact_dir / "participant-conformance.json"
            participant_receipt.write_text(
                json.dumps(
                    {
                        "schema": KAYA_PARTICIPANT_CONFORMANCE_SCHEMA,
                        "identity": KAYA_DIRECT_PARTICIPANT_ID,
                        "prototype_identity": KAYA_PROTOTYPE_ID,
                        "status": "VERIFIED",
                        "scoreable": False,
                        "unscored_population_authorized": True,
                        "external_claim_authorized": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            identity["participant_conformance_artifact"] = str(
                participant_receipt.relative_to(tmp_path)
            )
            identity["participant_conformance_sha256"] = hashlib.sha256(
                participant_receipt.read_bytes()
            ).hexdigest()
        distribution_evidence = {
            "schema": "flashpatch-comparator-distribution-provenance-v1",
            "name": identity["name"],
            "repository_url": identity["repository_url"],
            "revision": identity["revision"],
            "source_checkout": identity["source_checkout"],
            "license": identity["license"],
            "license_sha256": identity["license_sha256"],
            "distribution": identity["distribution"],
            "distribution_revision": identity["distribution_revision"],
            "distribution_source_revision": identity["distribution_source_revision"],
            "release_asset_sha256": identity["release_asset_sha256"],
            "binary_sha256": identity["binary_sha256"],
            "configuration_sha256": identity["configuration_sha256"],
            "environment_sha256": identity["environment_sha256"],
            "command_sha256": identity["command_sha256"],
            "participant_conformance_sha256": identity["participant_conformance_sha256"],
        }
        distribution_artifact = artifact_dir / "distribution.json"
        distribution_artifact.write_text(json.dumps(distribution_evidence, indent=2, sort_keys=True) + "\n")
        identity["distribution_artifact"] = str(distribution_artifact.relative_to(tmp_path))
        identity["distribution_sha256"] = hashlib.sha256(distribution_artifact.read_bytes()).hexdigest()
    return {
        "schema": COMPARATOR_CENSUS_SCHEMA,
        "detector_population": list(DIRECT_DETECTOR_POPULATION),
        "conformance_oracle_population": list(CONFORMANCE_ORACLE_POPULATION),
        "excluded_semantic_mismatch_population": list(EXCLUDED_SEMANTIC_MISMATCH_POPULATION),
        "mitigation_population": ["FFmpeg vf_photosensitivity"],
        "reserve_detector_population": ["EPI-LENS"],
        "comparators": identities,
    }


def _census_entry(manifest: dict[str, object], name: str) -> dict[str, object]:
    entries = manifest["comparators"]
    assert isinstance(entries, list)
    return next(entry for entry in entries if entry["name"] == name)


def _rebind_distribution_artifact(entry: dict[str, object], artifact_root: Path) -> None:
    evidence = {
        "schema": "flashpatch-comparator-distribution-provenance-v1",
        "name": entry["name"],
        "repository_url": entry["repository_url"],
        "revision": entry["revision"],
        "source_checkout": entry["source_checkout"],
        "license": entry["license"],
        "license_sha256": entry["license_sha256"],
        "distribution": entry["distribution"],
        "distribution_revision": entry["distribution_revision"],
        "distribution_source_revision": entry["distribution_source_revision"],
        "release_asset_sha256": entry["release_asset_sha256"],
        "binary_sha256": entry["binary_sha256"],
        "configuration_sha256": entry["configuration_sha256"],
        "environment_sha256": entry["environment_sha256"],
        "command_sha256": entry["command_sha256"],
        "participant_conformance_sha256": entry["participant_conformance_sha256"],
    }
    artifact = artifact_root / str(entry["distribution_artifact"])
    artifact.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    entry["distribution_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()


def _bind_iris_spec_to_census(
    tmp_path: Path,
    spec: IrisReleaseSpec,
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "census-artifacts"
    manifest = _valid_comparator_census(artifact_root)
    iris = _census_entry(manifest, EA_IRIS_RELEASE_ORACLE_ID)
    binary_artifact = artifact_root / str(iris["binary_artifact"])
    configuration_artifact = artifact_root / str(iris["configuration_artifact"])
    shutil.copy2(spec.executable, binary_artifact)
    shutil.copy2(spec.appsettings, configuration_artifact)
    iris["binary_sha256"] = hashlib.sha256(binary_artifact.read_bytes()).hexdigest()
    iris["configuration_sha256"] = hashlib.sha256(configuration_artifact.read_bytes()).hexdigest()
    iris["distribution_source_revision"] = spec.source_revision
    iris["release_asset_sha256"] = spec.release_asset_sha256
    _rebind_distribution_artifact(iris, artifact_root)
    manifest_path = tmp_path / "iris-census.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    census_path = tmp_path / "iris-census-receipt.json"
    _ = write_comparator_census_receipt(manifest_path, artifact_root, census_path)
    return artifact_root, census_path


def _bind_tooflashy_spec_to_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ComparatorSpec, Path, Path]:
    artifact_root = tmp_path / "census-artifacts"
    manifest = _valid_comparator_census(artifact_root)
    manifest_path = tmp_path / "census.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    census_path = tmp_path / "census-receipt.json"
    _ = write_comparator_census_receipt(manifest_path, artifact_root, census_path)
    too_flashy = _census_entry(manifest, "TooFlashy")
    frozen_command = json.loads((artifact_root / str(too_flashy["command_artifact"])).read_text())
    monkeypatch.setattr(
        external_league,
        "_checkout_provenance",
        lambda spec: {
            "status": "VERIFIED",
            "path": str(spec.source_checkout),
            "head": spec.revision,
            "clean": True,
            "reason": None,
        },
    )
    spec = ComparatorSpec(
        name="TooFlashy",
        repository_url=str(too_flashy["repository_url"]),
        revision=str(too_flashy["revision"]),
        license=str(too_flashy["license"]),
        mode="detection",
        raw_output_mode="stdout",
        command=tuple(frozen_command),
        source_checkout=Path(str(too_flashy["source_checkout"])),
        working_directory=Path(str(too_flashy["source_checkout"])),
        distribution=str(too_flashy["distribution"]),
        distribution_revision=str(too_flashy["distribution_revision"]),
        configuration_sha256=str(too_flashy["configuration_sha256"]),
        environment_sha256=str(too_flashy["environment_sha256"]),
    )
    return spec, artifact_root, census_path


def _run_fair_runtime_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    _require_bubblewrap_namespaces()
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    protocol = _fair_runtime_protocol(tmp_path / "fair-runtime.lock")
    flashpatch = execute_repeated_flashpatch_detector(
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "flashpatch-fair-repeats",
        runtime_protocol=protocol,
    )
    spec, artifact_root, census_path = _bind_tooflashy_spec_to_census(tmp_path, monkeypatch)
    too_flashy = execute_repeated_comparator(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "tooflashy-fair-repeats",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
        runtime_protocol=protocol,
    )
    return flashpatch, too_flashy


def _run_scheduled_fair_runtime_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], Path]:
    _require_bubblewrap_namespaces()
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    video = tmp_path / "lane" / "canonical.ffv1.mkv"
    conversion = tmp_path / "lane" / "conversion-receipt.json"
    protocol = _fair_runtime_protocol(tmp_path / "fair-runtime.lock")
    schedule_path = tmp_path / "fair-runtime-schedule.json"
    schedule = write_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy"],
        protocol,
        hashlib.sha256(video.read_bytes()).hexdigest(),
        schedule_path,
        seed=20260802,
    )
    spec, artifact_root, census_path = _bind_tooflashy_spec_to_census(tmp_path, monkeypatch)
    child_receipts: dict[str, list[Path]] = {"FlashPatch": [], "TooFlashy": []}
    for assignment in schedule["slots"]:
        comparator = str(assignment["comparator"])
        slot = int(assignment["slot"])
        repeat = int(assignment["repeat_ordinal"])
        output = tmp_path / "scheduled-runs" / f"slot-{slot:02d}-{comparator.lower()}"
        if comparator == "FlashPatch":
            result = execute_flashpatch_detector(
                video,
                conversion,
                output,
                runtime_protocol=protocol,
                scheduled_repeat_ordinal=repeat,
                runtime_schedule=schedule_path,
                schedule_slot=slot,
            )
        else:
            result = execute_comparator(
                spec,
                video,
                conversion,
                output,
                census_receipt=census_path,
                census_artifact_root=artifact_root,
                runtime_protocol=protocol,
                scheduled_repeat_ordinal=repeat,
                runtime_schedule=schedule_path,
                schedule_slot=slot,
            )
        child_receipts[comparator].append(Path(str(result["receipt"])))
    flashpatch = write_scheduled_runtime_repeat_receipt(
        "FlashPatch",
        child_receipts["FlashPatch"],
        protocol,
        tmp_path / "scheduled-flashpatch-repeats.json",
    )
    too_flashy = write_scheduled_runtime_repeat_receipt(
        "TooFlashy",
        child_receipts["TooFlashy"],
        protocol,
        tmp_path / "scheduled-tooflashy-repeats.json",
    )
    return flashpatch, too_flashy, schedule_path


def _synthetic_external_population_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    participants: tuple[str, ...] = DIRECT_DETECTOR_POPULATION,
) -> tuple[list[Path], Path, dict[str, object], list[tuple[str, int]]]:
    frozen = freeze_fair_runtime_protocol(_fair_runtime_protocol(tmp_path / "join.lock"))
    input_sha256 = "f" * 64
    schedule_path = tmp_path / "join-schedule.json"
    schedule = freeze_fair_runtime_schedule(
        participants,
        frozen,
        input_sha256,
        seed=20260804,
    )
    _write_json(schedule_path, schedule)
    schedule_sha256 = external_league._canonical_json_sha256(schedule)
    schedule_artifact_sha256 = external_league._sha256_file(schedule_path)
    observed_stat = schedule_path.stat()
    schedule_stat = {
        "device": observed_stat.st_dev,
        "inode": observed_stat.st_ino,
        "size": observed_stat.st_size,
        "mtime_ns": observed_stat.st_mtime_ns,
        "ctime_ns": observed_stat.st_ctime_ns,
    }
    implementation = tmp_path / "synthetic-normalizer.py"
    implementation.write_text("# frozen synthetic normalizer\n", encoding="utf-8")
    protocol_sha256 = external_league._canonical_json_sha256(frozen)
    environment_sha256 = external_league._runtime_environment_sha256(frozen)
    child_by_key: dict[tuple[str, int], dict[str, object]] = {}
    children_by_comparator: dict[str, list[Path]] = {
        comparator: [] for comparator in participants
    }
    child_schema = {
        "FlashPatch": "flashpatch-l7-direct-detector-run-v1",
        KAYA_DIRECT_PARTICIPANT_ID: external_league.KAYA_FAIR_RUNTIME_RUN_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-run-v1",
    }
    repeat_schema = {
        "FlashPatch": "flashpatch-l7-direct-detector-repeats-v1",
        KAYA_DIRECT_PARTICIPANT_ID: external_league.KAYA_FAIR_RUNTIME_REPEATS_SCHEMA,
        "TooFlashy": "flashpatch-external-comparator-repeats-v1",
    }
    for assignment in schedule["slots"]:
        comparator = str(assignment["comparator"])
        ordinal = int(assignment["repeat_ordinal"])
        slot = int(assignment["slot"])
        binding = {
            "path": str(schedule_path.resolve()),
            "artifact_sha256": schedule_artifact_sha256,
            "schedule_sha256": schedule_sha256,
            "stat": schedule_stat,
            **assignment,
        }
        observation = {
            "schema": "flashpatch-l7-synthetic-parser-observation-v1",
            "comparator": comparator,
            "prediction": "SAFE",
        }
        terminal = {
            "schema": "flashpatch-l7-normalized-terminal-observation-v1",
            "normalizer": f"synthetic-{comparator}-v1",
            "implementation": {
                "path": str(implementation.resolve()),
                "sha256": external_league._sha256_file(implementation),
            },
            "sha256": external_league._canonical_json_sha256(observation),
        }
        started = slot * 1_000_000
        runtime = {
            "schema": external_league.FAIR_RUNTIME_RUN_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "measurement_boundary": dict(external_league.FAIR_RUNTIME_BOUNDARY),
            "environment_policy_sha256": environment_sha256,
            "observed_environment": {
                "child_probe": {
                    "effective_environment_sha256": "e" * 64,
                    "child_timing": {
                        "probe_started_monotonic_ns": started + 10,
                        "tool_started_monotonic_ns": started + 20,
                        "tool_finished_monotonic_ns": started + 80,
                    },
                }
            },
            "timeout_seconds": 120,
            "scheduled_repeat_ordinal": ordinal,
            "schedule_binding": binding,
            "attempt_ordinal": 1,
            "retry_count": 0,
            "retry_policy": "NO_RETRY",
            "started_monotonic_ns": started,
            "finished_monotonic_ns": started + 100,
            "wall_time_ns": 100,
            "timed_out": False,
            "input_identity_sha256": input_sha256,
            "normalized_terminal_observation": terminal,
        }
        child = {
            "schema": child_schema[comparator],
            "comparator": {"name": comparator},
            "fair_runtime": runtime,
            "observation": observation,
            "status": "PROCESS_VALID",
        }
        child_path = tmp_path / "join-children" / f"slot-{slot:02d}.json"
        _write_json(child_path, child)
        children_by_comparator[comparator].append(child_path)
        child_by_key[(comparator, ordinal)] = {
            "path": child_path,
            "sha256": external_league._sha256_file(child_path),
            "binding": binding,
            "observation": observation,
        }
    repeat_paths: list[Path] = []
    for comparator in participants:
        runs = []
        for child_path in children_by_comparator[comparator]:
            child = json.loads(child_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "repeat": child["fair_runtime"]["scheduled_repeat_ordinal"],
                    "status": "PROCESS_VALID",
                    "receipt": str(child_path.resolve()),
                    "receipt_sha256": external_league._sha256_file(child_path),
                    "normalized_observation_sha256": child["fair_runtime"]
                    ["normalized_terminal_observation"]["sha256"],
                    "fair_runtime": child["fair_runtime"],
                }
            )
        runs.sort(key=lambda row: row["repeat"])
        repeat_payload = {
            "schema": repeat_schema[comparator],
            "repeats_required": 3,
            "comparator": comparator,
            "fair_runtime_protocol": frozen,
            "fair_runtime_protocol_sha256": protocol_sha256,
            "runs": runs,
            "status": "PROCESS_REPRODUCIBLE",
            "scoreable": False,
        }
        if comparator == KAYA_DIRECT_PARTICIPANT_ID:
            repeat_payload.update(
                {
                    "claim_status": "NOT_SCOREABLE",
                    "comparison_eligible": False,
                    "external_claim_authorized": False,
                }
            )
        repeat_path = tmp_path / f"join-{comparator}-repeats.json"
        _write_json(repeat_path, repeat_payload)
        repeat_paths.append(repeat_path)
    verified_slots: list[dict[str, object]] = []
    for assignment in schedule["slots"]:
        comparator = str(assignment["comparator"])
        ordinal = int(assignment["repeat_ordinal"])
        child = child_by_key[(comparator, ordinal)]
        result_payload = {
            "schema": external_league.EXTERNAL_SLOT_CHILD_JOIN_SCHEMA,
            "status": "PROCESS_VALID",
            "slot": assignment["slot"],
            "comparator": comparator,
            "repeat_ordinal": ordinal,
            "child_receipt_sha256": child["sha256"],
            "ffv1_input_sha256": input_sha256,
            "schedule_binding": external_league._portable_schedule_binding(
                child["binding"]
            ),
            "parser_observation": child["observation"],
        }
        result_path = (
            tmp_path
            / "join-external-results"
            / f"slot-{int(assignment['slot']):02d}.json"
        )
        _write_json(result_path, result_payload)
        verified_slots.append(
            {
                **assignment,
                "result": {
                    "path": str(result_path.resolve()),
                    "sha256": external_league._sha256_file(result_path),
                    "size": result_path.stat().st_size,
                },
            }
        )
    witnessed = {
        "schema": EXTERNAL_HOST_VERIFICATION_SCHEMA_V2,
        "status": "VERIFIED",
        "witness_verified": True,
        "independent_host_identity_verified": True,
        "host": {"identity_sha256": "a" * 64},
        "request": str(tmp_path / "external-request.json"),
        "receipt": str(tmp_path / "external-receipt.json"),
        "verified_slots": verified_slots,
        "failures": [],
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "independent_gold_receipt_missing",
            "fair_population_receipt_conditions_unproven",
            "receipt_bound_score_verifier_missing",
        ],
    }
    reparsed: list[tuple[str, int]] = []

    def reopen(
        _path: Path,
        child: dict[str, object],
        comparator: str,
        observed_input_sha256: str,
    ) -> dict[str, object]:
        assert observed_input_sha256 == input_sha256
        runtime = child["fair_runtime"]
        reparsed.append((comparator, int(runtime["scheduled_repeat_ordinal"])))
        return dict(child["observation"])

    monkeypatch.setattr(
        external_league,
        "_observed_environment_matches_protocol",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        external_league,
        "_child_probe_and_command_are_bound",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        external_league, "_reopen_child_normalized_observation", reopen
    )
    monkeypatch.setattr(
        external_league,
        "verify_external_host_witness",
        lambda *_args, **_kwargs: witnessed,
    )
    return repeat_paths, schedule_path, witnessed, reparsed


def _run_decoder_parity_triplet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Path], Path, Path]:
    _require_bubblewrap_namespaces()
    lane = materialize_cfr_ffv1(
        _frames(tmp_path / "frames.npz"),
        tmp_path / "lane",
        fps=60,
    )
    video = tmp_path / "lane" / "canonical.ffv1.mkv"
    conversion = tmp_path / "lane" / "conversion-receipt.json"
    protocol = _fair_runtime_protocol(tmp_path / "decoder-parity.lock")
    flashpatch = execute_flashpatch_detector(
        video,
        conversion,
        tmp_path / "flashpatch-decoder-run",
        runtime_protocol=protocol,
        scheduled_repeat_ordinal=1,
    )

    release_asset = tmp_path / "iris-release.tar.gz"
    release_asset.write_bytes(b"official-release-asset")
    appsettings = tmp_path / "iris-appsettings.json"
    appsettings.write_text("{}")
    iris_executable = tmp_path / "IrisApp"
    iris_rows = "".join(
        f"{index + 1},00:00:00.{int(index * 1000 / 60):03d}000,0,0,0\n"
        for index in range(8)
    )
    iris_executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"rows = {iris_rows!r}\n"
        "out = Path('Results/canonical.ffv1.mkv/framedata.csv')\n"
        "out.write_text('Frame,TimeStamp,LuminanceFrameResult,RedFrameResult,PatternFrameResult\\n' + rows)\n"
        "print('Video FPS: 60')\n"
        "print('Total frames: 8')\n"
        "print('Video Result: PASS')\n"
    )
    iris_executable.chmod(0o755)
    iris_spec = IrisReleaseSpec(
        repository_url="https://github.com/electronicarts/IRIS",
        source_revision="a" * 40,
        release_tag="1.1.0",
        release_asset=release_asset,
        release_asset_sha256=hashlib.sha256(release_asset.read_bytes()).hexdigest(),
        executable=iris_executable,
        appsettings=appsettings,
        expected_fps=60,
    )
    iris_artifact_root, iris_census = _bind_iris_spec_to_census(tmp_path / "iris", iris_spec)
    iris = execute_iris_release(
        iris_spec,
        video,
        conversion,
        tmp_path / "iris-decoder-run",
        census_receipt=iris_census,
        census_artifact_root=iris_artifact_root,
    )

    too_spec, artifact_root, census_path = _bind_tooflashy_spec_to_census(
        tmp_path / "tooflashy",
        monkeypatch,
    )
    too_flashy = execute_comparator(
        too_spec,
        video,
        conversion,
        tmp_path / "tooflashy-decoder-run",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
    )
    return [
        Path(str(flashpatch["receipt"])),
        Path(str(iris["receipt"])),
        Path(str(too_flashy["receipt"])),
    ], video, conversion


def test_comparator_census_freezes_capability_separated_population_without_score_path(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    receipt = validate_comparator_census(manifest, artifact_root=tmp_path)

    assert receipt["status"] == "CENSUS_VALID"
    assert receipt["league_status"] == "NOT_SCOREABLE"
    assert receipt["scoreable"] is False
    assert receipt["external_claim_authorized"] is False
    assert receipt["detector_population"] == list(DIRECT_DETECTOR_POPULATION)
    assert receipt["conformance_oracle_population"] == list(CONFORMANCE_ORACLE_POPULATION)
    release = next(entry for entry in receipt["comparators"] if entry["name"] == EA_IRIS_RELEASE_ORACLE_ID)
    excluded = next(entry for entry in receipt["comparators"] if entry["name"] == EA_IRIS_SOURCE_ADAPTER_ID)
    kaya = next(entry for entry in receipt["comparators"] if entry["name"] == KAYA_DIRECT_PARTICIPANT_ID)
    assert release["lane"] == "conformance-oracle"
    assert release["name"] not in receipt["detector_population"]
    assert receipt["excluded_semantic_mismatch_population"] == [EA_IRIS_SOURCE_ADAPTER_ID]
    assert excluded["lane"] == "excluded-semantic-mismatch"
    assert excluded["name"] not in receipt["detector_population"]
    assert excluded["unscorable_reason"] == "semantic_conformance_mismatch_excluded"
    assert kaya["execution_status"] == "UNSCORABLE"
    assert kaya["unscorable_reason"] == "natural_corpus_gold_parity_and_fair_repeats_missing"
    assert kaya["binary_artifact"] is None
    assert kaya["participant_conformance"]["status"] == "VERIFIED"
    assert kaya["participant_conformance"]["scoreable"] is False
    assert receipt["mitigation_population"] == ["FFmpeg vf_photosensitivity"]
    assert len(receipt["manifest_sha256"]) == 64
    assert all(len(entry["provenance_sha256"]) == 64 for entry in receipt["comparators"])
    assert not {"winner", "ranking", "scores"}.intersection(receipt)


def test_comparator_census_rejects_lgpl_claim_for_actual_ffmpeg_distribution(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    _census_entry(manifest, "FFmpeg vf_photosensitivity")["license"] = "LGPL-2.1-or-later"

    with pytest.raises(ExternalLeagueError, match="GPL-3.0-or-later"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("FFmpeg vf_photosensitivity", "capability", "detector"),
        ("TooFlashy", "lane", "mitigation"),
        (KAYA_DIRECT_PARTICIPANT_ID, "capability", "mitigation"),
    ],
)
def test_comparator_census_rejects_detector_mitigation_mixing(tmp_path: Path, name: str, field: str, value: str) -> None:
    manifest = _valid_comparator_census(tmp_path)
    _census_entry(manifest, name)[field] = value

    with pytest.raises(ExternalLeagueError, match="capability or league lane"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_rejects_missing_provenance_hash(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    _census_entry(manifest, "TooFlashy")["command_sha256"] = None

    with pytest.raises(ExternalLeagueError, match="TooFlashy:command_sha256"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_reopens_artifacts_instead_of_accepting_fabricated_hashes(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    too_flashy = _census_entry(manifest, "TooFlashy")
    too_flashy["binary_sha256"] = "0" * 64

    with pytest.raises(ExternalLeagueError, match="artifact hash mismatches"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_rejects_flashpatch_revision_other_than_executing_checkout(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    _census_entry(manifest, "FlashPatch")["revision"] = "f" * 40

    with pytest.raises(ExternalLeagueError, match="executing checkout"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_rejects_dirty_or_unpushed_flashpatch_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    flashpatch = _census_entry(manifest, "FlashPatch")
    monkeypatch.setattr(
        external_league,
        "_flashpatch_checkout_provenance",
        lambda: {
            "revision": flashpatch["revision"],
            "tree": "a" * 40,
            "origin": flashpatch["repository_url"],
            "clean": False,
            "pushed": False,
        },
    )

    with pytest.raises(ExternalLeagueError, match="clean checkout at a pushed revision"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("artifact_field", "hash_field", "material", "message"),
    [
        ("license_artifact", "license_sha256", "GNU LGPL 2.1 or later\n", "effective GPL-3"),
        ("configuration_artifact", "configuration_sha256", "configuration: --enable-shared\n", "GPL-enabled"),
        ("distribution_artifact", "distribution_sha256", "unrelated package\n", "does not bind"),
    ],
)
def test_comparator_census_rejects_semantically_unbound_ffmpeg_build_evidence(
    tmp_path: Path,
    artifact_field: str,
    hash_field: str,
    material: str,
    message: str,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    ffmpeg = _census_entry(manifest, "FFmpeg vf_photosensitivity")
    artifact = tmp_path / str(ffmpeg[artifact_field])
    artifact.write_text(material)
    ffmpeg[hash_field] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_field != "distribution_artifact":
        _rebind_distribution_artifact(ffmpeg, tmp_path)
    else:
        message = "distribution, command, or environment artifact"

    with pytest.raises(ExternalLeagueError, match=message):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_detects_artifact_tampering_after_manifest_freeze(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    iris = _census_entry(manifest, EA_IRIS_RELEASE_ORACLE_ID)
    (tmp_path / str(iris["command_artifact"])).write_text("tampered command\n")

    with pytest.raises(ExternalLeagueError, match="artifact hash mismatches"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_rejects_ambiguous_tooflashy_or_epi_lens_identity(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    _census_entry(manifest, "TooFlashy")["name"] = "TooFlashy_or_EPI_LENS"

    with pytest.raises(ExternalLeagueError, match="ambiguous comparator identity"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_keeps_epi_lens_unscorable_without_full_runner(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    epi_lens = _census_entry(manifest, "EPI-LENS")
    epi_lens["execution_status"] = "RUNNABLE"
    epi_lens["unscorable_reason"] = None

    with pytest.raises(ExternalLeagueError, match="remains UNSCORABLE"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_keeps_iris_source_adapter_excluded_after_semantic_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    adapter = _census_entry(manifest, EA_IRIS_SOURCE_ADAPTER_ID)
    adapter["execution_status"] = "RUNNABLE"
    adapter["unscorable_reason"] = None

    with pytest.raises(ExternalLeagueError, match="excluded semantic-mismatch baseline"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_requires_verified_kaya_conformance_receipt(
    tmp_path: Path,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    kaya = _census_entry(manifest, KAYA_DIRECT_PARTICIPANT_ID)
    receipt_path = tmp_path / str(kaya["participant_conformance_artifact"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "NOT_VERIFIED"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    kaya["participant_conformance_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    _rebind_distribution_artifact(kaya, tmp_path)

    with pytest.raises(ExternalLeagueError, match="cannot authorize"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_cannot_promote_kaya_to_scoreable_or_runnable(
    tmp_path: Path,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    kaya = _census_entry(manifest, KAYA_DIRECT_PARTICIPANT_ID)
    kaya["execution_status"] = "RUNNABLE"
    kaya["unscorable_reason"] = None

    with pytest.raises(ExternalLeagueError, match="UNSCORABLE participant"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "forbidden_identity",
    [EA_IRIS_RELEASE_ORACLE_ID, EA_IRIS_SOURCE_ADAPTER_ID, KAYA_PROTOTYPE_ID],
)
def test_comparator_census_rejects_nonparticipant_identity_in_direct_detector_population(
    tmp_path: Path,
    forbidden_identity: str,
) -> None:
    manifest = _valid_comparator_census(tmp_path)
    manifest["detector_population"] = [
        "FlashPatch",
        forbidden_identity,
        "TooFlashy",
    ]

    with pytest.raises(ExternalLeagueError, match=KAYA_DIRECT_PARTICIPANT_ID):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_comparator_census_rejects_detector_inside_mitigation_population(tmp_path: Path) -> None:
    manifest = _valid_comparator_census(tmp_path)
    manifest["mitigation_population"] = ["FFmpeg vf_photosensitivity", "FlashPatch"]

    with pytest.raises(ExternalLeagueError, match="mitigation population"):
        validate_comparator_census(manifest, artifact_root=tmp_path)


def test_tooflashy_distribution_rejects_any_environment_key_outside_allowlist(tmp_path: Path) -> None:
    checkout = tmp_path / "TooFlashy"
    checkout.mkdir()
    # 실행 파일 위치는 기계마다 다르다. 없으면 그 검사만 건너뛴다.
    uv_path = shutil.which("uv")
    if uv_path is None:
        pytest.skip("uv executable is unavailable")
    uv = Path(uv_path)
    if not uv.is_file():
        pytest.skip("canonical uv executable is unavailable")
    entry = {
        "source_checkout": str(checkout),
        "binary_sha256": hashlib.sha256(uv.read_bytes()).hexdigest(),
    }
    command = [str(uv), "run", "tooflashy", "--json", "{input}"]
    environment = {
        "PATH": "/usr/bin:/bin",
        "UV_PROJECT": str(checkout.resolve()),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "foreign-venv"),
    }

    with pytest.raises(ExternalLeagueError, match="canonical uv command"):
        _REAL_VERIFY_TOOFLASHY_DISTRIBUTION(entry, command, environment)


def test_comparator_census_rejects_score_or_winner_fields(tmp_path: Path) -> None:
    for forbidden in ("scoreable", "scores", "winner"):
        manifest = copy.deepcopy(_valid_comparator_census(tmp_path / forbidden))
        manifest[forbidden] = True
        with pytest.raises(ExternalLeagueError, match="manifest fields"):
            validate_comparator_census(manifest, artifact_root=tmp_path / forbidden)


def test_comparator_census_writer_binds_source_manifest_and_durable_receipt(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = _valid_comparator_census(artifact_root)
    manifest_path = tmp_path / "census-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt_path = tmp_path / "census-receipt.json"

    receipt = write_comparator_census_receipt(manifest_path, artifact_root, receipt_path)

    assert receipt["receipt"] == str(receipt_path.resolve())
    assert receipt["source_manifest"]["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert json.loads(receipt_path.read_text())["league_status"] == "NOT_SCOREABLE"


def test_materializer_freezes_lossless_cfr_input_and_proves_roundtrip(tmp_path: Path) -> None:
    receipt = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)

    assert receipt["roundtrip"]["byte_identical"] is True
    assert receipt["cfr"]["frame_count"] == 8
    assert len(receipt["renderer_rgb"]["frame_sha256"]) == 8
    saved = json.loads((tmp_path / "lane" / "conversion-receipt.json").read_text())
    assert saved["canonical_video"]["sha256"] == receipt["canonical_video"]["sha256"]


def test_materializer_rejects_non_cfr_renderer_timeline(tmp_path: Path) -> None:
    with pytest.raises(ExternalLeagueError, match="exact CFR"):
        materialize_cfr_ffv1(_frames(tmp_path / "vfr.npz", cfr=False), tmp_path / "lane", fps=60)


def test_decoder_timeline_parity_reopens_native_outputs_and_reports_upstream_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts, video, conversion = _run_decoder_parity_triplet(tmp_path, monkeypatch)
    destination = tmp_path / "decoder-timeline-parity.json"

    result = verify_decoder_timeline_parity(
        receipts,
        video,
        conversion,
        destination=destination,
    )

    assert result["status"] == "NOT_VERIFIED"
    assert result["primary_case_level_comparison"] == "NOT_VERIFIED"
    assert result["secondary_interval_comparison"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("run_receipt_comparator_invalid:")
        for failure in result["failures"]
    )
    assert f"run_receipt_missing:{KAYA_DIRECT_PARTICIPANT_ID}" in result["failures"]
    assert result["claim_status"] == "NOT_SCOREABLE"
    assert len(result["canonical_contract"]["frame_map"]) == 8
    assert len(result["canonical_contract"]["frame_map_sha256"]) == 64
    rows = {row["comparator"]: row for row in result["comparators"]}
    assert rows["FlashPatch"]["status"] == "VERIFIED"
    assert rows["FlashPatch"]["secondary_interval_endpoint"]["status"] == "VERIFIED"
    assert EA_IRIS_RELEASE_ORACLE_ID not in rows
    assert EA_IRIS_SOURCE_ADAPTER_ID not in rows
    assert KAYA_DIRECT_PARTICIPANT_ID not in rows
    assert rows["TooFlashy"]["status"] == "NOT_VERIFIED"
    assert rows["TooFlashy"]["decoder_timeline"]["parity_reason"] == "native_json_omits_per_frame_decode_identity_and_timestamps"
    assert rows["TooFlashy"]["secondary_interval_endpoint"] == {
        "status": "NOT_VERIFIED",
        "reason": "native_tool_does_not_expose_interval_endpoint",
        "value": None,
    }
    assert len(result["parity_blockers"]) == 1
    assert json.loads(destination.read_text())["status"] == "NOT_VERIFIED"
    forbidden = {"score", "scores", "rank", "winner", "gold"}

    def assert_no_forbidden_keys(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & forbidden)
            for child in value.values():
                assert_no_forbidden_keys(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_forbidden_keys(child)

    assert_no_forbidden_keys(result)


def test_decoder_timeline_parity_rejects_untrusted_audit_wrong_source_and_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts, video, conversion = _run_decoder_parity_triplet(tmp_path, monkeypatch)
    too_receipt = next(path for path in receipts if "tooflashy" in path.name or "tooflashy" in str(path.parent))
    original_child_bytes = too_receipt.read_bytes()
    original_child = json.loads(original_child_bytes)
    raw_path = too_receipt.parent / original_child["raw_output"]["path"]
    original_raw_bytes = raw_path.read_bytes()
    original_raw = json.loads(original_raw_bytes)

    conversion_payload = json.loads(conversion.read_text())
    forged = copy.deepcopy(original_raw)
    forged["decode_audit"] = {
        "schema": "tooflashy-native-decode-audit-v1",
        "evidence_origin": "native_decoder_callback",
        "canonical_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "conversion_receipt_sha256": hashlib.sha256(conversion.read_bytes()).hexdigest(),
        "decoded_rgb_sha256": conversion_payload["renderer_rgb"]["raw_sha256"],
        "frames": conversion_payload["renderer_rgb"]["frame_sha256"],
    }
    raw_path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    forged_child = copy.deepcopy(original_child)
    forged_child["raw_output"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    too_receipt.write_text(json.dumps(forged_child, indent=2, sort_keys=True) + "\n")
    forged_result = verify_decoder_timeline_parity(receipts, video, conversion)
    forged_rows = {row["comparator"]: row for row in forged_result["comparators"]}
    assert forged_result["status"] == "NOT_VERIFIED"
    assert forged_rows["TooFlashy"]["status"] == "NOT_VERIFIED"
    assert forged_rows["TooFlashy"]["decoder_timeline"]["parity_reason"] == "untrusted_unfrozen_decode_audit_augmentation"

    raw_path.write_bytes(original_raw_bytes)
    wrong_source = tmp_path / "same-bytes-different-source.mkv"
    shutil.copy2(video, wrong_source)
    wrong_source_raw = copy.deepcopy(original_raw)
    wrong_source_raw["path"] = str(wrong_source)
    raw_path.write_text(json.dumps(wrong_source_raw, sort_keys=True) + "\n")
    wrong_source_child = copy.deepcopy(original_child)
    wrong_source_child["input"]["path"] = str(wrong_source)
    wrong_source_child["raw_output"]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    too_receipt.write_text(json.dumps(wrong_source_child, indent=2, sort_keys=True) + "\n")
    wrong_source_result = verify_decoder_timeline_parity(receipts, video, conversion)
    assert wrong_source_result["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("decoder_timeline_evidence_invalid:TooFlashy:")
        for failure in wrong_source_result["failures"]
    )

    raw_path.write_bytes(original_raw_bytes)
    too_receipt.write_bytes(original_child_bytes)
    escaped_raw = too_receipt.parent.parent / "escaped-tooflashy.json"
    escaped_raw.write_bytes(original_raw_bytes)
    escaped_child = copy.deepcopy(original_child)
    escaped_child["raw_output"]["path"] = "../escaped-tooflashy.json"
    escaped_child["raw_output"]["sha256"] = hashlib.sha256(escaped_raw.read_bytes()).hexdigest()
    too_receipt.write_text(json.dumps(escaped_child, indent=2, sort_keys=True) + "\n")
    escaped_result = verify_decoder_timeline_parity(receipts, video, conversion)
    assert escaped_result["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("decoder_timeline_evidence_invalid:TooFlashy:")
        and "escapes the child run root" in failure
        for failure in escaped_result["failures"]
    )

    too_receipt.write_bytes(original_child_bytes)
    raw_path.unlink()
    missing_raw_result = verify_decoder_timeline_parity(receipts, video, conversion)
    assert missing_raw_result["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("decoder_timeline_evidence_invalid:TooFlashy:")
        for failure in missing_raw_result["failures"]
    )


@pytest.mark.parametrize("mutation", ["duplicate", "reorder", "timestamp"])
def test_decoder_timeline_parity_never_accepts_release_oracle_as_source_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipts, video, conversion = _run_decoder_parity_triplet(tmp_path, monkeypatch)
    iris_receipt = next(path for path in receipts if "iris" in str(path.parent) and "tooflashy" not in str(path.parent))
    child = json.loads(iris_receipt.read_text())
    csv_path = iris_receipt.parent / child["frame_report"]["path"]
    lines = csv_path.read_text().splitlines()
    if mutation == "duplicate":
        fields = lines[4].split(",")
        fields[0] = "3"
        lines[4] = ",".join(fields)
    elif mutation == "reorder":
        lines[3], lines[4] = lines[4], lines[3]
    else:
        fields = lines[4].split(",")
        fields[1] = "00:00:00.052000"
        lines[4] = ",".join(fields)
    csv_path.write_text("\n".join(lines) + "\n")
    child["frame_report"]["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    iris_receipt.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")

    result = verify_decoder_timeline_parity(receipts, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("run_receipt_comparator_invalid:")
        for failure in result["failures"]
    )
    assert f"run_receipt_missing:{KAYA_DIRECT_PARTICIPANT_ID}" in result["failures"]


def test_decoder_timeline_parity_rejects_flashpatch_mask_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts, video, conversion = _run_decoder_parity_triplet(tmp_path, monkeypatch)
    flashpatch_receipt = next(path for path in receipts if "flashpatch-decoder" in str(path.parent))
    child = json.loads(flashpatch_receipt.read_text())
    mask_path = flashpatch_receipt.parent / child["hazard_mask"]["path"]
    escaped_mask = flashpatch_receipt.parent.parent / "escaped-hazard-mask.npy"
    shutil.copy2(mask_path, escaped_mask)
    child["hazard_mask"]["path"] = "../escaped-hazard-mask.npy"
    child["hazard_mask"]["sha256"] = hashlib.sha256(escaped_mask.read_bytes()).hexdigest()
    flashpatch_receipt.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")

    result = verify_decoder_timeline_parity(receipts, video, conversion)

    assert result["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("decoder_timeline_evidence_invalid:FlashPatch:")
        and "escapes the child run root" in failure
        for failure in result["failures"]
    )


def test_decoder_timeline_parity_rejects_conversion_drift_and_missing_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts, video, conversion = _run_decoder_parity_triplet(tmp_path, monkeypatch)
    flashpatch_receipt = next(path for path in receipts if "flashpatch-decoder" in str(path.parent))
    child = json.loads(flashpatch_receipt.read_text())
    child["conversion_receipt"]["sha256"] = "0" * 64
    flashpatch_receipt.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")

    drift = verify_decoder_timeline_parity(receipts, video, conversion)
    missing = verify_decoder_timeline_parity(receipts[:-1], video, conversion)

    assert drift["status"] == "NOT_VERIFIED"
    assert any(
        failure.startswith("decoder_timeline_evidence_invalid:FlashPatch:")
        for failure in drift["failures"]
    )
    assert missing["status"] == "NOT_VERIFIED"
    assert "run_receipt_missing:TooFlashy" in missing["failures"]


def test_renderer_png_packer_requires_a_contiguous_actual_capture_sequence(tmp_path: Path) -> None:
    import cv2

    source = tmp_path / "capture"
    source.mkdir()
    for index, value in enumerate((12, 34, 56)):
        frame = np.full((4, 6, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(source / f"frame_{index:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    result = pack_renderer_png_sequence(source, tmp_path / "packed", fps=60)

    assert result["status"] == "RENDERER_SEQUENCE_PACKED"
    assert result["cfr"]["timestamps_us"] == [0, 16667, 33333]
    with np.load(tmp_path / "packed" / "renderer-frames.npz") as packed:
        assert packed["frames"].shape == (3, 4, 6, 3)
        assert packed["frames"][1, 0, 0, 0] == 34

    (source / "frame_00001.png").rename(source / "frame_00003.png")
    with pytest.raises(ExternalLeagueError, match="contiguous"):
        pack_renderer_png_sequence(source, tmp_path / "rejected", fps=60)


def test_comparator_receipt_binds_pinned_binary_input_and_raw_output(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(
        name="test-detector", repository_url="https://example.invalid/test-detector", revision="a" * 40,
        license="MIT", mode="detection",
        command=(sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes(b'raw:' + pathlib.Path(sys.argv[1]).read_bytes()[:8])", "{input}", "{output}"),
    )
    result = execute_comparator(spec, tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "detector")

    assert result["status"] == "PROCESS_VALID"
    assert result["input"]["sha256"] == lane["canonical_video"]["sha256"]
    assert result["raw_output"]["exists"] is True
    assert len(result["comparator"]["binary_sha256"]) == 64


def test_comparator_can_bind_stdout_as_its_declared_raw_output(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(
        name="stdout-detector", repository_url="https://example.invalid/stdout-detector", revision="b" * 40,
        license="MIT", mode="detection", raw_output_mode="stdout",
        command=(sys.executable, "-c", "print('detected')", "{input}"),
    )
    result = execute_comparator(spec, tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "detector")

    assert result["status"] == "PROCESS_VALID"
    assert result["raw_output"]["mode"] == "stdout"
    assert (tmp_path / "detector" / "raw-output.bin").read_bytes() == b"detected\n"


def test_comparator_can_declare_hazard_exit_code_as_valid_execution(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(
        name="hazard-exit-detector", repository_url="https://example.invalid/hazard-exit", revision="c" * 40,
        license="MIT", mode="detection", raw_output_mode="stdout", expected_exit_codes=(0, 1),
        command=(sys.executable, "-c", "print('hazard'); raise SystemExit(1)", "{input}"),
    )
    result = execute_comparator(spec, tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "detector")

    assert result["exit_code"] == 1
    assert result["status"] == "PROCESS_VALID"


def test_comparator_rejects_video_not_owned_by_conversion_receipt(tmp_path: Path) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    fake_video = tmp_path / "fake.mkv"
    fake_video.write_bytes(b"not the receipt-owned video")
    spec = ComparatorSpec(name="detector", repository_url="https://example.invalid/d", revision="d" * 40, license="MIT", mode="detection", command=(sys.executable, "-c", "print(1)", "{input}"), raw_output_mode="stdout")
    with pytest.raises(ExternalLeagueError, match="hash does not match"):
        execute_comparator(spec, fake_video, tmp_path / "lane" / "conversion-receipt.json", tmp_path / "detector")


def test_tooflashy_parser_requires_receipt_bound_case_level_output(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    raw = tmp_path / "tooflashy.json"
    raw.write_text(json.dumps({"path": str(tmp_path / "lane" / "canonical.ffv1.mkv"), "passes": False, "fps": 60.0, "frame_count": 8, "event_count": 7, "failures": ["flash"]}))
    parsed = parse_tooflashy_json(raw, tmp_path / "lane" / "canonical.ffv1.mkv", expected_fps=60, expected_frame_count=8)
    assert parsed["prediction"] == "HAZARDOUS"
    assert parsed["timestamp_metrics"] == "NOT_APPLICABLE"


def _tooflashy_oldfilm_repeat(root: Path, video: bytes, *, event_count: int = 1) -> Path:
    root.mkdir()
    (root / "repo").mkdir()
    staged = root / "canonical.ffv1.mkv"
    staged.write_bytes(video)
    (root / "tooflashy.json").write_text(
        json.dumps(
            {
                "path": str(staged.resolve()),
                "passes": True,
                "fps": 60.0,
                "frame_count": 150,
                "event_count": event_count,
                "failures": [],
            }
        )
        + "\n"
    )
    return root


def test_tooflashy_oldfilm_repeats_reject_staged_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "canonical.ffv1.mkv"
    source.write_bytes(b"canonical")
    monkeypatch.setattr(
        external_league,
        "TOOFLASHY_OLDFILM_CANONICAL_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    roots = [
        _tooflashy_oldfilm_repeat(tmp_path / f"run-{ordinal}", b"canonical")
        for ordinal in range(1, 4)
    ]
    (roots[0] / "canonical.ffv1.mkv").write_bytes(b"different")

    with pytest.raises(ExternalLeagueError, match="staged input hash mismatch"):
        verify_tooflashy_oldfilm_repeats(source, roots)


def test_tooflashy_oldfilm_repeats_reject_normalized_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "canonical.ffv1.mkv"
    source.write_bytes(b"canonical")
    monkeypatch.setattr(
        external_league,
        "TOOFLASHY_OLDFILM_CANONICAL_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(external_league, "_verify_tooflashy_repeat_checkout", lambda checkout: None)
    roots = [
        _tooflashy_oldfilm_repeat(
            tmp_path / f"run-{ordinal}",
            b"canonical",
            event_count=2 if ordinal == 3 else 1,
        )
        for ordinal in range(1, 4)
    ]

    with pytest.raises(ExternalLeagueError, match="repeat observations disagree"):
        verify_tooflashy_oldfilm_repeats(source, roots)


def _native_main_comparator_case(tmp_path: Path) -> Path:
    case = tmp_path / "native-case"
    case.mkdir()
    frames = np.arange(3 * 8 * 8 * 3, dtype=np.uint8).reshape(3, 8, 8, 3)
    np.savez_compressed(
        case / "renderer-frames.npz",
        frames=frames,
        timestamps=np.arange(3, dtype=np.float64) / 60.0,
    )
    (case / "trace.json").write_text(json.dumps({"fixed_fps": 60}), encoding="utf-8")
    (case / "capture.json").write_text("{}", encoding="utf-8")
    native = {
        "frame_artifact_path": "renderer-frames.npz",
        "trace_path": "trace.json",
        "execution_receipt_path": "capture.json",
    }
    for field, filename in (
        ("frame_artifact_sha256", "renderer-frames.npz"),
        ("trace_sha256", "trace.json"),
        ("execution_receipt_sha256", "capture.json"),
    ):
        native[field] = hashlib.sha256((case / filename).read_bytes()).hexdigest()
    (case / "native-main-natural-case.json").write_text(
        json.dumps({"native_main": native}), encoding="utf-8"
    )
    return case


def _native_main_assessment() -> dict[str, object]:
    return {
        "case_id": "oldfilm-native-main",
        "status": "NOT_SCOREABLE",
        "scoreable": False,
        "native_equivalence": "NOT_ESTABLISHED",
        "external_claim_authorized": False,
        "renderer_rgb_sha256": "a" * 64,
        "timestamps_sha256": "b" * 64,
    }


def test_native_main_comparator_input_binds_reopened_case_to_ffv1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _native_main_comparator_case(tmp_path)
    monkeypatch.setattr(l7_verify, "verify_native_main_natural_case_bundle", lambda root: _native_main_assessment())

    result = materialize_native_main_comparator_input(case, tmp_path / "lane")

    assert result["status"] == "NOT_SCOREABLE"
    assert result["scoreable"] is False
    assert result["case"]["fixed_fps"] == 60
    assert result["conversion"]["roundtrip"]["byte_identical"] is True
    assert (tmp_path / "lane" / "native-main-comparator-input-receipt.json").is_file()


def test_native_main_comparator_input_rejects_changed_case_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _native_main_comparator_case(tmp_path)
    monkeypatch.setattr(l7_verify, "verify_native_main_natural_case_bundle", lambda root: _native_main_assessment())
    ledger = json.loads((case / "native-main-natural-case.json").read_text())
    ledger["native_main"]["trace_sha256"] = "0" * 64
    (case / "native-main-natural-case.json").write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ExternalLeagueError, match="hash binding changed"):
        materialize_native_main_comparator_input(case, tmp_path / "lane")


def test_comparator_marks_undeclared_source_checkout_unverified(tmp_path: Path) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(name="detector", repository_url="https://example.invalid/d", revision="e" * 40, license="MIT", mode="detection", raw_output_mode="stdout", command=(sys.executable, "-c", "print(1)", "{input}"))
    result = execute_comparator(spec, tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "detector")
    assert result["comparator"]["source_checkout"]["status"] == "UNVERIFIED"


def test_repeated_comparator_rejects_raw_byte_only_repeat_equivalence(tmp_path: Path) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(name="detector", repository_url="https://example.invalid/d", revision="f" * 40, license="MIT", mode="detection", raw_output_mode="stdout", command=(sys.executable, "-c", "print('fixed')", "{input}"))
    result = execute_repeated_comparator(spec, tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "repeats")
    assert result["status"] == "INCONCLUSIVE"
    assert len(result["runs"]) == 3
    assert "normalized_terminal_observation_missing" in result["scoreable_blockers"]


def test_iris_parser_separates_case_failure_from_frame_level_locations(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    frames = tmp_path / "frameData.json"
    result.write_text(json.dumps({"OverallResult": "Fail", "TotalFrame": 3}))
    frames.write_text(json.dumps({"LineGraphFrameData": {"LuminanceFrameResult": [0, 3, 0], "RedFrameResult": [0, 0, 0], "PatternFrameResult": [0, 0, 0]}}))
    parsed = parse_iris_json(result, frames, expected_frame_count=3)
    assert parsed["prediction"] == "HAZARDOUS"
    assert parsed["hazard_frame_indices"] == [1]
    assert parsed["tool"] == EA_IRIS_LEGACY_JSON_ID
    assert parsed["tool"] != EA_IRIS_SOURCE_ADAPTER_ID
    assert parsed["comparison_eligible"] is False
    assert parsed["comparison_blocker"] == "source_build_frame_ledger_and_release_conformance_missing"


def test_iris_release_parser_binds_csv_and_terminal_stdout(tmp_path: Path) -> None:
    csv_path = tmp_path / "framedata.csv"
    stdout = tmp_path / "stdout.txt"
    csv_path.write_bytes(
        b"Frame,TimeStamp,LuminanceFrameResult,RedFrameResult,PatternFrameResult\n"
        b"1,00:00:00.000000,0,0,0\x00\n"
        b"2,00:00:00.016000,3,0,0\x00\n"
        b"3,00:00:00.033000,0,0,0\x00\n"
    )
    stdout.write_text("Video FPS: 60\nTotal frames: 3\nVideo Result: FAIL\n")
    parsed = parse_iris_release_csv(csv_path, stdout, expected_frame_count=3, expected_fps=60)
    assert parsed["prediction"] == "HAZARDOUS"
    assert parsed["hazard_frame_indices"] == [1]


def test_iris_release_parser_rejects_terminal_frame_conflict(tmp_path: Path) -> None:
    csv_path = tmp_path / "framedata.csv"
    stdout = tmp_path / "stdout.txt"
    csv_path.write_text("Frame,TimeStamp,LuminanceFrameResult,RedFrameResult,PatternFrameResult\n1,00:00:00.000000,3,0,0\n")
    stdout.write_text("Video FPS: 60\nTotal frames: 1\nVideo Result: PASS\n")
    with pytest.raises(ExternalLeagueError, match="conflicts"):
        parse_iris_release_csv(csv_path, stdout, expected_frame_count=1, expected_fps=60)


def test_iris_release_runner_binds_release_asset_input_and_csv(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    release_asset = tmp_path / "iris-release.tar.gz"
    release_asset.write_bytes(b"official-release-asset")
    appsettings = tmp_path / "appsettings.json"
    appsettings.write_text("{}")
    executable = tmp_path / "IrisApp"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "out = Path('Results/canonical.ffv1.mkv/framedata.csv')\n"
        "out.write_text('Frame,TimeStamp,LuminanceFrameResult,RedFrameResult,PatternFrameResult\\n' + ''.join(f'{i},00:00:00.000000,{3 if i == 2 else 0},0,0\\n' for i in range(1, 9)))\n"
        "print('Video FPS: 60')\nprint('Total frames: 8')\nprint('Video Result: FAIL')\n"
    )
    executable.chmod(0o755)
    spec = IrisReleaseSpec(
        repository_url="https://github.com/electronicarts/IRIS",
        source_revision="a" * 40,
        release_tag="1.1.0",
        release_asset=release_asset,
        release_asset_sha256=hashlib.sha256(release_asset.read_bytes()).hexdigest(),
        executable=executable,
        appsettings=appsettings,
        expected_fps=60,
    )
    artifact_root, census_path = _bind_iris_spec_to_census(tmp_path, spec)
    result = execute_iris_release(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "iris-run",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
    )
    assert result["status"] == "PROCESS_VALID"
    assert result["parsed_observation"]["prediction"] == "HAZARDOUS"
    assert result["input"]["sha256"] == lane["canonical_video"]["sha256"]


def test_repeated_iris_release_requires_same_frame_report_and_observation(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    release_asset = tmp_path / "iris-release.tar.gz"
    release_asset.write_bytes(b"official-release-asset")
    appsettings = tmp_path / "appsettings.json"
    appsettings.write_text("{}")
    executable = tmp_path / "IrisApp"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('Results/canonical.ffv1.mkv/framedata.csv').write_text('Frame,TimeStamp,LuminanceFrameResult,RedFrameResult,PatternFrameResult\\n' + ''.join(f'{i},00:00:00.000000,{3 if i == 2 else 0},0,0\\n' for i in range(1, 9)))\n"
        "print('Video FPS: 60')\nprint('Total frames: 8')\nprint('Video Result: FAIL')\n"
    )
    executable.chmod(0o755)
    spec = IrisReleaseSpec("https://github.com/electronicarts/IRIS", "a" * 40, "1.1.0", release_asset, hashlib.sha256(release_asset.read_bytes()).hexdigest(), executable, appsettings, 60)
    artifact_root, census_path = _bind_iris_spec_to_census(tmp_path, spec)
    result = execute_repeated_iris_release(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "iris-repeats",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
    )
    assert result["status"] == "PROCESS_REPRODUCIBLE"
    assert len(result["runs"]) == 3


def test_flashpatch_detector_binds_renderer_npz_to_canonical_video(tmp_path: Path) -> None:
    lane = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    result = execute_flashpatch_detector(tmp_path / "lane" / "canonical.ffv1.mkv", tmp_path / "lane" / "conversion-receipt.json", tmp_path / "flashpatch")
    assert result["status"] == "PROCESS_VALID"
    assert result["input"]["canonical_video_sha256"] == lane["canonical_video"]["sha256"]
    assert (tmp_path / "flashpatch" / "hazard-mask.npy").is_file()


def test_repeated_flashpatch_detector_requires_identical_mask_and_observation(tmp_path: Path) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    result = execute_repeated_flashpatch_detector(
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "flashpatch-repeats",
    )
    assert result["status"] == "PROCESS_REPRODUCIBLE"
    assert len(result["runs"]) == 3


def test_fair_runtime_protocol_freezes_full_environment_budget_and_boundary(tmp_path: Path) -> None:
    frozen = freeze_fair_runtime_protocol(_fair_runtime_protocol(tmp_path / "runtime.lock"))

    assert frozen["measurement_boundary"] == external_league.FAIR_RUNTIME_BOUNDARY
    assert frozen["machine"]["id"]
    assert frozen["machine"]["operating_system"]
    assert frozen["machine"]["architecture"]
    assert frozen["cpu"]["model"]
    assert frozen["cpu"]["logical_count"] == os.cpu_count()
    assert set(frozen["cpu"]["affinity"]).issubset(os.sched_getaffinity(0))
    assert frozen["threads"] == {"limit": 1}
    assert frozen["gpu"] == {
        "policy": "DISABLED",
        "device": None,
        "isolation": "BWRAP_EMPTY_DEV",
    }
    assert frozen["effective_environment_policy"] == external_league.FAIR_RUNTIME_EFFECTIVE_ENVIRONMENT_POLICY
    assert frozen["cache"] == {"policy": "WARM_INPUT_PRETOUCHED"}
    assert frozen["concurrency"] == {
        "limit": 1,
        "lock_path": str((tmp_path / "runtime.lock").resolve()),
        "process_isolation": "FRESH_SUBPROCESS_PER_REPEAT",
    }
    assert frozen["budget"] == {
        "timeout_seconds": 120,
        "scheduled_repeats": 3,
        "retry_policy": "NO_RETRY",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repeats_required", 2, "three scheduled runs"),
        ("retry_policy", "RETRY_ONCE", "no retries"),
        ("concurrency_limit", 2, "concurrency limit one"),
        ("process_isolation", "SHARED_PROCESS", "process isolation"),
        ("gpu_policy", "FIXED_DEVICE", "CPU-only"),
    ],
)
def test_fair_runtime_protocol_rejects_unfair_policy(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    unfair = FairRuntimeProtocol(**{**protocol.__dict__, field: value})

    with pytest.raises(ExternalLeagueError, match=message):
        freeze_fair_runtime_protocol(unfair)


def test_flashpatch_and_external_detector_emit_symmetric_fair_runtime_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)

    assert flashpatch["status"] == "PROCESS_REPRODUCIBLE"
    assert too_flashy["status"] == "PROCESS_REPRODUCIBLE"
    assert flashpatch["fair_runtime_protocol"] == too_flashy["fair_runtime_protocol"]
    assert [run["repeat"] for run in flashpatch["runs"]] == [1, 2, 3]
    assert [run["repeat"] for run in too_flashy["runs"]] == [1, 2, 3]
    for result in (flashpatch, too_flashy):
        for run in result["runs"]:
            runtime = run["fair_runtime"]
            assert runtime["measurement_boundary"] == external_league.FAIR_RUNTIME_BOUNDARY
            assert runtime["attempt_ordinal"] == 1
            assert runtime["retry_count"] == 0
            assert runtime["retry_policy"] == "NO_RETRY"
            assert runtime["timed_out"] is False
            assert runtime["wall_time_ns"] > 0
            assert runtime["normalized_terminal_observation"]["sha256"]
            assert runtime["observed_environment"]["parent_precondition"]["concurrency"]["lock_acquired"] is True
            assert runtime["observed_environment"]["child_probe"]["cpu_affinity"] == result["fair_runtime_protocol"]["cpu"]["affinity"]
    flashpatch_child = json.loads(Path(str(flashpatch["runs"][0]["receipt"])).read_text())
    conversion = json.loads((tmp_path / "lane" / "conversion-receipt.json").read_text())
    assert flashpatch_child["worker_decode"]["receipt"]["decoded_rgb_sha256"] == conversion["renderer_rgb"]["raw_sha256"]
    assert Path(flashpatch_child["command"][0]).name == "taskset"

    verified = verify_fair_runtime_receipts([flashpatch["receipt"], too_flashy["receipt"]])
    assert verified["failures"] == []
    assert verified["status"] == "NOT_VERIFIED"
    assert verified["receipts_verified"] is True
    assert verified["fair_runtime_verified"] is False
    assert verified["runtime_comparison_ready"] is False
    assert "balanced_interleaved_schedule_missing" in verified["runtime_comparison_blockers"]
    assert "unequal_effective_environment" not in verified["runtime_comparison_blockers"]
    assert len(verified["effective_environment_sha256"]) == 2
    assert len({value for values in verified["effective_environment_sha256"].values() for value in values}) == 1
    assert verified["scoreable"] is False
    assert not {"scores", "ranking", "winner"}.intersection(verified)


def test_fair_runtime_schedule_is_deterministic_balanced_and_pre_frozen(tmp_path: Path) -> None:
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    first = freeze_fair_runtime_schedule(
        ["TooFlashy", "FlashPatch", KAYA_DIRECT_PARTICIPANT_ID],
        protocol,
        "a" * 64,
        seed=20260802,
    )
    second = freeze_fair_runtime_schedule(
        [KAYA_DIRECT_PARTICIPANT_ID, "FlashPatch", "TooFlashy"],
        protocol,
        "a" * 64,
        seed=20260802,
    )

    assert first == second
    assert first["policy"]["freeze_state"] == "PRE_FROZEN"
    assert [entry["slot"] for entry in first["slots"]] == list(range(1, 10))
    for comparator in first["participants"]:
        rows = [entry for entry in first["slots"] if entry["comparator"] == comparator]
        assert [entry["repeat_ordinal"] for entry in rows] == [1, 2, 3]
        assert len({entry["position"] for entry in rows}) == 3


def test_release_oracle_cannot_enter_direct_scheduled_repeat_or_runtime_bundle(
    tmp_path: Path,
) -> None:
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    release_child = tmp_path / "release-child.json"
    release_child.write_text(
        json.dumps(
            {
                "schema": "flashpatch-ea-iris-release-run-v1",
                "comparator": {"name": EA_IRIS_RELEASE_ORACLE_ID},
                "status": "PROCESS_VALID",
            }
        )
        + "\n"
    )

    with pytest.raises(ExternalLeagueError, match="unsupported"):
        write_scheduled_runtime_repeat_receipt(
            EA_IRIS_RELEASE_ORACLE_ID,
            [release_child, release_child, release_child],
            protocol,
            tmp_path / "release-direct-repeat.json",
        )

    release_repeat = tmp_path / "release-conformance-repeat.json"
    release_repeat.write_text(
        json.dumps(
            {
                "schema": "flashpatch-ea-iris-release-conformance-repeats-v1",
                "comparator": EA_IRIS_RELEASE_ORACLE_ID,
                "repeats_required": 3,
                "status": "PROCESS_REPRODUCIBLE",
            }
        )
        + "\n"
    )
    result = verify_fair_runtime_receipts([release_repeat, release_repeat])
    assert result["status"] == "INCONCLUSIVE"
    assert all(
        failure.startswith("repeat_receipt_schema_invalid:")
        for failure in result["failures"]
        if failure.startswith("repeat_receipt_schema_invalid:")
    )
    assert any(
        failure.startswith("repeat_receipt_schema_invalid:")
        for failure in result["failures"]
    )


def test_interleaved_schedule_and_child_effective_environment_remain_unverified_without_independent_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy, schedule_path = _run_scheduled_fair_runtime_pair(tmp_path, monkeypatch)

    verified = verify_fair_runtime_receipts(
        [flashpatch["receipt"], too_flashy["receipt"]],
        schedule_receipt=schedule_path,
    )

    assert verified["status"] == "NOT_VERIFIED"
    assert verified["receipts_verified"] is True
    assert verified["schedule_environment_verified"] is True
    assert verified["fair_runtime_verified"] is False
    assert verified["runtime_comparison_ready"] is False
    assert verified["failures"] == []
    assert verified["runtime_comparison_blockers"] == ["independent_execution_witness_missing"]
    assert len({value for values in verified["effective_environment_sha256"].values() for value in values}) == 1
    assert verified["schedule"]["artifact_sha256"] == hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    assert verified["scoreable"] is False
    assert not {"scores", "ranking", "winner"}.intersection(verified)
    for repeat in (flashpatch, too_flashy):
        for row in repeat["runs"]:
            child = json.loads(Path(str(row["receipt"])).read_text())
            probe = child["runtime_probe"]["observation"]
            visible_nodes = probe["effective_environment"]["gpu"]["visible_device_nodes"]
            assert not any(
                path.startswith(("/dev/dri/", "/dev/accel/", "/dev/vfio/"))
                or Path(path).name.startswith(("nvidia", "mali", "xpu", "cuda", "nvhost"))
                or path in {"/dev/kfd", "/dev/dxg"}
                for path in visible_nodes
            )
            assert probe["effective_environment"]["cache"]["input_sha256"] == row["fair_runtime"]["input_identity_sha256"]
            assert probe["schedule_observation"]["artifact_sha256"] == verified["schedule"]["artifact_sha256"]


def test_valid_external_host_lane_closes_only_the_execution_witness_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy, schedule_path = _run_scheduled_fair_runtime_pair(tmp_path, monkeypatch)
    witnessed = {
        "schema": EXTERNAL_HOST_VERIFICATION_SCHEMA_V2,
        "status": "VERIFIED",
        "witness_verified": True,
        "independent_host_identity_verified": True,
        "host": {"identity_sha256": "a" * 64},
        "request": str(tmp_path / "external-request.json"),
        "receipt": str(tmp_path / "external-receipt.json"),
        "failures": [],
        "claim_status": "NOT_SCOREABLE",
        "scoreable": False,
        "comparison_eligible": False,
        "claim_blockers": [
            "independent_gold_receipt_missing",
            "fair_population_receipt_conditions_unproven",
            "receipt_bound_score_verifier_missing",
        ],
    }
    monkeypatch.setattr(external_league, "verify_external_host_witness", lambda *_args, **_kwargs: witnessed)

    verified = verify_fair_runtime_receipts(
        [flashpatch["receipt"], too_flashy["receipt"]],
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": tmp_path / "external-request.json",
            "receipt": tmp_path / "external-receipt.json",
        },
    )

    assert verified["failures"] == []
    assert verified["independent_execution_witness_verified"] is True
    assert verified["fair_runtime_verified"] is False
    assert verified["runtime_comparison_ready"] is False
    assert verified["external_host_witness"] == witnessed
    assert "independent_execution_witness_missing" not in verified["runtime_comparison_blockers"]
    assert verified["runtime_comparison_blockers"] == [
        "fair_population_receipt_conditions_unproven"
    ]
    assert verified["claim_status"] == "NOT_SCOREABLE"
    assert verified["scoreable"] is False
    assert verified["comparison_eligible"] is False
    assert not {"scores", "ranking", "winner"}.intersection(verified)

    legacy_witnessed = {
        **witnessed,
        "schema": EXTERNAL_HOST_VERIFICATION_SCHEMA_V1,
    }
    monkeypatch.setattr(
        external_league,
        "verify_external_host_witness",
        lambda *_args, **_kwargs: legacy_witnessed,
    )

    legacy_verified = verify_fair_runtime_receipts(
        [flashpatch["receipt"], too_flashy["receipt"]],
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": tmp_path / "external-request.json",
            "receipt": tmp_path / "external-receipt.json",
        },
    )

    assert legacy_verified["receipts_verified"] is True
    assert legacy_verified["independent_execution_witness_verified"] is False
    assert legacy_verified["runtime_comparison_blockers"] == [
        "external_host_witness_v2_required"
    ]
    assert legacy_verified["claim_status"] == "NOT_SCOREABLE"
    assert legacy_verified["scoreable"] is False
    assert legacy_verified["comparison_eligible"] is False


def test_full_population_external_slot_results_join_one_to_one_to_reparsed_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeats, schedule_path, witnessed, reparsed = _synthetic_external_population_join(
        tmp_path, monkeypatch
    )

    verified = verify_fair_runtime_receipts(
        repeats,
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": witnessed["request"],
            "receipt": witnessed["receipt"],
        },
    )

    assert verified["failures"] == []
    assert verified["status"] == "NOT_VERIFIED"
    assert verified["receipts_verified"] is True
    assert verified["independent_execution_witness_verified"] is True
    assert verified["fair_runtime_verified"] is True
    assert verified["runtime_comparison_ready"] is False
    assert verified["runtime_comparison_blockers"] == []
    assert len(verified["external_slot_child_joins"]) == 9
    assert set(reparsed) == {
        (comparator, ordinal)
        for comparator in DIRECT_DETECTOR_POPULATION
        for ordinal in (1, 2, 3)
    }
    assert verified["claim_status"] == "NOT_SCOREABLE"
    assert verified["scoreable"] is False
    assert verified["comparison_eligible"] is False
    assert not {"scores", "ranking", "winner"}.intersection(verified)


def test_external_slot_result_semantic_tamper_fails_closed_after_witness_hash_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeats, schedule_path, witnessed, _ = _synthetic_external_population_join(
        tmp_path, monkeypatch
    )
    slot = witnessed["verified_slots"][0]
    result_path = Path(slot["result"]["path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["parser_observation"]["prediction"] = "HAZARDOUS"
    _write_json(result_path, payload)
    slot["result"]["sha256"] = external_league._sha256_file(result_path)
    slot["result"]["size"] = result_path.stat().st_size

    verified = verify_fair_runtime_receipts(
        repeats,
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": witnessed["request"],
            "receipt": witnessed["receipt"],
        },
    )

    assert verified["fair_runtime_verified"] is False
    assert verified["status"] == "INCONCLUSIVE"
    assert any(
        failure.startswith("external_slot_child_join_mismatch:")
        for failure in verified["failures"]
    )
    assert "external_slot_child_join_invalid" in verified["runtime_comparison_blockers"]


def test_external_slot_join_keeps_two_participant_population_not_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeats, schedule_path, witnessed, _ = _synthetic_external_population_join(
        tmp_path,
        monkeypatch,
        participants=("FlashPatch", "TooFlashy"),
    )

    verified = verify_fair_runtime_receipts(
        repeats,
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": witnessed["request"],
            "receipt": witnessed["receipt"],
        },
    )

    assert verified["failures"] == []
    assert verified["status"] == "NOT_VERIFIED"
    assert verified["independent_execution_witness_verified"] is True
    assert verified["fair_runtime_verified"] is False
    assert verified["external_slot_child_joins"] == []
    assert verified["runtime_comparison_blockers"] == [
        "fair_population_receipt_conditions_unproven"
    ]


def test_external_slot_child_receipt_swap_fails_one_to_one_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeats, schedule_path, witnessed, _ = _synthetic_external_population_join(
        tmp_path, monkeypatch
    )
    first, second = witnessed["verified_slots"][:2]
    first_path = Path(first["result"]["path"])
    second_path = Path(second["result"]["path"])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second_path.read_text(encoding="utf-8"))
    first_payload["child_receipt_sha256"], second_payload["child_receipt_sha256"] = (
        second_payload["child_receipt_sha256"],
        first_payload["child_receipt_sha256"],
    )
    for slot, path, payload in (
        (first, first_path, first_payload),
        (second, second_path, second_payload),
    ):
        _write_json(path, payload)
        slot["result"]["sha256"] = external_league._sha256_file(path)
        slot["result"]["size"] = path.stat().st_size

    verified = verify_fair_runtime_receipts(
        repeats,
        schedule_receipt=schedule_path,
        external_host_witness={
            "request": witnessed["request"],
            "receipt": witnessed["receipt"],
        },
    )

    assert verified["fair_runtime_verified"] is False
    assert verified["status"] == "INCONCLUSIVE"
    assert sum(
        failure.startswith("external_slot_child_join_mismatch:")
        for failure in verified["failures"]
    ) == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unbalanced"])
def test_fair_runtime_schedule_rejects_missing_duplicate_and_unbalanced_slots(
    tmp_path: Path,
    mutation: str,
) -> None:
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    payload = freeze_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy", KAYA_DIRECT_PARTICIPANT_ID],
        protocol,
        "a" * 64,
        seed=7,
    )
    if mutation == "missing":
        payload["slots"].pop()
    elif mutation == "duplicate":
        payload["slots"][1]["slot"] = payload["slots"][0]["slot"]
    else:
        payload["slots"][0]["comparator"] = payload["slots"][1]["comparator"]

    with pytest.raises(ExternalLeagueError, match="not deterministic and balanced"):
        external_league._validate_frozen_runtime_schedule(payload)


def test_fair_runtime_schedule_rejects_mixed_type_participants_as_contract_error(tmp_path: Path) -> None:
    payload = freeze_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy"],
        _fair_runtime_protocol(tmp_path / "runtime.lock"),
        "a" * 64,
        seed=7,
    )
    payload["participants"] = ["FlashPatch", 1]

    with pytest.raises(ExternalLeagueError, match="participant population"):
        external_league._validate_frozen_runtime_schedule(payload)


def test_post_hoc_schedule_cannot_upgrade_unscheduled_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    video = tmp_path / "lane" / "canonical.ffv1.mkv"
    schedule_path = tmp_path / "post-hoc-schedule.json"
    _ = write_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy"],
        flashpatch["fair_runtime_protocol"],
        hashlib.sha256(video.read_bytes()).hexdigest(),
        schedule_path,
        seed=9,
    )

    result = verify_fair_runtime_receipts(
        [flashpatch["receipt"], too_flashy["receipt"]],
        schedule_receipt=schedule_path,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["fair_runtime_verified"] is False
    assert any(failure.startswith("schedule_binding_missing:") for failure in result["failures"])
    assert "schedule_slot_set_incomplete" in result["failures"]


def test_coherent_post_hoc_binding_still_fails_schedule_creation_time_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    video = tmp_path / "lane" / "canonical.ffv1.mkv"
    schedule_path = tmp_path / "late-schedule.json"
    schedule = write_fair_runtime_schedule(
        ["FlashPatch", "TooFlashy"],
        flashpatch["fair_runtime_protocol"],
        hashlib.sha256(video.read_bytes()).hexdigest(),
        schedule_path,
        seed=11,
    )
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    row = repeat["runs"][0]
    assignment = next(
        entry
        for entry in schedule["slots"]
        if entry["comparator"] == "TooFlashy" and entry["repeat_ordinal"] == 1
    )
    schedule_stat = schedule_path.stat()
    binding = {
        "path": str(schedule_path.resolve()),
        "artifact_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "schedule_sha256": hashlib.sha256(
            json.dumps(
                {key: value for key, value in schedule.items() if key not in {"receipt", "artifact_sha256", "schedule_sha256"}},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "stat": {
            "device": schedule_stat.st_dev,
            "inode": schedule_stat.st_ino,
            "size": schedule_stat.st_size,
            "mtime_ns": schedule_stat.st_mtime_ns,
            "ctime_ns": schedule_stat.st_ctime_ns,
        },
        **assignment,
    }
    child_path = Path(str(row["receipt"]))
    child = json.loads(child_path.read_text())
    probe_path = child_path.parent / child["runtime_probe"]["path"]
    probe = json.loads(probe_path.read_text())
    probe["schedule_observation"] = {
        "path": binding["path"],
        "artifact_sha256": binding["artifact_sha256"],
        "stat": binding["stat"],
        "schedule_sha256": binding["schedule_sha256"],
        "slot": str(binding["slot"]),
        "round": str(binding["round"]),
        "position": str(binding["position"]),
        "comparator": binding["comparator"],
        "repeat_ordinal": str(binding["repeat_ordinal"]),
    }
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    child["runtime_probe"]["observation"] = probe
    child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    child["fair_runtime"]["schedule_binding"] = binding
    child["fair_runtime"]["observed_environment"]["child_probe"] = probe
    script_index = child["command"].index(external_league._RUNTIME_PROBE_SCRIPT)
    child["command"][script_index + 3] = str(schedule_path.resolve())
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    row["fair_runtime"] = child["fair_runtime"]
    row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_path = tmp_path / "late-bound-repeat.json"
    forged_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts(
        [flashpatch["receipt"], forged_path],
        schedule_receipt=schedule_path,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert "observed_environment_invalid:TooFlashy:1" in result["failures"]


def test_undeclared_effective_environment_difference_is_rejected_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    row = repeat["runs"][0]
    child_path = Path(str(row["receipt"]))
    child = json.loads(child_path.read_text())
    probe_path = child_path.parent / child["runtime_probe"]["path"]
    probe = json.loads(probe_path.read_text())
    probe["effective_environment"]["process_environment"]["LD_PRELOAD"] = "/tmp/not-declared.so"
    probe["effective_environment_sha256"] = hashlib.sha256(
        json.dumps(probe["effective_environment"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    child["runtime_probe"]["observation"] = probe
    child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    child["fair_runtime"]["observed_environment"]["child_probe"] = probe
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    row["fair_runtime"] = child["fair_runtime"]
    row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_path = tmp_path / "undeclared-env-repeat.json"
    forged_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "observed_environment_invalid:TooFlashy:1" in result["failures"]


def test_pwd_exclusion_is_accepted_only_when_bound_to_declared_launcher_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    flashpatch_child = json.loads(Path(str(flashpatch["runs"][0]["receipt"])).read_text())
    too_flashy_child = json.loads(Path(str(too_flashy["runs"][0]["receipt"])).read_text())
    assert flashpatch_child["runtime_probe"]["observation"]["launcher_identity_environment"]["PWD"] != too_flashy_child["runtime_probe"]["observation"]["launcher_identity_environment"]["PWD"]
    baseline = verify_fair_runtime_receipts([flashpatch["receipt"], too_flashy["receipt"]])
    assert baseline["failures"] == []

    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    row = repeat["runs"][0]
    child_path = Path(str(row["receipt"]))
    child = json.loads(child_path.read_text())
    probe_path = child_path.parent / child["runtime_probe"]["path"]
    probe = json.loads(probe_path.read_text())
    probe["launcher_identity_environment"]["PWD"] = str(tmp_path)
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    child["runtime_probe"]["observation"] = probe
    child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    child["fair_runtime"]["observed_environment"]["child_probe"] = probe
    child["comparator"]["working_directory"] = str(tmp_path)
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    row["fair_runtime"] = child["fair_runtime"]
    row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_path = tmp_path / "wrong-pwd-repeat.json"
    forged_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "child_runtime_probe_or_command_unbound:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_returns_inconclusive_for_malformed_receipt_collections() -> None:
    none_result = verify_fair_runtime_receipts(None)  # type: ignore[arg-type]
    mixed_result = verify_fair_runtime_receipts([None, "missing.json"])  # type: ignore[list-item]

    assert none_result["status"] == "INCONCLUSIVE"
    assert "runtime_bundle_requires_at_least_two_comparators" in none_result["failures"]
    assert mixed_result["status"] == "INCONCLUSIVE"
    assert "repeat_receipt_reference_invalid" in mixed_result["failures"]


@pytest.mark.parametrize("schedule_reference", [{}, 1, []])
def test_fair_runtime_verifier_returns_inconclusive_for_malformed_schedule_reference(
    schedule_reference: object,
) -> None:
    result = verify_fair_runtime_receipts([], schedule_receipt=schedule_reference)  # type: ignore[arg-type]

    assert result["status"] == "INCONCLUSIVE"
    assert "fair_runtime_schedule_invalid" in result["failures"]


def test_fair_runtime_protocol_rejects_unhashable_affinity_as_contract_error(tmp_path: Path) -> None:
    payload = freeze_fair_runtime_protocol(_fair_runtime_protocol(tmp_path / "runtime.lock"))
    payload["cpu"]["affinity"] = [[]]

    with pytest.raises(ExternalLeagueError, match="CPU affinity"):
        external_league._validate_frozen_runtime_protocol(payload)


def test_fair_runtime_protocol_rejects_none_affinity_as_contract_error(tmp_path: Path) -> None:
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    malformed = FairRuntimeProtocol(**{**protocol.__dict__, "cpu_affinity": None})  # type: ignore[arg-type]

    with pytest.raises(ExternalLeagueError, match="CPU affinity"):
        freeze_fair_runtime_protocol(malformed)


def test_fair_runtime_verifier_returns_inconclusive_for_mixed_repeat_ordinal_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    repeat["runs"][1]["repeat"] = "2"
    malformed_path = tmp_path / "mixed-repeat-ordinal.json"
    malformed_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], malformed_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "scheduled_repeat_ordinals_invalid:TooFlashy" in result["failures"]


def test_fair_runtime_verifier_returns_inconclusive_for_malformed_child_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    row = repeat["runs"][0]
    child_path = Path(str(row["receipt"]))
    child = json.loads(child_path.read_text())
    probe_path = child_path.parent / child["runtime_probe"]["path"]
    probe = json.loads(probe_path.read_text())
    probe["child_timing"]["tool_started_monotonic_ns"] = "not-an-integer"
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    child["runtime_probe"]["observation"] = probe
    child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    child["fair_runtime"]["observed_environment"]["child_probe"] = probe
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    row["fair_runtime"] = child["fair_runtime"]
    row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    malformed_path = tmp_path / "malformed-child-timing-repeat.json"
    malformed_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], malformed_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "observed_environment_invalid:TooFlashy:1" in result["failures"]
    assert "child_parent_timing_mismatch:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_returns_inconclusive_for_unhashable_parent_affinity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    row = repeat["runs"][0]
    child_path = Path(str(row["receipt"]))
    child = json.loads(child_path.read_text())
    child["fair_runtime"]["observed_environment"]["parent_precondition"]["cpu"]["available_affinity"] = [[]]
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    row["fair_runtime"] = child["fair_runtime"]
    row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    malformed_path = tmp_path / "unhashable-parent-affinity-repeat.json"
    malformed_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], malformed_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "observed_environment_invalid:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_observed_gpu_device_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    forged = json.loads(Path(str(too_flashy["receipt"])).read_text())
    forged["runs"][0]["fair_runtime"]["observed_environment"]["child_probe"]["effective_environment"]["gpu"]["visible_device_nodes"].append("/dev/dri/card0")
    path = tmp_path / "gpu-allowed-repeat.json"
    path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], path])

    assert result["status"] == "INCONCLUSIVE"
    assert "observed_environment_invalid:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_schedule_order_not_actual_execution_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy, schedule_path = _run_scheduled_fair_runtime_pair(tmp_path, monkeypatch)
    repeat_paths = [Path(str(flashpatch["receipt"])), Path(str(too_flashy["receipt"]))]
    payloads = [json.loads(path.read_text()) for path in repeat_paths]
    rows = [row for payload in payloads for row in payload["runs"]]
    slot_one = next(row for row in rows if row["fair_runtime"]["schedule_binding"]["slot"] == 1)
    slot_two = next(row for row in rows if row["fair_runtime"]["schedule_binding"]["slot"] == 2)
    slot_one_child = json.loads(Path(str(slot_one["receipt"])).read_text())
    previous_tool_finish = slot_one_child["runtime_probe"]["observation"]["child_timing"]["tool_finished_monotonic_ns"]
    child_path = Path(str(slot_two["receipt"]))
    child = json.loads(child_path.read_text())
    probe_path = child_path.parent / child["runtime_probe"]["path"]
    probe = json.loads(probe_path.read_text())
    child_timing = probe["child_timing"]
    tool_duration = child_timing["tool_finished_monotonic_ns"] - child_timing["tool_started_monotonic_ns"]
    new_tool_start = previous_tool_finish - 1
    child_timing["probe_started_monotonic_ns"] = new_tool_start - 2
    child_timing["tool_started_monotonic_ns"] = new_tool_start
    child_timing["tool_finished_monotonic_ns"] = new_tool_start + tool_duration
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
    child["runtime_probe"]["observation"] = probe
    child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    runtime = slot_two["fair_runtime"]
    runtime["observed_environment"]["child_probe"] = probe
    runtime["started_monotonic_ns"] = child_timing["probe_started_monotonic_ns"] - 1
    runtime["finished_monotonic_ns"] = child_timing["tool_finished_monotonic_ns"] + 1
    runtime["wall_time_ns"] = runtime["finished_monotonic_ns"] - runtime["started_monotonic_ns"]
    child["fair_runtime"] = runtime
    child["wall_time_ns"] = runtime["wall_time_ns"]
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    slot_two["fair_runtime"] = runtime
    slot_two["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    for path, payload in zip(repeat_paths, payloads):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts(repeat_paths, schedule_receipt=schedule_path)

    assert result["status"] == "INCONCLUSIVE"
    assert "schedule_execution_order_or_isolation_invalid" in result["failures"]


def test_coherent_local_timing_rewrite_cannot_reach_fair_runtime_verified_without_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy, schedule_path = _run_scheduled_fair_runtime_pair(tmp_path, monkeypatch)
    repeat_paths = [Path(str(flashpatch["receipt"])), Path(str(too_flashy["receipt"]))]
    payloads = [json.loads(path.read_text()) for path in repeat_paths]
    rows = sorted(
        (row for payload in payloads for row in payload["runs"]),
        key=lambda row: row["fair_runtime"]["schedule_binding"]["slot"],
    )
    rewritten_start = 1_000_000_000_000
    for index, row in enumerate(rows):
        child_path = Path(str(row["receipt"]))
        child = json.loads(child_path.read_text())
        probe_path = child_path.parent / child["runtime_probe"]["path"]
        probe = json.loads(probe_path.read_text())
        tool_started = rewritten_start + index * 1_000
        tool_finished = tool_started + 100
        probe["child_timing"]["probe_started_monotonic_ns"] = tool_started - 10
        probe["child_timing"]["tool_started_monotonic_ns"] = tool_started
        probe["child_timing"]["tool_finished_monotonic_ns"] = tool_finished
        probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
        child["runtime_probe"]["observation"] = probe
        child["runtime_probe"]["sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        runtime = child["fair_runtime"]
        runtime["observed_environment"]["child_probe"] = probe
        runtime["started_monotonic_ns"] = tool_started - 11
        runtime["finished_monotonic_ns"] = tool_finished + 1
        runtime["wall_time_ns"] = runtime["finished_monotonic_ns"] - runtime["started_monotonic_ns"]
        child["fair_runtime"] = runtime
        child["wall_time_ns"] = runtime["wall_time_ns"]
        child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
        row["fair_runtime"] = runtime
        row["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    for path, payload in zip(repeat_paths, payloads):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts(repeat_paths, schedule_receipt=schedule_path)

    assert result["failures"] == []
    assert result["status"] == "NOT_VERIFIED"
    assert result["schedule_environment_verified"] is True
    assert result["fair_runtime_verified"] is False
    assert result["runtime_comparison_ready"] is False
    assert result["runtime_comparison_blockers"] == ["independent_execution_witness_missing"]


def test_fair_runtime_verifier_rejects_budget_boundary_environment_repeat_retry_and_normalization_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    source = json.loads(Path(str(too_flashy["receipt"])).read_text())

    def protocol_sha256(payload: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    mutations: list[tuple[str, object, str]] = []

    unequal_budget = copy.deepcopy(source)
    unequal_budget["fair_runtime_protocol"]["budget"]["timeout_seconds"] = 121
    unequal_budget["fair_runtime_protocol_sha256"] = protocol_sha256(unequal_budget["fair_runtime_protocol"])
    mutations.append(("unequal-budget", unequal_budget, "unequal_runtime_protocol"))

    unequal_environment = copy.deepcopy(source)
    unequal_environment["fair_runtime_protocol"]["machine"]["id"] = "other-machine"
    unequal_environment["fair_runtime_protocol_sha256"] = protocol_sha256(unequal_environment["fair_runtime_protocol"])
    mutations.append(("unequal-environment", unequal_environment, "unequal_environment_policy"))

    boundary_drift = copy.deepcopy(source)
    boundary_drift["fair_runtime_protocol"]["measurement_boundary"]["start"] = "after_decode"
    mutations.append(("boundary-drift", boundary_drift, "runtime_protocol_invalid:TooFlashy"))

    missing_repeat = copy.deepcopy(source)
    missing_repeat["runs"].pop()
    mutations.append(("missing-repeat", missing_repeat, "scheduled_repeat_count_invalid:TooFlashy"))

    duplicate_repeat = copy.deepcopy(source)
    duplicate_repeat["runs"][2]["repeat"] = 2
    mutations.append(("duplicate-repeat", duplicate_repeat, "scheduled_repeat_ordinals_invalid:TooFlashy"))

    hidden_retry = copy.deepcopy(source)
    hidden_retry["runs"][1]["fair_runtime"]["retry_count"] = 1
    mutations.append(("hidden-retry", hidden_retry, "retry_detected:TooFlashy:2"))

    wall_time_over_budget = copy.deepcopy(source)
    wall_time_over_budget["runs"][0]["fair_runtime"]["wall_time_ns"] = 121 * 1_000_000_000
    mutations.append(("wall-time-over-budget", wall_time_over_budget, "wall_time_exceeds_budget:TooFlashy:1"))

    raw_only = copy.deepcopy(source)
    for run in raw_only["runs"]:
        run["fair_runtime"]["normalized_terminal_observation"] = None
    mutations.append(("raw-only", raw_only, "normalized_terminal_observation_missing:TooFlashy:1"))

    normalized_drift = copy.deepcopy(source)
    normalized_drift["runs"][2]["fair_runtime"]["normalized_terminal_observation"]["sha256"] = "0" * 64
    mutations.append(("normalized-drift", normalized_drift, "normalized_repeat_disagreement:TooFlashy"))

    fake_enforcement = copy.deepcopy(source)
    fake_enforcement["runs"][0]["fair_runtime"]["observed_environment"]["child_probe"]["cpu_affinity"] = []
    mutations.append(("fake-enforcement", fake_enforcement, "observed_environment_invalid:TooFlashy:1"))

    for name, payload, expected_failure in mutations:
        modified = tmp_path / f"{name}.json"
        modified.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        result = verify_fair_runtime_receipts([flashpatch["receipt"], modified])
        assert result["status"] == "INCONCLUSIVE"
        assert result["fair_runtime_verified"] is False
        assert expected_failure in result["failures"]
        assert not {"scores", "ranking", "winner"}.intersection(result)


def test_fair_runtime_verifier_reparses_raw_output_instead_of_trusting_self_consistent_child_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    run = repeat["runs"][0]
    child_path = Path(str(run["receipt"]))
    child = json.loads(child_path.read_text())
    forged_observation = dict(child["parsed_observation"])
    forged_observation["prediction"] = "HAZARDOUS"
    forged_sha256 = hashlib.sha256(
        json.dumps(forged_observation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    child["parsed_observation"] = forged_observation
    child["fair_runtime"]["normalized_terminal_observation"]["sha256"] = forged_sha256
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    run["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    run["normalized_observation_sha256"] = forged_sha256
    run["fair_runtime"] = child["fair_runtime"]
    forged_repeat = tmp_path / "forged-repeat.json"
    forged_repeat.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_repeat])

    assert result["status"] == "INCONCLUSIVE"
    assert "normalized_terminal_observation_unbound:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_parent_enforcement_claim_without_child_probe_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    run = repeat["runs"][0]
    child_path = Path(str(run["receipt"]))
    child = json.loads(child_path.read_text())
    child["command"] = child["command"][7:]
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    run["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_repeat = tmp_path / "command-forged-repeat.json"
    forged_repeat.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_repeat])

    assert result["status"] == "INCONCLUSIVE"
    assert "child_runtime_probe_or_command_unbound:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_changed_comparator_command_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    run = repeat["runs"][0]
    child_path = Path(str(run["receipt"]))
    child = json.loads(child_path.read_text())
    child["command"][-1] = "--text-instead-of-json"
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    run["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_repeat = tmp_path / "tail-forged-repeat.json"
    forged_repeat.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_repeat])

    assert result["status"] == "INCONCLUSIVE"
    assert "child_runtime_probe_or_command_unbound:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_inserted_and_reordered_command_envelope_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    source_repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    source_child_path = Path(str(source_repeat["runs"][0]["receipt"]))
    source_child = json.loads(source_child_path.read_text())
    mutations = {
        "inserted": [*source_child["command"][:3], "--unexpected", *source_child["command"][3:]],
        "reordered": [
            *source_child["command"][:7],
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            *source_child["command"][12:],
        ],
    }
    for name, command in mutations.items():
        repeat = copy.deepcopy(source_repeat)
        child = copy.deepcopy(source_child)
        child["command"] = command
        child_path = source_child_path.with_name(f"{name}-envelope-child.json")
        child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
        repeat["runs"][0]["receipt"] = str(child_path)
        repeat["runs"][0]["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
        repeat_path = tmp_path / f"{name}-envelope-repeat.json"
        repeat_path.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

        result = verify_fair_runtime_receipts([flashpatch["receipt"], repeat_path])

        assert result["status"] == "INCONCLUSIVE"
        assert "child_runtime_probe_or_command_unbound:TooFlashy:1" in result["failures"]


def test_fair_runtime_verifier_rejects_self_declared_substitute_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    repeat = json.loads(Path(str(too_flashy["receipt"])).read_text())
    run = repeat["runs"][0]
    child_path = Path(str(run["receipt"]))
    child = json.loads(child_path.read_text())
    substitute = tmp_path / "substitute-detector"
    substitute.write_text("#!/bin/sh\nprintf '{\"passes\": true}'\n")
    substitute.chmod(0o755)
    substitute_sha256 = hashlib.sha256(substitute.read_bytes()).hexdigest()
    child["comparator"]["binary"] = str(substitute)
    child["comparator"]["binary_sha256"] = substitute_sha256
    child["command"][7] = str(substitute)
    child_path.write_text(json.dumps(child, indent=2, sort_keys=True) + "\n")
    run["receipt_sha256"] = hashlib.sha256(child_path.read_bytes()).hexdigest()
    forged_repeat = tmp_path / "executable-forged-repeat.json"
    forged_repeat.write_text(json.dumps(repeat, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], forged_repeat])

    assert result["status"] == "INCONCLUSIVE"
    assert "child_runtime_probe_or_command_unbound:TooFlashy:1" in result["failures"]


def test_external_total_boundary_timeout_is_inconclusive_at_child_and_repeat_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec, artifact_root, census_path = _bind_tooflashy_spec_to_census(tmp_path, monkeypatch)
    protocol = _fair_runtime_protocol(tmp_path / "runtime.lock")
    clock = iter((0, 121 * 1_000_000_000))
    monkeypatch.setattr(external_league.time, "monotonic_ns", lambda: next(clock))

    result = execute_comparator(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "timeout-run",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
        runtime_protocol=protocol,
        scheduled_repeat_ordinal=1,
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["fair_runtime"]["timed_out"] is True
    assert "runtime_timeout" in result["scoreable_blockers"]


def test_failed_scheduled_run_remains_inconclusive_not_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flashpatch, too_flashy = _run_fair_runtime_pair(tmp_path, monkeypatch)
    failed = json.loads(Path(str(too_flashy["receipt"])).read_text())
    failed["runs"][0]["status"] = "INCONCLUSIVE"
    failed_path = tmp_path / "failed-repeat.json"
    failed_path.write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n")

    result = verify_fair_runtime_receipts([flashpatch["receipt"], failed_path])

    assert result["status"] == "INCONCLUSIVE"
    assert "scheduled_run_inconclusive:TooFlashy:1" in result["failures"]
    assert "score" not in result
    assert "scores" not in result
    assert result["scoreable"] is False


def test_detection_aggregate_is_diagnostic_until_it_reads_bound_receipts() -> None:
    result = aggregate_detection_cases([
        {"case_id": "a", "repository_id": "repo-a", "gold": "HAZARDOUS", "predictions": {"FlashPatch": ["HAZARDOUS"] * 3, KAYA_DIRECT_PARTICIPANT_ID: ["SAFE"] * 3, "TooFlashy": ["SAFE"] * 3}},
        {"case_id": "b", "repository_id": "repo-b", "gold": "SAFE", "predictions": {"FlashPatch": ["SAFE"] * 3, KAYA_DIRECT_PARTICIPANT_ID: ["HAZARDOUS"] * 3, "TooFlashy": ["SAFE"] * 3}},
    ], comparators=list(DIRECT_DETECTOR_POPULATION))
    assert result["comparators"]["FlashPatch"]["fn"] == 0
    assert result["comparators"][KAYA_DIRECT_PARTICIPANT_ID]["fn"] == 1
    assert result["status"] == "DIAGNOSTIC_UNBOUND"
    assert result["scoreable"] is False
    assert "receipt_bound_runs_missing" in result["scoreable_blockers"]


@pytest.mark.parametrize(
    "comparators",
    [
        ["FlashPatch", EA_IRIS_RELEASE_ORACLE_ID, "TooFlashy"],
        ["FlashPatch", KAYA_DIRECT_PARTICIPANT_ID, "FFmpeg vf_photosensitivity"],
        ["FlashPatch", KAYA_DIRECT_PARTICIPANT_ID, "EPI-LENS"],
        ["FlashPatch", KAYA_DIRECT_PARTICIPANT_ID, "TooFlashy_or_EPI_LENS"],
        ["FlashPatch", EA_IRIS_SOURCE_ADAPTER_ID, "TooFlashy"],
    ],
)
def test_detection_aggregate_rejects_mitigation_reserve_and_ambiguous_population(comparators: list[str]) -> None:
    with pytest.raises(ExternalLeagueError, match="fixed direct detector population"):
        aggregate_detection_cases(
            [{"case_id": "a", "repository_id": "repo-a", "gold": "SAFE", "predictions": {name: ["SAFE"] * 3 for name in comparators}}],
            comparators=comparators,
        )


@pytest.mark.parametrize(
    ("name", "mode", "message"),
    [
        ("FFmpeg vf_photosensitivity", "detection", "mitigation-only"),
        ("EPI-LENS", "detection", "UNSCORABLE"),
        ("TooFlashy_or_EPI_LENS", "detection", "ambiguous comparator identity"),
        ("EA IRIS", "detection", "ambiguous legacy EA IRIS identity"),
        (EA_IRIS_RELEASE_ORACLE_ID, "detection", "pinned release runner"),
        (EA_IRIS_SOURCE_ADAPTER_ID, "detection", "excluded semantic-mismatch baseline"),
        (KAYA_DIRECT_PARTICIPANT_ID, "detection", "unscored participant"),
        ("TooFlashy", "detection", "validated census receipt"),
    ],
)
def test_comparator_execution_rejects_forbidden_census_identity_or_lane(
    tmp_path: Path,
    name: str,
    mode: str,
    message: str,
) -> None:
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    spec = ComparatorSpec(
        name=name,
        repository_url="https://example.invalid/comparator",
        revision="f" * 40,
        license="test-only",
        mode=mode,
        raw_output_mode="stdout",
        command=(sys.executable, "-c", "print('result')", "{input}"),
    )
    with pytest.raises(ExternalLeagueError, match=message):
        execute_comparator(
            spec,
            tmp_path / "lane" / "canonical.ffv1.mkv",
            tmp_path / "lane" / "conversion-receipt.json",
            tmp_path / "forbidden-run",
        )


def test_comparator_execution_reopens_census_and_binds_external_detector_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "census-artifacts"
    manifest = _valid_comparator_census(artifact_root)
    manifest_path = tmp_path / "census.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    census_path = tmp_path / "census-receipt.json"
    _ = write_comparator_census_receipt(manifest_path, artifact_root, census_path)
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    too_flashy = _census_entry(manifest, "TooFlashy")
    frozen_command = json.loads((artifact_root / str(too_flashy["command_artifact"])).read_text())
    monkeypatch.setattr(
        external_league,
        "_checkout_provenance",
        lambda spec: {
            "status": "VERIFIED",
            "path": str(spec.source_checkout),
            "head": spec.revision,
            "clean": True,
            "reason": None,
        },
    )
    spec = ComparatorSpec(
        name="TooFlashy",
        repository_url=str(too_flashy["repository_url"]),
        revision=str(too_flashy["revision"]),
        license=str(too_flashy["license"]),
        mode="detection",
        raw_output_mode="stdout",
        command=tuple(frozen_command),
        source_checkout=Path(str(too_flashy["source_checkout"])),
        working_directory=Path(str(too_flashy["source_checkout"])),
        distribution=str(too_flashy["distribution"]),
        distribution_revision=str(too_flashy["distribution_revision"]),
        configuration_sha256=str(too_flashy["configuration_sha256"]),
        environment_sha256=str(too_flashy["environment_sha256"]),
    )

    result = execute_comparator(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "tooflashy-run",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
    )

    assert result["status"] == "PROCESS_VALID"
    assert result["census_receipt"]["sha256"] == hashlib.sha256(census_path.read_bytes()).hexdigest()
    repeated = execute_repeated_comparator(
        spec,
        tmp_path / "lane" / "canonical.ffv1.mkv",
        tmp_path / "lane" / "conversion-receipt.json",
        tmp_path / "tooflashy-repeats",
        census_receipt=census_path,
        census_artifact_root=artifact_root,
    )
    assert repeated["status"] == "PROCESS_REPRODUCIBLE"


def test_comparator_execution_rejects_census_receipt_modified_after_validation(tmp_path: Path) -> None:
    artifact_root = tmp_path / "census-artifacts"
    manifest = _valid_comparator_census(artifact_root)
    manifest_path = tmp_path / "census.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    census_path = tmp_path / "census-receipt.json"
    _ = write_comparator_census_receipt(manifest_path, artifact_root, census_path)
    stored = json.loads(census_path.read_text())
    stored["detector_population"] = [EA_IRIS_RELEASE_ORACLE_ID, "TooFlashy"]
    census_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    _ = materialize_cfr_ffv1(_frames(tmp_path / "frames.npz"), tmp_path / "lane", fps=60)
    too_flashy = _census_entry(manifest, "TooFlashy")
    frozen_command = json.loads((artifact_root / str(too_flashy["command_artifact"])).read_text())
    spec = ComparatorSpec(
        name="TooFlashy",
        repository_url=str(too_flashy["repository_url"]),
        revision=str(too_flashy["revision"]),
        license=str(too_flashy["license"]),
        mode="detection",
        raw_output_mode="stdout",
        command=tuple(frozen_command),
        source_checkout=Path(str(too_flashy["source_checkout"])),
        working_directory=Path(str(too_flashy["source_checkout"])),
        distribution=str(too_flashy["distribution"]),
        distribution_revision=str(too_flashy["distribution_revision"]),
        configuration_sha256=str(too_flashy["configuration_sha256"]),
        environment_sha256=str(too_flashy["environment_sha256"]),
    )

    with pytest.raises(ExternalLeagueError, match="altered after validation"):
        execute_comparator(
            spec,
            tmp_path / "lane" / "canonical.ffv1.mkv",
            tmp_path / "lane" / "conversion-receipt.json",
            tmp_path / "tooflashy-run",
            census_receipt=census_path,
            census_artifact_root=artifact_root,
        )
