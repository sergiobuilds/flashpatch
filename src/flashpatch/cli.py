from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import analyze, repair as repair_frames, repair_with_evidence
from .independent import verify_video_independently
from .media import read_video, write_video
from .renderer_artifact import RendererArtifactError
from .renderer_intake import inspect_renderer_capture, write_renderer_intake_receipt
from .safety_ci import compile_project, write_receipt
from .unity_preflight import UnityPreflightError, verify_unity_source_preflight
from .verify import verify as verify_frames


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_clip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fps = 30
    frame_count = 60
    height, width = 48, 64
    frames = np.full((frame_count, height, width, 3), 96, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[8:40, 8:56] = True
    for index in range(frame_count):
        value = 255 if (index // 3) % 2 else 0
        frames[index, mask] = value
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    return frames, timestamps, mask


def _run_demo(output: Path) -> dict[str, object]:
    frames, timestamps, expected_mask = _synthetic_clip()
    parameters = {
        "luminance_delta": 0.1,
        "max_flashes_per_second": 3,
        "area_threshold": 0.25,
        "repair_max_luminance_step": 0.08,
        "window_boundary": "closed",
    }
    detection = analyze(
        frames,
        timestamps,
        luminance_delta=parameters["luminance_delta"],
        max_flashes_per_second=parameters["max_flashes_per_second"],
        area_threshold=parameters["area_threshold"],
    )
    repaired = repair_frames(
        frames,
        timestamps,
        detection,
        max_luminance_step=parameters["repair_max_luminance_step"],
    )
    verifier_parameters = {
        "luminance_delta": parameters["luminance_delta"],
        "max_flashes_per_second": parameters["max_flashes_per_second"],
        "area_threshold": parameters["area_threshold"],
    }
    original_check = verify_frames(frames, timestamps, **verifier_parameters)
    repaired_check = verify_frames(repaired, timestamps, **verifier_parameters)
    unchanged_outside = bool(np.array_equal(frames[:, ~expected_mask], repaired[:, ~expected_mask]))
    unchanged_outside_time_mask = bool(
        np.array_equal(frames[~detection.hazard_mask], repaired[~detection.hazard_mask])
    )

    output.mkdir(parents=True, exist_ok=True)
    original_path = output / "original.npz"
    repaired_path = output / "repaired.npz"
    mask_path = output / "hazard-mask.npy"
    receipt_path = output / "receipt.json"
    np.savez_compressed(original_path, frames=frames, timestamps=timestamps)
    np.savez_compressed(repaired_path, frames=repaired, timestamps=timestamps)
    np.save(mask_path, detection.hazard_mask)

    verified = (
        detection.hazardous
        and not original_check.passed
        and repaired_check.passed
        and unchanged_outside
        and unchanged_outside_time_mask
    )
    receipt: dict[str, object] = {
        "profile": "wcag22-general-flash-bootstrap",
        "profile_version": 1,
        "parameters": parameters,
        "input_manifest": {
            "frames_sha256": _array_sha256(frames),
            "timestamps_sha256": _array_sha256(timestamps),
            "shape": list(frames.shape),
            "dtype": frames.dtype.str,
        },
        "output_manifest": {
            "frames_sha256": _array_sha256(repaired),
            "shape": list(repaired.shape),
            "dtype": repaired.dtype.str,
        },
        "artifact_sha256": {
            "original.npz": _file_sha256(original_path),
            "repaired.npz": _file_sha256(repaired_path),
            "hazard-mask.npy": _file_sha256(mask_path),
        },
        "input_hazardous": detection.hazardous,
        "detected_area_fraction": detection.max_affected_fraction,
        "detected_max_flash_count": detection.max_flash_count,
        "independent_original_passed": original_check.passed,
        "independent_repaired_passed": repaired_check.passed,
        "independent_repaired_max_transition_count": repaired_check.max_transition_count,
        "outside_spatial_mask_unchanged": unchanged_outside,
        "outside_spatiotemporal_mask_unchanged": unchanged_outside_time_mask,
        "status": "VERIFIED" if verified else "FAILED",
    }
    receipt["receipt_payload_sha256"] = _payload_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _scan_file(path: Path, mask_path: Path | None = None) -> dict[str, object]:
    video = read_video(path)
    result = analyze(video.frames, video.timestamps)
    if mask_path is not None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mask_path,
            aggregate=result.hazard_mask,
            timestamps=video.timestamps,
            **result.kind_masks,
        )
    return {
        "input": str(path),
        "hazardous": result.hazardous,
        "hazard_kinds": sorted(result.kind_masks),
        "mask": str(mask_path) if mask_path is not None else None,
        "max_affected_fraction": result.max_affected_fraction,
        "max_flash_count": result.max_flash_count,
        "windows": [
            {
                "start": window.start,
                "end": window.end,
                "affected_fraction": window.affected_fraction,
                "flash_count": window.flash_count,
                "kind": window.kind,
            }
            for window in result.windows
        ],
    }


def _verify_file(path: Path) -> dict[str, object]:
    video = read_video(path)
    result = verify_video_independently(path, video)
    return {
        "input": str(path),
        "passed": result.passed,
        "decoder": result.decoder,
        "decoder_agreement": result.decoder_agreement,
        "disagreement_reasons": list(result.disagreement_reasons),
        "max_affected_fraction": result.max_affected_fraction,
        "max_transition_count": result.max_transition_count,
    }


def _repair_file(source: Path, destination: Path, receipt_path: Path) -> dict[str, object]:
    video = read_video(source)
    detection = analyze(video.frames, video.timestamps)
    repair_outcome = repair_with_evidence(video.frames, video.timestamps, detection)
    repaired = repair_outcome.frames
    write_video(destination, repaired, video.timestamps, metadata=video.metadata)
    reopened = read_video(destination)
    verification = verify_video_independently(destination, reopened)
    timestamp_preserved = bool(
        np.allclose(reopened.timestamps, video.timestamps, rtol=0.0, atol=1e-6)
    )
    color_metadata_preserved = reopened.metadata == video.metadata
    status = (
        "VERIFIED"
        if repair_outcome.verified
        and verification.passed
        and timestamp_preserved
        and color_metadata_preserved
        else "FAILED"
    )
    receipt: dict[str, object] = {
        "profile": "wcag22-general-flash-file-v1",
        "input": str(source),
        "output": str(destination),
        "input_sha256": _file_sha256(source),
        "output_sha256": _file_sha256(destination),
        "input_hazardous": detection.hazardous,
        "repair_verified": repair_outcome.verified,
        "fallback_used": repair_outcome.fallback_used,
        "changed_fraction": repair_outcome.changed_fraction,
        "outside_hazard_unchanged": repair_outcome.outside_hazard_unchanged,
        "structural_similarity": repair_outcome.structural_similarity,
        "hazard_band_power_before": repair_outcome.hazard_band_power_before,
        "hazard_band_power_after": repair_outcome.hazard_band_power_after,
        "output_verified": verification.passed,
        "independent_decoder": verification.decoder,
        "decoder_agreement": verification.decoder_agreement,
        "decoder_disagreement_reasons": list(verification.disagreement_reasons),
        "timestamp_preserved": timestamp_preserved,
        "color_metadata_preserved": color_metadata_preserved,
        "frame_count": len(reopened.frames),
        "display_dimensions": [int(reopened.frames.shape[2]), int(reopened.frames.shape[1])],
        "status": status,
    }
    receipt["receipt_payload_sha256"] = _payload_sha256(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(prog="flashpatch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_command = subparsers.add_parser(
        "compile",
        help="run the fail-closed Godot Safety CI contract",
    )
    compile_command.add_argument("project", type=Path, help="Godot project directory")
    compile_command.add_argument(
        "trace_or_contract",
        type=Path,
        help="FlashPatch contract JSON, or a trace when project/flashpatch.json exists",
    )
    compile_command.add_argument("--workspace", type=Path, default=Path("artifacts/flashpatch-ci"))
    compile_command.add_argument("--receipt", type=Path, default=Path("artifacts/flashpatch-ci/receipt.json"))
    safety_demo = subparsers.add_parser(
        "safety-demo",
        help="show input, hazard localization, one patch, revalidation, and all terminal receipts",
    )
    safety_demo.add_argument("--output", type=Path, default=Path("artifacts/safety-demo"))
    safety_demo.add_argument("--json", action="store_true", dest="json_output")
    demo = subparsers.add_parser("demo", help="run the deterministic proof clip")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    scan = subparsers.add_parser("scan", help="scan an MP4 file for flash hazards")
    scan.add_argument("input", type=Path)
    scan.add_argument("--mask", type=Path)
    renderer_intake = subparsers.add_parser(
        "renderer-intake",
        help="analyze a strict frame_npz_v1 capture without inferring an engine or source",
    )
    renderer_intake.add_argument("input", type=Path)
    renderer_intake.add_argument("--receipt", type=Path, required=True)
    unity_preflight = subparsers.add_parser(
        "unity-preflight",
        help="bind pinned Unity source files without importing or running the editor",
    )
    unity_preflight.add_argument("manifest", type=Path)
    unity_preflight.add_argument("project", type=Path)
    l10_verify = subparsers.add_parser(
        "l10-verify",
        help="re-open a fail-closed L10 engine receipt or three-engine bundle",
    )
    l10_verify.add_argument("input", type=Path)
    l10_verify.add_argument("--bundle", action="store_true")
    l10_verify.add_argument("--trust-policy", type=Path, required=True)
    l10_verify.add_argument("--expected-trust-policy-sha256", required=True)
    l10_unity_run = subparsers.add_parser(
        "l10-unity-run",
        help="run the frozen eight-execution Unity L10 matrix on one idle GPU",
    )
    l10_unity_run.add_argument("--factual-template", type=Path, required=True)
    l10_unity_run.add_argument("--counterfactual-template", type=Path, required=True)
    l10_unity_run.add_argument("--factual-manifest", type=Path, required=True)
    l10_unity_run.add_argument("--counterfactual-manifest", type=Path, required=True)
    l10_unity_run.add_argument("--runtime-output", type=Path, required=True)
    l10_unity_run.add_argument("--entitlement", type=Path, required=True)
    l10_unity_run.add_argument("--vulkan-loader", type=Path, required=True)
    l10_unity_run.add_argument("--gpu-index", type=int, required=True)
    l10_unity_run.add_argument("--display", required=True)
    l10_unity_run.add_argument("--timeout-seconds", type=int, default=1800)
    repair = subparsers.add_parser("repair", help="repair an MP4 file and write a receipt")
    repair.add_argument("input", type=Path)
    repair.add_argument("output", type=Path)
    repair.add_argument("--receipt", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an MP4 file")
    verify.add_argument("input", type=Path)
    web = subparsers.add_parser("web", help="serve the judge-visible browser demo")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8080)
    web.add_argument("--workspace", type=Path, default=Path("artifacts/web"))
    args = parser.parse_args()

    if args.command == "compile":
        receipt = compile_project(args.project, args.trace_or_contract, workspace=args.workspace)
        write_receipt(receipt, args.receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        if receipt["verdict"] not in {"PASS", "SAFE"}:
            raise SystemExit(2)
    elif args.command == "safety-demo":
        from .product_demo import format_safety_demo, run_safety_demo

        report = run_safety_demo(args.output)
        print(
            json.dumps(report, indent=2, sort_keys=True)
            if args.json_output
            else format_safety_demo(report)
        )
    elif args.command == "demo":
        receipt = _run_demo(args.output)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        if receipt["status"] != "VERIFIED":
            raise SystemExit(1)
    elif args.command == "scan":
        print(json.dumps(_scan_file(args.input, args.mask), indent=2, sort_keys=True))
    elif args.command == "renderer-intake":
        try:
            receipt = inspect_renderer_capture(args.input)
        except (OSError, RendererArtifactError) as exc:
            print(json.dumps({"verdict": "INCONCLUSIVE", "reason": str(exc)}, sort_keys=True))
            raise SystemExit(2) from exc
        write_renderer_intake_receipt(receipt, args.receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.command == "unity-preflight":
        try:
            print(json.dumps(verify_unity_source_preflight(args.manifest, args.project), indent=2, sort_keys=True))
        except (OSError, UnityPreflightError) as exc:
            print(json.dumps({"verdict": "INCONCLUSIVE", "reason": str(exc)}, sort_keys=True))
            raise SystemExit(2) from exc
    elif args.command == "l10-verify":
        from .l10_receipt import L10ReceiptError, verify_l10_bundle, verify_l10_receipt

        try:
            result = (
                verify_l10_bundle(
                    args.input,
                    args.trust_policy,
                    args.expected_trust_policy_sha256,
                )
                if args.bundle
                else verify_l10_receipt(
                    args.input,
                    args.trust_policy,
                    args.expected_trust_policy_sha256,
                )
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            if result.get("verified") is not True:
                raise SystemExit(2)
        except (OSError, L10ReceiptError) as exc:
            print(json.dumps({"verified": False, "verdict": "INCONCLUSIVE", "reason": str(exc)}, sort_keys=True))
            raise SystemExit(2) from exc
    elif args.command == "l10-unity-run":
        from .l10_unity_runner import UnityL10RunError, run_unity_l10_matrix

        try:
            result = run_unity_l10_matrix(
                args.factual_template,
                args.counterfactual_template,
                args.factual_manifest,
                args.counterfactual_manifest,
                args.runtime_output,
                args.entitlement,
                args.vulkan_loader,
                gpu_index=args.gpu_index,
                display=args.display,
                timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (OSError, UnityL10RunError) as exc:
            print(json.dumps({"verdict": "INCONCLUSIVE", "reason": str(exc)}, sort_keys=True))
            raise SystemExit(2) from exc
    elif args.command == "repair":
        receipt = _repair_file(args.input, args.output, args.receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        if receipt["status"] != "VERIFIED":
            raise SystemExit(1)
    elif args.command == "verify":
        result = _verify_file(args.input)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "web":
        from .web import serve

        serve(args.host, args.port, workspace=args.workspace)


if __name__ == "__main__":
    main()
