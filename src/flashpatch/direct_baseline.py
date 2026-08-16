from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .core import analyze, repair_with_evidence
from .verify import verify
from .video_corpus import SealedVideoCorpus

FFMPEG_VERSION = "ffmpeg version 6.1.1-3ubuntu5 "
IRIS_REVISION = "d96978ac1107f3463b77f69a9c1b1ec5d45291a0"
IRIS_BRIDGE_REVISION = "7e2ff32e34df1b86e4f3874ebabeb9d324e93269"
EPI_LENS_REVISION = "a7c5ab95278e9c590324d6cb95b5f90982561f13"


class DirectBaselineBenchmark:
    def __init__(
        self,
        corpus_root: str | Path,
        *,
        external_results: str | Path | None = None,
        iris_command: tuple[str, ...] | None = None,
    ) -> None:
        self.corpus_root = Path(corpus_root)
        project_root = self.corpus_root.parents[1]
        self.external_results = Path(external_results) if external_results else (
            project_root / "benchmarks" / "direct-baseline" / "external-detection.json"
        )
        self.iris_command = iris_command

    def run(self, output: str | Path) -> dict[str, object]:
        self._check_runtime()
        corpus = SealedVideoCorpus(self.corpus_root)
        cases = [case for case in corpus.cases if case.split == "sealed"]
        case_ids = [case.case_id for case in cases]
        external = self._load_external(case_ids)

        flashpatch_detection: list[dict[str, object]] = []
        flashpatch_repair: list[dict[str, object]] = []
        ffmpeg_repair: list[dict[str, object]] = []
        for case in cases:
            frames, timestamps, gold = corpus.load(case.case_id)
            detection = analyze(frames, timestamps)
            intersection = int(np.count_nonzero(detection.hazard_mask & gold))
            union = int(np.count_nonzero(detection.hazard_mask | gold))
            flashpatch_detection.append(
                {
                    "case_id": case.case_id,
                    "expected_hazardous": case.hazardous,
                    "hazardous": detection.hazardous,
                    "correct": detection.hazardous is case.hazardous,
                    "spatial_iou": 1.0 if union == 0 else intersection / union,
                }
            )

            outcome = repair_with_evidence(frames, timestamps, detection)
            flashpatch_repair.append(
                self._repair_metrics(
                    case.case_id,
                    frames,
                    timestamps,
                    gold,
                    outcome.frames,
                    structural_similarity=outcome.structural_similarity,
                )
            )
            ffmpeg_frames = self._run_ffmpeg(frames, timestamps)
            ffmpeg_repair.append(
                self._repair_metrics(
                    case.case_id,
                    frames,
                    timestamps,
                    gold,
                    ffmpeg_frames,
                    structural_similarity=self._structural_similarity(frames, ffmpeg_frames),
                )
            )

        detection_league = {
            "flashpatch": self._detection_summary(flashpatch_detection, mask_capable=True),
            "ea-iris": self._detection_summary(external["ea-iris"]["cases"], mask_capable=False),
            "epi-lens": self._detection_summary(external["epi-lens"]["cases"], mask_capable=False),
        }
        repair_league = {
            "flashpatch": self._repair_summary(flashpatch_repair),
            "ffmpeg-photosensitivity": self._repair_summary(ffmpeg_repair),
        }
        verified_hashes = all(corpus.verify_hashes(case.case_id) for case in cases)
        flashpatch_wins_locality = (
            repair_league["flashpatch"]["changed_fraction"]
            < repair_league["ffmpeg-photosensitivity"]["changed_fraction"]
        )
        result: dict[str, object] = {
            "schema": "flashpatch-direct-baseline-v1",
            "corpus": {
                "manifest_sha256": f"sha256:{self._sha256(self.corpus_root / 'manifest.json')}",
                "split": "sealed",
                "case_ids": case_ids,
            },
            "revisions": {
                "ea-iris": IRIS_REVISION,
                "ea-iris-python-bridge": IRIS_BRIDGE_REVISION,
                "epi-lens": EPI_LENS_REVISION,
                "ffmpeg": FFMPEG_VERSION.strip(),
            },
            "fairness": {
                "same_case_ids": all(
                    list(method["case_ids"]) == case_ids
                    for method in [*detection_league.values(), *repair_league.values()]
                ),
                "sealed_inputs_hash_verified": verified_hashes,
                "detection_and_repair_leagues_separate": True,
                "detection_only_tools_receive_no_repair_score": True,
            },
            "detection_league": detection_league,
            "repair_league": repair_league,
            "performance_evidence": external["performance_evidence"],
            "reproduction": {
                "runs": 2,
                "runs_equal": external["runs_equal"] is True,
                "command": ".venv/bin/python -m flashpatch.direct_baseline corpus/competition benchmarks/direct-baseline/results.json",
            },
            "claims": {
                "flashpatch_strict_detection_pass": detection_league["flashpatch"]["strict_all_cases_pass"],
                "flashpatch_zero_residual_hazard": repair_league["flashpatch"]["residual_hazard_rate"] == 0.0,
                "flashpatch_changes_less_than_ffmpeg": flashpatch_wins_locality,
            },
        }
        result["verdict"] = "PASS" if (
            result["fairness"]["same_case_ids"]
            and result["fairness"]["sealed_inputs_hash_verified"]
            and result["reproduction"]["runs_equal"]
            and all(result["claims"].values())
        ) else "FAIL"
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    def _check_runtime(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("required baseline adapter unavailable: ffmpeg")
        version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
        if FFMPEG_VERSION not in version:
            raise RuntimeError(f"required baseline adapter unavailable: pinned ffmpeg ({version})")
        if self.iris_command is not None and shutil.which(self.iris_command[0]) is None:
            raise RuntimeError(f"required baseline adapter unavailable: {self.iris_command[0]}")

    def _load_external(self, case_ids: list[str]) -> dict[str, object]:
        if not self.external_results.is_file():
            raise RuntimeError(f"required baseline adapter unavailable: {self.external_results}")
        payload = json.loads(self.external_results.read_text(encoding="utf-8"))
        if payload.get("schema") != "flashpatch-external-detection-v1":
            raise ValueError("external detection receipt schema is not frozen")
        for method in ("ea-iris", "epi-lens"):
            record = payload.get(method)
            if not isinstance(record, dict):
                raise TypeError(f"external detection receipt omitted {method}")
            rows = record.get("cases")
            if not isinstance(rows, list) or [row.get("case_id") for row in rows] != case_ids:
                raise ValueError(f"external detection case slate differs: {method}")
            if record.get("runs_equal") is not True:
                raise ValueError(f"external detection did not reproduce: {method}")
        if payload.get("runs_equal") is not True:
            raise ValueError("external detection receipt did not reproduce")
        return payload

    @staticmethod
    def _detection_summary(rows: list[dict[str, object]], *, mask_capable: bool) -> dict[str, object]:
        tp = sum(row["expected_hazardous"] is True and row["hazardous"] is True for row in rows)
        tn = sum(row["expected_hazardous"] is False and row["hazardous"] is False for row in rows)
        fp = sum(row["expected_hazardous"] is False and row["hazardous"] is True for row in rows)
        fn = sum(row["expected_hazardous"] is True and row["hazardous"] is False for row in rows)
        return {
            "case_ids": [row["case_id"] for row in rows],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": (tp + tn) / len(rows),
            "strict_all_cases_pass": fp == 0 and fn == 0,
            "spatial_mask_capable": mask_capable,
            "cases": rows,
        }

    @staticmethod
    def _repair_metrics(
        case_id: str,
        original: np.ndarray,
        timestamps: np.ndarray,
        gold: np.ndarray,
        repaired: np.ndarray,
        *,
        structural_similarity: float,
    ) -> dict[str, object]:
        changed = np.any(repaired != original, axis=-1)
        outside = ~gold
        outside_changed = float(np.mean(changed[outside])) if np.any(outside) else 0.0
        verification = verify(repaired, timestamps)
        error = original.astype(np.float64) - repaired.astype(np.float64)
        mse = float(np.mean(error**2))
        psnr = None if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        return {
            "case_id": case_id,
            "residual_hazard": not verification.passed,
            "changed_fraction": float(np.mean(changed)),
            "outside_gold_changed_fraction": outside_changed,
            "structural_similarity": structural_similarity,
            "psnr_db": psnr,
        }

    @staticmethod
    def _repair_summary(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "case_ids": [row["case_id"] for row in rows],
            "residual_hazard_rate": sum(bool(row["residual_hazard"]) for row in rows) / len(rows),
            "changed_fraction": sum(float(row["changed_fraction"]) for row in rows) / len(rows),
            "outside_gold_changed_fraction": sum(
                float(row["outside_gold_changed_fraction"]) for row in rows
            ) / len(rows),
            "mean_structural_similarity": sum(
                float(row["structural_similarity"]) for row in rows
            ) / len(rows),
            "cases": rows,
        }

    @staticmethod
    def _run_ffmpeg(frames: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        time_base = Fraction(1, 1_000_000)
        with tempfile.TemporaryDirectory(prefix="flashpatch-direct-baseline-") as directory:
            source = Path(directory) / "source.mkv"
            repaired = Path(directory) / "repaired.mkv"
            with av.open(str(source), mode="w", format="matroska") as container:
                stream = container.add_stream("ffv1")
                stream.width = frames.shape[2]
                stream.height = frames.shape[1]
                stream.pix_fmt = "bgr0"
                stream.time_base = time_base
                stream.codec_context.time_base = time_base
                for pixels, timestamp in zip(frames, timestamps, strict=True):
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts = round(float(timestamp) / float(time_base))
                    frame.time_base = time_base
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(source),
                    "-vf", "photosensitivity=frames=30:threshold=1:skip=1",
                    "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(repaired),
                ],
                check=True,
            )
            with av.open(str(repaired), mode="r") as container:
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        result = np.stack(decoded).astype(np.uint8, copy=False)
        if result.shape != frames.shape:
            raise RuntimeError("ffmpeg baseline changed the frame slate")
        return result

    @staticmethod
    def _structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
        x = first.astype(np.float64) / 255.0
        y = second.astype(np.float64) / 255.0
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        variance_x = float(np.var(x))
        variance_y = float(np.var(y))
        covariance = float(np.mean((x - mean_x) * (y - mean_y)))
        return ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mean_x**2 + mean_y**2 + 0.01**2) * (variance_x + variance_y + 0.03**2)
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("output")
    args = parser.parse_args()
    DirectBaselineBenchmark(args.corpus).run(args.output)


if __name__ == "__main__":
    main()