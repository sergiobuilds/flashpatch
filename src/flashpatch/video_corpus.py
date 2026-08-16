from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_ALLOWED_SPLITS = {"development", "sealed"}
_ALLOWED_CATEGORIES = {"real", "synthetic", "boundary", "safe-negative", "transformed"}
_ALLOWED_REDISTRIBUTION = {"allowed", "generated-in-project"}


@dataclass(frozen=True)
class CorpusRights:
    spdx_id: str
    attribution: str
    source_url: str
    redistribution: str


@dataclass(frozen=True)
class VideoCorpusCase:
    case_id: str
    source_family: str
    split: str
    category: str
    hazardous: bool
    artifact: Path
    artifact_sha256: str
    rights: CorpusRights


class SealedVideoCorpus:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported video corpus manifest schema")
        records = payload.get("cases")
        if not isinstance(records, list) or not records:
            raise ValueError("video corpus manifest must contain cases")

        cases: list[VideoCorpusCase] = []
        case_ids: set[str] = set()
        family_splits: dict[str, set[str]] = {}
        for record in records:
            case_id = str(record["id"])
            if case_id in case_ids:
                raise ValueError(f"duplicate video corpus case: {case_id}")
            case_ids.add(case_id)
            split = str(record["split"])
            category = str(record["category"])
            if split not in _ALLOWED_SPLITS:
                raise ValueError(f"unsupported corpus split: {split}")
            if category not in _ALLOWED_CATEGORIES:
                raise ValueError(f"unsupported corpus category: {category}")
            family = str(record["source_family"])
            family_splits.setdefault(family, set()).add(split)
            rights_record = record["rights"]
            rights = CorpusRights(
                spdx_id=str(rights_record["spdx_id"]),
                attribution=str(rights_record["attribution"]),
                source_url=str(rights_record["source_url"]),
                redistribution=str(rights_record["redistribution"]),
            )
            if not rights.spdx_id or not rights.attribution or not rights.source_url:
                raise ValueError(f"incomplete rights record: {case_id}")
            if rights.redistribution not in _ALLOWED_REDISTRIBUTION:
                raise ValueError(f"redistribution is not allowed: {case_id}")
            artifact = self.root / str(record["artifact"])
            if not artifact.is_file():
                raise ValueError(f"missing corpus artifact: {artifact}")
            cases.append(
                VideoCorpusCase(
                    case_id=case_id,
                    source_family=family,
                    split=split,
                    category=category,
                    hazardous=bool(record["hazardous"]),
                    artifact=artifact,
                    artifact_sha256=str(record["sha256"]),
                    rights=rights,
                )
            )
        if any(len(splits) != 1 for splits in family_splits.values()):
            raise ValueError("a source family crosses corpus splits")
        if {case.category for case in cases} != _ALLOWED_CATEGORIES:
            raise ValueError("corpus must cover every required category")
        self.cases = tuple(cases)
        self._by_id = {case.case_id: case for case in cases}

    def split_ids(self, split: str) -> tuple[str, ...]:
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported corpus split: {split}")
        return tuple(case.case_id for case in self.cases if case.split == split)

    def verify_hashes(self, case_id: str) -> bool:
        case = self._by_id[case_id]
        digest = hashlib.sha256(case.artifact.read_bytes()).hexdigest()
        return digest == case.artifact_sha256

    def load(self, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        case = self._by_id[case_id]
        if not self.verify_hashes(case_id):
            raise ValueError(f"corpus artifact hash mismatch: {case_id}")
        with np.load(case.artifact, allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "timestamps", "gold_mask"}:
                raise ValueError(f"unexpected corpus artifact members: {case_id}")
            frames = archive["frames"]
            timestamps = archive["timestamps"]
            gold = archive["gold_mask"]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"invalid frame array: {case_id}")
        if timestamps.shape != (len(frames),) or not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"invalid timestamps: {case_id}")
        if gold.dtype != np.bool_ or gold.shape != frames.shape[:3]:
            raise ValueError(f"invalid gold mask: {case_id}")
        return frames, timestamps, gold
