from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .godot import GodotReplayRunner


@dataclass(frozen=True)
class CorpusFamily:
    family_id: str
    project_file: Path
    scene_file: Path
    source_file: Path
    trace_file: Path
    causal_node: str
    causal_parameter: str
    interaction_cause: bool
    action_space: tuple[dict[str, object], ...]
    horizon: int
    risk_threshold: float


class GodotCorpus:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        records = manifest.get("families")
        if not isinstance(records, list) or len(records) < 2:
            raise ValueError("corpus manifest must contain at least two scene families")
        families: dict[str, CorpusFamily] = {}
        for record in records:
            family_id = record["id"]
            if family_id in families:
                raise ValueError(f"duplicate corpus family: {family_id}")
            project = self.root / record["project"]
            scene = project / "main.tscn"
            source = self.root / record["source"]
            trace = self.root / record["trace"]
            for path in (project / "project.godot", scene, source, trace):
                if not path.is_file():
                    raise ValueError(f"missing corpus artifact: {path}")
            parameter = record["ground_truth"]["parameter"]
            if parameter not in source.read_text(encoding="utf-8"):
                raise ValueError(f"causal parameter {parameter!r} is absent from {source}")
            families[family_id] = CorpusFamily(
                family_id=family_id,
                project_file=project / "project.godot",
                scene_file=scene,
                source_file=source,
                trace_file=trace,
                causal_node=record["ground_truth"]["node"],
                causal_parameter=parameter,
                interaction_cause=bool(record["ground_truth"]["interaction"]),
                action_space=tuple(record["exploration"]["action_space"]),
                horizon=int(record["exploration"]["horizon"]),
                risk_threshold=float(record["exploration"]["risk_threshold"]),
            )
        self._families = families

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._families))

    def family(self, family_id: str) -> CorpusFamily:
        return self._families[family_id]

    def replay(
        self,
        family_id: str,
        output: Path,
        *,
        trace: Path | None = None,
    ) -> dict[str, object]:
        family = self.family(family_id)
        runner = GodotReplayRunner(family.project_file.parent)
        return runner.replay(trace or family.trace_file, output)
