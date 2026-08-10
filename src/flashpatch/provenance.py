from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import GodotCorpus


_EXPORTED_PARAMETER = re.compile(
    r"^\s*@export\s+var\s+(?P<name>[A-Za-z_]\w*)\s*(?::[^=]+)?=\s*(?P<value>.+?)\s*$"
)
_ROOT_NODE = re.compile(r'^\[node\s+name="(?P<name>[^"]+)"')


@dataclass(frozen=True)
class RenderContributor:
    node_path: str
    parameter: str
    source_file: Path
    source_line: int
    value: object
    hazard_frames: tuple[int, ...]


@dataclass(frozen=True)
class RenderProvenance:
    family_id: str
    hazard_frames: tuple[int, ...]
    contributors: tuple[RenderContributor, ...]


class RenderProvenanceCollector:
    def __init__(self, corpus: GodotCorpus, *, workspace: Path) -> None:
        self.corpus = corpus
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        family_id: str,
        action_trace: tuple[dict[str, object], ...],
    ) -> RenderProvenance:
        family = self.corpus.family(family_id)
        trace_path = self.workspace / f"{family_id}-trace.json"
        output_path = self.workspace / f"{family_id}-provenance-replay.json"
        trace_path.write_text(
            json.dumps({"fixed_fps": 60, "actions": action_trace}, sort_keys=True),
            encoding="utf-8",
        )
        replay = self.corpus.replay(family_id, output_path, trace=trace_path)
        observations = replay.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError(f"replay for {family_id} omitted observations")
        hazard_frames_list = []
        for action, risk in zip(action_trace, observations, strict=True):
            frame = action.get("frame")
            if not isinstance(frame, int):
                raise ValueError("action trace frame must be an integer")
            if float(risk) >= family.risk_threshold:
                hazard_frames_list.append(frame)
        hazard_frames = tuple(hazard_frames_list)
        if not hazard_frames:
            return RenderProvenance(family_id, (), ())

        node_path = self._root_node_path(family.scene_file)
        contributors = tuple(
            RenderContributor(
                node_path=node_path,
                parameter=name,
                source_file=family.source_file,
                source_line=line_number,
                value=value,
                hazard_frames=hazard_frames,
            )
            for name, line_number, value in self._exported_parameters(family.source_file)
        )
        return RenderProvenance(family_id, hazard_frames, contributors)

    @staticmethod
    def _root_node_path(scene_file: Path) -> str:
        for line in scene_file.read_text(encoding="utf-8").splitlines():
            match = _ROOT_NODE.match(line)
            if match:
                return f"/root/{match.group('name')}"
        raise ValueError(f"scene has no root node: {scene_file}")

    @staticmethod
    def _exported_parameters(source_file: Path) -> tuple[tuple[str, int, object], ...]:
        parameters = []
        for line_number, line in enumerate(
            source_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _EXPORTED_PARAMETER.match(line)
            if match:
                raw_value = match.group("value")
                try:
                    value = ast.literal_eval(raw_value)
                except (SyntaxError, ValueError):
                    value = raw_value
                parameters.append((match.group("name"), line_number, value))
        return tuple(parameters)