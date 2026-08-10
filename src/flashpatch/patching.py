from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .corpus import GodotCorpus
from .provenance import RenderContributor


@dataclass(frozen=True)
class SourcePatch:
    family_id: str
    parameter: str
    replacement: object
    changed_parameter_count: int
    project_dir: Path
    patched_scene: Path
    patch_file: Path


class SourcePatchSynthesizer:
    def __init__(self, corpus: GodotCorpus, *, allowed_parameters: set[str]) -> None:
        self.corpus = corpus
        self.allowed_parameters = frozenset(allowed_parameters)

    def synthesize(
        self,
        family_id: str,
        contributor: RenderContributor,
        *,
        replacement: object,
        output_dir: Path,
    ) -> SourcePatch:
        family = self.corpus.family(family_id)
        if contributor.parameter not in self.allowed_parameters:
            raise ValueError(f"source parameter {contributor.parameter!r} is not allowed")
        if contributor.node_path != family.causal_node:
            raise ValueError(f"contributor node {contributor.node_path!r} is not in {family_id}")

        output_dir = Path(output_dir).resolve()
        project_dir = output_dir / "project"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(family.project_file.parent, project_dir)
        patched_scene = project_dir / family.scene_file.name
        original = patched_scene.read_text(encoding="utf-8").splitlines(keepends=True)
        updated = self._set_node_parameter(original, contributor.parameter, replacement)
        patched_scene.write_text("".join(updated), encoding="utf-8")

        relative = f"{family_id}/{family.scene_file.name}"
        diff = "".join(
            difflib.unified_diff(
                original,
                updated,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
        patch_file = output_dir / f"{family_id}.diff"
        patch_file.write_text(diff, encoding="utf-8")
        return SourcePatch(
            family_id=family_id,
            parameter=contributor.parameter,
            replacement=replacement,
            changed_parameter_count=1,
            project_dir=project_dir,
            patched_scene=patched_scene,
            patch_file=patch_file,
        )

    @staticmethod
    def _set_node_parameter(
        scene_lines: list[str],
        parameter: str,
        replacement: object,
    ) -> list[str]:
        replacement_text = str(replacement).lower() if isinstance(replacement, bool) else str(replacement)
        assignment = f"{parameter} = {replacement_text}\n"
        prefix = f"{parameter} ="
        updated = list(scene_lines)
        existing = [index for index, line in enumerate(updated) if line.startswith(prefix)]
        if len(existing) > 1:
            raise ValueError(f"scene contains duplicate parameter {parameter!r}")
        if existing:
            updated[existing[0]] = assignment
            return updated
        script_lines = [index for index, line in enumerate(updated) if line.startswith("script = ")]
        if len(script_lines) != 1:
            raise ValueError("scene must bind exactly one root script")
        updated.insert(script_lines[0] + 1, assignment)
        return updated
