"""Declarative dataset discovery for repeatable preparation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    enabled: bool = True
    input_globs: list[str] = Field(min_length=1)
    scenario_strategy: str = "filename"
    note: str | None = None

    def scenario_id(self, path: Path) -> str:
        if self.scenario_strategy == "filename":
            return path.stem.replace(".", "_").replace(" ", "_").lower()
        if self.scenario_strategy == "parent_directory":
            return path.parent.name.replace(" ", "_").lower()
        raise ValueError(f"Unsupported scenario strategy: {self.scenario_strategy}")


class DatasetCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    datasets: list[DatasetSpec]


def load_catalog(path: Path) -> DatasetCatalog:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return DatasetCatalog.model_validate(payload)


def discover_files(spec: DatasetSpec, repo_root: Path) -> list[Path]:
    matches: set[Path] = set()
    for pattern in spec.input_globs:
        matches.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(matches)
