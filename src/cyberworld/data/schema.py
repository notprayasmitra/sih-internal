"""Validated metadata for derived trajectory artifacts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrajectoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    extractor_version: str
    source_checksums: list[str]
    dataset_id: str
    scenario_id: str
    window_seconds: int = Field(gt=0)
    feature_names: list[str]
    label_mapping_version: str
    created_at_utc: datetime

    @property
    def trajectory_id(self) -> str:
        return f"{self.dataset_id}:{self.scenario_id}"
