"""Typed experiment configuration with inheritance and environment expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    name: str
    seed: int = 42
    deterministic: bool = True


class PathsConfig(StrictModel):
    data_root: Path
    trajectory_dir: Path
    artifact_root: Path
    checkpoint_root: Path
    run_root: Path


class DataConfig(StrictModel):
    feature_dim: int = Field(gt=0)
    num_stages: int = Field(default=9, ge=2)
    history_windows: int = Field(gt=0)
    forecast_windows: int = Field(gt=0)
    window_seconds: int = Field(gt=0)
    label_coverage_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    synthetic_if_missing: bool = False
    synthetic_trajectories: int = Field(default=32, ge=2)
    synthetic_length: int = Field(default=120, ge=8)

    @model_validator(mode="after")
    def validate_sequence_length(self) -> DataConfig:
        required = self.history_windows + self.forecast_windows
        if self.synthetic_length < required:
            raise ValueError(f"synthetic_length must be at least {required}")
        return self


class ModelConfig(StrictModel):
    architecture: str = "latent_jepa"
    hidden_dim: int = Field(gt=0)
    projection_dim: int = Field(gt=0)
    latent_dim: int = Field(default=96, gt=1)
    num_layers: int = Field(gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    min_log_variance: float = -8.0
    max_log_variance: float = 5.0

    @model_validator(mode="after")
    def validate_variance_bounds(self) -> ModelConfig:
        if self.architecture not in {
            "probabilistic_lstm",
            "latent_jepa",
            "temporal_transformer",
            "temporal_cnn",
        }:
            raise ValueError(
                "architecture must be probabilistic_lstm, latent_jepa, "
                "temporal_transformer, or temporal_cnn"
            )
        if self.min_log_variance >= self.max_log_variance:
            raise ValueError("min_log_variance must be below max_log_variance")
        return self


class TrainingConfig(StrictModel):
    device: str = "auto"
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    accumulate_steps: int = Field(default=1, gt=0)
    teacher_forcing_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    mixed_precision: bool = True
    compile: bool = False
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    train_trajectory_ids: tuple[str, ...] | None = None
    validation_trajectory_ids: tuple[str, ...] | None = None
    stage_class_weighting: Literal["none", "inverse_frequency"] = "none"
    lambda_state: float = Field(default=1.0, ge=0)
    lambda_stage: float = Field(default=0.5, ge=0)
    lambda_malicious: float = Field(default=0.5, ge=0)
    lambda_compromise: float = Field(default=0.5, ge=0)
    lambda_jepa: float = Field(default=0.25, ge=0)
    vicreg_invariance: float = Field(default=25.0, ge=0)
    vicreg_variance: float = Field(default=25.0, ge=0)
    vicreg_covariance: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_explicit_trajectory_split(self) -> TrainingConfig:
        train_ids = self.train_trajectory_ids
        validation_ids = self.validation_trajectory_ids
        if (train_ids is None) != (validation_ids is None):
            raise ValueError(
                "train_trajectory_ids and validation_trajectory_ids must be configured together"
            )
        if train_ids is None or validation_ids is None:
            return self
        if not train_ids or not validation_ids:
            raise ValueError("explicit trajectory split lists must be non-empty")
        if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(
            validation_ids
        ):
            raise ValueError("explicit trajectory split lists must not contain duplicates")
        overlap = set(train_ids) & set(validation_ids)
        if overlap:
            raise ValueError(f"explicit trajectory split lists overlap: {sorted(overlap)}")
        return self


class LoggingConfig(StrictModel):
    log_every_steps: int = Field(default=25, gt=0)
    save_every_epochs: int = Field(default=1, gt=0)


class AppConfig(StrictModel):
    experiment: ExperimentConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.getenv(name, default)
        if resolved is None:
            raise ValueError(f"Environment variable {name} is required")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def _read_yaml(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config inheritance involving {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {resolved}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = (resolved.parent / str(parent)).resolve()
    return _deep_merge(_read_yaml(parent_path, seen | {resolved}), payload)


def load_config(path: str | Path, *, repo_root: Path | None = None) -> AppConfig:
    """Load, inherit, expand, validate, and root relative paths."""
    config_path = Path(path).resolve()
    root = (repo_root or config_path.parents[2]).resolve()
    payload = _expand_env(_read_yaml(config_path))
    config = AppConfig.model_validate(payload)
    path_values = {}
    for field, value in config.paths.model_dump().items():
        candidate = Path(value)
        path_values[field] = candidate if candidate.is_absolute() else root / candidate
    return config.model_copy(update={"paths": PathsConfig.model_validate(path_values)})
