"""Attribution utilities for world-model prediction explanations.

The model is recurrent and the feature space is a masked state vector. We therefore
prefer gradient-based integrated gradients over a baseline instead of KernelSHAP:
SHAP's kernel approximations are not well matched to a temporal model with masked
input semantics and they are substantially slower to evaluate per prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from cyberworld.models import (
    TemporalConvolutionalWorldModel,
    TemporalJEPAWorldModel,
    TemporalTransformerWorldModel,
    WorldModel,
)
from cyberworld.preprocessing import STATE_FEATURES

TargetHead = Literal["state", "stage_logits", "malicious_risk", "compromise_risk"]


def _coerce_history(
    history: Tensor | np.ndarray,
    availability: Tensor | np.ndarray | None,
) -> tuple[Tensor, Tensor]:
    history_tensor = torch.as_tensor(history, dtype=torch.float32)
    if history_tensor.ndim == 1:
        history_tensor = history_tensor.unsqueeze(0)
    if history_tensor.ndim != 2:
        raise ValueError("history must be rank-1 or rank-2 with shape [time, feature_dim]")
    if availability is None:
        availability_tensor = torch.ones_like(history_tensor)
    else:
        availability_tensor = torch.as_tensor(availability, dtype=torch.float32)
        if availability_tensor.shape != history_tensor.shape:
            raise ValueError("availability must match history shape")
    if history_tensor.shape[-1] != len(STATE_FEATURES):
        raise ValueError(
            f"history feature dimension {history_tensor.shape[-1]} does not match "
            f"STATE_FEATURES ({len(STATE_FEATURES)})"
        )
    return history_tensor, availability_tensor


def _selection_fn(
    model: WorldModel,
    history: Tensor,
    availability: Tensor,
    *,
    target_head: TargetHead,
    forecast_step: int,
) -> Tensor:
    if forecast_step < 0:
        raise ValueError("forecast_step must be >= 0")
    steps = max(1, forecast_step + 1)
    rollout = model.rollout(history.unsqueeze(0), availability.unsqueeze(0), steps)
    if target_head == "state":
        return rollout.state_mean[0, forecast_step].sum()
    if target_head == "stage_logits":
        return rollout.stage_logits[0, forecast_step].sum()
    if target_head == "malicious_risk":
        return rollout.malicious_logits[0, forecast_step].sum()
    if target_head == "compromise_risk":
        return rollout.compromise_logits[0, forecast_step].sum()
    raise ValueError(f"Unsupported target_head {target_head!r}")


def attribute_prediction(
    model: WorldModel,
    history: Tensor | np.ndarray,
    availability: Tensor | np.ndarray | None = None,
    *,
    target_head: TargetHead = "state",
    forecast_step: int = 0,
    n_steps: int = 50,
    baseline: Tensor | np.ndarray | None = None,
) -> np.ndarray:
    """Return integrated-gradient attribution for a selected prediction head.

    Masked features are respected by zeroing their contribution at every interpolation
    step and by zeroing the final attribution vector for unavailable dimensions.
    """
    if not isinstance(model, (WorldModel, TemporalJEPAWorldModel)):
        raise TypeError("model must be a WorldModel or TemporalJEPAWorldModel instance")

    history_tensor, availability_tensor = _coerce_history(history, availability)
    base_state = history_tensor[-1].clone().detach()
    base_mask = availability_tensor[-1].clone().detach()
    reference = (
        torch.zeros_like(base_state)
        if baseline is None
        else torch.as_tensor(baseline, dtype=torch.float32)
    )
    if reference.shape != base_state.shape:
        raise ValueError("baseline shape must match the final observed state")

    state_var = base_state.clone().detach().requires_grad_(True)
    integrated = torch.zeros_like(base_state)
    steps = max(1, int(n_steps))
    model.eval()

    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        interpolated = reference + float(alpha) * (state_var - reference)
        interpolated = interpolated * base_mask
        modified_history = history_tensor.clone()
        modified_history[-1] = interpolated
        modified_availability = availability_tensor.clone()
        modified_availability[-1] = base_mask
        score = _selection_fn(
            model,
            modified_history,
            modified_availability,
            target_head=target_head,
            forecast_step=forecast_step,
        )
        grad = torch.autograd.grad(score, state_var, retain_graph=True, allow_unused=False)[0]
        integrated = integrated + grad * (state_var - reference) / steps

    integrated = integrated * base_mask
    return integrated.detach().cpu().numpy()


def load_model_from_checkpoint(path: str | Path) -> tuple[WorldModel, dict]:
    """Load a checkpoint and reconstruct the matching model architecture."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a payload dictionary")
    config = payload.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    feature_dim = int(data_cfg.get("feature_dim", len(STATE_FEATURES)))
    num_stages = int(data_cfg.get("num_stages", 9))
    architecture = str(model_cfg.get("architecture", "probabilistic_lstm"))
    if architecture == "latent_jepa":
        model = TemporalJEPAWorldModel(
            feature_dim=feature_dim,
            num_stages=num_stages,
            projection_dim=int(model_cfg.get("projection_dim", 32)),
            latent_dim=int(model_cfg.get("latent_dim", 96)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            min_log_variance=float(model_cfg.get("min_log_variance", -8.0)),
            max_log_variance=float(model_cfg.get("max_log_variance", 5.0)),
        )
    elif architecture == "temporal_transformer":
        model = TemporalTransformerWorldModel(
            feature_dim=feature_dim,
            num_stages=num_stages,
            projection_dim=int(model_cfg.get("projection_dim", 32)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            min_log_variance=float(model_cfg.get("min_log_variance", -8.0)),
            max_log_variance=float(model_cfg.get("max_log_variance", 5.0)),
        )
    elif architecture == "temporal_cnn":
        model = TemporalConvolutionalWorldModel(
            feature_dim=feature_dim,
            num_stages=num_stages,
            projection_dim=int(model_cfg.get("projection_dim", 32)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            min_log_variance=float(model_cfg.get("min_log_variance", -8.0)),
            max_log_variance=float(model_cfg.get("max_log_variance", 5.0)),
        )
    else:
        model = WorldModel(
            feature_dim=feature_dim,
            num_stages=num_stages,
            projection_dim=int(model_cfg.get("projection_dim", 32)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            min_log_variance=float(model_cfg.get("min_log_variance", -8.0)),
            max_log_variance=float(model_cfg.get("max_log_variance", 5.0)),
        )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
