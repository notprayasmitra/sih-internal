"""Masked multi-task objectives for probabilistic rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from cyberworld.models.world_model import RolloutOutput


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gaussian_nll(mean: Tensor, log_variance: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    elementwise = 0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))
    return masked_mean(elementwise, mask)


@dataclass(frozen=True)
class LossWeights:
    state: float
    stage: float
    malicious: float
    compromise: float
    jepa: float
    vicreg_invariance: float
    vicreg_variance: float
    vicreg_covariance: float


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    state: Tensor
    stage: Tensor
    malicious: Tensor
    compromise: Tensor
    jepa: Tensor


def vicreg_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    invariance_weight: float,
    variance_weight: float,
    covariance_weight: float,
) -> Tensor:
    """VICReg over flattened batch/time embeddings."""
    prediction = prediction.flatten(0, 1)
    target = target.flatten(0, 1)
    invariance = F.mse_loss(prediction, target)

    def variance_term(value: Tensor) -> Tensor:
        std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
        return torch.relu(1.0 - std).mean()

    def covariance_term(value: Tensor) -> Tensor:
        centered = value - value.mean(dim=0)
        denominator = max(1, value.shape[0] - 1)
        covariance = centered.T @ centered / denominator
        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance.square().sum() - diagonal.square().sum()
        return off_diagonal / value.shape[1]

    variance = variance_term(prediction) + variance_term(target)
    covariance = covariance_term(prediction) + covariance_term(target)
    return (
        invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance
    )


def compute_loss(
    prediction: RolloutOutput,
    batch: dict[str, Tensor],
    weights: LossWeights,
) -> LossOutput:
    state = gaussian_nll(
        prediction.state_mean,
        prediction.state_log_variance,
        batch["future_states"],
        batch["future_availability"],
    )
    valid = batch["future_label_valid"]
    stage_raw = F.cross_entropy(
        prediction.stage_logits.flatten(0, 1),
        batch["future_stages"].flatten(),
        reduction="none",
    ).view_as(valid)
    stage = masked_mean(stage_raw, valid)
    malicious = masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction.malicious_logits, batch["future_malicious"], reduction="none"
        ),
        valid,
    )
    compromise = masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction.compromise_logits, batch["future_compromise"], reduction="none"
        ),
        valid,
    )
    jepa = state.new_zeros(())
    if prediction.predicted_latents is not None and prediction.target_latents is not None:
        jepa = vicreg_loss(
            prediction.predicted_latents,
            prediction.target_latents,
            invariance_weight=weights.vicreg_invariance,
            variance_weight=weights.vicreg_variance,
            covariance_weight=weights.vicreg_covariance,
        )
    total = (
        weights.state * state
        + weights.stage * stage
        + weights.malicious * malicious
        + weights.compromise * compromise
        + weights.jepa * jepa
    )
    return LossOutput(total, state, stage, malicious, compromise, jepa)
