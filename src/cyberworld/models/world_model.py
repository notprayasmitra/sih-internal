"""Probabilistic recurrent network-state world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RolloutOutput:
    state_mean: Tensor
    state_log_variance: Tensor
    stage_logits: Tensor
    malicious_logits: Tensor
    compromise_logits: Tensor
    predicted_latents: Tensor | None = None
    target_latents: Tensor | None = None


class WorldModel(nn.Module):
    """Encode observed state history and recursively predict future distributions."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_stages: int,
        projection_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        min_log_variance: float,
        max_log_variance: float,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_stages = num_stages
        self.projection_dim = projection_dim
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.dynamics = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.state_mean_head = nn.Linear(hidden_dim, feature_dim)
        self.state_log_variance_head = nn.Linear(hidden_dim, feature_dim)
        self.stage_head = nn.Linear(hidden_dim, num_stages)
        self.malicious_head = nn.Linear(hidden_dim, 1)
        self.compromise_head = nn.Linear(hidden_dim, 1)

    def _encode_state(self, state: Tensor, availability: Tensor) -> Tensor:
        if state.shape != availability.shape:
            raise ValueError("state and availability must have identical shapes")
        return cast(
            Tensor,
            self.state_encoder(torch.cat((state * availability, availability), dim=-1)),
        )

    def _decode(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mean = self.state_mean_head(hidden)
        log_variance = self.state_log_variance_head(hidden).clamp(
            self.min_log_variance, self.max_log_variance
        )
        stage = self.stage_head(hidden)
        malicious = self.malicious_head(hidden).squeeze(-1)
        compromise = self.compromise_head(hidden).squeeze(-1)
        return mean, log_variance, stage, malicious, compromise

    def rollout(
        self,
        history: Tensor,
        history_availability: Tensor,
        steps: int,
        *,
        teacher_states: Tensor | None = None,
        teacher_availability: Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> RolloutOutput:
        if history.ndim != 3 or history.shape[-1] != self.feature_dim:
            raise ValueError("history must have shape [batch, time, feature_dim]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if teacher_forcing_ratio < 0.0 or teacher_forcing_ratio > 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if teacher_states is not None and teacher_states.shape[1] < steps:
            raise ValueError("teacher_states has fewer time steps than requested rollout")

        encoded_history = self._encode_state(history, history_availability)
        output, recurrent = self.dynamics(encoded_history)
        hidden = output[:, -1]
        means: list[Tensor] = []
        log_variances: list[Tensor] = []
        stages: list[Tensor] = []
        malicious: list[Tensor] = []
        compromise: list[Tensor] = []

        for step in range(steps):
            mean, log_variance, stage, mal, comp = self._decode(hidden)
            means.append(mean)
            log_variances.append(log_variance)
            stages.append(stage)
            malicious.append(mal)
            compromise.append(comp)

            next_state = mean
            next_availability = torch.ones_like(mean)
            if teacher_states is not None and teacher_forcing_ratio > 0.0:
                selector = torch.rand(mean.shape[0], 1, device=mean.device) < teacher_forcing_ratio
                next_state = torch.where(selector, teacher_states[:, step], mean)
                if teacher_availability is not None:
                    next_availability = torch.where(
                        selector, teacher_availability[:, step], next_availability
                    )
            recurrent_input = self._encode_state(next_state, next_availability).unsqueeze(1)
            output, recurrent = self.dynamics(recurrent_input, recurrent)
            hidden = output[:, -1]

        return RolloutOutput(
            state_mean=torch.stack(means, dim=1),
            state_log_variance=torch.stack(log_variances, dim=1),
            stage_logits=torch.stack(stages, dim=1),
            malicious_logits=torch.stack(malicious, dim=1),
            compromise_logits=torch.stack(compromise, dim=1),
        )

    def forward(self, history: Tensor, history_availability: Tensor, steps: int) -> RolloutOutput:
        return self.rollout(history, history_availability, steps)


class TemporalJEPAWorldModel(WorldModel):
    """Hybrid JEPA: latent future prediction plus probabilistic state decoding.

    The same state encoder represents observed and future target states. The context
    LSTM predicts future embeddings, while VICReg prevents representation collapse.
    A decoder retains the explicit future-state rollout required by the problem.
    """

    def __init__(self, *, latent_dim: int, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        hidden_dim = self.dynamics.hidden_size
        self.latent_predictor = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.target_projector = nn.Linear(self.projection_dim, latent_dim)

    def rollout(
        self,
        history: Tensor,
        history_availability: Tensor,
        steps: int,
        *,
        teacher_states: Tensor | None = None,
        teacher_availability: Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> RolloutOutput:
        if history.ndim != 3 or history.shape[-1] != self.feature_dim:
            raise ValueError("history must have shape [batch, time, feature_dim]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if teacher_forcing_ratio < 0.0 or teacher_forcing_ratio > 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        encoded = self._encode_state(history, history_availability)
        output, recurrent = self.dynamics(encoded)
        hidden = output[:, -1]
        means, variances, stages, malicious, compromise = [], [], [], [], []
        predicted_latents, target_latents = [], []

        for step in range(steps):
            predicted_latent = self.latent_predictor(hidden)
            decode_hidden = self.latent_to_hidden(predicted_latent)
            mean, log_variance, stage, mal, comp = self._decode(decode_hidden)
            means.append(mean)
            variances.append(log_variance)
            stages.append(stage)
            malicious.append(mal)
            compromise.append(comp)
            predicted_latents.append(predicted_latent)

            if teacher_states is not None:
                target_mask = (
                    teacher_availability[:, step]
                    if teacher_availability is not None
                    else torch.ones_like(teacher_states[:, step])
                )
                target_encoded = self._encode_state(teacher_states[:, step], target_mask)
                target_latents.append(self.target_projector(target_encoded))

            next_state = mean
            next_mask = torch.ones_like(mean)
            if teacher_states is not None and teacher_forcing_ratio > 0:
                selector = torch.rand(mean.shape[0], 1, device=mean.device) < teacher_forcing_ratio
                next_state = torch.where(selector, teacher_states[:, step], mean)
                if teacher_availability is not None:
                    next_mask = torch.where(selector, teacher_availability[:, step], next_mask)
            recurrent_input = self._encode_state(next_state, next_mask).unsqueeze(1)
            output, recurrent = self.dynamics(recurrent_input, recurrent)
            hidden = output[:, -1]

        return RolloutOutput(
            state_mean=torch.stack(means, dim=1),
            state_log_variance=torch.stack(variances, dim=1),
            stage_logits=torch.stack(stages, dim=1),
            malicious_logits=torch.stack(malicious, dim=1),
            compromise_logits=torch.stack(compromise, dim=1),
            predicted_latents=torch.stack(predicted_latents, dim=1),
            target_latents=torch.stack(target_latents, dim=1) if target_latents else None,
        )
