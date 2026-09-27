"""Probabilistic temporal world-model architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class TemporalTransformerWorldModel(WorldModel):
    """Autoregressive world model with a short-history Transformer backbone."""

    def __init__(
        self,
        *,
        max_sequence_length: int = 64,
        attention_heads: int = 4,
        **kwargs: int | float,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        hidden_dim = self.dynamics.hidden_size
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.dynamics = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=2 * hidden_dim,
                dropout=float(kwargs.get("dropout", 0.1)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(kwargs.get("num_layers", 2)),
        )
        self.temporal_projection = (
            nn.Identity()
            if self.projection_dim == hidden_dim
            else nn.Linear(self.projection_dim, hidden_dim)
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.attention_heads = attention_heads

    def _transform(self, tokens: list[Tensor]) -> Tensor:
        sequence = torch.cat(tokens, dim=1)
        if sequence.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("rollout sequence exceeds configured Transformer position capacity")
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]
        return self.dynamics(sequence)[:, -1]

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
        if history.shape[1] + steps > self.position_embedding.shape[1]:
            raise ValueError("history plus rollout exceeds Transformer position capacity")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if teacher_states is not None and teacher_states.shape[1] < steps:
            raise ValueError("teacher_states has fewer time steps than requested rollout")
        tokens = [self.temporal_projection(self._encode_state(history, history_availability))]
        means: list[Tensor] = []
        variances: list[Tensor] = []
        stages: list[Tensor] = []
        malicious: list[Tensor] = []
        compromise: list[Tensor] = []
        for step in range(steps):
            hidden = self._transform(tokens)
            mean, log_variance, stage, mal, comp = self._decode(hidden)
            means.append(mean)
            variances.append(log_variance)
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
            next_token = self.temporal_projection(
                self._encode_state(next_state, next_availability)
            ).unsqueeze(1)
            tokens.append(next_token)
        return RolloutOutput(
            state_mean=torch.stack(means, dim=1),
            state_log_variance=torch.stack(variances, dim=1),
            stage_logits=torch.stack(stages, dim=1),
            malicious_logits=torch.stack(malicious, dim=1),
            compromise_logits=torch.stack(compromise, dim=1),
        )


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        residual = sequence
        value = self.conv1(F.pad(sequence, (self.left_padding, 0)))
        value = self.dropout(F.gelu(self.norm1(value)))
        value = self.conv2(F.pad(value, (self.left_padding, 0)))
        value = self.dropout(F.gelu(self.norm2(value)))
        return residual + value


class TemporalConvolutionalWorldModel(WorldModel):
    """Autoregressive world model with causal dilated residual convolutions."""

    def __init__(self, **kwargs: int | float) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        hidden_dim = self.dynamics.hidden_size
        self.temporal_projection = (
            nn.Identity()
            if self.projection_dim == hidden_dim
            else nn.Linear(self.projection_dim, hidden_dim)
        )
        num_layers = int(kwargs.get("num_layers", 2))
        dropout = float(kwargs.get("dropout", 0.1))
        self.dynamics = nn.ModuleList(
            _CausalResidualBlock(hidden_dim, 2**layer, dropout)
            for layer in range(num_layers)
        )

    def _transform(self, sequence: Tensor) -> Tensor:
        value = sequence.transpose(1, 2)
        for block in self.dynamics:
            value = block(value)
        return value.transpose(1, 2)[:, -1]

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
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")
        if teacher_states is not None and teacher_states.shape[1] < steps:
            raise ValueError("teacher_states has fewer time steps than requested rollout")
        tokens = [self.temporal_projection(self._encode_state(history, history_availability))]
        means: list[Tensor] = []
        variances: list[Tensor] = []
        stages: list[Tensor] = []
        malicious: list[Tensor] = []
        compromise: list[Tensor] = []
        for step in range(steps):
            hidden = self._transform(torch.cat(tokens, dim=1))
            mean, log_variance, stage, mal, comp = self._decode(hidden)
            means.append(mean)
            variances.append(log_variance)
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
            tokens.append(
                self.temporal_projection(
                    self._encode_state(next_state, next_availability)
                ).unsqueeze(1)
            )
        return RolloutOutput(
            state_mean=torch.stack(means, dim=1),
            state_log_variance=torch.stack(variances, dim=1),
            stage_logits=torch.stack(stages, dim=1),
            malicious_logits=torch.stack(malicious, dim=1),
            compromise_logits=torch.stack(compromise, dim=1),
        )
