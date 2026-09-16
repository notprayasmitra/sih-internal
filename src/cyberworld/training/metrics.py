"""Streaming, dependency-light validation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    state_squared_error: float = 0.0
    state_count: float = 0.0
    stage_correct: float = 0.0
    stage_count: float = 0.0
    batches: int = 0

    def update(
        self,
        *,
        loss: Tensor,
        state_mean: Tensor,
        state_target: Tensor,
        state_mask: Tensor,
        stage_logits: Tensor,
        stage_target: Tensor,
        label_valid: Tensor,
    ) -> None:
        self.loss_sum += float(loss.detach().cpu())
        squared = (state_mean.detach() - state_target).square() * state_mask
        self.state_squared_error += float(squared.sum().cpu())
        self.state_count += float(state_mask.sum().cpu())
        correct = (stage_logits.detach().argmax(-1) == stage_target) * label_valid.bool()
        self.stage_correct += float(correct.sum().cpu())
        self.stage_count += float(label_valid.sum().cpu())
        self.batches += 1

    def compute(self) -> dict[str, float]:
        return {
            "loss": self.loss_sum / max(1, self.batches),
            "state_rmse": (self.state_squared_error / max(1.0, self.state_count)) ** 0.5,
            "stage_accuracy": self.stage_correct / max(1.0, self.stage_count),
        }
