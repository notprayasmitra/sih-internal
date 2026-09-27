"""Streaming, dependency-light validation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    state_loss_sum: float = 0.0
    stage_loss_sum: float = 0.0
    malicious_loss_sum: float = 0.0
    compromise_loss_sum: float = 0.0
    jepa_loss_sum: float = 0.0
    state_squared_error: float = 0.0
    state_count: float = 0.0
    stage_correct: float = 0.0
    stage_count: float = 0.0
    batches: int = 0

    def update(
        self,
        *,
        loss: Tensor,
        state_loss: Tensor | None = None,
        stage_loss: Tensor | None = None,
        malicious_loss: Tensor | None = None,
        compromise_loss: Tensor | None = None,
        jepa_loss: Tensor | None = None,
        state_mean: Tensor,
        state_target: Tensor,
        state_mask: Tensor,
        stage_logits: Tensor,
        stage_target: Tensor,
        label_valid: Tensor,
    ) -> None:
        self.loss_sum += float(loss.detach().cpu())
        self.state_loss_sum += float(
            (state_loss if state_loss is not None else loss).detach().cpu()
        )
        self.stage_loss_sum += float(
            (stage_loss if stage_loss is not None else loss.new_zeros(())).detach().cpu()
        )
        self.malicious_loss_sum += float(
            (malicious_loss if malicious_loss is not None else loss.new_zeros(())).detach().cpu()
        )
        self.compromise_loss_sum += float(
            (compromise_loss if compromise_loss is not None else loss.new_zeros(())).detach().cpu()
        )
        self.jepa_loss_sum += float(
            (jepa_loss if jepa_loss is not None else loss.new_zeros(())).detach().cpu()
        )
        squared = (state_mean.detach() - state_target).square() * state_mask
        self.state_squared_error += float(squared.sum().cpu())
        self.state_count += float(state_mask.sum().cpu())
        correct = (stage_logits.detach().argmax(-1) == stage_target) * label_valid.bool()
        self.stage_correct += float(correct.sum().cpu())
        self.stage_count += float(label_valid.sum().cpu())
        self.batches += 1

    def compute(self) -> dict[str, float | int | str | None]:
        stage_accuracy: float | None = None
        stage_accuracy_status = "measured"
        if self.stage_count > 0:
            stage_accuracy = self.stage_correct / self.stage_count
        else:
            stage_accuracy_status = "excluded_no_valid_labels"
        return {
            "loss": self.loss_sum / max(1, self.batches),
            "state_loss": self.state_loss_sum / max(1, self.batches),
            "stage_loss": self.stage_loss_sum / max(1, self.batches),
            "malicious_loss": self.malicious_loss_sum / max(1, self.batches),
            "compromise_loss": self.compromise_loss_sum / max(1, self.batches),
            "jepa_loss": self.jepa_loss_sum / max(1, self.batches),
            "state_rmse": (self.state_squared_error / max(1.0, self.state_count)) ** 0.5,
            "stage_accuracy": stage_accuracy,
            "stage_valid_count": int(self.stage_count),
            "stage_accuracy_status": stage_accuracy_status,
        }
