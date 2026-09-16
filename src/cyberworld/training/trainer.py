"""Portable CUDA/MPS/CPU training loop."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW

from cyberworld.config import AppConfig
from cyberworld.models import WorldModel
from cyberworld.runtime import DeviceInfo
from cyberworld.training.checkpoint import save_checkpoint_atomic
from cyberworld.training.losses import LossWeights, compute_loss
from cyberworld.training.metrics import MetricAccumulator


class Trainer:
    def __init__(self, model: WorldModel, config: AppConfig, device: DeviceInfo) -> None:
        self.model = model.to(device.device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.amp_enabled = config.training.mixed_precision and device.accelerator == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.weights = LossWeights(
            state=config.training.lambda_state,
            stage=config.training.lambda_stage,
            malicious=config.training.lambda_malicious,
            compromise=config.training.lambda_compromise,
            jepa=config.training.lambda_jepa,
            vicreg_invariance=config.training.vicreg_invariance,
            vicreg_variance=config.training.vicreg_variance,
            vicreg_covariance=config.training.vicreg_covariance,
        )
        self.run_dir = config.paths.run_root / config.experiment.name
        self.checkpoint_dir = config.paths.checkpoint_root / config.experiment.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _move(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        non_blocking = self.device.accelerator == "cuda"
        return {
            key: value.to(self.device.device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    def _run_epoch(self, loader: Iterable[dict[str, Tensor]], training: bool) -> dict[str, float]:
        self.model.train(training)
        accumulator = MetricAccumulator()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.training.accumulate_steps
        pending_steps = 0

        def optimizer_step() -> None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(loader, start=1):
            batch = self._move(raw_batch)
            context = torch.amp.autocast("cuda", enabled=self.amp_enabled)
            with torch.set_grad_enabled(training), context:
                prediction = self.model.rollout(
                    batch["history"],
                    batch["history_availability"],
                    self.config.data.forecast_windows,
                    teacher_states=batch["future_states"],
                    teacher_availability=batch["future_availability"],
                    teacher_forcing_ratio=(
                        self.config.training.teacher_forcing_ratio if training else 0.0
                    ),
                )
                losses = compute_loss(prediction, batch, self.weights)
                scaled_loss = losses.total / accumulation

            if training:
                self.scaler.scale(scaled_loss).backward()  # type: ignore[no-untyped-call]
                pending_steps += 1
                if step % accumulation == 0:
                    optimizer_step()
                    pending_steps = 0

            accumulator.update(
                loss=losses.total,
                state_mean=prediction.state_mean,
                state_target=batch["future_states"],
                state_mask=batch["future_availability"],
                stage_logits=prediction.stage_logits,
                stage_target=batch["future_stages"],
                label_valid=batch["future_label_valid"],
            )

        if training and pending_steps:
            optimizer_step()
        return accumulator.compute()

    def fit(
        self,
        train_loader: Iterable[dict[str, Tensor]],
        validation_loader: Iterable[dict[str, Tensor]],
    ) -> dict[str, float]:
        best_loss = float("inf")
        patience = 0
        final_metrics: dict[str, float] = {}
        history_path = self.run_dir / "metrics.jsonl"

        for epoch in range(1, self.config.training.epochs + 1):
            started = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            elapsed = time.perf_counter() - started
            record = {
                "epoch": epoch,
                "seconds": elapsed,
                "train": train_metrics,
                "validation": validation_metrics,
                "device": self.device.accelerator,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            final_metrics = validation_metrics

            payload = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.model_dump(mode="json"),
                "validation": validation_metrics,
            }
            if epoch % self.config.logging.save_every_epochs == 0:
                save_checkpoint_atomic(payload, self.checkpoint_dir / "last.pt")
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                patience = 0
                save_checkpoint_atomic(payload, self.checkpoint_dir / "best.pt")
            else:
                patience += 1
                if patience >= self.config.training.early_stopping_patience:
                    break
        return final_metrics
