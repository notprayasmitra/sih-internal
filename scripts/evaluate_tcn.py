#!/usr/bin/env python3
"""Evaluate the isolated full-combined TCN checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cyberworld.baseline.metrics import binary_classification_report
from cyberworld.cli import _resolve_trajectory_split
from cyberworld.config import load_config
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset, load_trajectories
from cyberworld.explain import load_model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/experiment/full_combined_tcn.yaml", repo_root=ROOT)
CHECKPOINT = ROOT / "checkpoints/world_model_full_combined_tcn/best.pt"
NORMALIZER = StandardNormalizer.load(ROOT / "checkpoints/world_model_full_combined_tcn/normalizer.npz")
MODEL, META = load_model_from_checkpoint(CHECKPOINT)
MODEL.eval()
all_trajectories = load_trajectories(CONFIG.paths.trajectory_dir, CONFIG.data.feature_dim, CONFIG.data.num_stages)
_, validation = _resolve_trajectory_split(all_trajectories, CONFIG)
THRESHOLDS = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]


def predict(item: Trajectory) -> dict[str, np.ndarray]:
    transformed = NORMALIZER.transform(item)
    dataset = TrajectoryWindowDataset([transformed], history_windows=CONFIG.data.history_windows, forecast_windows=CONFIG.data.forecast_windows)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    values: dict[str, list[np.ndarray]] = {"risk": [], "risk_label": [], "stage": [], "stage_label": [], "stage_valid": [], "compromise": [], "compromise_label": []}
    with torch.no_grad():
        for batch in loader:
            out = MODEL.rollout(batch["history"], batch["history_availability"], CONFIG.data.forecast_windows)
            risk_valid = batch["future_risk_valid"].numpy().astype(bool)
            stage_valid = batch["future_label_valid"].numpy().astype(bool)
            malicious = torch.sigmoid(out.malicious_logits).numpy()
            compromise = torch.sigmoid(out.compromise_logits).numpy()
            values["risk"].append(malicious[risk_valid]); values["risk_label"].append(batch["future_risk_malicious"].numpy()[risk_valid])
            values["compromise"].append(compromise[risk_valid]); values["compromise_label"].append(batch["future_risk_compromise"].numpy()[risk_valid])
            values["stage"].append(out.stage_logits.argmax(-1).numpy()[stage_valid]); values["stage_label"].append(batch["future_stages"].numpy()[stage_valid]); values["stage_valid"].append(stage_valid)
    return {key: np.concatenate(parts) for key, parts in values.items()}


def sweep(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    return {f"{t:.1f}": binary_classification_report(labels, (scores >= t).astype(int)) for t in THRESHOLDS}


def stats(labels: np.ndarray, scores: np.ndarray) -> dict:
    result = {}
    for name, value in (("benign", 0), ("malicious", 1)):
        selected = scores[labels == value]
        result[name] = {"count": int(selected.size), "mean": float(selected.mean()) if selected.size else None, "median": float(np.median(selected)) if selected.size else None, "min": float(selected.min()) if selected.size else None, "max": float(selected.max()) if selected.size else None}
    return result


results = {}
for item in validation:
    data = predict(item)
    risk_sweep = sweep(data["risk_label"], data["risk"])
    best_threshold, best_metrics = max(risk_sweep.items(), key=lambda pair: pair[1]["f1"])
    compromise_sweep = sweep(data["compromise_label"], data["compromise"])
    compromise_threshold, compromise_best = max(compromise_sweep.items(), key=lambda pair: pair[1]["f1"])
    stage_accuracy = float((data["stage"] == data["stage_label"]).mean()) if data["stage"].size else None
    results[item.trajectory_id] = {"risk_stats": stats(data["risk_label"], data["risk"]), "threshold_sweep": risk_sweep, "best": {"threshold": float(best_threshold), **best_metrics}, "stage_accuracy": stage_accuracy, "stage_valid_count": int(data["stage"].size), "compromise_positive_count": int(data["compromise_label"].sum()), "compromise_stats": stats(data["compromise_label"], data["compromise"]), "compromise_threshold_sweep": compromise_sweep, "compromise_best": {"threshold": float(compromise_threshold), **compromise_best}}

metrics = [json.loads(line) for line in (ROOT / "runs/world_model_full_combined_tcn/metrics.jsonl").read_text().splitlines()]
print(json.dumps({"checkpoint": str(CHECKPOINT), "training_curve": metrics, "validation": results}, indent=2))
