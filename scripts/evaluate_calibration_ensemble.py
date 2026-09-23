#!/usr/bin/env python3
"""Evaluate train-only Platt calibration and LSTM+JEPA risk ensembles.

This is deliberately an offline analysis. It never changes model checkpoints,
runtime scoring, preprocessing, or application behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from cyberworld.baseline.metrics import binary_classification_report
from cyberworld.cli import _resolve_trajectory_split
from cyberworld.config import load_config
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.trajectory import (
    Trajectory,
    TrajectoryWindowDataset,
    load_trajectories,
)
from cyberworld.explain import load_model_from_checkpoint
from cyberworld.runtime import select_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment/full_combined.yaml"
LSTM_CHECKPOINT = ROOT / "checkpoints/world_model_full_combined_lstm/best.pt"
JEPA_CHECKPOINT = ROOT / "checkpoints/world_model_full_combined_jepa/best.pt"
LSTM_NORMALIZER = ROOT / "checkpoints/world_model_full_combined_lstm/normalizer.npz"
JEPA_NORMALIZER = ROOT / "checkpoints/world_model_full_combined_jepa/normalizer.npz"
OUTPUT_PATH = (
    ROOT / "runs/world_model_full_combined_lstm/calibration_ensemble_analysis.json"
)
THRESHOLDS = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))
PROTECTED = {
    "cic_ids2018:wednesday-21-02-2018_trafficforml_cicflowmeter": {
        "f1": 0.6319,
        "precision": 1.0,
        "fpr": 0.0,
    },
    "cic_ids2018:thursday-01-03-2018_trafficforml_cicflowmeter": {
        "f1": 0.5685,
        "precision": 0.6414,
        "fpr": 0.1076,
    },
    "ctu13:scenario_09": {"f1": 0.6982, "precision": 0.7792, "fpr": 0.1870},
    "ctu13:scenario_11": {"f1": 0.5274, "precision": 0.4109, "fpr": 0.1693},
}


def predict(
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    trajectories: list[Trajectory],
    *,
    history_windows: int,
    forecast_windows: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    model = model.to(device)
    model.eval()
    for original in trajectories:
        trajectory = normalizer.transform(original)
        dataset = TrajectoryWindowDataset(
            [trajectory],
            history_windows=history_windows,
            forecast_windows=forecast_windows,
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        logits: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                prediction = model.rollout(
                    batch["history"].to(device),
                    batch["history_availability"].to(device),
                    forecast_windows,
                )
                valid = batch["future_risk_valid"].numpy() > 0
                logits.append(prediction.malicious_logits.cpu().numpy()[valid])
                labels.append(batch["future_risk_malicious"].numpy()[valid])
        logit_array = np.concatenate(logits).astype(np.float64, copy=False)
        results[original.trajectory_id] = {
            "logits": logit_array,
            "scores": 1.0 / (1.0 + np.exp(-logit_array)),
            "labels": np.concatenate(labels).astype(np.int64, copy=False),
        }
    return results


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return binary_classification_report(labels, (scores >= threshold).astype(np.int64))


def sweep(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    return {f"{threshold:.1f}": metrics_at(labels, scores, threshold) for threshold in THRESHOLDS}


def best(sweep_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    threshold, metrics = max(sweep_results.items(), key=lambda item: item[1]["f1"])
    return {"threshold": float(threshold), **metrics}


def score_stats(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    output = {}
    for name, value in (("negative", 0), ("positive", 1)):
        selected = scores[labels == value]
        output[name] = {
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "median": float(np.median(selected)),
            "min": float(selected.min()),
            "max": float(selected.max()),
        }
    return output


def assert_protected(trajectory_id: str, observed: dict[str, Any]) -> None:
    expected = PROTECTED[trajectory_id]
    for key, rounded in expected.items():
        if abs(observed[key] - rounded) > 0.00011:
            raise AssertionError(
                f"Protected {trajectory_id} {key} changed: {observed[key]} vs {rounded}"
            )


config = load_config(CONFIG_PATH, repo_root=ROOT)
all_trajectories = load_trajectories(
    config.paths.trajectory_dir, config.data.feature_dim, config.data.num_stages
)
train, validation = _resolve_trajectory_split(all_trajectories, config)
device_info = select_device(config.training.device)
lstm, lstm_checkpoint = load_model_from_checkpoint(LSTM_CHECKPOINT)
jepa, jepa_checkpoint = load_model_from_checkpoint(JEPA_CHECKPOINT)
lstm_normalizer = StandardNormalizer.load(LSTM_NORMALIZER)
jepa_normalizer = StandardNormalizer.load(JEPA_NORMALIZER)

lstm_train = predict(
    lstm,
    lstm_normalizer,
    train,
    history_windows=config.data.history_windows,
    forecast_windows=config.data.forecast_windows,
    device=device_info.device,
)
lstm_validation = predict(
    lstm,
    lstm_normalizer,
    validation,
    history_windows=config.data.history_windows,
    forecast_windows=config.data.forecast_windows,
    device=device_info.device,
)
jepa_validation = predict(
    jepa,
    jepa_normalizer,
    validation,
    history_windows=config.data.history_windows,
    forecast_windows=config.data.forecast_windows,
    device=device_info.device,
)

calibration_logits = np.concatenate([lstm_train[item.trajectory_id]["logits"] for item in train])
calibration_labels = np.concatenate([lstm_train[item.trajectory_id]["labels"] for item in train])
calibrator = LogisticRegression(C=1_000_000.0, solver="liblinear", max_iter=1000)
calibrator.fit(calibration_logits.reshape(-1, 1), calibration_labels)

trajectory_results = {}
calibrated_thresholds = []
for trajectory in validation:
    trajectory_id = trajectory.trajectory_id
    lstm_item = lstm_validation[trajectory_id]
    jepa_item = jepa_validation[trajectory_id]
    if not np.array_equal(lstm_item["labels"], jepa_item["labels"]):
        raise AssertionError(f"LSTM/JEPA target alignment differs for {trajectory_id}")
    labels = lstm_item["labels"]
    lstm_scores = lstm_item["scores"]
    jepa_scores = jepa_item["scores"]
    calibrated_scores = calibrator.predict_proba(lstm_item["logits"].reshape(-1, 1))[:, 1]
    average_scores = (lstm_scores + jepa_scores) / 2.0
    maximum_scores = np.maximum(lstm_scores, jepa_scores)

    methods = {}
    for name, scores in (
        ("lstm_uncalibrated", lstm_scores),
        ("lstm_platt_calibrated", calibrated_scores),
        ("jepa_uncalibrated", jepa_scores),
        ("ensemble_average", average_scores),
        ("ensemble_max", maximum_scores),
    ):
        method_sweep = sweep(labels, scores)
        methods[name] = {
            "score_stats": score_stats(labels, scores),
            "threshold_sweep": method_sweep,
            "best": best(method_sweep),
        }

    assert_protected(trajectory_id, methods["lstm_uncalibrated"]["best"])
    lstm_best = methods["lstm_uncalibrated"]["best"]
    calibrated_best = methods["lstm_platt_calibrated"]["best"]
    jepa_best = methods["jepa_uncalibrated"]["best"]
    better_individual_f1 = max(lstm_best["f1"], jepa_best["f1"])
    def protects_lstm(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
        return (
            candidate["f1"] + 1e-12 >= baseline["f1"]
            and candidate["precision"] + 1e-12 >= baseline["precision"]
            and candidate["fpr"] <= baseline["fpr"] + 1e-12
        )

    calibration_adoptable = protects_lstm(calibrated_best, lstm_best)
    average_adoptable = (
        methods["ensemble_average"]["best"]["f1"] + 1e-12 >= better_individual_f1
        and protects_lstm(methods["ensemble_average"]["best"], lstm_best)
    )
    max_adoptable = (
        methods["ensemble_max"]["best"]["f1"] + 1e-12 >= better_individual_f1
        and protects_lstm(methods["ensemble_max"]["best"], lstm_best)
    )
    if calibration_adoptable:
        calibrated_thresholds.append(calibrated_best["threshold"])

    candidates = {"lstm_uncalibrated": lstm_best}
    if calibration_adoptable:
        candidates["lstm_platt_calibrated"] = calibrated_best
    if average_adoptable:
        candidates["ensemble_average"] = methods["ensemble_average"]["best"]
    if max_adoptable:
        candidates["ensemble_max"] = methods["ensemble_max"]["best"]
    recommended_name, recommended_metrics = max(
        candidates.items(), key=lambda item: item[1]["f1"]
    )
    trajectory_results[trajectory_id] = {
        "rows": int(labels.size),
        "methods": methods,
        "decisions": {
            "adoption_guard": (
                "F1 and precision must match/beat LSTM; FPR must match/improve. "
                "Ensembles must additionally match/beat the better individual F1."
            ),
            "calibration_adoptable_without_regression": calibration_adoptable,
            "ensemble_average_adoptable": average_adoptable,
            "ensemble_max_adoptable": max_adoptable,
            "better_individual_f1": better_individual_f1,
            "recommended_method": recommended_name,
            "recommended_metrics": recommended_metrics,
        },
    }

payload = {
    "event": "train_only_calibration_and_lstm_jepa_ensemble_analysis",
    "additive_only": True,
    "runtime_behavior_changed": False,
    "config": str(CONFIG_PATH),
    "device": device_info.accelerator,
    "checkpoints": {
        "lstm": {"path": str(LSTM_CHECKPOINT), "epoch": int(lstm_checkpoint["epoch"])},
        "jepa": {"path": str(JEPA_CHECKPOINT), "epoch": int(jepa_checkpoint["epoch"])},
    },
    "calibration": {
        "method": "Platt scaling (unregularized logistic fit over LSTM logits)",
        "fit_scope": "seven training trajectories only",
        "fit_rows": int(calibration_labels.size),
        "fit_label_counts": {
            "negative": int((calibration_labels == 0).sum()),
            "positive": int((calibration_labels == 1).sum()),
        },
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "iterations": int(calibrator.n_iter_[0]),
        "adoptable_optimal_thresholds": calibrated_thresholds,
        "adoptable_threshold_range": (
            [min(calibrated_thresholds), max(calibrated_thresholds)]
            if calibrated_thresholds
            else None
        ),
    },
    "protected_reference": PROTECTED,
    "trajectories": trajectory_results,
    "output": str(OUTPUT_PATH),
}
OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"output": str(OUTPUT_PATH), "device": device_info.accelerator}))
