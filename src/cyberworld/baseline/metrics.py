"""Evaluation utilities for tabular baselines and world-model predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def binary_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have equal length")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    fpr = _safe_divide(fp, fp + tn)
    return {
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
        "f1": float(np.clip(f1, 0.0, 1.0)),
        "fpr": float(np.clip(fpr, 0.0, 1.0)),
        "support": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return binary_classification_report(y_true, y_pred)


def benchmark_table(report: dict[str, Any]) -> str:
    rows = [
        "model,precision,recall,f1,fpr",
    ]
    for name, metrics in report.items():
        rows.append(
            ",".join(
                [
                    name,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['fpr']:.4f}",
                ]
            )
        )
    return "\n".join(rows)
