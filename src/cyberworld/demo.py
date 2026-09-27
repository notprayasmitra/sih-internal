"""Reusable data-loading and inference services for the offline demonstration app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyberworld.adapters import read_flow_csv
from cyberworld.config import AppConfig
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset, load_trajectories
from cyberworld.explain import attribute_prediction
from cyberworld.labels import AttackStage
from cyberworld.pcap import extract_pcap
from cyberworld.preprocessing import STATE_FEATURES, build_trajectory

PACKET_DERIVED_FEATURES = frozenset(
    {
        "mean_tcp_win_fwd",
        "mean_tcp_win_bwd",
        "tcp_window_var_fwd",
        "tcp_window_var_bwd",
        "retransmission_rate",
        "mean_ttl",
        "ttl_variance",
        "fragment_rate",
        "fragment_presence",
        "payload_mean",
        "payload_std",
        "payload_min",
        "payload_max",
    }
)


@dataclass(frozen=True)
class DemoInput:
    name: str
    source_kind: str
    trajectory: Trajectory
    origin: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class DemoAnalysis:
    timeline: pd.DataFrame
    flagged: pd.DataFrame
    peak_window_index: int
    peak_context_start: int
    peak_score: float
    peak_stage: AttackStage
    peak_horizon: pd.DataFrame


FEATURED_SAMPLE_NAMES = (
    "CIC-IDS2018 — DDoS Attack (Feb 21)",
    "CIC-IDS2018 — Infiltration (Mar 1)",
)

BUNDLED_SAMPLES = {
    "CIC-IDS2018 — Brute Force (Feb 14)": {
        "filename": "cic_ids2018__wednesday-14-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-14T00:00:00Z",
    },
    "CIC-IDS2018 — DoS (Feb 16)": {
        "filename": "cic_ids2018__friday-16-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-16T00:00:00Z",
    },
    "CIC-IDS2018 — DDoS LOIC-HTTP (Feb 20)": {
        "filename": "cic_ids2018__thuesday-20-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-20T00:00:00Z",
    },
    "CIC-IDS2018 — DDoS Attack (Feb 21)": {
        "filename": "cic_ids2018__wednesday-21-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-21T01:55:46Z",
    },
    "CIC-IDS2018 — Web Attacks (Feb 22)": {
        "filename": "cic_ids2018__thursday-22-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-22T00:00:00Z",
    },
    "CIC-IDS2018 — Infiltration (Feb 28)": {
        "filename": "cic_ids2018__wednesday-28-02-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-02-28T00:00:00Z",
    },
    "CIC-IDS2018 — Infiltration (Mar 1)": {
        "filename": "cic_ids2018__thursday-01-03-2018_trafficforml_cicflowmeter.npz",
        "origin": "2018-03-01T01:00:00Z",
    },
    "CTU-13 — Scenario 1, Neris": {
        "filename": "ctu13__scenario_01.npz",
        "origin": "2011-08-10T00:00:00Z",
    },
    "CTU-13 — Scenario 9, Neris": {
        "filename": "ctu13__scenario_09.npz",
        "origin": "2011-08-17T00:00:00Z",
    },
    "CTU-13 — Scenario 10, Rbot": {
        "filename": "ctu13__scenario_10.npz",
        "origin": "2011-08-18T00:00:00Z",
    },
    "CTU-13 — Scenario 11, Rbot ICMP DoS": {
        "filename": "ctu13__scenario_11.npz",
        "origin": "2011-08-18T00:00:00Z",
    },
}


def load_bundled_sample(name: str, config: AppConfig) -> DemoInput:
    spec = BUNDLED_SAMPLES[name]
    path = config.paths.trajectory_dir / spec["filename"]
    trajectories = load_trajectories(path.parent, config.data.feature_dim, config.data.num_stages)
    trajectory = next(
        item for item in trajectories if item.trajectory_id.replace(":", "__") == path.stem
    )
    origin = pd.Timestamp(spec["origin"])
    end = origin + pd.Timedelta(seconds=(len(trajectory.states) - 1) * config.data.window_seconds)
    return DemoInput(name, "Prepared local trajectory", trajectory, origin, end)


def load_uploaded_input(
    path: Path,
    *,
    dataset_id: str,
    config: AppConfig,
) -> DemoInput:
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        frame = extract_pcap(path)
        source_kind = "Uploaded PCAP"
    elif suffix == ".csv":
        frame = read_flow_csv(path, dataset_id)
        source_kind = f"Uploaded {dataset_id} CSV"
    else:
        raise ValueError("Upload must be a .pcap, .pcapng, or .csv file")
    origin = pd.Timestamp(frame["timestamp"].min())
    end = pd.Timestamp(frame["timestamp"].max())
    with TemporaryDirectory(prefix="cyberworld-demo-") as directory:
        output = Path(directory) / "uploaded.npz"
        build_trajectory(
            frame,
            trajectory_id=f"upload:{path.stem}",
            window_seconds=config.data.window_seconds,
            output_path=output,
        )
        trajectory = load_trajectories(
            output.parent, config.data.feature_dim, config.data.num_stages
        )[0]
    return DemoInput(path.name, source_kind, trajectory, origin, end)


def feature_matrix(loaded: DemoInput) -> pd.DataFrame:
    frame = pd.DataFrame(loaded.trajectory.states, columns=STATE_FEATURES)
    frame.insert(0, "window_index", np.arange(len(frame)))
    return frame


def feature_availability(loaded: DemoInput) -> pd.DataFrame:
    available = loaded.trajectory.availability.any(axis=0)
    rows = []
    for feature, populated in zip(STATE_FEATURES, available, strict=True):
        rows.append(
            {
                "feature": feature,
                "family": "packet-derived" if feature in PACKET_DERIVED_FEATURES else "flow-level",
                "populated": bool(populated),
            }
        )
    return pd.DataFrame(rows)


def _dominant_protocol(state: np.ndarray) -> str:
    counts = state[
        [
            STATE_FEATURES.index("tcp_flows"),
            STATE_FEATURES.index("udp_flows"),
            STATE_FEATURES.index("other_flows"),
        ]
    ]
    return ("TCP", "UDP", "Other")[int(np.argmax(counts))]


def analyze_trajectory(
    loaded: DemoInput,
    *,
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    config: AppConfig,
    device: torch.device,
    threshold: float = 0.2,
) -> DemoAnalysis:
    normalized = normalizer.transform(loaded.trajectory)
    dataset = TrajectoryWindowDataset(
        [normalized],
        history_windows=config.data.history_windows,
        forecast_windows=config.data.forecast_windows,
    )
    if not dataset:
        raise ValueError("Trajectory is shorter than the configured history and forecast span")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    first_scores: list[np.ndarray] = []
    first_stages: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model.rollout(
                batch["history"].to(device),
                batch["history_availability"].to(device),
                config.data.forecast_windows,
            )
            scores = torch.sigmoid(prediction.malicious_logits).cpu().numpy()
            stages = prediction.stage_logits.argmax(dim=-1).cpu().numpy()
            first_scores.append(scores[:, 0])
            first_stages.append(stages[:, 0])
            all_scores.append(scores)
    scores = np.concatenate(first_scores)
    stages = np.concatenate(first_stages)
    horizon_scores = np.concatenate(all_scores)
    context_starts = np.arange(len(scores))
    window_indices = context_starts + config.data.history_windows
    timestamps = loaded.origin + pd.to_timedelta(
        window_indices * config.data.window_seconds, unit="s"
    )
    timeline = pd.DataFrame(
        {
            "window_index": window_indices,
            "timestamp": timestamps,
            "risk_score": scores,
            "threshold": threshold,
            "above_threshold": scores >= threshold,
            "predicted_stage": [AttackStage(int(value)).name for value in stages],
        }
    )
    peak_position = int(np.argmax(scores))
    peak_window_index = int(window_indices[peak_position])
    peak_context_start = int(context_starts[peak_position])
    peak_stage = AttackStage(int(stages[peak_position]))
    peak_horizon = pd.DataFrame(
        {
            "forecast_step": np.arange(1, config.data.forecast_windows + 1),
            "target_window": peak_window_index + np.arange(config.data.forecast_windows),
            "risk_score": horizon_scores[peak_position],
        }
    )
    raw_states = loaded.trajectory.states
    flagged = timeline.loc[timeline["above_threshold"]].copy()
    flagged["protocol"] = [
        _dominant_protocol(raw_states[int(index)]) for index in flagged["window_index"]
    ]
    for feature in ("unique_dst_ports", "mean_duration", "total_flows"):
        feature_index = STATE_FEATURES.index(feature)
        flagged[feature] = [
            raw_states[int(index), feature_index] for index in flagged["window_index"]
        ]
    return DemoAnalysis(
        timeline=timeline,
        flagged=flagged,
        peak_window_index=peak_window_index,
        peak_context_start=peak_context_start,
        peak_score=float(scores[peak_position]),
        peak_stage=peak_stage,
        peak_horizon=peak_horizon,
    )


def explain_peak(
    loaded: DemoInput,
    analysis: DemoAnalysis,
    *,
    model: torch.nn.Module,
    normalizer: StandardNormalizer,
    config: AppConfig,
    top_k: int = 10,
) -> pd.DataFrame:
    normalized = normalizer.transform(loaded.trajectory)
    start = analysis.peak_context_start
    split = start + config.data.history_windows
    attribution = attribute_prediction(
        model.cpu(),
        normalized.states[start:split],
        normalized.availability[start:split],
        target_head="malicious_risk",
        forecast_step=0,
    )
    indices = np.argsort(np.abs(attribution))[::-1][:top_k]
    raw_window = loaded.trajectory.states[split - 1]
    return pd.DataFrame(
        {
            "feature": [STATE_FEATURES[index] for index in indices],
            "attribution": attribution[indices],
            "absolute_attribution": np.abs(attribution[indices]),
            "raw_value": raw_window[indices],
            "available": loaded.trajectory.availability[split - 1, indices].astype(bool),
        }
    )


def load_benchmark(path: Path, trajectory_id: str) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["trajectories"].get(trajectory_id)


def benchmark_views(report: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    malicious = report["malicious_risk"]
    fixed_rows = []
    best_rows = []
    for label, key in (
        ("Logistic regression", "logistic_baseline"),
        ("Probabilistic LSTM", "world_model_epoch_16"),
    ):
        fixed = malicious[key]["fixed_threshold_0.5"]
        fixed_rows.append({"model": label, "threshold": 0.5, **fixed})
        threshold, metrics = max(
            malicious[key]["threshold_sweep"].items(), key=lambda item: item[1]["f1"]
        )
        best_rows.append({"model": label, "threshold": float(threshold), **metrics})
    columns = ["model", "threshold", "precision", "recall", "f1", "fpr"]
    return pd.DataFrame(fixed_rows)[columns], pd.DataFrame(best_rows)[columns]
