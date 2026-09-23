"""Aggregate canonical flow records into fixed-dimensional trajectories."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from cyberworld.labels import COMPROMISE_STAGES, AttackStage

STATE_FEATURES = (
    "total_flows",
    "tcp_flows",
    "udp_flows",
    "other_flows",
    "total_packets",
    "fwd_packets",
    "bwd_packets",
    "total_bytes",
    "fwd_bytes",
    "bwd_bytes",
    "packet_rate",
    "byte_rate",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_src_ports",
    "unique_dst_ports",
    "mean_duration",
    "std_duration",
    "max_duration",
    "mean_pkt_size",
    "std_pkt_size",
    "min_pkt_size",
    "max_pkt_size",
    "mean_iat",
    "std_iat",
    "max_iat",
    "syn_rate",
    "ack_rate",
    "fin_rate",
    "rst_rate",
    "psh_rate",
    "urg_rate",
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
    "dst_port_fanout",
    "dst_host_fanout",
    "sequential_port_ratio",
    "randomized_port_score",
    "incomplete_handshake_ratio",
    "bytes_per_packet",
)

RISK_MALICIOUS_FRACTION_THRESHOLD = 0.01


def _available(series: pd.Series) -> bool:
    return bool(series.notna().any())


def _sum(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    series = frame[column]
    return (float(series.sum()), 1.0) if _available(series) else (0.0, 0.0)


def _stat(frame: pd.DataFrame, column: str, operation: str) -> tuple[float, float]:
    series = frame[column].dropna()
    if series.empty:
        return 0.0, 0.0
    value = getattr(series, operation)()
    return (float(value) if np.isfinite(value) else 0.0), 1.0


def _unique(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    series = frame[column].dropna()
    return (float(series.nunique()), 1.0) if not series.empty else (0.0, 0.0)


def _protocol_counts(frame: pd.DataFrame) -> tuple[float, float, float]:
    text = frame["protocol"].astype(str).str.lower()
    tcp = text.isin({"6", "tcp", "6.0"})
    udp = text.isin({"17", "udp", "17.0"})
    return float(tcp.sum()), float(udp.sum()), float((~tcp & ~udp).sum())


def _moment_stat(
    frame: pd.DataFrame, count_column: str, sum_column: str, sumsq_column: str
) -> tuple[float, float]:
    if not all(column in frame.columns for column in (count_column, sum_column, sumsq_column)):
        return 0.0, 0.0
    count = float(frame[count_column].fillna(0).sum())
    if count <= 0:
        return 0.0, 0.0
    total = float(frame[sum_column].fillna(0).sum())
    sumsq = float(frame[sumsq_column].fillna(0).sum())
    mean = total / count
    return max(0.0, sumsq / count - mean * mean), 1.0


def _scan_statistics(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    valid = frame.dropna(subset=["src_ip", "dst_port"])
    if valid.empty:
        return 0.0, 0.0, 0.0, 0.0
    max_ports = 0
    max_hosts = 0
    sequential_scores = []
    entropy_scores = []
    for _, group in valid.groupby("src_ip", sort=False):
        ports = np.sort(group["dst_port"].astype(int).unique())
        max_ports = max(max_ports, len(ports))
        max_hosts = max(max_hosts, group["dst_ip"].nunique())
        sequential_scores.append(float(np.mean(np.diff(ports) == 1)) if len(ports) > 1 else 0.0)
        counts = group["dst_port"].value_counts().to_numpy(dtype=np.float64)
        probability = counts / counts.sum()
        entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
        entropy_scores.append(entropy / math.log(len(counts)) if len(counts) > 1 else 0.0)
    return (
        float(max_ports),
        float(max_hosts),
        float(np.mean(sequential_scores)),
        float(np.mean(entropy_scores)),
    )


def aggregate_window(frame: pd.DataFrame, window_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, tuple[float, float]] = {}
    total_flows = float(len(frame))
    values["total_flows"] = (total_flows, 1.0)
    tcp, udp, other = _protocol_counts(frame)
    values.update({"tcp_flows": (tcp, 1.0), "udp_flows": (udp, 1.0), "other_flows": (other, 1.0)})
    for target, source in (
        ("fwd_packets", "fwd_packets"),
        ("bwd_packets", "bwd_packets"),
        ("fwd_bytes", "fwd_bytes"),
        ("bwd_bytes", "bwd_bytes"),
    ):
        values[target] = _sum(frame, source)
    for target, supplied, left, right in (
        ("total_packets", "total_packets", "fwd_packets", "bwd_packets"),
        ("total_bytes", "total_bytes", "fwd_bytes", "bwd_bytes"),
    ):
        direct = _sum(frame, supplied)
        if direct[1]:
            values[target] = direct
        elif values[left][1] and values[right][1]:
            values[target] = (values[left][0] + values[right][0], 1.0)
        else:
            values[target] = (0.0, 0.0)
    values["packet_rate"] = (
        values["total_packets"][0] / window_seconds,
        values["total_packets"][1],
    )
    values["byte_rate"] = (values["total_bytes"][0] / window_seconds, values["total_bytes"][1])
    for target, source in (
        ("unique_src_ips", "src_ip"),
        ("unique_dst_ips", "dst_ip"),
        ("unique_src_ports", "src_port"),
        ("unique_dst_ports", "dst_port"),
    ):
        values[target] = _unique(frame, source)
    for target, source, operation in (
        ("mean_duration", "duration_s", "mean"),
        ("std_duration", "duration_s", "std"),
        ("max_duration", "duration_s", "max"),
        ("mean_pkt_size", "packet_length_mean", "mean"),
        ("std_pkt_size", "packet_length_std", "mean"),
        ("min_pkt_size", "packet_length_min", "min"),
        ("max_pkt_size", "packet_length_max", "max"),
        ("mean_iat", "flow_iat_mean_s", "mean"),
        ("std_iat", "flow_iat_std_s", "mean"),
        ("max_iat", "flow_iat_max_s", "max"),
        ("mean_tcp_win_fwd", "tcp_window_fwd", "mean"),
        ("mean_tcp_win_bwd", "tcp_window_bwd", "mean"),
        ("tcp_window_var_fwd", "tcp_window_var_fwd", "mean"),
        ("tcp_window_var_bwd", "tcp_window_var_bwd", "mean"),
    ):
        values[target] = _stat(frame, source, operation) if source in frame.columns else (0.0, 0.0)
    for target, prefix in (
        ("tcp_window_var_fwd", "fwd"),
        ("tcp_window_var_bwd", "bwd"),
    ):
        moment = _moment_stat(
            frame,
            f"tcp_window_count_{prefix}",
            f"tcp_window_sum_{prefix}",
            f"tcp_window_sumsq_{prefix}",
        )
        if moment[1]:
            values[target] = moment
    packets = values["total_packets"]
    for target, source in (
        ("syn_rate", "syn_count"),
        ("ack_rate", "ack_count"),
        ("fin_rate", "fin_count"),
        ("rst_rate", "rst_count"),
        ("psh_rate", "psh_count"),
        ("urg_rate", "urg_count"),
    ):
        count = _sum(frame, source)
        available = count[1] * packets[1]
        values[target] = (count[0] / max(1.0, packets[0]), available)
    ttl = pd.concat([frame["ttl_fwd"], frame["ttl_bwd"]]).dropna()
    values["mean_ttl"] = (float(ttl.mean()), 1.0) if not ttl.empty else (0.0, 0.0)
    ttl_moment = _moment_stat(frame, "ttl_count", "ttl_sum", "ttl_sumsq")
    if ttl_moment[1]:
        values["ttl_variance"] = ttl_moment
    elif "ttl_variance" in frame.columns and frame["ttl_variance"].notna().any():
        values["ttl_variance"] = _stat(frame, "ttl_variance", "mean")
    else:
        values["ttl_variance"] = (float(ttl.var(ddof=0)), 1.0) if not ttl.empty else (0.0, 0.0)
    for target, source in (
        ("payload_mean", "payload_size_mean"),
        ("payload_std", "payload_size_std"),
        ("payload_min", "payload_size_min"),
        ("payload_max", "payload_size_max"),
    ):
        values[target] = (
            _stat(frame, source, "mean") if source in frame.columns else (0.0, 0.0)
        )
    retransmissions = (
        _sum(frame, "retransmission_count")
        if "retransmission_count" in frame.columns
        else (0.0, 0.0)
    )
    fragments = (
        _sum(frame, "ip_fragment_count")
        if "ip_fragment_count" in frame.columns
        else (0.0, 0.0)
    )
    values["retransmission_rate"] = (
        retransmissions[0] / max(1.0, packets[0]),
        retransmissions[1] * packets[1],
    )
    values["fragment_rate"] = (
        fragments[0] / max(1.0, packets[0]),
        fragments[1] * packets[1],
    )
    if "ip_fragment_count" in frame.columns and fragments[1]:
        values["fragment_presence"] = (
            float((frame["ip_fragment_count"].fillna(0) > 0).any()),
            1.0,
        )
    else:
        values["fragment_presence"] = (0.0, 0.0)
    scan = _scan_statistics(frame)
    for name, value in zip(
        ("dst_port_fanout", "dst_host_fanout", "sequential_port_ratio", "randomized_port_score"),
        scan,
        strict=True,
    ):
        values[name] = (value, float(not frame.dropna(subset=["src_ip", "dst_port"]).empty))
    syn, ack = values["syn_rate"], values["ack_rate"]
    values["incomplete_handshake_ratio"] = (max(0.0, syn[0] - ack[0]), syn[1] * ack[1])
    values["bytes_per_packet"] = (
        values["total_bytes"][0] / max(1.0, values["total_packets"][0]),
        values["total_bytes"][1] * values["total_packets"][1],
    )
    return (
        np.asarray([values[name][0] for name in STATE_FEATURES], dtype=np.float32),
        np.asarray([values[name][1] for name in STATE_FEATURES], dtype=np.float32),
    )


def build_trajectory(
    frame: pd.DataFrame,
    *,
    trajectory_id: str,
    window_seconds: int,
    output_path: Path,
) -> Path:
    if frame.empty:
        raise ValueError("Cannot build a trajectory from an empty frame")
    origin = frame["timestamp"].min()
    window_index = ((frame["timestamp"] - origin).dt.total_seconds() // window_seconds).astype(int)
    groups = {int(index): group for index, group in frame.groupby(window_index, sort=True)}
    final_index = max(groups)
    states, availability, stages, label_valid = [], [], [], []
    risk_malicious, risk_compromise, risk_valid = [], [], []
    empty = frame.iloc[0:0]
    for index in range(final_index + 1):
        group = groups.get(index, empty)
        state, mask = aggregate_window(group, window_seconds)
        states.append(state)
        availability.append(mask)
        if group.empty:
            stages.append(int(AttackStage.UNKNOWN))
            label_valid.append(0.0)
            risk_malicious.append(0.0)
            risk_compromise.append(0.0)
            risk_valid.append(0.0)
            continue
        valid = group[group["label_valid"] > 0]
        coverage = len(valid) / len(group)
        risk_valid.append(float(not valid.empty))
        if valid.empty:
            risk_malicious.append(0.0)
            risk_compromise.append(0.0)
        else:
            valid_stages = valid["operational_stage"]
            malicious_count = int(
                (~valid_stages.isin((int(AttackStage.NORMAL), int(AttackStage.UNKNOWN)))).sum()
            )
            compromise_count = int(
                valid_stages.isin(tuple(int(stage) for stage in COMPROMISE_STAGES)).sum()
            )
            risk_malicious.append(
                float(malicious_count / len(group) >= RISK_MALICIOUS_FRACTION_THRESHOLD)
            )
            risk_compromise.append(
                float(compromise_count / len(group) >= RISK_MALICIOUS_FRACTION_THRESHOLD)
            )
        if valid.empty or coverage < 0.8:
            stages.append(int(AttackStage.UNKNOWN))
            label_valid.append(0.0)
        else:
            mode = valid["operational_stage"].mode()
            stages.append(int(mode.iloc[0]))
            label_valid.append(1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trajectory_id=np.asarray(trajectory_id),
        states=np.stack(states),
        availability=np.stack(availability),
        stages=np.asarray(stages, dtype=np.int64),
        label_valid=np.asarray(label_valid, dtype=np.float32),
        risk_malicious=np.asarray(risk_malicious, dtype=np.float32),
        risk_compromise=np.asarray(risk_compromise, dtype=np.float32),
        risk_valid=np.asarray(risk_valid, dtype=np.float32),
        feature_names=np.asarray(STATE_FEATURES),
    )
    return output_path
