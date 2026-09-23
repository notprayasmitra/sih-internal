from pathlib import Path

import numpy as np
import pandas as pd

from cyberworld.labels import AttackStage
from cyberworld.preprocessing import (
    RISK_MALICIOUS_FRACTION_THRESHOLD,
    STATE_FEATURES,
    aggregate_window,
    build_trajectory,
)


def test_build_trajectory_retains_empty_windows(tmp_path: Path) -> None:
    columns = {
        "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:25Z"]),
        "protocol": [6, 17],
        "src_ip": ["a", "b"],
        "dst_ip": ["c", "d"],
        "src_port": [1, 2],
        "dst_port": [80, 53],
        "operational_stage": [0, 1],
        "label_valid": [1.0, 1.0],
    }
    numeric = [
        "duration_s",
        "fwd_packets",
        "bwd_packets",
        "total_packets",
        "fwd_bytes",
        "bwd_bytes",
        "total_bytes",
        "flow_iat_mean_s",
        "flow_iat_std_s",
        "flow_iat_max_s",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",
        "packet_length_mean",
        "packet_length_std",
        "packet_length_min",
        "packet_length_max",
        "tcp_window_fwd",
        "tcp_window_bwd",
        "ttl_fwd",
        "ttl_bwd",
    ]
    for name in numeric:
        columns[name] = [1.0, 1.0]
    frame = pd.DataFrame(columns)
    output = build_trajectory(
        frame, trajectory_id="test:x", window_seconds=10, output_path=tmp_path / "x.npz"
    )
    with np.load(output) as payload:
        assert payload["states"].shape == (3, len(STATE_FEATURES))
        assert payload["stages"][1] == 8
        assert payload["label_valid"][1] == 0


def test_packet_features_have_hand_verified_values() -> None:
    frame = pd.DataFrame(
        {
            "protocol": [6, 6, 6],
            "src_ip": ["a", "a", "a"],
            "dst_ip": ["b", "c", "d"],
            "src_port": [1, 1, 1],
            "dst_port": [80, 81, 82],
            "fwd_packets": [1, 1, 1],
            "bwd_packets": [0, 0, 0],
            "total_packets": [1, 1, 1],
            "fwd_bytes": [60, 62, 58],
            "bwd_bytes": [0, 0, 0],
            "total_bytes": [60, 62, 58],
            "ttl_fwd": [60, 62, 58],
            "ttl_bwd": [np.nan, np.nan, np.nan],
            "tcp_window_fwd": [10.0, 20.0, 30.0],
            "tcp_window_bwd": [np.nan, np.nan, np.nan],
            "tcp_window_var_fwd": [4.0, 4.0, 4.0],
            "tcp_window_var_bwd": [np.nan, np.nan, np.nan],
            "ip_fragment_count": [0, 1, 0],
        }
    )
    required_defaults = {
        "duration_s": 1.0,
        "flow_iat_mean_s": 1.0,
        "flow_iat_std_s": 0.0,
        "flow_iat_max_s": 1.0,
        "syn_count": 0.0,
        "ack_count": 0.0,
        "fin_count": 0.0,
        "rst_count": 0.0,
        "psh_count": 0.0,
        "urg_count": 0.0,
        "packet_length_mean": 60.0,
        "packet_length_std": 0.0,
        "packet_length_min": 60.0,
        "packet_length_max": 60.0,
        "retransmission_count": 0.0,
        "payload_size_mean": 0.0,
        "payload_size_std": 0.0,
        "payload_size_min": 0.0,
        "payload_size_max": 0.0,
    }
    for name, value in required_defaults.items():
        frame[name] = value

    state, availability = aggregate_window(frame, window_seconds=10)
    positions = {name: index for index, name in enumerate(STATE_FEATURES)}
    assert state[positions["mean_ttl"]] == np.float32(60.0)
    np.testing.assert_allclose(state[positions["ttl_variance"]], np.float32(8.0 / 3.0))
    assert state[positions["tcp_window_var_fwd"]] == np.float32(4.0)
    assert state[positions["fragment_rate"]] == np.float32(1.0 / 3.0)
    assert state[positions["fragment_presence"]] == np.float32(1.0)
    assert availability[positions["tcp_window_var_fwd"]] == 1.0


def test_binary_risk_uses_one_percent_flow_fraction_without_changing_stage_mode(
    tmp_path: Path,
) -> None:
    assert RISK_MALICIOUS_FRACTION_THRESHOLD == 0.01
    row_count = 300
    timestamps = pd.to_datetime(
        ["2025-01-01T00:00:00Z"] * 100
        + ["2025-01-01T00:00:10Z"] * 200
    )
    stages = np.full(row_count, int(AttackStage.NORMAL), dtype=np.int64)
    stages[0] = int(AttackStage.INITIAL_ACCESS)  # 1/100: positive at the boundary.
    stages[100] = int(AttackStage.INITIAL_ACCESS)  # 1/200: below the boundary.
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "protocol": np.full(row_count, 6),
            "src_ip": np.full(row_count, "a"),
            "dst_ip": np.full(row_count, "b"),
            "src_port": np.full(row_count, 1234),
            "dst_port": np.full(row_count, 443),
            "operational_stage": stages,
            "label_valid": np.ones(row_count, dtype=np.float32),
        }
    )
    for name in (
        "duration_s",
        "fwd_packets",
        "bwd_packets",
        "total_packets",
        "fwd_bytes",
        "bwd_bytes",
        "total_bytes",
        "flow_iat_mean_s",
        "flow_iat_std_s",
        "flow_iat_max_s",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",
        "packet_length_mean",
        "packet_length_std",
        "packet_length_min",
        "packet_length_max",
        "tcp_window_fwd",
        "tcp_window_bwd",
        "ttl_fwd",
        "ttl_bwd",
    ):
        frame[name] = 1.0

    output = build_trajectory(
        frame,
        trajectory_id="test:risk-fraction",
        window_seconds=10,
        output_path=tmp_path / "risk_fraction.npz",
    )

    with np.load(output) as payload:
        np.testing.assert_array_equal(payload["risk_malicious"], [1.0, 0.0])
        np.testing.assert_array_equal(payload["risk_compromise"], [1.0, 0.0])
        np.testing.assert_array_equal(payload["risk_valid"], [1.0, 1.0])
        np.testing.assert_array_equal(
            payload["stages"],
            [int(AttackStage.NORMAL), int(AttackStage.NORMAL)],
        )
        np.testing.assert_array_equal(payload["label_valid"], [1.0, 1.0])
