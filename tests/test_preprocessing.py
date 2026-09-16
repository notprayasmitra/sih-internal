from pathlib import Path

import numpy as np
import pandas as pd

from cyberworld.preprocessing import STATE_FEATURES, build_trajectory


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
