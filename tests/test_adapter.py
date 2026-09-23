import warnings
from pathlib import Path

import pandas as pd

from cyberworld.adapters import read_flow_csv
from cyberworld.labels import AttackStage


def test_dapt_adapter_converts_units_and_stage(tmp_path: Path) -> None:
    path = tmp_path / "dapt.csv"
    pd.DataFrame(
        {
            "Timestamp": ["17/07/2019 02:35:09 PM"],
            "Flow Duration": [2_000_000],
            "Protocol": [6],
            "Total Fwd Packet": [2],
            "Total Bwd packets": [3],
            "Stage": ["Lateral Movement"],
            "Label": ["Attack"],
        }
    ).to_csv(path, index=False)
    frame = read_flow_csv(path, "dapt2020")
    assert frame.loc[0, "duration_s"] == 2.0
    assert frame.loc[0, "fwd_packets"] == 2
    assert frame.loc[0, "operational_stage"] == int(AttackStage.LATERAL_MOVEMENT)


def test_cic_adapter_maps_exact_init_window_headers(tmp_path: Path) -> None:
    path = tmp_path / "cic.csv"
    pd.DataFrame(
        {
            "Timestamp": ["01/03/2018 01:00:00 AM", "01/03/2018 01:00:01 AM"],
            "Flow Duration": [1_000_000, 2_000_000],
            "Protocol": [6, 17],
            "Dst Port": [443, 53],
            "Tot Fwd Pkts": [2, 1],
            "Tot Bwd Pkts": [1, 0],
            "TotLen Fwd Pkts": [100, 20],
            "TotLen Bwd Pkts": [50, 0],
            "Init Fwd Win Byts": [4096, -1],
            "Init Bwd Win Byts": [8192, -1],
            "Label": ["Benign", "Benign"],
        }
    ).to_csv(path, index=False)

    frame = read_flow_csv(path, "cic_ids2018")

    assert frame.loc[0, "tcp_window_fwd"] == 4096
    assert frame.loc[0, "tcp_window_bwd"] == 8192
    assert pd.isna(frame.loc[1, "tcp_window_fwd"])
    assert pd.isna(frame.loc[1, "tcp_window_bwd"])


def test_cic_and_ctu_dos_labels_both_map_to_impact(tmp_path: Path) -> None:
    labels = (("cic_ids2018", "DDoS attacks-LOIC-HTTP"), ("ctu13", "DDoS"))
    for dataset_id, label in labels:
        path = tmp_path / f"{dataset_id}.csv"
        pd.DataFrame({"StartTime": ["2011/08/18 15:52:39"], "Label": [label]}).to_csv(
            path, index=False
        )
        frame = read_flow_csv(path, dataset_id)
        assert frame.loc[0, "operational_stage"] == int(AttackStage.IMPACT)


def test_cic_explicit_intrusion_labels_map_to_initial_access(tmp_path: Path) -> None:
    path = tmp_path / "cic.csv"
    labels = ["FTP-BruteForce", "SSH-Bruteforce", "SQL Injection"]
    pd.DataFrame(
        {
            "Timestamp": [f"14/02/2018 01:00:0{index}" for index in range(len(labels))],
            "Label": labels,
        }
    ).to_csv(path, index=False)

    frame = read_flow_csv(path, "cic_ids2018")

    assert frame["operational_stage"].tolist() == [int(AttackStage.INITIAL_ACCESS)] * 3


def test_ctu_adapter_uses_conservative_explicit_label_mapping(tmp_path: Path) -> None:
    labels_and_stages = {
        "From-Normal-V42-Stribrek": AttackStage.NORMAL,
        "Background": AttackStage.UNKNOWN,
        "To-Botnet-V42-TCP-Established": AttackStage.UNKNOWN,
        "C&C Channels": AttackStage.COMMAND_AND_CONTROL,
        "IRC": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC1-HTTP-Not-Encrypted": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC6-Plain-HTTP-Encrypted-Data": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC7-Custom-Encryption": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC12-HTTP-Not-Encrypted": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC16-HTTP-Not-Encrypted": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V50-1-TCP-CC20-Plain-HTTP-Encrypted-Data": (
            AttackStage.COMMAND_AND_CONTROL
        ),
        "flow=From-Botnet-V50-1-TCP-CC23-Plain-HTTP-Encrypted-Data": (
            AttackStage.COMMAND_AND_CONTROL
        ),
        "flow=From-Botnet-V42-TCP-CC53-HTTP-Not-Encrypted": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V42-TCP-CC54-Custom-Encryption": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V42-TCP-CC55-Custom-Encryption": AttackStage.COMMAND_AND_CONTROL,
        "flow=From-Botnet-V51-1-TCP-CC106-IRC-Not-Encrypted": AttackStage.COMMAND_AND_CONTROL,
        "PortScan": AttackStage.RECONNAISSANCE,
        "UDP-Scan": AttackStage.RECONNAISSANCE,
        "DDoS": AttackStage.IMPACT,
        "From-Botnet-V42-TCP-Established": AttackStage.OTHER_MALICIOUS,
        "Spam": AttackStage.OTHER_MALICIOUS,
        "ClickFraud": AttackStage.OTHER_MALICIOUS,
        "novel-label": AttackStage.UNKNOWN,
    }
    path = tmp_path / "ctu.csv"
    timestamps = [
        f"2011/08/18 15:52:{second:02d}" for second in range(len(labels_and_stages))
    ]
    pd.DataFrame(
        {
            "StartTime": timestamps,
            "Label": list(labels_and_stages),
        }
    ).to_csv(path, index=False)

    frame = read_flow_csv(path, "ctu13")

    assert frame["operational_stage"].tolist() == [
        int(stage) for stage in labels_and_stages.values()
    ]


def test_cic_adapter_drops_rows_outside_filename_capture_date(tmp_path: Path) -> None:
    path = tmp_path / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
    pd.DataFrame(
        {
            "Timestamp": ["14/02/2018 01:00:00"] * 100 + ["05/01/1970 03:01:17"],
            "Label": ["Benign"] * 101,
        }
    ).to_csv(path, index=False)

    with warnings.catch_warnings(record=True) as caught:
        frame = read_flow_csv(path, "cic_ids2018")

    assert len(frame) == 100
    assert str(frame.loc[0, "timestamp"]) == "2018-02-14 01:00:00+00:00"
    assert any("dropping 1 rows" in str(item.message) for item in caught)
