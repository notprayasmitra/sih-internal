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
