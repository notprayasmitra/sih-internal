"""Schema-aware adapters for supported flow CSV formats."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from cyberworld.labels import AttackStage

CANONICAL_COLUMNS = (
    "timestamp",
    "duration_s",
    "protocol",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
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
    "source_label",
    "source_stage",
)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


COMMON_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "starttime", "stime"),
    "protocol": ("protocol", "proto"),
    "src_ip": ("srcip", "srcaddr", "sourceip"),
    "dst_ip": ("dstip", "dstaddr", "destinationip"),
    "src_port": ("srcport", "sport", "sourceport"),
    "dst_port": ("dstport", "dport", "destinationport"),
    "fwd_packets": ("totfwdpkts", "totalfwdpacket", "totalfwdpackets", "spkts"),
    "bwd_packets": ("totbwdpkts", "totalbwdpackets", "totalbackwardpackets", "dpkts"),
    "total_packets": ("totpkts", "totalpackets", "pkts"),
    "fwd_bytes": ("totlenfwdpkts", "totallengthoffwdpacket", "sbytes", "srcbytes"),
    "bwd_bytes": ("totlenbwdpkts", "totallengthofbwdpacket", "dbytes"),
    "total_bytes": ("totbytes", "totalbytes", "bytes"),
    "flow_iat_mean_s": ("flowiatmean",),
    "flow_iat_std_s": ("flowiatstd",),
    "flow_iat_max_s": ("flowiatmax",),
    "syn_count": ("synflagcnt", "synflagcount"),
    "ack_count": ("ackflagcnt", "ackflagcount"),
    "fin_count": ("finflagcnt", "finflagcount"),
    "rst_count": ("rstflagcnt", "rstflagcount"),
    "psh_count": ("pshflagcnt", "pshflagcount"),
    "urg_count": ("urgflagcnt", "urgflagcount"),
    "packet_length_mean": ("pktlenmean", "packetlengthmean", "smeansz"),
    "packet_length_std": ("pktlenstd", "packetlengthstd"),
    "packet_length_min": ("pktlenmin", "minpacketlength", "packetlengthmin"),
    "packet_length_max": ("pktlenmax", "maxpacketlength", "packetlengthmax"),
    "tcp_window_fwd": (
        "fwdinitwinbytes",
        "fwdinitwinbyts",
        "initfwdwinbytes",
        "initfwdwinbyts",
        "swin",
    ),
    "tcp_window_bwd": (
        "bwdinitwinbytes",
        "bwdinitwinbyts",
        "initbwdwinbytes",
        "initbwdwinbyts",
        "dwin",
    ),
    "ttl_fwd": ("sttl",),
    "ttl_bwd": ("dttl",),
    "source_label": ("label", "attackcat"),
    "source_stage": ("stage",),
}


def _map_dapt(label: str, stage: str) -> AttackStage:
    value = stage.strip().lower()
    if value in {"benign", "normal"}:
        return AttackStage.NORMAL
    if "recon" in value:
        return AttackStage.RECONNAISSANCE
    if "foothold" in value:
        return AttackStage.INITIAL_ACCESS
    if "lateral" in value:
        return AttackStage.LATERAL_MOVEMENT
    if "exfil" in value:
        return AttackStage.EXFILTRATION
    return AttackStage.UNKNOWN


def _map_cic(label: str, stage: str) -> AttackStage:
    value = label.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", value)
    if value in {"benign", "normal"}:
        return AttackStage.NORMAL
    if "portscan" in value or "port scan" in value:
        return AttackStage.RECONNAISSANCE
    if "ddos" in value or value.startswith("dos") or "dos-" in value:
        return AttackStage.IMPACT
    if normalized in {"ftpbruteforce", "sshbruteforce"} or "sqlinjection" in normalized:
        return AttackStage.INITIAL_ACCESS
    if value:
        return AttackStage.OTHER_MALICIOUS
    return AttackStage.UNKNOWN


def _map_ctu(label: str, stage: str) -> AttackStage:
    value = label.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", value)
    if "fromnormal" in normalized or normalized == "normal":
        return AttackStage.NORMAL
    if (
        normalized in {"cc", "ccchannel", "ccchannels", "irc", "p2p"}
        or "commandandcontrol" in normalized
        or "commandcontrol" in normalized
        or re.search(r"(?:^|[^a-z0-9])cc\d+(?:[^a-z0-9]|$)", value)
        or re.search(r"(?:^|[^a-z0-9])irc(?:[^a-z0-9]|$)", value)
    ):
        return AttackStage.COMMAND_AND_CONTROL
    if "portscan" in normalized or "udpscan" in normalized:
        return AttackStage.RECONNAISSANCE
    if "ddos" in normalized or normalized.startswith("dos"):
        return AttackStage.IMPACT
    if "background" in normalized or normalized.startswith("to"):
        return AttackStage.UNKNOWN
    if any(
        token in normalized
        for token in ("spam", "clickfraud", "fastflux", "frombotnet")
    ) or normalized == "botnet":
        return AttackStage.OTHER_MALICIOUS
    return AttackStage.UNKNOWN


def _map_unsw(label: str, stage: str) -> AttackStage:
    value = label.strip().lower()
    if value in {"normal", "0", "benign"}:
        return AttackStage.NORMAL
    if "recon" in value:
        return AttackStage.RECONNAISSANCE
    if value == "dos":
        return AttackStage.IMPACT
    if value:
        return AttackStage.OTHER_MALICIOUS
    return AttackStage.UNKNOWN


MAPPERS: dict[str, Callable[[str, str], AttackStage]] = {
    "dapt2020": _map_dapt,
    "cic_ids2018": _map_cic,
    "ctu13": _map_ctu,
    "unsw_nb15": _map_unsw,
}


def _parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        if numeric.dropna().median() > 10_000_000_000:
            return cast(
                pd.Series,
                pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce"),
            )
        return cast(
            pd.Series,
            pd.to_datetime(numeric, unit="s", utc=True, errors="coerce"),
        )
    return cast(
        pd.Series,
        pd.to_datetime(series, utc=True, errors="coerce", dayfirst=True, format="mixed"),
    )


def _filter_cic_capture_date(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Reject timestamp corruption that would create multi-year empty trajectories."""
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", path.name)
    if match is None:
        return frame
    day, month, year = (int(value) for value in match.groups())
    expected = pd.Timestamp(year=year, month=month, day=day, tz="UTC").date()
    matches_date = frame["timestamp"].dt.date == expected
    rejected = int((~matches_date).sum())
    if rejected == 0:
        return frame
    rejection_fraction = rejected / len(frame)
    if rejection_fraction > 0.01:
        raise ValueError(
            f"{path}: {rejected}/{len(frame)} CIC rows fall outside capture date {expected}"
        )
    warnings.warn(
        f"{path}: dropping {rejected} rows outside CIC capture date {expected}",
        RuntimeWarning,
        stacklevel=2,
    )
    return frame.loc[matches_date].copy()


def read_flow_csv(path: Path, dataset_id: str, *, chunksize: int | None = None) -> pd.DataFrame:
    """Read one supported flow file into canonical columns.

    This function intentionally rejects unknown dataset identifiers. Adding a dataset
    requires an explicit mapping policy rather than accidental column matching.
    """
    if dataset_id not in MAPPERS:
        raise ValueError(f"Unsupported dataset_id {dataset_id!r}")
    if chunksize is not None:
        raise NotImplementedError("Chunk aggregation is handled by the preprocessing CLI")
    header = pd.read_csv(path, nrows=0)
    normalized = {_normalize(column): column for column in header.columns}
    selected_columns = {
        normalized[alias]
        for aliases in COMMON_ALIASES.values()
        for alias in aliases
        if alias in normalized
    }
    selected_columns.update(
        normalized[alias]
        for alias in ("flowduration", "duration", "dur")
        if alias in normalized
    )
    if not selected_columns:
        raise ValueError(f"No recognized flow columns in {path}")
    raw = pd.read_csv(path, usecols=sorted(selected_columns), low_memory=False)
    output = pd.DataFrame(index=raw.index)
    for canonical in CANONICAL_COLUMNS:
        source = next(
            (
                normalized[alias]
                for alias in COMMON_ALIASES.get(canonical, ())
                if alias in normalized
            ),
            None,
        )
        output[canonical] = raw[source] if source is not None else np.nan

    duration_alias = "duration_s"
    duration_source = next(
        (normalized[x] for x in ("flowduration", "duration", "dur") if x in normalized),
        None,
    )
    if duration_source is not None:
        duration = pd.to_numeric(raw[duration_source], errors="coerce")
        if dataset_id in {"dapt2020", "cic_ids2018"}:
            duration = duration / 1_000_000.0
        output[duration_alias] = duration

    if dataset_id in {"dapt2020", "cic_ids2018"}:
        for column in ("flow_iat_mean_s", "flow_iat_std_s", "flow_iat_max_s"):
            output[column] = pd.to_numeric(output[column], errors="coerce") / 1_000_000.0

    output["timestamp"] = _parse_timestamp(output["timestamp"])
    for column in CANONICAL_COLUMNS:
        non_numeric = {"timestamp", "protocol", "src_ip", "dst_ip", "source_label", "source_stage"}
        if column not in non_numeric:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    # CICFlowMeter uses -1 for an unavailable TCP window, not a negative window.
    for column in ("tcp_window_fwd", "tcp_window_bwd"):
        output[column] = output[column].mask(output[column] < 0)

    output["source_label"] = output["source_label"].fillna("").astype(str)
    output["source_stage"] = output["source_stage"].fillna("").astype(str)
    mapper = MAPPERS[dataset_id]
    output["operational_stage"] = [
        int(mapper(label, stage))
        for label, stage in zip(output["source_label"], output["source_stage"], strict=True)
    ]
    output["label_valid"] = (output["operational_stage"] != int(AttackStage.UNKNOWN)).astype(
        np.float32
    )
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    if dataset_id == "cic_ids2018":
        output = _filter_cic_capture_date(output, path)
    return output.reset_index(drop=True)
