"""Memory-bounded bidirectional IPv4 flow reconstruction from PCAP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return np.nan, np.nan, np.nan, np.nan
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std()), float(array.min()), float(array.max())


@dataclass
class FlowAccumulator:
    protocol: int
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    started: float
    ended: float
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    timestamps: list[float] = field(default_factory=list)
    ttl_fwd: list[float] = field(default_factory=list)
    ttl_bwd: list[float] = field(default_factory=list)
    windows_fwd: list[float] = field(default_factory=list)
    windows_bwd: list[float] = field(default_factory=list)
    packet_lengths: list[float] = field(default_factory=list)
    payload_sizes: list[float] = field(default_factory=list)
    flags: CounterLike = field(default_factory=dict)
    fragment_count: int = 0
    retransmission_count: int = 0
    seen_tcp_segments: set[tuple[bool, int, int]] = field(default_factory=set)

    def add(self, packet: Any, timestamp: float, forward: bool) -> None:
        from scapy.layers.inet import IP, TCP, UDP

        self.ended = timestamp
        self.timestamps.append(timestamp)
        ip = packet[IP]
        packet_bytes = int(ip.len) if getattr(ip, "len", None) else len(bytes(ip))
        self.packet_lengths.append(float(packet_bytes))
        ttl = float(ip.ttl)
        if forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_bytes
            self.ttl_fwd.append(ttl)
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_bytes
            self.ttl_bwd.append(ttl)
        self.fragment_count += int(int(ip.frag) > 0 or bool(int(ip.flags) & 0x1))

        payload_size = 0
        if TCP in packet:
            tcp = packet[TCP]
            payload_size = len(bytes(tcp.payload))
            window_list = self.windows_fwd if forward else self.windows_bwd
            window_list.append(float(tcp.window))
            for flag in ("S", "A", "F", "R", "P", "U"):
                if flag in str(tcp.flags):
                    self.flags[flag] = self.flags.get(flag, 0) + 1
            if payload_size > 0:
                segment = (forward, int(tcp.seq), payload_size)
                if segment in self.seen_tcp_segments:
                    self.retransmission_count += 1
                self.seen_tcp_segments.add(segment)
        elif UDP in packet:
            payload_size = len(bytes(packet[UDP].payload))
        self.payload_sizes.append(float(payload_size))

    def to_record(self) -> dict[str, Any]:
        ordered_times = np.sort(np.asarray(self.timestamps, dtype=np.float64))
        iats = np.diff(ordered_times) if len(ordered_times) > 1 else np.asarray([])
        pkt_mean, pkt_std, pkt_min, pkt_max = _stats(self.packet_lengths)
        payload_mean, payload_std, payload_min, payload_max = _stats(self.payload_sizes)
        _, tcp_window_var_fwd, _, _ = _stats(self.windows_fwd)
        _, tcp_window_var_bwd, _, _ = _stats(self.windows_bwd)
        _, ttl_var, _, _ = _stats(self.ttl_fwd + self.ttl_bwd)
        ttl_values = self.ttl_fwd + self.ttl_bwd
        total_packets = self.fwd_packets + self.bwd_packets
        total_bytes = self.fwd_bytes + self.bwd_bytes
        return {
            "timestamp": pd.to_datetime(self.started, unit="s", utc=True),
            "duration_s": max(0.0, self.ended - self.started),
            "protocol": self.protocol,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "fwd_packets": self.fwd_packets,
            "bwd_packets": self.bwd_packets,
            "total_packets": total_packets,
            "fwd_bytes": self.fwd_bytes,
            "bwd_bytes": self.bwd_bytes,
            "total_bytes": total_bytes,
            "flow_iat_mean_s": float(iats.mean()) if len(iats) else np.nan,
            "flow_iat_std_s": float(iats.std()) if len(iats) else np.nan,
            "flow_iat_max_s": float(iats.max()) if len(iats) else np.nan,
            "syn_count": self.flags.get("S", 0),
            "ack_count": self.flags.get("A", 0),
            "fin_count": self.flags.get("F", 0),
            "rst_count": self.flags.get("R", 0),
            "psh_count": self.flags.get("P", 0),
            "urg_count": self.flags.get("U", 0),
            "packet_length_mean": pkt_mean,
            "packet_length_std": pkt_std,
            "packet_length_min": pkt_min,
            "packet_length_max": pkt_max,
            "tcp_window_fwd": np.mean(self.windows_fwd) if self.windows_fwd else np.nan,
            "tcp_window_bwd": np.mean(self.windows_bwd) if self.windows_bwd else np.nan,
            "tcp_window_var_fwd": tcp_window_var_fwd**2 if self.windows_fwd else np.nan,
            "tcp_window_var_bwd": tcp_window_var_bwd**2 if self.windows_bwd else np.nan,
            "tcp_window_count_fwd": len(self.windows_fwd),
            "tcp_window_count_bwd": len(self.windows_bwd),
            "tcp_window_sum_fwd": sum(self.windows_fwd),
            "tcp_window_sum_bwd": sum(self.windows_bwd),
            "tcp_window_sumsq_fwd": sum(value * value for value in self.windows_fwd),
            "tcp_window_sumsq_bwd": sum(value * value for value in self.windows_bwd),
            "ttl_fwd": np.mean(self.ttl_fwd) if self.ttl_fwd else np.nan,
            "ttl_bwd": np.mean(self.ttl_bwd) if self.ttl_bwd else np.nan,
            "ttl_variance": ttl_var**2 if self.ttl_fwd or self.ttl_bwd else np.nan,
            "ttl_count": len(ttl_values),
            "ttl_sum": sum(ttl_values),
            "ttl_sumsq": sum(value * value for value in ttl_values),
            "retransmission_count": self.retransmission_count,
            "ip_fragment_count": self.fragment_count,
            "payload_size_mean": payload_mean,
            "payload_size_std": payload_std,
            "payload_size_min": payload_min,
            "payload_size_max": payload_max,
            "source_label": "",
            "source_stage": "",
            "operational_stage": 8,
            "label_valid": 0.0,
        }


CounterLike = dict[str, int]


def _flow_key(
    protocol: int,
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
) -> tuple[Any, ...]:
    source = (src_ip, src_port)
    destination = (dst_ip, dst_port)
    endpoints = (source, destination) if source <= destination else (destination, source)
    return protocol, *endpoints


def extract_pcap(
    path: Path,
    *,
    tcp_inactive_timeout: float = 120.0,
    udp_inactive_timeout: float = 60.0,
    active_timeout: float = 300.0,
    packet_limit: int | None = None,
) -> pd.DataFrame:
    """Extract one row per reconstructed flow.

    Only IPv4 TCP/UDP packets are currently modeled. Labels remain unknown and
    must be attached from an audited source timeline before supervised use.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    active: dict[tuple[Any, ...], FlowAccumulator] = {}
    completed: list[dict[str, Any]] = []
    processed = 0
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if IP not in packet or (TCP not in packet and UDP not in packet):
                continue
            timestamp = float(packet.time)
            ip = packet[IP]
            transport = packet[TCP] if TCP in packet else packet[UDP]
            protocol = 6 if TCP in packet else 17
            src_ip, dst_ip = str(ip.src), str(ip.dst)
            src_port, dst_port = int(transport.sport), int(transport.dport)
            key = _flow_key(protocol, src_ip, src_port, dst_ip, dst_port)
            flow = active.get(key)
            inactive_timeout = tcp_inactive_timeout if protocol == 6 else udp_inactive_timeout
            if flow is not None and (
                timestamp - flow.ended > inactive_timeout
                or timestamp - flow.started > active_timeout
            ):
                completed.append(flow.to_record())
                flow = None
            if flow is None:
                flow = FlowAccumulator(
                    protocol, src_ip, src_port, dst_ip, dst_port, timestamp, timestamp
                )
                active[key] = flow
            forward = src_ip == flow.src_ip and src_port == flow.src_port
            flow.add(packet, timestamp, forward)
            processed += 1
            if packet_limit is not None and processed >= packet_limit:
                break
    completed.extend(flow.to_record() for flow in active.values())
    if not completed:
        raise ValueError(f"No IPv4 TCP/UDP packets found in {path}")
    return pd.DataFrame(completed).sort_values("timestamp", kind="stable").reset_index(drop=True)
