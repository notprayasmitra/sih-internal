from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import wrpcap

from cyberworld.pcap import extract_pcap


def test_pcap_extractor_is_bidirectional_and_counts_retransmission(tmp_path: Path) -> None:
    path = tmp_path / "sample.pcap"
    packets = [
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=64)
        / TCP(sport=1234, dport=80, flags="S", seq=1, window=1024),
        IP(src="10.0.0.2", dst="10.0.0.1", ttl=63)
        / TCP(sport=80, dport=1234, flags="SA", seq=2, window=2048),
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=64)
        / TCP(sport=1234, dport=80, flags="PA", seq=3)
        / Raw(b"hello"),
        IP(src="10.0.0.1", dst="10.0.0.2", ttl=64)
        / TCP(sport=1234, dport=80, flags="PA", seq=3)
        / Raw(b"hello"),
        IP(src="10.0.0.3", dst="10.0.0.4") / UDP(sport=53, dport=5353) / Raw(b"dns"),
    ]
    for index, packet in enumerate(packets):
        packet.time = 1_700_000_000 + index * 0.1
    wrpcap(str(path), packets)
    frame = extract_pcap(path)
    assert len(frame) == 2
    tcp = frame.loc[frame.protocol == 6].iloc[0]
    assert tcp.fwd_packets == 3
    assert tcp.bwd_packets == 1
    assert tcp.retransmission_count == 1
    assert tcp.payload_size_max == 5
