"""Column names and small helpers shared by the simulator, the models and the SDN runtime.

One row is one *flow record*: what an SDN controller learns about one flow entry
(source, destination, protocol) from one poll of the switch's flow statistics.
"""
from __future__ import annotations

PROTO_TCP, PROTO_UDP, PROTO_ICMP = 0, 1, 2
PROTO_NAMES = {PROTO_TCP: "tcp", PROTO_UDP: "udp", PROTO_ICMP: "icmp"}

#: The controller polls the switch once per window of this many seconds.
POLL_INTERVAL_S = 1.0

#: Columns every flow record must have. `scenario_id` is optional outside the simulator.
RAW_COLUMNS = ["scenario_id", "t", "src", "dst", "proto", "pkts", "bytes", "flow_age", "pair_flow"]
LABEL = "label"          # 0 = normal, 1 = attack
ATTACK_TYPE = "attack_type"


def ip_to_int(address: str) -> int:
    """Dotted quad -> integer (IPv4 only)."""
    a, b, c, d = (int(part) for part in address.split("."))
    for octet in (a, b, c, d):
        if not 0 <= octet <= 255:
            raise ValueError(f"not an IPv4 address: {address!r}")
    return (a << 24) | (b << 16) | (c << 8) | d


def int_to_ip(value: int) -> str:
    """Integer -> dotted quad."""
    value = int(value)
    return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))


#: The protected server in every scenario and in the Mininet topology.
VICTIM_IP = ip_to_int("10.0.0.1")
