"""Flow-level simulator of one SDN edge switch under normal load, flash crowds and DDoS floods.

The simulator plays the part of the Mininet + Ryu set-up in the original course project: every
second the "controller" polls the switch and sees one record per active flow entry. It does
not simulate packets; each flow-window is drawn from a simple stochastic model instead. All
numbers it produces are therefore properties of *these models*, not measurements of real
traffic (see the README for what that means for the results).

Traffic that is generated
-------------------------
* **Normal**: clients talk to one protected server (10.0.0.1) in sessions: web browsing, bulk
  download, upload, one-way UDP media, DNS-style request / response and ping. Each session is
  a pair of flows (client -> server and server -> client) that persist for a while. A few
  "heavy hitter" hosts transfer at rates comparable to a modest flood.
* **Flash crowd** (still normal!): a surge of legitimate traffic, with many new short-lived
  visitors. It looks like an attack in aggregate and is what makes false positives costly.
* **Attacks** aimed at the server:
  ``syn_spoof``  TCP SYN flood from random spoofed source addresses (one new flow per source),
  ``udp``        UDP flood from a few bots with fixed addresses,
  ``icmp``       ICMP (ping) flood from a few bots,
  ``http``       application-layer flood: TCP requests from bots at a moderate rate, with normal-looking
                 packet sizes and replies, close to a flash crowd,
  ``pulse``      low-rate pulsing flood: bots send short bursts with silent gaps in between,
  ``mixed``      two of the above at the same time.
  Attacks ramp up over a few seconds, like real botnets do.

A flow's age is the number of consecutive windows in which it carried traffic; a window of silence
resets it (the flow entry expires), exactly as the controller-side `FlowTracker` counts it.
Everything is deterministic given the scenario seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import (ATTACK_TYPE, LABEL, PROTO_ICMP, PROTO_TCP, PROTO_UDP, VICTIM_IP, ip_to_int)

ATTACK_TYPES = ("syn_spoof", "udp", "icmp", "http", "pulse")
KINDS = ("normal", "flash") + ATTACK_TYPES + ("mixed",)

_CLIENT_BASE = ip_to_int("10.0.0.10")      # legitimate clients: 10.0.0.10, 10.0.0.11, ...
_VISITOR_BASE = ip_to_int("10.0.2.0")      # short-lived flash-crowd visitors
_BOT_BASES = (ip_to_int("10.0.1.0"), ip_to_int("10.0.3.0"))   # one block per simultaneous attack
_SPOOF_LOW, _SPOOF_HIGH = ip_to_int("11.0.0.0"), ip_to_int("223.255.255.255")


@dataclass(frozen=True)
class Scenario:
    """One simulated run of the network, 1 poll per second."""
    scenario_id: int
    kind: str                   # one of KINDS
    seed: int
    duration: int = 90          # seconds (= windows)
    flash_crowd: bool = False   # add a legitimate surge (always on for kind == "flash")

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown scenario kind {self.kind!r}")


# --------------------------------------------------------------------------- normal traffic
class _Flow:
    """One direction of a session: a flow entry with a mean rate and packet size."""
    __slots__ = ("src", "dst", "proto", "rate", "size", "size_sd", "pair", "age")

    def __init__(self, src, dst, proto, rate, size, size_sd, pair):
        self.src, self.dst, self.proto = src, dst, proto
        self.rate, self.size, self.size_sd, self.pair = rate, size, size_sd, pair
        self.age = 0


_PROFILES = ("web", "bulk", "upload", "udp_up", "dns", "ping")
_PROFILE_P = (0.52, 0.12, 0.08, 0.08, 0.12, 0.08)


def _make_session(rng, host: int, profile: str, rate: float) -> list[_Flow]:
    s = VICTIM_IP
    if profile == "web":
        return [_Flow(host, s, PROTO_TCP, rate * 0.45, 140, 60, 1),
                _Flow(s, host, PROTO_TCP, rate, 1050, 300, 1)]
    if profile == "bulk":
        return [_Flow(host, s, PROTO_TCP, rate * 0.3, 66, 10, 1),
                _Flow(s, host, PROTO_TCP, rate * 2.2, 1400, 60, 1)]
    if profile == "upload":
        return [_Flow(host, s, PROTO_TCP, rate * 1.6, 1380, 100, 1),
                _Flow(s, host, PROTO_TCP, rate * 0.6, 66, 10, 1)]
    if profile == "udp_up":      # one-way media stream towards the server
        return [_Flow(host, s, PROTO_UDP, rate * 0.8, 700, 250, 0)]
    if profile == "dns":
        r = max(0.8, rate * 0.08)
        return [_Flow(host, s, PROTO_UDP, r, 80, 15, 1), _Flow(s, host, PROTO_UDP, r, 180, 40, 1)]
    r = float(rng.uniform(1.0, 3.0))   # ping
    return [_Flow(host, s, PROTO_ICMP, r, 84, 2, 1), _Flow(s, host, PROTO_ICMP, r, 84, 2, 1)]


def _emit(rng, flow: _Flow, boost: float = 1.0):
    """One window of a flow: a (src, dst, proto, pkts, bytes, age, pair) tuple, or None if idle."""
    n = int(rng.poisson(flow.rate * boost * np.exp(rng.normal(0.0, 0.35))))
    if n == 0:
        flow.age = 0              # silent for a window: the entry expires and the flow starts over
        return None
    flow.age += 1
    size = float(np.clip(rng.normal(flow.size, flow.size_sd), 40, 1500))
    return (flow.src, flow.dst, flow.proto, n, int(n * size), flow.age, flow.pair)


def _simulate_normal(rng, duration: int, flash: tuple | None, rows: list):
    """Append normal-traffic rows (t, src, dst, proto, pkts, bytes, age, pair)."""
    n_hosts = int(rng.integers(10, 26))
    hosts = [_CLIENT_BASE + i for i in range(n_hosts)]
    profile = rng.choice(len(_PROFILES), size=n_hosts, p=_PROFILE_P)
    base_rate = np.clip(rng.lognormal(np.log(18.0), 0.7, n_hosts), 3.0, 120.0)
    heavy = rng.random(n_hosts) < 0.08            # power users: transfers as fast as a modest flood
    base_rate = np.where(heavy, np.minimum(base_rate * rng.uniform(3.0, 8.0, n_hosts), 400.0), base_rate)
    p_start = rng.uniform(0.06, 0.25, n_hosts)
    sessions: list[list[_Flow] | None] = [None] * n_hosts
    left = [0] * n_hosts
    visitors: list[list] = []

    for t in range(duration):
        boost_rate, boost_start, visitor_rate = 1.0, 1.0, 0.0
        if flash is not None:
            f0, f1, ramp_len, m_rate, m_start, visitor_rate_peak = flash
            if f0 <= t < f1:
                ramp = min(1.0, (t - f0 + 1) / ramp_len)
                boost_rate = 1.0 + (m_rate - 1.0) * ramp
                boost_start = 1.0 + (m_start - 1.0) * ramp
                visitor_rate = visitor_rate_peak * ramp

        for h in range(n_hosts):
            if left[h] <= 0:
                if rng.random() >= min(1.0, p_start[h] * boost_start):
                    sessions[h] = None
                    continue
                sessions[h] = _make_session(rng, hosts[h], _PROFILES[profile[h]], float(base_rate[h]))
                left[h] = int(rng.geometric(1 / 14))
            boost = boost_rate if _PROFILES[profile[h]] in ("web", "bulk") else 1.0
            for flow in sessions[h]:
                row = _emit(rng, flow, boost)
                if row is not None:
                    rows.append((t,) + row)
            left[h] -= 1

        for _ in range(int(rng.poisson(visitor_rate))):
            host = _VISITOR_BASE + int(rng.integers(1, 60000))
            rate = float(np.clip(rng.lognormal(np.log(10.0), 0.5), 2.0, 60.0))
            visitors.append([_make_session(rng, host, "web", rate), int(rng.integers(1, 4))])
        alive = []
        for session in visitors:
            for flow in session[0]:
                row = _emit(rng, flow)
                if row is not None:
                    rows.append((t,) + row)
            session[1] -= 1
            if session[1] > 0:
                alive.append(session)
        visitors = alive


# --------------------------------------------------------------------------------- attacks
class _Attack:
    """Common bookkeeping: an attack window with a ramp-up."""
    label = ""

    def __init__(self, rng, start: int, end: int):
        self.start, self.end = start, end
        self.ramp_len = int(rng.integers(3, 9))

    def intensity(self, t: int) -> float:
        if not self.start <= t < self.end:
            return 0.0
        return min(1.0, (t - self.start + 1) / self.ramp_len)


class _SynSpoof(_Attack):
    label = "syn_spoof"

    def __init__(self, rng, start, end, bot_base):
        super().__init__(rng, start, end)
        self.rate = float(rng.uniform(30, 150))      # new spoofed flows per second at full strength

    def emit(self, rng, t):
        k = self.intensity(t)
        if k == 0.0:
            return None
        n = int(rng.poisson(self.rate * k))
        if n == 0:
            return None
        src = rng.integers(_SPOOF_LOW, _SPOOF_HIGH, n)
        pkts = 1 + rng.poisson(0.5, n)
        size = rng.integers(40, 67, n)
        return (src, np.full(n, VICTIM_IP), np.full(n, PROTO_TCP), pkts, pkts * size,
                np.ones(n, dtype=int), np.zeros(n, dtype=int))


class _FixedBots(_Attack):
    """Bots with fixed addresses, one persistent flow each (udp / icmp / http / pulse)."""

    def __init__(self, rng, start, end, bot_base, kind):
        super().__init__(rng, start, end)
        self.label = kind
        k = int(rng.integers(3, 11))
        self.src = bot_base + 1 + np.arange(k)
        lo, hi = {"udp": (60, 1500), "icmp": (40, 1000), "http": (20, 200), "pulse": (100, 1200)}[kind]
        self.rate = np.exp(rng.uniform(np.log(lo), np.log(hi), k))
        if kind == "udp":
            self.proto, self.pair = PROTO_UDP, 0
            self.size, self.size_sd = rng.uniform(300, 1400, k), 80.0
        elif kind == "icmp":
            self.proto, self.pair = PROTO_ICMP, int(rng.random() < 0.5)
            self.size, self.size_sd = rng.uniform(64, 1024, k), 5.0
        elif kind == "http":
            self.proto, self.pair = PROTO_TCP, 1
            self.size, self.size_sd = rng.uniform(150, 500, k), 120.0
        else:                                    # pulse
            self.proto, self.pair = (PROTO_UDP if rng.random() < 0.7 else PROTO_TCP), 0
            self.size, self.size_sd = rng.uniform(60, 1200, k), 80.0
            self.on, self.off = int(rng.integers(2, 6)), int(rng.integers(4, 11))
            synced = rng.random() < 0.5
            self.phase = np.zeros(k, dtype=int) if synced else rng.integers(0, self.on + self.off, k)
        self.age = np.zeros(k, dtype=int)
        self.was_active = np.zeros(k, dtype=bool)

    def _active(self, t) -> np.ndarray:
        if self.label != "pulse":
            return np.ones(len(self.src), dtype=bool)
        return ((t - self.start + self.phase) % (self.on + self.off)) < self.on

    def emit(self, rng, t):
        k = self.intensity(t)
        if k == 0.0:
            return None
        active = self._active(t)
        # a flow that was silent for a window has expired from the flow table and starts over
        self.age = np.where(active, np.where(self.was_active, self.age + 1, 1), 0)
        self.was_active = active
        lam = self.rate * k * np.exp(rng.normal(0.0, 0.25, len(self.src)))
        pkts = rng.poisson(lam)
        keep = active & (pkts > 0)
        if not keep.any():
            return None
        size = np.clip(rng.normal(self.size, self.size_sd), 40, 1500)
        n = int(keep.sum())
        return (self.src[keep], np.full(n, VICTIM_IP), np.full(n, self.proto), pkts[keep],
                (pkts[keep] * size[keep]).astype(int), self.age[keep], np.full(n, self.pair))


def _make_attack(rng, kind: str, start: int, end: int, bot_base: int):
    if kind == "syn_spoof":
        return _SynSpoof(rng, start, end, bot_base)
    return _FixedBots(rng, start, end, bot_base, kind)


# --------------------------------------------------------------------------------- scenario
def simulate_scenario(sc: Scenario) -> pd.DataFrame:
    """Simulate one scenario and return its flow records (one row per flow and window)."""
    rng = np.random.default_rng(sc.seed)
    duration = sc.duration

    flash = None
    if sc.flash_crowd or sc.kind == "flash":
        f0 = int(rng.integers(25, 45))
        length = int(rng.integers(20, 36))
        flash = (f0, min(duration - 2, f0 + length), int(rng.integers(4, 9)),
                 float(rng.uniform(2.5, 6.0)), float(rng.uniform(2.0, 4.0)), float(rng.uniform(15, 70)))

    attacks = []
    if sc.kind in ATTACK_TYPES or sc.kind == "mixed":
        kinds = [sc.kind] if sc.kind != "mixed" else [str(k) for k in rng.choice(ATTACK_TYPES, 2, replace=False)]
        start = int(rng.integers(25, 41))
        for i, kind in enumerate(kinds):
            a_start = start if i == 0 else int(np.clip(start + rng.integers(-8, 9), 15, 45))
            a_end = min(duration - 3, a_start + int(rng.integers(35, 51)))
            attacks.append(_make_attack(rng, kind, a_start, a_end, _BOT_BASES[i]))

    rows: list[tuple] = []
    _simulate_normal(rng, duration, flash, rows)
    cols = ["t", "src", "dst", "proto", "pkts", "bytes", "flow_age", "pair_flow"]
    normal = pd.DataFrame(rows, columns=cols)
    normal[LABEL] = 0
    normal[ATTACK_TYPE] = "none"
    frames = [normal]

    for t in range(duration):
        for attack in attacks:
            chunk = attack.emit(rng, t)
            if chunk is None:
                continue
            frame = pd.DataFrame(dict(zip(cols[1:], chunk)))
            frame.insert(0, "t", t)
            frame[LABEL] = 1
            frame[ATTACK_TYPE] = attack.label
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("t", kind="stable", ignore_index=True)
    df.insert(0, "scenario_id", sc.scenario_id)
    df = df.astype({"src": "int64", "dst": "int64", "proto": "int8", "pkts": "int64", "bytes": "int64",
                    "flow_age": "int32", "pair_flow": "int8", "t": "int32", LABEL: "int8"})
    df["scenario_id"] = df["scenario_id"].astype("int32")
    return df


# ----------------------------------------------------------------------------- the data set
#: How many scenarios of each kind per 120 (scaled for other sizes): normal traffic, then attacks.
_MIX = {"normal": 16, "flash": 8, "syn_spoof": 18, "udp": 18, "icmp": 18, "http": 18, "pulse": 12, "mixed": 12}


def plan_scenarios(n_scenarios: int = 120, seed: int = 42, duration: int = 90) -> list[Scenario]:
    """The list of scenarios in a data set: a fixed mix of kinds, each with its own seed.

    About 30% of the attack scenarios also contain a flash crowd, so that legitimate surges and
    attacks overlap in time.
    """
    total = sum(_MIX.values())
    counts = {kind: max(2, round(n * n_scenarios / total)) for kind, n in _MIX.items()}
    plan_rng = np.random.default_rng(seed)
    plan, sid = [], 0
    for kind, count in counts.items():
        for _ in range(count):
            flash = kind == "flash" or (kind in ATTACK_TYPES + ("mixed",) and plan_rng.random() < 0.3)
            plan.append(Scenario(sid, kind, seed=int(seed) * 100_003 + sid, duration=duration,
                                 flash_crowd=bool(flash)))
            sid += 1
    return plan


def main(argv=None) -> None:
    """Write a few simulated scenarios as a CSV in the layout `load_flows_csv` reads (for inspection)."""
    import argparse

    from .schema import int_to_ip

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--out", default="data/sample_flows.csv")
    ap.add_argument("--kinds", nargs="+", default=["normal", "syn_spoof", "udp", "http"], choices=KINDS)
    ap.add_argument("--duration", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    frames = [simulate_scenario(Scenario(i, kind, seed=args.seed * 100_003 + i, duration=args.duration))
              for i, kind in enumerate(args.kinds)]
    df = pd.concat(frames, ignore_index=True)
    for column in ("src", "dst"):
        df[column] = df[column].map(int_to_ip)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} flow records of {len(frames)} scenarios to {args.out}")


if __name__ == "__main__":
    main()
