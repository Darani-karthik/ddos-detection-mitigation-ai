"""Mitigation policy: turn per-flow attack scores into block rules, and replay it on recorded traffic.

Detection alone does not stop a flood. The controller has to decide *what to drop*, and a careless
policy hurts legitimate users as much as the attack does. Two mechanisms:

1. **Source blocking with hysteresis.** A source whose flow is flagged in `strikes_to_block`
   consecutive windows is blocked for `block_ttl` windows. Requiring more than one strike keeps a
   single false alarm from cutting off a legitimate client. This is the right tool against bots with
   fixed addresses (UDP / ICMP / HTTP / pulsing floods).
2. **Protect mode** against spoofed floods. When one window has at least `spoof_min_sources` flagged
   sources, blocking addresses is pointless: each spoofed address sends one packet and never again.
   Instead, for `protect_ttl` windows only *trusted* sources (seen unflagged in at least `trust_after`
   windows) are let through; every unknown source is dropped. Unknown but legitimate newcomers are the
   price, and `replay` measures it.

A decision taken after window ``t`` takes effect from window ``t + 1``: the controller needs one poll to
see the attack, so some attack traffic always gets through at the start.

The class has no dependency on NumPy-only data or on Ryu, so the same code runs in the replay below and
inside the controller app in `sdn/`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .schema import VICTIM_IP


@dataclass(frozen=True)
class MitigationConfig:
    threshold: float = 0.5        # attack score at or above which a flow record counts as flagged
    strikes_to_block: int = 2     # consecutive flagged windows before a source is blocked
    block_ttl: int = 30           # windows a block rule lives
    protect: bool = True          # enable protect mode against spoofed floods
    spoof_min_sources: int = 20   # distinct flagged sources in one window that trigger protect mode
    protect_ttl: int = 8          # windows protect mode stays on after the last trigger
    trust_after: int = 3          # unflagged windows a source needs before protect mode trusts it
    allowlist: frozenset = frozenset({VICTIM_IP})   # never blocked: above all the protected server itself


class MitigationController:
    """Stateful policy: feed it every poll window with `observe`, ask it what to drop with `is_dropped`."""

    _PURGE_EVERY, _FORGET_AFTER = 20, 60

    def __init__(self, config: MitigationConfig | None = None):
        self.config = config or MitigationConfig()
        self.strikes: dict[int, int] = {}
        self.good: dict[int, int] = {}            # source -> windows seen unflagged
        self.blocked_until: dict[int, int] = {}   # source -> first window the block no longer applies
        self.last_seen: dict[int, int] = {}
        self.protect_until = -1
        self.rules_installed = 0

    # ----------------------------------------------------------------------------- decisions
    def is_dropped(self, src: int, t: int, dst: int | None = None) -> bool:
        """Is a flow from `src` (to `dst`) dropped in window `t` under the rules decided so far?

        A flow is dead if either end is a blocked or, in protect mode, untrusted client: when the requests
        of a client are dropped, the replies to it stop as well.
        """
        for endpoint in (src, dst):
            if endpoint is None or endpoint in self.config.allowlist:
                continue
            if self.blocked_until.get(endpoint, -1) > t:
                return True
            if t <= self.protect_until and self.good.get(endpoint, 0) < self.config.trust_after:
                return True
        return False

    def observe(self, t: int, src: np.ndarray, score: np.ndarray) -> dict:
        """Update the policy with the scored flow records of window `t`.

        Returns what changed: `new_blocks` (sources to block now) and `protect` (True if protect mode
        was triggered or extended in this window).
        """
        cfg = self.config
        src = np.asarray(src)
        flagged_row = np.asarray(score) >= cfg.threshold
        keep = ~np.isin(src, list(cfg.allowlist))
        src, flagged_row = src[keep], flagged_row[keep]
        sources, inverse = np.unique(src, return_inverse=True)
        flagged = np.bincount(inverse, weights=flagged_row.astype(float), minlength=len(sources)) > 0

        new_blocks = []
        for s, is_flagged in zip(sources.tolist(), flagged.tolist()):
            self.last_seen[s] = t
            if is_flagged:
                strikes = self.strikes.get(s, 0) + 1
                self.strikes[s] = strikes
                if strikes >= cfg.strikes_to_block:
                    # a source that keeps sending flagged traffic under a block keeps it: no gap at expiry
                    if self.blocked_until.get(s, -1) <= t:
                        new_blocks.append(s)
                        self.rules_installed += 1
                    self.blocked_until[s] = t + 1 + cfg.block_ttl
            else:
                self.strikes.pop(s, None)
                self.good[s] = self.good.get(s, 0) + 1

        protect = bool(cfg.protect and flagged.sum() >= cfg.spoof_min_sources)
        if protect:
            self.protect_until = t + cfg.protect_ttl
        if t % self._PURGE_EVERY == 0:
            self._purge(t)
        return {"new_blocks": new_blocks, "protect": protect}

    def _purge(self, t: int) -> None:
        """Forget sources not seen for a while (spoofed addresses would otherwise pile up)."""
        stale = [s for s, seen in self.last_seen.items() if t - seen > self._FORGET_AFTER
                 and self.blocked_until.get(s, -1) <= t and self.good.get(s, 0) < self.config.trust_after]
        for s in stale:
            for table in (self.last_seen, self.strikes, self.good, self.blocked_until):
                table.pop(s, None)


# --------------------------------------------------------------------------------- policies
def policies(threshold: float) -> dict[str, MitigationConfig]:
    """The policies compared in the README: a naive one, blocking only, and the full policy."""
    full = MitigationConfig(threshold=threshold)
    return {
        "naive (block on first flag)": replace(full, strikes_to_block=1, protect=False),
        "block sources (2 strikes)": replace(full, protect=False),
        "block + protect mode": full,
    }


# ----------------------------------------------------------------------------------- replay
def replay(df: pd.DataFrame, score: np.ndarray, config: MitigationConfig) -> np.ndarray:
    """Run the policy over recorded traffic and return which flow records it would have dropped.

    One fresh controller per scenario. At window ``t`` the controller first applies the rules decided
    in earlier windows, then sees the window's scores. Simplification: dropped traffic keeps showing up in
    the statistics, as it does for OpenFlow drop rules (their counters keep counting matched packets).
    """
    sid = df["scenario_id"].to_numpy()
    t = df["t"].to_numpy()
    src = df["src"].to_numpy()
    dst = df["dst"].to_numpy()
    score = np.asarray(score)
    dropped = np.zeros(len(df), dtype=bool)

    order = np.lexsort((t, sid))
    boundaries = np.nonzero((np.diff(sid[order]) != 0) | (np.diff(t[order]) != 0))[0] + 1
    controller, current = None, None
    for rows in np.split(order, boundaries):
        if sid[rows[0]] != current:
            current, controller = sid[rows[0]], MitigationController(config)
        window = int(t[rows[0]])
        dropped[rows] = [controller.is_dropped(s, window, d)
                         for s, d in zip(src[rows].tolist(), dst[rows].tolist())]
        controller.observe(window, src[rows], score[rows])
    return dropped


def summarize_replay(df: pd.DataFrame, dropped: np.ndarray) -> dict:
    """Outcome of a replay: attack traffic stopped, legitimate traffic lost, and how fast."""
    df = df.reset_index(drop=True)
    pkts = df["pkts"].to_numpy(dtype=float)
    attack = df["label"].to_numpy() == 1
    to_victim = (df["dst"].to_numpy() == VICTIM_IP)

    def share(mask_num, mask_den):
        den = pkts[mask_den].sum()
        return float(pkts[mask_num].sum() / den) if den else float("nan")

    out = {
        "attack_packets_dropped": share(attack & dropped, attack),
        "legit_packets_dropped": share(~attack & dropped, ~attack),
        "legit_flows_dropped": float((~attack & dropped).sum() / max((~attack).sum(), 1)),
    }
    flash = (df["kind"] == "flash").to_numpy() if "kind" in df else np.zeros(len(df), bool)
    out["legit_packets_dropped_flash"] = share(~attack & dropped & flash, ~attack & flash) if flash.any() else float("nan")

    # load on the protected server: packets per second arriving at it, before and after mitigation
    load_before, load_after, times_to_mitigate = [], [], {}
    for sid, scenario in df.groupby("scenario_id"):
        victim = to_victim[scenario.index]
        per_window_before = pd.Series(pkts[scenario.index][victim]).groupby(scenario["t"].to_numpy()[victim]).sum()
        keep = ~dropped[scenario.index][victim]
        per_window_after = pd.Series(pkts[scenario.index][victim][keep]).groupby(
            scenario["t"].to_numpy()[victim][keep]).sum()
        if attack[scenario.index].any():
            load_before.append(per_window_before.max())
            load_after.append(per_window_after.max() if len(per_window_after) else 0.0)
            kind = scenario["kind"].iloc[0] if "kind" in scenario else "attack"
            times_to_mitigate.setdefault(kind, []).append(_time_to_mitigate(scenario, attack, dropped, pkts))
    out["peak_victim_load_reduction"] = float(1 - np.sum(load_after) / np.sum(load_before)) if load_before else float("nan")
    out["time_to_mitigate"] = {
        kind: (float(np.nanmedian(values)) if not np.all(np.isnan(values)) else float("nan"))
        for kind, values in times_to_mitigate.items()}
    flat = [v for values in times_to_mitigate.values() for v in values]
    out["time_to_mitigate_median"] = float(np.nanmedian(flat)) if flat and not np.all(np.isnan(flat)) else float("nan")
    out["attack_scenarios_never_mitigated"] = int(np.sum(np.isnan(flat))) if flat else 0
    return out


def _time_to_mitigate(scenario: pd.DataFrame, attack: np.ndarray, dropped: np.ndarray, pkts: np.ndarray,
                      target: float = 0.9) -> float:
    """Windows from the first attack packet until a window has >= 90% of its attack packets dropped."""
    idx = scenario.index.to_numpy()
    is_attack = attack[idx]
    t = scenario["t"].to_numpy()
    start = t[is_attack].min()
    for window in np.unique(t[is_attack]):
        in_window = is_attack & (t == window)
        total = pkts[idx][in_window].sum()
        if total > 0 and pkts[idx][in_window & dropped[idx]].sum() / total >= target:
            return float(window - start)
    return float("nan")


def victim_timeline(df: pd.DataFrame, dropped: np.ndarray) -> pd.DataFrame:
    """Packets per second arriving at the server in one scenario: attack and legitimate, before / after."""
    df = df.reset_index(drop=True)
    victim = df["dst"].to_numpy() == VICTIM_IP
    d = pd.DataFrame({"t": df["t"], "pkts": df["pkts"], "attack": df["label"] == 1,
                      "delivered": ~dropped})[victim]
    out = pd.DataFrame({
        "attack_offered": d[d.attack].groupby("t").pkts.sum(),
        "legit_offered": d[~d.attack].groupby("t").pkts.sum(),
        "attack_delivered": d[d.attack & d.delivered].groupby("t").pkts.sum(),
        "legit_delivered": d[~d.attack & d.delivered].groupby("t").pkts.sum(),
    })
    return out.reindex(range(int(df["t"].max()) + 1)).fillna(0.0)
