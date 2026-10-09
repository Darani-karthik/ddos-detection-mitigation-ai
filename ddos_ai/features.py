"""Feature engineering: per-flow statistics, switch-level context and flow histories.

The same code turns simulator output, captures from a real Mininet / Ryu run and the live
OpenFlow statistics of the controller app into model inputs, so there is one definition of
every feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import POLL_INTERVAL_S, PROTO_ICMP, PROTO_TCP, PROTO_UDP

#: Model inputs, in this order. IP addresses are deliberately *not* features.
FEATURES = [
    "pktcount",       # packets of this flow in the window
    "bytecount",      # bytes of this flow in the window
    "avg_pkt_size",   # bytes per packet
    "pktrate",        # packets per second
    "kbps",           # kilobits per second
    "flow_age",       # consecutive windows this flow entry has existed
    "pair_flow",      # 1 if the reverse flow (dst -> src) is also active: a two-way conversation
    "proto_tcp", "proto_udp", "proto_icmp",
    "sw_flows",       # switch context: active flow entries in the window
    "sw_packetins",   # switch context: new flow entries (= packet-in messages to the controller)
    "sw_pktrate",     # switch context: total packets per second
    "sw_kbps",        # switch context: total kilobits per second
    "sw_entropy",     # switch context: normalised entropy of the packet share per flow record
]
#: Heavy-tailed count / rate features that the neural models and GNB see on a log scale.
LOG_FEATURES = ["pktcount", "bytecount", "avg_pkt_size", "pktrate", "kbps", "flow_age",
                "sw_flows", "sw_packetins", "sw_pktrate", "sw_kbps"]
_LOG_IDX = [FEATURES.index(name) for name in LOG_FEATURES]


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Flow records -> feature table (one row per record, columns `FEATURES`).

    Switch-level features are computed per poll window (per `scenario_id` and `t` if the
    records come from several scenarios), so each record only uses information that the
    controller had at the end of its own window: nothing from the future.
    """
    n = len(df)
    pkts = df["pkts"].to_numpy(dtype=float)
    nbytes = df["bytes"].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=np.int64)
    sid = df["scenario_id"].to_numpy(dtype=np.int64) if "scenario_id" in df.columns else np.zeros(n, np.int64)
    window = pd.factorize(sid * 1_000_003 + t)[0]
    n_win = window.max() + 1 if n else 0

    flows = np.bincount(window, minlength=n_win).astype(float)
    new_flows = np.bincount(window, weights=(df["flow_age"].to_numpy() == 1).astype(float), minlength=n_win)
    win_pkts = np.bincount(window, weights=pkts, minlength=n_win)
    win_bytes = np.bincount(window, weights=nbytes, minlength=n_win)

    share = pkts / np.maximum(win_pkts[window], 1.0)
    plogp = np.where(share > 0, share * np.log2(np.maximum(share, 1e-12)), 0.0)
    entropy = -np.bincount(window, weights=plogp, minlength=n_win)
    entropy = np.where(flows > 1, entropy / np.log2(np.maximum(flows, 2.0)), 0.0)

    proto = df["proto"].to_numpy()
    out = pd.DataFrame({
        "pktcount": pkts,
        "bytecount": nbytes,
        "avg_pkt_size": nbytes / np.maximum(pkts, 1.0),
        "pktrate": pkts / POLL_INTERVAL_S,
        "kbps": nbytes * 8 / 1000 / POLL_INTERVAL_S,
        "flow_age": df["flow_age"].to_numpy(dtype=float),
        "pair_flow": df["pair_flow"].to_numpy(dtype=float),
        "proto_tcp": (proto == PROTO_TCP).astype(float),
        "proto_udp": (proto == PROTO_UDP).astype(float),
        "proto_icmp": (proto == PROTO_ICMP).astype(float),
        "sw_flows": flows[window],
        "sw_packetins": new_flows[window],
        "sw_pktrate": win_pkts[window] / POLL_INTERVAL_S,
        "sw_kbps": win_bytes[window] * 8 / 1000 / POLL_INTERVAL_S,
        "sw_entropy": entropy[window],
    }, index=df.index)
    return out[FEATURES]


class Preprocessor:
    """log1p on the heavy-tailed features, then standardise. Fit on training rows only."""

    def fit(self, X: np.ndarray) -> "Preprocessor":
        Z = self._log(np.asarray(X, dtype=np.float64))
        self.mean_ = Z.mean(axis=0)
        self.std_ = np.maximum(Z.std(axis=0), 1e-6)
        return self

    @staticmethod
    def _log(X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        Z[:, _LOG_IDX] = np.log1p(Z[:, _LOG_IDX])
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (self._log(np.asarray(X, dtype=np.float64)) - self.mean_) / self.std_
        return Z.astype(np.float32)


def history_index(df: pd.DataFrame, length: int) -> np.ndarray:
    """For every record, the row numbers of the same flow's previous `length - 1` windows.

    Returns an int array of shape (n, length - 1), oldest window first and the window just
    before the record last. An entry is -1 where the flow was not active in that window (it was
    new, or it went silent and expired). A spoofed-source flood has no history at all, since each
    spoofed flow is seen exactly once. Only past windows are ever referenced.
    """
    n = len(df)
    sid = df["scenario_id"].to_numpy(dtype=np.int64) if "scenario_id" in df.columns else np.zeros(n, np.int64)
    key = (pd.DataFrame({"s": sid, "a": df["src"].to_numpy(), "b": df["dst"].to_numpy(),
                         "p": df["proto"].to_numpy()})
           .groupby(["s", "a", "b", "p"], sort=False).ngroup().to_numpy())
    t = df["t"].to_numpy(dtype=np.int64)
    order = np.lexsort((t, key))
    sorted_key, sorted_t = key[order], t[order]
    hist = np.full((n, max(length - 1, 0)), -1, dtype=np.int64)
    for j in range(1, length):
        ok = np.zeros(n, dtype=bool)
        ok[j:] = (sorted_key[j:] == sorted_key[:-j]) & (sorted_t[j:] - sorted_t[:-j] == j)
        pos = np.nonzero(ok)[0]
        hist[order[pos], length - 1 - j] = order[pos - j]
    return hist


def make_sequences(Xs: np.ndarray, hist: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Model-ready sequences for the rows `idx`: shape (len(idx), length, n_features + 1).

    `Xs` are the preprocessed features of *all* rows and `hist` comes from `history_index`.
    The extra last channel is 1 for real steps and 0 for padding (a flow without history).
    """
    length = hist.shape[1] + 1
    n_feat = Xs.shape[1]
    H = hist[idx]
    valid = H >= 0
    seq = np.zeros((len(idx), length, n_feat + 1), dtype=np.float32)
    past = seq[:, : length - 1, :n_feat]
    past[valid] = Xs[H[valid]]
    seq[:, : length - 1, n_feat] = valid
    seq[:, length - 1, :n_feat] = Xs[idx]
    seq[:, length - 1, n_feat] = 1.0
    return seq
