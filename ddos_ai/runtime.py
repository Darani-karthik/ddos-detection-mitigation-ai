"""Online pieces used by the controller app: flow statistics -> records -> attack scores.

`FlowTracker` turns the cumulative counters an OpenFlow switch reports into the per-window flow
records the models were trained on. `OnlineDetector` scores them with the saved LightGBM model, which is
the model meant for deployment: it needs no TensorFlow inside the controller process and scores a window
in milliseconds. Both are plain Python so they can be tested without a switch or Ryu.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, featurize
from .schema import RAW_COLUMNS

FlowKey = tuple[int, int, int]      # (src ip, dst ip, protocol)


class FlowTracker:
    """Per-flow bookkeeping across polls.

    Call `update(t, counters)` once per poll with the *cumulative* (packets, bytes) of every flow entry, keyed by
    (src, dst, proto). It returns the records of that window: packets and bytes since the previous poll,
    how many consecutive windows the flow has been seen, and whether the reverse flow is active too.
    `add_unmatched` lets the controller report packets it handled itself (packet-ins of flows that never got a
    flow entry because they were dropped), which then show up as records in the next `update`.
    """

    def __init__(self):
        self._last: dict[FlowKey, tuple[int, int]] = {}
        self._age: dict[FlowKey, int] = {}
        self._unmatched: dict[FlowKey, list[int]] = {}

    def add_unmatched(self, src: int, dst: int, proto: int, nbytes: int) -> None:
        entry = self._unmatched.setdefault((src, dst, proto), [0, 0])
        entry[0] += 1
        entry[1] += int(nbytes)

    def update(self, t: int, counters: dict[FlowKey, tuple[int, int]]) -> pd.DataFrame:
        deltas: dict[FlowKey, tuple[int, int]] = {}
        for key, (packets, nbytes) in counters.items():
            last = self._last.get(key)
            # counters that went backwards mean the entry was removed and installed again
            d_pkts, d_bytes = (packets, nbytes) if last is None or packets < last[0] else (packets - last[0], nbytes - last[1])
            if d_pkts > 0:
                deltas[key] = (d_pkts, d_bytes)
        for key, (packets, nbytes) in self._unmatched.items():
            old = deltas.get(key, (0, 0))
            deltas[key] = (old[0] + packets, old[1] + nbytes)
        self._unmatched = {}
        self._last = dict(counters)

        # a flow that was silent in the previous window starts over at age 1
        self._age = {key: self._age.get(key, 0) + 1 for key in deltas}
        rows = [(t, src, dst, proto, d_pkts, d_bytes, self._age[(src, dst, proto)],
                 int((dst, src, proto) in deltas))
                for (src, dst, proto), (d_pkts, d_bytes) in deltas.items()]
        return pd.DataFrame(rows, columns=RAW_COLUMNS[1:])


class OnlineDetector:
    """Scores flow records with the saved LightGBM model."""

    def __init__(self, model_dir: str | Path = "models"):
        import lightgbm as lgb

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "lightgbm_threshold.json").read_text())
        if meta["features"] != FEATURES:
            raise ValueError("the saved model was trained on a different feature list; retrain it")
        self.threshold: float = float(meta["threshold"])
        self.booster = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))

    def score(self, records: pd.DataFrame) -> np.ndarray:
        """Attack probability of every record of ONE poll window."""
        if records.empty:
            return np.zeros(0)
        return self.booster.predict(featurize(records)[FEATURES].to_numpy())
