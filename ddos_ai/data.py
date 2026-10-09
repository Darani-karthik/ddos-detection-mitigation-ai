"""Data sets: building, caching, loading captures and the scenario-wise split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, featurize, history_index
from .schema import ATTACK_TYPE, LABEL, RAW_COLUMNS, ip_to_int
from .simulate import plan_scenarios, simulate_scenario

#: Length of the flow history fed to the LSTM (the record's window plus the 4 windows before it).
SEQUENCE_LENGTH = 5


def build_simulated(n_scenarios: int = 120, seed: int = 42, duration: int = 90,
                    cache_dir: str | Path | None = "cache") -> pd.DataFrame:
    """Simulate (or load from cache) the flow records of `n_scenarios` scenarios."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"flows_{n_scenarios}x{duration}s_seed{seed}.pkl"
        if cache.exists():
            return pd.read_pickle(cache)
    plan = plan_scenarios(n_scenarios, seed, duration)
    kinds = {sc.scenario_id: sc.kind for sc in plan}
    df = pd.concat([simulate_scenario(sc) for sc in plan], ignore_index=True)
    df["kind"] = df["scenario_id"].map(kinds)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def load_flows_csv(path: str | Path, block: int = 60) -> pd.DataFrame:
    """Load flow records captured from a real run (see `sdn/`) in the same layout as the simulator.

    Needs the columns t, src, dst, proto, pkts, bytes, flow_age, pair_flow and label. IP addresses may
    be dotted quads or integers. Without a `scenario_id` the capture is cut into consecutive blocks of
    `block` windows, which then play the role of scenarios when splitting: the model is never tested
    on windows that sit between training windows.
    """
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS[1:] + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for col in ("src", "dst"):
        if df[col].dtype == object:
            df[col] = df[col].map(ip_to_int)
    if "scenario_id" not in df.columns:
        df["scenario_id"] = (df["t"] - df["t"].min()) // block
    if ATTACK_TYPE not in df.columns:
        df[ATTACK_TYPE] = np.where(df[LABEL] == 1, "attack", "none")
    has_attack = df.groupby("scenario_id")[LABEL].transform("max") == 1
    df["kind"] = np.where(has_attack, "attack", "normal")
    return df.sort_values(["scenario_id", "t"], kind="stable", ignore_index=True)


def split_scenarios(df: pd.DataFrame, seed: int = 0, val: float = 0.15, test: float = 0.25) -> dict[int, str]:
    """Assign whole scenarios to train / val / test, stratified by scenario kind.

    A scenario is never cut in two, so the flows of one run (and the bots of one attack) are never
    on both sides of the split.
    """
    rng = np.random.default_rng(seed)
    parts: dict[int, str] = {}
    scen = df.groupby("scenario_id")["kind"].first()
    for _, ids in scen.groupby(scen):
        ids = rng.permutation(ids.index.to_numpy())
        n = len(ids)
        n_test = max(1, round(test * n)) if n >= 3 else (1 if n == 2 else 0)
        n_val = max(1, round(val * n)) if n >= 3 else 0
        for i, sid in enumerate(ids):
            parts[int(sid)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return parts


@dataclass
class FlowData:
    """A data set prepared for modelling: features, labels, flow histories and scenario info."""
    df: pd.DataFrame
    X: np.ndarray          # (n, len(FEATURES)) raw features
    y: np.ndarray          # (n,) 0 = normal, 1 = attack
    hist: np.ndarray       # (n, SEQUENCE_LENGTH - 1) row numbers of the flow's previous windows

    @classmethod
    def from_frame(cls, df: pd.DataFrame, length: int = SEQUENCE_LENGTH) -> "FlowData":
        df = df.reset_index(drop=True)
        return cls(df=df, X=featurize(df)[FEATURES].to_numpy(dtype=np.float64),
                   y=df[LABEL].to_numpy(dtype=np.int64), hist=history_index(df, length))

    def rows(self, part: dict[int, str], name: str) -> np.ndarray:
        """Row numbers of the scenarios assigned to `name`."""
        wanted = np.array([sid for sid, p in part.items() if p == name])
        return np.nonzero(np.isin(self.df["scenario_id"].to_numpy(), wanted))[0]
