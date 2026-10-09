import pandas as pd
import pytest

from ddos_ai.data import FlowData, build_simulated, load_flows_csv
from ddos_ai.features import FEATURES
from ddos_ai.schema import ip_to_int


def test_flowdata_shapes_and_labels(small_df, small_fd):
    n = len(small_df)
    assert small_fd.X.shape == (n, len(FEATURES)) and small_fd.y.shape == (n,)
    assert small_fd.hist.shape == (n, 4)
    assert (small_fd.y == small_df["label"].to_numpy()).all()


def test_build_simulated_is_cached_and_reproducible(tmp_path):
    a = build_simulated(8, seed=3, duration=30, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.pkl"))
    b = build_simulated(8, seed=3, duration=30, cache_dir=tmp_path)           # from the cache
    c = build_simulated(8, seed=3, duration=30, cache_dir=None)               # simulated again
    assert a.equals(b) and a.equals(c)


def _capture(tmp_path, with_scenarios=False):
    rows = []
    for t in range(130):
        rows.append(("10.0.0.10", "10.0.0.1", 0, 20, 2000, 5, 1, int(t >= 100)))
        rows.append(("10.0.0.1", "10.0.0.10", 0, 20, 20000, 5, 1, 0))
    df = pd.DataFrame(rows, columns=["src", "dst", "proto", "pkts", "bytes", "flow_age", "pair_flow", "label"])
    df.insert(0, "t", [i // 2 + 1000 for i in range(len(df))])               # t does not start at zero
    if with_scenarios:
        df["scenario_id"] = df["t"] // 50
    path = tmp_path / "capture.csv"
    df.to_csv(path, index=False)
    return path


def test_load_flows_csv_converts_addresses_and_makes_blocks(tmp_path):
    df = load_flows_csv(_capture(tmp_path), block=60)
    assert df["src"].iloc[0] == ip_to_int("10.0.0.10") and df["dst"].iloc[0] == ip_to_int("10.0.0.1")
    assert sorted(df["scenario_id"].unique()) == [0, 1, 2]                    # 130 windows in blocks of 60
    assert set(df.loc[df["scenario_id"] == 0, "kind"]) == {"normal"}
    assert set(df.loc[df["scenario_id"] == 1, "kind"]) == {"attack"}          # attack windows 100-129
    assert df["attack_type"].eq("attack").sum() == df["label"].sum()
    fd = FlowData.from_frame(df)                                              # runs through the same pipeline
    assert fd.X.shape[1] == len(FEATURES)


def test_load_flows_csv_keeps_given_scenarios(tmp_path):
    df = load_flows_csv(_capture(tmp_path, with_scenarios=True))
    assert sorted(df["scenario_id"].unique()) == [20, 21, 22]


def test_load_flows_csv_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"t": [0], "src": ["10.0.0.2"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_flows_csv(path)
