import json

import numpy as np
import pandas as pd
import pytest

from ddos_ai.features import FEATURES
from ddos_ai.labeling import label_capture
from ddos_ai.runtime import FlowTracker
from ddos_ai.schema import PROTO_TCP, PROTO_UDP, VICTIM_IP, ip_to_int


def test_tracker_reports_deltas_since_the_last_poll():
    tracker = FlowTracker()
    key = (1, 9, PROTO_TCP)
    first = tracker.update(0, {key: (100, 10_000)})
    assert first.iloc[0][["pkts", "bytes", "flow_age"]].tolist() == [100, 10_000, 1]
    second = tracker.update(1, {key: (130, 13_500)})
    assert second.iloc[0][["pkts", "bytes", "flow_age"]].tolist() == [30, 3_500, 2]


def test_tracker_skips_idle_flows_and_restarts_their_age():
    tracker = FlowTracker()
    key = (1, 9, PROTO_TCP)
    tracker.update(0, {key: (10, 1000)})
    idle = tracker.update(1, {key: (10, 1000)})            # entry still in the table, no new packets
    assert idle.empty
    back = tracker.update(2, {key: (15, 1500)})
    assert back.iloc[0]["flow_age"] == 1 and back.iloc[0]["pkts"] == 5


def test_tracker_handles_a_counter_reset():
    tracker = FlowTracker()
    key = (1, 9, PROTO_TCP)
    tracker.update(0, {key: (500, 50_000)})
    again = tracker.update(1, {key: (20, 2_000)})          # the entry was removed and installed again
    assert again.iloc[0]["pkts"] == 20 and again.iloc[0]["flow_age"] == 2


def test_tracker_marks_two_way_conversations():
    tracker = FlowTracker()
    rec = tracker.update(0, {(1, 9, PROTO_TCP): (5, 500), (9, 1, PROTO_TCP): (5, 5000), (2, 9, PROTO_UDP): (5, 500)})
    pair = dict(zip(zip(rec["src"], rec["dst"]), rec["pair_flow"]))
    assert pair[(1, 9)] == 1 and pair[(9, 1)] == 1 and pair[(2, 9)] == 0


def test_tracker_turns_controller_handled_packets_into_records():
    tracker = FlowTracker()
    tracker.add_unmatched(7, 9, PROTO_TCP, 60)
    tracker.add_unmatched(7, 9, PROTO_TCP, 60)
    rec = tracker.update(0, {})
    assert rec.iloc[0][["src", "pkts", "bytes"]].tolist() == [7, 2, 120]
    assert tracker.update(1, {}).empty                      # reported once


def test_label_capture_marks_unknown_sources_towards_the_victim_during_the_attack():
    legit = ["10.0.0.10", "10.0.0.11"]
    rec = pd.DataFrame({
        "t":    [1, 5, 5, 5, 5, 9],
        "src":  [ip_to_int("10.0.0.10"), ip_to_int("10.0.0.10"), ip_to_int("66.1.2.3"), VICTIM_IP,
                 ip_to_int("66.1.2.3"), ip_to_int("66.1.2.3")],
        "dst":  [VICTIM_IP, VICTIM_IP, VICTIM_IP, ip_to_int("10.0.0.10"), ip_to_int("10.0.0.99"), VICTIM_IP],
    })
    out = label_capture(rec, legit, attack_start=3, attack_end=8, attack_name="syn_spoof")
    assert out["label"].tolist() == [0, 0, 1, 0, 0, 0]
    assert out["attack_type"].tolist() == ["none", "none", "syn_spoof", "none", "none", "none"]


lightgbm = pytest.importorskip("lightgbm")


def test_online_detector_matches_the_training_pipeline(small_df, small_fd, tmp_path):
    from ddos_ai.data import split_scenarios
    from ddos_ai.evaluation import pick_threshold
    from ddos_ai.models import fit_lightgbm, save_lightgbm
    from ddos_ai.runtime import OnlineDetector

    part = split_scenarios(small_df, 0)
    train, val, test = (small_fd.rows(part, n) for n in ("train", "val", "test"))
    scorer = fit_lightgbm(small_fd, train, val, 0)
    save_lightgbm(scorer, pick_threshold(small_fd.y[val], scorer.score(val)), tmp_path)
    assert json.loads((tmp_path / "lightgbm_threshold.json").read_text())["features"] == FEATURES

    detector = OnlineDetector(tmp_path)
    # a single poll window, scored on its own, as the controller does it
    sid = small_df["scenario_id"].iloc[test[0]]
    window = small_df[(small_df["scenario_id"] == sid) & (small_df["t"] == 30)]
    expected = scorer.score(window.index.to_numpy())
    assert np.allclose(detector.score(window), expected, atol=1e-9)
    assert detector.score(window.iloc[0:0]).size == 0


def test_labeling_cli_round_trip(tmp_path):
    from ddos_ai.data import load_flows_csv
    from ddos_ai.labeling import main

    rows = []
    for t in range(10):
        ts = 1000.0 + t
        rows.append((ts, t, "10.0.0.10", "10.0.0.1", 0, 10, 1000, 3, 1))
        if 4 <= t < 8:
            rows.append((ts, t, "66.6.6.6", "10.0.0.1", 0, 1, 60, 1, 0))
    capture = tmp_path / "capture.csv"
    pd.DataFrame(rows, columns=["ts", "t", "src", "dst", "proto", "pkts", "bytes", "flow_age", "pair_flow"]).to_csv(
        capture, index=False)
    meta = tmp_path / "experiment.json"
    meta.write_text(json.dumps({"attack": "syn_spoof", "victim": "10.0.0.1", "legit_hosts": ["10.0.0.10"],
                                "attack_start": 1004.0, "attack_end": 1007.5}))
    out = tmp_path / "labeled.csv"
    main([str(capture), str(meta), str(out), "--scenario-id", "5"])
    df = load_flows_csv(out)
    assert set(df["scenario_id"]) == {5}
    assert df["label"].sum() == 4 and (df.loc[df["label"] == 1, "attack_type"] == "syn_spoof").all()
    assert (df.loc[df["src"] == ip_to_int("10.0.0.10"), "label"] == 0).all()
