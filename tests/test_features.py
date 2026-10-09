import numpy as np
import pandas as pd

from ddos_ai.features import FEATURES, LOG_FEATURES, Preprocessor, featurize, history_index, make_sequences


def _records(rows):
    return pd.DataFrame(rows, columns=["t", "src", "dst", "proto", "pkts", "bytes", "flow_age", "pair_flow"])


def test_featurize_flow_and_switch_context():
    df = _records([
        (0, 1, 9, 0, 10, 1000, 3, 1),
        (0, 2, 9, 1, 30, 3000, 1, 0),
        (1, 1, 9, 0, 5, 500, 4, 1),
    ])
    f = featurize(df)
    assert list(f.columns) == FEATURES
    first = f.iloc[0]
    assert first["pktcount"] == 10 and first["avg_pkt_size"] == 100 and first["pktrate"] == 10
    assert first["kbps"] == 8.0                                  # 1000 B * 8 / 1000 per second
    assert first["proto_tcp"] == 1 and first["proto_udp"] == 0
    # window 0 has two records, one of them new (age 1); window 1 has one
    assert first["sw_flows"] == 2 and first["sw_packetins"] == 1 and first["sw_pktrate"] == 40
    assert f.iloc[2]["sw_flows"] == 1 and f.iloc[2]["sw_entropy"] == 0.0
    # entropy of shares 0.25 / 0.75, normalised by log2(2 flows)
    expected = -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
    assert np.isclose(first["sw_entropy"], expected)


def test_switch_features_are_computed_per_scenario_and_window():
    a = _records([(0, 1, 9, 0, 10, 100, 1, 0)])
    b = _records([(0, 2, 9, 0, 10, 100, 1, 0), (0, 3, 9, 0, 10, 100, 1, 0)])
    a["scenario_id"], b["scenario_id"] = 0, 1
    f = featurize(pd.concat([a, b], ignore_index=True))
    assert list(f["sw_flows"]) == [1, 2, 2]


def test_uniform_traffic_has_maximal_entropy():
    df = _records([(0, i, 9, 0, 5, 500, 1, 0) for i in range(8)])
    assert np.allclose(featurize(df)["sw_entropy"], 1.0)


def test_preprocessor_uses_only_training_statistics():
    rng = np.random.default_rng(0)
    train = np.abs(rng.normal(5, 2, (200, len(FEATURES))))
    pre = Preprocessor().fit(train)
    z = pre.transform(train)
    assert z.dtype == np.float32 and np.allclose(z.mean(axis=0), 0, atol=1e-4)
    shifted = pre.transform(train + 100)
    assert shifted.mean() > 1                                    # the shift is not normalised away
    assert len(LOG_FEATURES) < len(FEATURES)


def test_history_index_follows_consecutive_windows_of_one_flow():
    df = _records([
        (0, 1, 9, 0, 1, 10, 1, 0),   # row 0  flow A
        (1, 1, 9, 0, 1, 10, 2, 0),   # row 1  flow A
        (2, 1, 9, 0, 1, 10, 3, 0),   # row 2  flow A
        (4, 1, 9, 0, 1, 10, 1, 0),   # row 3  flow A after a gap: starts over
        (1, 2, 9, 0, 1, 10, 1, 0),   # row 4  another flow
        (2, 1, 9, 1, 1, 10, 1, 0),   # row 5  same hosts, other protocol: a different flow
    ])
    hist = history_index(df, length=4)
    assert hist.shape == (6, 3)
    assert list(hist[2]) == [-1, 0, 1]          # oldest first, the previous window last
    assert list(hist[1]) == [-1, -1, 0]
    assert list(hist[0]) == [-1, -1, -1]
    assert list(hist[3]) == [-1, -1, -1]        # gap: no history
    assert list(hist[4]) == [-1, -1, -1] and list(hist[5]) == [-1, -1, -1]


def test_history_never_points_to_the_future(small_fd):
    hist, df = small_fd.hist, small_fd.df
    t = df["t"].to_numpy()
    sid = df["scenario_id"].to_numpy()
    rows, cols = np.nonzero(hist >= 0)
    prev = hist[rows, cols]
    assert (t[prev] < t[rows]).all() and (sid[prev] == sid[rows]).all()


def test_make_sequences_pads_and_masks():
    Xs = np.arange(12, dtype=np.float32).reshape(4, 3)
    hist = np.array([[-1, -1], [-1, 0], [0, 1], [-1, -1]])
    seq = make_sequences(Xs, hist, np.array([2, 0]))
    assert seq.shape == (2, 3, 4)
    assert np.allclose(seq[0, 2, :3], Xs[2]) and seq[0, 2, 3] == 1       # the record itself
    assert np.allclose(seq[0, 0, :3], Xs[0]) and seq[0, 0, 3] == 1       # two windows ago
    assert np.allclose(seq[0, 1, :3], Xs[1])
    assert (seq[1, :2] == 0).all() and seq[1, 2, 3] == 1                 # no history: padding, mask 0
