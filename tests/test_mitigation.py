import numpy as np
import pandas as pd

from ddos_ai.mitigation import MitigationConfig, MitigationController, policies, replay, summarize_replay
from ddos_ai.schema import VICTIM_IP

BOT, CLIENT = 500, 600


def _observe(ctl, t, srcs, scores):
    return ctl.observe(t, np.array(srcs), np.array(scores, dtype=float))


def test_block_needs_consecutive_strikes_and_applies_from_the_next_window():
    ctl = MitigationController(MitigationConfig(strikes_to_block=2, block_ttl=5))
    # as in `replay`: ask about window t first, then show the controller what window t looked like
    assert not ctl.is_dropped(BOT, 0)
    _observe(ctl, 0, [BOT], [0.9])
    assert not ctl.is_dropped(BOT, 1)                      # one strike is not enough
    out = _observe(ctl, 1, [BOT], [0.9])
    assert out["new_blocks"] == [BOT]
    assert all(ctl.is_dropped(BOT, t) for t in range(2, 7))   # the decision after window 1 acts from window 2 on
    assert not ctl.is_dropped(BOT, 7)                      # expired after block_ttl windows


def test_a_single_false_alarm_does_not_block_a_client():
    ctl = MitigationController(MitigationConfig(strikes_to_block=2))
    _observe(ctl, 0, [CLIENT], [0.9])
    _observe(ctl, 1, [CLIENT], [0.1])                     # back to normal: the strike counter resets
    _observe(ctl, 2, [CLIENT], [0.9])
    assert not any(ctl.is_dropped(CLIENT, t) for t in range(0, 20))


def test_block_is_renewed_while_the_source_keeps_flagging():
    ctl = MitigationController(MitigationConfig(strikes_to_block=2, block_ttl=3))
    for t in range(10):
        _observe(ctl, t, [BOT], [0.9])
    assert all(ctl.is_dropped(BOT, t) for t in range(2, 11))        # no gap when the first block would expire
    assert ctl.rules_installed == 1                                   # one rule, renewed


def test_the_protected_server_is_never_blocked():
    ctl = MitigationController(MitigationConfig(strikes_to_block=1))
    for t in range(5):
        _observe(ctl, t, [VICTIM_IP], [0.99])
    assert not any(ctl.is_dropped(VICTIM_IP, t) for t in range(10))
    assert ctl.rules_installed == 0


def test_a_flow_to_a_blocked_client_is_dropped_too():
    ctl = MitigationController(MitigationConfig(strikes_to_block=1, block_ttl=5))
    _observe(ctl, 0, [CLIENT], [0.9])
    assert ctl.is_dropped(CLIENT, 1, VICTIM_IP)            # its requests
    assert ctl.is_dropped(VICTIM_IP, 1, CLIENT)            # the server's replies to it
    assert not ctl.is_dropped(VICTIM_IP, 1, 777)


def test_protect_mode_triggers_on_many_flagged_sources_and_only_lets_trusted_sources_through():
    cfg = MitigationConfig(spoof_min_sources=5, protect_ttl=4, trust_after=2, strikes_to_block=2)
    ctl = MitigationController(cfg)
    _observe(ctl, 0, [CLIENT], [0.0])
    _observe(ctl, 1, [CLIENT], [0.0])                      # CLIENT is now trusted (seen unflagged twice)
    out = _observe(ctl, 2, list(range(1000, 1010)), [0.9] * 10)
    assert out["protect"]
    assert not ctl.is_dropped(CLIENT, 3)                   # known client: let through
    assert ctl.is_dropped(4242, 3)                         # unknown source: dropped
    assert ctl.is_dropped(1003, 3)
    assert not ctl.is_dropped(4242, 3 + cfg.protect_ttl + 1)   # protect mode runs out
    assert ctl.rules_installed == 0                        # no per-source rules for one-shot spoofed sources


def test_protect_mode_can_be_switched_off():
    ctl = MitigationController(MitigationConfig(spoof_min_sources=5, protect=False))
    out = _observe(ctl, 0, list(range(1000, 1010)), [0.9] * 10)
    assert not out["protect"] and not ctl.is_dropped(4242, 1)


def test_stale_untrusted_sources_are_forgotten():
    ctl = MitigationController(MitigationConfig())
    _observe(ctl, 1, list(range(100, 200)), [0.0] * 100)    # each seen once: never trusted
    for t in range(2, 100):
        _observe(ctl, t, [CLIENT], [0.0])
    assert len(ctl.last_seen) < 20


def _frame(rows):
    df = pd.DataFrame(rows, columns=["scenario_id", "t", "src", "dst", "pkts", "label"])
    df["kind"] = "udp"
    return df


def test_replay_drops_the_attacker_after_two_windows_and_counts_packets():
    rows = [(0, t, BOT, VICTIM_IP, 100, 1) for t in range(6)] + [(0, t, CLIENT, VICTIM_IP, 10, 0) for t in range(6)]
    df = _frame(rows)
    score = df["label"].to_numpy(dtype=float)
    dropped = replay(df, score, MitigationConfig(strikes_to_block=2, block_ttl=30))
    attack_dropped = df.loc[dropped & (df["label"] == 1), "t"].tolist()
    assert attack_dropped == [2, 3, 4, 5]                  # windows 0 and 1 got through
    assert not dropped[df["label"] == 0].any()
    s = summarize_replay(df, dropped)
    assert np.isclose(s["attack_packets_dropped"], 4 / 6) and s["legit_packets_dropped"] == 0
    assert s["time_to_mitigate_median"] == 2
    assert s["peak_victim_load_reduction"] == 0.0           # the peak (window 0) arrives before any rule exists


def test_replay_keeps_scenarios_independent():
    rows = [(0, t, BOT, VICTIM_IP, 100, 1) for t in range(5)] + [(1, t, BOT, VICTIM_IP, 100, 0) for t in range(5)]
    df = _frame(rows)
    dropped = replay(df, (df["scenario_id"] == 0).to_numpy(dtype=float), MitigationConfig())
    assert dropped[df["scenario_id"] == 0].any() and not dropped[df["scenario_id"] == 1].any()


def test_no_mitigation_summary():
    df = _frame([(0, t, BOT, VICTIM_IP, 100, 1) for t in range(4)])
    s = summarize_replay(df, np.zeros(len(df), dtype=bool))
    assert s["attack_packets_dropped"] == 0 and s["attack_scenarios_never_mitigated"] == 1


def test_policies_differ_in_the_expected_way():
    p = policies(0.3)
    naive, blocking, full = p["naive (block on first flag)"], p["block sources (2 strikes)"], p["block + protect mode"]
    assert naive.strikes_to_block == 1 and not naive.protect
    assert blocking.strikes_to_block == 2 and not blocking.protect
    assert full.protect and full.threshold == 0.3
