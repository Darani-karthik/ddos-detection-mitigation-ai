import numpy as np
import pytest

from ddos_ai.schema import PROTO_ICMP, PROTO_TCP, PROTO_UDP, RAW_COLUMNS, VICTIM_IP, int_to_ip, ip_to_int
from ddos_ai.simulate import ATTACK_TYPES, KINDS, Scenario, plan_scenarios, simulate_scenario


def test_ip_round_trip():
    assert int_to_ip(ip_to_int("10.0.0.1")) == "10.0.0.1"
    assert VICTIM_IP == ip_to_int("10.0.0.1")
    with pytest.raises(ValueError):
        ip_to_int("10.0.0.300")


def test_scenario_is_deterministic():
    sc = Scenario(0, "udp", seed=123, duration=40)
    assert simulate_scenario(sc).equals(simulate_scenario(sc))
    assert not simulate_scenario(sc).equals(simulate_scenario(Scenario(0, "udp", seed=124, duration=40)))


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        Scenario(0, "ping_of_death", seed=1)


@pytest.mark.parametrize("kind", KINDS)
def test_records_are_well_formed(kind):
    df = simulate_scenario(Scenario(3, kind, seed=5, duration=60))
    assert set(RAW_COLUMNS) <= set(df.columns)
    assert (df["scenario_id"] == 3).all()
    assert df["t"].between(0, 59).all()
    assert (df["pkts"] >= 1).all() and (df["bytes"] > 0).all() and (df["flow_age"] >= 1).all()
    assert df["proto"].isin([PROTO_TCP, PROTO_UDP, PROTO_ICMP]).all()
    assert df["pair_flow"].isin([0, 1]).all()
    assert (df["t"].diff().dropna() >= 0).all()            # sorted by window
    # the label and the attack type always agree
    assert ((df["label"] == 1) == (df["attack_type"] != "none")).all()


def test_normal_scenarios_have_no_attack():
    for kind in ("normal", "flash"):
        assert simulate_scenario(Scenario(0, kind, seed=9, duration=60))["label"].sum() == 0


@pytest.mark.parametrize("kind", ATTACK_TYPES)
def test_attack_scenarios_contain_that_attack_aimed_at_the_victim(kind):
    df = simulate_scenario(Scenario(0, kind, seed=11, duration=90))
    attack = df[df["label"] == 1]
    assert len(attack) > 0
    assert set(attack["attack_type"]) == {kind}
    assert (attack["dst"] == VICTIM_IP).all()
    # the attack starts after the network has run normally for a while
    assert attack["t"].min() >= 25


def test_spoofed_flood_uses_new_sources_every_time():
    df = simulate_scenario(Scenario(0, "syn_spoof", seed=2, duration=90))
    attack = df[df["label"] == 1]
    assert (attack["flow_age"] == 1).all() and (attack["pair_flow"] == 0).all()
    assert attack["src"].nunique() > 0.99 * len(attack)


def test_bot_floods_keep_their_source_addresses():
    attack = simulate_scenario(Scenario(0, "udp", seed=2, duration=90)).query("label == 1")
    assert attack["src"].nunique() <= 10 and attack["flow_age"].max() > 10


def test_pulsing_flood_has_silent_gaps():
    attack = simulate_scenario(Scenario(0, "pulse", seed=4, duration=90)).query("label == 1")
    active = attack["t"].nunique()
    assert active < attack["t"].max() - attack["t"].min() + 1      # some windows have no attack traffic


def test_mixed_scenario_has_two_attack_types():
    df = simulate_scenario(Scenario(0, "mixed", seed=8, duration=90))
    assert df.loc[df["label"] == 1, "attack_type"].nunique() == 2


def test_flash_crowd_multiplies_the_traffic_peak_without_any_attack():
    flash = simulate_scenario(Scenario(0, "flash", seed=3, duration=90))
    plain = simulate_scenario(Scenario(0, "normal", seed=3, duration=90))
    assert flash.groupby("t")["pkts"].sum().max() > 2 * plain.groupby("t")["pkts"].sum().max()
    assert flash["label"].sum() == 0


def test_plan_covers_every_kind_with_unique_ids_and_seeds():
    plan = plan_scenarios(120, seed=42)
    assert {sc.kind for sc in plan} == set(KINDS)
    assert len({sc.scenario_id for sc in plan}) == len(plan)
    assert len({sc.seed for sc in plan}) == len(plan)
    assert len(plan_scenarios(30, seed=42)) < len(plan)


def test_attacks_are_the_minority_class(small_df):
    assert 0.05 < small_df["label"].mean() < 0.5
    assert set(small_df["kind"]) == set(KINDS)
    assert np.isfinite(small_df[["pkts", "bytes"]].to_numpy()).all()
