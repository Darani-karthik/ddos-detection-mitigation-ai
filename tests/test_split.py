import numpy as np

from ddos_ai.data import split_scenarios


def test_scenarios_are_never_split_across_parts(small_df, small_fd):
    part = split_scenarios(small_df, seed=0)
    train, val, test = (set(small_df["scenario_id"].iloc[small_fd.rows(part, n)]) for n in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(small_df["scenario_id"])


def test_every_kind_is_in_train_and_test(small_df):
    part = split_scenarios(small_df, seed=0)
    kind_of = small_df.groupby("scenario_id")["kind"].first()
    for kind in kind_of.unique():
        parts = {part[sid] for sid in kind_of.index[kind_of == kind]}
        assert {"train", "test"} <= parts, kind


def test_split_is_reproducible_and_depends_on_the_seed(small_df):
    assert split_scenarios(small_df, 0) == split_scenarios(small_df, 0)
    assert split_scenarios(small_df, 0) != split_scenarios(small_df, 1)


def test_part_sizes_are_roughly_60_15_25(small_df):
    part = split_scenarios(small_df, seed=3)
    counts = {name: sum(p == name for p in part.values()) for name in ("train", "val", "test")}
    total = sum(counts.values())
    assert 0.45 < counts["train"] / total < 0.75
    assert counts["val"] > 0 and counts["test"] > 0
    assert np.isclose(sum(counts.values()), small_df["scenario_id"].nunique())
