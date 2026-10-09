import numpy as np
import pytest

from ddos_ai.data import split_scenarios
from ddos_ai.evaluation import evaluate_run
from ddos_ai.models import FITTERS, MODEL_NAMES


@pytest.fixture(scope="module")
def parts(small_df, small_fd):
    part = split_scenarios(small_df, 0)
    return tuple(small_fd.rows(part, n) for n in ("train", "val", "test"))


def _check(name, small_df, small_fd, parts, min_auc):
    train, val, test = parts
    scorer = FITTERS[name](small_fd, train, val, 0)
    assert scorer.name == name
    score = scorer.score(test)
    assert score.shape == (len(test),) and np.isfinite(score).all() and 0 <= score.min() and score.max() <= 1
    run = evaluate_run(small_fd.y[val], scorer.score(val), small_df.iloc[test].reset_index(drop=True), score)
    assert run["roc_auc"] > min_auc, run["roc_auc"]
    return scorer


def test_gnb_learns_the_task(small_df, small_fd, parts):
    _check("GNB", small_df, small_fd, parts, 0.8)


def test_lightgbm_learns_the_task_and_is_reproducible(small_df, small_fd, parts):
    scorer = _check("LightGBM", small_df, small_fd, parts, 0.95)
    again = FITTERS["LightGBM"](small_fd, parts[0], parts[1], 0)
    assert np.allclose(scorer.score(parts[2]), again.score(parts[2]))
    assert scorer.info["trees"] > 0


def test_fnn_learns_the_task(small_df, small_fd, parts):
    pytest.importorskip("tensorflow")
    _check("FNN", small_df, small_fd, parts, 0.95)


def test_lstm_learns_the_task(small_df, small_fd, parts):
    pytest.importorskip("tensorflow")
    scorer = _check("LSTM", small_df, small_fd, parts, 0.95)
    # scoring must not depend on how the rows are batched
    rows = parts[2][:500]
    assert np.allclose(scorer.score(rows), scorer.score(rows[:250]).tolist() + scorer.score(rows[250:]).tolist(),
                       atol=1e-5)


def test_all_four_models_are_registered():
    assert set(MODEL_NAMES) == set(FITTERS) == {"GNB", "LightGBM", "FNN", "LSTM"}
