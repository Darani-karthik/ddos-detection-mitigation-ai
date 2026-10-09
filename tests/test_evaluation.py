import numpy as np
import pandas as pd

from ddos_ai.evaluation import breakdown, evaluate_run, metrics_at, pick_threshold


def test_metrics_at_known_confusion_matrix():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
    m = metrics_at(y, score, 0.5)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["accuracy"] == 0.75 and m["recall"] == 0.75 and m["precision"] == 0.75
    assert m["false_positive_rate"] == 0.25 and m["specificity"] == 0.75
    assert np.isclose(m["f1"], 0.75) and np.isclose(m["balanced_accuracy"], 0.75)


def test_metrics_handle_a_detector_that_flags_nothing():
    m = metrics_at(np.array([0, 1, 1]), np.zeros(3), 0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["specificity"] == 1.0


def test_pick_threshold_separates_the_classes():
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.6, 0.9, 50)])
    thr = pick_threshold(y, score)
    assert 0.3 <= thr <= 0.6
    assert metrics_at(y, score, thr)["balanced_accuracy"] == 1.0


def test_pick_threshold_ignores_class_imbalance():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    score = np.concatenate([rng.beta(2, 5, 900), rng.beta(5, 2, 100)])
    thr = pick_threshold(y, score)
    m = metrics_at(y, score, thr)
    assert m["recall"] > 0.7 and m["specificity"] > 0.7


def test_breakdown_reports_recall_per_attack_and_false_alarms():
    frame = pd.DataFrame({
        "label": [1, 1, 1, 1, 0, 0, 0, 0],
        "attack_type": ["udp", "udp", "http", "http", "none", "none", "none", "none"],
        "kind": ["udp"] * 2 + ["http"] * 2 + ["normal", "normal", "flash", "flash"],
    }, index=[10, 11, 12, 13, 14, 15, 16, 17])             # a non-default index must not matter
    score = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.8, 0.1, 0.1])
    b = breakdown(frame, score, 0.5)
    assert b["recall_by_attack"] == {"http": 0.5, "udp": 1.0}
    assert b["false_alarm_by_kind"] == {"normal": 0.5, "flash": 0.0}
    assert np.isclose(b["macro_recall"], 0.75)


def test_evaluate_run_ties_it_together():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    score = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 400), 0, 1)
    frame = pd.DataFrame({"label": y, "attack_type": np.where(y == 1, "udp", "none"),
                          "kind": np.where(y == 1, "udp", "normal")})
    run = evaluate_run(y, score, frame, score)
    assert run["roc_auc"] > 0.9 and run["pr_auc"] > 0.9
    assert run["tuned"]["f1"] > 0.85 and set(run) >= {"tuned", "default", "roc_auc", "pr_auc", "breakdown"}
