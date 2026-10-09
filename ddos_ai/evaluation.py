"""Metrics shared by every model: validation-tuned threshold, scores per attack type."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from .schema import ATTACK_TYPE


def pick_threshold(y_val, score_val, default: float = 0.5) -> float:
    """Score threshold that maximises balanced accuracy on the validation scenarios.

    Balanced accuracy weighs a missed attack and a wrongly flagged legitimate flow equally, whatever the
    class ratio. Ties go to the threshold closest to `default`.
    """
    fpr, tpr, thr = roc_curve(y_val, score_val)
    keep = np.isfinite(thr)
    fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    balanced = (tpr + 1.0 - fpr) / 2.0
    best = balanced >= balanced.max() - 1e-9
    candidates = thr[best]
    return float(candidates[np.argmin(np.abs(candidates - default))])


def metrics_at(y_true, score, threshold: float) -> dict:
    """Test metrics with 'attack' as the positive class."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_positive_rate": float(1 - specificity),
        "f1": float(f1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def breakdown(frame: pd.DataFrame, score, threshold: float) -> dict:
    """Detection rate per attack type and false-alarm rate on normal traffic.

    `frame` holds the evaluated rows (columns `label`, `attack_type`, `kind`); `score` is aligned to it.
    Pooled metrics are dominated by the spoofed SYN flood, which has far more flow records than any
    other attack, so these per-type numbers are the fairer picture.
    """
    frame = frame.reset_index(drop=True)
    flagged = np.asarray(score) >= threshold
    out = {"recall_by_attack": {}, "false_alarm_by_kind": {}}
    for attack, group in frame[frame["label"] == 1].groupby(ATTACK_TYPE):
        out["recall_by_attack"][attack] = float(flagged[group.index.to_numpy()].mean())
    legit = frame["label"] == 0
    for kind in ("normal", "flash"):
        rows = np.nonzero((legit & (frame["kind"] == kind)).to_numpy())[0]
        if len(rows):
            out["false_alarm_by_kind"][kind] = float(flagged[rows].mean())
    if out["recall_by_attack"]:
        out["macro_recall"] = float(np.mean(list(out["recall_by_attack"].values())))
    return out


def evaluate_run(y_val, score_val, frame_test: pd.DataFrame, score_test, default: float = 0.5) -> dict:
    """Score one trained model on the test scenarios.

    The decision threshold is chosen on the validation scenarios ("tuned"); the metrics at the model's
    natural 0.5 threshold ("default") are kept alongside. ROC-AUC and PR-AUC need no threshold.
    """
    y_test = frame_test["label"].to_numpy()
    tuned = pick_threshold(y_val, score_val, default)
    run = {
        "tuned": metrics_at(y_test, score_test, tuned),
        "default": metrics_at(y_test, score_test, default),
        "roc_auc": float(roc_auc_score(y_test, score_test)),
        "pr_auc": float(average_precision_score(y_test, score_test)),
    }
    run["breakdown"] = breakdown(frame_test, score_test, tuned)
    return run
