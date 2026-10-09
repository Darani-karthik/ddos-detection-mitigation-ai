"""The experiments: main comparison, unseen-attack generalisation, leakage demo and mitigation replay."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .data import FlowData, split_scenarios
from .evaluation import breakdown, evaluate_run, metrics_at, pick_threshold
from .mitigation import MitigationConfig, policies, replay, summarize_replay, victim_timeline
from .models import FITTERS, MODEL_NAMES, fit_lightgbm
from .simulate import ATTACK_TYPES

SEEDS = (0, 1, 2)


def _rows(fd: FlowData, df: pd.DataFrame, seed: int):
    part = split_scenarios(df, seed)
    return part, tuple(fd.rows(part, name) for name in ("train", "val", "test"))


def run_main(df: pd.DataFrame, fd: FlowData, seeds=SEEDS, log=print) -> tuple[dict, dict]:
    """Every model on three different scenario splits (split seed = model seed).

    Returns (runs per model, primary) where `primary` holds the scores of the first split, which the
    plots and the mitigation replay use.
    """
    runs: dict[str, list] = {name: [] for name in MODEL_NAMES}
    primary: dict = {}
    for seed in seeds:
        part, (train, val, test) = _rows(fd, df, seed)
        frame = df.iloc[test].reset_index(drop=True)
        log(f"split {seed}: {len(train):,} train / {len(val):,} val / {len(test):,} test flow records "
            f"({sum(p == 'test' for p in part.values())} test scenarios)")
        for name in MODEL_NAMES:
            start = time.time()
            scorer = FITTERS[name](fd, train, val, seed)
            score_val, score_test = scorer.score(val), scorer.score(test)
            run = evaluate_run(fd.y[val], score_val, frame, score_test)
            run.update(seed=seed, info=scorer.info, test_rows=int(len(test)),
                       test_scenarios=int(sum(p == "test" for p in part.values())))
            runs[name].append(run)
            log(f"  {name:9s} F1 {run['tuned']['f1']:.4f}  AUC {run['roc_auc']:.4f}  "
                f"macro recall {run['breakdown'].get('macro_recall', float('nan')):.3f}  ({time.time() - start:.0f}s)")
            if seed == seeds[0]:
                primary.setdefault("scores", {})[name] = score_test
                primary.setdefault("val_scores", {})[name] = score_val
                primary.setdefault("thresholds", {})[name] = run["tuned"]["threshold"]
                if name == "LightGBM":
                    primary["lightgbm"] = scorer
        if seed == seeds[0]:
            primary.update(part=part, train=train, val=val, test=test, frame=frame)
    return runs, primary


def run_loao(df: pd.DataFrame, fd: FlowData, seed: int = 0, log=print) -> dict:
    """Leave-one-attack-out: train without one attack type, then test on scenarios of exactly that attack.

    The decision threshold is tuned on validation scenarios that do not contain the attack either, so
    this is the situation of meeting an attack the detector has never seen.
    """
    part = split_scenarios(df, seed)
    attacks_in = df[df["label"] == 1].groupby("scenario_id")["attack_type"].agg(set)
    scenario_of = df["scenario_id"].to_numpy()
    results: dict = {}
    for held_out in ATTACK_TYPES:
        contains = {sid for sid, kinds in attacks_in.items() if held_out in kinds}
        usable = ~np.isin(scenario_of, list(contains))
        train = np.nonzero(usable & np.isin(scenario_of, [s for s, p in part.items() if p == "train"]))[0]
        val = np.nonzero(usable & np.isin(scenario_of, [s for s, p in part.items() if p == "val"]))[0]
        test = np.nonzero((df["kind"] == held_out).to_numpy())[0]
        frame = df.iloc[test].reset_index(drop=True)
        log(f"held-out {held_out}: train {len(train):,}, val {len(val):,}, test {len(test):,} records")
        results[held_out] = {}
        for name in MODEL_NAMES:
            scorer = FITTERS[name](fd, train, val, seed)
            score_val, score_test = scorer.score(val), scorer.score(test)
            threshold = pick_threshold(fd.y[val], score_val)
            m = metrics_at(frame["label"], score_test, threshold)
            results[held_out][name] = {
                "recall": m["recall"], "false_positive_rate": m["false_positive_rate"], "f1": m["f1"],
                "precision": m["precision"], "roc_auc": float(roc_auc_score(frame["label"], score_test)),
                "attack_records": int(frame["label"].sum()), "scenarios": int(frame["scenario_id"].nunique())}
            log(f"  {name:9s} recall {m['recall']:.3f}  false-alarm {m['false_positive_rate']:.4f}  "
                f"AUC {results[held_out][name]['roc_auc']:.4f}")
    return results


def run_leaky(df: pd.DataFrame, fd: FlowData, seeds=SEEDS, log=print) -> dict:
    """The same LightGBM, once on a scenario-wise split and once on a leaky random split of the records.

    The random split puts records of the same scenario, and of the same bot flow, in both training and
    test data. Only the evaluation differs; model, features and tuning are identical.
    """
    out = {"scenario_wise": [], "random_rows": []}
    for seed in seeds:
        _, (train, val, test) = _rows(fd, df, seed)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        n_test, n_val = len(test), len(val)
        r_test, r_val, r_train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
        for key, (tr, va, te) in {"scenario_wise": (train, val, test), "random_rows": (r_train, r_val, r_test)}.items():
            scorer = fit_lightgbm(fd, tr, va, seed)
            score_test = scorer.score(te)
            frame = df.iloc[te].reset_index(drop=True)
            run = evaluate_run(fd.y[va], scorer.score(va), frame, score_test)
            out[key].append({"f1": run["tuned"]["f1"], "recall": run["tuned"]["recall"],
                             "roc_auc": run["roc_auc"], "breakdown": run["breakdown"],
                             "test_scenarios_also_in_training": int(
                                 len(set(df["scenario_id"].iloc[te]) & set(df["scenario_id"].iloc[tr])))})
        log(f"  split {seed}: scenario-wise F1 {out['scenario_wise'][-1]['f1']:.4f}, "
            f"random-rows F1 {out['random_rows'][-1]['f1']:.4f}")
    return out


def run_mitigation(primary: dict, log=print) -> tuple[dict, dict]:
    """Replay the mitigation policies on the test scenarios of the first split.

    Detector scores come from the models trained for the main comparison; the thresholds are the ones
    tuned on validation scenarios. The oracle uses the true labels as scores: it shows what the policy
    itself costs when detection is perfect.
    """
    frame = primary["frame"]
    results: dict = {}
    examples: dict = {}
    results["no mitigation"] = summarize_replay(frame, np.zeros(len(frame), dtype=bool))
    for detector in ("LightGBM", "LSTM"):
        score, threshold = primary["scores"][detector], primary["thresholds"][detector]
        for policy, cfg in policies(threshold).items():
            dropped = replay(frame, score, cfg)
            results[f"{detector} / {policy}"] = summarize_replay(frame, dropped)
            if detector == "LightGBM":
                examples[policy] = dropped
    oracle_cfg = replace(MitigationConfig(), threshold=0.5)
    results["oracle detector / block + protect mode"] = summarize_replay(
        frame, replay(frame, frame["label"].to_numpy(dtype=float), oracle_cfg))
    for name, r in results.items():
        log(f"  {name:48s} attack dropped {r['attack_packets_dropped']:.3f}  "
            f"legit dropped {r['legit_packets_dropped']:.3f}")
    return results, examples


def example_timelines(primary: dict, dropped: np.ndarray, kinds=("syn_spoof", "udp", "http")) -> dict:
    """Server-side packet rates over time for one test scenario of each given kind."""
    frame = primary["frame"]
    out = {}
    for kind in kinds:
        candidates = frame.loc[frame["kind"] == kind, "scenario_id"].unique()
        if len(candidates) == 0:
            continue
        mask = (frame["scenario_id"] == candidates[0]).to_numpy()
        out[kind] = victim_timeline(frame[mask], dropped[mask])
    return out
