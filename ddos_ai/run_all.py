"""Run every experiment and rebuild `results/`.

    python -m ddos_ai.run_all                      # simulated data, the full study (about 15 minutes)
    python -m ddos_ai.run_all --quick              # small smoke run (about 2 minutes)
    python -m ddos_ai.run_all --data my_capture.csv --out results_capture
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import plots, report
from .data import FlowData, build_simulated, load_flows_csv
from .experiments import SEEDS, example_timelines, run_leaky, run_loao, run_main, run_mitigation
from .models import MODEL_NAMES, save_lightgbm
from .simulate import ATTACK_TYPES


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def dataset_summary(df) -> dict:
    by_kind = {}
    for kind, g in df.groupby("kind"):
        by_kind[kind] = {"scenarios": int(g["scenario_id"].nunique()), "records": int(len(g)),
                         "attack_records": int(g["label"].sum())}
    attack = df[df["label"] == 1].groupby("attack_type").size().to_dict()
    return {"scenarios": int(df["scenario_id"].nunique()), "records": int(len(df)),
            "attack_records": int(df["label"].sum()), "by_kind": by_kind,
            "attack_records_by_type": {k: int(v) for k, v in attack.items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV of flow records captured from a real run instead of the simulator")
    ap.add_argument("--scenarios", type=int, default=120)
    ap.add_argument("--duration", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42, help="seed of the simulated data set")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--quick", action="store_true", help="small data set, one split, no leave-one-out")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    start = time.time()
    out = Path(args.out)
    simulated = args.data is None
    if args.quick:
        args.scenarios = min(args.scenarios, 40)
    seeds = SEEDS[:1] if args.quick else SEEDS

    print("== data set")
    df = build_simulated(args.scenarios, args.seed, args.duration) if simulated else load_flows_csv(args.data)
    fd = FlowData.from_frame(df)
    summary = dataset_summary(df)
    print(f"{summary['scenarios']} scenarios, {summary['records']:,} flow records, "
          f"{summary['attack_records'] / summary['records']:.1%} attack")
    _dump(out / "metrics" / "dataset.json", summary)

    print("== main comparison")
    runs, primary = run_main(df, fd, seeds)
    _dump(out / "metrics" / "main.json", runs)
    save_lightgbm(primary["lightgbm"], primary["thresholds"]["LightGBM"], args.models_dir)

    loao = None
    if simulated and not args.quick:
        print("== leave-one-attack-out")
        loao = run_loao(df, fd)
        _dump(out / "metrics" / "leave_one_attack_out.json", loao)

    print("== leaky vs scenario-wise evaluation")
    leaky = run_leaky(df, fd, seeds)
    _dump(out / "metrics" / "leakage.json", leaky)

    print("== mitigation replay")
    mitigation, dropped = run_mitigation(primary)
    _dump(out / "metrics" / "mitigation.json", mitigation)

    print("== figures and summary")
    fig = out / "figures"
    plots.confusion_matrices(runs, fig / "confusion_matrices.png")
    plots.precision_recall(primary, fig / "precision_recall.png")
    plots.feature_importance(primary["lightgbm"].booster, fig / "feature_importance.png")
    plots.leakage_comparison(leaky, fig / "leakage_comparison.png")
    plots.mitigation_policies(mitigation, fig / "mitigation_policies.png")
    if simulated:
        plots.traffic_overview(df, fig / "traffic_overview.png")
        plots.attack_recall(runs, fig / "attack_recall.png")
        plots.mitigation_timeline(example_timelines(primary, dropped["block + protect mode"]),
                                  fig / "mitigation_timeline.png")
    if loao:
        plots.loao_heatmap(loao, fig / "leave_one_attack_out.png")
    report.write_summary(out / "summary.md", summary, runs, loao, leaky, mitigation)
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out}/")


if __name__ == "__main__":
    main()
