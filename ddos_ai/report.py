"""Builds results/summary.md from the experiment results, so the tables are never typed by hand."""
from __future__ import annotations

import numpy as np

from .models import MODEL_NAMES
from .simulate import ATTACK_TYPES

ATTACK_NAME = {"syn_spoof": "SYN flood (spoofed)", "udp": "UDP flood", "icmp": "ICMP flood",
               "http": "HTTP flood", "pulse": "Pulsing flood"}


def _ms(values, digits: int = 3, pct: bool = False) -> str:
    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        return "n/a"
    scale = 100 if pct else 1
    mean, std = np.nanmean(values) * scale, np.nanstd(values) * scale
    fmt = f"{{:.{digits}f}}"
    return fmt.format(mean) if len(values) == 1 or std == 0 else f"{fmt.format(mean)} ± {fmt.format(std)}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def dataset_table(summary: dict) -> str:
    rows = [[kind, str(v["scenarios"]), f"{v['records']:,}", f"{v['attack_records']:,}"]
            for kind, v in summary["by_kind"].items()]
    rows.append(["**total**", str(summary["scenarios"]), f"{summary['records']:,}", f"{summary['attack_records']:,}"])
    return _table(["Scenario kind", "Scenarios", "Flow records", "of which attack"], rows)


def main_table(runs: dict) -> str:
    rows = []
    for name in MODEL_NAMES:
        r = runs[name]
        rows.append([
            name, _ms([x["tuned"]["accuracy"] * 100 for x in r], 2),
            _ms([x["tuned"]["precision"] * 100 for x in r], 2), _ms([x["tuned"]["recall"] * 100 for x in r], 2),
            _ms([x["tuned"]["f1"] * 100 for x in r], 2), _ms([x["tuned"]["false_positive_rate"] * 100 for x in r], 2),
            _ms([x["roc_auc"] for x in r], 4), _ms([x["pr_auc"] for x in r], 4)])
    return _table(["Model", "Accuracy %", "Precision %", "Recall %", "F1 %", "False-alarm rate %", "ROC-AUC", "PR-AUC"], rows)


def attack_table(runs: dict) -> str:
    rows = []
    for name in MODEL_NAMES:
        r = runs[name]
        row = [name]
        for attack in ATTACK_TYPES:
            row.append(_ms([x["breakdown"]["recall_by_attack"].get(attack, np.nan) * 100 for x in r], 1))
        row.append(_ms([x["breakdown"].get("macro_recall", np.nan) * 100 for x in r], 1))
        row.append(_ms([x["breakdown"]["false_alarm_by_kind"].get("normal", np.nan) * 100 for x in r], 2))
        row.append(_ms([x["breakdown"]["false_alarm_by_kind"].get("flash", np.nan) * 100 for x in r], 2))
        rows.append(row)
    return _table(["Model"] + [ATTACK_NAME[a] + " recall %" for a in ATTACK_TYPES]
                  + ["Macro recall %", "False alarms, normal %", "False alarms, flash crowd %"], rows)


def default_threshold_table(runs: dict) -> str:
    rows = []
    for name in MODEL_NAMES:
        r = runs[name]
        rows.append([name, _ms([x["default"]["f1"] * 100 for x in r], 2),
                     _ms([x["default"]["recall"] * 100 for x in r], 2),
                     _ms([x["default"]["false_positive_rate"] * 100 for x in r], 2),
                     _ms([x["tuned"]["threshold"] for x in r], 3)])
    return _table(["Model", "F1 % at 0.5", "Recall % at 0.5", "False-alarm rate % at 0.5", "tuned threshold"], rows)


def loao_table(loao: dict) -> str:
    rows = []
    for name in MODEL_NAMES:
        rows.append([name] + [f"{loao[a][name]['recall'] * 100:.1f} / {loao[a][name]['false_positive_rate'] * 100:.2f}"
                              for a in loao])
    return _table(["Model"] + [f"{ATTACK_NAME[a]} held out" for a in loao], rows)


def loao_auc_table(loao: dict) -> str:
    rows = [[name] + [f"{loao[a][name]['roc_auc']:.3f}" for a in loao] for name in MODEL_NAMES]
    return _table(["Model"] + [f"{ATTACK_NAME[a]} held out" for a in loao], rows)


def leakage_table(leaky: dict) -> str:
    rows = []
    for key, label in (("scenario_wise", "Scenario-wise split (used everywhere else)"),
                       ("random_rows", "Random split of flow records (leaky)")):
        r = leaky[key]
        rows.append([label, _ms([x["f1"] * 100 for x in r], 2), _ms([x["roc_auc"] for x in r], 4),
                     _ms([x["breakdown"]["macro_recall"] * 100 for x in r], 1),
                     _ms([x["breakdown"]["recall_by_attack"].get("http", np.nan) * 100 for x in r], 1),
                     _ms([x["test_scenarios_also_in_training"] for x in r], 0)])
    return _table(["Evaluation", "F1 %", "ROC-AUC", "Macro recall %", "HTTP-flood recall %",
                   "Test scenarios also in training"], rows)


def mitigation_table(results: dict) -> str:
    def pct(v):
        return "n/a" if v != v else f"{v * 100:.1f}"

    rows = []
    for name, r in results.items():
        rows.append([name, pct(r["attack_packets_dropped"]), pct(r["legit_packets_dropped"]),
                     pct(r["legit_packets_dropped_flash"]), pct(r["peak_victim_load_reduction"]),
                     "n/a" if r["time_to_mitigate_median"] != r["time_to_mitigate_median"]
                     else f"{r['time_to_mitigate_median']:.0f}",
                     str(r["attack_scenarios_never_mitigated"])])
    return _table(["Detector / policy", "Attack packets dropped %", "Legit packets dropped %",
                   "Legit dropped in flash crowds %", "Peak server load cut %",
                   "Median windows to mitigate", "Attack scenarios never mitigated"], rows)


def time_to_mitigate_table(results: dict, key: str = "LightGBM / block + protect mode") -> str:
    ttm = results[key]["time_to_mitigate"]
    return _table(["Scenario kind", "Median windows until 90% of attack packets are dropped"],
                  [[kind, "never" if v != v else f"{v:.1f}"] for kind, v in ttm.items()])


def write_summary(path, dataset: dict, runs: dict, loao: dict | None, leaky: dict, mitigation: dict) -> None:
    parts = ["# Results summary", "",
             "Generated by `python -m ddos_ai.run_all`. Mean ± std over 3 scenario splits where noted.", "",
             "## Data set", "", dataset_table(dataset), "",
             "## Detection on held-out scenarios", "", main_table(runs), "",
             "Decision threshold tuned on validation scenarios (maximum balanced accuracy). Positive class = attack.", "",
             "### Detection rate per attack type", "", attack_table(runs), "",
             "### At the natural 0.5 threshold", "", default_threshold_table(runs), ""]
    if loao:
        parts += ["## Attacks never seen in training (leave-one-attack-out)", "",
                  "Cells: recall % / false-alarm rate % at the threshold tuned on validation scenarios without the attack.", "",
                  loao_table(loao), "", "ROC-AUC on the same test scenarios (threshold-free):", "", loao_auc_table(loao), ""]
    parts += ["## Leaky vs scenario-wise evaluation (LightGBM)", "", leakage_table(leaky), "",
              "## Mitigation replay", "", mitigation_table(mitigation), "",
              "### Time to mitigate by attack kind", "", time_to_mitigate_table(mitigation), ""]
    path.write_text("\n".join(parts), encoding="utf-8")
