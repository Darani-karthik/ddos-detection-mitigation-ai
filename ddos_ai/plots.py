"""Figures for the README. One colour per entity, thin marks, recessive grid, direct labels where it helps."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .features import FEATURES  # noqa: E402
from .models import MODEL_NAMES  # noqa: E402
from .simulate import ATTACK_TYPES  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_COLOR = {"GNB": BLUE, "LightGBM": ORANGE, "FNN": AQUA, "LSTM": YELLOW}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
ATTACK_LABEL = {"syn_spoof": "SYN flood\n(spoofed)", "udp": "UDP flood", "icmp": "ICMP flood",
                "http": "HTTP flood", "pulse": "Pulsing flood"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 160,
    })


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _mean_std(runs, getter):
    values = np.array([getter(r) for r in runs], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values))


# ------------------------------------------------------------------------------- the data
def traffic_overview(df, path):
    """What the simulated traffic looks like: packets per second at the switch, normal vs attack."""
    _style()
    kinds = ["flash", "syn_spoof", "udp", "icmp", "http", "pulse"]
    titles = {"flash": "Flash crowd (normal)", "syn_spoof": "SYN flood (spoofed)", "udp": "UDP flood",
              "icmp": "ICMP flood", "http": "HTTP flood", "pulse": "Pulsing flood"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.8), sharex=True)
    for ax, kind in zip(axes.ravel(), kinds):
        sid = df.loc[df["kind"] == kind, "scenario_id"].iloc[0]
        d = df[df["scenario_id"] == sid]
        for label, color, name in ((0, BLUE, "normal"), (1, ORANGE, "attack")):
            series = d[d["label"] == label].groupby("t")["pkts"].sum().reindex(range(int(d["t"].max()) + 1))
            if series.notna().any():
                ax.plot(series.index, series.fillna(0), color=color, lw=1.6, label=name)
        ax.set_title(titles[kind])
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("packets / s at the switch")
    axes[0, 1].legend(loc="upper left", ncols=2)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- main results
def attack_recall(runs, path):
    """Detection rate per attack type for the four models (mean and std over three scenario splits)."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.2
    x = np.arange(len(ATTACK_TYPES))
    for i, name in enumerate(MODEL_NAMES):
        means, stds = zip(*[_mean_std(runs[name], lambda r, a=a: r["breakdown"]["recall_by_attack"].get(a, np.nan))
                            for a in ATTACK_TYPES])
        ax.bar(x + (i - 1.5) * width, means, width * 0.88, yerr=stds, color=MODEL_COLOR[name], label=name,
               error_kw={"elinewidth": 0.9, "ecolor": INK2, "capsize": 2})
    http = ATTACK_TYPES.index("http")
    for i, name in enumerate(MODEL_NAMES):
        mean, _ = _mean_std(runs[name], lambda r: r["breakdown"]["recall_by_attack"].get("http", np.nan))
        ax.text(http + (i - 1.5) * width, mean + 0.045, f"{mean:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [ATTACK_LABEL[a] for a in ATTACK_TYPES])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of attack flow records detected")
    ax.set_title("Share of attack flow records detected, by attack type")
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, -0.28))
    _save(fig, path)


def confusion_matrices(runs, path):
    """Row-normalised confusion matrices on the test scenarios of the first split."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, name in zip(axes, MODEL_NAMES):
        (tn, fp), (fn, tp) = runs[name][0]["tuned"]["confusion_matrix"]
        m = np.array([[tn, fp], [fn, tp]], dtype=float)
        m = m / m.sum(axis=1, keepdims=True)
        ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.5 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["normal", "attack"])
        ax.set_yticks([0, 1], ["normal", "attack"], rotation=90, va="center")
        ax.set_xlabel("predicted")
        ax.set_title(name)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    _save(fig, path)


def precision_recall(primary, path):
    from sklearn.metrics import precision_recall_curve

    _style()
    y = primary["frame"]["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(y, primary["scores"][name])
        ax.plot(recall, precision, color=MODEL_COLOR[name], lw=1.8, label=name)
    ax.set_xlabel("recall (share of attack records detected)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.01)
    ax.set_title("Precision-recall on the test scenarios")
    ax.legend(loc="lower left")
    _save(fig, path)


def feature_importance(booster, path):
    _style()
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(np.arange(len(order)), gain[order] / gain.sum(), color=ORANGE, height=0.62)
    ax.set_yticks(np.arange(len(order)), [FEATURES[i] for i in order])
    ax.set_xlabel("share of total split gain")
    ax.set_title("What the LightGBM detector relies on")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ------------------------------------------------------------------------ generalisation
def loao_heatmap(loao, path):
    _style()
    attacks = list(loao)
    grid = np.array([[loao[a][m]["recall"] for a in attacks] for m in MODEL_NAMES])
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else INK, fontsize=10)
    ax.set_xticks(range(len(attacks)), [ATTACK_LABEL[a] for a in attacks])
    ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES)
    ax.set_xlabel("attack type held out of training (and of threshold tuning)")
    ax.set_title("Detection rate on an attack the model has never seen")
    _save(fig, path)


def leakage_comparison(leaky, path):
    _style()
    metrics = [("F1 score", lambda r: r["f1"]),
               ("Macro recall over attack types", lambda r: r["breakdown"]["macro_recall"]),
               ("HTTP-flood recall", lambda r: r["breakdown"]["recall_by_attack"].get("http", np.nan))]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    width = 0.34
    for i, (key, label, color) in enumerate((("scenario_wise", "scenario-wise split (used in this repo)", BLUE),
                                              ("random_rows", "random split of records (leaky)", ORANGE))):
        stats = [_mean_std(leaky[key], getter) for _, getter in metrics]
        xs = np.arange(len(metrics)) + (i - 0.5) * width
        ax.bar(xs, [s[0] for s in stats], width * 0.9, yerr=[s[1] for s in stats], color=color, label=label,
               error_kw={"elinewidth": 0.9, "ecolor": INK2, "capsize": 2})
        for x, (mean, _) in zip(xs, stats):
            ax.text(x, mean + 0.04, f"{mean:.3f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Same LightGBM, two ways of splitting the data")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    _save(fig, path)


# -------------------------------------------------------------------------- mitigation
def mitigation_policies(results, path, detector: str = "LightGBM"):
    _style()
    names = [("naive (block on first flag)", "Naive:\nblock on first flag"),
             ("block sources (2 strikes)", "Block sources\n(2 strikes)"),
             ("block + protect mode", "Block +\nprotect mode")]
    rows = [(label, results[f"{detector} / {key}"]) for key, label in names]
    rows.append(("Oracle detector,\nblock + protect", results["oracle detector / block + protect mode"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (key, title, color) in zip(axes, (("attack_packets_dropped", "Attack packets dropped (higher is better)", ORANGE),
                                              ("legit_packets_dropped", "Legitimate packets dropped (lower is better)", BLUE))):
        values = [r[key] for _, r in rows]
        ax.bar(range(len(rows)), np.array(values) * 100, color=color, width=0.62)
        for x, v in enumerate(values):
            ax.text(x, v * 100 + 1.2, f"{v * 100:.1f}%", ha="center", fontsize=9, color=INK2)
        ax.set_xticks(range(len(rows)), [label for label, _ in rows], fontsize=8)
        ax.set_ylim(0, 112 if key == "attack_packets_dropped" else max(values) * 100 * 1.3)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("% of packets")
    fig.tight_layout()
    _save(fig, path)


def mitigation_timeline(timelines, path):
    """Packets per second reaching the server, with the full policy on."""
    _style()
    titles = {"syn_spoof": "SYN flood (spoofed)", "udp": "UDP flood", "http": "HTTP flood"}
    fig, axes = plt.subplots(1, len(timelines), figsize=(10.5, 3.5))
    axes = np.atleast_1d(axes)
    for ax, (kind, tl) in zip(axes, timelines.items()):
        ax.plot(tl.index, tl["attack_offered"], color=ORANGE, lw=1.2, ls=(0, (4, 2)), label="attack, offered")
        ax.plot(tl.index, tl["attack_delivered"], color=ORANGE, lw=2.0, label="attack, delivered")
        ax.plot(tl.index, tl["legit_offered"], color=MUTED, lw=1.2, ls=(0, (4, 2)), label="legitimate, offered")
        ax.plot(tl.index, tl["legit_delivered"], color=BLUE, lw=2.0, label="legitimate, delivered")
        ax.set_title(titles.get(kind, kind))
        ax.set_xlabel("time (s)")
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("packets / s reaching the server")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    _save(fig, path)
