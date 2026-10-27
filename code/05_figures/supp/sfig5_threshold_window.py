#!/usr/bin/env python3
"""Supplementary Figure 5: width of the intermediate noise window.

Produces : figures/supp/supp_threshold_window.png
Run      : python sfig5_threshold_window.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "supp"
FIGS.mkdir(parents=True, exist_ok=True)


"""Supplementary figure: width of the intermediate noise window for all configurations.

Top row: worst-case and best-case thresholds (the window between them).
Bottom row: ratio r*_best / r*_worst.

Usage: python sfig5_threshold_window.py [thresholds.csv] [output_stem]
Requires: pandas, numpy, matplotlib.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA / "fits" / "thresholds.csv"
out_stem = sys.argv[2] if len(sys.argv) > 2 else str(FIGS / "supp_threshold_window")
df_threshold_minlogr = pd.read_csv(csv_path)

df_supp_threshold = df_threshold_minlogr.rename(columns={
    "Noise Type": "channel", "Graph Structure": "family",
    "Node": "n", "QAOA Layers": "p", "Threshold": "r"})

summary = (df_supp_threshold.groupby(["channel", "family", "n"])["r"]
             .agg(best="max", worst="min").reset_index())
summary["ratio"] = summary["best"] / summary["worst"]
median_ratio = summary["ratio"].median()

channels = [("AmplitudeDamping", "Amplitude damping"),
            ("Depolarization", "Depolarising"),
            ("BitFlip", "Bit-flip")]
families = [("ER random", "Erdős-Rényi random", "#CC79A7"),
            ("Path", "Path", "#E69F00"),
            ("Complete bipartite", "Complete bipartite", "#0072B2")]
orders = [6, 9, 12, 18]
# x position: four orders per family, with a gap between families
pos = {(f, n): fi * (len(orders) + 1) + k
       for fi, (f, _, _) in enumerate(families) for k, n in enumerate(orders)}

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6), sharex="col",
                         gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})

for c_i, (ch, ch_label) in enumerate(channels):
    ax_w, ax_r = axes[0, c_i], axes[1, c_i]
    for fam, _, colour in families:
        sub = summary[(summary.channel == ch) & (summary.family == fam)].sort_values("n")
        x = np.array([pos[(fam, n)] for n in sub["n"]])
        # Top: window between worst and best
        ax_w.vlines(x, sub["worst"], sub["best"], color=colour, lw=4.5, alpha=0.45, zorder=2)
        ax_w.plot(x, sub["best"], ls="none", marker="^", ms=5.5, mfc=colour, mec="white", mew=0.5, zorder=3)
        ax_w.plot(x, sub["worst"], ls="none", marker="v", ms=5.5, mfc=colour, mec="white", mew=0.5, zorder=3)
        # Bottom: ratio, drawn as a stem from 1
        ax_r.vlines(x, 1, sub["ratio"], color=colour, lw=1.2, zorder=2)
        ax_r.plot(x, sub["ratio"], ls="none", marker="o", ms=5, mfc=colour, mec="white", mew=0.5, zorder=3)

    ax_w.set_yscale("log")
    ax_w.set_ylim(0.0027, 0.65)
    ax_w.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3]))
    ax_w.yaxis.set_major_formatter(FixedFormatter(["0.003", "0.01", "0.03", "0.1", "0.3"]))
    ax_w.yaxis.set_minor_locator(NullLocator())
    ax_w.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    ax_w.set_title(ch_label, fontsize=9, fontweight="bold", pad=10)

    ax_r.set_yscale("log")
    ax_r.set_ylim(1, 11)
    ax_r.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
    ax_r.yaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10"]))
    ax_r.yaxis.set_minor_locator(NullLocator())
    ax_r.axhline(median_ratio, color="0.4", lw=0.9, ls="--", zorder=1)
    ax_r.grid(axis="y", color="0.92", lw=0.6, zorder=0)

    xt = [pos[(f, n)] for f, _, _ in families for n in orders]
    ax_r.set_xticks(xt)
    ax_r.set_xticklabels([str(n) for _ in families for n in orders], fontsize=6.5)
    ax_r.set_xlim(-0.8, pos[(families[-1][0], orders[-1])] + 0.8)
    for f, label, colour in families:
        xc = np.mean([pos[(f, n)] for n in orders])
        ax_r.annotate(label, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -14),
                      textcoords="offset points", ha="center", va="top", fontsize=7,
                      color=colour, fontweight="bold")
    for ax in (ax_w, ax_r):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_w.tick_params(labelbottom=False)
    if c_i == 0:
        ax_w.set_ylabel("Noise rate $r$")
        ax_r.set_ylabel("$r^\\star_{\\mathrm{best}}/r^\\star_{\\mathrm{worst}}$")
    ax_w.text(0.0, 1.03, "abc"[c_i], transform=ax_w.transAxes, fontweight="bold",
              fontsize=10, va="bottom", ha="left")
    ax_r.text(0.0, 1.03, "def"[c_i], transform=ax_r.transAxes, fontweight="bold",
              fontsize=10, va="bottom", ha="left")

handles = [
    Line2D([0], [0], marker="^", color="none", mfc="0.35", mec="white", ms=7, label="$r^\\star_{\\mathrm{best}}$"),
    Line2D([0], [0], marker="v", color="none", mfc="0.35", mec="white", ms=7, label="$r^\\star_{\\mathrm{worst}}$"),
    Line2D([0], [0], color="0.35", lw=4.5, alpha=0.45, label="Intermediate window"),
    Line2D([0], [0], color="0.4", lw=0.9, ls="--", label=f"Median ratio ({median_ratio:.1f})"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.17, wspace=0.18, hspace=0.12)
fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
print("saved", out_stem, "median ratio", round(median_ratio, 3))
