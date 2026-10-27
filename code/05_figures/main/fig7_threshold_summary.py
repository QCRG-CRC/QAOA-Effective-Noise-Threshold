#!/usr/bin/env python3
"""Figure 7: dependence of the threshold on depth, channel and graph order (2 x 3 panels).

Produces : figures/main/threshold_summary.{pdf,png} and data/fits/best_worst_case_thresholds.csv
Run      : python fig7_threshold_summary.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "main"
FIGS.mkdir(parents=True, exist_ok=True)


"""Threshold summary figure, 2 x 3 panels plus a right-hand legend column.

Columns: one noise channel each (amplitude damping, depolarising, bit-flip).
Row 1, (a) to (c): r*(p)/r*(1) against depth p.
    hue and marker shape = graph family, marker size = graph order,
    thick black line = median of the 12 configurations of that channel,
    thick grey dashed line = global median of all 36 configurations (identical in the three panels).
Row 2, (d) to (f): best and worst thresholds over depth against graph order n (log axis).
    r*_best solid with filled markers, r*_worst dashed with open markers, gap shaded in the family hue.
Input: thresholds.csv (Noise Type, Graph Structure, Node, QAOA Layers, Threshold)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

df_threshold_minlogr = pd.read_csv(DATA / "fits" / "thresholds.csv")
OUT = FIGS

FIG_H_MM = 100      # figure height in mm (width fixed at 180 mm, Nature double column)
HSPACE = 0.50       # vertical gap between rows, as a fraction of the mean axes height
LABEL_Y = -0.27     # vertical position of "(a)" to "(f)" in axes coordinates

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.title_fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "lines.linewidth": 1.0, "lines.markersize": 3.0, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

CH = ['AmplitudeDamping', 'Depolarization', 'BitFlip']
CH_LABEL = {'AmplitudeDamping': 'Amplitude damping', 'Depolarization': 'Depolarising', 'BitFlip': 'Bit-flip'}
FAM = ['ER random', 'Path', 'Complete bipartite']
FAM_LABEL = {'ER random': 'Erdős-Rényi random', 'Path': 'Path', 'Complete bipartite': 'Complete bipartite'}
FAM_COL = {'ER random': '#2a78d6', 'Path': '#eb6834', 'Complete bipartite': '#1baf7a'}
FAM_MARK = {'ER random': 'o', 'Path': 's', 'Complete bipartite': '^'}
NS = [6, 9, 12, 18]
MS = {6: 1.6, 9: 2.5, 12: 3.4, 18: 4.6}     # marker size encodes graph order in (a) to (c)
ps = np.arange(1, 11)
INK, MUTED = '#2b2b2b', '#8a8984'


def panel_label(ax, s):
    # centred below the axes, under the x-axis label
    ax.text(0.5, LABEL_Y, f"({s})", transform=ax.transAxes, fontsize=8, ha="center", va="top")


# ---------------------------------------------------------------- data
thr = df_threshold_minlogr.rename(columns={'Noise Type': 'ch', 'Graph Structure': 'fam', 'Node': 'n',
                                           'QAOA Layers': 'p', 'Threshold': 'r'})
thr['p'] = thr['p'].astype(int)
P = thr.pivot_table(index=['ch', 'fam', 'n'], columns='p', values='r')        # r*(p), one row per configuration
REL = P.div(P[1], axis=0)                                                       # r*(p)/r*(1)
GLOBAL_MED = REL.median(axis=0)                                                 # median over all 36 configurations
ext = pd.DataFrame({'r_best': P.max(axis=1), 'p_best': P.idxmax(axis=1),
                    'r_worst': P.min(axis=1), 'p_worst': P.idxmin(axis=1)})
ext['ratio'] = ext['r_best'] / ext['r_worst']
ext.reset_index().to_csv(DATA / 'fits' / 'best_worst_case_thresholds.csv', index=False)

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(180 / 25.4, FIG_H_MM / 25.4))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.77, bottom=0.125, top=0.955, wspace=0.12, hspace=HSPACE)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
for j in range(1, 3):
    axes[0, j].sharey(axes[0, 0]); axes[1, j].sharey(axes[1, 0])

# ---------------------------------------------------------------- (a) to (c)
for j, ch in enumerate(CH):
    ax = axes[0, j]
    ax.axhline(1, color=MUTED, lw=0.6, ls=':', zorder=0)
    for fam in FAM:
        for n in NS:
            ax.plot(ps, REL.loc[(ch, fam, n)].values, color=FAM_COL[fam], marker=FAM_MARK[fam], ms=MS[n],
                    lw=0.7, alpha=0.75, zorder=2)
    ax.plot(ps, GLOBAL_MED.values, color=MUTED, lw=1.8, ls='--', zorder=3)
    ax.plot(ps, REL.loc[ch].median(axis=0).values, color=INK, lw=1.8, zorder=4)
    ax.set_title(CH_LABEL[ch], pad=3)
    ax.set_xticks(ps); ax.set_xlim(0.6, 10.4)
    ax.set_xlabel('QAOA depth $p$')
ax0 = axes[0, 0]
ax0.set_yscale('log'); ax0.set_ylim(0.1, 3.0)
ax0.set_yticks([0.1, 0.2, 0.5, 1, 2]); ax0.set_yticklabels(['0.1', '0.2', '0.5', '1', '2']); ax0.minorticks_off()
ax0.set_ylabel(r'$r^\star(p)\,/\,r^\star(1)$')

# ---------------------------------------------------------------- (d) to (f)
for j, ch in enumerate(CH):
    ax = axes[1, j]
    E = ext.loc[ch]
    for fam in FAM:
        e = E.loc[fam].reindex(NS)
        col = FAM_COL[fam]
        ax.fill_between(NS, e['r_worst'], e['r_best'], color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot(NS, e['r_best'], color=col, marker=FAM_MARK[fam], ls='-', lw=1.0, ms=3.2, zorder=3)
        ax.plot(NS, e['r_worst'], color=col, marker=FAM_MARK[fam], ls='--', lw=1.0, ms=3.2, mfc='white',
                mew=0.8, zorder=3)
    ax.set_xticks(NS); ax.set_xlim(5, 19)
    ax.set_xlabel('Graph order $n$')
ax3 = axes[1, 0]
ax3.set_yscale('log'); ax3.set_ylim(2.5e-3, 0.6)
ax3.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3]); ax3.set_yticklabels(['0.003', '0.01', '0.03', '0.1', '0.3'])
ax3.minorticks_off()
ax3.set_ylabel(r'Threshold $r^\star$')

for ax, s in zip(axes.ravel(), 'abcdef'):
    panel_label(ax, s)
    if ax not in (axes[0, 0], axes[1, 0]):
        ax.tick_params(labelleft=False)

# ---------------------------------------------------------------- legend column (right of both rows)
h_fam = [Line2D([], [], color=FAM_COL[f], marker=FAM_MARK[f], ms=3.2, lw=1.0, label=FAM_LABEL[f]) for f in FAM]
h_ord = [Line2D([], [], color=MUTED, marker='o', ms=MS[n], lw=0, label=f'$n={n}$') for n in NS]
h_med = [Line2D([], [], color=INK, lw=1.8, label='Median, each channel (12)'),
         Line2D([], [], color=MUTED, lw=1.8, ls='--', label='Median, all channels (36)')]
h_thr = [Line2D([], [], color=INK, marker='o', ls='-', lw=1.0, ms=3.2,
                label=r'$r^\star_{\mathrm{best}}=\max_p r^\star(p)$'),
         Line2D([], [], color=INK, marker='o', ls='--', lw=1.0, ms=3.2, mfc='white', mew=0.8,
                label=r'$r^\star_{\mathrm{worst}}=\min_p r^\star(p)$'),
         Patch(facecolor=MUTED, alpha=0.3, lw=0, label='Gap between them')]

x_leg = 0.79
kw = dict(loc='upper left', alignment='left', handlelength=2.2, labelspacing=0.4, borderaxespad=0)
fig.legend(handles=h_fam, title='Graph family', bbox_to_anchor=(x_leg, 0.955), **kw)
fig.legend(handles=h_ord, title='Graph order (marker size,\npanels a to c)', bbox_to_anchor=(x_leg, 0.78),
           ncol=2, columnspacing=0.8, **kw)
fig.legend(handles=h_med, title='Panels a to c', bbox_to_anchor=(x_leg, 0.625), **kw)
fig.legend(handles=h_thr, title='Panels d to f', bbox_to_anchor=(x_leg, 0.44), **kw)

fig.savefig(OUT / 'threshold_summary.pdf'); fig.savefig(OUT / 'threshold_summary.png', dpi=300)
