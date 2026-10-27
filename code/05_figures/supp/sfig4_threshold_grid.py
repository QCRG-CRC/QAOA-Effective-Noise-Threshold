#!/usr/bin/env python3
"""Supplementary Figure 4: layerwise thresholds r*(p) for all configurations.

Produces : figures/supp/supp_threshold_grid.png
Run      : python sfig4_threshold_grid.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "supp"
FIGS.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = FIGS
df_threshold_minlogr = pd.read_csv(DATA / "fits" / "thresholds.csv")


plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'legend.title_fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.dpi': 300})

CH = ['AmplitudeDamping', 'Depolarization', 'BitFlip']
CH_LABEL = {'AmplitudeDamping': 'Amplitude damping', 'Depolarization': 'Depolarising', 'BitFlip': 'Bit-flip'}
CH_COL = {'AmplitudeDamping': '#1f4fd8', 'Depolarization': '#d62728', 'BitFlip': '#2a8f2a'}
CH_MARK = {'AmplitudeDamping': 'o', 'Depolarization': 's', 'BitFlip': '^'}
FAM = ['ER random', 'Path', 'Complete bipartite']
FAM_LABEL = {'ER random': 'Erdős-Rényi random', 'Path': 'Path', 'Complete bipartite': 'Complete bipartite'}
FAM_MARK = {'ER random': 'o', 'Path': 's', 'Complete bipartite': '^'}
NS = [6, 9, 12, 18]
ps = np.arange(1, 11)

# ---------------------------------------------------------------- data
thr = df_threshold_minlogr.rename(columns={'Noise Type': 'ch', 'Graph Structure': 'fam', 'Node': 'n',
                                           'QAOA Layers': 'p', 'Threshold': 'r'})
thr['p'] = thr['p'].astype(int)
P = thr.pivot_table(index=['ch', 'fam', 'n'], columns='p', values='r')   # r*(p), one row per configuration

extremes = pd.DataFrame({'r_best': P.max(axis=1), 'p_best': P.idxmax(axis=1),
                         'r_worst': P.min(axis=1), 'p_worst': P.idxmin(axis=1)}).reset_index()
B = extremes.set_index(['ch', 'fam', 'n'])['r_best']
W = extremes.set_index(['ch', 'fam', 'n'])['r_worst']


# ---------------------------------------------------------------- supplementary grid
fig, axes = plt.subplots(len(NS), 3, figsize=(7.4, 8.0), sharex=True, sharey=True, constrained_layout=True)
for i, n in enumerate(NS):
    for j, fam in enumerate(FAM):
        axx = axes[i, j]
        for ch in CH:
            axx.plot(ps, P.loc[(ch, fam, n)].values, color=CH_COL[ch], marker=CH_MARK[ch], ms=3, lw=1.3)
        axx.set_yscale('log'); axx.set_xticks(ps); axx.set_ylim(2e-3, 0.7)
        if i == 0: axx.set_title(FAM_LABEL[fam])
        if j == 0: axx.set_ylabel(f'$n={n}$\n' + r'$r^\star(p)$')
        if i == len(NS) - 1: axx.set_xlabel('QAOA depth $p$')
fig.legend(handles=[Line2D([], [], color=CH_COL[c], marker=CH_MARK[c], ms=4, lw=1.3, label=CH_LABEL[c]) for c in CH],
           loc='outside lower center', ncol=3, frameon=False)
fig.savefig(OUT / 'supp_threshold_grid.png')

print('done')
