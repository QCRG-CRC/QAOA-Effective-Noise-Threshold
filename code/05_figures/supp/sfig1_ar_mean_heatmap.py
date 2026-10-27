#!/usr/bin/env python3
"""Supplementary Figure 1: mean AR heatmaps.

Produces : figures/supp/supp_ar_mean_heatmap.png
Run      : python sfig1_ar_mean_heatmap.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "supp"
FIGS.mkdir(parents=True, exist_ok=True)

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, brentq
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(DATA / "processed" / "qaoa_unified_dataset_new.csv")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

node_order = [6, 9, 12, 18]
structure_order = ['ER random', 'Path', 'Complete bipartite']
noise_types = ['AmplitudeDamping', 'Depolarization', 'BitFlip']
noise_dict_paper = {'AmplitudeDamping': 'Amplitude Damping', 'Depolarization': 'Depolarising', 'BitFlip': 'Bit-Flip'}
layer_order = sorted(df['# Layers'].unique())
noise_levels = sorted(df['Noise intensity'].unique())

n_rows = len(node_order)
n_cols_per_block = len(structure_order)
n_blocks = len(noise_types)

# width ratios: 3 real columns per block, then a thin spacer, repeated, no spacer after last block
width_ratios = []
for b in range(n_blocks):
    width_ratios += [1, 1, 1]
    if b < n_blocks - 1:
        width_ratios += [0.25]

fig = plt.figure(figsize=(16.5, 8.27))  # A3-ish landscape width, A4 height
gs = gridspec.GridSpec(n_rows, len(width_ratios), width_ratios=width_ratios, figure=fig, wspace=0.15, hspace=0.15)

cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])

col_start = {noise_types[b]: b * (n_cols_per_block + 1) for b in range(n_blocks)}

for b, noise_type in enumerate(noise_types):
    sub = df[df['Noise Type'] == noise_type]
    base_col = col_start[noise_type]

    for i, node in enumerate(node_order):
        for j, structure in enumerate(structure_order):
            ax = fig.add_subplot(gs[i, base_col + j])
            cell = sub[(sub['Node'] == node) & (sub['Structure'] == structure)]
            pivot = cell.pivot_table(index='Noise intensity', columns='# Layers', values='Approx. Ratio', aggfunc='mean')
            pivot = pivot.reindex(index=noise_levels, columns=layer_order)

            is_first_panel = (b == 0 and i == 0 and j == 0)
            sns.heatmap(pivot, ax=ax, vmin=0, vmax=1, cmap='viridis', cbar=is_first_panel, cbar_ax=(cbar_ax if is_first_panel else None), linewidths=0.2, linecolor='white')

            if i == 0:
                ax.set_title(structure, fontsize=9)
            if j == 0 and noise_type == 'AmplitudeDamping':
                ax.set_ylabel(f'n={node}\nNoise rate', fontsize=8)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if i == n_rows - 1:
                ax.set_xlabel('# Layers', fontsize=8)
            else:
                ax.set_xlabel('')

            ax.tick_params(labelsize=6)

    # noise-type label centred above its block
    mid_col = base_col + n_cols_per_block / 2 - 0.5
    fig.text((mid_col + 0.5) / len(width_ratios) * 0.92 + 0.02, 0.965, noise_dict_paper[noise_type], ha='center', fontsize=11, fontweight='bold')

# fig.suptitle('Mean approximation ratio across noise channels, graph sizes and structures', fontsize=13, y=1.02)
fig.savefig(FIGS / 'supp_ar_mean_heatmap.png', dpi=200, bbox_inches='tight')
