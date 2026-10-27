#!/usr/bin/env python3
"""Figure 3: AR against noise rate and depth, and the AR surfaces, for the 18-vertex complete bipartite graph K_{9,9}.

Produces : figures/main/2D_noise_bipartite_18.png
Run      : python fig3_ar_landscape_K99.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "main"
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

import matplotlib as mpl
from numpy.linalg import norm
elev, azim = 15, 315
n_layers = 10

color_dict = {'Depolarization': 'red', 'BitFlip': 'green', 'AmplitudeDamping': 'blue'}
noise_dict = {'Depolarization' : "Depolarising", 'BitFlip': "Bit Flip", 'AmplitudeDamping': "Amplitude Damping"}
structure_dict = {'ER random': "Erdős-Rényi Random", 'Path': "Path", 'Complete bipartite': "Complete Balanced Bipartite"}

df_bipa = df[(df["Structure"] == "Complete bipartite") & (df["Node"] == 18)]

fig = plt.figure(figsize=(6 * len(df["Noise Type"].unique()), 4 * 3))
for n, noise in enumerate(df["Noise Type"].unique()):
    ax1, ax2, ax3 = fig.add_subplot(3, 3, n+1), fig.add_subplot(3, 3, n+1 + 3), fig.add_subplot(3, 3, n+1 + 6, projection='3d')
    df_subset = df_bipa[(df_bipa["Noise Type"] == noise) & (df_bipa["Noise intensity"]>0)]
    
    sns.lineplot(data=df_subset, hue="# Layers", y="Approx. Ratio", x="Noise intensity", ax = ax1, palette = "plasma")
    ax1.set_title(f"({chr(ord('a')+n)}) AR vs Noise Rate for {noise_dict[noise]} Noise", y=-0.25)#, pad = -10)
    ax1.set_xscale('log')

    norm = mpl.colors.LogNorm(vmin=df_subset['Noise intensity'].min(), vmax=df_subset['Noise intensity'].max())
    sns.lineplot(data=df_subset, x="# Layers", y="Approx. Ratio", hue="Noise intensity", hue_norm=norm, palette='viridis', ax=ax2, legend=False)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, label='Noise Rate (Log Scale)')
    cbar.set_label('Noise Rate', size=8)
    ax2.set_title(f"({chr(ord('a')+n+3)}) AR vs QAOA Layers for {noise_dict[noise]} Noise", y=-0.25)

    # generate 3D plot
    df_subset = df_bipa[(df_bipa["Noise Type"] == noise) & (df_bipa["Noise intensity"]>0) & (df_bipa["Noise intensity"]<=0.1)]
    df_subset_pivot = pd.pivot_table(df_subset, values='Approx. Ratio', index=['Noise intensity', '# Layers'], aggfunc=np.mean)
    x_agg = df_subset_pivot.index.get_level_values(1)
    y_agg = -np.log10(1*(df_subset_pivot.index.get_level_values(0)))#/np.log10(15)
    z_agg = df_subset_pivot["Approx. Ratio"]
    
    scat_agg = ax3.scatter(x_agg, y_agg, z_agg, color = color_dict[noise], marker='o')# edgecolor='k', alpha=0.7, label='Data Points')
    surf1 = ax3.plot_trisurf(x_agg, y_agg, z_agg, color=color_dict[noise], alpha=0.4)
    for i in range(0, len(x_agg), n_layers):
        ax3.plot3D(x_agg[i:i+n_layers], y_agg[i:i+n_layers], z_agg[i:i+n_layers], color='black', linewidth=0.5)
    for i in range(0, n_layers):
        ax3.plot3D(x_agg[i:len(x_agg):n_layers], y_agg[i:len(x_agg):n_layers], z_agg[i:len(x_agg):n_layers], color='black', linewidth=0.5)

    ax3.set_xlabel('Number of QAOA Layers')
    ax3.set_ylabel('\n-Log (Noise Rate)')
    ax3.set_zlabel('Approx. Ratio')
    ax3.set_title(f"({chr(ord('a')+n+6)}) 3D AR surfaces for {noise_dict[noise]} Noise", y=-0.25)
    ax3.view_init(elev=elev, azim=azim)  # Adjust the viewing angle for better visualization
    ax3.set_box_aspect((1.1, 1.8, 1))
plt.tight_layout(h_pad=3.0)
plt.savefig(FIGS / "2D_noise_bipartite_18")
