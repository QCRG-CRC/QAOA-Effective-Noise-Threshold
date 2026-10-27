#!/usr/bin/env python3
"""Figure 6: fitted logistic surfaces and layerwise thresholds for the path family under depolarising noise.

Produces : figures/main/logistic_fit_thresholds_snapshot.png
Run      : python fig6_fit_snapshot_path_depol.py   (from any working directory)
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

df_threshold_curve_minlogr = pd.read_csv(DATA / "fits" / "threshold_curve.csv")
df_logistic_minlogr=pd.read_csv(DATA / "fits" / "logistic_fit_parameters.csv")
df_threshold_minlogr=pd.read_csv(DATA / "fits" / "thresholds.csv")
n_layers = 10

# Figure 6 code

elev, azim = 15, 315
chosen_rate = [0.0010, 0.0050, 0.0100,  0.0500, 0.100, 0.5000]
x_limit, y_limit = (1, 10), (0, 4)
chosen_nodes = [9, 12, 18]

graph_structure = "Path"
noise_type = "Depolarization"
df_subset_logistic = df_logistic_minlogr[(df_logistic_minlogr["Graph Structure"] == graph_structure) & (df_logistic_minlogr["Noise Type"] == noise_type)]

# plot 3D surface of logistic function for this subset
# x_logistic = np.linspace(x_limit[0], x_limit[1], 100)
# y_logistic = np.linspace(y_limit[0], y_limit[1], 100)
# X, Y = np.meshgrid(x_logistic, y_logistic)
# X2, Y2 = X**2, Y**2
n_rows = 2
n_cols = len(chosen_nodes)
fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
for i, node in enumerate(chosen_nodes):
    df_subset_node = df_subset_logistic[df_subset_logistic["Node"] == node]
    df_subset = df[(df["Noise Type"] == noise_type) & (df["Structure"] == graph_structure) & (df["Node"] == node) & (df["Noise intensity"].isin(chosen_rate))]
    df_subset_pivot = pd.pivot_table(df_subset, values='Approx. Ratio', index=['Noise intensity', '# Layers'], aggfunc=np.mean)
    df_threshold_node = df_threshold_minlogr[(df_threshold_minlogr["Noise Type"] == noise_type) & (df_threshold_minlogr["Graph Structure"] == graph_structure) & (df_threshold_minlogr["Node"] == node)]
    x_agg = df_subset_pivot.index.get_level_values(1)
    y_agg = -np.log10(df_subset_pivot.index.get_level_values(0))  
    z_agg = df_subset_pivot["Approx. Ratio"].to_numpy()

    params = df_subset_node.iloc[0][["v", "L", "b0", "b1", "b2", "b3", "b4", "b5"]].to_numpy()
    p_mean, p_std, x_mean, x_std = df_subset_node.iloc[0][["p_mean", "p_std", "x_mean", "x_std"]].to_numpy()

    # plotting the fitted surface, in raw (unstandardised) coordinates
    x_logistic = np.linspace(1, 10, 91)
    y_logistic = np.linspace(np.min(y_agg), np.max(y_agg), 101)
    Xg, Yg = np.meshgrid(x_logistic, y_logistic)
    Xg_s = (Xg - p_mean) / p_std
    Yg_s = (Yg - x_mean) / x_std
    X2g, Y2g = Xg_s**2, Yg_s**2
    Zg = params[0] + params[1] / (1 + np.exp(-np.clip(
        params[2] + params[3]*Xg_s + params[4]*Yg_s + params[5]*X2g + params[6]*Y2g + params[7]*Xg_s*Yg_s,
        -35, 35
    )))

    
    # Z = params[0]  + params[1] / (1 + np.exp(-(params[2] + params[3]*X + params[4]*Y + params[5]*X2 + params[6]*Y2 + params[7]*X*Y)))

    # thetas = params[2:]  # Extract b0, b1, b2, ...
    # p_range = np.linspace(1, 10, 100)
    # # thresholds = np.array([threshold(p, thetas) for p in p_range])
    # thresholds = df_threshold_node.set_index("QAOA Layers").reindex(p_range)["Threshold"].to_numpy()

    # X_thresholds = np.array([p_range, thresholds, p_range**2, thresholds**2, p_range*thresholds])
    # z_thresholds = logistic_model(X_thresholds, *params)
    df_threshold_curve_node = df_threshold_curve_minlogr[
        (df_threshold_curve_minlogr["Noise Type"] == noise_type) &
        (df_threshold_curve_minlogr["Graph Structure"] == graph_structure) &
        (df_threshold_curve_minlogr["Node"] == node)
    ].sort_values("QAOA Layers")

    p_range = df_threshold_curve_node["QAOA Layers"].to_numpy()
    thresholds = df_threshold_curve_node["Threshold (x)"].to_numpy()
    z_thresholds = df_threshold_curve_node["AR (z)"].to_numpy()
    
    ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
    ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha = 0.5, edgecolor='k', linewidth=0.1, label='Logistic Fit')
    ax.scatter(x_agg, y_agg, z_agg, marker='o', label='Data Points', color='black', s=7.5, alpha=0.7)
    
    # for t in range(len(p_range)):
    #     if thresholds[t] < y_limit[0] or thresholds[t] > y_limit[1] * 1.1:
    #         continue
    #     threshold_line[0].append(p_range[t])
    #     threshold_line[1].append(thresholds[t])
    #     threshold_line[2].append(z_thresholds[t])
    threshold_line = [[], [], []]
    for t in range(len(p_range)):
        if thresholds[t] < y_limit[0] or thresholds[t] > y_limit[1] * 1.1:
            continue
        threshold_line[0].append(p_range[t])
        threshold_line[1].append(thresholds[t])
        threshold_line[2].append(z_thresholds[t])
    ax.plot3D(threshold_line[0], threshold_line[1], threshold_line[2], color='red', linewidth=3, label='Noise Threshold')

    for ii in range(0, len(x_agg), n_layers):
        ax.plot3D(x_agg[ii:ii+n_layers], y_agg[ii:ii+n_layers], z_agg[ii:ii+n_layers], color='black', linewidth=0.5)
    for ii in range(0, n_layers):
        ax.plot3D(x_agg[ii:len(x_agg):n_layers], y_agg[ii:len(x_agg):n_layers], z_agg[ii:len(x_agg):n_layers], color='black', linewidth=0.5)
    ax.set_xlabel('Number of QAOA Layers')
    ax.set_ylabel('\n-Log (Noise Rate)')
    ax.set_zlabel('Approx. Ratio')
    ax.set_title(f"({chr(ord('a')+i)}) Logistic fit for graph of order {node} (MAE = {df_subset_node.iloc[0]['MAE']:.4f})", y=-0.01)
    ax.legend(loc='upper right')
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    ax.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(n_rows, n_cols, i+1+n_cols)
    picked_layers = [2, 5, 9]
    # picked_threshold = [1 - 10 ** (-threshold(picked_layer, thetas) ) for picked_layer in picked_layers]
    picked_threshold = df_threshold_node.set_index("QAOA Layers").reindex(picked_layers)["Threshold"].to_numpy()
    # df_subset = df_all[(df_all["Noise Type"] == noise_type) & (df_all["graph_structure"] == graph_structure) & (df_all["Node"] == node) & (df_all["Noise intensity"]>0) & (df_all["Noise intensity"]<=0.1)]
    df_subset_layer = df_subset[df_subset["# Layers"].isin(picked_layers)]
    df_subset_layer["# Layers"].replace({picked_layer: f"QAOA with {picked_layer} Layers" for picked_layer in picked_layers}, inplace=True)
    sns.lineplot(data=df_subset_layer, x="Noise intensity", y="Approx. Ratio", palette="viridis", ax=ax2, hue="# Layers")
    line_styles = ['dashed', 'dashdot', 'dotted']
    for j, (threshold_val, layer) in enumerate(zip(picked_threshold, picked_layers)):
        ax2.vlines(x=threshold_val, ymin=0.8 * min(df_subset_layer["Approx. Ratio"]), ymax=1, colors='red', linestyles=line_styles[j], label=f'threshold for {layer} layers')
    ax2.set_xscale('log')
    ax2.set_xlabel("Noise Rate")
    ax2.set_title(f"({chr(ord('a')+i+n_cols)}) Layerwise Noise Thresholds for graph of order {node}", y=-0.2)
    ax2.set_box_aspect(0.7)
    ax2.legend()
plt.subplots_adjust(hspace=0.02) 
plt.tight_layout(h_pad=0.1)
plt.savefig(FIGS / "logistic_fit_thresholds_snapshot.png")
