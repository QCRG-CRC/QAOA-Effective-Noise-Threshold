#!/usr/bin/env python3
"""Fit the generalised logistic surface to every (noise channel, graph family, graph order) and extract the layerwise noise thresholds.

Produces : results/fits/{logistic_fit_parameters,thresholds,threshold_curve}.csv and
           figures/supp/3D_logistic_minlogr_fit_all_data.png (Supplementary Figure 3)
Run      : python fit_logistic_thresholds.py   (from any working directory)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "supp"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "fits"
OUT.mkdir(parents=True, exist_ok=True)

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

from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def logistic_model(X, v, L, b0, *b_coefficients):
    """
    X: standardised feature matrix, shape (n_features, n_samples)
    v: lower asymptote
    L: amplitude (upper asymptote = v + L)
    b0: intercept
    *b_coefficients: b1..b5, matching (p_s, x_s, p_s^2, x_s^2, p_s*x_s)
    """
    b_coefficients = np.asarray(b_coefficients)
    z = b0 + np.dot(b_coefficients, X)
    z = np.clip(z, -35, 35)
    return v + L / (1 + np.exp(-z))

def build_features(p, x):
    p = np.asarray(p, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.vstack([p, x, p**2, x**2, p * x])

def threshold_scaled(p_s, thetas):
    """
    Solve b0 + b1*p_s + b2*x_s + b3*p_s^2 + b4*x_s^2 + b5*p_s*x_s = 0 for x_s,
    given a fixed (standardised) depth p_s. Returns np.nan if no physically
    valid real root exists.
    """
    b0, b1, b2, b3, b4, b5 = thetas
    a = b4
    be = b2 + b5 * p_s
    ce = b0 + b1 * p_s + b3 * p_s**2

    if np.isclose(a, 0):
        if np.isclose(be, 0):
            return np.nan
        return -ce / be

    D = be**2 - 4 * a * ce
    if D < 0:
        return np.nan

    sqrtD = np.sqrt(D)
    roots = [(-be + sqrtD) / (2 * a), (-be - sqrtD) / (2 * a)]
    return roots  # both candidates; caller filters against the valid x_s range

n_nodes = 4
n_structures = 3
n_noise = 3
n_layers = 10

elev, azim = 15, 315
fig = plt.figure(figsize=(7 * n_nodes, 5 * n_structures * n_noise))
ctr = 1
tab_logistic = []
tab_threshold = []
tab_threshold_curve = []
noise_dict_paper = {'AmplitudeDamping': 'Amplitude Damping', 'Depolarization': 'Depolarising', 'BitFlip': 'Bit-Flip'}

for n, noise_type in enumerate(df["Noise Type"].unique()):
    for j, graph_structure in enumerate(df["Structure"].unique()):
        for k, node in enumerate(df["Node"].unique()):
            ax = fig.add_subplot(n_structures * n_noise, n_nodes, ctr, projection='3d')
            ctr += 1

            df_subset = df[
                (df["Noise Type"] == noise_type) &
                (df["Structure"] == graph_structure) &
                (df["Node"] == node) &
                (df["Noise intensity"] > 0) &
                (df["Noise intensity"] <= 1)
            ]

            df_subset_pivot = pd.pivot_table(
                df_subset, values='Approx. Ratio',
                index=['Noise intensity', '# Layers'], aggfunc=np.mean
            )
            x_agg = df_subset_pivot.index.get_level_values(1)  # layers
            y_agg = -np.log10(df_subset_pivot.index.get_level_values(0))  # -log10(r)
            z_agg = df_subset_pivot["Approx. Ratio"]

            p_raw = df_subset["# Layers"].to_numpy(dtype=float)
            x_raw = -np.log10(df_subset["Noise intensity"].to_numpy(dtype=float))
            y = df_subset["Approx. Ratio"].to_numpy(dtype=float)

            p_mean, p_std = p_raw.mean(), p_raw.std()
            x_mean, x_std = x_raw.mean(), x_raw.std()
            p_std = p_std if p_std > 1e-8 else 1.0
            x_std = x_std if x_std > 1e-8 else 1.0

            p_s = (p_raw - p_mean) / p_std
            x_s = (x_raw - x_mean) / x_std
            X = build_features(p_s, x_s)

            # v, L, b0, *b_coefficients
            lower_bounds = [0, 0, -np.inf, -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,]
            upper_bounds = [ 2, 2,  np.inf,  np.inf,np.inf,np.inf,np.inf,np.inf]
            po = [0.0, 1.0, 0.0, 1.0, -1.0, -0.1, -0.5, 0.0]

            try:
                popt, pcov = curve_fit(
                    logistic_model, X, y,
                    p0=po,
                    bounds=(lower_bounds, upper_bounds),
                    method='trf',
                    loss='soft_l1',
                    # f_scale=0.05,
                    # x_scale='jac',
                    max_nfev=20000 * X.shape[1],
                )
            except RuntimeError as e:
                print(f"Fit failed for {noise_type}, {graph_structure}, node={node}: {e}")
                ax.set_title(f"{graph_structure} order {node}\n{noise_type} noise\nFIT FAILED", y=0.95)
                ax.view_init(elev=elev, azim=azim)
                continue

            params = popt
            y_pred = logistic_model(X, *params)
            rmse = root_mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            print(f"Noise type = {noise_type}, Graph structure = {graph_structure}, Node = {node}, Fitted parameters: {params}, RMSE: {rmse}, MAE: {mae}")
            tab_logistic.append((
                noise_type, graph_structure, node, *params, rmse, mae,
                p_mean, p_std, x_mean, x_std
            ))

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
            ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.5, edgecolor='k', linewidth=0.5)
            ax.scatter(x_agg, y_agg, z_agg, marker='o', label='Data Points', color='black', s=5)

            for ii in range(0, len(x_agg), n_layers):
                ax.plot3D(x_agg[ii:ii+n_layers], y_agg[ii:ii+n_layers], z_agg[ii:ii+n_layers], color='black', linewidth=0.5)
            for ii in range(0, n_layers):
                ax.plot3D(x_agg[ii:len(x_agg):n_layers], y_agg[ii:len(x_agg):n_layers], z_agg[ii:len(x_agg):n_layers], color='black', linewidth=0.5)

            # threshold extraction: solve in standardised space, then unscale
            thetas = params[2:]
            p_range = np.linspace(1, 10, 91)
            x_s_min, x_s_max = x_s.min(), x_s.max()

            thresholds_raw = []
            for p in p_range:
                p_scaled = (p - p_mean) / p_std
                roots = threshold_scaled(p_scaled, thetas)
                if isinstance(roots, float):  # np.nan or single linear-case root
                    candidates = [roots]
                else:
                    candidates = roots
                valid = [r for r in candidates if np.isfinite(r) and x_s_min <= r <= x_s_max]
                if not valid:
                    thresholds_raw.append(np.nan)
                    continue
                mid = 0.5 * (x_s_min + x_s_max)
                x_s_best = min(valid, key=lambda r: abs(r - mid))
                thresholds_raw.append(x_s_best * x_std + x_mean)  # unscale to raw x

            thresholds_raw = np.array(thresholds_raw)
            mask = np.isfinite(thresholds_raw)

            for p, x_thr in zip(p_range[mask], thresholds_raw[mask]):
                t = 10 ** (-x_thr)  # back to noise rate
                if p - int(p) < 1e-6:  # only record integer layer counts
                    tab_threshold.append((noise_type, graph_structure, node, p, t))

            if mask.any():
                p_plot = p_range[mask]
                x_plot = thresholds_raw[mask]
                p_plot_s = (p_plot - p_mean) / p_std
                x_plot_s = (x_plot - x_mean) / x_std
                X_thr_s = build_features(p_plot_s, x_plot_s)
                z_thr = logistic_model(X_thr_s, *params)
                ax.plot3D(p_plot, x_plot, z_thr, color='red', linewidth=2)
                for pp, xx, zz in zip(p_plot, x_plot, z_thr):
                    tab_threshold_curve.append((noise_type, graph_structure, node, pp, xx, zz))

            ax.set_xlabel('Number of QAOA Layers')
            ax.set_ylabel('\n-Log (Noise Rate)')
            ax.set_zlabel('Approx. Ratio')
            ax.set_title(f"{graph_structure} order {node}\nwith {noise_dict_paper[noise_type]} noise\nMAE = {mae:.4f}", y=0.95)
            ax.view_init(elev=elev, azim=azim)

df_logistic_minlogr = pd.DataFrame(
    tab_logistic,
    columns=["Noise Type", "Graph Structure", "Node", "v", "L",
             "b0", "b1", "b2", "b3", "b4", "b5", "RMSE", "MAE",
             "p_mean", "p_std", "x_mean", "x_std"]
)
df_threshold_minlogr = pd.DataFrame(
    tab_threshold,
    columns=["Noise Type", "Graph Structure", "Node", "QAOA Layers", "Threshold"]
)
df_threshold_curve_minlogr = pd.DataFrame(
    tab_threshold_curve,
    columns=["Noise Type", "Graph Structure", "Node", "QAOA Layers", "Threshold (x)", "AR (z)"]
)
df_threshold_curve_minlogr.to_csv(OUT / "threshold_curve.csv", index=False)
df_logistic_minlogr.to_csv(OUT / "logistic_fit_parameters.csv", index=False)
df_threshold_minlogr.to_csv(OUT / "thresholds.csv", index=False)

plt.tight_layout()
plt.savefig(FIGS / "3D_logistic_minlogr_fit_all_data.png")
