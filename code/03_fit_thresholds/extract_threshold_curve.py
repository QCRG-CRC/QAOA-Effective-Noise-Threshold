#!/usr/bin/env python3
"""Compute the threshold curves from the fitted logistic parameters, without refitting.

Produces : data/fits/threshold_curve.csv
Reads    : data/processed/qaoa_unified_dataset_new.csv, data/fits/logistic_fit_parameters.csv,
           data/fits/thresholds.csv (used only as a check)

The threshold extraction is the one of fit_logistic_thresholds.py (logistic_model, build_features,
threshold_scaled and the extraction block), applied to the parameters in
data/fits/logistic_fit_parameters.csv instead of to a new fit.

"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


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


# ---------------------------------------------------------------- loop over the released fits
df = pd.read_csv(DATA / "processed" / "qaoa_unified_dataset_new.csv")
df_logistic = pd.read_csv(DATA / "fits" / "logistic_fit_parameters.csv")
released = pd.read_csv(DATA / "fits" / "thresholds.csv")

tab_threshold = []
tab_threshold_curve = []
for _, row in df_logistic.iterrows():
    noise_type, graph_structure, node = row["Noise Type"], row["Graph Structure"], row["Node"]
    params = row[["v", "L", "b0", "b1", "b2", "b3", "b4", "b5"]].to_numpy(dtype=float)
    p_mean, p_std, x_mean, x_std = row[["p_mean", "p_std", "x_mean", "x_std"]].to_numpy(dtype=float)

    # same subset and transformation as fit_logistic_thresholds.py, needed for the admissible range of x_s
    df_subset = df[
        (df["Noise Type"] == noise_type) &
        (df["Structure"] == graph_structure) &
        (df["Node"] == node) &
        (df["Noise intensity"] > 0) &
        (df["Noise intensity"] <= 1)
    ]
    x_raw = -np.log10(df_subset["Noise intensity"].to_numpy(dtype=float))
    x_s = (x_raw - x_mean) / x_std

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
        for pp, xx, zz in zip(p_plot, x_plot, z_thr):
            tab_threshold_curve.append((noise_type, graph_structure, node, pp, xx, zz))

df_threshold = pd.DataFrame(tab_threshold, columns=["Noise Type", "Graph Structure", "Node", "QAOA Layers", "Threshold"])
df_threshold_curve = pd.DataFrame(
    tab_threshold_curve,
    columns=["Noise Type", "Graph Structure", "Node", "QAOA Layers", "Threshold (x)", "AR (z)"]
)

df_threshold_curve.to_csv(DATA / "fits" / "threshold_curve.csv", index=False)
print(f"wrote {len(df_threshold_curve)} curve points to data/fits/threshold_curve.csv")
