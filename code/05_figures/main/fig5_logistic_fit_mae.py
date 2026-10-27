#!/usr/bin/env python3
"""Figure 5: mean absolute error of the fitted logistic surfaces.

Produces : figures/main/logistic_fit_mae.png
Run      : python fig5_logistic_fit_mae.py   (from any working directory)
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

df_threshold_curve_minlogr = pd.read_csv(DATA / "fits" / "threshold_curve.csv")
df_logistic_minlogr=pd.read_csv(DATA / "fits" / "logistic_fit_parameters.csv")
df_threshold_minlogr=pd.read_csv(DATA / "fits" / "thresholds.csv")

noise_dict = {'Depolarization' : "Depolarising", 'BitFlip': "Bit Flip", 'AmplitudeDamping': "Amplitude Damping"}
structure_dict = {'ER random': "Erdős-Rényi Random", 'Path': "Path", 'Complete bipartite': "Complete Balanced Bipartite"}

print(df_logistic_minlogr.head())
df_logistic_pivot = df_logistic_minlogr.pivot_table(index=["Graph Structure", "Node"], columns="Noise Type", values="MAE").reset_index()
print(df_logistic_pivot)
df_logistic_plot = df_logistic_minlogr.replace({"Graph Structure": structure_dict, "Noise Type": noise_dict})
sns.catplot(data=df_logistic_plot, x="Graph Structure", y="MAE", col="Noise Type", kind="bar", hue="Node", height=4, aspect=1.6)
plt.savefig(FIGS / "logistic_fit_mae.png")
