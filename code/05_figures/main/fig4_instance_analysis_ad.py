#!/usr/bin/env python3
"""Figure 4: AR against depth under amplitude damping, across graph family and graph order.

Produces : figures/main/2D_instance_analysis_AmplitudeDamping.png
Run      : python fig4_instance_analysis_ad.py   (from any working directory)
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

noise_dict = {'Depolarization' : "Depolarising", 'BitFlip': "Bit Flip", 'AmplitudeDamping': "Amplitude Damping"}
structure_dict = {'ER random': "Erdős-Rényi Random", 'Path': "Path", 'Complete bipartite': "Complete Balanced Bipartite"}

noise_type = "AmplitudeDamping"
noise_rate_chosen = [0.01, 0.05, 0.1, 0.5]
node_chosen = [6, 12, 18]
df_analyse = df[(df["Noise Type"] == noise_type) & (df["Noise intensity"].isin(noise_rate_chosen)) & (df["Node"].isin(node_chosen))]
df_analyse["Structure"] = df_analyse["Structure"].replace({name : structure_dict[name] + " graph" for name in structure_dict.keys()})
g = sns.catplot(data=df_analyse, row = "Node",x="# Layers", hue = "Structure", y="Approx. Ratio", col = "Noise intensity", kind="point", height=3, aspect=1.7)
sns.move_legend(
    g, 
    "lower center",
    bbox_to_anchor=(0.5, -0.02), # (x, y) coordinates
    ncol=3,                      # Adjust number of columns
    title=None,                  # Optional: remove title
    frameon=False                # Optional: remove box
)
plt.savefig(FIGS / "2D_instance_analysis_AmplitudeDamping.png", bbox_inches='tight')
