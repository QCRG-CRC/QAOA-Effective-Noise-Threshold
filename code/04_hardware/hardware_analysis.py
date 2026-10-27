#!/usr/bin/env python3
"""Hardware section: threshold-based assessment of ibm_marrakesh (Table 1, Figure 8 and the numbers quoted in the text).

Produces : figures/main/fig_hardware_threshold.{pdf,png} (Figure 8),
           data/hardware/hardware_section_numbers.csv (every number the hardware section quotes),
           data/hardware/ibm_marrakesh_calibration_*.csv (the calibration used for Table 1 and r_dev)
Reads    : data/hardware/ibm_marrakesh_ar.csv, data/processed/qaoa_unified_dataset_new.csv,
           data/fits/thresholds.csv, data/fits/logistic_fit_parameters.csv
Run      : python hardware_analysis.py   (needs requirements-hardware.txt; no IBM account is needed)
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HW_DATA = DATA / "hardware"
FIGS = ROOT / "figures" / "main"
FIGS.mkdir(parents=True, exist_ok=True)
display = print

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda v: f"{v:.4g}")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Calibration -------------------------------------------------------------------------
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN")
IBM_INSTANCE = None            # optional instance CRN
BACKEND_NAME = "ibm_marrakesh"
# Calibration date of the hardware run. With live access, IBM returns the calibration
# valid at this time. None uses the current calibration (live) or the bundled snapshot (offline).
CALIBRATION_DATETIME = None    # e.g. "2026-03-13 12:00"
USE_LIVE_CALIBRATION = os.environ.get("USE_LIVE_CALIBRATION") == "1"

# --- Simulation outputs ------------------------------------------------------------------
DATASET_CSV = DATA / "processed" / "qaoa_unified_dataset_new.csv"
THRESHOLDS_CSV = DATA / "fits" / "thresholds.csv"
# How the threshold column is stored: "r", "minlog10" (-log10 r) or "minln" (-ln r).
THRESHOLD_SCALE = "r"
INSTANCE = dict(family="ER random", n=18)

# --- Hardware run ------------------------------------------------------------------------
QUANTUM_AR = pd.read_csv(HW_DATA / "ibm_marrakesh_ar.csv").sort_values("QAOA Layers")["Approx. Ratio"].tolist()   # p = 1, ..., 10

# --- Fitted logistic-surface parameters -------------------------------------------------
# One row per (noise channel, graph family, graph order). The lower asymptote y0 is stored as
# column "v"; b0 to b5 are the coefficients of (1, p~, x~, p~^2, x~^2, p~ x~) with x = -log10 r,
# standardised with p_mean, p_std, x_mean, x_std.
FIT_PARAMS_CSV = DATA / "fits" / "logistic_fit_parameters.csv"
FIT_COLS = dict(channel="Noise Type", family="Graph Structure", n="Node", y0="v", L="L")

CHANNELS = ["AmplitudeDamping", "Depolarization", "BitFlip"]
LABEL = {"AmplitudeDamping": "Amplitude damping", "Depolarization": "Depolarising", "BitFlip": "Bit-flip"}
DEPTHS = np.arange(1, 11)

from qiskit_ibm_runtime import QiskitRuntimeService

try:
    if not USE_LIVE_CALIBRATION:
        raise LookupError("live calibration not requested")
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN, instance=IBM_INSTANCE)
               if IBM_TOKEN else QiskitRuntimeService())
    backend = service.backend(BACKEND_NAME)
    props = (backend.properties(datetime=pd.Timestamp(CALIBRATION_DATETIME).to_pydatetime())
             if CALIBRATION_DATETIME else backend.properties())
    SOURCE = "live"
except Exception as exc:
    print(f"Live access unavailable ({type(exc).__name__}). Using the FakeMarrakesh snapshot.")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    backend = FakeMarrakesh()
    props = backend.properties()
    SOURCE = "snapshot (FakeMarrakesh)"

CALIB_TIME = props.last_update_date
print(f"Backend: {props.backend_name} | source: {SOURCE} | calibration: {CALIB_TIME}")

def gate_errors(props, gate):
    rows = []
    for g in props.gates:
        if g.gate != gate:
            continue
        par = {p.name: p.value for p in g.parameters}
        rows.append({"qubits": tuple(sorted(g.qubits)), "error": par.get("gate_error", np.nan),
                     "length_ns": par.get("gate_length", np.nan)})
    df = pd.DataFrame(rows)
    return df.groupby("qubits", as_index=False).mean()   # one entry per undirected pair

TWO_Q_GATE = next(g for g in ("cz", "ecr", "cx") if any(x.gate == g for x in props.gates))
err_1q_all, err_2q_all = gate_errors(props, "sx"), gate_errors(props, TWO_Q_GATE)
err_1q = err_1q_all[err_1q_all.error < 1]
err_2q = err_2q_all[err_2q_all.error < 1]
readout = pd.Series([props.readout_error(q) for q in range(len(props.qubits))])

print(f"one-qubit gate: sx | entries {len(err_1q_all)}, excluded (error >= 1): {len(err_1q_all) - len(err_1q)}")
print(f"two-qubit gate: {TWO_Q_GATE} | couplers {len(err_2q_all)}, excluded (error >= 1): {len(err_2q_all) - len(err_2q)}")

err_1q_all.assign(qubits=err_1q_all["qubits"].map(lambda q: " ".join(map(str, q)))).to_csv(HW_DATA / "ibm_marrakesh_calibration_1q_sx.csv", index=False)
err_2q_all.assign(qubits=err_2q_all["qubits"].map(lambda q: " ".join(map(str, q)))).to_csv(HW_DATA / f"ibm_marrakesh_calibration_2q_{TWO_Q_GATE}.csv", index=False)
readout.rename_axis("qubit").rename("readout_error").to_csv(HW_DATA / "ibm_marrakesh_calibration_readout.csv")
print(f"calibration tables written to {HW_DATA} (source: {SOURCE}, {CALIB_TIME})")

def describe(s):
    s = pd.Series(s).dropna()
    return pd.Series({"Mean": s.mean(), "Std": s.std(), "Min": s.min(), "25%": s.quantile(0.25),
                      "50%": s.quantile(0.5), "75%": s.quantile(0.75), "Max": s.max()})

table1 = pd.DataFrame({"1-qubit gate error": describe(err_1q.error),
                       "2-qubit gate error": describe(err_2q.error),
                       "Readout error": describe(readout)}).T
display(table1)

# LaTeX rows for Table 1 (readout row optional)
fmt = lambda v: f"{v:.6f}"
for name, row in table1.iterrows():
    print(f"{name} & " + " & ".join(fmt(v) for v in row.values) + r" \\")

R_DEV = dict(q1=err_2q.error.quantile(0.25), med=err_2q.error.median(), q3=err_2q.error.quantile(0.75))
R_1Q_MED = err_1q.error.median()
print("r_dev (two-qubit gate error): median {med:.5f}, IQR {q1:.5f} to {q3:.5f}".format(**R_DEV))
print(f"median one-qubit gate error: {R_1Q_MED:.6f} (ratio to r_dev: {R_1Q_MED / R_DEV['med']:.3f})")

try:
    df_threshold = df_threshold_minlogr.copy()
    print("Using df_threshold_minlogr from the session.")
except NameError:
    df_threshold = pd.read_csv(THRESHOLDS_CSV)
    print(f"Using {THRESHOLDS_CSV}.")

def to_rate(v, scale=THRESHOLD_SCALE):
    v = np.asarray(v, float)
    return {"r": v, "minlog10": 10.0 ** (-v), "minln": np.exp(-v)}[scale]

df_threshold["r_star"] = to_rate(df_threshold["Threshold"])
assert df_threshold["r_star"].between(0, 1).all(), "Thresholds outside [0, 1]: check THRESHOLD_SCALE."

sub = df_threshold[(df_threshold["Graph Structure"] == INSTANCE["family"]) & (df_threshold["Node"] == INSTANCE["n"])]
R_STAR = sub.pivot(index="QAOA Layers", columns="Noise Type", values="r_star")[CHANNELS]
R_STAR.index = R_STAR.index.astype(int)

margin = {k: (R_STAR / R_DEV[v]).rename(columns=LABEL) for k, v in
          [("median", "med"), ("lower quartile r_dev", "q1"), ("upper quartile r_dev", "q3")]}
print(f"Thresholds r*(p), {INSTANCE}")
display(R_STAR.rename(columns=LABEL))
for k, m in margin.items():
    print(f"Margin r*(p) / r_dev at the {k}")
    display(m)
    below = {c: list(m.index[m[c] < 1]) for c in m.columns}
    print("  depths with margin < 1:", below)

worst = df_threshold.groupby(["Noise Type", "Graph Structure", "Node"])["r_star"].min()
print(f"Configurations with r*_worst > r_dev:")
for k in ("q1", "med", "q3"):
    print(f"  at r_dev {k} = {R_DEV[k]:.5f}: {(worst > R_DEV[k]).sum()} of {len(worst)}")
print("\nConfigurations with r*_worst <= upper-quartile r_dev:")
display(worst[worst <= R_DEV["q3"]].to_frame("r*_worst"))
print("Lowest r*_worst per channel:")
display(worst.groupby(level=0).agg(["min", "idxmin"]))

fit = pd.read_csv(FIT_PARAMS_CSV)
fi = fit[(fit[FIT_COLS["family"]] == INSTANCE["family"]) & (fit[FIT_COLS["n"]] == INSTANCE["n"])]
fi = fi.set_index(FIT_COLS["channel"])
missing = [c for c in CHANNELS if c not in fi.index]
assert not missing, f"Fit parameters missing for {missing}: check FIT_COLS and the channel names."

MIDPOINT = {c: float(fi.loc[c, FIT_COLS["y0"]] + fi.loc[c, FIT_COLS["L"]] / 2) for c in CHANNELS}
ASYMPTOTES = {c: (float(fi.loc[c, FIT_COLS["y0"]]), float(fi.loc[c, FIT_COLS["y0"]] + fi.loc[c, FIT_COLS["L"]]))
              for c in CHANNELS}
for c in CHANNELS:
    lo, hi = sorted(ASYMPTOTES[c])
    print(f"{LABEL[c]:>18}: asymptotes {lo:.3f} and {hi:.3f}, midpoint AR = {MIDPOINT[c]:.3f}")
    if not (0 <= MIDPOINT[c] <= 1):
        print("                    warning: midpoint outside [0, 1]; check the fit for this channel")

def thresholds_from_fit(row, x_grid):
    pm, ps, xm, xs = row["p_mean"], row["p_std"], row["x_mean"], row["x_std"]
    b = [row[f"b{i}"] for i in range(6)]
    out = []
    for p in DEPTHS:
        pt = (p - pm) / ps
        roots = np.roots([b[4], b[2] + b[5] * pt, b[0] + b[1] * pt + b[3] * pt**2])
        roots = roots[np.isreal(roots)].real * xs + xm
        ok = [10.0 ** (-x) for x in roots if x_grid.min() <= x <= x_grid.max()]
        out.append(ok[0] if len(ok) == 1 else np.nan)
    return np.array(out)

X_GRID = -np.log10(np.array(sorted(pd.read_csv(DATASET_CSV, usecols=["Noise intensity"])["Noise intensity"].unique())))
rec = pd.DataFrame({LABEL[c]: thresholds_from_fit(fi.loc[c], X_GRID) for c in CHANNELS}, index=DEPTHS)
rel = (rec.values / R_STAR.values - 1)
print(f"Maximum relative difference between recomputed and tabulated thresholds: {np.nanmax(np.abs(rel)):.2e}")
assert np.nanmax(np.abs(rel)) < 1e-3, "Thresholds do not match the fitted parameters: check the column mapping."

data = pd.read_csv(DATASET_CSV)
inst = data[(data["Structure"] == INSTANCE["family"]) & (data["Node"] == INSTANCE["n"])]
GRID_R = np.array(sorted(data["Noise intensity"].unique()))

grid = inst.groupby(["Noise Type", "Noise intensity"])["Approx. Ratio"].mean().unstack(0)[CHANNELS].rename(columns=LABEL)
print("Run- and depth-averaged AR per noise rate (hardware instance):")
display(grid)

fl = inst[(inst["Noise intensity"] == 0.5) & inst["Noise Type"].isin(["BitFlip", "Depolarization"])]["Approx. Ratio"]
FLOOR = dict(mean=fl.mean(), sd=fl.std())
CEIL = inst[inst["Noise intensity"] == GRID_R.min()]["Approx. Ratio"].mean()
print(f"Decohered floor (r = 0.5, unital channels): mean {FLOOR['mean']:.3f}, s.d. {FLOOR['sd']:.3f}, n = {len(fl)}")
print(f"Low-noise reference (r = {GRID_R.min():g}, all channels): mean AR {CEIL:.3f}")

# Where each threshold sits relative to the sampled grid
bracket = {}
for c in CHANNELS:
    lo = [GRID_R[GRID_R <= v].max() for v in R_STAR[c]]
    hi = [GRID_R[GRID_R >= v].min() for v in R_STAR[c]]
    bracket[LABEL[c]] = sorted(set(zip(lo, hi)))
print("Grid points bracketing r*(p):", bracket)

hw = pd.Series(QUANTUM_AR, index=DEPTHS, name="hardware AR")
cls = pd.DataFrame({"hardware AR": hw})
for c in CHANNELS:
    cls[f"margin > 1 ({LABEL[c]})"] = R_STAR[c].values / R_DEV["med"] > 1
    cls[f"AR > midpoint ({LABEL[c]})"] = (hw.values > MIDPOINT[c]) if np.isfinite(MIDPOINT[c]) else pd.NA
cls["inside floor band"] = (hw - FLOOR["mean"]).abs() <= FLOOR["sd"]
display(cls)

for c in CHANNELS:
    if not np.isfinite(MIDPOINT[c]):
        print(f"{LABEL[c]:>18}: no midpoint available, agreement not assessed")
        continue
    pred, obs = cls[f"margin > 1 ({LABEL[c]})"], cls[f"AR > midpoint ({LABEL[c]})"].astype(bool)
    print(f"{LABEL[c]:>18}: prediction agrees with outcome at {(pred == obs).sum()} of 10 depths")

HW = dict(mean=hw.mean(), sd=hw.std(ddof=1), min=hw.min(), max=hw.max(),
          mean_p1_7=hw.loc[1:7].mean(), mean_p8_10=hw.loc[8:10].mean(),
          n_in_floor=int(cls["inside floor band"].sum()), n_below_floor_mean=int((hw < FLOOR["mean"]).sum()))
HW["drop"] = HW["mean_p1_7"] - HW["mean_p8_10"]
print("\nHardware AR summary:", {k: round(v, 3) for k, v in HW.items()})
print(f"Drop between p <= 7 and p >= 8: {HW['drop']:.3f}, i.e. {HW['drop'] / FLOOR['sd']:.2f} floor s.d.")
sim_sd = inst[inst["Noise intensity"].isin([0.001, 0.005])].groupby(["Noise Type", "# Layers"])["Approx. Ratio"].std()
print(f"Simulated run-to-run s.d. at r = 0.001 and 0.005: median {sim_sd.median():.3f}, max {sim_sd.max():.3f}")

COLOUR = {"AmplitudeDamping": "#2a78d6", "Depolarization": "#eb6834", "BitFlip": "#1baf7a"}
MARKER = {"AmplitudeDamping": "o", "Depolarization": "s", "BitFlip": "^"}
INK, MUTED = "#2b2b2b", "#8a8984"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

def panel_label(ax, s):
    # centred below the axes, under the x-axis label
    ax.text(0.5, -0.22, f"({s})", transform=ax.transAxes, fontsize=8, ha="center", va="top")

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4), constrained_layout=True)
p = DEPTHS

# (a) thresholds against the device error rate
for c in CHANNELS:
    ax_a.plot(p, R_STAR[c], color=COLOUR[c], marker=MARKER[c], label=LABEL[c])
ax_a.axhspan(R_DEV["q1"], R_DEV["q3"], color=MUTED, alpha=0.25, lw=0)
ax_a.axhline(R_DEV["med"], color=INK, lw=0.8, ls="--")
ax_a.axhline(R_1Q_MED, color=INK, lw=0.7, ls=":")
ax_a.text(0.8, R_DEV["q3"] * 1.05, "two-qubit gate error (median and IQR)", fontsize=5.5, color=INK, va="bottom")
ax_a.text(0.8, R_1Q_MED * 1.05, "one-qubit gate error (median)", fontsize=5.5, color=INK, va="bottom")
ax_a.set(yscale="log", ylim=(1e-4, 0.4), xlabel="QAOA depth $p$", ylabel="Noise rate $r$")

# (b) hardware outcome
ax_b.axhspan(FLOOR["mean"] - FLOOR["sd"], FLOOR["mean"] + FLOOR["sd"], color=MUTED, alpha=0.25, lw=0)
ax_b.axhline(FLOOR["mean"], color=MUTED, lw=0.8, ls=":")
for c in CHANNELS:
    ax_b.axhline(MIDPOINT[c], color=COLOUR[c], lw=0.9, ls="--")
unital_top = max(MIDPOINT["BitFlip"], MIDPOINT["Depolarization"])
ax_b.text(1, unital_top + 0.01, "midpoints, bit-flip and depolarising", fontsize=5.5, color=INK, va="bottom")
ax_b.text(1, MIDPOINT["AmplitudeDamping"] + 0.01, "midpoint, amplitude damping", fontsize=5.5,
          color=COLOUR["AmplitudeDamping"], va="bottom")
ax_b.plot(p, QUANTUM_AR, color=INK, marker="D", lw=1.0, label=r"$\mathtt{ibm\_marrakesh}$")
ax_b.text(10.3, min(QUANTUM_AR) - 0.04,
          "grey band: decohered floor, simulated $r=0.5$,\nunital channels, mean $\\pm$ s.d.",
          fontsize=5.5, color=INK, ha="right", va="top")
y_lo = min(0.4, np.floor((min(MIDPOINT.values()) - 0.05) * 10) / 10)
ax_b.set(ylim=(y_lo, 1.0), xlabel="QAOA depth $p$", ylabel="Approximation ratio")

for ax, s in zip((ax_a, ax_b), "ab"):
    ax.set_xticks(range(1, 11)); ax.set_xlim(0.6, 10.4); panel_label(ax, s)
h, l = ax_a.get_legend_handles_labels(); hb, lb = ax_b.get_legend_handles_labels()
fig.legend(h + hb, l + lb, loc="outside upper center", ncol=4, handlelength=1.8)
fig.savefig(FIGS / "fig_hardware_threshold.pdf"); fig.savefig(FIGS / "fig_hardware_threshold.png", dpi=300)

numbers = {
    "calibration source": SOURCE,
    "calibration timestamp": str(CALIB_TIME),
    "two-qubit gate": TWO_Q_GATE,
    "couplers used / excluded": f"{len(err_2q)} / {len(err_2q_all) - len(err_2q)}",
    "r_dev median": R_DEV["med"], "r_dev Q1": R_DEV["q1"], "r_dev Q3": R_DEV["q3"],
    "one-qubit error median": R_1Q_MED,
    "configs with r*_worst > r_dev (median)": f"{(worst > R_DEV['med']).sum()} of {len(worst)}",
    "configs with r*_worst > r_dev (Q3)": f"{(worst > R_DEV['q3']).sum()} of {len(worst)}",
}
for c in CHANNELS:
    numbers[f"min margin, {LABEL[c]} (median r_dev)"] = (R_STAR[c] / R_DEV["med"]).min()
    numbers[f"depth of min margin, {LABEL[c]}"] = int((R_STAR[c] / R_DEV["med"]).idxmin())
    numbers[f"min margin, {LABEL[c]} (Q3 r_dev)"] = (R_STAR[c] / R_DEV["q3"]).min()
    numbers[f"midpoint AR, {LABEL[c]}"] = MIDPOINT[c]
numbers.update({"floor mean": FLOOR["mean"], "floor sd": FLOOR["sd"], "low-noise reference AR": CEIL,
                "hardware AR mean": HW["mean"], "hardware AR min": HW["min"], "hardware AR max": HW["max"],
                "hardware mean p1-7": HW["mean_p1_7"], "hardware mean p8-10": HW["mean_p8_10"],
                "hardware depths inside floor band": f"{HW['n_in_floor']} of 10"})
summary = pd.Series(numbers, name="value")
summary.to_csv(HW_DATA / "hardware_section_numbers.csv")
display(summary.to_frame())
