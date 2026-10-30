# Code for the noisy QAOA experiments

Code and data for the paper available at https://arxiv.org/abs/2610.36597v1. The paper characterises QAOA performance on MaxCut by an effective noise threshold rather than by the choice of circuit depth alone. Noisy simulations of 12 graphs under three noise channels, with noise acting during both training and sampling, are fitted with logistic surfaces of the approximation ratio over noise rate and depth, from which a threshold noise rate r*(p) is extracted for each depth. The thresholds are then compared with the gate errors of the ibm_marrakesh device in a hardware run.

```
code/
  common/runtag.py                    filename tags, columns, seed rule (imported by all scripts)
  01_run_noisy_experiments/
    run_noisy_qaoa.py                 run one block of the grid from a config file
    config_template.txt               all experimental factors
  02_compile_results/
    compile_results.py                merge run files into the unified dataset, with checks
  03_fit_thresholds/
    fit_logistic_thresholds.py        fit the 36 logistic surfaces, extract r*(p), draw Supplementary Fig. 3
    extract_threshold_curve.py        threshold curves from the fitted parameters (no refit)
  04_hardware/
    qaoa_hardware.py                  QAOA on IBM hardware or Aer
    run_hardware.py                   run graph 80 at depths 1..10 with the settings of the Methods
    hardware_analysis.py              Table 1, r_dev, margins, midpoints, Figure 8 and the quoted numbers
  05_figures/
    main/   fig2 ... fig7             one script per main-text figure
    supp/   sfig1, sfig2, sfig4, sfig5  one script per supplementary figure
  run_analysis.sh                     compile, threshold curve and all figures in one go (--refit to refit as well)
```

The analysis steps (02, 03, 05) need only the packages in `requirements-analysis.txt`; step 01 needs `requirements-simulation.txt`
(CUDA-Q 0.12.0 or later, as in the experiments) and, for the full grid, an NVIDIA GPU;
step 04 also needs `requirements-hardware.txt` (and an IBM Quantum account only for new hardware runs).

## Graph instances

`data/graphs/graph_instances.csv` holds the 12 graphs used in the experiments, one per graph family and order:
`Graph_ID` (the `Graph ID` of the AR dataset), `Node`, `Graph Family` (Erdős-Rényi random, path, balanced complete bipartite),
`Structure` (the label used in the AR dataset, for joining), `Adjacency Matrix` and `Brute Force` (the maximum cut, by exhaustive
search). 

## Run an experiment block

```
python run_noisy_qaoa.py config.txt --dry-run   # shows the output name and the number of jobs
python run_noisy_qaoa.py config.txt             # runs; rerunning the same command resumes
```

The output file is named after its configuration, for example

```
noise=depol__structure=path__node=6__graph=91__rates=1e-4+5e-4+...__layers=1-10__reps=10__seedbase=0__traj=1024__shots=16384__iters=250.csv
```

and holds one row per job with the columns `Node, Structure, Graph ID, Seed, Run ID, # Layers, Noise Type, Noise intensity, Approx. Ratio`.
To spread work over several machines, give each a config with a different subset of `rates` and/or `layers`.
Seeds follow `seed = seedbase + 1 + (rate_index * 10 + layers - 1) * 10 + replicate`, where `rate_index` is the position in the fixed ladder
`1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1`; `Run ID` equals `Seed`.

The experiments were run on NVIDIA GPUs.

## Compile

```
python compile_results.py data/runs data/processed/qaoa_unified_dataset_new.csv --expect-pairs 36
```

Checks: columns and constant fields against the filename; exactly 10 rows per (rate, layers); unique, in-range seeds; `Run ID == Seed`;
`0 <= AR <= 1`; identical (trajectories, shots, iterations) in all files; no cell covered twice; every graph has every noise model;
36 (graph, noise) pairs of 800 rows each.

## Notes on the released data

* `data/runs/` holds the 36 result files of the experiments, one per (graph, noise model), in the format above. Compiling them gives
  `data/processed/qaoa_unified_dataset_new.csv`.
* Seeds are unique within each (graph, noise model) pair; the same numbers are reused across graphs of the same structure (1 to 800 for path and
  ER graphs, 801 to 1600 for complete bipartite graphs).
* Per-iteration energies and the 30 most frequent sampled bitstrings were logged during the runs but are not released.

## Fit the logistic surfaces and extract the thresholds

```
python 03_fit_thresholds/extract_threshold_curve.py      # quick: threshold curves from the released parameters
python 03_fit_thresholds/fit_logistic_thresholds.py      # full refit, a few minutes; writes results/fits/
```

`data/fits/` holds the results of the fit: `logistic_fit_parameters.csv` (Supplementary Table 1) and `thresholds.csv` (Supplementary
Table 2). Two further tables are derived from them: `threshold_curve.csv` (the red threshold curves, computed by
`extract_threshold_curve.py`, which stops unless it reproduces `thresholds.csv` to a relative 1e-9) and `best_worst_case_thresholds.csv`
(Supplementary Tables 3 and 4, written by the Figure 7 script).

Each surface is fitted to the 800 individual runs of its configuration (8 noise rates x 10 depths x 10 runs). MAE and RMSE are
computed over those 800 points. The 3D panels show the 80 run averages.

## Figures

```
bash run_analysis.sh            # everything from data/runs; all figures from the released fits
bash run_analysis.sh --refit    # the same, plus the full refit (needed for Supplementary Fig. 3)
```

| Figure | Script | Output file |
|---|---|---|
| Fig. 2 | `05_figures/main/fig2_graph_instances.py` | `figures/main/graphs.{pdf,png}` |
| Fig. 3 | `05_figures/main/fig3_ar_landscape_K99.py` | `figures/main/2D_noise_bipartite_18.png` |
| Fig. 4 | `05_figures/main/fig4_instance_analysis_ad.py` | `figures/main/2D_instance_analysis_AmplitudeDamping.png` |
| Fig. 5 | `05_figures/main/fig5_logistic_fit_mae.py` | `figures/main/logistic_fit_mae.png` |
| Fig. 6 | `05_figures/main/fig6_fit_snapshot_path_depol.py` | `figures/main/logistic_fit_thresholds_snapshot.png` |
| Fig. 7 | `05_figures/main/fig7_threshold_summary.py` | `figures/main/threshold_summary.{pdf,png}` |
| Fig. 8 | `04_hardware/hardware_analysis.py` | `figures/main/fig_hardware_threshold.{pdf,png}` |
| Supp. Fig. 1 | `05_figures/supp/sfig1_ar_mean_heatmap.py` | `figures/supp/supp_ar_mean_heatmap.png` |
| Supp. Fig. 2 | `05_figures/supp/sfig2_ar_std_heatmap.py` | `figures/supp/supp_ar_std_heatmap.png` |
| Supp. Fig. 3 | `03_fit_thresholds/fit_logistic_thresholds.py` | `figures/supp/3D_logistic_minlogr_fit_all_data.png` |
| Supp. Fig. 4 | `05_figures/supp/sfig4_threshold_grid.py` | `figures/supp/supp_threshold_grid.png` |
| Supp. Fig. 5 | `05_figures/supp/sfig5_threshold_window.py` | `figures/supp/supp_threshold_window.png` |

Output files keep the names used in the manuscript sources. Fig. 1 is a drawn illustration. Fig. 2 draws graphs 46, 81, 63 and 80
(panels a to d), 91 (e) and 94 (f), each with an optimal cut checked against the stored maximum.

## Hardware runs and the hardware section

```
python 04_hardware/run_hardware.py                     # new runs on ibm_marrakesh (saved IBM Quantum account required)
python 04_hardware/hardware_analysis.py                # Table 1, r_dev, margins, Figure 8; no account needed
```

`data/hardware/ibm_marrakesh_ar.csv` holds the hardware result: one execution per depth p = 1 to 10 on graph 80 (18-vertex
Erdős-Rényi), AR to two decimals as reported in the manuscript. `hardware_analysis.py` computes every number of the hardware section
(r_dev median 0.003234 and IQR 0.002437 to 0.004253 over 163 couplers, midpoint ARs 0.468, 0.897 and 0.907, decohered floor
0.815 +/- 0.062, 36 of 36 configurations with r*_worst above the median r_dev), writes them to `data/hardware/hardware_section_numbers.csv`
and draws Figure 8. Table 1 uses the `FakeMarrakesh` calibration snapshot of qiskit-ibm-runtime 0.50.0 (dated 2025-02-26); live
calibration is used only with `USE_LIVE_CALIBRATION=1`. The calibration used is exported to `data/hardware/ibm_marrakesh_calibration_*.csv`.
An API key is read only from `IBM_QUANTUM_TOKEN` and is never stored in the code.

The defaults of `run_QAOA()` in `qaoa_hardware.py` are the settings of the Methods (ibm_marrakesh, training and evaluation on
hardware, 125 COBYLA iterations, 4096 estimator shots, 16384 sampler shots, optimisation level 3, resilience level 0). `run_hardware.py`
uses these defaults unless an option overrides them, and records the AR as the cut of the most frequent bitstring, ties broken towards
the larger cut.

## Use of a large language model

Claude Sonnet, a large language model by Anthropic, was used to reorganise the original code into a repository structure that follows GitHub best practice and to improve the comprehensibility of the code. It was not used to design or run the experiments, nor to produce the results. All changes were reviewed by the authors.
