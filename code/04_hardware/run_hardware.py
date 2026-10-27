#!/usr/bin/env python3
"""Run QAOA on the 18-vertex Erdős-Rényi instance (graph 80) for a range of depths, with the settings of the Methods.

    python run_hardware.py                          # ibm_marrakesh, p = 1..10 (needs an IBM Quantum account)
    python run_hardware.py --depths 1 2 3           # a subset of depths
    python run_hardware.py --mode noiseless --maxiter 5 --depths 1     # local check on Aer, no hardware use

Settings: the defaults of run_QAOA() in qaoa_hardware.py, which are those of the Methods ("Real Quantum
Hardware Implementation"): backend ibm_marrakesh, COBYLA with at most 125 iterations, EstimatorV2 with
4096 shots per evaluation, SamplerV2 with 16384 shots for the final circuit, transpilation at
optimisation level 3, no error mitigation (resilience level 0). The options below override them only
when given.

Each depth writes results/hardware/p{p}.json (optimal parameters, energy history, the 30 most frequent
bitstrings) and appends one row to results/hardware/ar.csv, in the format of
data/hardware/ibm_marrakesh_ar.csv. The account is read by QiskitRuntimeService() from the saved
configuration (QiskitRuntimeService.save_account); no key is stored in this repository.
"""
import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from qaoa_hardware import run_QAOA  

GRAPH_ID = 80


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--depths", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--backend", help="default: ibm_marrakesh")
    ap.add_argument("--mode", choices=["real", "noisy_sim", "noiseless"],
                    help="training and evaluation mode passed to run_QAOA (default: real)")
    ap.add_argument("--maxiter", type=int, help="default: 125")
    ap.add_argument("--estimator-shots", type=int, help="default: 4096")
    ap.add_argument("--sampler-shots", type=int, help="default: 16384")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hardware")
    args = ap.parse_args()

    graphs = pd.read_csv(ROOT / "data" / "graphs" / "graph_instances.csv").set_index("Graph_ID")
    row = graphs.loc[GRAPH_ID]
    G = nx.from_numpy_array(np.array(ast.literal_eval(row["Adjacency Matrix"]), dtype=float))
    c_opt = int(row["Brute Force"])
    args.out.mkdir(parents=True, exist_ok=True)

    overrides = {"training_mode": args.mode, "evaluation_mode": args.mode, "backend_real_name": args.backend,
                 "maxiter": args.maxiter, "sampler_shot": args.sampler_shots, "estimator_shot": args.estimator_shots}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    defaults = {k: v.default for k, v in inspect.signature(run_QAOA).parameters.items()}
    settings = {**{k: defaults[k] for k in ("training_mode", "evaluation_mode", "backend_real_name", "maxiter",
                                            "sampler_shot", "estimator_shot")}, **overrides}
    print("settings:", settings)

    for p in args.depths:
        res = run_QAOA(G, p, **overrides)
        bitstring, cut, count = res["top30"][0]          # most frequent, ties towards the larger cut
        ar = cut / c_opt
        record = {"backend": settings["backend_real_name"], "mode": settings["training_mode"], "graph_id": GRAPH_ID,
                  "p": p, "maxiter": settings["maxiter"], "estimator_shots": settings["estimator_shot"],
                  "sampler_shots": settings["sampler_shot"],
                  "approx_ratio": ar, "cut": cut, "c_opt": c_opt,
                  "optimal_params": np.asarray(res["optimal_params"]).tolist(),
                  "energy_history": [float(e) for e in res["energy_history"]],
                  "top30": [[b, int(c), int(n)] for b, c, n in res["top30"]]}
        (args.out / f"p{p}.json").write_text(json.dumps(record, indent=1))
        line = pd.DataFrame([{"Backend": settings["backend_real_name"], "Graph ID": GRAPH_ID, "Node": int(row["Node"]),
                              "Structure": row["Structure"], "QAOA Layers": p, "Approx. Ratio": ar}])
        csv = args.out / "ar.csv"
        line.to_csv(csv, mode="a", header=not csv.exists(), index=False)
        print(f"p = {p}: most frequent bitstring cuts {cut} of {c_opt} edges, AR = {ar:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
