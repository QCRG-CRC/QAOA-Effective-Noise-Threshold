#!/usr/bin/env python3
"""Noisy QAOA for MaxCut with noise applied during training and sampling.

One invocation runs one block of the experimental grid, described entirely by a
configuration file, and writes one CSV whose *filename* records the
configuration (see ../common/runtag.py).  Run it on as many machines as you like
with different configuration files (for example different noise rates or layer
ranges); ../02_compile_results/compile_results.py merges and checks the outputs.

    python run_noisy_qaoa.py config.txt            # run (resumes if the output exists)
    python run_noisy_qaoa.py config.txt --dry-run  # print output name and job count only

Protocol
  * QAOA with p layers on the MaxCut Hamiltonian, COBYLA optimiser.
  * The noise channel acts on h, x, y, z, rx, ry, rz of every qubit, both in every
    energy evaluation during training (noisy trajectories) and in the final sampling.
  * The parameters with the lowest observed (noisy) energy are used for sampling.
  * Approximation ratio (AR) = cut value of the most frequently sampled bitstring
    (ties broken by the larger cut value) divided by the brute-force optimum.

"""

from __future__ import annotations

import argparse
import ast
import configparser
import csv
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import cudaq
from cudaq import spin  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from runtag import COLUMNS, RunTag  # noqa: E402

# ---------------------------------------------------------------------------
# Problem and circuit
# ---------------------------------------------------------------------------


def objective_value(x, w):
    """Cut value of the 0/1 assignment x on the graph with weight matrix w."""
    X = np.outer(x, (1 - x))
    w_01 = np.where(w != 0, 1, 0)
    return np.sum(w_01 * X)


@cudaq.kernel
def kernel_qaoa(
    qubit_count: int,
    layer_count: int,
    edges_src: List[int],
    edges_tgt: List[int],
    thetas: List[float],
):
    qvector = cudaq.qvector(qubit_count)

    # Create superposition
    h(qvector)

    # Apply QAOA layers
    for layer in range(layer_count):
        for edge in range(len(edges_src)):
            qubitu = edges_src[edge]
            qubitv = edges_tgt[edge]
            x.ctrl(qvector[qubitu], qvector[qubitv])
            rz(2.0 * thetas[layer], qvector[qubitv])
            x.ctrl(qvector[qubitu], qvector[qubitv])

        for qubit in range(qubit_count):
            rx(2.0 * thetas[layer + layer_count], qvector[qubit])


def get_operator(edges_src, edges_tgt):
    hamiltonian = cudaq.SpinOperator()
    for u, v in zip(edges_src, edges_tgt):
        hamiltonian += 0.5 * (cudaq.spin.z(u) * cudaq.spin.z(v) - cudaq.spin.i(u) * cudaq.spin.i(v))
    return hamiltonian


def get_edges_src_tgt_from_matrix(weight_matrix):
    edges_src, edges_tgt = [], []
    num_nodes = len(weight_matrix)
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                edges_src.append(i)
                edges_tgt.append(j)
    return edges_src, edges_tgt


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
NOISE_CHANNELS = {
    "bitflip": lambda p: cudaq.BitFlipChannel(p),
    "depol": lambda p: cudaq.DepolarizationChannel(p),
    "ampdamp": lambda p: cudaq.AmplitudeDampingChannel(p),
}
NOISY_GATES = ["h", "x", "y", "z", "rx", "ry", "rz"]


@cudaq.kernel
def _noise_probe():
    q = cudaq.qvector(1)
    x(q)


def check_noise_active() -> None:
    """Abort if the current CUDA-Q target silently ignores noise models.

    Some targets (for example the default CPU state-vector simulator) accept a
    noise model but apply no noise, which would produce noiseless results.  A
    single X gate followed by a 50% bit flip must return both outcomes.
    """
    model = cudaq.NoiseModel()
    model.add_channel("x", [0], cudaq.BitFlipChannel(0.5))
    counts = cudaq.sample(_noise_probe, noise_model=model, shots_count=2000)
    if "0" not in counts or "1" not in counts:
        raise RuntimeError(
            "the selected CUDA-Q target does not apply noise models. Set `target` in the "
            "[simulation] section to a noise-capable target (for example a GPU target "
            "with trajectories, or density-matrix-cpu)."
        )


def make_noise_model(noise: str, rate: float, qubit_count: int):
    model = cudaq.NoiseModel()
    channel = NOISE_CHANNELS[noise](rate)
    for q in range(qubit_count):
        for gate in NOISY_GATES:
            model.add_channel(gate, [q], channel)
    return model


# ---------------------------------------------------------------------------
# One job: train QAOA under noise, sample, return the cut value of the answer
# ---------------------------------------------------------------------------
def run_qaoa(w, qubit_count, layer_count, seed, rate, tag: RunTag, init_bound: float) -> int:
    edges_src, edges_tgt = get_edges_src_tgt_from_matrix(w)
    parameter_count = 2 * layer_count

    cudaq.set_random_seed(seed)
    rng = np.random.RandomState(seed)

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.initial_parameters = rng.uniform(-init_bound, init_bound, parameter_count)
    optimizer.max_iterations = tag.iters

    hamiltonian = get_operator(edges_src, edges_tgt)
    noise = make_noise_model(tag.noise, rate, qubit_count)

    best = {"energy": float("inf"), "params": None}

    def objective(params):
        try:
            if np.any(np.isnan(params)) or np.any(np.isinf(params)):
                return 1e6
            result = cudaq.observe(
                kernel_qaoa,
                hamiltonian,
                qubit_count,
                layer_count,
                edges_src,
                edges_tgt,
                params,
                noise_model=noise,
                num_trajectories=tag.traj,
            )
            energy = result.expectation()
            if np.isnan(energy) or np.isinf(energy):
                return 1e6
            if energy < best["energy"]:
                best["energy"] = energy
                best["params"] = np.copy(params)
            return energy
        except Exception as exc:  # penalise and continue
            print(f"[objective] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1e6

    optimizer.optimize(dimensions=parameter_count, function=objective)
    if best["params"] is None:
        raise RuntimeError("optimiser never produced a valid energy")

    counts = cudaq.sample(
        kernel_qaoa,
        qubit_count,
        layer_count,
        edges_src,
        edges_tgt,
        best["params"],
        noise_model=noise,
        shots_count=tag.shots,
    )

    # Most frequent bitstring; ties broken by the larger cut value.
    best_key, best_cut = None, None
    for bitstring, count in counts.items():
        x = np.array([int(bit) for bit in bitstring])
        cut = objective_value(x, w)
        key = (count, cut)
        if best_key is None or key > best_key:
            best_key, best_cut = key, cut
    return int(best_cut)


# ---------------------------------------------------------------------------
# Configuration and graph table
# ---------------------------------------------------------------------------
def _parse_layers(text: str):
    text = text.strip()
    if "-" in text:
        lo, hi = text.split("-", 1)
        return int(lo), int(hi)
    return int(text), int(text)


def load_config(config_path: Path):
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not parser.read(config_path):
        raise FileNotFoundError(config_path)
    base = config_path.resolve().parent

    def need(section, key):
        try:
            return parser[section][key].strip()
        except KeyError as exc:
            raise KeyError(f"missing '{key}' in section [{section}] of {config_path}") from exc

    graph_table = (base / need("instance", "graph_table")).resolve()
    graph_id = int(need("instance", "graph_id"))
    out_dir = (base / need("output", "out_dir")).resolve()

    table = pd.read_csv(graph_table)
    rows = table[table["Graph_ID"] == graph_id]
    if len(rows) != 1:
        raise ValueError(f"graph {graph_id} found {len(rows)} times in {graph_table}")
    row = rows.iloc[0]
    w = np.array(ast.literal_eval(row["Adjacency Matrix"]), dtype=int)
    node = int(row["Node"])
    optimum = float(row["Brute Force"])
    if w.shape != (node, node) or not np.array_equal(w != 0, (w != 0).T):
        raise ValueError(f"graph {graph_id}: adjacency matrix is not a symmetric {node}x{node} matrix")

    tag = RunTag(
        noise=need("instance", "noise"),
        structure=need("instance", "structure"),
        node=node,
        graph=graph_id,
        rates=tuple(float(s) for s in need("grid", "rates").split(",")),
        layers=_parse_layers(need("grid", "layers")),
        reps=int(need("grid", "reps")),
        seedbase=int(need("grid", "seedbase")),
        traj=int(need("simulation", "trajectories")),
        shots=int(need("simulation", "shots")),
        iters=int(need("simulation", "max_iterations")),
    )
    tag.validate()
    init_bound = float(need("simulation", "init_angle_bound")) * np.pi
    target = parser["simulation"].get("target", "").strip()  # optional
    return tag, w, optimum, init_bound, out_dir, target


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print output name and job count, run nothing")
    args = ap.parse_args()

    tag, w, optimum, init_bound, out_dir, target = load_config(args.config)
    out_path = out_dir / tag.filename()
    jobs = list(tag.grid())

    done = set()
    if out_path.exists():
        with open(out_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != COLUMNS:
                raise ValueError(f"{out_path} exists with a different header; refusing to append")
            seed_col = COLUMNS.index("Seed")
            done = {int(float(r[seed_col])) for r in reader if r}

    print(f"output : {out_path}")
    print(f"jobs   : {len(jobs)} total, {len(done)} already done, {len(jobs) - len(done)} to run")
    if args.dry_run:
        return 0

    if target:
        cudaq.set_target(target)
    print(f"target : {cudaq.get_target().name}")
    check_noise_active()

    out_dir.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    failures = 0
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)
            f.flush()
        for k, (rate, layers, rep, seed) in enumerate(jobs, 1):
            if seed in done:
                continue
            t0 = time.time()
            try:
                cut = run_qaoa(w, tag.node, layers, seed, rate, tag, init_bound)
            except Exception as exc:  # leave a gap; rerunning the same config retries it
                failures += 1
                print(f"[{k}/{len(jobs)}] seed {seed} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            writer.writerow(
                [tag.node, tag.structure_label, tag.graph, seed, seed, layers, tag.noise_label, rate, cut / optimum]
            )
            f.flush()
            print(f"[{k}/{len(jobs)}] rate={rate:g} layers={layers} rep={rep} seed={seed} "
                  f"AR={cut / optimum:.4f} ({time.time() - t0:.1f}s)")

    if failures:
        print(f"{failures} job(s) failed; rerun the same command to retry them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
