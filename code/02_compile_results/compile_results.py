#!/usr/bin/env python3
"""Merge noisy-QAOA run files into the unified dataset, with strict checks.

    python compile_results.py RUNS_DIR OUTPUT.csv [--allow-partial] [--strict-seeds] [--expect-pairs N]

Every file in RUNS_DIR must carry its configuration in its name (see
../common/runtag.py).  The script

  1. parses each filename and checks that the file content matches it exactly
     (columns, constant fields, exactly `reps` rows for every (rate, layers)
     cell, unique seeds inside the legal range, Run ID == Seed, AR in [0, 1]);
     files whose seeds also follow the closed-form seed rule are counted, and
     --strict-seeds makes that rule mandatory;
  2. checks that all files share the same physical settings (trajectories,
     shots, iterations) and that no cell is covered twice;
  3. checks that every (graph, noise) pair present is fully covered, i.e. all
     8 rates x 10 layers x 10 replicates, that every graph has every noise
     model, and (with --expect-pairs N) that exactly N pairs exist; all of
     this is skipped with --allow-partial;
  4. writes the unified file with the columns of common/runtag.py.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from runtag import (  
    COLUMNS,
    LAYER_SLOTS,
    RATE_LADDER,
    REP_SLOTS,
    parse_filename,
    settings_signature,
)

FULL_CELLS = len(RATE_LADDER) * LAYER_SLOTS * REP_SLOTS  # 800 per (graph, noise)

# Row order of the compiled dataset, as (Noise Type, Node, Structure) blocks: orders 6, 9 and 12 by
# channel, order and family (path, complete bipartite, ER random), then the 18-vertex blocks as listed.
_FAMILIES = ["Path", "Complete bipartite", "ER random"]
BLOCK_ORDER = (
    [(nt, n, s) for nt in ["AmplitudeDamping", "BitFlip", "Depolarization"] for n in [6, 9, 12] for s in _FAMILIES]
    + [("AmplitudeDamping", 18, s) for s in ["Path", "Complete bipartite", "ER random"]]
    + [("BitFlip", 18, s) for s in ["ER random", "Path", "Complete bipartite"]]
    + [("Depolarization", 18, s) for s in ["ER random", "Path", "Complete bipartite"]]
)


class CompileError(Exception):
    pass


def check_file(path: Path, tag):
    """Read one run file and verify it against its own filename tag.

    Returns (dataframe, follows_seed_rule).
    """
    df = pd.read_csv(path)
    where = path.name
    if list(df.columns) != COLUMNS:
        raise CompileError(f"{where}: columns {list(df.columns)} differ from {COLUMNS}")
    if df.isna().any().any():
        raise CompileError(f"{where}: contains missing values")

    constants = {
        "Node": tag.node,
        "Structure": tag.structure_label,
        "Graph ID": tag.graph,
        "Noise Type": tag.noise_label,
    }
    for col, expected in constants.items():
        bad = df[col] != expected
        if bad.any():
            raise CompileError(f"{where}: {int(bad.sum())} row(s) with {col} != {expected!r}")

    if not (df["Seed"] == df["Run ID"]).all():
        raise CompileError(f"{where}: Run ID differs from Seed")

    ar = df["Approx. Ratio"]
    if not (np.isfinite(ar).all() and (ar >= 0).all() and (ar <= 1).all()):
        raise CompileError(f"{where}: Approx. Ratio outside [0, 1] or not finite")

    # Every (rate, layers) cell must hold exactly `reps` rows.
    expected_counts = {(r, l): tag.reps for r in tag.rates for l in tag.layer_values()}
    observed_counts = df.groupby(["Noise intensity", "# Layers"]).size().to_dict()
    if observed_counts != expected_counts:
        missing = sorted(set(expected_counts) - set(observed_counts))
        extra = sorted(set(observed_counts) - set(expected_counts))
        wrong = sorted(k for k in expected_counts if k in observed_counts and observed_counts[k] != tag.reps)
        raise CompileError(
            f"{where}: grid mismatch (cells missing: {missing[:2]}, unexpected: {extra[:2]}, "
            f"wrong row count: {wrong[:2]})"
        )

    # Seeds: unique, inside the range reserved by seedbase.
    seeds = df["Seed"].astype(int)
    if seeds.duplicated().any():
        raise CompileError(f"{where}: duplicated seeds")
    lo, hi = tag.seedbase + 1, tag.seedbase + len(RATE_LADDER) * LAYER_SLOTS * REP_SLOTS
    if seeds.min() < lo or seeds.max() > hi:
        raise CompileError(f"{where}: seeds outside the range {lo}..{hi}")

    # Does the file also follow the closed-form seed rule exactly?
    formula = {(rate, layers, seed) for rate, layers, _rep, seed in tag.grid()}
    observed = set(zip(df["Noise intensity"], df["# Layers"], seeds))
    return df, observed == formula


def compile_dir(runs_dir: Path, allow_partial: bool, strict_seeds: bool, expect_pairs=None):
    files = sorted(runs_dir.glob("*.csv"))
    if not files:
        raise CompileError(f"no csv files in {runs_dir}")

    frames, signatures, off_rule = [], {}, []
    for path in files:
        try:
            tag = parse_filename(path)
        except ValueError as exc:
            raise CompileError(str(exc)) from exc
        frame, follows_rule = check_file(path, tag)
        if not follows_rule:
            if strict_seeds:
                raise CompileError(f"{path.name}: seeds do not follow the closed-form seed rule")
            off_rule.append(path.name)
        frames.append(frame)
        signatures.setdefault(settings_signature(tag), []).append(path.name)

    if len(signatures) > 1:
        lines = [f"  (traj, shots, iters) = {sig}: {len(names)} file(s), e.g. {names[0]}"
                 for sig, names in signatures.items()]
        raise CompileError("files were produced with different physical settings:\n" + "\n".join(lines))

    df = pd.concat(frames, ignore_index=True)

    # No cell may be covered by two files.
    key = ["Graph ID", "Noise Type", "Noise intensity", "# Layers", "Seed"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        raise CompileError(f"{int(dup.sum())} row(s) are covered by more than one file")
    # Seeds must also be unique within each (graph, noise) pair.
    dup = df.duplicated(["Graph ID", "Noise Type", "Seed"], keep=False)
    if dup.any():
        raise CompileError(f"{int(dup.sum())} row(s) share a seed within one (graph, noise) pair")

    # Coverage per (graph, noise).
    counts = df.groupby(["Graph ID", "Noise Type"]).size()
    short = counts[counts != FULL_CELLS]
    if len(short) and not allow_partial:
        lines = [f"  graph {int(g)} {n}: {c} of {FULL_CELLS} rows" for (g, n), c in short.items()]
        raise CompileError("incomplete coverage (use --allow-partial to accept):\n" + "\n".join(lines))

    # Every graph present must have every noise model present, and optionally a fixed number of pairs.
    pairs = set(counts.index)
    universe = {(g, n) for g in df["Graph ID"].unique() for n in df["Noise Type"].unique()}
    if universe - pairs and not allow_partial:
        lines = [f"  graph {int(g)} {n}" for g, n in sorted(universe - pairs)]
        raise CompileError("missing (graph, noise) pairs (use --allow-partial to accept):\n" + "\n".join(lines))
    if expect_pairs is not None and len(pairs) != expect_pairs and not allow_partial:
        raise CompileError(f"found {len(pairs)} (graph, noise) pairs, expected {expect_pairs}")

    rank = {blk: i for i, blk in enumerate(BLOCK_ORDER)}
    df["_block"] = [rank.get(k, len(rank)) for k in zip(df["Noise Type"], df["Node"], df["Structure"])]
    df = df.sort_values(["_block", "Noise Type", "Node", "Graph ID", "Seed"], kind="mergesort")
    df = df.drop(columns="_block").reset_index(drop=True)
    return df, off_rule, len(files)


def write_unified(df: pd.DataFrame, out_path: Path) -> None:
    out = df[COLUMNS].copy()
    out["Graph ID"] = out["Graph ID"].astype(float)  # float formatting, see module docstring
    out["Seed"] = out["Seed"].astype(float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--allow-partial", action="store_true", help="accept (graph, noise) pairs with incomplete grids")
    ap.add_argument("--expect-pairs", type=int, help="required number of (graph, noise) pairs, e.g. 36 for the paper")
    ap.add_argument("--strict-seeds", action="store_true", help="require every file to follow the closed-form seed rule")
    args = ap.parse_args()
    try:
        df, off_rule, n_files = compile_dir(args.runs_dir, args.allow_partial, args.strict_seeds, args.expect_pairs)
    except CompileError as exc:
        print(f"COMPILE FAILED: {exc}", file=sys.stderr)
        return 1
    write_unified(df, args.output)
    n_pairs = df.groupby(["Graph ID", "Noise Type"]).ngroups
    print(f"OK: {len(df)} rows from {n_files} files, {n_pairs} (graph, noise) pairs, written to {args.output}")
    if off_rule:
        print(f"note: {len(off_rule)} file(s) have unique, in-range seeds that do not follow the closed-form rule:")
        for name in off_rule:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
