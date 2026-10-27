"""Shared conventions for the noisy QAOA experiments.

This module is the single source of truth for:
  * the column layout of every run results file,
  * the filename tag that records the configuration of a run,
  * the seed rule,
  * the fixed ladder of noise rates used to index seeds.

It is imported by the run script and the compile script, so that both always
agree.  It has no dependency on CUDA-Q.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Output columns, shared by every run results file and by the compiled dataset
# data/processed/qaoa_unified_dataset_new.csv.
# ---------------------------------------------------------------------------
COLUMNS = [
    "Node",
    "Structure",
    "Graph ID",
    "Seed",
    "Run ID",
    "# Layers",
    "Noise Type",
    "Noise intensity",
    "Approx. Ratio",
]

# Tag value -> label stored in the "Noise Type" / "Structure" columns.
NOISE_LABELS = {
    "bitflip": "BitFlip",
    "depol": "Depolarization",
    "ampdamp": "AmplitudeDamping",
}
STRUCTURE_LABELS = {
    "path": "Path",
    "bipartite": "Complete bipartite",
    "er": "ER random",
}

# ---------------------------------------------------------------------------
# Seed rule.
#
# seed = seedbase + 1 + (rate_index * LAYER_SLOTS + (layers - 1)) * REP_SLOTS + rep
#
# rate_index is the position of the noise rate in RATE_LADDER (not in the subset
# being run), so the seed of a job does not depend on how the work is split
# between files or machines.  With the ladder below, seedbase 0 gives seeds 1..800
# and seedbase 800 gives 801..1600, the ranges used in the released data.
# ---------------------------------------------------------------------------
RATE_LADDER = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
LAYER_SLOTS = 10  # layers must lie in 1..10
REP_SLOTS = 10  # replicates are numbered 0..9


def seed_for(seedbase: int, rate: float, layers: int, rep: int) -> int:
    """Seed of one job (also stored as Run ID)."""
    i = RATE_LADDER.index(rate)
    return seedbase + 1 + (i * LAYER_SLOTS + (layers - 1)) * REP_SLOTS + rep


# ---------------------------------------------------------------------------
# Filename tag.
# ---------------------------------------------------------------------------
TAG_KEYS = [
    "noise",
    "structure",
    "node",
    "graph",
    "rates",
    "layers",
    "reps",
    "seedbase",
    "traj",
    "shots",
    "iters",
]
PAIR_SEP = "__"
LIST_SEP = "+"


def fmt_rate(x: float) -> str:
    """Compact exponent form, e.g. 0.0001 -> '1e-4', 0.5 -> '5e-1'."""
    mantissa, exponent = f"{x:.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    s = f"{mantissa}e{int(exponent)}"
    if float(s) != x:
        raise ValueError(f"noise rate {x!r} cannot be written exactly as a tag ({s})")
    return s


@dataclass(frozen=True)
class RunTag:
    noise: str
    structure: str
    node: int
    graph: int
    rates: tuple
    layers: tuple  # (first, last), inclusive
    reps: int
    seedbase: int
    traj: int
    shots: int
    iters: int

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        if self.noise not in NOISE_LABELS:
            raise ValueError(f"unknown noise '{self.noise}'; use {list(NOISE_LABELS)}")
        if self.structure not in STRUCTURE_LABELS:
            raise ValueError(
                f"unknown structure '{self.structure}'; use {list(STRUCTURE_LABELS)}"
            )
        if not self.rates:
            raise ValueError("no noise rates given")
        for r in self.rates:
            if r not in RATE_LADDER:
                raise ValueError(
                    f"noise rate {r} is not on the fixed ladder {RATE_LADDER}"
                )
        if list(self.rates) != sorted(set(self.rates)):
            raise ValueError("rates must be unique and in increasing order")
        lo, hi = self.layers
        if not (1 <= lo <= hi <= LAYER_SLOTS):
            raise ValueError(f"layers must satisfy 1 <= first <= last <= {LAYER_SLOTS}")
        if not (1 <= self.reps <= REP_SLOTS):
            raise ValueError(f"reps must lie in 1..{REP_SLOTS}")
        if self.seedbase < 0:
            raise ValueError("seedbase must be non-negative")
        for name in ("node", "graph", "traj", "shots", "iters"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    # -- labels and grid ----------------------------------------------------
    @property
    def noise_label(self) -> str:
        return NOISE_LABELS[self.noise]

    @property
    def structure_label(self) -> str:
        return STRUCTURE_LABELS[self.structure]

    def layer_values(self):
        return range(self.layers[0], self.layers[1] + 1)

    def grid(self):
        """Yield (rate, layers, rep, seed) in the order the jobs are run."""
        for rate in self.rates:
            for layers in self.layer_values():
                for rep in range(self.reps):
                    yield rate, layers, rep, seed_for(self.seedbase, rate, layers, rep)

    # -- (de)serialisation --------------------------------------------------
    def filename(self) -> str:
        self.validate()
        values = {
            "noise": self.noise,
            "structure": self.structure,
            "node": str(self.node),
            "graph": str(self.graph),
            "rates": LIST_SEP.join(fmt_rate(r) for r in self.rates),
            "layers": f"{self.layers[0]}-{self.layers[1]}",
            "reps": str(self.reps),
            "seedbase": str(self.seedbase),
            "traj": str(self.traj),
            "shots": str(self.shots),
            "iters": str(self.iters),
        }
        return PAIR_SEP.join(f"{k}={values[k]}" for k in TAG_KEYS) + ".csv"


_LAYERS_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_filename(path) -> RunTag:
    """Recover the run configuration from a results filename."""
    name = Path(path).name
    if not name.endswith(".csv"):
        raise ValueError(f"not a csv file: {name}")
    pairs = name[: -len(".csv")].split(PAIR_SEP)
    fields = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"malformed tag element '{pair}' in {name}")
        key, value = pair.split("=", 1)
        if key in fields:
            raise ValueError(f"repeated tag key '{key}' in {name}")
        fields[key] = value
    if list(fields) != TAG_KEYS:
        raise ValueError(f"tag keys {list(fields)} differ from expected {TAG_KEYS}: {name}")
    m = _LAYERS_RE.match(fields["layers"])
    if not m:
        raise ValueError(f"layers tag must look like '1-10': {name}")
    try:
        tag = RunTag(
            noise=fields["noise"],
            structure=fields["structure"],
            node=int(fields["node"]),
            graph=int(fields["graph"]),
            rates=tuple(float(s) for s in fields["rates"].split(LIST_SEP)),
            layers=(int(m.group(1)), int(m.group(2))),
            reps=int(fields["reps"]),
            seedbase=int(fields["seedbase"]),
            traj=int(fields["traj"]),
            shots=int(fields["shots"]),
            iters=int(fields["iters"]),
        )
    except ValueError as exc:
        raise ValueError(f"cannot parse tag of {name}: {exc}") from exc
    tag.validate()
    if tag.filename() != name:
        raise ValueError(f"filename is not in canonical form: {name} (expected {tag.filename()})")
    return tag


def settings_signature(tag: RunTag) -> tuple:
    """Physical settings that must be identical across all files of a release."""
    return (tag.traj, tag.shots, tag.iters)


