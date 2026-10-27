#!/usr/bin/env python3
"""Figure 2: representative graph instances with an optimal MaxCut bipartition.

Produces : figures/main/graphs.png and figures/main/graphs.pdf
Reads    : data/graphs/graph_instances.csv (the 12 instances used in the experiments)
Panels   : (a) to (d) Erdős-Rényi random graphs with 6, 9, 12 and 18 vertices (graphs 46, 81, 63, 80),
           (e) the 6-vertex path graph (graph 91), (f) the 9-vertex balanced complete bipartite graph K_{4,5} (graph 94).

Node colour marks
the side of an optimal cut found by exhaustive search; bold dark edges are cut, thin grey edges are
not. For every panel the script checks that the cut it draws equals the stored optimum
("Brute Force" column). Where several optimal cuts exist, the first one in the enumeration order
(vertex 0 fixed on the first side) is drawn. Layouts are deterministic (fixed seed).
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIGS = ROOT / "figures" / "main"
FIGS.mkdir(parents=True, exist_ok=True)

PANELS = [46, 81, 63, 80, 91, 94]
SIDE_COLOUR = {0: "#7fb8d6", 1: "#e07a7a"}
CUT_EDGE, OTHER_EDGE = "#2b2b2b", "#c8c8c8"
SEED = 7

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


def load_graph(row):
    A = np.array(ast.literal_eval(row["Adjacency Matrix"]), dtype=float)
    return nx.from_numpy_array(A)


def optimal_cut(G):
    """Exhaustive search with vertex 0 fixed to side 0; returns (cut value, side of each vertex)."""
    n = G.number_of_nodes()
    E = np.array(G.edges())
    best, best_x = -1, None
    for b in range(2 ** (n - 1)):
        x = (b << 1 >> np.arange(n)) & 1          # bit i of 2b, so vertex 0 is always 0
        c = int((x[E[:, 0]] != x[E[:, 1]]).sum())
        if c > best:
            best, best_x = c, x
    return best, best_x


def layout(G, family, side):
    n = G.number_of_nodes()
    if family == "path":
        return {v: (v, 0.0) for v in G.nodes()}
    if family == "balanced complete bipartite":
        left = [v for v in G.nodes() if side[v] == 1]
        right = [v for v in G.nodes() if side[v] == 0]
        pos = {v: (0.0, -i * 1.0) for i, v in enumerate(left)}
        pos.update({v: (2.4, -i * 1.0 + (len(left) - len(right)) / 2) for i, v in enumerate(right)})
        return pos
    return nx.spring_layout(G, seed=SEED, k=1.6 / np.sqrt(n), iterations=300)


df = pd.read_csv(ROOT / "data" / "graphs" / "graph_instances.csv").set_index("Graph_ID")

fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 105 / 25.4))
for k, (ax, gid) in enumerate(zip(axes.ravel(), PANELS)):
    row = df.loc[gid]
    G = load_graph(row)
    value, side = optimal_cut(G)
    assert value == int(row["Brute Force"]), f"graph {gid}: cut {value} differs from stored optimum {row['Brute Force']}"
    pos = layout(G, row["Graph Family"], side)
    cut = [(u, v) for u, v in G.edges() if side[u] != side[v]]
    other = [(u, v) for u, v in G.edges() if side[u] == side[v]]
    nx.draw_networkx_edges(G, pos, edgelist=other, ax=ax, edge_color=OTHER_EDGE, width=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=cut, ax=ax, edge_color=CUT_EDGE, width=1.1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=28 if G.number_of_nodes() > 12 else 40,
                           node_color=[SIDE_COLOUR[int(side[v])] for v in G.nodes()], linewidths=0)
    ax.set_axis_off()
    ax.margins(0.08)
    ax.text(0.5, -0.06, f"({'abcdef'[k]})", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    print(f"({'abcdef'[k]}) graph {gid}: {row['Graph Family']}, n = {G.number_of_nodes()}, "
          f"{G.number_of_edges()} edges, optimal cut {value}")

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.07, wspace=0.08, hspace=0.25)
fig.savefig(FIGS / "graphs.pdf")
fig.savefig(FIGS / "graphs.png", dpi=300)
