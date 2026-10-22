"""Shared helpers for Stage-5 publication figures.

Reads only from results/statistics.json, results/tables/, and the dataset npz.
Colorblind-safe (Okabe-Ito) palette; vector PDF + 300-DPI PNG output.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "operators"))

import train_and_evaluate_suite as T  # noqa: E402
from operators.losses_and_metrics import GridTensors  # noqa: E402
from operators.constants import annular_weights, make_grid  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
MODELS = os.path.join(RESULTS, "models")
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
FIGDATA = os.path.join(RESULTS, "figure_data")
DATA = os.path.join(ROOT, "data/processed/pnp_parametric_240.npz")

# Okabe-Ito colorblind-safe palette
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F",
}
MODEL_STYLE = {
    "deeponet_reparam": (C["blue"], "o", "PI-DeepONet (reparam)"),
    "fno_reparam": (C["orange"], "s", "FNO-2D (reparam)"),
    "deeponet_direct_c": (C["vermillion"], "^", "PI-DeepONet (direct c)"),
    "fno_direct_c": (C["purple"], "v", "FNO-2D (direct c)"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})


def load_statistics():
    with open(os.path.join(RESULTS, "statistics.json")) as fh:
        return json.load(fh)


def load_dataset():
    return T.load_data()


def load_table(name):
    import csv
    with open(os.path.join(TABLES, name), newline="") as fh:
        return list(csv.DictReader(fh))


def load_model(kind, mode, seed=43, condition="data_plus_physics", epochs=None):
    if epochs is not None:
        name = f"{kind}_{mode}_seed{seed}_{epochs}ep.pt"
    else:
        name = f"{kind}_{mode}_{condition}_seed{seed}.pt"
    ck = torch.load(os.path.join(MODELS, name), weights_only=False)
    model = T.build_model(ck["kind"], ck["mode"], ck["stats"], **ck["size"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def predict_mu(model, mu):
    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    idx = np.arange(mu.shape[0])
    pred = T.predict(model, idx, mu)
    ipred = T.current_from_pred(pred, idx, mu, GRID)
    return pred, ipred


GRID = GridTensors()
R, Z, *_ = make_grid(T.NR, T.NZ)
AREA = annular_weights(R)[:, None] * (Z[1] - Z[0])


def panel_label(ax, letter, x=-0.06, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", clip_on=False)


def save_fig(fig, name, data=None):
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(FIGDATA, exist_ok=True)
    pdf = os.path.join(FIGDIR, name + ".pdf")
    png = os.path.join(FIGDIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    if data is not None:
        np.savez_compressed(os.path.join(FIGDATA, name + "_data.npz"), **data)
    print(f"[fig] wrote {os.path.relpath(pdf, ROOT)} + .png", flush=True)


def summary_global(stat, key, split, metric):
    return stat["summary_multiseed_global_rel_l2"][key][split][metric]
