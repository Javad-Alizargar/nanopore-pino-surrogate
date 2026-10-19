"""TASK 5.3a — assemble the static client-side simulator assets.

Copies the exported ONNX operators into ``web_simulator/models/`` and writes
``web_simulator/data/model_meta.json`` containing the dimensionless grid, the
physical constants needed for the in-browser SG current evaluator, the
in-distribution and slider ranges, and a small set of verified SG/FV benchmark
states (fields + current) for one-click side-by-side comparison.

This script performs no inference; it only packages already-verified artifacts.
"""

import json
import os
import shutil

import numpy as np

from operators.constants import (
    A_LAMBDA_M2,
    DIFF_ANION,
    DIFF_CATION,
    EPSILON_0,
    FARADAY,
    L_NM,
    NR,
    NZ,
    REL_PERMITTIVITY,
    V_T,
    annular_weights,
    make_grid,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
NPZ = os.path.join(PROJECT_ROOT, "data", "processed", "pnp_parametric_240.npz")
MODELS_SRC = os.path.join(PROJECT_ROOT, "results", "models")
WEB = os.path.join(PROJECT_ROOT, "web_simulator")
WEB_MODELS = os.path.join(WEB, "models")
WEB_DATA = os.path.join(WEB, "data")

ONNX_FILES = {
    "deeponet": "deeponet_reparam_seed42.onnx",
    "fno": "fno_reparam_seed42.onnx",
}

# idx -> short human label. 144 is the v0.007 fixed-state DA-PINN benchmark.
BENCHMARK_IDX = [
    (144, "Benchmark #0 (interp)"),
    (150, "Interp in-box"),
    (0, "Train: V=0, q_w=0 (symmetric)"),
    (3, "Train: V=0, mid charge"),
    (6, "Train: corner max"),
    (7, "Train: corner max, opposite V"),
    (192, "OOD charge/salt"),
    (216, "OOD geometry/voltage"),
]


def _round(x, sig=6):
    return [float(f"{v:.{sig}g}") for v in np.asarray(x).ravel()]


def main() -> None:
    os.makedirs(WEB_MODELS, exist_ok=True)
    os.makedirs(WEB_DATA, exist_ok=True)

    copied = {}
    for key, name in ONNX_FILES.items():
        src = os.path.join(MODELS_SRC, name)
        dst = os.path.join(WEB_MODELS, name)
        shutil.copy2(src, dst)
        copied[key] = {
            "file": f"models/{name}",
            "bytes": os.path.getsize(dst),
            "source": f"results/models/{name}",
        }

    r, z, *_ = make_grid(NR, NZ)
    weights = annular_weights(r)
    dz = float(z[1] - z[0])
    dr = float(r[1] - r[0])

    data = np.load(NPZ, allow_pickle=True)
    mu = data["mu"]

    states = []
    for idx, label in BENCHMARK_IDX:
        states.append(
            {
                "id": int(idx),
                "label": label,
                "split": str(data["splits"][idx]),
                "mu": [float(v) for v in mu[idx]],
                "lambda_D_over_R": float(data["lambda_D_over_R"][idx]),
                "mean_current_pA": float(data["mean_current_pA"][idx]),
                "psi": _round(data["psi"][idx]),
                "c_plus": _round(data["c_plus"][idx]),
                "rho": _round(data["rho"][idx]),
                "i_z": _round(data["i_z"][idx]),
            }
        )

    meta = {
        "stage": 5,
        "task": "5.3a",
        "note": "No inference performed; packaging of verified ONNX + SG/FV states.",
        "grid": {
            "nr": NR,
            "nz": NZ,
            "r": _round(r),
            "z": _round(z),
            "weights": _round(weights),
            "dr": dr,
            "dz": dz,
        },
        "constants": {
            "V_T": V_T,
            "FARADAY": FARADAY,
            "EPSILON_0": EPSILON_0,
            "REL_PERMITTIVITY": REL_PERMITTIVITY,
            "A_LAMBDA_M2": A_LAMBDA_M2,
            "L_nm": L_NM,
            "D_cation": DIFF_CATION,
            "D_anion": DIFF_ANION,
        },
        "ranges_in_distribution": {
            "V_app_mV": [-50.0, 50.0],
            "C0": [20.0, 200.0],
            "q_w": [-0.05, 0.0],
            "R_nm": [3.0, 8.0],
        },
        "slider_ranges": {
            "V_app_mV": [-75.0, 75.0],
            "C0": [10.0, 200.0],
            "q_w": [-0.10, 0.0],
            "R_nm": [2.0, 8.0],
        },
        "models": copied,
        "benchmark_states": states,
    }

    out = os.path.join(WEB_DATA, "model_meta.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    for k, v in copied.items():
        print(f"copied {v['file']} ({v['bytes']} bytes)")


if __name__ == "__main__":
    main()
