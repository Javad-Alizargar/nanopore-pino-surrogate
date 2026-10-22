#!/usr/bin/env python3
"""Run the real 3-way Stage-21 bias diagnostic on the final code and save every row.

D1: noise-free 21x61 observations inverted on 21x61 (pure optimisation error)
D2: noise-free 41x121 observations inverted on 21x61 (mesh-discretisation bias)
D3: warm- vs cold-started noisy 1% dual inversions (initialisation dependence),
    summarised from the per-realization inverse records.

Output: results/stage21_bias_diagnostic_real.json with non-empty `rows`,
`summary`, and run metadata.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_stage19_hybrid_solver_and_reversal as S19  # noqa: E402
import run_stage10_upgrades as S10  # noqa: E402

ONNX = str(ROOT / "web_simulator/models/deeponet_reparam_seed42.onnx")
RES = ROOT / "results"
CKPT = RES / "checkpoints"
OUT = RES / "stage21_bias_diagnostic_real.json"


def build_states():
    rng = np.random.default_rng(200020)
    bands = [("weak", 0.04, 0.15), ("transitional", 0.15, 0.60), ("strong", 0.60, 1.50)]
    states = []
    for band, lo, hi in bands:
        for _ in range(5):
            R = float(rng.uniform(3.0, 7.0))
            C0 = float(10 ** rng.uniform(np.log10(15.0), np.log10(150.0)))
            sigma = -float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
            q = float(S19.q_from_sigma(sigma, R))
            states.append({"band": band, "R": R, "C0": C0, "sigma": sigma, "q": q})
    return states


def load_records():
    npz = RES / "stage21_inverse_records_real.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        keys = [k for k in z.files if k not in ("y_obs", "sigmas")]
        n = len(z[keys[0]])
        return [{k: z[k][i].item() if hasattr(z[k][i], "item") else z[k][i] for k in keys}
                for i in range(n)]
    pkl = CKPT / "inverse_records.pkl"
    if pkl.exists():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)["records"]
    raise SystemExit("no inverse records available; run the inverse study first")


def main():
    t0 = time.perf_counter()
    with open(CKPT / "truth_states.pkl", "rb") as fh:
        truth = pickle.load(fh)
    with open(CKPT / "state_data.pkl", "rb") as fh:
        state_data = pickle.load(fh)

    op = S10.OnnxOperator(ONNX)
    data = S10.load_npz(S10.DATA240)
    rev_sur, _ = S19.fit_reversal_surrogate(op, data)
    S19._OP, S19._REV, S19._STATE_DATA = op, rev_sur, state_data

    states = build_states()
    records = load_records()

    result = S19.run_bias_diagnostics(op, rev_sur, states, truth, state_data, records)
    if not result.get("rows"):
        raise SystemExit("bias diagnostic returned no rows")

    required = {"state", "band", "diagnostic", "q_rel", "R_rel", "sigma_rel",
                "converged", "boundary_hit", "se_lnq"}
    missing = required - set(result["rows"][0])
    if missing:
        raise SystemExit(f"bias rows missing keys: {missing}")

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                capture_output=True, text=True).stdout.strip() or "uncommitted",
            "n_rows": len(result["rows"]),
            "n_states": len(states),
            "elapsed_time_s": time.perf_counter() - t0,
            "noise_for_D3": 0.01,
        },
        "rows": result["rows"],
        "summary": result["summary"],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} with {len(result['rows'])} rows")
    for k, v in result["summary"].items():
        if isinstance(v, dict):
            print(k, json.dumps(v))


if __name__ == "__main__":
    main()
