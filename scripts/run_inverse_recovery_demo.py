"""Stage 6A.2 - inverse parameter-recovery demo on held-out states.

For five held-out `interp_test` states we synthesise a five-bias I-V curve with
the exact SG/FV reference solver, add 1 % Gaussian measurement noise, and invert
the curve with the packaged PI-DeepONet (ONNX):

    measurement:  I_meas(V_k; C0, q_w*, R*),  V_k = -50 ... +50 mV
    search:       400 candidate (q_w, R) pairs x 5 biases = 2000 forward evals
    refinement:   Nelder-Mead from the best three grid candidates

The grid sweep is evaluated as one ONNX batch (the operator is batched), so the
wall time is compared with the 2000 equivalent SG/FV solves.

Outputs
    results/inverse_recovery_demo.json
    results/tables/table6_inverse_parameter_recovery.csv
    results/statistics.json  ->  key "inverse_demo"

The measured current is almost independent of q_w at fixed bulk salt (symmetric
electrolyte with nearly equal ionic diffusivities), so the demo reports the
identifiability of each parameter explicitly instead of hiding a flat direction.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from generate_parametric_dataset import solve_state  # noqa: E402
from pnp_da_pinn.conservative import physical_current  # noqa: E402
from operators.constants import (  # noqa: E402
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

DATASET = os.path.join(ROOT, "data", "processed", "pnp_parametric_240.npz")
ONNX = os.path.join(ROOT, "web_simulator", "models", "deeponet_reparam_seed42.onnx")
OUT_JSON = os.path.join(ROOT, "results", "inverse_recovery_demo.json")
OUT_CSV = os.path.join(ROOT, "results", "tables", "table6_inverse_parameter_recovery.csv")
STATS = os.path.join(ROOT, "results", "statistics.json")

V_BIAS = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
N_CAND = 400
N_Q = 20
N_R = 20
NOISE_LEVEL = 0.01
# Five interp_test states spanning |q_w| ~ 0.002-0.046 and R ~ 3.4-7.3 nm.
STATE_IDS = [175, 147, 154, 172, 161]
T_SGFV_MEDIAN_S = 0.2013  # table1; used only for the counterfactual cost

_R, _Z, *_ = make_grid(NR, NZ)
_WEIGHTS = annular_weights(_R)
_DZ = float(_Z[1] - _Z[0])
_SESSION = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
_INPUT = _SESSION.get_inputs()[0].name
_OUTPUTS = [o.name for o in _SESSION.get_outputs()]


def _bernoulli(x):
    xc = np.clip(x, -80.0, 80.0)
    small = np.abs(xc) < 1e-8
    series = 1.0 - xc / 2.0 + xc ** 2 / 12.0
    xs = np.where(small, 1.0, xc)
    return np.where(small, series, xs / np.expm1(xs))


def operator_current(mu):
    """Mean axial current (pA) predicted by the ONNX DeepONet.

    mu: (B, 4) array of (V_app [mV], C0 [mol/m^3], q_w, R [nm]).
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    out = _SESSION.run(None, {_INPUT: mu.astype(np.float32)})
    fields = dict(zip(_OUTPUTS, out))
    cp, cm, psi = fields["c_plus"], fields["c_minus"], fields["psi"]
    dpsi = psi[:, :, 1:] - psi[:, :, :-1]
    jp = -(cp[:, :, 1:] * _bernoulli(-dpsi)
           - cp[:, :, :-1] * _bernoulli(dpsi)) / _DZ
    jm = -(cm[:, :, 1:] * _bernoulli(dpsi)
           - cm[:, :, :-1] * _bernoulli(-dpsi)) / _DZ
    dimless = 2.0 * np.pi * np.sum(
        (DIFF_CATION * jp - DIFF_ANION * jm) * _WEIGHTS[None, :, None], axis=1)
    scale = FARADAY * mu[:, 1] * (mu[:, 3] * 1e-9) ** 2 / (L_NM * 1e-9) * 1e12
    return (dimless * scale[:, None]).mean(axis=1)


def sigma_mC(q_w, R_nm):
    return float(q_w) * REL_PERMITTIVITY * EPSILON_0 * V_T / (float(R_nm) * 1e-9) * 1.0e3


def measured_curve(C0, q, R, rng):
    values = []
    for V in V_BIAS:
        sol, cfg, grid, _, _ = solve_state(V, C0, q, R)
        values.append(float(physical_current(sol, grid, cfg).mean()) * 1e12)
    clean = np.asarray(values, dtype=float)
    noisy = clean * (1.0 + NOISE_LEVEL * rng.standard_normal(clean.size))
    return clean, noisy


def candidate_box(dataset):
    mu = dataset["mu"]
    q_lo = float(mu[:, 2].min())
    q_hi = 0.0
    r_lo = float(mu[:, 3].min())
    r_hi = float(mu[:, 3].max())
    qs = np.linspace(q_lo, q_hi, N_Q)
    # stratified jitter inside each cell, deterministic
    rng = np.random.default_rng(20261019)
    dq = (q_hi - q_lo) / (N_Q - 1)
    dr = (r_hi - r_lo) / (N_R - 1)
    q_grid = (qs + rng.uniform(-0.35 * dq, 0.35 * dq, N_Q)).clip(q_lo, q_hi)
    r_grid = (np.linspace(r_lo, r_hi, N_R)
              + rng.uniform(-0.35 * dr, 0.35 * dr, N_R)).clip(r_lo, r_hi)
    qq, rr = np.meshgrid(q_grid, r_grid, indexing="ij")
    return np.column_stack([qq.ravel(), rr.ravel()]), (q_lo, q_hi, r_lo, r_hi)


def invert_state(idx, dataset, candidates, box, with_refine=True):
    row = dataset["mu"][idx]
    C0, q_true, R_true = float(row[1]), float(row[2]), float(row[3])
    rng = np.random.default_rng(1000 + idx)
    clean, meas = measured_curve(C0, q_true, R_true, rng)

    q_lo, q_hi, r_lo, r_hi = box
    qc = candidates[:, 0]
    rc = candidates[:, 1]
    mu = np.empty((len(qc) * V_BIAS.size, 4), dtype=float)
    mu[:, 0] = np.tile(V_BIAS, len(qc))
    mu[:, 1] = C0
    mu[:, 2] = np.repeat(qc, V_BIAS.size)
    mu[:, 3] = np.repeat(rc, V_BIAS.size)

    t0 = time.perf_counter()
    pred = operator_current(mu).reshape(len(qc), V_BIAS.size)
    grid_wall = time.perf_counter() - t0
    scale = float(np.mean(meas ** 2))
    cost = np.mean((pred - meas) ** 2, axis=1) / scale
    order = np.argsort(cost)
    best = order[0]

    refine_wall = 0.0
    n_refine = 0
    q_hat, R_hat = float(qc[best]), float(rc[best])
    if with_refine:
        def objective(x):
            xq = float(np.clip(x[0], q_lo, q_hi))
            xr = float(np.clip(x[1], r_lo, r_hi))
            p = operator_current(np.column_stack(
                [V_BIAS, np.full(V_BIAS.size, C0),
                 np.full(V_BIAS.size, xq), np.full(V_BIAS.size, xr)]))
            return float(np.mean((p - meas) ** 2) / scale)

        t0 = time.perf_counter()
        options = {"xatol": 1e-7, "fatol": 1e-12, "maxiter": 400}
        best_res = None
        for j in order[:3]:
            res = minimize(objective, [qc[j], rc[j]], method="Nelder-Mead",
                           options=options)
            n_refine += int(res.nfev)
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        refine_wall = time.perf_counter() - t0
        q_hat, R_hat = (float(np.clip(best_res.x[0], q_lo, q_hi)),
                        float(np.clip(best_res.x[1], r_lo, r_hi)))

    # Identifiability: profile the cost over q_w at the recovered R.
    q_profile = np.linspace(q_lo, q_hi, 41)
    prof_pred = operator_current(np.column_stack(
        [np.tile(V_BIAS, q_profile.size), np.full(q_profile.size * V_BIAS.size, C0),
         np.repeat(q_profile, V_BIAS.size), np.full(q_profile.size * V_BIAS.size, R_hat)]))
    prof_pred = prof_pred.reshape(q_profile.size, V_BIAS.size)
    prof_cost = np.mean((prof_pred - meas) ** 2, axis=1) / scale
    cmin = float(prof_cost.min())
    # width of the set allowed within the 1 % noise floor (chi-square +1)
    allowed = q_profile[prof_cost <= cmin + NOISE_LEVEL ** 2]
    q_span = float(allowed.max() - allowed.min()) if allowed.size else 0.0

    # Sensitivity of the measured curve to q_w across the full prior box.
    q_sens = np.array([q_lo, q_hi])
    sens_pred = operator_current(np.column_stack(
        [np.tile(V_BIAS, 2), np.full(2 * V_BIAS.size, C0),
         np.repeat(q_sens, V_BIAS.size), np.full(2 * V_BIAS.size, R_hat)]))
    sens_pred = sens_pred.reshape(2, V_BIAS.size)
    ref = np.abs(sens_pred[0])
    mask = ref > 1e-2 * ref.max()
    d_i_over_i = float(np.max(
        np.abs(sens_pred[0][mask] - sens_pred[1][mask]) / ref[mask]))

    q_rel = abs(q_hat - q_true) / abs(q_true)
    r_rel = abs(R_hat - R_true) / R_true
    s_true, s_hat = sigma_mC(q_true, R_true), sigma_mC(q_hat, R_hat)
    s_rel = abs(s_hat - s_true) / abs(s_true)

    return {
        "state_index": int(idx),
        "V_app_mV": V_BIAS.tolist(),
        "C0_mol_m3": C0,
        "lambda_D_over_R": float(dataset["lambda_D_over_R"][idx]),
        "true": {"q_w": q_true, "sigma_mC_m2": s_true, "R_nm": R_true},
        "measured_pA": meas.tolist(),
        "clean_pA": clean.tolist(),
        "recovered": {"q_w": q_hat, "sigma_mC_m2": s_hat, "R_nm": R_hat},
        "relative_error": {"q_w": q_rel, "sigma": s_rel, "R_nm": r_rel},
        "grid": {
            "n_candidates": int(len(qc)),
            "n_forward_evals": int(len(mu)),
            "operator_wall_s": grid_wall,
            "best_cost": float(cost[best]),
        },
        "refine": {
            "n_forward_evals": n_refine,
            "operator_wall_s": refine_wall,
            "cost": float(best_res.fun) if with_refine else float(cost[best]),
        },
        "identifiability": {
            "q_w_flat_direction": bool(q_span > 0.5 * (q_hi - q_lo)),
            "q_w_1sigma_span": q_span,
            "max_relative_current_change_across_q_box": d_i_over_i,
            "noise_level": NOISE_LEVEL,
        },
    }


def main():
    with np.load(DATASET) as z:
        dataset = {k: z[k] for k in z.files}
    candidates, box = candidate_box(dataset)
    rows = [invert_state(i, dataset, candidates, box) for i in STATE_IDS]

    n_grid = int(sum(r["grid"]["n_forward_evals"] for r in rows))
    wall_grid = float(sum(r["grid"]["operator_wall_s"] for r in rows))
    wall_refine = float(sum(r["refine"]["operator_wall_s"] for r in rows))
    n_refine = int(sum(r["refine"]["n_forward_evals"] for r in rows))
    sgfv_grid = n_grid * T_SGFV_MEDIAN_S

    def med(key, sub="relative_error"):
        return float(np.median([r[sub][key] for r in rows]))

    summary = {
        "n_states": len(rows),
        "protocol": {
            "biases_mV": V_BIAS.tolist(),
            "noise_level": NOISE_LEVEL,
            "candidates_per_state": N_CAND,
            "forward_evals_per_state": N_CAND * V_BIAS.size,
            "operator": "PI-DeepONet (reparam, seed 42, ONNX)",
        },
        "recovery": {
            "median_relative_error_R": med("R_nm"),
            "median_relative_error_q_w": med("q_w"),
            "median_relative_error_sigma": med("sigma"),
            "R_recovered_within_2pct": bool(med("R_nm") < 0.02),
            "q_w_identifiable": bool(med("q_w") < 0.10),
        },
        "cost": {
            "operator_grid_wall_s": wall_grid,
            "operator_grid_wall_s_per_state": wall_grid / len(rows),
            "operator_refine_wall_s": wall_refine,
            "operator_refine_evals": n_refine,
            "sgfv_equivalent_wall_s": sgfv_grid,
            "sgfv_equivalent_wall_s_per_state": sgfv_grid / len(rows),
            "speedup_grid": sgfv_grid / wall_grid if wall_grid else None,
            "t_sgfv_median_s": T_SGFV_MEDIAN_S,
        },
        "identifiability": {
            "max_relative_current_change_across_q_box": float(np.median(
                [r["identifiability"]["max_relative_current_change_across_q_box"]
                 for r in rows])),
            "noise_level": NOISE_LEVEL,
            "note": ("Total conductance is nearly invariant to fixed surface charge "
                     "for a symmetric electrolyte with near-equal ionic diffusivities; "
                     "the q_w direction is flat at the 1% noise level, while R is "
                     "well identified."),
        },
        "states": rows,
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "state_index", "C0_mol_m3", "lambda_D_over_R",
            "q_w_true", "q_w_hat", "q_w_rel_err",
            "sigma_true_mC_m2", "sigma_hat_mC_m2", "sigma_rel_err",
            "R_true_nm", "R_hat_nm", "R_rel_err",
            "grid_forward_evals", "grid_wall_s", "refine_forward_evals",
        ])
        for r in rows:
            writer.writerow([
                r["state_index"], f"{r['C0_mol_m3']:.6g}",
                f"{r['lambda_D_over_R']:.8g}",
                f"{r['true']['q_w']:.8g}", f"{r['recovered']['q_w']:.8g}",
                f"{r['relative_error']['q_w']:.6g}",
                f"{r['true']['sigma_mC_m2']:.8g}",
                f"{r['recovered']['sigma_mC_m2']:.8g}",
                f"{r['relative_error']['sigma']:.6g}",
                f"{r['true']['R_nm']:.8g}", f"{r['recovered']['R_nm']:.8g}",
                f"{r['relative_error']['R_nm']:.6g}",
                r["grid"]["n_forward_evals"],
                f"{r['grid']['operator_wall_s']:.6g}",
                r["refine"]["n_forward_evals"],
            ])

    with open(STATS) as fh:
        stats = json.load(fh)
    stats["inverse_demo"] = summary
    with open(STATS, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(summary["recovery"], indent=2))
    print(json.dumps(summary["cost"], indent=2))
    print(json.dumps(summary["identifiability"], indent=2))
    print(f"[inverse] wrote {OUT_JSON}")
    print(f"[inverse] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
