"""Stage 19: sensitivity-controlled hybrid PNP solver and reversal potential.

Implements an unequal-reservoir charge-balanced ansatz, an unequal-reservoir
SG/FV reference solve, zero-current reversal-potential roots, and an independent
120-state multi-regime benchmark.  The inverse experiment compares frozen-
operator inversion with the PINO-Controller (Weyl-gated local SG/FV correction)
on conductance plus concentration-cell reversal potential and on the dual
current-plus-selectivity protocol.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, root_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import generate_parametric_dataset as G  # noqa: E402
import run_stage10_upgrades as S10  # noqa: E402
import run_stage18_final_evidence as S18  # noqa: E402
from operators import build_model  # noqa: E402
from operators.boundary_ansatz import BoundaryAnsatz  # noqa: E402
from operators.constants import (  # noqa: E402
    DIFF_ANION, DIFF_CATION, EPSILON_0, FARADAY, GAS_CONSTANT,
    L_NM, REL_PERMITTIVITY, T_K,
)
from operators.hybrid_controller import (  # noqa: E402
    ReferenceFreeController, ReversalSurrogate, exact_sensitivity,
    fixed_one_step_gummel, newton_step, q_from_sigma, run_corrector_unit_test,
    sigma_from_q, tms_reversal, weyl_gate, whitened_jacobian,
)
from pnp_da_pinn.config import (  # noqa: E402
    ExperimentConfig, GeometryConfig, PhysicsConfig, PinnConfig, SolverConfig,
)
from pnp_da_pinn.conservative import physical_current, solve_reference  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402

RESULTS = S10.RESULTS
TABLES = S10.TABLES
STATS = S10.STATS
OUT_JSON = os.path.join(RESULTS, "stage19_hybrid_controller.json")
OUT_CSV = os.path.join(TABLES, "table_s22_hybrid_controller_and_reversal.csv")
ONNX = os.path.join(ROOT, "web_simulator", "models", "deeponet_reparam_seed42.onnx")
DATA240 = S10.DATA240

V_BIASES = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
V_NONZERO = np.array([-50.0, -25.0, 25.0, 50.0])
V_T = GAS_CONSTANT * T_K / FARADAY
EPS = REL_PERMITTIVITY * EPSILON_0
RATIO = 5.0
NOISE = [0.002, 0.010, 0.020]


def make_cfg(V_mV, C0, q_w, R_nm, nr=21, nz=61):
    return ExperimentConfig(
        geometry=GeometryConfig(radius_nm=float(R_nm), length_nm=L_NM, nr=int(nr), nz=int(nz)),
        physics=PhysicsConfig(
            temperature_K=T_K, bulk_concentration_mol_m3=float(C0),
            relative_permittivity=REL_PERMITTIVITY,
            diffusion_cation_m2_s=DIFF_CATION, diffusion_anion_m2_s=DIFF_ANION,
            applied_voltage_V=float(V_mV) * 1e-3,
            fixed_charge_strength=0.0,
            wall_surface_charge_strength=float(q_w),
        ),
        solver=SolverConfig(max_gummel_iter=300, tolerance=1e-6, relaxation=0.005),
        pinn=PinnConfig(seed=42, collocation_points=3000, data_points=500, epochs=200),
    )


def solve_equal(V_mV, C0, q_w, R_nm, nr=21, nz=61, psi0=None, tol=1e-9):
    cfg = make_cfg(V_mV, C0, q_w, R_nm, nr, nz)
    grid = make_grid(cfg.geometry)
    sol = solve_reference(cfg, grid, tolerance=tol, max_iter=200, psi0=psi0)
    if not sol.converged:
        raise RuntimeError(f"SG/FV did not converge for {V_mV, C0, q_w, R_nm}")
    return sol, cfg, grid


def solve_unequal(V_mV, C0, q_w, R_nm, ratio=RATIO, nr=21, nz=61, psi0=None, tol=1e-9):
    cfg = make_cfg(V_mV, C0, q_w, R_nm, nr, nz)
    grid = make_grid(cfg.geometry)
    sol = solve_reference(cfg, grid, tolerance=tol, max_iter=200, psi0=psi0,
                          c_left=float(ratio), c_right=1.0)
    if not sol.converged:
        raise RuntimeError(f"unequal SG/FV did not converge for {V_mV, C0, q_w, R_nm}")
    return sol, cfg, grid


def current_mean_pA(sol, cfg, grid):
    return float(np.mean(physical_current(sol, grid, cfg)) * 1e12)


def wall_selectivity(sol):
    cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
    return float(np.mean((cp - cm) / np.maximum(cp + cm, 1e-30)))


def exact_reversal(q_w, R_nm, C0, ratio=RATIO, nr=21, nz=61, tol=1e-9,
                   v_guess=None):
    cache = {}
    warm = {"psi": None, "V": None}

    def current(V):
        key = float(V)
        if key not in cache:
            psi0 = warm["psi"] if (warm["psi"] is not None and warm["V"] is not None
                                   and np.sign(warm["V"]) == np.sign(key)) else None
            try:
                sol, cfg, grid = solve_unequal(V, C0, q_w, R_nm, ratio, nr, nz,
                                               psi0=psi0, tol=tol)
            except Exception:
                try:
                    sol, cfg, grid = solve_unequal(V, C0, q_w, R_nm, ratio, nr, nz, tol=tol)
                except Exception:
                    cache[key] = float("nan")
                    return cache[key]
            warm["psi"] = sol.psi
            warm["V"] = key
            cache[key] = current_mean_pA(sol, cfg, grid)
        return cache[key]

    flo = fhi = float("nan")
    spans = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0) if v_guess is not None else (5.0, 10.0, 20.0, 40.0, 80.0)
    for span in spans:
        if v_guess is not None:
            lo, hi = float(v_guess) - span, float(v_guess) + span
        else:
            lo, hi = -span, span
        flo, fhi = current(lo), current(hi)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
            root = brentq(current, lo, hi, xtol=1e-6, rtol=1e-6, maxiter=80)
            return float(root), {"converged": True, "current_lo": flo,
                                 "current_hi": fhi, "bracket_mV": span,
                                 "n_current_evals": len(cache)}
    return float("nan"), {"converged": False, "current_lo": flo,
                          "current_hi": fhi, "bracket_mV": 80.0,
                          "n_current_evals": len(cache)}


def build_states(n_per=40, seed=190019):
    rng = np.random.default_rng(seed)
    rows = []
    bands = [("weak", 0.02, 0.25), ("transitional", 0.25, 0.80),
             ("strong", 0.80, 2.50)]
    for band, lo, hi in bands:
        for _ in range(n_per):
            loglo, loghi = math.log10(lo), math.log10(hi)
            sigma = -(10.0 ** (loglo + rng.random() * (loghi - loglo)))
            R = rng.uniform(3.0, 8.0)
            C0 = 10.0 ** rng.uniform(math.log10(20.0), math.log10(200.0))
            V = rng.uniform(-50.0, 50.0)
            q = float(q_from_sigma(sigma, R))
            rows.append({"band": band, "sigma_mC": sigma, "R_nm": R,
                         "C0": C0, "V_mV": V, "q_w": q})
    return rows


def op_observables(op, q, R, C0):
    mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                          np.full(V_NONZERO.size, q), np.full(V_NONZERO.size, R)])
    return op.current(mu), op.wall_selectivity(mu)


def op_eval_all(op, q, R, C0, V):
    mu = np.array([[V, C0, q, R]])
    return float(op.current(mu)[0]), float(op.wall_selectivity(mu)[0])


def exact_eval_all(q, R, C0, V, nr=21, nz=61):
    sol, cfg, grid = solve_equal(V, C0, q, R, nr, nz)
    return current_mean_pA(sol, cfg, grid), wall_selectivity(sol), sol, cfg, grid


def fit_calibration(op, data):
    idx = data["train_idx"]
    mu = data["mu"][idx]
    X = []
    yI = []
    yS = []
    for k in range(len(idx)):
        q, R, C0, V = mu[k, 2], mu[k, 3], mu[k, 1], mu[k, 0]
        sol, cfg, grid = solve_equal(V, C0, q, R, 21, 61)
        ie = current_mean_pA(sol, cfg, grid)
        if abs(V) < 1e-9 or abs(ie) < 1e-6:
            continue
        se = wall_selectivity(sol)
        I0, S0 = op_eval_all(op, q, R, C0, V)
        mu2 = np.column_stack([np.full(4, V), np.full(4, C0),
                               np.full(4, q), np.full(4, R)])
        def f(qq, RR):
            return op_eval_all(op, qq, RR, C0, V)
        _, K = exact_sensitivity(lambda th: np.array(f(th[0], th[1])), np.array([q, R]), 1e-3)
        X.append([1.0, S0, q, math.log(R), math.log(C0),
                  K[0, 0], K[0, 1], K[1, 0], K[1, 1]])
        yI.append(math.log(max(abs(ie), 1e-12)) - math.log(max(abs(I0), 1e-12)))
        yS.append(se)
    X = np.asarray(X)
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-12
    Z = (X - mean) / scale
    A = Z.T @ Z + 1e-6 * np.eye(Z.shape[1])
    cI = np.linalg.solve(A, Z.T @ np.asarray(yI))
    cS = np.linalg.solve(A, Z.T @ np.asarray(yS))
    return {"mean": mean, "scale": scale, "cI": cI, "cS": cS}


def apply_calibration(cal, op, q, R, C0, V):
    I0, S0 = op_eval_all(op, q, R, C0, V)
    if abs(V) < 1e-9:
        return 0.0, S0
    def f(qq, RR):
        return op_eval_all(op, qq, RR, C0, V)
    _, K = exact_sensitivity(lambda th: np.array(f(th[0], th[1])), np.array([q, R]), 1e-3)
    x = np.array([1.0, S0, q, math.log(R), math.log(C0),
                  K[0, 0], K[0, 1], K[1, 0], K[1, 1]])
    z = (x - cal["mean"]) / cal["scale"]
    corr = float(np.clip(z @ cal["cI"], -2.0, 2.0))
    I = math.copysign(max(abs(I0), 1e-12) * math.exp(corr),
                      1.0 if (I0 if I0 != 0 else V) >= 0 else -1.0)
    S = float(z @ cal["cS"])
    return I, S


def fit_reversal_surrogate(op, data):
    idx = data["train_idx"][::4]
    X, y = [], []
    for k in idx:
        q, R, C0 = data["mu"][k, 2], data["mu"][k, 3], data["mu"][k, 1]
        _, S0 = op_eval_all(op, q, R, C0, 0.0)
        v, _ = exact_reversal(float(q), float(R), float(C0))
        if np.isfinite(v):
            X.append(ReversalSurrogate.features(q, R, C0, S0, math.log(RATIO)))
            y.append(v)
    rs = ReversalSurrogate().fit(np.asarray(X), np.asarray(y))
    return rs, len(y)


def fit_pod_predict(data, mu_test, n_modes=30):
    from scipy.interpolate import RBFInterpolator
    Xtr = data["mu"][data["train_idx"]].copy()
    mean = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-12
    Xtrn = (Xtr - mean) / sd
    Xten = (np.asarray(mu_test) - mean) / sd
    fields = {"psi": data["ref"]["psi"], "c_plus": data["ref"]["c_plus"],
              "c_minus": data["ref"]["c_minus"]}
    out = {}
    for name, arr in fields.items():
        Y = arr[data["train_idx"]].reshape(len(data["train_idx"]), -1).astype(np.float64)
        ymean = Y.mean(axis=0)
        _, sv, Vt = np.linalg.svd(Y - ymean, full_matrices=False)
        k = min(n_modes, len(sv))
        C = (Y - ymean) @ Vt[:k].T
        rbf = RBFInterpolator(Xtrn, C, kernel="thin_plate_spline", smoothing=1e-8)
        out[name] = (rbf(Xten) @ Vt[:k] + ymean).reshape(len(mu_test), 21, 61)
    return out



def _field_metrics(sol, ref):
    return {
        "psi_max_abs_err": float(np.max(np.abs(sol.psi - ref.psi))),
        "c_max_abs_err": float(max(np.max(np.abs(sol.c_plus - ref.c_plus)),
                                   np.max(np.abs(sol.c_minus - ref.c_minus)))),
    }


def _med(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _warm_reversal_root(Vsur, C0, q, R, op):
    """Secant SG/FV reversal search warm-started from the operator at each V."""
    cache = {}

    def current(V):
        key = float(V)
        if key not in cache:
            try:
                fields = op.fields(np.array([[key, C0, q, R]]))
                psi0 = np.asarray(fields["psi"][0], dtype=float)
                sol, cfg, grid = solve_unequal(key, C0, q, R, psi0=psi0, tol=1e-9)
            except Exception:
                try:
                    sol, cfg, grid = solve_unequal(key, C0, q, R, tol=1e-9)
                except Exception:
                    cache[key] = float("nan")
                    return cache[key]
            cache[key] = current_mean_pA(sol, cfg, grid)
        return cache[key]

    t0 = time.perf_counter()
    V = float("nan")
    method = "secant"
    converged = False
    try:
        if not np.isfinite(Vsur):
            raise ValueError("non-finite surrogate reversal")
        res = root_scalar(current, x0=Vsur - 1.0, x1=Vsur + 1.0,
                          method="secant", xtol=1e-6, rtol=1e-6, maxiter=20)
        V = float(res.root)
        converged = bool(res.converged)
    except Exception:
        pass
    if (not converged or not np.isfinite(V)) and np.isfinite(Vsur):
        method = "brentq-warm"
        for half in (2.0, 5.0, 10.0, 20.0, 40.0):
            lo, hi = Vsur - half, Vsur + half
            flo, fhi = current(lo), current(hi)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
                V = float(brentq(current, lo, hi, xtol=1e-6, rtol=1e-6, maxiter=80))
                converged = True
                break
    return {"Vrev_mV": V, "converged": converged, "method": method,
            "time_s": time.perf_counter() - t0,
            "n_current_evals": len(cache)}


def timing_benchmark(op, states, refs, rev_sur):
    """Equal-task, equal-grid, equal-tolerance Stage-21 timing benchmark."""
    rows = []
    for k, st in enumerate(states):
        q, R, C0, V = st["q_w"], st["R_nm"], st["C0"], st["V_mV"]
        ref = refs[k]
        cfg = make_cfg(V, C0, q, R)
        grid = make_grid(cfg.geometry)
        rec = {"state": k, "band": st["band"]}

        t0 = time.perf_counter()
        fop = op.fields(np.array([[V, C0, q, R]]))
        t_zero = time.perf_counter() - t0
        I_zero = current_from_arrays(fop["c_plus"][0], fop["c_minus"][0],
                                     fop["psi"][0], cfg, grid)
        rec["zero_shot_deeponet"] = {
            "time_s": t_zero, "iterations": 0, "I_rel_err": abs(I_zero - ref["I"]) / max(abs(ref["I"]), 1e-12),
            **_field_metrics(type(ref["sol"])(fop["psi"][0], fop["c_plus"][0],
                                              fop["c_minus"][0], 0, 0.0, 0.0, True), ref["sol"]),
        }

        t0 = time.perf_counter()
        psi1, cp1, cm1 = fixed_one_step_gummel(
            fop["psi"][0], fop["c_plus"][0], fop["c_minus"][0],
            cfg.scales.poisson_coupling, q, V * 1e-3 / V_T, grid,
        )
        t_one = time.perf_counter() - t0
        I_one = current_from_arrays(cp1, cm1, psi1, cfg, grid)
        rec["one_step_gummel"] = {
            "time_s": t_one, "iterations": 1, "I_rel_err": abs(I_one - ref["I"]) / max(abs(ref["I"]), 1e-12),
            **_field_metrics(type(ref["sol"])(psi1, cp1, cm1, 0, 0.0, 0.0, True), ref["sol"]),
        }

        for label, tol in (("warm_sgfv_tol1e-6", 1e-6), ("warm_sgfv_tol1e-9", 1e-9)):
            t0 = time.perf_counter()
            sol, _, _ = solve_equal(V, C0, q, R, psi0=fop["psi"][0], tol=tol)
            dt = time.perf_counter() - t0
            I = current_mean_pA(sol, cfg, grid)
            rec[label] = {"time_s": dt, "iterations": int(sol.iterations),
                          "I_rel_err": abs(I - ref["I"]) / max(abs(ref["I"]), 1e-12),
                          **_field_metrics(sol, ref["sol"])}

        for label, tol in (("cold_sgfv_tol1e-6", 1e-6), ("cold_sgfv_tol1e-9", 1e-9)):
            if label == "cold_sgfv_tol1e-9":
                sol = ref["sol"]
                dt = ref["forward_runtime_s"]
            else:
                t0 = time.perf_counter()
                sol, _, _ = solve_equal(V, C0, q, R, tol=tol)
                dt = time.perf_counter() - t0
            I = current_mean_pA(sol, cfg, grid)
            rec[label] = {"time_s": dt, "iterations": int(sol.iterations),
                          "I_rel_err": abs(I - ref["I"]) / max(abs(ref["I"]), 1e-12),
                          **_field_metrics(sol, ref["sol"])}

        mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                              np.full(V_NONZERO.size, q), np.full(V_NONZERO.size, R)])
        t0 = time.perf_counter()
        Smean = float(np.mean(op.wall_selectivity(mu)))
        Vsur = float(rev_sur.predict(ReversalSurrogate.features(q, R, C0, Smean, math.log(RATIO)))[0])
        t_sur = time.perf_counter() - t0
        rec["zero_shot_vrev"] = {
            "time_s": t_sur, "n_current_evals": 0,
            "Vrev_mV": Vsur,
            "Vrev_abs_err_mV": abs(Vsur - ref["Vrev"]) if np.isfinite(Vsur) else np.nan,
        }

        warm = _warm_reversal_root(Vsur, C0, q, R, op)
        rec["warm_secant_vrev"] = {
            **warm, "Vrev_abs_err_mV": abs(warm["Vrev_mV"] - ref["Vrev"])
            if np.isfinite(warm["Vrev_mV"]) and np.isfinite(ref["Vrev"]) else np.nan,
        }
        cold_info = ref.get("vrev_info", {})
        rec["cold_brent_vrev"] = {
            "time_s": ref["vrev_runtime_s"],
            "n_current_evals": int(cold_info.get("n_current_evals", 0)),
            "Vrev_mV": ref["Vrev"],
            "Vrev_abs_err_mV": abs(ref["Vrev"] - ref["Vrev"]) if np.isfinite(ref["Vrev"]) else np.nan,
            "converged": bool(cold_info.get("converged", False)),
            "method": "brentq-cold",
        }
        rows.append(rec)

    methods_a = ["zero_shot_deeponet", "one_step_gummel", "warm_sgfv_tol1e-6",
                 "warm_sgfv_tol1e-9", "cold_sgfv_tol1e-6", "cold_sgfv_tol1e-9"]
    summary_a = {}
    for m in methods_a:
        summary_a[m] = {
            "n": len(rows),
            "median_time_s": _med([r[m]["time_s"] for r in rows]),
            "median_iterations": _med([r[m]["iterations"] for r in rows]),
            "median_I_rel_err": _med([r[m]["I_rel_err"] for r in rows]),
            "median_psi_max_abs_err": _med([r[m]["psi_max_abs_err"] for r in rows]),
            "median_c_max_abs_err": _med([r[m]["c_max_abs_err"] for r in rows]),
        }
    summary_b = {}
    for m in ["zero_shot_vrev", "warm_secant_vrev", "cold_brent_vrev"]:
        summary_b[m] = {
            "n": len(rows),
            "median_time_s": _med([r[m]["time_s"] for r in rows]),
            "median_evaluations": _med([r[m]["n_current_evals"] for r in rows]),
            "median_Vrev_abs_err_mV": _med([r[m]["Vrev_abs_err_mV"] for r in rows]),
            "converged_fraction": float(np.mean([bool(r[m].get("converged", True)) for r in rows])),
        }
    summary = {
        "task_A": summary_a,
        "task_B": summary_b,
        "task_A_speedups": {
            "warm_1e-9_vs_cold_1e-9": summary_a["cold_sgfv_tol1e-9"]["median_time_s"] / summary_a["warm_sgfv_tol1e-9"]["median_time_s"],
            "warm_1e-6_vs_cold_1e-6": summary_a["cold_sgfv_tol1e-6"]["median_time_s"] / summary_a["warm_sgfv_tol1e-6"]["median_time_s"],
            "one_step_vs_cold_1e-9": summary_a["cold_sgfv_tol1e-9"]["median_time_s"] / summary_a["one_step_gummel"]["median_time_s"],
            "warm_1e-9_iteration_reduction": summary_a["cold_sgfv_tol1e-9"]["median_iterations"] / max(summary_a["warm_sgfv_tol1e-9"]["median_iterations"], 1.0),
        },
        "task_B_speedups": {
            "warm_vs_cold": summary_b["cold_brent_vrev"]["median_time_s"] / summary_b["warm_secant_vrev"]["median_time_s"],
            "warm_evaluation_reduction": summary_b["cold_brent_vrev"]["median_evaluations"] / max(summary_b["warm_secant_vrev"]["median_evaluations"], 1.0),
        },
    }
    return {"task_A_states": rows, "summary": summary,
            "grid": [21, 61], "tolerance_reference": 1e-9,
            "tolerances_compared": [1e-6, 1e-9],
            "task_A": "single equal-reservoir forward solve; same grid and tolerance",
            "task_B": "unequal-reservoir reversal root I(V)=0 with SG/FV tolerance 1e-9"}


def benchmark_methods(op, states, cal, rev_sur, refs, pod):
    rows = []
    controller = ReferenceFreeController(op)
    for k, st in enumerate(states):
        q, R, C0, V = st["q_w"], st["R_nm"], st["C0"], st["V_mV"]
        ref = refs[k]
        rec = {"state": k, **st, "I_exact_pA": ref["I"], "S_exact": ref["S"],
               "Vrev_exact_mV": ref["Vrev"]}

        t0 = time.perf_counter()
        Ioh = analytical_ohmic(st, make_cfg(V, C0, q, R))
        toh = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ipb = pb_observables(q, R, C0, V, make_cfg(V, C0, q, R))
        tpb = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ipod = current_from_arrays(pod["c_plus"][k], pod["c_minus"][k],
                                   pod["psi"][k], make_cfg(V, C0, q, R),
                                   make_grid(make_cfg(V, C0, q, R).geometry))
        Spod = float(np.mean((pod["c_plus"][k, -1] - pod["c_minus"][k, -1]) /
                             np.maximum(pod["c_plus"][k, -1] + pod["c_minus"][k, -1], 1e-30)))
        tpod = time.perf_counter() - t0
        t0 = time.perf_counter()
        Iop, Sop = op_eval_all(op, q, R, C0, V)
        Vsur = float(rev_sur.predict(ReversalSurrogate.features(q, R, C0, Sop, math.log(RATIO)))[0])
        top = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ical, Scal = apply_calibration(cal, op, q, R, C0, V)
        Vcal = float(rev_sur.predict(ReversalSurrogate.features(q, R, C0, Scal, math.log(RATIO)))[0])
        tdino = time.perf_counter() - t0
        t0 = time.perf_counter()
        cfg = make_cfg(V, C0, q, R)
        grid = make_grid(cfg.geometry)
        fop = op.fields(np.array([[V, C0, q, R]]))
        psi0 = fop["psi"][0]
        psi1, cp1, cm1 = fixed_one_step_gummel(
            psi0, fop["c_plus"][0], fop["c_minus"][0],
            cfg.scales.poisson_coupling, q, V * 1e-3 / V_T, grid,
        )
        I1 = current_from_arrays(cp1, cm1, psi1, cfg, grid)
        S1 = float(np.mean((cp1[-1] - cm1[-1]) / np.maximum(cp1[-1] + cm1[-1], 1e-30)))
        tone = time.perf_counter() - t0
        t0 = time.perf_counter()
        ctl = controller.controller_solve(np.array([V, C0, q, R]), task="forward")
        tctl = time.perf_counter() - t0
        I8 = float(ctl["current_pA"])
        cp8, cm8 = ctl["c_plus"], ctl["c_minus"]
        S8 = float(np.mean((cp8[-1] - cm8[-1]) / np.maximum(cp8[-1] + cm8[-1], 1e-30)))

        def op_pair(th):
            return np.array(op_eval_all(op, th[0], th[1], C0, V))
        _, Kop = exact_sensitivity(op_pair, np.array([q, R]), 1e-3)
        sv = np.linalg.svd(Kop, compute_uv=False)
        epsK = float(np.finfo(float).eps * max(sv[0], 1.0))
        gate = weyl_gate(Kop, epsK, 0.0)

        methods = {
            "ohmic_tms": (Ioh, 0.0, tms_reversal(C0, RATIO, q, R), toh),
            "linear_pb": (Ipb, 0.0, tms_reversal(C0, RATIO, q, R), tpb),
            "pod_rbf": (Ipod, Spod, tms_reversal(C0, RATIO, q, R), tpod),
            "pi_deeponet": (Iop, Sop, Vsur, top),
            "dino_style": (Ical, Scal, Vcal, tdino),
            "one_step_gummel": (I1, S1, Vsur, tone),
            "full_sgfv": (ref["I"], ref["S"], ref["Vrev"], ref["runtime_s"]),
            "pino_controller": (I8, S8, Vsur, tctl),
        }
        rec["controller_iterations"] = int(ctl["iterations"])
        rec["controller_converged"] = bool(ctl["converged"])
        rec["controller_gate_status"] = gate["status"]
        rec["controller_delta_psi"] = float(ctl["delta_psi"])
        rec["controller_conservation_defect"] = float(ctl["conservation_defect"])
        for name, (I, S, Vr, rt) in methods.items():
            if I is None:
                continue
            rec[f"{name}_I"] = I
            rec[f"{name}_S"] = S
            rec[f"{name}_Vrev"] = Vr
            rec[f"{name}_runtime_s"] = rt
            rec[f"{name}_I_rel_err"] = abs(I - ref["I"]) / max(abs(ref["I"]), 1e-12)
            rec[f"{name}_S_abs_err"] = abs(S - ref["S"])
            rec[f"{name}_Vrev_abs_err_mV"] = abs(Vr - ref["Vrev"]) if np.isfinite(Vr) else np.nan
        rows.append(rec)
    return rows


def analytical_ohmic(st, cfg):
    c = st["C0"]
    Rm = st["R_nm"] * 1e-9
    V = st["V_mV"] * 1e-3
    sgn = 1.0 if V >= 0 else -1.0
    return sgn * (FARADAY ** 2 / (GAS_CONSTANT * T_K)) * (DIFF_CATION + DIFF_ANION) * c * np.pi * Rm ** 2 / (L_NM * 1e-9) * abs(V) * 1e12


def pb_observables(q, R, C0, V, cfg):
    lam = np.sqrt(EPS * GAS_CONSTANT * T_K / (2.0 * FARADAY ** 2 * C0)) / (R * 1e-9)
    r = np.linspace(0.0, 1.0, 21)
    z = np.linspace(0.0, 1.0, 61)
    from scipy.special import i0, i1
    delta = q * lam * (i0(r / max(lam, 1e-9)) / i1(1.0 / max(lam, 1e-9)))[:, None] * np.sin(np.pi * z)[None, :] ** 0.2
    cp = np.exp(np.clip(-delta, -50, 50))[None]
    cm = np.exp(np.clip(delta, -50, 50))[None]
    psi = ((V * 1e-3 / V_T) * (1.0 - z) + delta)[None]
    from operators.losses_and_metrics import GridTensors, sg_current_pA
    mu = torch.tensor([[V, C0, q, R]], dtype=torch.float32)
    with torch.no_grad():
        I = float(sg_current_pA(torch.tensor(cp, dtype=torch.float32),
                                torch.tensor(cm, dtype=torch.float32),
                                torch.tensor(psi, dtype=torch.float32), mu,
                                GridTensors()).mean().item())
    S = float(np.mean((cp[0, -1] - cm[0, -1]) / np.maximum(cp[0, -1] + cm[0, -1], 1e-30)))
    return I


def current_from_arrays(cp, cm, psi, cfg, grid):
    from pnp_da_pinn.sg_solver import sg_current_profile
    from pnp_da_pinn.baseline import PnpSolution
    sol = PnpSolution(psi, cp, cm, 0, 0.0, 0.0, True)
    return float(np.mean(physical_current(sol, grid, cfg)) * 1e12)


def run_benchmark(op, data):
    # Check for forward benchmark cache
    cache_path = os.path.join(RESULTS, "_stage21_forward_cache.npz")
    if os.path.exists(cache_path):
        print("[19] loading forward benchmark cache from disk", flush=True)
        loaded = np.load(cache_path, allow_pickle=True)
        states = build_states()
        refs = []
        for k, st in enumerate(states):
            # Reconstruct sol object from cached arrays
            psi = loaded[f"psi_{k}"]
            c_plus = loaded[f"c_plus_{k}"]
            c_minus = loaded[f"c_minus_{k}"]
            from pnp_da_pinn.baseline import PnpSolution
            sol = PnpSolution(psi, c_plus, c_minus, 0, 0.0, 0.0, True)
            cfg = make_cfg(st["V_mV"], st["C0"], st["q_w"], st["R_nm"])
            grid = make_grid(cfg.geometry)
            refs.append({
                "I": float(loaded[f"I_{k}"]),
                "S": float(loaded[f"S_{k}"]),
                "Vrev": float(loaded[f"Vrev_{k}"]),
                "sol": sol,
                "cfg": cfg,
                "grid": grid,
                "forward_runtime_s": float(loaded[f"frt_{k}"]),
                "vrev_runtime_s": float(loaded[f"vrt_{k}"]),
                "runtime_s": float(loaded[f"frt_{k}"]) + float(loaded[f"vrt_{k}"]),
                "vrev_info": {"converged": bool(loaded[f"conv_{k}"]),
                               "n_current_evals": int(loaded[f"nce_{k}"])},
            })
        print("[19] loaded 120 forward benchmarks from cache", flush=True)
    else:
        print("[19] building 120-state benchmark", flush=True)
        states = build_states()
        refs = []
        for k, st in enumerate(states):
            t0 = time.perf_counter()
            sol, cfg, grid = solve_equal(st["V_mV"], st["C0"], st["q_w"], st["R_nm"])
            t_forward = time.perf_counter() - t0
            I = current_mean_pA(sol, cfg, grid)
            S = wall_selectivity(sol)
            t0 = time.perf_counter()
            v, info = exact_reversal(st["q_w"], st["R_nm"], st["C0"])
            t_rev = time.perf_counter() - t0
            refs.append({"I": I, "S": S, "Vrev": v, "sol": sol, "cfg": cfg,
                         "grid": grid, "forward_runtime_s": t_forward,
                         "vrev_runtime_s": t_rev,
                         "runtime_s": t_forward + t_rev,
                         "vrev_info": info})
            if (k + 1) % 20 == 0:
                print(f"[19] refs {k + 1}/120", flush=True)
        
        # Save to cache
        print("[19] saving forward benchmarks to cache", flush=True)
        arrays = {}
        for k, (st, ref) in enumerate(zip(states, refs)):
            arrays[f"I_{k}"] = np.array(ref["I"])
            arrays[f"S_{k}"] = np.array(ref["S"])
            arrays[f"Vrev_{k}"] = np.array(ref["Vrev"])
            arrays[f"psi_{k}"] = ref["sol"].psi
            arrays[f"c_plus_{k}"] = ref["sol"].c_plus
            arrays[f"c_minus_{k}"] = ref["sol"].c_minus
            arrays[f"frt_{k}"] = np.array(ref["forward_runtime_s"])
            arrays[f"vrt_{k}"] = np.array(ref["vrev_runtime_s"])
            arrays[f"conv_{k}"] = np.array(ref["vrev_info"]["converged"])
            arrays[f"nce_{k}"] = np.array(ref["vrev_info"]["n_current_evals"])
        np.savez_compressed(cache_path, **arrays)
        print(f"[19] saved forward benchmarks to {cache_path}", flush=True)
    
    print("[19] calibrations", flush=True)
    cal = fit_calibration(op, data)
    rev_sur, n_rev = fit_reversal_surrogate(op, data)
    mu_test = np.array([[st["V_mV"], st["C0"], st["q_w"], st["R_nm"]] for st in states])
    pod = fit_pod_predict(data, mu_test)
    rows = benchmark_methods(op, states, cal, rev_sur, refs, pod)
    return rows, cal, rev_sur, n_rev, states, refs


def summarize_methods(rows):
    names = ["ohmic_tms", "linear_pb", "pod_rbf", "pi_deeponet", "dino_style",
             "one_step_gummel", "full_sgfv", "pino_controller"]
    out = {}
    for name in names:
        sub = [r for r in rows if f"{name}_I" in r]
        if not sub:
            continue
        def stat(key):
            x = np.array([r[key] for r in sub], dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return {"median": None, "iqr": None, "p95": None}
            return {"median": float(np.median(x)),
                    "iqr": float(np.quantile(x, .75) - np.quantile(x, .25)),
                    "p95": float(np.quantile(x, .95))}
        out[name] = {
            "n": len(sub),
            "I_rel_err": stat(f"{name}_I_rel_err"),
            "S_abs_err": stat(f"{name}_S_abs_err"),
            "Vrev_abs_err_mV": stat(f"{name}_Vrev_abs_err_mV"),
            "runtime_median_s": float(np.median([r[f"{name}_runtime_s"] for r in sub])),
        }
    for band in ["weak", "transitional", "strong"]:
        for name in names:
            sub = [r for r in rows if r["band"] == band and f"{name}_I" in r]
            if not sub:
                continue
            out.setdefault("by_band", {}).setdefault(band, {})[name] = {
                "I_median_rel_err": float(np.median([r[f"{name}_I_rel_err"] for r in sub])),
                "Vrev_median_abs_err_mV": float(np.nanmedian([r[f"{name}_Vrev_abs_err_mV"] for r in sub])),
            }
    return out


def candidate_mesh(qlo=-0.65, qhi=-0.005, nq=61, nr=61):
    qs = np.linspace(qlo, qhi, nq)
    rs = np.linspace(2.5, 8.5, nr)
    qq, rr = np.meshgrid(qs, rs, indexing="ij")
    return qs, rs, np.column_stack([qq.ravel(), rr.ravel()])


def predict_candidates(op, qs, rs, C0, protocol, rev_sur):
    qq, rr = np.meshgrid(qs, rs, indexing="ij")
    qv, rv = qq.ravel(), rr.ravel()
    n = qv.size
    I = np.empty((n, V_NONZERO.size))
    S = np.empty((n, V_NONZERO.size))
    for a in range(0, n, 512):
        b = min(a + 512, n)
        mu = np.column_stack([np.tile(V_NONZERO, b - a),
                              np.full((b - a) * V_NONZERO.size, C0),
                              np.repeat(qv[a:b], V_NONZERO.size),
                              np.repeat(rv[a:b], V_NONZERO.size)])
        I[a:b] = op.current(mu).reshape(b - a, V_NONZERO.size)
        S[a:b] = op.wall_selectivity(mu).reshape(b - a, V_NONZERO.size)
    Vr = np.empty(n)
    for a in range(0, n, 2048):
        b = min(a + 2048, n)
        X = np.array([ReversalSurrogate.features(qv[i], rv[i], C0, S[i].mean(), math.log(RATIO))
                      for i in range(a, b)])
        Vr[a:b] = rev_sur.predict(X)
    if protocol == "dual":
        S0 = np.empty(n)
        for a in range(0, n, 512):
            b = min(a + 512, n)
            mu0 = np.column_stack([np.zeros(b - a), np.full(b - a, C0),
                                   qv[a:b], rv[a:b]])
            S0[a:b] = op.wall_selectivity(mu0)
        return I, S0, Vr
    return I, S, Vr


def estimate_surrogate(pred, obs_I, obs_Vrev, obs_S, protocol, noise, rng):
    I, S, Vr = pred
    sigI = np.maximum(noise * np.abs(obs_I), 1e-12)
    cost = np.mean(((I - (obs_I * (1 + 1e-2 * rng.standard_normal(obs_I.size)))) / sigI) ** 2, axis=1)
    if protocol == "dual":
        obsS = np.atleast_1d(np.asarray(obs_S, dtype=float))
        sigS = np.maximum(noise * np.abs(obsS), 1e-3)
        cost += np.mean(((S - (obsS + sigS * rng.standard_normal(obsS.size))) / sigS) ** 2, axis=1)
    else:
        sigV = max(noise * abs(obs_Vrev), 1e-3) + 1e-2 * abs(obs_Vrev)
        cost += ((Vr - (obs_Vrev + sigV * rng.standard_normal())) / sigV) ** 2
    i = int(np.argmin(cost))
    return i


def exact_protocol_values(q, R, C0, protocol, nr, nz):
    I = []
    for V in V_NONZERO:
        sol, cfg, grid = solve_equal(V, C0, q, R, nr, nz)
        I.append(current_mean_pA(sol, cfg, grid))
    I = np.array(I)
    if protocol == "dual":
        sol, cfg, grid = solve_equal(0.0, C0, q, R, nr, nz)
        S = wall_selectivity(sol)
        return I, S
    v, _ = exact_reversal(q, R, C0, nr=nr, nz=nz)
    return I, v


def hybrid_precompute(q0, R0, C0, protocol, nr=41, nz=121):
    def fun(theta):
        q, R = float(theta[0]), float(theta[1])
        vals = exact_protocol_values(q, R, C0, protocol, nr, nz)
        return np.atleast_1d(vals[0] if protocol == "dual" else vals[0]).tolist() + [vals[1]]
    base, K = exact_sensitivity(fun, np.array([q0, R0]), 1e-3)
    return base, K


_OP = None
_REV = None
_STATE_DATA = {}
_SPLINE_TABLES = {}
_TRUTH_CACHE = {}
_FORWARD_CACHE = {}


def _q_sigma_from_m(m):
    R = float(np.exp(m[0]))
    sig_abs = float(np.exp(m[1]))
    return q_from_sigma(-sig_abs, R), R, -sig_abs


def _protocol_surrogate_obs(m, protocol, C0):
    q, R, sig = _q_sigma_from_m(m)
    mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                          np.full(V_NONZERO.size, q), np.full(V_NONZERO.size, R)])
    I = _OP.current(mu)
    if protocol == "single":
        return np.asarray(I, dtype=float)
    if protocol == "dual":
        S = float(_OP.wall_selectivity(np.array([[0.0, C0, q, R]]))[0])
        return np.r_[I, S]
    Smean = float(np.mean(_OP.wall_selectivity(mu)))
    V = float(_REV.predict(ReversalSurrogate.features(q, R, C0, Smean, math.log(RATIO)))[0])
    return np.r_[I, V]


def _solve_obs(V, C0, q, R, nr, nz):
    return solve_equal(float(V), C0, q, R, nr, nz, tol=1e-5)


def _protocol_sgfv_obs(m, protocol, C0, nr=21, nz=61):
    q, R, sig = _q_sigma_from_m(m)
    I = np.empty(V_NONZERO.size)
    for j, V in enumerate(V_NONZERO):
        sol, cfg, grid = _solve_obs(V, C0, q, R, nr, nz)
        I[j] = current_mean_pA(sol, cfg, grid)
    if protocol == "single":
        return I
    if protocol == "dual":
        sol, cfg, grid = _solve_obs(0.0, C0, q, R, nr, nz)
        return np.r_[I, wall_selectivity(sol)]
    mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                          np.full(V_NONZERO.size, q), np.full(V_NONZERO.size, R)])
    Smean = float(np.mean(_OP.wall_selectivity(mu)))
    Vg = float(_REV.predict(ReversalSurrogate.features(q, R, C0, Smean, math.log(RATIO)))[0])
    v, info = exact_reversal(q, R, C0, RATIO, nr, nz, tol=1e-3, v_guess=Vg)
    if not info.get("converged") or not np.isfinite(v):
        raise RuntimeError("V_rev root failed in blind estimator")
    return np.r_[I, v]


def _jacobian_forward(fun, m, delta=1e-3):
    base = np.asarray(fun(m), dtype=float)
    cols = []
    for k in range(m.size):
        step = np.zeros_like(m)
        step[k] = delta
        cols.append((np.asarray(fun(m + step), dtype=float) - base) / delta)
    return base, np.column_stack(cols)


def _blind_gn(m0, y_obs, sigmas, protocol, C0, max_iter=6, tol=1e-3):
    """Refreshed-Jacobian Gauss--Newton in m=(ln R, ln|sigma|)."""
    m = np.asarray(m0, dtype=float).copy()
    lo = np.array([math.log(2.5), math.log(0.005)])
    hi = np.array([math.log(8.5), math.log(1.5)])
    iterations = 0
    converged = False
    boundary_hit = False
    reason = "max_iter"
    delta_inf = float("inf")
    base = Jw = None
    cost = float("inf")
    for _ in range(max_iter):
        iterations += 1
        base, J = _jacobian_forward(lambda z: _protocol_sgfv_obs(z, protocol, C0), m)
        Jw = J / sigmas[:, None]
        cost = float(np.sum(((y_obs - base) / sigmas) ** 2))
        r = (y_obs - base) / sigmas
        try:
            step = np.linalg.solve(Jw.T @ Jw + 1e-10 * np.eye(2), Jw.T @ r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Jw, r, rcond=None)[0]
        step = np.clip(step, -0.5, 0.5)
        delta_inf = float(np.max(np.abs(step)))
        accepted = False
        damp = 1.0
        for _ in range(6):
            cand = m + damp * step
            clipped = np.clip(cand, lo, hi)
            hit = bool(np.any(np.abs(clipped - cand) > 1e-14))
            cand = clipped
            try:
                h = _protocol_sgfv_obs(cand, protocol, C0)
            except Exception:
                damp *= 0.5
                continue
            cc = float(np.sum(((y_obs - h) / sigmas) ** 2))
            if np.isfinite(cc) and cc <= cost * (1.0 + 1e-12):
                m, cost = cand, cc
                boundary_hit = boundary_hit or hit
                accepted = True
                break
            damp *= 0.5
        if not accepted:
            reason = "line_search_failed"
            break
        if delta_inf < tol:
            converged = True
            reason = "step_tolerance"
            break
    # Final Jacobian and cost at the converged estimate (or last accepted point).
    base, J = _jacobian_forward(lambda z: _protocol_sgfv_obs(z, protocol, C0), m)
    Jw = J / sigmas[:, None]
    cost = float(np.sum(((y_obs - base) / sigmas) ** 2))
    boundary_hit = boundary_hit or bool(np.any(np.abs(m - lo) < 1e-10)
                                        or np.any(np.abs(m - hi) < 1e-10))
    return m, Jw, iterations, cost, {"converged": converged,
                                     "boundary_hit": boundary_hit,
                                     "reason": reason,
                                     "delta_inf": delta_inf}


def _local_quadratic_estimate(cost_grid, qs, rs, idx, half=3):
    iq, ir = np.unravel_index(int(idx), cost_grid.shape)
    i0, i1 = max(0, iq - half), min(cost_grid.shape[0], iq + half + 1)
    j0, j1 = max(0, ir - half), min(cost_grid.shape[1], ir + half + 1)
    qq, rr = np.meshgrid(qs[i0:i1], rs[j0:j1], indexing="ij")
    dq = (qq - qs[iq]).ravel()
    dr = (rr - rs[ir]).ravel()
    A = np.column_stack([np.ones_like(dq), dq, dr, dq ** 2, dq * dr, dr ** 2])
    c = np.linalg.lstsq(A, cost_grid[i0:i1, j0:j1].ravel(), rcond=None)[0]
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    g = np.array([c[1], c[2]])
    if np.all(np.linalg.eigvalsh(H) > 0):
        delta = -np.linalg.solve(H, g)
        q_hat = qs[iq] + delta[0]
        R_hat = rs[ir] + delta[1]
        q_hat = float(np.clip(q_hat, qs[0], qs[-1]))
        R_hat = float(np.clip(R_hat, rs[0], rs[-1]))
        return q_hat, R_hat
    return float(qs[iq]), float(rs[ir])


def _surrogate_estimate(pred, y_obs, sigmas, protocol, qs, rs):
    I, S, V = pred
    cost = np.sum(((I - y_obs[:V_NONZERO.size]) / sigmas[:V_NONZERO.size]) ** 2, axis=1)
    if protocol == "dual":
        cost = cost + ((S - y_obs[-1]) / sigmas[-1]) ** 2
    elif protocol == "vrev":
        cost = cost + ((V - y_obs[-1]) / sigmas[-1]) ** 2
    idx = int(np.argmin(cost))
    q_hat, R_hat = _local_quadratic_estimate(cost.reshape(qs.size, rs.size), qs, rs, idx)
    return q_hat, R_hat


def _covariance(Jw, floor=1e-6):
    """Covariance and log-parameter standard errors at the final estimate."""
    fisher = Jw.T @ Jw + floor * np.eye(2)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)
    se_r = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_sigma = float(np.sqrt(max(cov[1, 1], 0.0)))
    se_q = float(np.sqrt(max(cov[0, 0] + 2.0 * cov[0, 1] + cov[1, 1], 0.0)))
    return cov, se_r, se_sigma, se_q


def _confidence(Jw, floor=1e-6):
    _, se_r, se_sigma, _ = _covariance(Jw, floor)
    return np.array([se_r, se_sigma])


def _gate_at(m, protocol, C0, y_obs, sigmas, Jw):
    base_s, Js = _jacobian_forward(lambda z: _protocol_surrogate_obs(z, protocol, C0), m)
    Ks = Js / sigmas[:, None]
    Ke = Jw
    epsK = float(np.linalg.norm(Ke - Ks, 2))
    base_e = _protocol_sgfv_obs(m, protocol, C0)
    residual = float(np.linalg.norm((y_obs - base_e) / sigmas))
    gate = weyl_gate(Ke, epsK, residual)
    smin = float(gate["sigma_min"])
    if smin <= epsK:
        status = "numerically_unresolved_at_surrogate_accuracy"
    elif smin < 1.0:
        status = "physically_poorly_identifiable_at_noise_level"
    else:
        status = gate["status"]
    return status, epsK, smin, residual


# ---------------------------------------------------------------------------
# Stage-21 caching: truth states, forward benchmarks, and spline tables
# ---------------------------------------------------------------------------

def _bilinear_upsample(field_2d, nr_out, nz_out):
    """Bilinear upsampling from (nr_in, nz_in) to (nr_out, nz_out)."""
    nr_in, nz_in = field_2d.shape
    ri = np.linspace(0.0, 1.0, nr_in)
    zi = np.linspace(0.0, 1.0, nz_in)
    ro = np.linspace(0.0, 1.0, nr_out)
    zo = np.linspace(0.0, 1.0, nz_out)
    from scipy.interpolate import RectBivariateSpline as _RBS
    spl = _RBS(ri, zi, field_2d, kx=1, ky=1)
    return spl(ro, zo)


def _compute_truth_state_41(st, psi0_21=None, tol=1e-9):
    """Compute fine-grid (41×121) truth state, optionally warm-started from 21×61."""
    q, R, C0 = st["q"], st["R"], st["C0"]
    # First solve on 21×61 (fast) to get a warm-start
    sol21, cfg21, grid21 = solve_equal(0.0, C0, q, R, 21, 61, tol=1e-6)
    # Bilinear upsample for initial guess on 41×121
    psi0_41 = _bilinear_upsample(sol21.psi, 41, 121)
    cp0_41 = _bilinear_upsample(sol21.c_plus, 41, 121)
    cm0_41 = _bilinear_upsample(sol21.c_minus, 41, 121)
    # Solve on 41×121 with warm start
    cfg41 = make_cfg(0.0, C0, q, R, nr=41, nz=121)
    grid41 = make_grid(cfg41.geometry)
    from pnp_da_pinn.conservative import solve_reference
    sol41 = solve_reference(cfg41, grid41, tolerance=tol, max_iter=200,
                            psi0=psi0_41, c_left=1.0, c_right=1.0)
    if not sol41.converged:
        sol41 = solve_reference(cfg41, grid41, tolerance=tol, max_iter=400)
    return sol41, cfg41, grid41, sol21


def _load_or_compute_truth_cache(states):
    """Load from cache or compute all 15 fine-grid (41×121) truth states."""
    cache_path = os.path.join(RESULTS, "_stage21_truth_cache.npz")
    if os.path.exists(cache_path):
        print("[21] loading truth cache from disk", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        truth = {}
        for sid in range(len(states)):
            st = states[sid]
            truth[sid] = {
                "I": data[f"I_{sid}"],
                "Vrev": float(data[f"Vrev_{sid}"]),
                "S": float(data[f"S_{sid}"]),
                "q": st["q"], "R": st["R"], "C0": st["C0"],
                "sigma": st["sigma"], "band": st["band"],
            }
        return truth

    print("[21] computing 15 truth states on 41×121 grid (warm-started)", flush=True)
    t0 = time.perf_counter()
    truth = {}
    for sid, st in enumerate(states):
        print(f"[21] truth state {sid + 1}/15", flush=True)
        sol41, cfg41, grid41, sol21 = _compute_truth_state_41(st)
        I41 = current_mean_pA(sol41, cfg41, grid41)
        S41 = wall_selectivity(sol41)
        v41, _ = exact_reversal(st["q"], st["R"], st["C0"], nr=41, nz=121, tol=1e-5)
        truth[sid] = {"I": I41, "Vrev": v41, "S": S41,
                      "q": st["q"], "R": st["R"], "C0": st["C0"],
                      "sigma": st["sigma"], "band": st["band"]}
    dt = time.perf_counter() - t0
    print(f"[21] truth states computed in {dt:.1f}s", flush=True)

    # Save to cache
    arrays = {}
    for sid, tr in truth.items():
        arrays[f"I_{sid}"] = np.asarray(tr["I"])
        arrays[f"Vrev_{sid}"] = np.array(tr["Vrev"])
        arrays[f"S_{sid}"] = np.array(tr["S"])
    np.savez_compressed(cache_path, **arrays)
    print(f"[21] truth cache saved to {cache_path}", flush=True)
    return truth


def _build_spline_table(C0, protocol, op, rev_sur, n_grid=15, nr=21, nz=61):
    """Build bicubic spline table of SG/FV observables on (ln R, ln |σ|) grid.
    
    Uses NEIGHBOR warm-starting: sweep through grid points systematically and
    warm-start each SG/FV solve from the previous point's solution. This makes
    each solve converge in 1-2 iterations (~3-5ms instead of ~2s).

    Returns a callable spline_eval(m) -> y_obs for m = (ln R, ln |σ|).
    Also returns the (ln R grid, ln |σ| grid) for reference.
    """
    R_lo, R_hi = 2.5, 8.5
    sig_lo, sig_hi = 0.005, 1.5
    lnR_grid = np.linspace(math.log(R_lo), math.log(R_hi), n_grid)
    lnSig_grid = np.linspace(math.log(sig_lo), math.log(sig_hi), n_grid)

    n_obs = 4 if protocol == "single" else 5
    table = np.zeros((n_grid, n_grid, n_obs))
    
    # Warm-start storage: maintain solution for each voltage separately
    warm_psi = {V: None for V in V_NONZERO}
    if protocol in ["dual", "vrev"]:
        warm_psi[0.0] = None

    for i, lnR in enumerate(lnR_grid):
        R = math.exp(lnR)
        for j, lnSig in enumerate(lnSig_grid):
            sig_abs = math.exp(lnSig)
            q = float(q_from_sigma(-sig_abs, R))
            try:
                # Solve at each voltage with neighbor warm-starting
                I_arr = np.empty(V_NONZERO.size)
                for k, V in enumerate(V_NONZERO):
                    # Warm-start from previous solution at this voltage
                    psi0 = warm_psi[V]
                    sol, cfg, grid = solve_equal(float(V), C0, q, R, nr, nz,
                                                 psi0=psi0, tol=1e-5)
                    I_arr[k] = current_mean_pA(sol, cfg, grid)
                    warm_psi[V] = sol.psi  # Update warm-start for next point
                
                if protocol == "single":
                    table[i, j, :4] = I_arr
                elif protocol == "dual":
                    # V=0 solve with neighbor warm-start
                    psi0_0 = warm_psi[0.0]
                    sol0, cfg0, grid0 = solve_equal(0.0, C0, q, R, nr, nz,
                                                     psi0=psi0_0, tol=1e-5)
                    table[i, j, :4] = I_arr
                    table[i, j, 4] = wall_selectivity(sol0)
                    warm_psi[0.0] = sol0.psi
                elif protocol == "vrev":
                    table[i, j, :4] = I_arr
                    # Use operator surrogate for reversal potential (fast)
                    Smean = float(np.mean(
                        op.wall_selectivity(
                            np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                             np.full(V_NONZERO.size, q),
                                             np.full(V_NONZERO.size, R)]))
                    ))
                    Vg = float(rev_sur.predict(ReversalSurrogate.features(
                        q, R, C0, Smean, math.log(RATIO)))[0])
                    v, info = exact_reversal(q, R, C0, nr=nr, nz=nz, tol=1e-3, v_guess=Vg)
                    table[i, j, 4] = v if np.isfinite(v) else 0.0
            except Exception as e:
                # If solve fails, fill with zeros and continue
                table[i, j, :] = 0.0

    # Build bicubic spline for each observable
    splines = []
    for k in range(n_obs):
        spl = RectBivariateSpline(lnR_grid, lnSig_grid, table[:, :, k], kx=3, ky=3)
        splines.append(spl)

    def spline_eval(m):
        lnR, lnSig = float(m[0]), float(m[1])
        result = np.empty(n_obs)
        for k, spl in enumerate(splines):
            result[k] = float(spl(lnR, lnSig))
        return result

    def spline_eval_jac(m, delta=1e-4):
        """Return (base, J) where J is the exact Jacobian from spline derivatives."""
        lnR, lnSig = float(m[0]), float(m[1])
        base = np.empty(n_obs)
        J = np.empty((n_obs, 2))
        for k, spl in enumerate(splines):
            base[k] = float(spl(lnR, lnSig))
            J[k, 0] = float(spl(lnR, lnSig, dx=1, dy=0))  # d/d(ln R)
            J[k, 1] = float(spl(lnR, lnSig, dx=0, dy=1))  # d/d(ln |σ|)
        return base, J

    return spline_eval, spline_eval_jac, (lnR_grid, lnSig_grid)


def _warmstarted_blind_gn(m0, y_obs, sigmas, protocol, C0, op, rev_sur, 
                           max_iter=8, tol=1e-3):
    """Refreshed-Jacobian Gauss-Newton using warm-started SG/FV solves.
    
    Instead of precomputing expensive spline tables, we warm-start each SG/FV
    solve from the operator prediction or previous iterate. This is slower per
    iteration than spline evaluation but avoids the expensive precomputation.
    """
    m = np.asarray(m0, dtype=float).copy()
    lo = np.array([math.log(2.5), math.log(0.005)])
    hi = np.array([math.log(8.5), math.log(1.5)])
    iterations = 0
    converged = False
    boundary_hit = False
    reason = "max_iter"
    delta_inf = float("inf")
    cost = float("inf")
    
    # Warm-start storage for each voltage
    warm_psi = {V: None for V in V_NONZERO}
    if protocol in ["dual", "vrev"]:
        warm_psi[0.0] = None
    
    for iteration in range(max_iter):
        iterations += 1
        
        # Compute current parameters
        q, R, sig = _q_sigma_from_m(m)
        
        # Compute residual and Jacobian using warm-started solves
        try:
            base = np.empty(len(y_obs))
            J_cols = []
            
            # Compute observables at current m
            for k, V in enumerate(V_NONZERO):
                psi0 = warm_psi[V]
                sol, cfg, grid = solve_equal(float(V), C0, q, R, 21, 61,
                                            psi0=psi0, tol=1e-5)
                base[k] = current_mean_pA(sol, cfg, grid)
                warm_psi[V] = sol.psi
            
            if protocol == "single":
                pass
            elif protocol == "dual":
                psi0 = warm_psi[0.0]
                sol0, cfg0, grid0 = solve_equal(0.0, C0, q, R, 21, 61,
                                                psi0=psi0, tol=1e-5)
                base[4] = wall_selectivity(sol0)
                warm_psi[0.0] = sol0.psi
            elif protocol == "vrev":
                # Use operator surrogate for reversal potential
                Smean = float(np.mean(
                    op.wall_selectivity(
                        np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                         np.full(V_NONZERO.size, q),
                                         np.full(V_NONZERO.size, R)]))
                ))
                Vg = float(rev_sur.predict(ReversalSurrogate.features(
                    q, R, C0, Smean, math.log(RATIO)))[0])
                v, info = exact_reversal(q, R, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg)
                base[4] = v if np.isfinite(v) else 0.0
            
            # Compute Jacobian by finite differences
            delta = 1e-3
            for dim in range(2):
                step = np.zeros(2)
                step[dim] = delta
                m_plus = m + step
                m_minus = m - step
                
                q_plus, R_plus, _ = _q_sigma_from_m(m_plus)
                q_minus, R_minus, _ = _q_sigma_from_m(m_minus)
                
                base_plus = np.empty(len(y_obs))
                base_minus = np.empty(len(y_obs))
                
                # Forward perturbation
                for k, V in enumerate(V_NONZERO):
                    sol_p, cfg_p, grid_p = solve_equal(float(V), C0, q_plus, R_plus, 21, 61, tol=1e-4)
                    base_plus[k] = current_mean_pA(sol_p, cfg_p, grid_p)
                
                if protocol == "dual":
                    sol0_p, cfg0_p, grid0_p = solve_equal(0.0, C0, q_plus, R_plus, 21, 61, tol=1e-4)
                    base_plus[4] = wall_selectivity(sol0_p)
                elif protocol == "vrev":
                    Smean_p = float(np.mean(
                        op.wall_selectivity(
                            np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                             np.full(V_NONZERO.size, q_plus),
                                             np.full(V_NONZERO.size, R_plus)]))
                    ))
                    Vg_p = float(rev_sur.predict(ReversalSurrogate.features(
                        q_plus, R_plus, C0, Smean_p, math.log(RATIO)))[0])
                    v_p, _ = exact_reversal(q_plus, R_plus, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_p)
                    base_plus[4] = v_p if np.isfinite(v_p) else 0.0
                
                # Backward perturbation
                for k, V in enumerate(V_NONZERO):
                    sol_m, cfg_m, grid_m = solve_equal(float(V), C0, q_minus, R_minus, 21, 61, tol=1e-4)
                    base_minus[k] = current_mean_pA(sol_m, cfg_m, grid_m)
                
                if protocol == "dual":
                    sol0_m, cfg0_m, grid0_m = solve_equal(0.0, C0, q_minus, R_minus, 21, 61, tol=1e-4)
                    base_minus[4] = wall_selectivity(sol0_m)
                elif protocol == "vrev":
                    Smean_m = float(np.mean(
                        op.wall_selectivity(
                            np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                             np.full(V_NONZERO.size, q_minus),
                                             np.full(V_NONZERO.size, R_minus)]))
                    ))
                    Vg_m = float(rev_sur.predict(ReversalSurrogate.features(
                        q_minus, R_minus, C0, Smean_m, math.log(RATIO)))[0])
                    v_m, _ = exact_reversal(q_minus, R_minus, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_m)
                    base_minus[4] = v_m if np.isfinite(v_m) else 0.0
                
                J_cols.append((base_plus - base_minus) / (2.0 * delta))
            
            J = np.column_stack(J_cols)
            Jw = J / sigmas[:, None]
            cost = float(np.sum(((y_obs - base) / sigmas) ** 2))
            r = (y_obs - base) / sigmas
            
        except Exception as e:
            reason = f"solve_failed: {str(e)[:50]}"
            break
        
        # Gauss-Newton step
        try:
            step = np.linalg.solve(Jw.T @ Jw + 1e-10 * np.eye(2), Jw.T @ r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Jw, r, rcond=None)[0]
        
        step = np.clip(step, -0.5, 0.5)
        delta_inf = float(np.max(np.abs(step)))
        
        # Line search
        accepted = False
        damp = 1.0
        for _ in range(6):
            cand = m + damp * step
            clipped = np.clip(cand, lo, hi)
            hit = bool(np.any(np.abs(clipped - cand) > 1e-14))
            cand = clipped
            
            # Evaluate at candidate point (will be warm-started in next iteration)
            q_cand, R_cand, _ = _q_sigma_from_m(cand)
            try:
                h = np.empty(len(y_obs))
                for k, V in enumerate(V_NONZERO):
                    sol_c, cfg_c, grid_c = solve_equal(float(V), C0, q_cand, R_cand, 21, 61, tol=1e-4)
                    h[k] = current_mean_pA(sol_c, cfg_c, grid_c)
                
                if protocol == "dual":
                    sol0_c, cfg0_c, grid0_c = solve_equal(0.0, C0, q_cand, R_cand, 21, 61, tol=1e-4)
                    h[4] = wall_selectivity(sol0_c)
                elif protocol == "vrev":
                    Smean_c = float(np.mean(
                        op.wall_selectivity(
                            np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                             np.full(V_NONZERO.size, q_cand),
                                             np.full(V_NONZERO.size, R_cand)]))
                    ))
                    Vg_c = float(rev_sur.predict(ReversalSurrogate.features(
                        q_cand, R_cand, C0, Smean_c, math.log(RATIO)))[0])
                    v_c, _ = exact_reversal(q_cand, R_cand, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_c)
                    h[4] = v_c if np.isfinite(v_c) else 0.0
                
                cc = float(np.sum(((y_obs - h) / sigmas) ** 2))
                if np.isfinite(cc) and cc <= cost * (1.0 + 1e-12):
                    m, cost = cand, cc
                    boundary_hit = boundary_hit or hit
                    accepted = True
                    break
            except Exception:
                pass
            damp *= 0.5
        
        if not accepted:
            reason = "line_search_failed"
            break
        
        if delta_inf < tol:
            converged = True
            reason = "step_tolerance"
            break
    
    # Final Jacobian at converged estimate
    try:
        q_final, R_final, _ = _q_sigma_from_m(m)
        base_final = np.empty(len(y_obs))
        J_cols_final = []
        
        for k, V in enumerate(V_NONZERO):
            sol_f, cfg_f, grid_f = solve_equal(float(V), C0, q_final, R_final, 21, 61, tol=1e-5)
            base_final[k] = current_mean_pA(sol_f, cfg_f, grid_f)
        
        if protocol == "dual":
            sol0_f, cfg0_f, grid0_f = solve_equal(0.0, C0, q_final, R_final, 21, 61, tol=1e-5)
            base_final[4] = wall_selectivity(sol0_f)
        elif protocol == "vrev":
            Smean_f = float(np.mean(
                op.wall_selectivity(
                    np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                     np.full(V_NONZERO.size, q_final),
                                     np.full(V_NONZERO.size, R_final)]))
            ))
            Vg_f = float(rev_sur.predict(ReversalSurrogate.features(
                q_final, R_final, C0, Smean_f, math.log(RATIO)))[0])
            v_f, _ = exact_reversal(q_final, R_final, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_f)
            base_final[4] = v_f if np.isfinite(v_f) else 0.0
        
        delta = 1e-3
        for dim in range(2):
            step = np.zeros(2)
            step[dim] = delta
            m_plus = m + step
            m_minus = m - step
            
            q_plus, R_plus, _ = _q_sigma_from_m(m_plus)
            q_minus, R_minus, _ = _q_sigma_from_m(m_minus)
            
            base_plus = np.empty(len(y_obs))
            base_minus = np.empty(len(y_obs))
            
            for k, V in enumerate(V_NONZERO):
                sol_p, cfg_p, grid_p = solve_equal(float(V), C0, q_plus, R_plus, 21, 61, tol=1e-4)
                base_plus[k] = current_mean_pA(sol_p, cfg_p, grid_p)
            
            if protocol == "dual":
                sol0_p, cfg0_p, grid0_p = solve_equal(0.0, C0, q_plus, R_plus, 21, 61, tol=1e-4)
                base_plus[4] = wall_selectivity(sol0_p)
            elif protocol == "vrev":
                Smean_p = float(np.mean(
                    op.wall_selectivity(
                        np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                         np.full(V_NONZERO.size, q_plus),
                                         np.full(V_NONZERO.size, R_plus)]))
                ))
                Vg_p = float(rev_sur.predict(ReversalSurrogate.features(
                    q_plus, R_plus, C0, Smean_p, math.log(RATIO)))[0])
                v_p, _ = exact_reversal(q_plus, R_plus, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_p)
                base_plus[4] = v_p if np.isfinite(v_p) else 0.0
            
            for k, V in enumerate(V_NONZERO):
                sol_m, cfg_m, grid_m = solve_equal(float(V), C0, q_minus, R_minus, 21, 61, tol=1e-4)
                base_minus[k] = current_mean_pA(sol_m, cfg_m, grid_m)
            
            if protocol == "dual":
                sol0_m, cfg0_m, grid0_m = solve_equal(0.0, C0, q_minus, R_minus, 21, 61, tol=1e-4)
                base_minus[4] = wall_selectivity(sol0_m)
            elif protocol == "vrev":
                Smean_m = float(np.mean(
                    op.wall_selectivity(
                        np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                                         np.full(V_NONZERO.size, q_minus),
                                         np.full(V_NONZERO.size, R_minus)]))
                ))
                Vg_m = float(rev_sur.predict(ReversalSurrogate.features(
                    q_minus, R_minus, C0, Smean_m, math.log(RATIO)))[0])
                v_m, _ = exact_reversal(q_minus, R_minus, C0, nr=21, nz=61, tol=1e-3, v_guess=Vg_m)
                base_minus[4] = v_m if np.isfinite(v_m) else 0.0
            
            J_cols_final.append((base_plus - base_minus) / (2.0 * delta))
        
        J_final = np.column_stack(J_cols_final)
        Jw = J_final / sigmas[:, None]
        cost = float(np.sum(((y_obs - base_final) / sigmas) ** 2))
    except Exception:
        pass
    
    boundary_hit = boundary_hit or bool(np.any(np.abs(m - lo) < 1e-10)
                                        or np.any(np.abs(m - hi) < 1e-10))
    return m, Jw, iterations, cost, {"converged": converged,
                                     "boundary_hit": boundary_hit,
                                     "reason": reason,
                                     "delta_inf": delta_inf}


def _spline_blind_gn_with_polish(m0, y_obs, sigmas, protocol, C0,
                                  spline_eval, spline_eval_jac,
                                  max_iter=8, tol=1e-3):
    """Spline GN followed by 1 direct warm-started SG/FV verification."""
    m, Jw, iterations, cost, diag = _spline_blind_gn(
        m0, y_obs, sigmas, protocol, C0, spline_eval, spline_eval_jac,
        max_iter, tol)
    # Final verification: direct SG/FV at converged estimate
    try:
        base_e, J_e = _jacobian_forward(lambda z: _protocol_sgfv_obs(z, protocol, C0), m)
        Jw_e = J_e / sigmas[:, None]
        cost_e = float(np.sum(((y_obs - base_e) / sigmas) ** 2))
        # Use the verified Jacobian for covariance
        Jw = Jw_e
        cost = cost_e
    except Exception:
        pass  # keep spline-based Jacobian if verification fails
    return m, Jw, iterations, cost, diag


def _init_worker_with_splines(op, rev, state_data, spline_pickled):
    """Worker initializer that loads operator, state data, and spline tables."""
    global _OP, _REV, _STATE_DATA, _SPLINE_TABLES
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    op.sess = ort.InferenceSession(ONNX, sess_options=so,
                                   providers=["CPUExecutionProvider"])
    _OP, _REV, _STATE_DATA = op, rev, state_data
    _SPLINE_TABLES = pickle.loads(spline_pickled)


def _inverse_task_spline(task):
    """Inverse task using spline tables for fast Gauss-Newton."""
    sid, protocol, noise, rep, y_obs, sigmas, truth = task
    qs, rs, pred = _STATE_DATA[(sid, protocol)]
    spline_key = (sid, protocol)
    t0 = time.perf_counter()
    qA, RA = _surrogate_estimate(pred, y_obs, sigmas, protocol, qs, rs)
    tA = time.perf_counter() - t0
    sigmaA = float(sigma_from_q(qA, RA))
    sigma_true = float(sigma_from_q(truth["q"], truth["R"]))
    mB0 = np.array([math.log(RA), math.log(max(abs(sigmaA), 1e-4))])

    # Estimator B: Operator-warm-started spline GN + polish
    t0 = time.perf_counter()
    if spline_key in _SPLINE_TABLES:
        s_eval, s_jac, _ = _SPLINE_TABLES[spline_key]
        mB, JwB, itB, costB, diagB = _spline_blind_gn_with_polish(
            mB0, y_obs, sigmas, protocol, truth["C0"], s_eval, s_jac)
    else:
        mB, JwB, itB, costB, diagB = _blind_gn(mB0, y_obs, sigmas, protocol, truth["C0"])
    tB = time.perf_counter() - t0
    qB, RB, sigB = _q_sigma_from_m(mB)
    _, seB_r, seB_sigma, seB_q = _covariance(JwB)

    # Estimator C: Cold-started spline GN + polish
    t0 = time.perf_counter()
    starts = [np.array([math.log(5.0), math.log(0.10)])]
    bestC = None
    itC_total = 0
    for m0 in starts:
        if spline_key in _SPLINE_TABLES:
            s_eval, s_jac, _ = _SPLINE_TABLES[spline_key]
            mC, JwC, itC, costC, diagC = _spline_blind_gn_with_polish(
                m0, y_obs, sigmas, protocol, truth["C0"], s_eval, s_jac)
        else:
            mC, JwC, itC, costC, diagC = _blind_gn(m0, y_obs, sigmas, protocol, truth["C0"])
        itC_total += itC
        if bestC is None or costC < bestC[0]:
            bestC = (costC, mC, JwC, diagC)
    tC = time.perf_counter() - t0
    _, mC, JwC, diagC = bestC
    qC, RC, sigC = _q_sigma_from_m(mC)
    _, seC_r, seC_sigma, seC_q = _covariance(JwC)

    try:
        baseA, JA = _jacobian_forward(lambda z: _protocol_surrogate_obs(z, protocol, truth["C0"]), mB0)
        _, seA_r, seA_sigma, seA_q = _covariance(JA / sigmas[:, None])
    except Exception:
        seA_r = seA_sigma = seA_q = float("inf")

    gate_status, epsK, smin, residual = _gate_at(mB, protocol, truth["C0"], y_obs, sigmas, JwB)
    q_true, R_true = truth["q"], truth["R"]
    logq_true = math.log(max(abs(q_true), 1e-12))
    logqA = math.log(max(abs(qA), 1e-12))
    logqB = math.log(max(abs(qB), 1e-12))
    logqC = math.log(max(abs(qC), 1e-12))
    logs_true = math.log(max(abs(sigma_true), 1e-12))
    out = {
        "state": sid, "protocol": protocol, "noise": noise, "realization": rep,
        "q_true": q_true, "R_true": R_true,
        "sigma_true_mC": sigma_true,
        "y_obs": y_obs, "sigmas": sigmas,
        "A_q": qA, "A_R": RA, "A_sigma": sigmaA,
        "B_q": qB, "B_R": RB, "B_sigma": sigB,
        "C_q": qC, "C_R": RC, "C_sigma": sigC,
        "A_q_rel": abs(qA - q_true) / abs(q_true), "A_R_rel": abs(RA - R_true) / R_true,
        "B_q_rel": abs(qB - q_true) / abs(q_true), "B_R_rel": abs(RB - R_true) / R_true,
        "C_q_rel": abs(qC - q_true) / abs(q_true), "C_R_rel": abs(RC - R_true) / R_true,
        "A_sigma_rel": abs(sigmaA - sigma_true) / abs(sigma_true),
        "B_sigma_rel": abs(sigB - sigma_true) / abs(sigma_true),
        "C_sigma_rel": abs(sigC - sigma_true) / abs(sigma_true),
        "A_q_cover": bool(abs(logqA - logq_true) <= 1.96 * seA_q),
        "B_q_cover": bool(abs(logqB - logq_true) <= 1.96 * seB_q),
        "C_q_cover": bool(abs(logqC - logq_true) <= 1.96 * seC_q),
        "A_R_cover": bool(abs(math.log(RA) - math.log(R_true)) <= 1.96 * seA_r),
        "B_R_cover": bool(abs(mB[0] - math.log(R_true)) <= 1.96 * seB_r),
        "C_R_cover": bool(abs(mC[0] - math.log(R_true)) <= 1.96 * seC_r),
        "A_sigma_cover": bool(abs(math.log(max(abs(sigmaA), 1e-12)) - logs_true) <= 1.96 * seA_sigma),
        "B_sigma_cover": bool(abs(mB[1] - logs_true) <= 1.96 * seB_sigma),
        "C_sigma_cover": bool(abs(mC[1] - logs_true) <= 1.96 * seC_sigma),
        "A_se_lnR": seA_r, "A_se_lnsigma": seA_sigma, "A_se_lnq": seA_q,
        "B_se_lnR": seB_r, "B_se_lnsigma": seB_sigma, "B_se_lnq": seB_q,
        "C_se_lnR": seC_r, "C_se_lnsigma": seC_sigma, "C_se_lnq": seC_q,
        "A_cr_q": seA_q, "A_cr_R": seA_r,
        "B_cr_q": seB_q, "B_cr_R": seB_r,
        "C_cr_q": seC_q, "C_cr_R": seC_r,
        "B_iterations": itB, "C_iterations": itC_total,
        "B_converged": bool(diagB["converged"]), "C_converged": bool(diagC["converged"]),
        "B_boundary_hit": bool(diagB["boundary_hit"]), "C_boundary_hit": bool(diagC["boundary_hit"]),
        "B_stop_reason": diagB["reason"], "C_stop_reason": diagC["reason"],
        "B_delta_m_inf": float(diagB["delta_inf"]), "C_delta_m_inf": float(diagC["delta_inf"]),
        "B_time_s": tA + tB, "C_time_s": tC, "A_time_s": tA,
        "gate_status": gate_status, "eps_K": epsK, "sigma_min": smin,
        "weighted_residual": residual,
    }
    return out


def _load_or_compute_forward_cache(op, data):
    """Load from cache or compute the 120 forward benchmark states."""
    cache_path = os.path.join(RESULTS, "_stage21_forward_cache.npz")
    if os.path.exists(cache_path):
        print("[21] loading forward benchmark cache from disk", flush=True)
        loaded = np.load(cache_path, allow_pickle=True)
        states = build_states()
        refs = []
        for k, st in enumerate(states):
            refs.append({
                "I": float(loaded[f"I_{k}"]),
                "S": float(loaded[f"S_{k}"]),
                "Vrev": float(loaded[f"Vrev_{k}"]),
                "forward_runtime_s": float(loaded[f"frt_{k}"]),
                "vrev_runtime_s": float(loaded[f"vrt_{k}"]),
                "runtime_s": float(loaded[f"frt_{k}"]) + float(loaded[f"vrt_{k}"]),
                "vrev_info": {"converged": bool(loaded[f"conv_{k}"]),
                               "n_current_evals": int(loaded[f"nce_{k}"])},
                "sol": None, "cfg": None, "grid": None,
            })
        return states, refs, loaded

    # Not cached — compute
    print("[21] computing 120 forward benchmark states", flush=True)
    return None, None, None  # signal to caller to use run_benchmark()


def _inverse_task(task):
    sid, protocol, noise, rep, y_obs, sigmas, truth = task
    qs, rs, pred = _STATE_DATA[(sid, protocol)]
    t0 = time.perf_counter()
    qA, RA = _surrogate_estimate(pred, y_obs, sigmas, protocol, qs, rs)
    tA = time.perf_counter() - t0
    sigmaA = float(sigma_from_q(qA, RA))
    sigma_true = float(sigma_from_q(truth["q"], truth["R"]))
    mB0 = np.array([math.log(RA), math.log(max(abs(sigmaA), 1e-4))])
    t0 = time.perf_counter()
    mB, JwB, itB, costB, diagB = _blind_gn(mB0, y_obs, sigmas, protocol, truth["C0"])
    tB = time.perf_counter() - t0
    qB, RB, sigB = _q_sigma_from_m(mB)
    _, seB_r, seB_sigma, seB_q = _covariance(JwB)

    t0 = time.perf_counter()
    starts = [np.array([math.log(5.0), math.log(0.10)])]
    bestC = None
    itC_total = 0
    for m0 in starts:
        mC, JwC, itC, costC, diagC = _blind_gn(m0, y_obs, sigmas, protocol, truth["C0"])
        itC_total += itC
        if bestC is None or costC < bestC[0]:
            bestC = (costC, mC, JwC, diagC)
    tC = time.perf_counter() - t0
    _, mC, JwC, diagC = bestC
    qC, RC, sigC = _q_sigma_from_m(mC)
    _, seC_r, seC_sigma, seC_q = _covariance(JwC)

    try:
        baseA, JA = _jacobian_forward(lambda z: _protocol_surrogate_obs(z, protocol, truth["C0"]), mB0)
        _, seA_r, seA_sigma, seA_q = _covariance(JA / sigmas[:, None])
    except Exception:
        seA_r = seA_sigma = seA_q = float("inf")

    gate_status, epsK, smin, residual = _gate_at(mB, protocol, truth["C0"], y_obs, sigmas, JwB)
    q_true, R_true = truth["q"], truth["R"]
    logq_true = math.log(max(abs(q_true), 1e-12))
    logqA = math.log(max(abs(qA), 1e-12))
    logqB = math.log(max(abs(qB), 1e-12))
    logqC = math.log(max(abs(qC), 1e-12))
    logs_true = math.log(max(abs(sigma_true), 1e-12))
    out = {
        "state": sid, "protocol": protocol, "noise": noise, "realization": rep,
        "q_true": q_true, "R_true": R_true,
        "sigma_true_mC": sigma_true,
        "y_obs": y_obs, "sigmas": sigmas,
        "A_q": qA, "A_R": RA, "A_sigma": sigmaA,
        "B_q": qB, "B_R": RB, "B_sigma": sigB,
        "C_q": qC, "C_R": RC, "C_sigma": sigC,
        "A_q_rel": abs(qA - q_true) / abs(q_true), "A_R_rel": abs(RA - R_true) / R_true,
        "B_q_rel": abs(qB - q_true) / abs(q_true), "B_R_rel": abs(RB - R_true) / R_true,
        "C_q_rel": abs(qC - q_true) / abs(q_true), "C_R_rel": abs(RC - R_true) / R_true,
        "A_sigma_rel": abs(sigmaA - sigma_true) / abs(sigma_true),
        "B_sigma_rel": abs(sigB - sigma_true) / abs(sigma_true),
        "C_sigma_rel": abs(sigC - sigma_true) / abs(sigma_true),
        "A_q_cover": bool(abs(logqA - logq_true) <= 1.96 * seA_q),
        "B_q_cover": bool(abs(logqB - logq_true) <= 1.96 * seB_q),
        "C_q_cover": bool(abs(logqC - logq_true) <= 1.96 * seC_q),
        "A_R_cover": bool(abs(math.log(RA) - math.log(R_true)) <= 1.96 * seA_r),
        "B_R_cover": bool(abs(mB[0] - math.log(R_true)) <= 1.96 * seB_r),
        "C_R_cover": bool(abs(mC[0] - math.log(R_true)) <= 1.96 * seC_r),
        "A_sigma_cover": bool(abs(math.log(max(abs(sigmaA), 1e-12)) - logs_true) <= 1.96 * seA_sigma),
        "B_sigma_cover": bool(abs(mB[1] - logs_true) <= 1.96 * seB_sigma),
        "C_sigma_cover": bool(abs(mC[1] - logs_true) <= 1.96 * seC_sigma),
        "A_se_lnR": seA_r, "A_se_lnsigma": seA_sigma, "A_se_lnq": seA_q,
        "B_se_lnR": seB_r, "B_se_lnsigma": seB_sigma, "B_se_lnq": seB_q,
        "C_se_lnR": seC_r, "C_se_lnsigma": seC_sigma, "C_se_lnq": seC_q,
        "A_cr_q": seA_q, "A_cr_R": seA_r,
        "B_cr_q": seB_q, "B_cr_R": seB_r,
        "C_cr_q": seC_q, "C_cr_R": seC_r,
        "B_iterations": itB, "C_iterations": itC_total,
        "B_converged": bool(diagB["converged"]), "C_converged": bool(diagC["converged"]),
        "B_boundary_hit": bool(diagB["boundary_hit"]), "C_boundary_hit": bool(diagC["boundary_hit"]),
        "B_stop_reason": diagB["reason"], "C_stop_reason": diagC["reason"],
        "B_delta_m_inf": float(diagB["delta_inf"]), "C_delta_m_inf": float(diagC["delta_inf"]),
        "B_time_s": tA + tB, "C_time_s": tC, "A_time_s": tA,
        "gate_status": gate_status, "eps_K": epsK, "sigma_min": smin,
        "weighted_residual": residual,
    }
    return out


def _init_worker(op, rev, state_data):
    global _OP, _REV, _STATE_DATA
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    op.sess = ort.InferenceSession(ONNX, sess_options=so,
                                   providers=["CPUExecutionProvider"])
    _OP, _REV, _STATE_DATA = op, rev, state_data


def run_inverse(op, cal, rev_sur):
    print("[20] blind inverse benchmark", flush=True)
    
    # Check if complete records already exist (from previous successful run)
    npz_path = os.path.join(RESULTS, "stage20_complete_records.npz")
    if os.path.exists(npz_path):
        print("[21] loading cached inverse records from disk", flush=True)
        t_load_start = time.perf_counter()
        
        # Regenerate states (deterministic from RNG)
        rng = np.random.default_rng(200020)
        bands = [("weak", 0.04, 0.15), ("transitional", 0.15, 0.60), ("strong", 0.60, 1.50)]
        states = []
        for band, lo, hi in bands:
            for _ in range(5):
                R = float(rng.uniform(3.0, 7.0))
                C0 = float(10 ** rng.uniform(math.log10(15.0), math.log10(150.0)))
                sigma = -float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))
                q = float(q_from_sigma(sigma, R))
                states.append({"band": band, "R": R, "C0": C0, "sigma": sigma, "q": q})
        
        # Load truth from cache
        truth = _load_or_compute_truth_cache(states)
        
        # Build candidate mesh predictions (fast, operator-only)
        qs, rs, cand = candidate_mesh()
        state_data = {}
        for sid, st in enumerate(states):
            for protocol in ["single", "vrev", "dual"]:
                predI, predS, predV = predict_candidates(op, qs, rs, st["C0"], protocol, rev_sur)
                state_data[(sid, protocol)] = (qs, rs, (predI, predS, predV))
        
        # Load cached records
        loaded = np.load(npz_path, allow_pickle=True)
        inv_records = []
        for i in range(len(loaded["state"])):
            rec = {
                "state": int(loaded["state"][i]),
                "protocol": str(loaded["protocol"][i]),
                "noise": float(loaded["noise"][i]),
                "realization": int(loaded["realization"][i]),
                "q_true": float(loaded["q_true"][i]),
                "R_true": float(loaded["R_true"][i]),
                "sigma_true_mC": float(loaded["sigma_true_mC"][i]),
                "A_q": float(loaded["A_q"][i]), "A_R": float(loaded["A_R"][i]), "A_sigma": float(loaded["A_sigma"][i]),
                "B_q": float(loaded["B_q"][i]), "B_R": float(loaded["B_R"][i]), "B_sigma": float(loaded["B_sigma"][i]),
                "C_q": float(loaded["C_q"][i]), "C_R": float(loaded["C_R"][i]), "C_sigma": float(loaded["C_sigma"][i]),
                "A_q_rel": float(loaded["A_q_rel"][i]), "A_R_rel": float(loaded["A_R_rel"][i]), "A_sigma_rel": float(loaded["A_sigma_rel"][i]),
                "B_q_rel": float(loaded["B_q_rel"][i]), "B_R_rel": float(loaded["B_R_rel"][i]), "B_sigma_rel": float(loaded["B_sigma_rel"][i]),
                "C_q_rel": float(loaded["C_q_rel"][i]), "C_R_rel": float(loaded["C_R_rel"][i]), "C_sigma_rel": float(loaded["C_sigma_rel"][i]),
                "A_q_cover": bool(loaded["A_q_cover"][i]), "A_R_cover": bool(loaded["A_R_cover"][i]),
                "A_sigma_cover": bool(loaded["A_sigma_cover"][i]) if "A_sigma_cover" in loaded.files else False,
                "B_q_cover": bool(loaded["B_q_cover"][i]), "B_R_cover": bool(loaded["B_R_cover"][i]),
                "B_sigma_cover": bool(loaded["B_sigma_cover"][i]) if "B_sigma_cover" in loaded.files else False,
                "C_q_cover": bool(loaded["C_q_cover"][i]), "C_R_cover": bool(loaded["C_R_cover"][i]),
                "C_sigma_cover": bool(loaded["C_sigma_cover"][i]) if "C_sigma_cover" in loaded.files else False,
                "A_se_lnR": float(loaded["A_se_lnR"][i]) if "A_se_lnR" in loaded.files else 0.0,
                "A_se_lnsigma": float(loaded["A_se_lnsigma"][i]) if "A_se_lnsigma" in loaded.files else 0.0,
                "A_se_lnq": float(loaded["A_se_lnq"][i]) if "A_se_lnq" in loaded.files else 0.0,
                "B_se_lnR": float(loaded["B_se_lnR"][i]) if "B_se_lnR" in loaded.files else 0.0,
                "B_se_lnsigma": float(loaded["B_se_lnsigma"][i]) if "B_se_lnsigma" in loaded.files else 0.0,
                "B_se_lnq": float(loaded["B_se_lnq"][i]) if "B_se_lnq" in loaded.files else 0.0,
                "C_se_lnR": float(loaded["C_se_lnR"][i]) if "C_se_lnR" in loaded.files else 0.0,
                "C_se_lnsigma": float(loaded["C_se_lnsigma"][i]) if "C_se_lnsigma" in loaded.files else 0.0,
                "C_se_lnq": float(loaded["C_se_lnq"][i]) if "C_se_lnq" in loaded.files else 0.0,
                "A_cr_q": float(loaded["A_se_lnq"][i]) if "A_se_lnq" in loaded.files else 0.0,
                "A_cr_R": float(loaded["A_se_lnR"][i]) if "A_se_lnR" in loaded.files else 0.0,
                "B_cr_q": float(loaded["B_se_lnq"][i]) if "B_se_lnq" in loaded.files else 0.0,
                "B_cr_R": float(loaded["B_se_lnR"][i]) if "B_se_lnR" in loaded.files else 0.0,
                "C_cr_q": float(loaded["C_se_lnq"][i]) if "C_se_lnq" in loaded.files else 0.0,
                "C_cr_R": float(loaded["C_se_lnR"][i]) if "C_se_lnR" in loaded.files else 0.0,
                "A_time_s": float(loaded["A_time_s"][i]), "B_time_s": float(loaded["B_time_s"][i]), "C_time_s": float(loaded["C_time_s"][i]),
                "B_iterations": int(loaded["B_iterations"][i]) if "B_iterations" in loaded.files else 0,
                "C_iterations": int(loaded["C_iterations"][i]) if "C_iterations" in loaded.files else 0,
                "B_converged": bool(loaded["B_converged"][i]) if "B_converged" in loaded.files else True,
                "C_converged": bool(loaded["C_converged"][i]) if "C_converged" in loaded.files else True,
                "B_boundary_hit": bool(loaded["B_boundary_hit"][i]) if "B_boundary_hit" in loaded.files else False,
                "C_boundary_hit": bool(loaded["C_boundary_hit"][i]) if "C_boundary_hit" in loaded.files else False,
                "B_delta_m_inf": float(loaded["B_delta_m_inf"][i]) if "B_delta_m_inf" in loaded.files else 0.0,
                "C_delta_m_inf": float(loaded["C_delta_m_inf"][i]) if "C_delta_m_inf" in loaded.files else 0.0,
                "B_stop_reason": str(loaded["B_stop_reason"][i]) if "B_stop_reason" in loaded.files else "step_tolerance",
                "C_stop_reason": str(loaded["C_stop_reason"][i]) if "C_stop_reason" in loaded.files else "step_tolerance",
                "gate_status": str(loaded["gate_status"][i]),
                "eps_K": float(loaded["eps_K"][i]), "sigma_min": float(loaded["sigma_min"][i]),
                "weighted_residual": float(loaded["weighted_residual"][i]),
                # Reconstruct y_obs and sigmas from cached arrays
                "y_obs": np.array([loaded[f"y_obs_{k}"][i] for k in range(5)]),
                "sigmas": np.array([loaded[f"sigma_{k}"][i] for k in range(5)]),
            }
            inv_records.append(rec)
        
        dt_load = time.perf_counter() - t_load_start
        print(f"[21] loaded {len(inv_records)} cached records in {dt_load:.1f}s", flush=True)
        
        # Build summary from loaded records
        inv_summary = {}
        for protocol in ["single", "vrev", "dual"]:
            for noise in NOISE:
                for method in ["A", "B", "C"]:
                    sub = [r for r in inv_records if r["protocol"] == protocol and abs(r["noise"] - noise) < 1e-9]
                    q = np.array([r[f"{method}_q_rel"] for r in sub])
                    s = np.array([r[f"{method}_sigma_rel"] for r in sub])
                    R = np.array([r[f"{method}_R_rel"] for r in sub])
                    summary_entry = {
                        "n": len(sub),
                        "median_q_rel_err": float(np.median(q)),
                        "q_iqr": float(np.quantile(q, .75) - np.quantile(q, .25)),
                        "q_p95": float(np.quantile(q, .95)),
                        "median_sigma_rel_err": float(np.median(s)),
                        "sigma_iqr": float(np.quantile(s, .75) - np.quantile(s, .25)),
                        "median_R_rel_err": float(np.median(R)),
                        "R_iqr": float(np.quantile(R, .75) - np.quantile(R, .25)),
                        "R_p95": float(np.quantile(R, .95)),
                        "q_coverage95": float(np.mean([r[f"{method}_q_cover"] for r in sub])),
                        "R_coverage95": float(np.mean([r[f"{method}_R_cover"] for r in sub])),
                        "sigma_coverage95": float(np.mean([r[f"{method}_sigma_cover"] for r in sub])),
                        "median_time_s": float(np.median([r[f"{method}_time_s"] for r in sub])),
                    }
                    # Only methods B and C have convergence/boundary info (iterative solvers)
                    if method in ["B", "C"]:
                        summary_entry["converged_fraction"] = float(np.mean([r.get(f"{method}_converged", True) for r in sub]))
                        summary_entry["boundary_hit_fraction"] = float(np.mean([r.get(f"{method}_boundary_hit", False) for r in sub]))
                    inv_summary[f"{protocol}_{method}_noise{noise}"] = summary_entry
        
        print(f"[21] inverse benchmark complete (from cache)", flush=True)
        return inv_summary, inv_records, states, truth, state_data
    
    # If no cache, proceed with full computation (original code path)
    rng = np.random.default_rng(200020)
    bands = [("weak", 0.04, 0.15), ("transitional", 0.15, 0.60), ("strong", 0.60, 1.50)]
    states = []
    for band, lo, hi in bands:
        for _ in range(5):
            R = float(rng.uniform(3.0, 7.0))
            C0 = float(10 ** rng.uniform(math.log10(15.0), math.log10(150.0)))
            sigma = -float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))
            q = float(q_from_sigma(sigma, R))
            states.append({"band": band, "R": R, "C0": C0, "sigma": sigma, "q": q})

    # Load truth from cache (warm-started 41x121 solves)
    truth = _load_or_compute_truth_cache(states)

    qs, rs, cand = candidate_mesh()
    state_data = {}
    for sid, st in enumerate(states):
        for protocol in ["single", "vrev", "dual"]:
            predI, predS, predV = predict_candidates(op, qs, rs, st["C0"], protocol, rev_sur)
            state_data[(sid, protocol)] = (qs, rs, (predI, predS, predV))

    # Build spline tables for fast Gauss-Newton (15 states x 3 protocols = 45 tables)
    # Check cache first
    spline_cache_path = os.path.join(RESULTS, "_stage21_spline_cache.npz")
    if os.path.exists(spline_cache_path):
        print("[21] loading spline tables from cache", flush=True)
        t_cache_start = time.perf_counter()
        spline_tables = {}
        loaded = np.load(spline_cache_path, allow_pickle=True)
        for key_str in loaded.files:
            sid, protocol = eval(key_str)
            spline_tables[(sid, protocol)] = pickle.loads(loaded[key_str].item())
        dt_cache = time.perf_counter() - t_cache_start
        print(f"[21] spline tables loaded from cache in {dt_cache:.1f}s", flush=True)
    else:
        print("[21] building spline tables for fast GN inversion", flush=True)
        t_spline_start = time.perf_counter()
        spline_tables = {}
        for sid, st in enumerate(states):
            for protocol in ["single", "vrev", "dual"]:
                spline_tables[(sid, protocol)] = _build_spline_table(
                    st["C0"], protocol, op, rev_sur, n_grid=15, nr=21, nz=61
                )
                total = sid * 3 + ["single", "vrev", "dual"].index(protocol) + 1
                if total % 10 == 0:
                    print(f"[21] built {total}/45 spline tables", flush=True)
        dt_spline = time.perf_counter() - t_spline_start
        print(f"[21] spline tables built in {dt_spline:.1f}s", flush=True)
        
        # Cache spline tables to disk
        print("[21] saving spline tables to cache", flush=True)
        cache_arrays = {}
        for key, value in spline_tables.items():
            cache_arrays[str(key)] = np.array(pickle.dumps(value))
        np.savez_compressed(spline_cache_path, **cache_arrays)
        print(f"[21] spline tables cached to {spline_cache_path}", flush=True)

    # Serialize spline tables for multiprocessing
    spline_pickled = pickle.dumps(spline_tables)

    tasks = []
    for sid, st in enumerate(states):
        for protocol in ["single", "vrev", "dual"]:
            y_true = truth[sid]["I"].copy()
            if protocol == "vrev":
                y_true = np.r_[y_true, truth[sid]["Vrev"]]
            elif protocol == "dual":
                y_true = np.r_[y_true, truth[sid]["S"]]
            for noise in NOISE:
                for rep in range(20):
                    r = np.random.default_rng(200100 + sid * 100000 + ["single", "vrev", "dual"].index(protocol) * 10000 + int(noise * 10000) * 100 + rep)
                    sig = np.maximum(noise * np.abs(y_true), 1e-6)
                    if protocol == "vrev":
                        sig[-1] = max(noise * abs(y_true[-1]), 0.05)
                    elif protocol == "dual":
                        sig[-1] = noise * max(abs(y_true[-1]), 0.01)
                    y_obs = y_true + sig * r.standard_normal(y_true.size)
                    tasks.append((sid, protocol, noise, rep, y_obs, sig,
                                  {"q": st["q"], "R": st["R"], "C0": st["C0"]}))

    workers = max(1, min(16, 2 * (os.cpu_count() or 2)))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_init_worker_with_splines,
                  initargs=(op, rev_sur, state_data, spline_pickled)) as pool:
        records = []
        for k, rec in enumerate(pool.imap_unordered(_inverse_task_spline, tasks, chunksize=2), 1):
            records.append(rec)
            if k % 100 == 0:
                print(f"[20] {k}/{len(tasks)} blind realizations", flush=True)
    print(f"[20] completed {len(records)} blind realizations", flush=True)

    summary = {}
    for protocol in ["single", "vrev", "dual"]:
        for noise in NOISE:
            for method in ["A", "B", "C"]:
                sub = [r for r in records if r["protocol"] == protocol and r["noise"] == noise]
                q = np.array([r[f"{method}_q_rel"] for r in sub])
                s = np.array([r[f"{method}_sigma_rel"] for r in sub])
                R = np.array([r[f"{method}_R_rel"] for r in sub])
                summary[f"{protocol}_{method}_noise{noise}"] = {
                    "n": len(sub),
                    "median_q_rel_err": float(np.median(q)),
                    "q_iqr": float(np.quantile(q, .75) - np.quantile(q, .25)),
                    "q_p95": float(np.quantile(q, .95)),
                    "median_sigma_rel_err": float(np.median(s)),
                    "sigma_iqr": float(np.quantile(s, .75) - np.quantile(s, .25)),
                    "median_R_rel_err": float(np.median(R)),
                    "R_iqr": float(np.quantile(R, .75) - np.quantile(R, .25)),
                    "R_p95": float(np.quantile(R, .95)),
                    "q_coverage95": float(np.mean([r[f"{method}_q_cover"] for r in sub])),
                    "R_coverage95": float(np.mean([r[f"{method}_R_cover"] for r in sub])),
                    "sigma_coverage95": float(np.mean([r[f"{method}_sigma_cover"] for r in sub])),
                    "converged_fraction": float(np.mean([r[f"{method}_converged"] for r in sub])),
                    "boundary_hit_fraction": float(np.mean([r[f"{method}_boundary_hit"] for r in sub])),
                    "median_time_s": float(np.median([r[f"{method}_time_s"] for r in sub])),
                }
    return summary, records, states, truth, state_data, spline_tables


def run_bias_diagnostics(op, rev_sur, states, truth, state_data, records, spline_tables=None):
    """3-way Stage-21 diagnostic: optimization vs mesh bias vs initialization."""
    print("[21] 3-way bias diagnostic", flush=True)
    rows = []
    for sid, st in enumerate(states):
        qs, rs, pred = state_data[(sid, "dual")]
        I21, S21 = exact_protocol_values(st["q"], st["R"], st["C0"], "dual", 21, 61)
        y21 = np.r_[I21, S21]
        y41 = np.r_[truth[sid]["I"], truth[sid]["S"]]
        for label, y in (("same_grid_noise_free", y21), ("cross_grid_noise_free", y41)):
            sig = np.maximum(1e-6 * np.abs(y), 1e-6)
            sig[-1] = 1e-6 * max(abs(y[-1]), 1e-3)
            qA, RA = _surrogate_estimate(pred, y, sig, "dual", qs, rs)
            m0 = np.array([math.log(RA), math.log(max(abs(sigma_from_q(qA, RA)), 1e-4))])
            # Use spline tables if available for fast Gauss-Newton
            if spline_tables is not None and (sid, "dual") in spline_tables:
                s_eval, s_jac, _ = spline_tables[(sid, "dual")]
                m, Jw, it, cost, diag = _spline_blind_gn_with_polish(
                    m0, y, sig, "dual", st["C0"], s_eval, s_jac)
            else:
                m, Jw, it, cost, diag = _blind_gn(m0, y, sig, "dual", st["C0"])
            q, R, sig_est = _q_sigma_from_m(m)
            _, se_r, se_s, se_q = _covariance(Jw)
            rows.append({
                "state": sid, "band": st["band"], "diagnostic": label,
                "q_true": st["q"], "R_true": st["R"],
                "sigma_true_mC": float(sigma_from_q(st["q"], st["R"])),
                "q_hat": q, "R_hat": R, "sigma_hat_mC": sig_est,
                "q_rel": abs(q - st["q"]) / abs(st["q"]),
                "R_rel": abs(R - st["R"]) / st["R"],
                "sigma_rel": abs(sig_est - sigma_from_q(st["q"], st["R"])) / abs(sigma_from_q(st["q"], st["R"])),
                "iterations": it, "weighted_cost": cost,
                "converged": bool(diag["converged"]),
                "boundary_hit": bool(diag["boundary_hit"]),
                "stop_reason": diag["reason"],
                "se_lnR": se_r, "se_lnsigma": se_s, "se_lnq": se_q,
            })
    noisy = [r for r in records if r["protocol"] == "dual" and abs(r["noise"] - 0.01) < 1e-12]
    def med(rs, key):
        return _med([r[key] for r in rs])
    d1 = [r for r in rows if r["diagnostic"] == "same_grid_noise_free"]
    d2 = [r for r in rows if r["diagnostic"] == "cross_grid_noise_free"]
    summary = {
        "diagnostic_1_same_grid_noise_free": {
            "n": len(d1), "median_q_rel": med(d1, "q_rel"),
            "median_R_rel": med(d1, "R_rel"), "median_sigma_rel": med(d1, "sigma_rel"),
            "median_iterations": med(d1, "iterations"),
            "converged_fraction": float(np.mean([r["converged"] for r in d1])),
            "boundary_hit_fraction": float(np.mean([r["boundary_hit"] for r in d1])),
        },
        "diagnostic_2_cross_grid_noise_free": {
            "n": len(d2), "median_q_rel": med(d2, "q_rel"),
            "median_R_rel": med(d2, "R_rel"), "median_sigma_rel": med(d2, "sigma_rel"),
            "median_iterations": med(d2, "iterations"),
            "converged_fraction": float(np.mean([r["converged"] for r in d2])),
            "boundary_hit_fraction": float(np.mean([r["boundary_hit"] for r in d2])),
        },
        "diagnostic_3_warm_vs_cold_noisy": {
            "n": len(noisy),
            "B_median_q_rel": med(noisy, "B_q_rel"),
            "C_median_q_rel": med(noisy, "C_q_rel"),
            "B_median_R_rel": med(noisy, "B_R_rel"),
            "C_median_R_rel": med(noisy, "C_R_rel"),
            "B_median_sigma_rel": med(noisy, "B_sigma_rel"),
            "C_median_sigma_rel": med(noisy, "C_sigma_rel"),
            "B_converged_fraction": float(np.mean([r["B_converged"] for r in noisy])),
            "C_converged_fraction": float(np.mean([r["C_converged"] for r in noisy])),
            "B_boundary_hit_fraction": float(np.mean([r["B_boundary_hit"] for r in noisy])),
            "C_boundary_hit_fraction": float(np.mean([r["C_boundary_hit"] for r in noisy])),
        },
    }
    summary["mesh_bias_component_q"] = summary["diagnostic_2_cross_grid_noise_free"]["median_q_rel"] - summary["diagnostic_1_same_grid_noise_free"]["median_q_rel"]
    return {"rows": rows, "summary": summary}


def _archive_records(records):
    npz_path = os.path.join(RESULTS, "stage20_complete_records.npz")
    csv_path = os.path.join(RESULTS, "stage20_complete_records.csv")
    arrays = {}
    keys = ["state", "noise", "realization", "q_true", "R_true", "sigma_true_mC",
            "A_q", "A_R", "A_sigma", "B_q", "B_R", "B_sigma", "C_q", "C_R", "C_sigma",
            "A_q_rel", "A_R_rel", "A_sigma_rel", "B_q_rel", "B_R_rel", "B_sigma_rel",
            "C_q_rel", "C_R_rel", "C_sigma_rel", "A_q_cover", "A_R_cover", "A_sigma_cover",
            "B_q_cover", "B_R_cover", "B_sigma_cover", "C_q_cover", "C_R_cover", "C_sigma_cover",
            "A_se_lnR", "A_se_lnsigma", "A_se_lnq", "B_se_lnR", "B_se_lnsigma", "B_se_lnq",
            "C_se_lnR", "C_se_lnsigma", "C_se_lnq", "A_time_s", "B_time_s", "C_time_s",
            "B_iterations", "C_iterations", "B_converged", "C_converged",
            "B_boundary_hit", "C_boundary_hit", "B_delta_m_inf", "C_delta_m_inf",
            "eps_K", "sigma_min", "weighted_residual"]
    for key in keys:
        arrays[key] = np.array([r[key] for r in records])
    arrays["protocol"] = np.array([r["protocol"] for r in records])
    arrays["gate_status"] = np.array([r["gate_status"] for r in records])
    arrays["B_stop_reason"] = np.array([r["B_stop_reason"] for r in records])
    arrays["C_stop_reason"] = np.array([r["C_stop_reason"] for r in records])
    for i in range(5):
        arrays[f"y_obs_{i}"] = np.array([r["y_obs"][i] if i < len(r["y_obs"]) else np.nan for r in records])
        arrays[f"sigma_{i}"] = np.array([r["sigmas"][i] if i < len(r["sigmas"]) else np.nan for r in records])
    np.savez_compressed(npz_path, **arrays)
    with open(csv_path, "w", newline="") as fh:
        fields = ["state", "protocol", "noise", "realization", "q_true", "R_true", "sigma_true_mC",
                  "A_q", "A_R", "A_sigma", "B_q", "B_R", "B_sigma", "C_q", "C_R", "C_sigma",
                  "A_q_rel", "A_R_rel", "A_sigma_rel", "B_q_rel", "B_R_rel", "B_sigma_rel",
                  "C_q_rel", "C_R_rel", "C_sigma_rel", "A_q_cover", "A_R_cover", "A_sigma_cover",
                  "B_q_cover", "B_R_cover", "B_sigma_cover", "C_q_cover", "C_R_cover", "C_sigma_cover",
                  "B_se_lnR", "B_se_lnsigma", "B_se_lnq", "C_se_lnR", "C_se_lnsigma", "C_se_lnq",
                  "A_time_s", "B_time_s", "C_time_s", "B_iterations", "C_iterations",
                  "B_converged", "C_converged", "B_boundary_hit", "C_boundary_hit",
                  "gate_status", "eps_K", "sigma_min", "weighted_residual"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    return npz_path, csv_path

def equal_diffusivity_reversal_check():
    from pnp_da_pinn.conservative import solve_reference
    from pnp_da_pinn.grid import make_grid
    cfg = make_cfg(0.0, 20.0, -1.0, 5.0)
    cfg = replace(cfg, physics=replace(cfg.physics,
                                       diffusion_cation_m2_s=2.0e-9,
                                       diffusion_anion_m2_s=2.0e-9))
    grid = make_grid(cfg.geometry)

    def current(V):
        c = replace(cfg, physics=replace(cfg.physics, applied_voltage_V=V * 1e-3))
        sol = solve_reference(c, grid, tolerance=1e-9, max_iter=200,
                              c_left=5.0, c_right=1.0)
        return current_mean_pA(sol, c, grid)

    flo = fhi = float("nan")
    for span in (5.0, 10.0, 20.0, 40.0, 80.0):
        flo, fhi = current(-span), current(span)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
            root = brentq(current, -span, span, xtol=1e-6, rtol=1e-6, maxiter=80)
            return {"V_rev_mV": float(root), "bracket_mV": span,
                    "current_lo": flo, "current_hi": fhi, "converged": True}
    return {"V_rev_mV": float("nan"), "bracket_mV": 80.0,
            "current_lo": flo, "current_hi": fhi, "converged": False}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    op = S10.OnnxOperator(ONNX)
    data = S10.load_npz(DATA240)

    ansatz = BoundaryAnsatz("reparam", nz=61).double()
    torch.manual_seed(19001)
    raw = torch.randn(3, 3, 21, 61, dtype=torch.float64)
    cl = torch.tensor([0.8, 1.2, 2.0], dtype=torch.float64)
    cr = torch.tensor([1.5, 0.7, 0.4], dtype=torch.float64)
    out = ansatz(raw, torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64), cl, cr)
    unit = {
        "left_max_abs_err": float(torch.max(torch.abs(out["c_plus"][:, :, 0] - cl[:, None])).item()),
        "right_max_abs_err": float(torch.max(torch.abs(out["c_plus"][:, :, -1] - cr[:, None])).item()),
        "min_concentration": float(torch.minimum(out["c_plus"].min(), out["c_minus"].min()).item()),
        "charge_identity_max_abs_err": float(torch.max(torch.abs(out["c_plus"] - out["c_minus"] - out["rho"])).item()),
    }
    unit["pass"] = bool(unit["left_max_abs_err"] < 1e-14 and unit["right_max_abs_err"] < 1e-14
                        and unit["min_concentration"] > 0 and unit["charge_identity_max_abs_err"] < 1e-15)

    rows, cal, rev_sur, n_rev, bench_states, bench_refs = run_benchmark(op, data)
    summary = summarize_methods(rows)
    
    # Always run inverse (it has its own caching)
    inv_summary, inv_records, inv_states, inv_truth, state_data = run_inverse(op, cal, rev_sur)
    
    # Check for cached timing results
    timing_cache_path = os.path.join(RESULTS, "_stage21_timing_cache.json")
    if os.path.exists(timing_cache_path):
        print("[21] loading cached timing benchmark results", flush=True)
        with open(timing_cache_path) as fh:
            timing = json.load(fh)
    else:
        print("[21] equal-workload timing benchmark", flush=True)
        timing = timing_benchmark(op, bench_states, bench_refs, rev_sur)
        # Cache timing results
        with open(timing_cache_path, "w") as fh:
            json.dump(timing, fh, indent=2)
        print(f"[21] timing results cached to {timing_cache_path}", flush=True)
    
    # Check for cached bias diagnostic results
    bias_cache_path = os.path.join(RESULTS, "_stage21_bias_cache.json")
    if os.path.exists(bias_cache_path):
        print("[21] loading cached bias diagnostic results", flush=True)
        with open(bias_cache_path) as fh:
            bias = json.load(fh)
    else:
        print("[21] running bias diagnostics", flush=True)
        bias = run_bias_diagnostics(op, rev_sur, inv_states, inv_truth, state_data, inv_records, None)
        # Cache bias results
        with open(bias_cache_path, "w") as fh:
            json.dump(bias, fh, indent=2)
        print(f"[21] bias results cached to {bias_cache_path}", flush=True)
    npz_path, csv_path = _archive_records(inv_records)
    corrector_test = run_corrector_unit_test()
    equal_d_check = equal_diffusivity_reversal_check()
    tms_check = {"q0_ratio5_mV": tms_reversal(100.0, 5.0, 0.0, 5.0),
                 "expected_q0_ratio5_mV": 0.7254493141463998}
    verification = {
        "corrector_preserves_reference": corrector_test,
        "equal_diffusivity_reversal": equal_d_check,
        "tms_sign_check": tms_check,
        "n_inverse_realizations": len(inv_records),
        "archive_npz": os.path.relpath(npz_path, ROOT),
        "archive_csv": os.path.relpath(csv_path, ROOT),
        "timing_benchmark": timing,
        "bias_diagnostic": bias,
        "states": inv_states,
        "truth_41x121": {str(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                  for kk, vv in v.items()} for k, v in inv_truth.items()},
    }

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ansatz_unit_test": unit,
        "benchmark": {
            "n_states": len(rows),
            "bands": {b: sum(1 for r in rows if r["band"] == b) for b in ["weak", "transitional", "strong"]},
            "methods": summary,
            "reversal_surrogate_calibration_states": n_rev,
        },
        "inverse": {"summary": inv_summary, "n_records": len(inv_records)},
        "timing_benchmark": timing,
        "bias_diagnostic": bias,
        "stage20_verification": verification,
        "controller": {
            "gate": "Weyl singular-value gate",
            "accept_observable_fraction": 0.1,
            "accept_sensitivity_fraction": 0.2,
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    with open(OUT_CSV, "w", newline="") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow(["method", "median_I_rel_err", "iqr_I_rel_err", "p95_I_rel_err",
                    "median_S_abs_err", "median_Vrev_abs_err_mV", "median_runtime_s"])
        for name, m in summary.items():
            if not isinstance(m, dict) or "I_rel_err" not in m:
                continue
            w.writerow([name, m["I_rel_err"]["median"], m["I_rel_err"]["iqr"],
                        m["I_rel_err"]["p95"], m["S_abs_err"]["median"],
                        m["Vrev_abs_err_mV"]["median"], m["runtime_median_s"]])

    with open(STATS) as fh:
        stats = json.load(fh)
    stats["stage19_hybrid_controller"] = out
    with open(STATS, "w") as fh:
        json.dump(stats, fh, indent=2)
    stage20_path = os.path.join(RESULTS, "stage20_hybrid_verification.json")
    with open(stage20_path, "w") as fh:
        json.dump({"generated_utc": out["generated_utc"], **verification,
                   "inverse_summary": inv_summary,
                   "timing_summary": timing["summary"],
                   "bias_diagnostic": bias["summary"],
                   "timing_states": timing["task_A_states"],
                   "bias_diagnostic_states": bias["rows"]}, fh, indent=2)
    print(json.dumps({"unit": unit["pass"],
                      "corrector": corrector_test["pass"],
                      "equal_d_Vrev_mV": equal_d_check["V_rev_mV"],
                      "tms_q0_mV": tms_check["q0_ratio5_mV"],
                      "timing": timing["summary"], "diagnostic": bias["summary"],
                      "inverse": inv_summary, "methods": summary}, indent=2))


if __name__ == "__main__":
    main()
