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
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.optimize import brentq

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


def exact_reversal(q_w, R_nm, C0, ratio=RATIO, nr=21, nz=61, tol=1e-9):
    cache = {}

    def current(V):
        key = float(V)
        if key not in cache:
            try:
                sol, cfg, grid = solve_unequal(V, C0, q_w, R_nm, ratio, nr, nz, tol=tol)
                cache[key] = current_mean_pA(sol, cfg, grid)
            except Exception:
                cache[key] = float("nan")
        return cache[key]

    flo = fhi = float("nan")
    for span in (5.0, 10.0, 20.0, 40.0, 80.0):
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
    print("[19] building 120-state benchmark", flush=True)
    states = build_states()
    refs = []
    for k, st in enumerate(states):
        t0 = time.perf_counter()
        sol, cfg, grid = solve_equal(st["V_mV"], st["C0"], st["q_w"], st["R_nm"])
        I = current_mean_pA(sol, cfg, grid)
        S = wall_selectivity(sol)
        v, info = exact_reversal(st["q_w"], st["R_nm"], st["C0"])
        refs.append({"I": I, "S": S, "Vrev": v, "sol": sol, "cfg": cfg,
                     "grid": grid, "runtime_s": time.perf_counter() - t0,
                     "vrev_info": info})
        if (k + 1) % 20 == 0:
            print(f"[19] refs {k + 1}/120", flush=True)
    print("[19] calibrations", flush=True)
    cal = fit_calibration(op, data)
    rev_sur, n_rev = fit_reversal_surrogate(op, data)
    mu_test = np.array([[st["V_mV"], st["C0"], st["q_w"], st["R_nm"]] for st in states])
    pod = fit_pod_predict(data, mu_test)
    rows = benchmark_methods(op, states, cal, rev_sur, refs, pod)
    return rows, cal, rev_sur, n_rev


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


def _protocol_sgfv_obs(m, protocol, C0, nr=21, nz=61):
    q, R, sig = _q_sigma_from_m(m)
    I = np.empty(V_NONZERO.size)
    for j, V in enumerate(V_NONZERO):
        sol, cfg, grid = solve_equal(float(V), C0, q, R, nr, nz, tol=1e-5)
        I[j] = current_mean_pA(sol, cfg, grid)
    if protocol == "single":
        return I
    if protocol == "dual":
        sol, cfg, grid = solve_equal(0.0, C0, q, R, nr, nz, tol=1e-5)
        return np.r_[I, wall_selectivity(sol)]
    v, info = exact_reversal(q, R, C0, RATIO, nr, nz, tol=1e-4)
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


def _blind_gn(m0, y_obs, sigmas, protocol, C0, max_iter=3):
    m = np.asarray(m0, dtype=float).copy()
    base, J = _jacobian_forward(lambda z: _protocol_sgfv_obs(z, protocol, C0), m)
    Jw = J / sigmas[:, None]
    iterations = 0
    cost = float(np.sum(((y_obs - base) / sigmas) ** 2))
    for _ in range(max_iter):
        iterations += 1
        r = (y_obs - base) / sigmas
        try:
            step = np.linalg.solve(Jw.T @ Jw + 1e-10 * np.eye(2), Jw.T @ r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Jw, r, rcond=None)[0]
        accepted = False
        damp = 1.0
        for _ in range(6):
            cand = m + damp * np.clip(step, -0.5, 0.5)
            cand[0] = np.clip(cand[0], math.log(2.5), math.log(8.5))
            cand[1] = np.clip(cand[1], math.log(0.005), math.log(1.5))
            h = _protocol_sgfv_obs(cand, protocol, C0)
            cc = float(np.sum(((y_obs - h) / sigmas) ** 2))
            if np.isfinite(cc) and cc <= cost * (1.0 + 1e-12):
                m, base, cost = cand, h, cc
                accepted = True
                break
            damp *= 0.5
        if not accepted:
            break
    return m, Jw, iterations, cost

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


def _confidence(Jw, floor=1e-6):
    fisher = Jw.T @ Jw + floor * np.eye(2)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


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


def _inverse_task(task):
    sid, protocol, noise, rep, y_obs, sigmas, truth = task
    qs, rs, pred = _STATE_DATA[(sid, protocol)]
    t0 = time.perf_counter()
    qA, RA = _surrogate_estimate(pred, y_obs, sigmas, protocol, qs, rs)
    tA = time.perf_counter() - t0
    # Warm start in m=(ln R, ln|sigma|) from the noisy-data surrogate estimate.
    sigmaA = float(sigma_from_q(qA, RA))
    mB0 = np.array([math.log(RA), math.log(max(abs(sigmaA), 1e-4))])
    t0 = time.perf_counter()
    mB, JwB, itB, costB = _blind_gn(mB0, y_obs, sigmas, protocol, truth["C0"])
    tB = time.perf_counter() - t0
    qB, RB, sigB = _q_sigma_from_m(mB)
    crB = _confidence(JwB)
    t0 = time.perf_counter()
    starts = [np.array([math.log(3.5), math.log(0.03)]),
              np.array([math.log(5.5), math.log(0.60)])]
    bestC = None
    itC_total = 0
    for m0 in starts:
        mC, JwC, itC, costC = _blind_gn(m0, y_obs, sigmas, protocol, truth["C0"])
        itC_total += itC
        if bestC is None or costC < bestC[0]:
            bestC = (costC, mC, JwC)
    tC = time.perf_counter() - t0
    _, mC, JwC = bestC
    qC, RC, sigC = _q_sigma_from_m(mC)
    crC = _confidence(JwC)
    crA = np.array([np.nan, np.nan])
    try:
        baseA, JA = _jacobian_forward(lambda z: _protocol_surrogate_obs(z, protocol, truth["C0"]), mB0)
        crA = _confidence(JA / sigmas[:, None])
    except Exception:
        pass
    gate_status, epsK, smin, residual = _gate_at(mB, protocol, truth["C0"], y_obs, sigmas, JwB)
    q_true, R_true = truth["q"], truth["R"]
    out = {
        "state": sid, "protocol": protocol, "noise": noise, "realization": rep,
        "q_true": q_true, "R_true": R_true,
        "sigma_true_mC": float(sigma_from_q(q_true, R_true)),
        "y_obs": y_obs, "sigmas": sigmas,
        "A_q": qA, "A_R": RA, "A_sigma": float(sigma_from_q(qA, RA)),
        "B_q": qB, "B_R": RB, "B_sigma": float(sigma_from_q(qB, RB)),
        "C_q": qC, "C_R": RC, "C_sigma": float(sigma_from_q(qC, RC)),
        "A_q_rel": abs(qA - q_true) / abs(q_true), "A_R_rel": abs(RA - R_true) / R_true,
        "B_q_rel": abs(qB - q_true) / abs(q_true), "B_R_rel": abs(RB - R_true) / R_true,
        "C_q_rel": abs(qC - q_true) / abs(q_true), "C_R_rel": abs(RC - R_true) / R_true,
        "A_sigma_rel": abs(sigma_from_q(qA, RA) - sigma_from_q(q_true, R_true)) / abs(sigma_from_q(q_true, R_true)),
        "B_sigma_rel": abs(sigma_from_q(qB, RB) - sigma_from_q(q_true, R_true)) / abs(sigma_from_q(q_true, R_true)),
        "C_sigma_rel": abs(sigma_from_q(qC, RC) - sigma_from_q(q_true, R_true)) / abs(sigma_from_q(q_true, R_true)),
        "A_q_cover": bool(abs(math.log(max(abs(qA), 1e-12)) - math.log(abs(q_true))) <= 1.96 * crA[1]),
        "B_q_cover": bool(abs(mB[1] - math.log(abs(sigma_from_q(q_true, R_true)))) <= 1.96 * crB[1]),
        "C_q_cover": bool(abs(mC[1] - math.log(abs(sigma_from_q(q_true, R_true)))) <= 1.96 * crC[1]),
        "A_R_cover": bool(abs(math.log(RA) - math.log(R_true)) <= 1.96 * crA[0]),
        "B_R_cover": bool(abs(mB[0] - math.log(R_true)) <= 1.96 * crB[0]),
        "C_R_cover": bool(abs(mC[0] - math.log(R_true)) <= 1.96 * crC[0]),
        "A_cr_q": float(crA[1]), "A_cr_R": float(crA[0]),
        "B_cr_q": float(crB[1]), "B_cr_R": float(crB[0]),
        "C_cr_q": float(crC[1]), "C_cr_R": float(crC[0]),
        "B_iterations": itB, "C_iterations": itC_total,
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

    qs, rs, cand = candidate_mesh()
    state_data = {}
    truth = {}
    for sid, st in enumerate(states):
        print(f"[20] truth state {sid + 1}/15", flush=True)
        I41, V41 = exact_protocol_values(st["q"], st["R"], st["C0"], "vrev", 41, 121)
        _, S41 = exact_protocol_values(st["q"], st["R"], st["C0"], "dual", 41, 121)
        truth[sid] = {"I": I41, "Vrev": V41, "S": S41,
                      "q": st["q"], "R": st["R"], "C0": st["C0"],
                      "sigma": st["sigma"], "band": st["band"]}
        for protocol in ["single", "vrev", "dual"]:
            predI, predS, predV = predict_candidates(op, qs, rs, st["C0"], protocol, rev_sur)
            state_data[(sid, protocol)] = (qs, rs, (predI, predS, predV))

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

    workers = max(1, min(8, os.cpu_count() or 2))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(op, rev_sur, state_data)) as pool:
        records = []
        for k, rec in enumerate(pool.imap_unordered(_inverse_task, tasks, chunksize=2), 1):
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
                    "median_time_s": float(np.median([r[f"{method}_time_s"] for r in sub])),
                }
    return summary, records, states, truth


def _archive_records(records):
    npz_path = os.path.join(RESULTS, "stage20_complete_records.npz")
    csv_path = os.path.join(RESULTS, "stage20_complete_records.csv")
    arrays = {}
    keys = ["state", "noise", "realization", "q_true", "R_true", "sigma_true_mC",
            "A_q", "A_R", "A_sigma", "B_q", "B_R", "B_sigma", "C_q", "C_R", "C_sigma",
            "A_q_rel", "A_R_rel", "A_sigma_rel", "B_q_rel", "B_R_rel", "B_sigma_rel",
            "C_q_rel", "C_R_rel", "C_sigma_rel", "A_q_cover", "A_R_cover", "B_q_cover",
            "B_R_cover", "C_q_cover", "C_R_cover", "A_time_s", "B_time_s", "C_time_s",
            "B_iterations", "C_iterations", "eps_K", "sigma_min", "weighted_residual"]
    for key in keys:
        arrays[key] = np.array([r[key] for r in records])
    arrays["protocol"] = np.array([r["protocol"] for r in records])
    arrays["gate_status"] = np.array([r["gate_status"] for r in records])
    for i in range(5):
        arrays[f"y_obs_{i}"] = np.array([r["y_obs"][i] if i < len(r["y_obs"]) else np.nan for r in records])
        arrays[f"sigma_{i}"] = np.array([r["sigmas"][i] if i < len(r["sigmas"]) else np.nan for r in records])
    np.savez_compressed(npz_path, **arrays)
    with open(csv_path, "w", newline="") as fh:
        fields = ["state", "protocol", "noise", "realization", "q_true", "R_true",
                  "A_q", "A_R", "B_q", "B_R", "C_q", "C_R",
                  "A_q_rel", "A_R_rel", "B_q_rel", "B_R_rel", "C_q_rel", "C_R_rel",
                  "A_q_cover", "A_R_cover", "B_q_cover", "B_R_cover", "C_q_cover", "C_R_cover",
                  "A_time_s", "B_time_s", "C_time_s", "B_iterations", "C_iterations",
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

    rows, cal, rev_sur, n_rev = run_benchmark(op, data)
    summary = summarize_methods(rows)
    inv_summary, inv_records, inv_states, inv_truth = run_inverse(op, cal, rev_sur)
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
                   "inverse_summary": inv_summary}, fh, indent=2)
    print(json.dumps({"unit": unit["pass"],
                      "corrector": corrector_test["pass"],
                      "equal_d_Vrev_mV": equal_d_check["V_rev_mV"],
                      "tms_q0_mV": tms_check["q0_ratio5_mV"],
                      "inverse": inv_summary, "methods": summary}, indent=2))


if __name__ == "__main__":
    main()
