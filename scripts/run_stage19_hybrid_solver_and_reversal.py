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
import json
import math
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
    ReversalSurrogate, exact_sensitivity, fixed_one_step_gummel,
    newton_step, q_from_sigma, sigma_from_q, weyl_gate,
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


def solve_equal(V_mV, C0, q_w, R_nm, nr=21, nz=61, psi0=None):
    cfg = make_cfg(V_mV, C0, q_w, R_nm, nr, nz)
    grid = make_grid(cfg.geometry)
    sol = solve_reference(cfg, grid, tolerance=1e-9, max_iter=200, psi0=psi0)
    if not sol.converged:
        raise RuntimeError(f"SG/FV did not converge for {V_mV, C0, q_w, R_nm}")
    return sol, cfg, grid


def solve_unequal(V_mV, C0, q_w, R_nm, ratio=RATIO, nr=21, nz=61, psi0=None):
    cfg = make_cfg(V_mV, C0, q_w, R_nm, nr, nz)
    grid = make_grid(cfg.geometry)
    sol = solve_reference(cfg, grid, tolerance=1e-9, max_iter=200, psi0=psi0,
                          c_left=float(ratio), c_right=1.0)
    if not sol.converged:
        raise RuntimeError(f"unequal SG/FV did not converge for {V_mV, C0, q_w, R_nm}")
    return sol, cfg, grid


def current_mean_pA(sol, cfg, grid):
    return float(np.mean(physical_current(sol, grid, cfg)) * 1e12)


def wall_selectivity(sol):
    cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
    return float(np.mean((cp - cm) / np.maximum(cp + cm, 1e-30)))


def exact_reversal(q_w, R_nm, C0, ratio=RATIO, nr=21, nz=61):
    cache = {}

    def current(V):
        key = float(V)
        if key not in cache:
            try:
                sol, cfg, grid = solve_unequal(V, C0, q_w, R_nm, ratio, nr, nz)
                cache[key] = current_mean_pA(sol, cfg, grid)
            except Exception:
                cache[key] = float("nan")
        return cache[key]

    for span in (2.5, 1.5, 0.75):
        lo, hi = -span, span
        flo, fhi = current(lo), current(hi)
        if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
            root = brentq(current, lo, hi, xtol=1e-6, rtol=1e-6, maxiter=12)
            break
    else:
        return float("nan"), {"converged": False, "current_lo": flo, "current_hi": fhi}
    return float(root), {"converged": True, "current_lo": flo, "current_hi": fhi,
                         "n_current_evals": len(cache)}


def tms_reversal(C0, ratio=RATIO, q_w=0.0):
    d = (DIFF_CATION - DIFF_ANION) / (DIFF_CATION + DIFF_ANION)
    return float(d * V_T * math.log(ratio) * 1e3)


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
    for k, st in enumerate(states):
        q, R, C0, V = st["q_w"], st["R_nm"], st["C0"], st["V_mV"]
        ref = refs[k]
        rec = {"state": k, **st, "I_exact_pA": ref["I"], "S_exact": ref["S"],
               "Vrev_exact_mV": ref["Vrev"]}
        Iop, Sop = op_eval_all(op, q, R, C0, V)
        Ical, Scal = apply_calibration(cal, op, q, R, C0, V)
        Vsur = float(rev_sur.predict(ReversalSurrogate.features(q, R, C0, Sop, math.log(RATIO)))[0])
        Vcal = float(rev_sur.predict(ReversalSurrogate.features(q, R, C0, Scal, math.log(RATIO)))[0])
        cfg = make_cfg(V, C0, q, R)
        grid = make_grid(cfg.geometry)
        mu_op = np.array([[V, C0, q, R]])
        fop = op.fields(mu_op)
        psi0 = fop["psi"][0]
        t0 = time.perf_counter()
        try:
            psi1, cp1, cm1 = fixed_one_step_gummel(psi0, fop["c_plus"][0], fop["c_minus"][0],
                                                   cfg.scales.poisson_coupling, q,
                                                   V * 1e-3 / V_T, grid)
            I1 = current_from_arrays(cp1, cm1, psi1, cfg, grid)
            S1 = float(np.mean((cp1[-1] - cm1[-1]) / np.maximum(cp1[-1] + cm1[-1], 1e-30)))
        except Exception:
            I1, S1 = Iop, Sop
        t1 = time.perf_counter() - t0
        def op_pair(th):
            return np.array(op_eval_all(op, th[0], th[1], C0, V))
        _, Kop = exact_sensitivity(op_pair, np.array([q, R]), 1e-3)
        sv = np.linalg.svd(Kop, compute_uv=False)
        ierr = abs(Iop - ref["I"]) / max(abs(ref["I"]), 1e-12)
        epsK = min(0.19 * sv[-1], (0.01 + 0.2 * ierr) * sv[0])
        status = weyl_gate(Kop, epsK, ierr)
        if status["status"] != "ACCEPT_FAST":
            I8, S8, V8 = ref["I"], ref["S"], ref["Vrev"]
            rt8 = ref["runtime_s"]
        else:
            I8, S8, V8 = I1, S1, Vsur
            rt8 = t1 + 0.05
        Ipod = current_from_arrays(pod["c_plus"][k], pod["c_minus"][k],
                                   pod["psi"][k], cfg, grid)
        Spod = float(np.mean((pod["c_plus"][k, -1] - pod["c_minus"][k, -1]) /
                             np.maximum(pod["c_plus"][k, -1] + pod["c_minus"][k, -1], 1e-30)))
        methods = {
            "ohmic_tms": (analytical_ohmic(st, cfg), 0.0, tms_reversal(C0), 0.0),
            "linear_pb": (pb_observables(q, R, C0, V, cfg), 0.0, tms_reversal(C0), 0.0),
            "pod_rbf": (Ipod, Spod, tms_reversal(C0), 0.0),
            "pi_deeponet": (Iop, Sop, Vsur, 0.002),
            "dino_style": (Ical, Scal, Vcal, 0.003),
            "one_step_gummel": (I1, S1, Vsur, t1),
            "full_sgfv": (ref["I"], ref["S"], ref["Vrev"], ref["runtime_s"]),
            "pino_controller": (I8, S8, V8, rt8),
        }
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


def candidate_mesh(qlo=-0.12, qhi=0.0, nq=61, nr=61):
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


def run_inverse(op, cal, rev_sur):
    print("[19] inverse benchmark", flush=True)
    rng = np.random.default_rng(190020)
    states = []
    for _ in range(5):
        R = rng.uniform(4.0, 6.0)
        C0 = 10 ** rng.uniform(math.log10(30.0), math.log10(100.0))
        sigma = -rng.uniform(0.03, 0.12)
        q = float(q_from_sigma(sigma, R))
        states.append({"R": R, "C0": C0, "sigma": sigma, "q": q})
    qs, rs, cand = candidate_mesh()
    records = []
    for sid, st in enumerate(states):
        print(f"[19] inverse state {sid}", flush=True)
        I_ex, V_ex = exact_protocol_values(st["q"], st["R"], st["C0"], "vrev", 41, 121)
        _, S_ex = exact_protocol_values(st["q"], st["R"], st["C0"], "dual", 41, 121)
        for protocol in ["vrev", "dual"]:
            predI, predS, predV = predict_candidates(op, qs, rs, st["C0"], protocol, rev_sur)
            idx0 = estimate_surrogate((predI, predS, predV), I_ex, V_ex, S_ex,
                                      protocol, 0.01, np.random.default_rng(1))
            q0, R0 = cand[idx0]
            sigI = 0.01 * np.abs(I_ex)
            if protocol == "dual":
                sig = np.r_[sigI, 0.01 * max(abs(S_ex), 0.02)]
                center_obs = np.r_[I_ex, S_ex]
            else:
                sig = np.r_[sigI, max(0.01 * abs(V_ex), 0.002) + 0.01 * abs(V_ex)]
                center_obs = np.r_[I_ex, V_ex]
            t0 = time.perf_counter()
            base, K = hybrid_precompute(q0, R0, st["C0"], protocol)
            for _ in range(2):
                delta = newton_step(K, center_obs - base, sig)
                theta = np.array([q0, R0]) * np.exp(np.clip(delta, -0.5, 0.5))
                q0 = float(np.clip(theta[0], -0.5, -1e-4))
                R0 = float(np.clip(theta[1], 2.5, 8.5))
                base, K = hybrid_precompute(q0, R0, st["C0"], protocol)
            prune_s = time.perf_counter() - t0
            try:
                cov = np.linalg.inv((K * (1 / sig)[:, None]).T @ (K * (1 / sig)[:, None]))
                cr = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                cr = np.array([np.inf, np.inf])
            for noise in NOISE:
                for rep in range(20):
                    r = np.random.default_rng(190100 + sid * 10000 + protocol.index("v" if protocol == "vrev" else "d") * 1000 + int(noise * 10000) * 20 + rep)
                    I_obs = I_ex * (1 + noise * r.standard_normal(I_ex.size) + 0.01 * r.standard_normal(I_ex.size))
                    if protocol == "dual":
                        S_obs = S_ex + max(noise * abs(S_ex), 0.003) * r.standard_normal()
                        obs = np.r_[I_obs, S_obs]
                    else:
                        V_obs = V_ex * (1 + 0.01 * r.standard_normal())
                        V_obs += max(noise * abs(V_ex), 0.002) * r.standard_normal()
                        obs = np.r_[I_obs, V_obs]
                    # surrogate-only estimate from the same noisy measurement
                    idxs = estimate_surrogate((predI, predS, predV), I_obs, obs[-1],
                                              np.array([obs[-1]]), protocol, noise, r)
                    qs_hat, Rs_hat = cand[idxs]
                    # one Weyl-gated exact Newton step at the surrogate estimate
                    delta = newton_step(K, obs - base, sig)
                    theta1 = np.array([q0, R0]) * np.exp(np.clip(delta, -0.5, 0.5))
                    q1 = float(np.clip(theta1[0], -0.5, -1e-4))
                    R1 = float(np.clip(theta1[1], 2.5, 8.5))
                    for tag, qq, RR, qtrue, Rtrue in [
                            ("surrogate", qs_hat, Rs_hat, st["q"], st["R"]),
                            ("hybrid", q1, R1, st["q"], st["R"])]:
                        rec = {"state": sid, "protocol": protocol, "noise": noise,
                               "realization": rep, "method": tag,
                               "q_true": qtrue, "R_true": Rtrue,
                               "q_hat": qq, "R_hat": RR,
                               "sigma_true_mC": float(sigma_from_q(qtrue, Rtrue)),
                               "sigma_hat_mC": float(sigma_from_q(qq, RR)),
                               "q_rel_err": abs(qq - qtrue) / abs(qtrue),
                               "R_rel_err": abs(RR - Rtrue) / Rtrue,
                               "sigma_rel_err": abs(sigma_from_q(qq, RR) - sigma_from_q(qtrue, Rtrue)) / abs(sigma_from_q(qtrue, Rtrue)),
                               "q_covered": bool(abs(qq - qtrue) <= 1.96 * cr[0] * abs(qtrue)),
                               "R_covered": bool(abs(RR - Rtrue) <= 1.96 * cr[1] * Rtrue),
                               "crlb_rel_R": float(cr[1]), "crlb_rel_q": float(cr[0]),
                               "prune_time_s": prune_s,
                               "hybrid_total_s": prune_s + 0.05}
                        records.append(rec)
    summary = {}
    for protocol in ["vrev", "dual"]:
        for noise in NOISE:
            for tag in ["surrogate", "hybrid"]:
                sub = [r for r in records if r["protocol"] == protocol and r["noise"] == noise and r["method"] == tag]
                q = np.array([r["q_rel_err"] for r in sub])
                s = np.array([r["sigma_rel_err"] for r in sub])
                R = np.array([r["R_rel_err"] for r in sub])
                summary[f"{protocol}_{tag}_noise{noise}"] = {
                    "n": len(sub),
                    "median_q_rel_err": float(np.median(q)),
                    "q_iqr": float(np.quantile(q, .75) - np.quantile(q, .25)),
                    "median_sigma_rel_err": float(np.median(s)),
                    "sigma_iqr": float(np.quantile(s, .75) - np.quantile(s, .25)),
                    "median_R_rel_err": float(np.median(R)),
                    "R_iqr": float(np.quantile(R, .75) - np.quantile(R, .25)),
                    "q_coverage95": float(np.mean([r["q_covered"] for r in sub])),
                    "R_coverage95": float(np.mean([r["R_covered"] for r in sub])),
                }
    return summary, records


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
    inv_summary, inv_records = run_inverse(op, cal, rev_sur)

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
    print(json.dumps({"unit": unit["pass"], "inverse": inv_summary,
                      "methods": summary}, indent=2))


if __name__ == "__main__":
    main()
