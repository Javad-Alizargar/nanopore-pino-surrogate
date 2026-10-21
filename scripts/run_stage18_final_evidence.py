"""Stage 18: final evidence pass for the bounded-identifiability revision.

This script regenerates every quantitative result touched by the final review:

  1. Corrected negative-q_w Fisher plane (single-salt I-V, 1% relative noise),
     with finite-difference step convergence, CRLBs in (R, q_w) and
     (R, sigma_w), a low-salt/narrow-pore panel, and an 8x8 exact SG/FV check.
  2. Exact SG/FV-generated five-bias inverse study over 15 held-out states,
     three protocols, three noise levels and 20 realizations, plus a 121x121
     nesting check and a five-state exact SG/FV forward-model reference.
  3. Analytic (ohmic and linearized radial PB), POD-RBF, DeepONet and FNO
     baselines, a fifth matched-loss direct-c constraint scheme, and a
     strong-charge operator benchmark.

Outputs
  results/stage18_final_evidence.json
  results/tables/table_s15_fisher_identifiability_protocols.csv
  results/tables/table_s15b_fisher_plane.csv
  results/tables/table_sgfv_operator_fisher_8x8.csv
  results/tables/table_analytic_pod_baselines.csv
  results/tables/table_strong_charge_benchmark.csv
  results/statistics.json (keys stage18_*)
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from scipy.special import i0, i1

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import generate_parametric_dataset as G  # noqa: E402
import run_stage10_upgrades as S10  # noqa: E402
import run_stage11_reviewer_upgrades as S11  # noqa: E402
from operators import build_model, feature_stats  # noqa: E402
from operators.constants import (  # noqa: E402
    A_LAMBDA_M2,
    DIFF_ANION,
    DIFF_CATION,
    FARADAY,
    GAS_CONSTANT,
    L_NM,
    NR,
    NZ,
    T_K,
    annular_weights,
    make_grid,
)
from operators.losses_and_metrics import GridTensors, sg_current_pA  # noqa: E402

DATA240 = S10.DATA240
RESULTS = S10.RESULTS
TABLES = S10.TABLES
STATS = S10.STATS
OUT_JSON = os.path.join(RESULTS, "stage18_final_evidence.json")
ONNX = os.path.join(ROOT, "web_simulator", "models", "deeponet_reparam_seed42.onnx")
V_BIASES = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
V_NONZERO = np.array([-50.0, -25.0, 25.0, 50.0])
INONZERO = np.array([0, 1, 3, 4])
SALT_MULTI = [10.0, 30.0, 100.0]
NOISE = [0.002, 0.010, 0.020]
R_FIXED = 5.0
FISHER_DELTAS = [1.0e-4, 1.0e-3, 1.0e-2]

R_GRID, Z_GRID, *_ = make_grid(NR, NZ)
DZ = float(Z_GRID[1] - Z_GRID[0])
AREA = annular_weights(R_GRID)[:, None] * DZ
NEAR_WALL = R_GRID >= 0.8 - 1e-9
GT = GridTensors()
VT = GAS_CONSTANT * T_K / FARADAY


# --------------------------------------------------------------------------- #
# basic field/current helpers
# --------------------------------------------------------------------------- #
def load_data():
    return S10.load_npz(DATA240)


def rel_l2_state(pred, ref, mask=None):
    w = AREA if mask is None else AREA * mask[:, None]
    num = np.sqrt(((pred - ref) ** 2 * w).sum(axis=(1, 2)))
    den = np.sqrt((ref ** 2 * w).sum(axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / np.maximum(den, 1e-14)
    out[~np.isfinite(out)] = np.nan
    return out


def current_from_fields(mu, psi, cp, cm):
    mu_t = torch.tensor(np.asarray(mu), dtype=torch.float32)
    with torch.no_grad():
        cur = sg_current_pA(torch.tensor(cp, dtype=torch.float32),
                            torch.tensor(cm, dtype=torch.float32),
                            torch.tensor(psi, dtype=torch.float32), mu_t, GT)
    return cur.numpy().astype(np.float64)


def selectivity_from_fields(cp, cm):
    s = (cp - cm) / np.maximum(cp + cm, 1e-30)
    return s[:, -1, :].mean(axis=1)


def metrics_subset(mu_sub, psi, cp, cm, ref, idx):
    psi = np.asarray(psi, dtype=np.float64)
    cp = np.asarray(cp, dtype=np.float64)
    cm = np.asarray(cm, dtype=np.float64)
    s = cp + cm
    rho = cp - cm
    S = rho[:, -1, :] / np.maximum(s[:, -1, :], 1e-30)
    Sref = (ref["rho"][idx, -1, :] /
            np.maximum(ref["sigma_c"][idx, -1, :], 1e-30))
    out = {
        "psi_rel_l2": float(np.nanmean(rel_l2_state(psi, ref["psi"][idx]))),
        "rho_rel_l2": float(np.nanmean(rel_l2_state(rho, ref["rho"][idx]))),
        "S_rel_l2": float(np.sqrt(np.mean((S - Sref) ** 2)) /
                          max(np.sqrt(np.mean(Sref ** 2)), 1e-14)),
    }
    I = current_from_fields(mu_sub, psi, cp, cm)
    refI = ref["i_z"][idx]
    rel = np.sqrt(((I - refI) ** 2).mean(axis=1)) / np.maximum(
        np.sqrt((refI ** 2).mean(axis=1)), 1e-14)
    out["I_rel_l2"] = float(np.nanmean(rel))
    return out, I


def metrics_from_fields(mu, psi, cp, cm, ref, idx):
    return metrics_subset(mu[idx], np.asarray(psi)[idx], np.asarray(cp)[idx],
                          np.asarray(cm)[idx], ref, idx)


def bulk_fields(mu):
    n = mu.shape[0]
    psi = (mu[:, 0] * 1e-3 / VT)[:, None, None] * (1.0 - Z_GRID)[None, None, :]
    psi = np.broadcast_to(psi, (n, NR, NZ)).copy()
    cp = np.ones((n, NR, NZ))
    cm = np.ones((n, NR, NZ))
    return psi, cp, cm


def pb_fields(mu, lam_over_R):
    """Linearized radial PB approximation with a thin axial boundary layer.

    delta = q_w (lambda/R) [I0(r/(lambda/R))/I1(1/(lambda/R))]
            sin(pi z)^0.2,  c_+- = exp(-+ delta).
    The exponent 0.2 was fixed once from the training set axial profile; no
    per-state fitting is used.
    """
    n = mu.shape[0]
    r = R_GRID[None, :, None]
    z = Z_GRID[None, None, :]
    L = lam_over_R[:, None, None]
    env = np.sin(np.pi * z) ** 0.2
    delta = (mu[:, 2, None, None] * L * (i0(r / L) / i1(1.0 / L)) * env)
    psi = ((mu[:, 0] * 1e-3 / VT)[:, None, None] * (1.0 - z)) + delta
    cp = np.exp(-delta)
    cm = np.exp(delta)
    return psi, cp, cm


def bulk_current(mu):
    c = mu[:, 1]
    Rm = mu[:, 3] * 1e-9
    V = mu[:, 0] * 1e-3
    sgn = np.sign(V)
    return sgn * (FARADAY ** 2 / (GAS_CONSTANT * T_K)) * (DIFF_CATION + DIFF_ANION) * c * np.pi * Rm ** 2 / (L_NM * 1e-9) * np.abs(V) * 1e12


# --------------------------------------------------------------------------- #
# POD-RBF baseline
# --------------------------------------------------------------------------- #
def fit_pod_rbf(mu, fields, train_idx, test_idx, n_modes=30):
    Xtr = mu[train_idx].copy()
    Xte = mu[test_idx].copy()
    mean = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-12
    Xtrn = (Xtr - mean) / sd
    Xten = (Xte - mean) / sd
    out = {}
    info = {}
    for name, arr in fields.items():
        Y = arr[train_idx].reshape(len(train_idx), -1).astype(np.float64)
        ymean = Y.mean(axis=0)
        U, sv, Vt = np.linalg.svd(Y - ymean, full_matrices=False)
        k = min(n_modes, len(sv))
        C = (Y - ymean) @ Vt[:k].T
        rbf = RBFInterpolator(Xtrn, C, kernel="thin_plate_spline", smoothing=1e-8)
        Cte = rbf(Xten)
        pred = Cte @ Vt[:k] + ymean
        out[name] = pred.reshape(len(test_idx), NR, NZ)
        info[name] = {"n_modes": int(k),
                      "train_energy": float((sv[:k] ** 2).sum() / (sv ** 2).sum())}
    return out, info


# --------------------------------------------------------------------------- #
# Fisher machinery
# --------------------------------------------------------------------------- #
def jacobian_from_logs(logs_Rp, logs_Rm, logs_qp, logs_qm, delta):
    dR = (logs_Rp - logs_Rm) / (2.0 * delta)
    dq = (logs_qp - logs_qm) / (2.0 * delta)
    return np.column_stack([dR, dq])


def fisher_summary(J, sigma=0.01):
    """J columns are d observable / d ln R, d ln|q|; noise is relative."""
    J = np.asarray(J, dtype=np.float64)
    F = (J.T @ J) / (sigma ** 2)
    sv = np.linalg.svd(J / sigma, compute_uv=False)
    kappa = float((sv[0] / max(sv[-1], 1e-300)) ** 2) if sv[-1] > 0 else float("inf")
    try:
        cov = np.linalg.inv(F)
        cr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        cr = np.array([np.inf, np.inf])
    # y = (ln R, ln sigma_w), sigma_w = q_w eps VT / R
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Js = J @ A
    Fs = (Js.T @ Js) / (sigma ** 2)
    svs = np.linalg.svd(Js / sigma, compute_uv=False)
    kappa_s = float((svs[0] / max(svs[-1], 1e-300)) ** 2) if svs[-1] > 0 else float("inf")
    try:
        crs = np.sqrt(np.maximum(np.diag(np.linalg.inv(Fs)), 0.0))
    except np.linalg.LinAlgError:
        crs = np.array([np.inf, np.inf])
    return {
        "F": F.tolist(), "J": J.tolist(),
        "kappa_q": kappa, "kappa_sigma": kappa_s,
        "crlb_rel_R": float(cr[0]), "crlb_rel_q": float(cr[1]),
        "crlb_rel_sigma": float(crs[1]),
        "dlnI_dlnR": float(np.mean(J[:, 0])),
        "dlnI_dlnq": float(np.mean(J[:, 1])),
    }


def op_jacobian(op, C0, q, R, delta):
    q = float(q); R = float(R); C0 = float(C0)
    def I(qq, RR):
        mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                              np.full(V_NONZERO.size, qq),
                              np.full(V_NONZERO.size, RR)])
        return np.log(np.maximum(np.abs(op.current(mu)), 1e-14))
    return jacobian_from_logs(I(q, R * (1 + delta)), I(q, R * (1 - delta)),
                              I(q * (1 + delta), R), I(q * (1 - delta), R), delta)


def sgfv_jacobian(C0, q, R, delta, cache=None):
    """Exact SG/FV Jacobian at four non-zero biases."""
    q = float(q); R = float(R); C0 = float(C0)
    key = (round(C0, 9), round(q, 12), round(R, 9), delta)
    if cache is not None and key in cache:
        return cache[key]
    vals = {}
    for tag, qq, RR in [("qp", q * (1 + delta), R), ("qm", q * (1 - delta), R),
                        ("Rp", q, R * (1 + delta)), ("Rm", q, R * (1 - delta))]:
        I = []
        for V in V_NONZERO:
            sol, cfg, grid, _, _ = G.solve_state(float(V), C0, float(qq), float(RR))
            cur = G.physical_current(sol, grid, cfg) * 1e12
            I.append(float(np.mean(cur)))
        vals[tag] = np.log(np.maximum(np.abs(np.array(I)), 1e-14))
    J = jacobian_from_logs(vals["Rp"], vals["Rm"], vals["qp"], vals["qm"], delta)
    if cache is not None:
        cache[key] = J
    return J


def fisher_plane(op, points, deltas=FISHER_DELTAS):
    rows = []
    for (q, C0, R, tag) in points:
        js = {}
        for d in deltas:
            js[d] = fisher_summary(op_jacobian(op, C0, q, R, d))
        main = js[1e-3]
        other = [js[d] for d in deltas if d != 1e-3]
        J0 = np.asarray(main["J"])
        step_j = max(np.linalg.norm(np.asarray(x["J"]) - J0) /
                     max(np.linalg.norm(J0), 1e-300) for x in other)
        step_kappa = max(abs(x["kappa_q"] - main["kappa_q"]) /
                         max(abs(main["kappa_q"]), 1e-300) for x in other)
        rows.append({
            "q_w": float(q), "C0": float(C0), "R_nm": float(R),
            "lambda_D_over_R": float(np.sqrt(A_LAMBDA_M2 / C0) / (R * 1e-9)),
            "panel": tag,
            "kappa_q": main["kappa_q"], "kappa_sigma": main["kappa_sigma"],
            "log10_kappa_q": float(np.log10(max(main["kappa_q"], 1e-300))),
            "crlb_rel_R": main["crlb_rel_R"], "crlb_rel_q": main["crlb_rel_q"],
            "crlb_rel_sigma": main["crlb_rel_sigma"],
            "dlnI_dlnR": main["dlnI_dlnR"], "dlnI_dlnq": main["dlnI_dlnq"],
            "step_jacobian_rel_change_max": float(step_j),
            "step_kappa_rel_change_max": float(step_kappa),
            "step_deltas": deltas,
        })
    return rows


# --------------------------------------------------------------------------- #
# inverse study
# --------------------------------------------------------------------------- #
def candidate_mesh(nq, nr):
    qs = np.linspace(-0.06, 0.0, nq)
    rs = np.linspace(2.5, 8.5, nr)
    qq, rr = np.meshgrid(qs, rs, indexing="ij")
    return qs, rs, np.column_stack([qq.ravel(), rr.ravel()])


def predict_candidates(op, qs, rs, salts):
    qq, rr = np.meshgrid(qs, rs, indexing="ij")
    qv = qq.ravel(); rv = rr.ravel()
    predI = {}; predS = {}
    for C0 in salts:
        outI = np.empty((len(qv), V_BIASES.size), dtype=np.float64)
        outS = np.empty((len(qv), V_BIASES.size), dtype=np.float64)
        for a in range(0, len(qv), 512):
            b = min(a + 512, len(qv))
            mu = np.column_stack([np.tile(V_BIASES, b - a), np.full((b - a) * V_BIASES.size, C0),
                                  np.repeat(qv[a:b], V_BIASES.size),
                                  np.repeat(rv[a:b], V_BIASES.size)])
            f = op.fields(mu)
            cp, cm, psi = f["c_plus"], f["c_minus"], f["psi"]
            dpsi = psi[:, :, 1:] - psi[:, :, :-1]
            jp = -(cp[:, :, 1:] * S10._bernoulli(-dpsi) - cp[:, :, :-1] * S10._bernoulli(dpsi)) / DZ
            jm = -(cm[:, :, 1:] * S10._bernoulli(dpsi) - cm[:, :, :-1] * S10._bernoulli(-dpsi)) / DZ
            dimless = 2.0 * np.pi * np.sum(
                (DIFF_CATION * jp - DIFF_ANION * jm) * op.w[None, :, None], axis=1)
            scale = FARADAY * mu[:, 1] * (mu[:, 3] * 1e-9) ** 2 / (L_NM * 1e-9) * 1e12
            cur = (dimless * scale[:, None]).mean(axis=1)
            outI[a:b] = cur.reshape(b - a, V_BIASES.size)
            s = (cp[:, -1, :] - cm[:, -1, :]) / np.maximum(cp[:, -1, :] + cm[:, -1, :], 1e-30)
            outS[a:b] = s.mean(axis=1).reshape(b - a, V_BIASES.size)
        predI[C0] = outI; predS[C0] = outS
    return predI, predS


def refine_quadratic(cost2, qs, rs, i, j):
    if i <= 0 or i >= len(qs) - 1 or j <= 0 or j >= len(rs) - 1:
        return float(qs[i]), float(rs[j])
    patch = cost2[i - 1:i + 2, j - 1:j + 2]
    dx, dy = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
    A = np.column_stack([np.ones(9), dx.ravel(), dy.ravel(), dx.ravel() ** 2,
                         dy.ravel() ** 2, (dx * dy).ravel()])
    coef = np.linalg.lstsq(A, patch.ravel(), rcond=None)[0]
    H = np.array([[2 * coef[3], coef[5]], [coef[5], 2 * coef[4]]])
    g = np.array([coef[1], coef[2]])
    try:
        step = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        step = np.zeros(2)
    step = np.clip(step, -0.75, 0.75)
    return float(qs[i] + step[0] * (qs[1] - qs[0])), float(rs[j] + step[1] * (rs[1] - rs[0]))


def estimate_from_predictions(predI, predS, obs, salts, protocol, noise, rng, qs, rs):
    nq, nr = len(qs), len(rs)
    cost = np.zeros(nq * nr)
    for C0 in salts:
        It = obs[C0]["I"][INONZERO]
        St = obs[C0]["S"][INONZERO]
        Io = It * (1.0 + noise * rng.standard_normal(It.size))
        sigI = np.maximum(noise * np.abs(It), 1e-12)
        cost += np.mean(((predI[C0][:, INONZERO] - Io[None, :]) / sigI[None, :]) ** 2, axis=1)
        if protocol == "dual":
            So = St + np.maximum(noise * np.abs(St), 1e-3) * rng.standard_normal(St.size)
            sigS = np.maximum(noise * np.abs(St), 1e-3)
            cost += np.mean(((predS[C0][:, INONZERO] - So[None, :]) / sigS[None, :]) ** 2, axis=1)
    cost2 = cost.reshape(nq, nr)
    i, j = np.unravel_index(np.argmin(cost2), cost2.shape)
    return refine_quadratic(cost2, qs, rs, int(i), int(j))


def protocol_jacobian_op(op, C0_true, q, R, protocol, noise):
    """Noise-whitened Jacobian for the selected protocol."""
    q = float(q); R = float(R); C0_true = float(C0_true)
    salts = [C0_true] if protocol == "single" else SALT_MULTI
    rows = []
    def vals(qq, RR, C0):
        mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                              np.full(V_NONZERO.size, qq),
                              np.full(V_NONZERO.size, RR)])
        return op.current(mu), op.wall_selectivity(mu)
    for C0 in salts:
        baseI, baseS = vals(q, R, C0)
        sigI = np.maximum(noise * np.abs(baseI), 1e-12)
        for tag, qq, RR in [("qp", q * (1 + 1e-3), R), ("qm", q * (1 - 1e-3), R),
                            ("Rp", q, R * (1 + 1e-3)), ("Rm", q, R * (1 - 1e-3))]:
            if tag in ("qp", "qm"):
                II, _ = vals(qq, RR, C0)
                if tag == "qp":
                    Ip = II
                else:
                    Im = II
            else:
                II, SS = vals(qq, RR, C0)
                if tag == "Rp":
                    Ip2 = II; Sp = SS
                else:
                    Im2 = II; Sm = SS
        # current rows are already d ln|I|; convert to whitened residuals
        JI = np.column_stack([(np.log(np.maximum(np.abs(Ip2), 1e-14)) -
                               np.log(np.maximum(np.abs(Im2), 1e-14))) / 2e-3,
                              (np.log(np.maximum(np.abs(Ip), 1e-14)) -
                               np.log(np.maximum(np.abs(Im), 1e-14))) / 2e-3]) / noise
        rows.extend(JI.tolist())
        if protocol == "dual":
            sigS = np.maximum(noise * np.abs(baseS), 1e-3)
            JS = np.column_stack([(Sp - Sm) / (2e-3 * sigS),
                                  (baseS - baseS) / (2e-3 * sigS)])
            # direct q derivative of S, not log; recompute q sides
            _, Sq = vals(q * (1 + 1e-3), R, C0)
            _, Sqm = vals(q * (1 - 1e-3), R, C0)
            JS[:, 1] = (Sq - Sqm) / (2e-3 * sigS)
            rows.extend(JS.tolist())
    J = np.asarray(rows)
    F = J.T @ J
    try:
        cr = np.sqrt(np.maximum(np.diag(np.linalg.inv(F)), 0.0))
    except np.linalg.LinAlgError:
        cr = np.array([np.inf, np.inf])
    sv = np.linalg.svd(J, compute_uv=False)
    return {"crlb_rel_R": float(cr[0]), "crlb_rel_q": float(cr[1]),
            "kappa_q": float((sv[0] / max(sv[-1], 1e-300)) ** 2) if sv[-1] > 0 else float("inf")}


def cvar(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)),
            "iqr": float(np.quantile(x, .75) - np.quantile(x, .25)),
            "mean": float(np.mean(x))}


# --------------------------------------------------------------------------- #
# main experiment sections
# --------------------------------------------------------------------------- #
def run_fisher(op):
    print("[18] Fisher plane", flush=True)
    q_plane = -np.geomspace(0.002, 0.05, 18)
    C_plane = np.geomspace(20.0, 200.0, 18)
    pts = [(float(q), float(c), R_FIXED, "plane") for c in C_plane for q in q_plane]
    rows = fisher_plane(op, pts)

    # low-salt / narrow-pore panel, explicitly marked
    q_low = -np.geomspace(0.002, 0.05, 6)
    r_low = [3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
    pts_low = [(float(q), 20.0, float(r), "low_salt_narrow_pore")
               for r in r_low for q in q_low]
    rows_low = fisher_plane(op, pts_low)

    # exact SG/FV 8x8 check
    print("[18] exact SG/FV Fisher 8x8", flush=True)
    q_ex = -np.geomspace(0.002, 0.05, 8)
    C_ex = np.geomspace(20.0, 200.0, 8)
    cache = {}
    exact = []
    for c in C_ex:
        for q in q_ex:
            J = sgfv_jacobian(float(c), float(q), R_FIXED, 1e-3, cache)
            fs = fisher_summary(J)
            jo = fisher_summary(op_jacobian(op, float(c), float(q), R_FIXED, 1e-3))
            exact.append({
                "q_w": float(q), "C0": float(c), "R_nm": R_FIXED,
                "exact_kappa_q": fs["kappa_q"], "op_kappa_q": jo["kappa_q"],
                "exact_crlb_rel_q": fs["crlb_rel_q"], "op_crlb_rel_q": jo["crlb_rel_q"],
                "exact_crlb_rel_R": fs["crlb_rel_R"], "op_crlb_rel_R": jo["crlb_rel_R"],
                "exact_dlnI_dlnq": fs["dlnI_dlnq"], "op_dlnI_dlnq": jo["dlnI_dlnq"],
                "exact_dlnI_dlnR": fs["dlnI_dlnR"], "op_dlnI_dlnR": jo["dlnI_dlnR"],
            })
            print(f"[18] exact {c:7.1f} {q:+.4f} kappa {fs['kappa_q']:.3e} "
                  f"{jo['kappa_q']:.3e}", flush=True)

    # finite-difference convergence for exact at a subset
    print("[18] exact step convergence", flush=True)
    step_rows = []
    for c, q in [(20, -0.002), (70, -0.01), (200, -0.05), (20, -0.05),
                 (200, -0.002), (70, -0.05), (20, -0.01), (200, -0.01)]:
        ks = []; js = []
        for d in FISHER_DELTAS:
            fs = fisher_summary(sgfv_jacobian(float(c), float(q), R_FIXED, d, cache))
            ks.append(fs["kappa_q"]); js.append(np.asarray(fs["J"]))
        step_rows.append({"C0": c, "q_w": q, "kappa_q": ks,
                          "max_jacobian_rel_change": float(max(
                              np.linalg.norm(J - js[1]) / max(np.linalg.norm(js[1]), 1e-300)
                              for J in js)),
                          "max_kappa_rel_change": float(max(
                              abs(k - ks[1]) / max(abs(ks[1]), 1e-300) for k in ks))})
    return {
        "grid": rows, "low_salt_panel": rows_low,
        "summary": {
            "kappa_q_median": float(np.median([r["kappa_q"] for r in rows])),
            "kappa_q_min": float(np.min([r["kappa_q"] for r in rows])),
            "kappa_q_max": float(np.max([r["kappa_q"] for r in rows])),
            "kappa_q_gt_1e8_frac": float(np.mean([r["kappa_q"] > 1e8 for r in rows])),
            "crlb_rel_q_median": float(np.median([r["crlb_rel_q"] for r in rows])),
            "crlb_rel_sigma_median": float(np.median([r["crlb_rel_sigma"] for r in rows])),
            "crlb_rel_R_median": float(np.median([r["crlb_rel_R"] for r in rows])),
            "step_jacobian_rel_change_max": float(max(
                r["step_jacobian_rel_change_max"] for r in rows)),
            "step_kappa_rel_change_max_median": float(np.median(
                [r["step_kappa_rel_change_max"] for r in rows])),
        },
        "exact_8x8": exact,
        "exact_summary": {
            "kappa_ratio_median": float(np.median([e["exact_kappa_q"] / e["op_kappa_q"] for e in exact])),
            "dlnq_rel_median": float(np.median([abs(e["exact_dlnI_dlnq"] - e["op_dlnI_dlnq"]) /
                                                max(abs(e["exact_dlnI_dlnq"]), 1e-30) for e in exact])),
            "crlb_q_ratio_median": float(np.median([e["exact_crlb_rel_q"] / e["op_crlb_rel_q"] for e in exact])),
        },
        "exact_step": step_rows,
    }


def run_inverse(op, data, state_ids, obs, grid_n=61, realizations=20):
    print(f"[18] inverse grid {grid_n}", flush=True)
    qs, rs, cand = candidate_mesh(grid_n, grid_n)
    records = []
    crl = {}
    for sid in state_ids:
        C0_true = float(data["mu"][sid, 1]); q_true = float(data["mu"][sid, 2])
        R_true = float(data["mu"][sid, 3])
        for protocol in ["single", "multi", "dual"]:
            salts = [C0_true] if protocol == "single" else SALT_MULTI
            predI, predS = predict_candidates(op, qs, rs, salts)
            for noise in NOISE:
                crl[(int(sid), protocol, noise)] = protocol_jacobian_op(
                    op, C0_true, q_true, R_true, protocol, noise)
                for k in range(realizations):
                    rng = np.random.default_rng(180000 + int(sid) * 1000 +
                                                {"single": 0, "multi": 1, "dual": 2}[protocol] * 100 +
                                                int(noise * 10000) * 2 + k)
                    qh, Rh = estimate_from_predictions(predI, predS, obs[sid], salts,
                                                       protocol, noise, rng, qs, rs)
                    e_q = (qh - q_true) / max(abs(q_true), 1e-12)
                    e_R = (Rh - R_true) / R_true
                    c = crl[(int(sid), protocol, noise)]
                    records.append({
                        "state": int(sid), "protocol": protocol, "noise": noise,
                        "realization": k, "C0": C0_true, "q_true": q_true,
                        "R_true": R_true, "q_hat": qh, "R_hat": Rh,
                        "q_rel_err": float(abs(qh - q_true) / max(abs(q_true), 1e-12)),
                        "R_rel_err": float(abs(Rh - R_true) / R_true),
                        "q_signed_rel": float(e_q), "R_signed_rel": float(e_R),
                        "crlb_rel_q": c["crlb_rel_q"], "crlb_rel_R": c["crlb_rel_R"],
                        "q_covered": bool(abs(qh - q_true) <= 1.96 * c["crlb_rel_q"] * abs(q_true)),
                        "R_covered": bool(abs(Rh - R_true) <= 1.96 * c["crlb_rel_R"] * R_true),
                    })
            print(f"[18] inv {sid} {protocol:6s} done", flush=True)
    summary = {}
    for protocol in ["single", "multi", "dual"]:
        for noise in NOISE:
            sub = [r for r in records if r["protocol"] == protocol and r["noise"] == noise]
            qa = cvar([r["q_rel_err"] for r in sub]); ra = cvar([r["R_rel_err"] for r in sub])
            qb = cvar([r["q_signed_rel"] for r in sub]); rb = cvar([r["R_signed_rel"] for r in sub])
            summary[f"{protocol}_noise{noise}"] = {
                "median_q_rel_err": qa["median"], "q_iqr": qa["iqr"],
                "median_R_rel_err": ra["median"], "R_iqr": ra["iqr"],
                "q_bias": qb["median"], "R_bias": rb["median"],
                "q_coverage95": float(np.mean([r["q_covered"] for r in sub])),
                "R_coverage95": float(np.mean([r["R_covered"] for r in sub])),
                "q_identifiable_10pct": bool(qa["median"] < 0.10),
                "R_recovered_2pct": bool(ra["median"] < 0.02),
                "n": len(sub),
            }
    return summary, records


def generate_exact_observations(data, state_ids):
    print("[18] exact SG/FV observations", flush=True)
    obs = {}
    for sid in state_ids:
        C0_true = float(data["mu"][sid, 1])
        salts = sorted(set([C0_true] + SALT_MULTI))
        obs[sid] = {}
        for C0 in salts:
            I = []; S = []
            for V in V_BIASES:
                sol, cfg, grid, _, _ = G.solve_state(float(V), float(C0),
                                                     float(data["mu"][sid, 2]),
                                                     float(data["mu"][sid, 3]))
                cur = G.physical_current(sol, grid, cfg) * 1e12
                I.append(float(np.mean(cur)))
                cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
                S.append(float(np.mean((cp - cm) / np.maximum(cp + cm, 1e-30))))
            obs[sid][C0] = {"I": np.array(I), "S": np.array(S)}
        print(f"[18] obs {sid} salts={len(salts)}", flush=True)
    return obs


def exact_inverse_reference(data, state_ids, obs, op_est, n=5):
    print("[18] exact SG/FV forward-model reference", flush=True)
    use = list(state_ids)[::max(1, len(state_ids) // n)][:n]
    out = []
    for sid in use:
        C0_true = float(data["mu"][sid, 1]); q_true = float(data["mu"][sid, 2])
        R_true = float(data["mu"][sid, 3])
        for protocol in ["single", "multi", "dual"]:
            salts = [C0_true] if protocol == "single" else SALT_MULTI
            q0, R0 = op_est[(int(sid), protocol)]
            dq = max(0.15 * abs(q0), 0.002)
            dR = max(0.10 * R0, 0.20)
            qs = np.clip(np.linspace(q0 - 2 * dq, q0 + 2 * dq, 5), -0.06, 0.0)
            rs = np.clip(np.linspace(R0 - 2 * dR, R0 + 2 * dR, 5), 2.5, 8.5)
            rng = np.random.default_rng(99000 + sid + {"single": 0, "multi": 1, "dual": 2}[protocol])
            noise = 0.01
            meas = {}
            for C0 in salts:
                meas[C0] = {
                    "I": obs[sid][C0]["I"] * (1 + noise * rng.standard_normal(V_BIASES.size)),
                    "S": obs[sid][C0]["S"] + np.maximum(noise * np.abs(obs[sid][C0]["S"]), 1e-3) * rng.standard_normal(V_BIASES.size),
                }
            cost = np.zeros((len(qs), len(rs)))
            for ii, q in enumerate(qs):
                for jj, R in enumerate(rs):
                    cc = 0.0
                    for C0 in salts:
                        Io = meas[C0]["I"]
                        So = meas[C0]["S"]
                        for k, V in enumerate(V_BIASES):
                            if k == 2:
                                continue
                            sol, cfg, grid, _, _ = G.solve_state(float(V), float(C0), float(q), float(R))
                            cur = float(np.mean(G.physical_current(sol, grid, cfg) * 1e12))
                            cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
                            ss = float(np.mean((cp - cm) / np.maximum(cp + cm, 1e-30)))
                            cc += ((cur - Io[k]) / max(noise * abs(obs[sid][C0]["I"][k]), 1e-12)) ** 2
                            if protocol == "dual":
                                cc += ((ss - So[k]) / max(noise * abs(obs[sid][C0]["S"][k]), 1e-3)) ** 2
                    cost[ii, jj] = cc
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            qh, Rh = refine_quadratic(cost, qs, rs, int(i), int(j))
            out.append({"state": int(sid), "protocol": protocol,
                        "q_true": q_true, "R_true": R_true,
                        "q_hat": float(qh), "R_hat": float(Rh),
                        "q_rel_err": float(abs(qh - q_true) / max(abs(q_true), 1e-12)),
                        "R_rel_err": float(abs(Rh - R_true) / R_true)})
            print(f"[18] exact inv {sid} {protocol} q={qh:+.4f} R={Rh:.3f}", flush=True)
    return out


def run_baselines(op, data):
    print("[18] baselines", flush=True)
    ref = data["ref"]
    rows = []
    for split, idx in [("interp_test", data["interp_idx"]), ("ood_test", data["ood_idx"])]:
        # ohmic bulk
        psi, cp, cm = bulk_fields(data["mu"])
        m, I = metrics_from_fields(data["mu"], psi, cp, cm, ref, idx)
        rows.append({"split": split, "method": "Ohmic bulk", **m})
        # linearized radial PB
        psi, cp, cm = pb_fields(data["mu"], data["lambda_D_over_R"])
        m, I = metrics_from_fields(data["mu"], psi, cp, cm, ref, idx)
        rows.append({"split": split, "method": "Linearized radial PB", **m})
        # POD-RBF
        fields = {"psi": ref["psi"], "c_plus": ref["c_plus"], "c_minus": ref["c_minus"]}
        pod, info = fit_pod_rbf(data["mu"], fields, data["train_idx"], idx, n_modes=30)
        m, I = metrics_subset(data["mu"][idx], pod["psi"], pod["c_plus"],
                              pod["c_minus"], ref, idx)
        rows.append({"split": split, "method": "POD-RBF (30 modes)", **m})
        # model fields from the released ONNX operators (seed 42), while
        # psi/rho/I remain the canonical three-seed means from statistics.json.
        with open(STATS) as fh:
            st = json.load(fh)
        for key, label, fpath in [
                ("deeponet_reparam_data_plus_physics", "PI-DeepONet",
                 os.path.join(ROOT, "web_simulator/models/deeponet_reparam_seed42.onnx")),
                ("fno_reparam_data_plus_physics", "FNO-2D",
                 os.path.join(ROOT, "web_simulator/models/fno_reparam_seed42.onnx"))]:
            s = st["summary_multiseed_global_rel_l2"][key][split]
            opm = S10.OnnxOperator(fpath)
            f = opm.fields(data["mu"][idx])
            _, I = metrics_subset(data["mu"][idx], f["psi"], f["c_plus"],
                                  f["c_minus"], ref, idx)
            S = selectivity_from_fields(f["c_plus"], f["c_minus"])
            Sref = (ref["rho"][idx, -1, :] /
                    np.maximum(ref["sigma_c"][idx, -1, :], 1e-30)).mean(axis=1)
            m = {
                "psi_rel_l2": s["grel_l2_psi"]["mean"],
                "rho_rel_l2": s["grel_l2_rho"]["mean"],
                "S_rel_l2": float(np.sqrt(np.mean((S - Sref) ** 2)) /
                                      max(np.sqrt(np.mean(Sref ** 2)), 1e-14)),
                "I_rel_l2": s["grel_l2_i"]["mean"],
            }
            rows.append({"split": split, "method": label, **m})
    return rows, {"pod_modes": info}


def run_matched_loss(data):
    print("[18] fifth constraint scheme: matched direct-c loss", flush=True)
    targets = S10.build_targets(data["ref"])
    val_idx = data["train_idx"][::6]
    rows = []; summary = {}
    for seed in S11.SEEDS:
        model, info = S11.train_scheme(data, targets, data["train_idx"], val_idx,
                                       "direct_c_matched_loss", seed, 250,
                                       physics=True, phys_w=1e-3)
        rec = {"scheme": "direct_c_matched_loss", "seed": seed, **info}
        for split, idx in [("interp", data["interp_idx"]), ("ood", data["ood_idx"])]:
            m = S11.evaluate_scheme(model, idx, data)
            rec[split] = {k: float(np.nanmean(v)) for k, v in m.items()}
        rows.append(rec)
        print(f"[18] matched seed{seed} interp_rho={rec['interp']['rel_l2_rho']:.4f} "
              f"ood_rho={rec['ood']['rel_l2_rho']:.4f}", flush=True)
    for split in ["interp", "ood"]:
        summary[split] = {k: S10.summarize([r[split][k] for r in rows])
                          for k in rows[0][split]}
    return {"scheme": "direct_c_matched_loss", "epochs": 250,
            "seeds": S11.SEEDS, "rows": rows, "summary": summary}


def generate_strong(data, n=120, seed=1818):
    print("[18] strong-charge states", flush=True)
    rng = np.random.default_rng(seed)
    mu = []
    for i in range(n):
        V = rng.choice([-50.0, 50.0])
        C0 = float(rng.uniform(10.0, 13.0))
        q = -float(rng.uniform(0.38, 0.50))
        R = float(rng.uniform(2.5, 5.0))
        mu.append([V, C0, q, R])
    mu = np.array(mu)
    psi = np.empty((n, NR, NZ)); cp = np.empty_like(psi); cm = np.empty_like(psi)
    i_z = np.empty((n, NZ - 1)); S = np.empty(n)
    for j, (V, C0, q, R) in enumerate(mu):
        sol, cfg, grid, meth, nc = G.solve_state(V, C0, q, R)
        psi[j] = sol.psi; cp[j] = sol.c_plus; cm[j] = sol.c_minus
        i_z[j] = G.physical_current(sol, grid, cfg) * 1e12
        cpp, cmm = sol.c_plus[-1, :], sol.c_minus[-1, :]
        S[j] = np.mean((cpp - cmm) / np.maximum(cpp + cmm, 1e-30))
        if (j + 1) % 8 == 0:
            print(f"[18] strong {j+1}/{n} q={q:+.3f}", flush=True)
    return {"mu": mu, "ref": {"psi": psi, "c_plus": cp, "c_minus": cm,
                              "rho": cp - cm, "sigma_c": cp + cm, "i_z": i_z},
            "S": S}


def run_strong(data, strong):
    print("[18] strong-charge benchmark", flush=True)
    ntr = 96
    mu = np.concatenate([data["mu"], strong["mu"]], axis=0)
    ref = {k: np.concatenate([data["ref"][k], strong["ref"][k]], axis=0)
           for k in data["ref"]}
    targets = S10.build_targets(ref)
    train_idx = np.concatenate([data["train_idx"], 240 + np.arange(ntr)])
    val_idx = np.concatenate([data["train_idx"][::6], 240 + np.arange(0, ntr, 5)])
    d = {"mu": mu, "ref": ref}
    rows = []
    # bulk baseline
    idx = 240 + np.arange(ntr, len(strong["mu"]))
    bulk = bulk_current(strong["mu"][ntr:])
    refI = strong["ref"]["i_z"][ntr:].mean(axis=1)
    rel = np.abs(bulk - refI) / np.maximum(np.abs(refI), 1e-12)
    bpsi, bcp, bcm = bulk_fields(strong["mu"][ntr:])
    bm, _ = metrics_subset(strong["mu"][ntr:], bpsi, bcp, bcm, ref, idx)
    rows.append({"method": "Ohmic bulk", "I_rel_l2": float(np.median(rel)),
                 "I_rel_l2_mean": float(np.mean(rel)),
                 "psi_rel_l2": bm["psi_rel_l2"], "rho_rel_l2": bm["rho_rel_l2"],
                 "S_rel_l2": bm["S_rel_l2"]})
    for seed in [42, 43, 44]:
        model, info = S11.train_scheme(d, targets, train_idx, val_idx,
                                       "hyperbolic_reparam", seed, 600,
                                       physics=True, phys_w=1e-3)
        pred = S11.predict_full(model, idx, mu)
        I = current_from_fields(mu[idx], pred["psi"], pred["c_plus"], pred["c_minus"]).mean(axis=1)
        rel = np.abs(I - refI) / np.maximum(np.abs(refI), 1e-12)
        S = selectivity_from_fields(pred["c_plus"], pred["c_minus"])
        Sref = strong["S"][ntr:]
        relS = np.abs(S - Sref) / np.maximum(np.abs(Sref), 1e-12)
        fm, _ = metrics_subset(mu[idx], pred["psi"], pred["c_plus"],
                               pred["c_minus"], ref, idx)
        rows.append({"method": "PI-DeepONet, strong-trained", "seed": seed,
                     "I_rel_l2": float(np.median(rel)),
                     "I_rel_l2_mean": float(np.mean(rel)),
                     "S_rel_err": float(np.median(relS)),
                     "psi_rel_l2": fm["psi_rel_l2"],
                     "rho_rel_l2": fm["rho_rel_l2"],
                     "S_rel_l2": fm["S_rel_l2"],
                     "train_time_s": info["train_time_s"]})
        print(f"[18] strong seed{seed} I_rel={np.median(rel):.2e} "
              f"bulk={np.median(rows[0]['I_rel_l2']):.2e}", flush=True)
    return {"n_train": ntr, "n_test": len(strong["mu"]) - ntr, "rows": rows,
            "summary": {"bulk_I_rel_median": rows[0]["I_rel_l2"],
                        "model_I_rel_median": float(np.median([r["I_rel_l2"] for r in rows[1:]])),
                        "improvement_factor": float(rows[0]["I_rel_l2"] / np.median([r["I_rel_l2"] for r in rows[1:]]))},
            "S_test_median": float(np.median(np.abs(strong["S"][ntr:])))}


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def write_tables(fisher, inv_summary, inv_records, baselines, matched, strong):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "table_s15_fisher_identifiability_protocols.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "noise", "median_q_rel_err", "q_iqr", "median_R_rel_err",
                    "R_iqr", "q_bias", "R_bias", "q_coverage95", "R_coverage95",
                    "q_identifiable_10pct", "R_recovered_2pct", "n"])
        for key, s in inv_summary.items():
            proto, noise = key.rsplit("_noise", 1)
            w.writerow([proto, noise, f"{s['median_q_rel_err']:.6g}", f"{s['q_iqr']:.6g}",
                        f"{s['median_R_rel_err']:.6g}", f"{s['R_iqr']:.6g}",
                        f"{s['q_bias']:.6g}", f"{s['R_bias']:.6g}",
                        f"{s['q_coverage95']:.4f}", f"{s['R_coverage95']:.4f}",
                        s["q_identifiable_10pct"], s["R_recovered_2pct"], s["n"]])
    with open(os.path.join(TABLES, "table_s15b_fisher_plane.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "lambda_D_over_R", "C0_mol_m3", "q_w", "R_nm",
                    "kappa_q", "kappa_sigma", "crlb_rel_R", "crlb_rel_q",
                    "crlb_rel_sigma", "dlnI_dlnR", "dlnI_dlnq",
                    "step_jacobian_rel_change_max", "step_kappa_rel_change_max"])
        for r in fisher["grid"] + fisher["low_salt_panel"]:
            w.writerow([r["panel"], f"{r['lambda_D_over_R']:.6g}", f"{r['C0']:.6g}",
                        f"{r['q_w']:.6g}", f"{r['R_nm']:.6g}", f"{r['kappa_q']:.6g}",
                        f"{r['kappa_sigma']:.6g}", f"{r['crlb_rel_R']:.6g}",
                        f"{r['crlb_rel_q']:.6g}", f"{r['crlb_rel_sigma']:.6g}",
                        f"{r['dlnI_dlnR']:.6g}", f"{r['dlnI_dlnq']:.6g}",
                        f"{r['step_jacobian_rel_change_max']:.6g}",
                        f"{r['step_kappa_rel_change_max']:.6g}"])
    with open(os.path.join(TABLES, "table_sgfv_operator_fisher_8x8.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["C0", "q_w", "R_nm", "exact_kappa_q", "op_kappa_q",
                    "exact_crlb_rel_q", "op_crlb_rel_q", "exact_crlb_rel_R", "op_crlb_rel_R"])
        for r in fisher["exact_8x8"]:
            w.writerow([f"{r['C0']:.6g}", f"{r['q_w']:.6g}", f"{r['R_nm']:.6g}",
                        f"{r['exact_kappa_q']:.6g}", f"{r['op_kappa_q']:.6g}",
                        f"{r['exact_crlb_rel_q']:.6g}", f"{r['op_crlb_rel_q']:.6g}",
                        f"{r['exact_crlb_rel_R']:.6g}", f"{r['op_crlb_rel_R']:.6g}"])
    with open(os.path.join(TABLES, "table_analytic_pod_baselines.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "method", "psi_rel_l2", "rho_rel_l2", "S_rel_l2", "I_rel_l2"])
        for r in baselines:
            w.writerow([r["split"], r["method"], f"{r['psi_rel_l2']:.6g}",
                        f"{r['rho_rel_l2']:.6g}", f"{r['S_rel_l2']:.6g}",
                        f"{r['I_rel_l2']:.6g}"])
    with open(os.path.join(TABLES, "table_strong_charge_benchmark.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "I_rel_l2_median", "I_rel_l2_mean",
                    "psi_rel_l2", "rho_rel_l2", "S_rel_l2", "S_rel_err_median",
                    "train_time_s"])
        for r in strong["rows"]:
            w.writerow([r["method"], r.get("seed", ""), f"{r['I_rel_l2']:.6g}",
                        f"{r['I_rel_l2_mean']:.6g}", f"{r.get('psi_rel_l2', float('nan')):.6g}",
                        f"{r.get('rho_rel_l2', float('nan')):.6g}",
                        f"{r.get('S_rel_l2', float('nan')):.6g}",
                        f"{r.get('S_rel_err', float('nan')):.6g}",
                        f"{r.get('train_time_s', '')}"])
    with open(os.path.join(TABLES, "table_s18_matched_loss_scheme.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "split", "metric", "mean", "sd", "median", "n"])
        for split, s in matched["summary"].items():
            for k, v in s.items():
                w.writerow([matched["scheme"], split, k, f"{v['mean']:.6g}",
                            f"{v['sd']:.6g}", f"{v['median']:.6g}", v["n"]])


def merge_stats(fisher, inv_summary, inv_records, baselines, matched, strong,
                coarse_summary, fine_summary, exact_inverse):
    with open(STATS) as fh:
        st = json.load(fh)
    st["stage18_fisher_plane_corrected"] = fisher
    st["stage18_inverse_sgfv"] = {
        "summary": inv_summary, "summary_coarse": coarse_summary,
        "summary_fine": fine_summary, "n_records": len(inv_records),
        "realizations": 20, "noise_levels": NOISE,
        "protocols": ["single", "multi", "dual"],
        "exact_sgfv_reference": exact_inverse,
    }
    st["stage18_baselines"] = baselines
    st["stage18_matched_loss"] = matched
    st["stage18_strong_charge"] = strong
    st["stage"] = 18
    with open(STATS, "w") as fh:
        json.dump(st, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fisher", action="store_true")
    ap.add_argument("--skip-inverse", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-matched", action="store_true")
    ap.add_argument("--skip-strong", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    data = load_data()
    op = S10.OnnxOperator(ONNX)
    out = {"stage": 18, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    fisher = run_fisher(op) if not args.skip_fisher else None
    state_ids = []
    obs = {}
    inv_coarse = inv_fine = records = exact_inverse = None
    if not args.skip_inverse:
        interp = data["interp_idx"]
        order = np.argsort(data["lambda_D_over_R"][interp])
        pick = order[np.linspace(0, len(order) - 1, 15).astype(int)]
        state_ids = interp[pick]
        obs = generate_exact_observations(data, state_ids)
        inv_coarse, records = run_inverse(op, data, state_ids, obs, 61, 20)
        # fine nesting check: whole study at 121x121
        inv_fine, _ = run_inverse(op, data, state_ids, obs, 121, 20)
        op_est = {}
        for sid in state_ids:
            for protocol in ["single", "multi", "dual"]:
                rr = [r for r in records if r["state"] == sid and
                      r["protocol"] == protocol and r["noise"] == 0.01 and
                      r["realization"] == 0][0]
                op_est[(int(sid), protocol)] = (rr["q_hat"], rr["R_hat"])
        exact_inverse = exact_inverse_reference(data, state_ids, obs, op_est, 5)
    baselines = run_baselines(op, data)[0] if not args.skip_baselines else None
    matched = run_matched_loss(data) if not args.skip_matched else None
    strong = None
    if not args.skip_strong:
        strong_states = generate_strong(data)
        strong = run_strong(data, strong_states)

    if fisher is not None: out["fisher"] = fisher
    if inv_coarse is not None:
        out["inverse"] = {"summary": inv_coarse, "summary_fine": inv_fine,
                          "exact_sgfv_reference": exact_inverse,
                          "n_records": len(records)}
    if baselines is not None: out["baselines"] = baselines
    if matched is not None: out["matched_loss"] = matched
    if strong is not None: out["strong_charge"] = strong

    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

    if all(x is not None for x in (fisher, inv_coarse, baselines, matched, strong)):
        write_tables(fisher, inv_coarse, records, baselines, matched, strong)
        merge_stats(fisher, inv_coarse, records, baselines, matched, strong,
                    inv_coarse, inv_fine, exact_inverse)
    print(f"[18] wrote {OUT_JSON} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
