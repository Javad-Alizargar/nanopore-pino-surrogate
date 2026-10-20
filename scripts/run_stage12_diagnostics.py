"""Stage 12 diagnostics (deterministic, local, $0 paid spend).

Three fast analyses that answer reviewer issues 2, 3 and 5:

  1. ``ood_mechanistic_breakdown`` (Issue 3): spatial decomposition of the
     out-of-distribution squared error into reservoir-entrance transitions
     (z in [0,0.10] U [0.90,1.0]), the charged-wall EDL (r >= 0.80,
     0.10 < z < 0.90) and the pore core, for both backbones; plus a
     five-category parametric extrapolation breakdown (combined charge+low
     salt, charge-only, low-salt-only, narrow radius, high voltage) with
     mutually exclusive shares of the total squared error.
  2. ``inversion_selectivity_validation`` (Issue 5): direct validation of the
     operator's near-wall Donnan selectivity S, wall concentrations and
     dS/dln|q_w| against the exact float64 SG/FV solver at the 15 inversion
     states and the three multi-salt concentrations.
  3. ``positivity_stress_test`` (Issue 2): hyperbolic reconstruction versus the
     linear orthogonal projection with Softplus clipping, including the
     analytic Softplus bias ln(2)/beta and its effect on the charge identity
     and on the curvature of c_- near Donnan co-ion exclusion.

Outputs
  results/stage12_diagnostics.json
  results/tables/table_ood_mechanistic_breakdown.csv
  results/tables/table_inversion_selectivity_validation.csv
  results/tables/table_positivity_stress_test.csv
  results/statistics.json  (keys ood_mechanistic_breakdown,
                            inversion_selectivity_validation,
                            positivity_stress_test)
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import generate_parametric_dataset as G  # noqa: E402
import run_stage10_upgrades as S10  # noqa: E402
import run_stage11_reviewer_upgrades as S11  # noqa: E402
from operators.constants import L_NM, NR, NZ, make_grid  # noqa: E402

DATA240 = S10.DATA240
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
MODELS = os.path.join(RESULTS, "models")
STATS = os.path.join(RESULTS, "statistics.json")
OUT_JSON = os.path.join(RESULTS, "stage12_diagnostics.json")
SEL_CACHE = os.path.join(RESULTS, "_stage12_selectivity_cache.json")
POS_CACHE = os.path.join(MODELS, "stage12_linear_ortho_seed42.pt")
SEEDS = [42, 43, 44]
SALT_MULTI = [10.0, 30.0, 100.0]
V_NONZERO = np.array([-50.0, -25.0, 25.0, 50.0])
FD_EPS = 1e-3

R, Z, *_ = make_grid(NR, NZ)
DZ = float(Z[1] - Z[0])
AREA = S10.AREA

Z1 = (Z <= 0.10 + 1e-12) | (Z >= 0.90 - 1e-12)
Z2 = (~Z1)[None, :] & (R[:, None] >= 0.80 - 1e-12)
Z3 = (~Z1)[None, :] & (R[:, None] < 0.80 - 1e-12)
ZONES = {"reservoir_corners": Z1[None, :] & np.ones((NR, 1), bool),
         "wall_edl": Z2, "core": Z3}
ZONE_LABEL = {"reservoir_corners": "Zone 1 reservoir transitions",
              "wall_edl": "Zone 2 charged-wall EDL",
              "core": "Zone 3 pore bulk core"}


def pct(x):
    return float(100.0 * x)


def rel_l2(pred, ref, idx=None):
    if idx is not None:
        pred, ref = pred[idx], ref[idx]
    num = np.sqrt(((pred - ref) ** 2 * AREA).sum(axis=(1, 2)))
    den = np.sqrt((ref ** 2 * AREA).sum(axis=(1, 2)))
    return num / np.maximum(den, 1e-30)


def glob_rel(pred, ref, mask=None):
    """Global area-weighted relative L2 over a set of states (and optional zone)."""
    w = AREA if mask is None else AREA * mask
    num = np.sqrt(((pred - ref) ** 2 * w).sum())
    den = np.sqrt((ref ** 2 * w).sum())
    return float(num / max(den, 1e-30))


def current_profile(op, mu):
    """Operator axial current I(z) for each state, in pA."""
    f = op.fields(mu)
    cp, cm, psi = f["c_plus"], f["c_minus"], f["psi"]
    dpsi = psi[:, :, 1:] - psi[:, :, :-1]
    jp = -(cp[:, :, 1:] * S10._bernoulli(-dpsi) - cp[:, :, :-1] * S10._bernoulli(dpsi)) / DZ
    jm = -(cm[:, :, 1:] * S10._bernoulli(dpsi) - cm[:, :, :-1] * S10._bernoulli(-dpsi)) / DZ
    w = S10.annular_weights(R)
    dimless = 2.0 * np.pi * np.sum(
        (S10.DIFF_CATION * jp - S10.DIFF_ANION * jm) * w[None, :, None], axis=1)
    scale = S10.FARADAY * mu[:, 1] * (mu[:, 3] * 1e-9) ** 2 / (L_NM * 1e-9) * 1e12
    return dimless * scale[:, None]


# --------------------------------------------------------------------------- #
# 1. OOD mechanistic breakdown
# --------------------------------------------------------------------------- #
def categorise(mu):
    """Axis-specific OOD category masks (overlapping)."""
    v = np.abs(mu[:, 0])
    c0 = mu[:, 1]
    q = np.abs(mu[:, 2])
    rr = mu[:, 3]
    return {
        "combined_charge_salt": (q > 0.05) & (c0 < 20.0),
        "charge_only": (q > 0.05) & (c0 >= 20.0),
        "salt_only": (q <= 0.05) & (c0 < 20.0),
        "narrow_radius": (rr >= 2.0) & (rr < 3.0),
        "high_voltage": (v > 50.0) & (v <= 75.0),
    }


def exclusive_labels(mu):
    """Mutually exclusive labels, priority combined > charge > salt > R > V."""
    masks = categorise(mu)
    labels = np.full(len(mu), "other", dtype=object)
    # lowest priority first, highest last so later assignments win
    for key in ["high_voltage", "narrow_radius", "salt_only",
                "charge_only", "combined_charge_salt"]:
        labels[masks[key]] = key
    return labels


def run_ood_breakdown():
    data = S10.load_npz(DATA240)
    ood = data["ood_idx"]
    mu = data["mu"][ood]
    ref = data["ref"]
    t0 = time.perf_counter()
    ops = {"deeponet": S10.OnnxOperator(os.path.join(MODELS, "deeponet_reparam_seed42.onnx")),
           "fno": S10.OnnxOperator(os.path.join(MODELS, "fno_reparam_seed42.onnx"))}
    n_ood = len(ood)
    rows = []
    zones_out = {}
    for name, op in ops.items():
        f = op.fields(mu)
        pred = {"psi": f["psi"], "rho": f["c_plus"] - f["c_minus"]}
        err = {k: (pred[k] - ref[k][ood]) for k in ("psi", "rho")}
        sq = {k: err[k] ** 2 * AREA for k in ("psi", "rho")}
        total = {k: sq[k].sum() for k in ("psi", "rho")}
        zones = {}
        for zk, mask in ZONES.items():
            zones[zk] = {}
            for k in ("psi", "rho"):
                share = sq[k][:, mask].sum() / max(total[k], 1e-30)
                zones[zk][k] = {"share_of_sq_error": float(share),
                                "zone_rel_l2": glob_rel(pred[k], ref[k][ood], mask)}
        ipred = current_profile(op, mu)
        iref = ref["i_z"][ood]
        i_err_state = np.sqrt(((ipred - iref) ** 2).mean(axis=1)) / np.maximum(
            np.sqrt((iref ** 2).mean(axis=1)), 1e-30)
        i_global = float(np.sqrt(((ipred - iref) ** 2).mean())
                         / max(np.sqrt((iref ** 2).mean()), 1e-30))
        # error cancellation: area-weighted cross-section mean of psi error
        w = AREA[:, 0] / AREA[:, 0].sum()
        cs_mean = np.abs((err["psi"] * w[None, :, None]).sum(axis=1))
        psi_rms = np.sqrt((ref["psi"][ood] ** 2).mean(axis=(1, 2)))
        cancel = float(np.median(cs_mean.max(axis=1) / np.maximum(psi_rms, 1e-30)))
        zones_out[name] = {"zones": zones,
                           "psi_rel_l2_global": glob_rel(pred["psi"], ref["psi"][ood]),
                           "rho_rel_l2_global": glob_rel(pred["rho"], ref["rho"][ood]),
                           "I_rel_l2_global": i_global,
                           "cross_section_psi_cancel_ratio": cancel}
        for zk in ZONES:
            rows.append({"backbone": name, "panel": "spatial", "category": zk,
                         "n_states": n_ood,
                         "share_sq_psi": zones[zk]["psi"]["share_of_sq_error"],
                         "share_sq_rho": zones[zk]["rho"]["share_of_sq_error"],
                         "zone_rel_l2_psi": zones[zk]["psi"]["zone_rel_l2"],
                         "zone_rel_l2_rho": zones[zk]["rho"]["zone_rel_l2"],
                         "mean_rel_l2_psi": zones_out[name]["psi_rel_l2_global"],
                         "mean_rel_l2_rho": zones_out[name]["rho_rel_l2_global"],
                         "mean_rel_l2_I": zones_out[name]["I_rel_l2_global"]})
        # parametric breakdown
        masks = categorise(mu)
        labels = exclusive_labels(mu)
        cats = {}
        for key, mask in masks.items():
            if mask.sum() == 0:
                continue
            cats[key] = {"n_states": int(mask.sum()),
                         "mean_rel_l2_psi": glob_rel(pred["psi"][mask], ref["psi"][ood][mask]),
                         "mean_rel_l2_rho": glob_rel(pred["rho"][mask], ref["rho"][ood][mask]),
                         "mean_rel_l2_I": float(np.sqrt(((ipred[mask] - iref[mask]) ** 2).mean())
                                                / max(np.sqrt((iref[mask] ** 2).mean()), 1e-30))}
        excl = {}
        for key in ["combined_charge_salt", "charge_only", "salt_only",
                    "narrow_radius", "high_voltage", "other"]:
            m = labels == key
            if m.sum() == 0:
                continue
            excl[key] = {"n_states": int(m.sum()),
                         "share_sq_psi": float(sq["psi"][m].sum() / max(total["psi"], 1e-30)),
                         "share_sq_rho": float(sq["rho"][m].sum() / max(total["rho"], 1e-30))}
            rows.append({"backbone": name, "panel": "parametric_exclusive",
                         "category": key, "n_states": int(m.sum()),
                         "share_sq_psi": excl[key]["share_sq_psi"],
                         "share_sq_rho": excl[key]["share_sq_rho"],
                         "zone_rel_l2_psi": glob_rel(pred["psi"][m], ref["psi"][ood][m]),
                         "zone_rel_l2_rho": glob_rel(pred["rho"][m], ref["rho"][ood][m]),
                         "mean_rel_l2_psi": cats.get(key, {}).get("mean_rel_l2_psi", float("nan")),
                         "mean_rel_l2_rho": cats.get(key, {}).get("mean_rel_l2_rho", float("nan")),
                         "mean_rel_l2_I": cats.get(key, {}).get("mean_rel_l2_I", float("nan"))})
        zones_out[name]["parametric_axis_specific"] = cats
        zones_out[name]["parametric_exclusive"] = excl
        print(f"[12.1a] {name}: psi zone shares " +
              " ".join(f"{z}={pct(zones[z]['psi']['share_of_sq_error']):.1f}%"
                       for z in ZONES), flush=True)
    with open(os.path.join(TABLES, "table_ood_mechanistic_breakdown.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_ood_states": n_ood, "zones_definition":
               {"reservoir_corners": "z in [0,0.10] U [0.90,1.0]",
                "wall_edl": "r >= 0.80 and 0.10 < z < 0.90",
                "core": "r < 0.80 and 0.10 < z < 0.90"},
               "backbones": zones_out, "wall_s": time.perf_counter() - t0}
    print(f"[12.1a] done in {summary['wall_s']:.1f}s", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# 2. inversion selectivity validation
# --------------------------------------------------------------------------- #
_SEL = {}


def load_sel_cache():
    if os.path.exists(SEL_CACHE):
        with open(SEL_CACHE) as fh:
            for k, v in json.load(fh).items():
                _SEL[tuple(json.loads(k))] = v


def save_sel_cache():
    with open(SEL_CACHE, "w") as fh:
        json.dump({json.dumps(list(k)): v for k, v in _SEL.items()}, fh)


def sg_state(V, C0, q, R):
    key = (round(float(V), 6), round(float(C0), 6), round(float(q), 9), round(float(R), 6))
    if key in _SEL:
        return _SEL[key]
    sol, cfg, grid, meth, nc = G.solve_state(key[0], key[1], key[2], key[3])
    cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
    S = (cp - cm) / np.maximum(cp + cm, 1e-30)
    rec = {"S": float(np.mean(S)), "S_profile": S.tolist(),
           "c_plus_wall": float(np.mean(cp)), "c_minus_wall": float(np.mean(cm)),
           "I": float((G.physical_current(sol, grid, cfg) * 1e12).mean())}
    _SEL[key] = rec
    return rec


def op_state(op, V, C0, q, R):
    f = op.fields(np.array([[V, C0, q, R]]))
    cp, cm = f["c_plus"][0, -1, :], f["c_minus"][0, -1, :]
    S = (cp - cm) / np.maximum(cp + cm, 1e-30)
    I = op.current(np.array([[V, C0, q, R]]))[0]
    return {"S": float(np.mean(S)), "S_profile": S, "c_plus_wall": float(cp.mean()),
            "c_minus_wall": float(cm.mean()), "I": float(I)}


def run_selectivity_validation():
    data = S10.load_npz(DATA240)
    interp = data["interp_idx"]
    lam = data["lambda_D_over_R"][interp]
    order = np.argsort(lam)
    pick = order[np.linspace(0, len(order) - 1, 15).astype(int)]
    state_ids = interp[pick]
    op = S10.OnnxOperator(os.path.join(MODELS, "deeponet_reparam_seed42.onnx"))
    load_sel_cache()
    t0 = time.perf_counter()
    rows = []
    for sid in state_ids:
        V0 = float(data["mu"][sid, 0])
        C0_true = float(data["mu"][sid, 1])
        q = float(data["mu"][sid, 2])
        R = float(data["mu"][sid, 3])
        for C0 in SALT_MULTI:
            ex = sg_state(50.0, C0, q, R)
            pr = op_state(op, 50.0, C0, q, R)
            ex_p = sg_state(50.0, C0, q * (1 + FD_EPS), R)
            ex_m = sg_state(50.0, C0, q * (1 - FD_EPS), R)
            op_p = op_state(op, 50.0, C0, q * (1 + FD_EPS), R)
            op_m = op_state(op, 50.0, C0, q * (1 - FD_EPS), R)
            dS_ex = (ex_p["S"] - ex_m["S"]) / (2 * FD_EPS)
            dS_op = (op_p["S"] - op_m["S"]) / (2 * FD_EPS)
            dcp_ex = (ex_p["c_plus_wall"] - ex_m["c_plus_wall"]) / (2 * FD_EPS)
            dcp_op = (op_p["c_plus_wall"] - op_m["c_plus_wall"]) / (2 * FD_EPS)
            rows.append({
                "state_index": int(sid), "C0": C0, "C0_true": C0_true,
                "q_w": q, "R_nm": R, "lambda_D_over_R": float(data["lambda_D_over_R"][sid]),
                "S_exact": ex["S"], "S_operator": pr["S"],
                "S_abs_err": abs(pr["S"] - ex["S"]),
                "S_rel_err": abs(pr["S"] - ex["S"]) / max(abs(ex["S"]), 1e-30),
                "c_plus_wall_exact": ex["c_plus_wall"], "c_plus_wall_op": pr["c_plus_wall"],
                "c_plus_wall_rel_err": abs(pr["c_plus_wall"] - ex["c_plus_wall"]) / ex["c_plus_wall"],
                "c_minus_wall_exact": ex["c_minus_wall"], "c_minus_wall_op": pr["c_minus_wall"],
                "c_minus_wall_rel_err": abs(pr["c_minus_wall"] - ex["c_minus_wall"]) / ex["c_minus_wall"],
                "dS_dlnq_exact": dS_ex, "dS_dlnq_op": dS_op,
                "dS_dlnq_rel_err": abs(dS_op - dS_ex) / max(abs(dS_ex), 1e-30),
                "dcp_dlnq_exact": dcp_ex, "dcp_dlnq_op": dcp_op,
                "I_exact_pA": ex["I"], "I_op_pA": pr["I"],
            })
    # field-level validation at the operator evaluation point (C0_true)
    field_rows = []
    for sid in state_ids:
        mu = data["mu"][sid:sid + 1]
        f = op.fields(mu)
        for field in ("c_plus", "c_minus"):
            rl = float(rel_l2(f[field], data["ref"][field][sid:sid + 1])[0])
            field_rows.append({"state_index": int(sid), "field": field, "rel_l2": rl})
    save_sel_cache()
    with open(os.path.join(TABLES, "table_inversion_selectivity_validation.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    def med(key):
        return float(np.median([r[key] for r in rows]))
    summary = {
        "n_states": 15, "n_salts": len(SALT_MULTI), "n_rows": len(rows),
        "wall_s": time.perf_counter() - t0,
        "median_S_rel_err": med("S_rel_err"),
        "median_S_abs_err": med("S_abs_err"),
        "max_S_abs_err": float(np.max([r["S_abs_err"] for r in rows])),
        "max_S_rel_err": float(np.max([r["S_rel_err"] for r in rows])),
        "median_c_plus_wall_rel_err": med("c_plus_wall_rel_err"),
        "max_c_plus_wall_rel_err": float(np.max([r["c_plus_wall_rel_err"] for r in rows])),
        "median_c_minus_wall_rel_err": med("c_minus_wall_rel_err"),
        "max_c_minus_wall_rel_err": float(np.max([r["c_minus_wall_rel_err"] for r in rows])),
        "median_dS_dlnq_rel_err": med("dS_dlnq_rel_err"),
        "median_I_rel_err": float(np.median(
            [abs(r["I_op_pA"] - r["I_exact_pA"]) / max(abs(r["I_exact_pA"]), 1e-30) for r in rows])),
        "max_field_rel_l2_c_plus": float(max(r["rel_l2"] for r in field_rows if r["field"] == "c_plus")),
        "max_field_rel_l2_c_minus": float(max(r["rel_l2"] for r in field_rows if r["field"] == "c_minus")),
        "field_validation": field_rows,
    }
    print(f"[12.1b] S rel err median {summary['median_S_rel_err']:.4f} "
          f"max {summary['max_S_rel_err']:.4f}; dS/dlnq median {summary['median_dS_dlnq_rel_err']:.4f}",
          flush=True)
    return summary


# --------------------------------------------------------------------------- #
# 3. positivity stress test
# --------------------------------------------------------------------------- #
def run_positivity_stress():
    t0 = time.perf_counter()
    from operators import build_model, feature_stats

    data = S10.load_npz(DATA240)
    targets = S10.build_targets(data["ref"])
    val_idx = data["train_idx"][::6]
    if os.path.exists(POS_CACHE):
        ck = torch.load(POS_CACHE, map_location="cpu", weights_only=False)
        model = build_model("deeponet", "linear_ortho_projection",
                            ck["stats"], p=64, use_radial_basis=True)
        model.load_state_dict(ck["state_dict"])
        train_s = ck.get("train_s", 0.0)
    else:
        model, info = S11.train_scheme(data, targets, data["train_idx"], val_idx,
                                       "linear_ortho_projection", 42, 250,
                                       physics=True)
        model.eval()
        torch.save({"state_dict": model.state_dict(),
                    "stats": feature_stats(data["mu"][data["train_idx"]]),
                    "train_s": info["train_time_s"]}, POS_CACHE)
        train_s = info["train_time_s"]
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(data["mu"][data["ood_idx"]], dtype=torch.float32))
    cp_l = out["c_plus"].numpy().astype(np.float64)
    cm_l = out["c_minus"].numpy().astype(np.float64)
    rho_l = out["rho_latent"].numpy().astype(np.float64)
    sig_l = out["s"].numpy().astype(np.float64)
    ident_lin = np.abs((cp_l - cm_l) - rho_l)
    raw_cm = 0.5 * (sig_l - rho_l)
    beta = 20.0
    bias_lin = cm_l - np.maximum(raw_cm, 0.0)
    # hyperbolic reconstruction from the same latent fields (float64)
    s_h = sig_l
    rho_h = rho_l
    root = np.sqrt(s_h ** 2 + rho_h ** 2)
    cp_h = 0.5 * (root + rho_h)
    cm_h = 0.5 * (root - rho_h)
    ident_hyp = np.abs((cp_h - cm_h) - rho_h)
    # curvature of c_- with respect to rho at fixed sigma
    d2_lin = beta / 4.0  # max of softplus''(beta x)/beta at x=0
    with np.errstate(divide="ignore", invalid="ignore"):
        d2_hyp = np.nanmax(s_h ** 2 / (2.0 * np.maximum(s_h ** 2 + rho_h ** 2, 1e-30) ** 1.5))
    # analytic worst case on the physical latent grid
    s_grid = np.linspace(0.2, 3.0, 280)
    r_grid = np.linspace(-2.9, 2.9, 581)
    Sg, Rg = np.meshgrid(s_grid, r_grid, indexing="ij")
    valid = np.ones_like(Rg, dtype=bool)
    raw = 0.5 * (Sg - Rg)
    cm_sp = np.logaddexp(0.0, beta * raw) / beta
    neg_frac = float((raw[valid] < 0).mean())
    bias_max = float(np.max(np.abs(cm_sp - np.maximum(raw, 0.0))[valid]))
    ident_max = float(np.max(np.abs((np.logaddexp(0.0, beta * 0.5 * (Sg + Rg)) / beta
                                     - cm_sp) - Rg)[valid]))
    summary = {
        "train_time_s": float(train_s), "wall_s": time.perf_counter() - t0,
        "linear_projection": {
            "min_c_minus": float(cm_l.min()),
            "n_points": int(cm_l.size), "n_negative_raw": int((raw_cm < 0).sum()),
            "fraction_negative_raw": float((raw_cm < 0).mean()),
            "max_abs_identity_mismatch": float(ident_lin.max()),
            "median_abs_identity_mismatch": float(np.median(ident_lin)),
            "max_softplus_bias": float(bias_lin.max()),
            "analytic_softplus_bias_at_zero": float(math.log(2.0) / beta),
            "max_curvature_d2c_drho2": float(d2_lin),
        },
        "hyperbolic": {
            "min_c_minus": float(cm_h.min()),
            "min_c_plus_times_c_minus": float((cp_h * cm_h).min()),
            "max_abs_identity_mismatch": float(ident_hyp.max()),
            "median_abs_identity_mismatch": float(np.median(ident_hyp)),
            "max_curvature_d2c_drho2": float(d2_hyp),
            "identity_mismatch_float64": float(np.abs((cp_h - cm_h) - rho_h).max()),
        },
        "analytic_grid": {
            "fraction_raw_negative": neg_frac,
            "max_softplus_bias": bias_max,
            "max_identity_mismatch": ident_max,
            "n_grid": int(valid.sum()),
        },
    }
    rows = []
    for key in ("linear_projection", "hyperbolic"):
        for mk, mv in summary[key].items():
            rows.append({"scheme": key, "metric": mk, "value": mv})
    with open(os.path.join(TABLES, "table_positivity_stress_test.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"[12.1c] linear min c- {summary['linear_projection']['min_c_minus']:.3e}, "
          f"raw negatives {summary['linear_projection']['n_negative_raw']}; "
          f"hyperbolic identity {summary['hyperbolic']['max_abs_identity_mismatch']:.2e}",
          flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force and os.path.exists(POS_CACHE):
        os.remove(POS_CACHE)
    t0 = time.perf_counter()
    stats = json.loads(open(STATS).read()) if os.path.exists(STATS) else {}
    ood = run_ood_breakdown()
    sel = run_selectivity_validation()
    pos = run_positivity_stress()
    stats["ood_mechanistic_breakdown"] = ood
    stats["inversion_selectivity_validation"] = sel
    stats["positivity_stress_test"] = pos
    open(STATS, "w").write(json.dumps(stats, indent=2, default=float) + "\n")
    payload = {"stage": 12, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "wall_s": time.perf_counter() - t0,
               "ood_mechanistic_breakdown": ood,
               "inversion_selectivity_validation": sel,
               "positivity_stress_test": pos}
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"wrote {OUT_JSON} ({payload['wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
