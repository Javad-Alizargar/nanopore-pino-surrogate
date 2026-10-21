"""Stage 10 upgrade experiments (deterministic, local, $0 paid spend).

Three bounded, high-impact studies addressing pre-submission review critiques:

  A. Multi-scale radial-trunk ablation: standard `(r, z)` MLP trunk vs. the
     multi-scale exponential EDL trunk, with single-scale kappa in {1,3,10,30}
     and the full multi-scale set, 3 seeds {42,43,44}. Metrics on interp_test,
     ood_test and the near-wall band r >= 0.8 (psi, rho, wall derivative,
     I(z)).
  B. Dataset expansion to 384 verified SG/FV states (N_train = 288) and an
     empirical sample-complexity scaling law over N_train in {36,72,144,288},
     3 seeds, on interp_test and ood_test. The 96 test states are byte-identical
     to the 240-state dataset so every comparison is directly comparable.
  C. Fisher-information spectrum and multi-regime inverse identifiability:
     2x2 dimensionless sensitivity Jacobian, FIM condition number across the
     (lambda_D/R, |q_w|) plane, and three inversion protocols (single-salt I-V,
     multi-salt I-V, joint conductance + near-wall Donnan selectivity).

Outputs
  data/processed/pnp_parametric_384.npz
  results/dataset_manifest_384.json
  results/stage10_upgrades.json
  results/tables/table_s14_radial_trunk_and_scaling_ablation.csv
  results/tables/table_s15_fisher_identifiability_protocols.csv
  results/statistics.json   (keys stage10_*)
  results/models/deeponet_radial_ntrain288_seed42.onnx (if it improves)
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
import train_and_evaluate_suite as T  # noqa: E402
from operators import build_model, feature_stats  # noqa: E402
from operators.constants import (  # noqa: E402
    DIFF_ANION,
    DIFF_CATION,
    FARADAY,
    L_NM,
    NR,
    NZ,
    annular_weights,
    make_grid,
)
from operators.losses_and_metrics import (  # noqa: E402
    GridTensors,
    pnp_physics_loss,
    sg_current_pA,
)

DATA240 = os.path.join(ROOT, "data/processed/pnp_parametric_240.npz")
DATA384 = os.path.join(ROOT, "data/processed/pnp_parametric_384.npz")
MANIFEST384 = os.path.join(ROOT, "results/dataset_manifest_384.json")
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
MODELS = os.path.join(RESULTS, "models")
STATS = os.path.join(RESULTS, "statistics.json")
OUT_JSON = os.path.join(RESULTS, "stage10_upgrades.json")
SEEDS = [42, 43, 44]
NEW_STATES_SEED = 20261020
N_NEW_TRAIN = 144

R, Z, *_ = make_grid(NR, NZ)
DZ = float(Z[1] - Z[0])
AREA = annular_weights(R)[:, None] * DZ
NEAR_WALL = (R >= 0.8 - 1e-9)  # radial band r >= 0.8
GRID = GridTensors()


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    mu = d["mu"].astype(np.float64)
    ref = {k: d[k].astype(np.float64) for k in
           ["psi", "c_plus", "c_minus", "rho", "sigma_c", "i_z"]}
    splits = d["splits"].astype(str)
    return {
        "mu": mu, "ref": ref, "splits": splits,
        "lambda_D_over_R": d["lambda_D_over_R"].astype(np.float64),
        "train_idx": np.where(splits == "train")[0],
        "interp_idx": np.where(splits == "interp_test")[0],
        "ood_idx": np.where(splits == "ood_test")[0],
        "oodA": np.array([i for i in np.where(splits == "ood_test")[0] if 192 <= i < 216]),
        "oodB": np.array([i for i in np.where(splits == "ood_test")[0] if 216 <= i < 240]),
    }


def build_targets(ref):
    cp, cm = ref["c_plus"], ref["c_minus"]
    return {"psi": ref["psi"], "c_plus": cp, "c_minus": cm,
            "rho": cp - cm, "s": 2.0 * np.sqrt(cp * cm), "sigma_c": cp + cm}


def per_state_scales(targets, fit_idx, fields, floor=0.1):
    scales = {}
    for f in fields:
        arr = targets[f][fit_idx]
        rms = np.sqrt((arr ** 2).mean(axis=(1, 2)))
        g = math.sqrt((arr ** 2).mean())
        scales[f] = np.maximum(rms, floor * g)
    return scales


# --------------------------------------------------------------------------- #
# training (adapted from train_and_evaluate_suite.train_run)
# --------------------------------------------------------------------------- #
def train_model(train_idx, val_idx, model_kwargs, fields, seed, epochs, physics,
                phys_w=1e-3, lr=2e-3, batch=36, stats=None, mode="reparam"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = TRAIN_DATA
    targets = TRAIN_TARGETS
    fit_idx = np.array([i for i in train_idx if i not in set(val_idx.tolist())])
    scales = {}
    for f in fields:
        rms = np.sqrt((targets[f] ** 2).mean(axis=(1, 2)))
        g = math.sqrt((targets[f][fit_idx] ** 2).mean())
        scales[f] = np.maximum(rms, 0.1 * g)

    model = build_model("deeponet", mode, stats, **model_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mu = data["mu"]
    best_val, best_state = float("inf"), None
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(fit_idx)
        for st in range(0, len(perm), batch):
            bidx = perm[st:st + batch]
            mu_b = torch.tensor(mu[bidx], dtype=torch.float32)
            opt.zero_grad()
            out = model(mu_b)
            loss = 0.0
            for f in fields:
                tgt = torch.tensor(targets[f][bidx], dtype=torch.float32)
                sc = torch.tensor(scales[f][bidx], dtype=torch.float32).reshape(-1, 1, 1)
                loss = loss + (((out[f] - tgt) / sc) ** 2).mean()
            loss = loss / len(fields)
            if physics:
                pl = pnp_physics_loss(out, mu_b, GRID)
                loss = loss + phys_w * (pl["poisson"] + pl["continuity"] + pl["wall"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vl = eval_loss(model, val_idx, fields, scales)
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, {"val_loss_best": best_val, "epochs": epochs,
                   "train_time_s": train_time, "n_train": int(len(train_idx))}


def eval_loss(model, idx, fields, scales):
    model.eval()
    data, targets = TRAIN_DATA, TRAIN_TARGETS
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bidx = idx[s:s + 64]
            mu_b = torch.tensor(data["mu"][bidx], dtype=torch.float32)
            out = model(mu_b)
            for f in fields:
                tgt = torch.tensor(targets[f][bidx], dtype=torch.float32)
                sc = torch.tensor(scales[f][bidx], dtype=torch.float32).reshape(-1, 1, 1)
                tot += (((out[f] - tgt) / sc) ** 2).mean().item() * len(bidx)
    return tot / max(len(idx), 1)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def predict(model, idx, mu):
    model.eval()
    outs = {}
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bidx = idx[s:s + 64]
            o = model(torch.tensor(mu[bidx], dtype=torch.float32))
            for k in ["psi", "c_plus", "c_minus", "rho", "s"]:
                outs.setdefault(k, []).append(o[k].numpy().astype(np.float64))
    return {k: np.concatenate(v, 0) for k, v in outs.items()}


def _rel_l2(pred, ref, mask=None):
    w = AREA if mask is None else AREA * mask[:, None]
    num = np.sqrt(((pred - ref) ** 2 * w).sum(axis=(1, 2)))
    den = np.sqrt((ref ** 2 * w).sum(axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def evaluate(model, idx, data, fields=("psi", "rho"), near_wall=True):
    pred = predict(model, idx, data["mu"])
    ref = data["ref"]
    res = {}
    for f in fields:
        res[f"rel_l2_{f}"] = _rel_l2(pred[f], ref[f][idx])
        if near_wall:
            res[f"nw_rel_l2_{f}"] = _rel_l2(pred[f], ref[f][idx], NEAR_WALL)
    # wall derivative + current
    psi = torch.tensor(pred["psi"], dtype=torch.float32)
    cp = torch.tensor(pred["c_plus"], dtype=torch.float32)
    cm = torch.tensor(pred["c_minus"], dtype=torch.float32)
    mu_b = torch.tensor(data["mu"][idx], dtype=torch.float32)
    with torch.no_grad():
        q_w = mu_b[:, 2].reshape(-1, 1)
        dpsi_dr = (psi[:, -1, :] - psi[:, -2, :]) / GRID.dr
        res["wall_deriv_err"] = (dpsi_dr - q_w).abs().mean(dim=1).numpy()
        ipred = sg_current_pA(cp, cm, psi, mu_b, GRID).numpy().astype(np.float64)
    iref = ref["i_z"][idx]
    res["i_rel_l2"] = (np.sqrt(((ipred - iref) ** 2).mean(axis=1))
                       / np.sqrt((iref ** 2).mean(axis=1)))
    res["i_abs_err_pA"] = np.abs(ipred - iref).mean(axis=1)
    return res, pred


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=float("nan"), median=float("nan"), sd=float("nan"), n=0)
    return dict(mean=float(arr.mean()), median=float(np.median(arr)),
                sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0, n=int(arr.size))


# --------------------------------------------------------------------------- #
# Experiment B: 384-state dataset
# --------------------------------------------------------------------------- #
def append_numpy(a, b):
    return np.concatenate([a, b], axis=0)


def generate_384():
    if os.path.exists(DATA384):
        print("[384] exists, reusing", flush=True)
        return
    d = np.load(DATA240, allow_pickle=True)
    keys = d.files
    mu0 = d["mu"].astype(np.float64)
    splits0 = d["splits"].astype(str)
    n0 = mu0.shape[0]
    assert n0 == 240

    rng = np.random.default_rng(NEW_STATES_SEED)
    new_mu = np.array([G._map_box(u) for u in G.lhs(N_NEW_TRAIN, 4, rng)])
    print(f"[384] generating {N_NEW_TRAIN} new in-distribution states", flush=True)

    psi = np.empty((N_NEW_TRAIN, NR, NZ), dtype=np.float32)
    c_plus = np.empty_like(psi)
    c_minus = np.empty_like(psi)
    rho = np.empty_like(psi)
    sigma_c = np.empty_like(psi)
    i_z = np.empty((N_NEW_TRAIN, NZ - 1), dtype=np.float64)
    residual = np.empty(N_NEW_TRAIN)
    iterations = np.empty(N_NEW_TRAIN, dtype=np.int64)
    runtime_s = np.empty(N_NEW_TRAIN)
    converged = np.empty(N_NEW_TRAIN, dtype=bool)
    lambda_d_m = np.empty(N_NEW_TRAIN)
    lambda_d_over_r = np.empty(N_NEW_TRAIN)
    beta = np.empty(N_NEW_TRAIN)
    alpha_r = np.empty(N_NEW_TRAIN)
    mean_current_pA = np.empty(N_NEW_TRAIN)
    rel_current_range = np.empty(N_NEW_TRAIN)
    n_cont = np.empty(N_NEW_TRAIN, dtype=np.int64)
    method = np.empty(N_NEW_TRAIN, dtype="<U16")

    t0 = time.perf_counter()
    for j, (V, C0, q, Rr) in enumerate(new_mu):
        sol, cfg, grid, meth, nc = G.solve_state(V, C0, q, Rr)
        cur = G.physical_current(sol, grid, cfg) * 1e12
        cp, cm = sol.c_plus, sol.c_minus
        psi[j] = sol.psi.astype(np.float32)
        c_plus[j] = cp.astype(np.float32)
        c_minus[j] = cm.astype(np.float32)
        rho[j] = (cp - cm).astype(np.float32)
        sigma_c[j] = (cp + cm).astype(np.float32)
        i_z[j] = cur
        residual[j] = sol.residual
        iterations[j] = sol.iterations
        runtime_s[j] = sol.runtime_s
        converged[j] = sol.converged
        lam = cfg.scales.debye_length_m
        lambda_d_m[j] = lam
        lambda_d_over_r[j] = lam / (Rr * 1e-9)
        beta[j] = cfg.scales.poisson_coupling
        alpha_r[j] = grid.radial_metric
        mean_current_pA[j] = float(cur.mean())
        rel_current_range[j] = float(np.ptp(cur) / max(np.max(np.abs(cur)), 1e-30))
        n_cont[j] = nc
        method[j] = meth
        if (j + 1) % 24 == 0 or j == 0:
            print(f"[384] {j+1:3d}/{N_NEW_TRAIN} V={V:6.1f} C0={C0:6.1f} q={q:+.3f} "
                  f"R={Rr:4.1f} res={sol.residual:.2e} I={mean_current_pA[j]:9.4f}",
                  flush=True)
    wall = time.perf_counter() - t0
    assert np.all(residual < 1e-9), float(residual.max())
    assert c_plus.min() > 0 and c_minus.min() > 0

    out = {}
    for k in keys:
        out[k] = d[k]
    # concatenate arrays along the state axis where applicable
    def cat(name, new):
        if name in d.files and getattr(d[name], "ndim", 0) >= 1 and d[name].shape[0] == n0:
            return np.concatenate([d[name], new], axis=0)
        return d[name]
    out["mu"] = np.concatenate([mu0, new_mu], axis=0)
    out["psi"] = cat("psi", psi)
    out["c_plus"] = cat("c_plus", c_plus)
    out["c_minus"] = cat("c_minus", c_minus)
    out["rho"] = cat("rho", rho)
    out["sigma_c"] = cat("sigma_c", sigma_c)
    out["i_z"] = cat("i_z", i_z)
    out["residual"] = cat("residual", residual)
    out["iterations"] = cat("iterations", iterations)
    out["runtime_s"] = cat("runtime_s", runtime_s)
    out["converged"] = cat("converged", converged)
    out["lambda_D_m"] = cat("lambda_D_m", lambda_d_m)
    out["lambda_D_over_R"] = cat("lambda_D_over_R", lambda_d_over_r)
    out["beta"] = cat("beta", beta)
    out["alpha_r"] = cat("alpha_r", alpha_r)
    out["mean_current_pA"] = cat("mean_current_pA", mean_current_pA)
    out["rel_current_range"] = cat("rel_current_range", rel_current_range)
    out["n_continuation_steps"] = cat("n_continuation_steps", n_cont)
    out["method"] = cat("method", method)
    splits_new = np.concatenate([splits0, np.array(["train"] * N_NEW_TRAIN)])
    out["splits"] = splits_new
    np.savez_compressed(DATA384, **out)
    manifest = {
        "stage": 10, "task": "10.1B_384_state_expansion",
        "base_dataset": "data/processed/pnp_parametric_240.npz",
        "n_states": int(out["mu"].shape[0]),
        "n_train": int((splits_new == "train").sum()),
        "n_interp_test": int((splits_new == "interp_test").sum()),
        "n_ood_test": int((splits_new == "ood_test").sum()),
        "new_train_seed": NEW_STATES_SEED,
        "new_train_residual_max": float(residual.max()),
        "new_train_wall_seconds": round(wall, 2),
        "test_states_identical_to_240": True,
        "size_bytes": os.path.getsize(DATA384),
    }
    with open(MANIFEST384, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[384] wrote {DATA384} ({manifest['size_bytes']/1e6:.2f} MB) "
          f"in {wall:.1f}s, max_res={residual.max():.2e}", flush=True)


# --------------------------------------------------------------------------- #
# Experiment A: radial-trunk ablation
# --------------------------------------------------------------------------- #
ABLATION_CONFIGS = {
    "mlp_trunk": dict(use_radial_basis=False),
    "radial_k1": dict(use_radial_basis=True, radial_scales=(1.0,)),
    "radial_k3": dict(use_radial_basis=True, radial_scales=(3.0,)),
    "radial_k10": dict(use_radial_basis=True, radial_scales=(10.0,)),
    "radial_k30": dict(use_radial_basis=True, radial_scales=(30.0,)),
    "radial_multiscale": dict(use_radial_basis=True, radial_scales=(1.0, 3.0, 10.0, 30.0)),
}


def run_ablation(epochs):
    data = load_npz(DATA240)
    global TRAIN_DATA, TRAIN_TARGETS
    TRAIN_DATA = data
    TRAIN_TARGETS = build_targets(data["ref"])
    stats = feature_stats(data["mu"][data["train_idx"]])
    val_idx = data["train_idx"][::6]
    fields = ["psi", "s", "rho"]
    rows = []
    summary = {}
    for name, kw in ABLATION_CONFIGS.items():
        summary[name] = {}
        for seed in SEEDS:
            model, info = train_model(data["train_idx"], val_idx, dict(p=64, **kw),
                                      fields, seed, epochs, physics=True, stats=stats)
            m_int, _ = evaluate(model, data["interp_idx"], data)
            m_ood, _ = evaluate(model, data["ood_idx"], data)
            rec = dict(config=name, seed=seed, n_params=int(sum(p.numel() for p in model.parameters())),
                       **{f"interp_{k}": float(np.nanmean(v)) for k, v in m_int.items()},
                       **{f"ood_{k}": float(np.nanmean(v)) for k, v in m_ood.items()},
                       **info)
            rows.append(rec)
            print(f"[A] {name:18s} seed{seed} nw_rho={np.nanmean(m_int['nw_rel_l2_rho']):.4f} "
                  f"rho={np.nanmean(m_int['rel_l2_rho']):.4e} wall={np.nanmean(m_int['wall_deriv_err']):.3e} "
                  f"({info['train_time_s']:.1f}s)", flush=True)
    # aggregate across seeds
    for name in ABLATION_CONFIGS:
        sub = [r for r in rows if r["config"] == name]
        summary[name] = {
            k: summarize([r[k] for r in sub])
            for k in sub[0] if k not in ("config", "seed", "epochs")
        }
    return {"configs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                        for k, v in ABLATION_CONFIGS.items()},
            "epochs": epochs, "seeds": SEEDS, "rows": rows, "summary": summary}


# --------------------------------------------------------------------------- #
# Experiment B: sample-complexity scaling law
# --------------------------------------------------------------------------- #
def run_scaling(epochs, n_list=(36, 72, 144, 288)):
    data = load_npz(DATA384)
    global TRAIN_DATA, TRAIN_TARGETS
    TRAIN_DATA = data
    TRAIN_TARGETS = build_targets(data["ref"])
    train_all = data["train_idx"]  # 0..143 then 240..383
    assert len(train_all) == 288
    fields = ["psi", "s", "rho"]
    rows = []
    summary = {}
    for n in n_list:
        sub = train_all[:n]
        stats = feature_stats(data["mu"][sub])
        val_idx = sub[::6]
        summary[n] = {}
        for seed in SEEDS:
            model, info = train_model(sub, val_idx,
                                      dict(p=64, use_radial_basis=True,
                                           radial_scales=(1.0, 3.0, 10.0, 30.0)),
                                      fields, seed, epochs, physics=True, stats=stats)
            m_int, _ = evaluate(model, data["interp_idx"], data)
            m_ood, _ = evaluate(model, data["ood_idx"], data)
            rows.append(dict(n_train=n, seed=seed,
                             interp_psi=float(np.nanmean(m_int["rel_l2_psi"])),
                             interp_rho=float(np.nanmean(m_int["rel_l2_rho"])),
                             interp_i=float(np.nanmean(m_int["i_rel_l2"])),
                             interp_wall=float(np.nanmean(m_int["wall_deriv_err"])),
                             ood_psi=float(np.nanmean(m_ood["rel_l2_psi"])),
                             ood_rho=float(np.nanmean(m_ood["rel_l2_rho"])),
                             ood_i=float(np.nanmean(m_ood["i_rel_l2"])),
                             ood_wall=float(np.nanmean(m_ood["wall_deriv_err"])),
                             nw_interp_rho=float(np.nanmean(m_int["nw_rel_l2_rho"])),
                             **{k: v for k, v in info.items() if k != "n_train"}))
            print(f"[B] N={n:3d} seed{seed} interp_rho={np.nanmean(m_int['rel_l2_rho']):.4e} "
                  f"ood_rho={np.nanmean(m_ood['rel_l2_rho']):.4e} ({info['train_time_s']:.1f}s)",
                  flush=True)
    for n in n_list:
        sub = [r for r in rows if r["n_train"] == n]
        summary[n] = {k: summarize([r[k] for r in sub])
                      for k in sub[0] if k not in ("seed", "epochs")}
    # empirical power-law fit: err = a * N^(-alpha)
    fits = {}
    for metric in ["interp_rho", "interp_psi", "ood_rho", "ood_psi", "interp_i", "ood_i",
                   "interp_wall", "ood_wall"]:
        xs = np.array([n for n in n_list], dtype=float)
        ys = np.array([summary[n][metric]["mean"] for n in n_list], dtype=float)
        m = np.isfinite(ys) & (ys > 0)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)
            fits[metric] = {"alpha": float(-slope), "log_a": float(intercept),
                            "a": float(np.exp(intercept))}
    best_model = None
    best_err = float("inf")
    for seed in SEEDS:
        sub = [r for r in rows if r["n_train"] == 288 and r["seed"] == seed]
        if sub and sub[0]["interp_rho"] < best_err:
            best_err = sub[0]["interp_rho"]
    return {"n_list": list(n_list), "epochs": epochs, "seeds": SEEDS,
            "rows": rows, "summary": summary, "power_law_fits": fits,
            "n_train_288_best_interp_rho": best_err}


# --------------------------------------------------------------------------- #
# Experiment C: Fisher information + inverse identifiability
# --------------------------------------------------------------------------- #
def load_onnx_session(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]


def _bernoulli(x):
    xc = np.clip(x, -80.0, 80.0)
    small = np.abs(xc) < 1e-8
    series = 1.0 - xc / 2.0 + xc ** 2 / 12.0
    xs = np.where(small, 1.0, xc)
    return np.where(small, series, xs / np.expm1(xs))


class OnnxOperator:
    def __init__(self, path):
        self.sess, self.input, self.outputs = load_onnx_session(path)
        self.w = annular_weights(R)

    def fields(self, mu):
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        out = self.sess.run(None, {self.input: mu.astype(np.float32)})
        return dict(zip(self.outputs, out))

    def current(self, mu):
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        f = self.fields(mu)
        cp, cm, psi = f["c_plus"], f["c_minus"], f["psi"]
        dpsi = psi[:, :, 1:] - psi[:, :, :-1]
        jp = -(cp[:, :, 1:] * _bernoulli(-dpsi) - cp[:, :, :-1] * _bernoulli(dpsi)) / DZ
        jm = -(cm[:, :, 1:] * _bernoulli(dpsi) - cm[:, :, :-1] * _bernoulli(-dpsi)) / DZ
        dimless = 2.0 * np.pi * np.sum(
            (DIFF_CATION * jp - DIFF_ANION * jm) * self.w[None, :, None], axis=1)
        scale = FARADAY * mu[:, 1] * (mu[:, 3] * 1e-9) ** 2 / (L_NM * 1e-9) * 1e12
        return (dimless * scale[:, None]).mean(axis=1)

    def wall_selectivity(self, mu):
        """Mean near-wall Donnan selectivity S = (c_+ - c_-)/(c_+ + c_-) at r=1."""
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        f = self.fields(mu)
        cp, cm = f["c_plus"][:, -1, :], f["c_minus"][:, -1, :]
        s = (cp - cm) / np.maximum(cp + cm, 1e-30)
        return s.mean(axis=1)


ALPHA = 1e-3  # relative step for finite differences
FIM_BIASES = np.array([-50.0, -25.0, 25.0, 50.0])  # exclude V=0 (I=0 by symmetry)


def fisher_analysis(op, C0, q_w, R_nm, v_biases=FIM_BIASES):
    """2x2 dimensionless sensitivity Jacobian J = d ln|I| / d ln(R, |q_w|) + FIM.

    Uses the non-zero bias series (the V=0 point carries no information). The
    observable is the signed axial current, compared on a log-magnitude scale so
    the Jacobian is dimensionless and comparable across voltages.
    """
    def logabs(q, Rr):
        mu = np.column_stack([v_biases, np.full(v_biases.size, C0),
                              np.full(v_biases.size, q), np.full(v_biases.size, Rr)])
        return np.log(np.maximum(np.abs(op.current(mu)), 1e-12))

    dI_dlnq = (logabs(q_w * (1 + ALPHA), R_nm) - logabs(q_w * (1 - ALPHA), R_nm)) / (2 * ALPHA)
    dI_dlnR = (logabs(q_w, R_nm * (1 + ALPHA)) - logabs(q_w, R_nm * (1 - ALPHA))) / (2 * ALPHA)
    J = np.column_stack([dI_dlnR, dI_dlnq])
    F = (J.T @ J) / (0.01 ** 2)
    ev = np.sort(np.abs(np.linalg.eigvalsh(F)))[::-1]
    cond = float(ev[0] / max(ev[-1], 1e-300))
    return {"J": J.tolist(), "F": F.tolist(),
            "eig": ev.tolist(), "cond": cond, "kappa_q": cond,
            "dlnI_dlnR": float(np.mean(dI_dlnR)),
            "dlnI_dlnq": float(np.mean(dI_dlnq))}


def fisher_plane(op, n=18):
    """Noise-whitened Fisher conditioning over negative q_w and C0 at R=5 nm."""
    from operators.constants import A_LAMBDA_M2
    R_nm = 5.0
    C0_grid = np.logspace(np.log10(200.0), np.log10(20.0), n)
    qw = -np.geomspace(0.002, 0.05, n)
    grid = []
    for c0 in C0_grid:
        lam_d_over_r = float(np.sqrt(A_LAMBDA_M2 / c0) / (R_nm * 1e-9))
        for q in qw:
            fa = fisher_analysis(op, float(c0), float(q), R_nm)
            grid.append({"lambda_D_over_R": lam_d_over_r,
                         "C0": float(c0), "q_w": float(q),
                         "cond": fa["cond"], "kappa_q": fa["cond"],
                         "dlnI_dlnR": fa["dlnI_dlnR"], "dlnI_dlnq": fa["dlnI_dlnq"]})
    return grid

def grid_candidates(q_lo, q_hi, r_lo, r_hi, nq=28, nr=28, seed=20261020):
    rng = np.random.default_rng(seed)
    qs = np.linspace(q_lo, q_hi, nq)
    rs = np.linspace(r_lo, r_hi, nr)
    dq = (q_hi - q_lo) / (nq - 1)
    dr = (r_hi - r_lo) / (nr - 1)
    qs = (qs + rng.uniform(-0.3 * dq, 0.3 * dq, nq)).clip(q_lo, q_hi)
    rs = (rs + rng.uniform(-0.3 * dr, 0.3 * dr, nr)).clip(r_lo, r_hi)
    qq, rr = np.meshgrid(qs, rs, indexing="ij")
    return np.column_stack([qq.ravel(), rr.ravel()]), (q_lo, q_hi, r_lo, r_hi)


def invert(op, protocol, truth, candidates, v_biases, salt_list, box, rng, noise,
           refine=True):
    """Return recovered (q_w, R) for a protocol given noisy synthetic data."""
    from scipy.optimize import minimize
    C0_true, q_true, R_true = truth
    q_lo, q_hi, r_lo, r_hi = box

    def synth_I(C0):
        mu = np.column_stack([v_biases, np.full(v_biases.size, C0),
                              np.full(v_biases.size, q_true), np.full(v_biases.size, R_true)])
        clean = op.current(mu)
        return clean, clean * (1.0 + noise * rng.standard_normal(clean.size))

    def synth_S(C0):
        mu = np.column_stack([v_biases, np.full(v_biases.size, C0),
                              np.full(v_biases.size, q_true), np.full(v_biases.size, R_true)])
        clean = op.wall_selectivity(mu)
        return clean, clean * (1.0 + noise * rng.standard_normal(clean.size))

    salts = salt_list if protocol != "single" else [C0_true]
    meas_I, meas_S = {}, {}
    for C0 in salts:
        _, meas_I[C0] = synth_I(C0)
        if protocol == "dual":
            _, meas_S[C0] = synth_S(C0)

    def cost_qr(qv, rv):
        qv = float(np.clip(qv, q_lo, q_hi)); rv = float(np.clip(rv, r_lo, r_hi))
        c = 0.0
        for C0 in salts:
            mu = np.column_stack([v_biases, np.full(v_biases.size, C0),
                                  np.full(v_biases.size, qv), np.full(v_biases.size, rv)])
            predI = op.current(mu)
            sI = float(np.mean(meas_I[C0] ** 2)) + 1e-12
            c += float(np.mean((predI - meas_I[C0]) ** 2)) / sI
            if protocol == "dual":
                predS = op.wall_selectivity(mu)
                sS = float(np.mean(meas_S[C0] ** 2)) + 1e-12
                c += float(np.mean((predS - meas_S[C0]) ** 2)) / sS
        return c

    qc = np.clip(candidates[:, 0], q_lo, q_hi)
    rc = np.clip(candidates[:, 1], r_lo, r_hi)
    costs = np.zeros(len(qc))
    for C0 in salts:
        mu = np.empty((len(qc) * v_biases.size, 4))
        mu[:, 0] = np.tile(v_biases, len(qc))
        mu[:, 1] = C0
        mu[:, 2] = np.repeat(qc, v_biases.size)
        mu[:, 3] = np.repeat(rc, v_biases.size)
        predI = op.current(mu).reshape(len(qc), v_biases.size)
        sI = float(np.mean(meas_I[C0] ** 2)) + 1e-12
        costs += np.mean((predI - meas_I[C0]) ** 2, axis=1) / sI
        if protocol == "dual":
            predS = op.wall_selectivity(mu).reshape(len(qc), v_biases.size)
            sS = float(np.mean(meas_S[C0] ** 2)) + 1e-12
            costs += np.mean((predS - meas_S[C0]) ** 2, axis=1) / sS
    best = int(np.argmin(costs))
    q_hat, R_hat = float(qc[best]), float(rc[best])
    n_refine = 0
    if refine:
        grid_step = max((q_hi - q_lo) / 40.0, (r_hi - r_lo) / 40.0)
        order = np.argsort(costs)
        base = None
        for j in order[:4]:
            res = minimize(lambda x: cost_qr(x[0], x[1]), [qc[j], rc[j]],
                           method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 300})
            n_refine += int(res.nfev)
            if base is None or res.fun < base.fun:
                base = res
        if base is not None:
            q_hat = float(np.clip(base.x[0], q_lo, q_hi))
            R_hat = float(np.clip(base.x[1], r_lo, r_hi))
    q_rel = abs(q_hat - q_true) / abs(q_true)
    R_rel = abs(R_hat - R_true) / R_true
    n_ev = int(len(candidates) * len(salts) * v_biases.size * (2 if protocol == "dual" else 1)) + n_refine * len(salts) * v_biases.size
    return {"q_hat": q_hat, "R_hat": R_hat, "q_rel_err": q_rel, "R_rel_err": R_rel,
            "n_evals": n_ev}


def run_identifiability(op, onnx_rel):
    data = load_npz(DATA240)
    interp = data["interp_idx"]
    # 15 held-out states spanning lambda_D/R and |q_w|
    lam = data["lambda_D_over_R"][interp]
    order = np.argsort(lam)
    pick = order[np.linspace(0, len(order) - 1, 15).astype(int)]
    state_ids = interp[pick]

    v_biases = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
    candidates, box = grid_candidates(-0.06, 0.0, 2.5, 8.5, nq=30, nr=30)
    salt_multi = [10.0, 30.0, 100.0]
    noise_levels = [0.002, 0.010, 0.020]

    protocols = {"single": None, "multi": salt_multi, "dual": salt_multi}
    records = []
    t0 = time.perf_counter()
    for sid in state_ids:
        C0_true = float(data["mu"][sid, 1])
        q_true = float(data["mu"][sid, 2])
        R_true = float(data["mu"][sid, 3])
        for noise in noise_levels:
            for proto, salts in protocols.items():
                rng = np.random.default_rng(1000 + sid + int(noise * 10000) +
                                            {"single": 0, "multi": 1, "dual": 2}[proto])
                r = invert(op, proto, (C0_true, q_true, R_true), candidates, v_biases,
                           salts if salts else [C0_true], box, rng, noise)
                r.update(state_index=int(sid), C0=C0_true, q_true=q_true, R_true=R_true,
                         lambda_D_over_R=float(data["lambda_D_over_R"][sid]),
                         noise=noise, protocol=proto)
                records.append(r)
    wall = time.perf_counter() - t0

    def med(key, pred):
        vals = [r[key] for r in records if pred(r)]
        return float(np.median(vals)) if vals else float("nan")

    summary = {}
    for proto in protocols:
        for noise in noise_levels:
            pred = lambda r, p=proto, n=noise: r["protocol"] == p and r["noise"] == n
            summary[f"{proto}_noise{noise}"] = {
                "median_q_rel_err": med("q_rel_err", pred),
                "median_R_rel_err": med("R_rel_err", pred),
                "q_identifiable_10pct": bool(med("q_rel_err", pred) < 0.10),
                "R_recovered_2pct": bool(med("R_rel_err", pred) < 0.02),
            }
    total_evals = int(sum(r["n_evals"] for r in records))
    # sensitivity of the single-salt conductance to q_w vs. bulk concentration
    salt_sens = {}
    for c in [10.0, 30.0, 100.0, 200.0]:
        vals = []
        for q in [-0.05, -0.03, -0.01, 0.0]:
            mu = np.column_stack([v_biases, np.full(v_biases.size, c),
                                  np.full(v_biases.size, q), np.full(v_biases.size, 5.0)])
            vals.append(float(np.mean(np.abs(op.current(mu)))))
        salt_sens[str(c)] = {"rel_spread_absI": float((max(vals) - min(vals)) / np.mean(vals))}
    result = {
        "salt_sensitivity": salt_sens,
        "protocol": {
            "v_biases_mV": v_biases.tolist(),
            "salt_multi_mol_m3": salt_multi,
            "noise_levels": noise_levels,
            "n_states": int(len(state_ids)),
            "state_ids": state_ids.tolist(),
            "candidates": int(len(candidates)),
            "operator": onnx_rel,
        },
        "summary": summary,
        "cost": {"total_forward_evals": total_evals,
                 "operator_wall_s": wall,
                 "sgfv_equivalent_wall_s": total_evals * 0.2013},
        "records": records,
    }
    return result


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def write_tables(ablation, scaling, ident, fisher_grid):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "table_s14_radial_trunk_and_scaling_ablation.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "config_or_Ntrain", "metric", "split", "mean", "sd", "median", "n"])
        for name in ABLATION_CONFIGS:
            s = ablation["summary"][name]
            for k in ["rel_l2_rho", "nw_rel_l2_rho", "rel_l2_psi", "nw_rel_l2_psi",
                      "wall_deriv_err", "i_rel_l2"]:
                for split in ["interp", "ood"]:
                    key = f"{split}_{k}"
                    if key in s:
                        w.writerow(["radial_trunk", name, k, split,
                                    f"{s[key]['mean']:.8g}", f"{s[key]['sd']:.6g}",
                                    f"{s[key]['median']:.8g}", s[key]['n']])
        for n in scaling["n_list"]:
            s = scaling["summary"][n]
            for k in ["interp_rho", "interp_psi", "interp_i", "ood_rho", "ood_psi", "ood_i",
                      "nw_interp_rho"]:
                if k in s:
                    w.writerow(["scaling", n, k, "interp" if k.startswith("interp") else "ood",
                                f"{s[k]['mean']:.8g}", f"{s[k]['sd']:.6g}",
                                f"{s[k]['median']:.8g}", s[k]['n']])
    with open(os.path.join(TABLES, "table_s15_fisher_identifiability_protocols.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "noise", "median_q_rel_err", "median_R_rel_err",
                    "q_identifiable_10pct", "R_recovered_2pct"])
        for key, s in ident["summary"].items():
            proto, noise = key.rsplit("_noise", 1)
            w.writerow([proto, noise, f"{s['median_q_rel_err']:.6g}",
                        f"{s['median_R_rel_err']:.6g}",
                        s["q_identifiable_10pct"], s["R_recovered_2pct"]])
    # Fisher plane summary (binned)
    with open(os.path.join(TABLES, "table_s15b_fisher_plane.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda_D_over_R", "C0_mol_m3", "q_w", "cond_FIM", "dlnI_dlnR", "dlnI_dlnq"])
        for g in fisher_grid:
            w.writerow([f"{g['lambda_D_over_R']:.6g}", f"{g['C0']:.6g}", f"{g['q_w']:.6g}",
                        f"{g['cond']:.6g}", f"{g['dlnI_dlnR']:.6g}", f"{g['dlnI_dlnq']:.6g}"])


def merge_statistics(ablation, scaling, ident, fisher):
    with open(STATS) as fh:
        stats = json.load(fh)
    stats["stage10_radial_trunk_ablation"] = ablation
    stats["stage10_scaling_law"] = scaling
    stats["stage10_fisher_identifiability"] = ident
    stats["stage10_fisher_plane"] = fisher
    stats["stage"] = 10
    with open(STATS, "w") as fh:
        json.dump(stats, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-ablation", type=int, default=250)
    ap.add_argument("--epochs-scaling", type=int, default=250)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore per-experiment cache")
    args = ap.parse_args()
    if args.quick:
        args.epochs_ablation = 60
        args.epochs_scaling = 60

    def cache_path(tag, epochs):
        return os.path.join(ROOT, "results", f"_stage10_cache_{tag}_{epochs}.json")

    def cached(tag, epochs):
        p = cache_path(tag, epochs)
        if not args.force and os.path.exists(p):
            with open(p) as fh:
                print(f"[cache] reusing {os.path.basename(p)}", flush=True)
                return json.load(fh)
        return None

    def store(tag, epochs, obj):
        with open(cache_path(tag, epochs), "w") as fh:
            json.dump(obj, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        return obj

    out = {"stage": 10,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ablation = scaling = ident = fisher = None
    if not args.skip_a:
        ablation = cached("A", args.epochs_ablation)
        if ablation is None:
            print("=== Experiment A: radial-trunk ablation ===", flush=True)
            ablation = store("A", args.epochs_ablation, run_ablation(args.epochs_ablation))
        out["radial_trunk_ablation"] = ablation
    if not args.skip_b:
        scaling = cached("B", args.epochs_scaling)
        if scaling is None:
            print("=== Experiment B: 384-state expansion + scaling law ===", flush=True)
            generate_384()
            scaling = store("B", args.epochs_scaling, run_scaling(args.epochs_scaling))
        out["scaling_law"] = scaling
    if not args.skip_c:
        payload = cached("C", 0)
        if payload is None:
            print("=== Experiment C: Fisher identifiability ===", flush=True)
            onnx_path = os.path.join(ROOT, "web_simulator", "models", "deeponet_reparam_seed42.onnx")
            op = OnnxOperator(onnx_path)
            payload = store("C", 0, {"fisher_plane": fisher_plane(op),
                                     "fisher_identifiability": run_identifiability(
                                         op, os.path.relpath(onnx_path, ROOT))})
        fisher = payload["fisher_plane"]
        ident = payload["fisher_identifiability"]
        out["fisher_plane"] = fisher
        out["fisher_identifiability"] = ident

    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if all(x is not None for x in (ablation, scaling, ident, fisher)):
        write_tables(ablation, scaling, ident, fisher)
        merge_statistics(ablation, scaling, ident, fisher)
    print(f"[stage10] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
