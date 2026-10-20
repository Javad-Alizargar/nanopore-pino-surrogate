"""Stage 4: multi-seed training, H1-H4 evaluation, ONNX export, canonical stats.

Deterministic, local, zero paid LLM calls. Produces:
  results/statistics.json
  results/tables/table1_dataset_and_solver_validation.csv
  results/tables/table2_benchmark_state_vs_dapinn.csv
  results/tables/table3_multiseed_h1_h2_h3_ablation.csv
  results/tables/table4_timing_and_amortization.csv
  results/tables/per_state_test_metrics.csv
  results/models/*.pt, results/models/*.onnx
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from operators import build_model, feature_stats  # noqa: E402
from operators.boundary_ansatz import (  # noqa: E402
    reconstruct_concentrations,
    salt_mean,
)
from operators.constants import (  # noqa: E402
    FARADAY,
    NR,
    NZ,
    annular_weights,
    make_grid,
)
from operators.losses_and_metrics import (  # noqa: E402
    GridTensors,
    continuity_residual,
    pnp_physics_loss,
    poisson_residual,
    sg_current_pA,
    wall_bc_residual,
    wall_flux_residual,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/processed/pnp_parametric_240.npz")
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
MODELS = os.path.join(RESULTS, "models")

SEEDS = [42, 43, 44]
COMBOS = [("fno", "reparam"), ("fno", "direct_c"),
          ("deeponet", "reparam"), ("deeponet", "direct_c")]
BENCHMARK_CURRENT_PA = 196.137225
PRIOR_DAPINN = {
    "psi_rel_l2": 3.82e-4,
    "c_plus_rel_l2": 1.06e-4,
    "c_minus_rel_l2": 1.06e-4,
    "rho_rel_l2": 0.145,
    "current_profile_rel_l2": 1.02e-3,
    "retrain_s_per_state_approx": 500.0,
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data():
    d = np.load(DATA, allow_pickle=True)
    mu = d["mu"].astype(np.float64)
    ref = {k: d[k].astype(np.float64) for k in
           ["psi", "c_plus", "c_minus", "rho", "sigma_c", "i_z"]}
    splits = d["splits"].astype(str)
    train_idx = np.where(splits == "train")[0]
    interp_idx = np.where(splits == "interp_test")[0]
    ood_idx = np.where(splits == "ood_test")[0]
    oodA = np.array([i for i in ood_idx if 192 <= i < 216])  # charge/salt
    oodB = np.array([i for i in ood_idx if 216 <= i < 240])  # geometry/voltage
    return {
        "mu": mu, "ref": ref, "splits": splits,
        "lambda_D_over_R": d["lambda_D_over_R"].astype(np.float64),
        "train_idx": train_idx, "interp_idx": interp_idx,
        "ood_idx": ood_idx, "oodA": oodA, "oodB": oodB,
        "bench_idx": int(interp_idx[0]),
        "n_continuation_steps": d["n_continuation_steps"],
    }


def build_targets(ref):
    cp, cm = ref["c_plus"], ref["c_minus"]
    return {
        "psi": ref["psi"],
        "c_plus": cp,
        "c_minus": cm,
        "rho": cp - cm,
        "s": 2.0 * np.sqrt(cp * cm),
        "sigma_c": cp + cm,
    }


def per_state_scales(targets, fit_idx, fields, floor=0.1):
    scales = {}
    for f in fields:
        arr = targets[f][fit_idx]
        rms = np.sqrt((arr ** 2).mean(axis=(1, 2)))
        g = math.sqrt((arr ** 2).mean())
        scales[f] = np.maximum(rms, floor * g)
    return scales


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def size_kwargs(kind, args):
    if kind == "fno":
        return dict(width=args.fno_width, modes_r=args.fno_modes_r,
                    modes_z=args.fno_modes_z, n_layers=4)
    return dict(p=args.deeponet_p)


def eval_loss(model, idx, fields, scales, mu, targets):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bidx = idx[s:s + 64]
            mu_b = torch.tensor(mu[bidx], dtype=torch.float32)
            out = model(mu_b)
            for f in fields:
                tgt = torch.tensor(targets[f][bidx], dtype=torch.float32)
                sc = torch.tensor(scales[f][bidx], dtype=torch.float32).reshape(-1, 1, 1)
                tot += (((out[f] - tgt) / sc) ** 2).mean().item() * len(bidx)
    return tot / len(idx)


def train_run(kind, mode, seed, physics, epochs, args, data, targets, grid, stats):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fields = ["psi", "s", "rho"] if mode == "reparam" else ["psi", "c_plus", "c_minus"]
    fit_idx = data["fit_idx"]
    val_idx = data["val_idx"]
    scales = per_state_scales(targets, data["train_idx"], fields)

    model = build_model(kind, mode, stats, **size_kwargs(kind, args))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mu = data["mu"]
    best_val, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(fit_idx)
        ep_loss, nb = 0.0, 0
        for st in range(0, len(perm), args.batch):
            bidx = perm[st:st + args.batch]
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
                pl = pnp_physics_loss(out, mu_b, grid)
                loss = loss + args.phys_w * (pl["poisson"] + pl["continuity"] + pl["wall"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
        vl = eval_loss(model, val_idx, fields, scales, mu, targets)
        hist.append((ep, ep_loss / nb, vl))
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, {"train_loss_first": hist[0][1], "train_loss_last": hist[-1][1],
                   "val_loss_best": best_val, "epochs": epochs,
                   "train_time_s": train_time, "fields": fields}


# --------------------------------------------------------------------------- #
# prediction + metrics
# --------------------------------------------------------------------------- #
def predict(model, idx, mu):
    model.eval()
    outs = {}
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bidx = idx[s:s + 64]
            mu_b = torch.tensor(mu[bidx], dtype=torch.float32)
            o = model(mu_b)
            for k in ["psi", "c_plus", "c_minus", "rho", "s"]:
                outs.setdefault(k, []).append(o[k].numpy().astype(np.float64))
    pred = {k: np.concatenate(v, 0) for k, v in outs.items()}
    pred["sigma_c"] = pred["c_plus"] + pred["c_minus"]
    return pred


def _area():
    r, z, *_ = make_grid(NR, NZ)
    _, _, _, _, _, dz = make_grid(NR, NZ)
    return annular_weights(r)[:, None] * dz


AREA = _area()


def rel_l2_per_state(pred, ref):
    num = np.sqrt(((pred - ref) ** 2 * AREA).sum(axis=(1, 2)))
    den = np.sqrt((ref ** 2 * AREA).sum(axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = num / den
    rel[den < 1e-9 * (AREA.sum())] = np.nan
    return rel


def global_rel_l2(pred, ref):
    num = float(np.sqrt(((pred - ref) ** 2 * AREA).sum()))
    den = float(np.sqrt((ref ** 2 * AREA).sum()))
    return num / max(den, 1e-300)


def rms_per_state(pred, ref):
    return np.sqrt(((pred - ref) ** 2).mean(axis=(1, 2)))


def current_from_pred(pred, idx, mu, grid):
    cp = torch.tensor(pred["c_plus"], dtype=torch.float32)
    cm = torch.tensor(pred["c_minus"], dtype=torch.float32)
    psi = torch.tensor(pred["psi"], dtype=torch.float32)
    mu_b = torch.tensor(mu[idx], dtype=torch.float32)
    with torch.no_grad():
        return sg_current_pA(cp, cm, psi, mu_b, grid).numpy().astype(np.float64)


def physical_consistency(pred, idx, mu, grid):
    cp = torch.tensor(pred["c_plus"], dtype=torch.float32)
    cm = torch.tensor(pred["c_minus"], dtype=torch.float32)
    psi = torch.tensor(pred["psi"], dtype=torch.float32)
    mu_b = torch.tensor(mu[idx], dtype=torch.float32)
    with torch.no_grad():
        q_w = mu_b[:, 2].reshape(-1, 1)
        dpsi_dr = (psi[:, -1, :] - psi[:, -2, :]) / grid.dr
        wall_deriv_err = (dpsi_dr - q_w).abs().mean(dim=1).numpy()
        wall_flux = (wall_flux_residual(cp, psi, grid, 1).abs().mean(dim=1)
                     + wall_flux_residual(cm, psi, grid, -1).abs().mean(dim=1)).numpy()
        poi = poisson_residual(psi, cp, cm, mu_b, grid).abs().mean(dim=(1, 2)).numpy()
        con = (0.5 * (continuity_residual(cp, psi, mu_b, grid, 1).abs().mean(dim=(1, 2))
                      + continuity_residual(cm, psi, mu_b, grid, -1).abs().mean(dim=(1, 2)))).numpy()
    return {"wall_deriv_err": wall_deriv_err, "wall_flux_residual": wall_flux,
            "poisson_residual": poi, "continuity_residual": con}


def evaluate_model(model, idx, data, targets, grid, with_consistency=True):
    pred = predict(model, idx, data["mu"])
    ref = data["ref"]
    out, glob = {}, {}
    for f in ["psi", "c_plus", "c_minus", "rho", "sigma_c"]:
        out[f"rel_l2_{f}"] = rel_l2_per_state(pred[f], ref[f][idx])
        out[f"rms_{f}"] = rms_per_state(pred[f], ref[f][idx])
        glob[f"grel_l2_{f}"] = global_rel_l2(pred[f], ref[f][idx])
        glob[f"rms_{f}_mean"] = float(out[f"rms_{f}"].mean())
    ipred = current_from_pred(pred, idx, data["mu"], grid)
    iref = ref["i_z"][idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["i_rel_l2"] = (np.sqrt(((ipred - iref) ** 2).mean(axis=1))
                           / np.sqrt((iref ** 2).mean(axis=1)))
    out["i_rel_l2"][~np.isfinite(out["i_rel_l2"])] = np.nan
    out["i_abs_err_pA"] = np.abs(ipred - iref).mean(axis=1)
    out["i_mean_pred_pA"] = ipred.mean(axis=1)
    out["i_mean_ref_pA"] = iref.mean(axis=1)
    glob["grel_l2_i"] = global_rel_l2(ipred[:, None, :], iref[:, None, :])
    glob["i_mean_pred_pA"] = float(ipred.mean())
    glob["i_mean_ref_pA"] = float(iref.mean())
    glob["i_abs_err_pA_mean"] = float(out["i_abs_err_pA"].mean())
    if with_consistency:
        cons = physical_consistency(pred, idx, data["mu"], grid)
        out.update(cons)
        for k, v in cons.items():
            glob[k + "_mean"] = float(np.nanmean(v))
    return out, glob


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=float("nan"), median=float("nan"), iqr=float("nan"),
                    sd=float("nan"), n=0)
    return dict(mean=float(arr.mean()), median=float(np.median(arr)),
                iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0, n=int(arr.size))


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def export_onnx(model, path, nr=NR, nz=NZ):
    model.eval()
    dummy = torch.zeros(1, 4, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path, input_names=["mu"],
        output_names=["psi", "c_plus", "c_minus", "rho", "s"],
        dynamic_axes={"mu": {0: "B"}, "psi": {0: "B"}, "c_plus": {0: "B"},
                      "c_minus": {0: "B"}, "rho": {0: "B"}, "s": {0: "B"}},
        opset_version=17, dynamo=False, do_constant_folding=True)
    return path


def onnx_parity(model, path, idx, mu):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    max_diff = 0.0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), 32):
            bidx = idx[s:s + 32]
            mu_b = torch.tensor(mu[bidx], dtype=torch.float32)
            o = model(mu_b)
            res = sess.run(names, {"mu": mu_b.numpy()})
            for name, r in zip(names, res):
                d = np.abs(o[name].numpy() - r).max()
                max_diff = max(max_diff, float(d))
    return max_diff


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def make_runs(seeds):
    runs = []
    for seed in seeds:
        for kind, mode in COMBOS:
            runs.append(dict(kind=kind, mode=mode, seed=seed,
                             condition="data_plus_physics", physics=True))
    for seed in seeds:
        for kind in ("fno", "deeponet"):
            runs.append(dict(kind=kind, mode="reparam", seed=seed,
                             condition="data_only", physics=False))
    return runs


def run_tag(r):
    return f"{r['kind']}_{r['mode']}_{r['condition']}_seed{r['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs-fno", type=int, default=110)
    ap.add_argument("--epochs-deeponet", type=int, default=500)
    ap.add_argument("--batch", type=int, default=36)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--phys-w", type=float, default=1e-3)
    ap.add_argument("--fno-width", type=int, default=24)
    ap.add_argument("--fno-modes-r", type=int, default=8)
    ap.add_argument("--fno-modes-z", type=int, default=12)
    ap.add_argument("--deeponet-p", type=int, default=64)
    args = ap.parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs_fno = 25
        args.epochs_deeponet = 40

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    data = load_data()
    targets = build_targets(data["ref"])
    grid = GridTensors()
    stats = feature_stats(data["mu"][data["train_idx"]])

    # algebraic identity check for reparam s target
    cp = torch.tensor(data["ref"]["c_plus"], dtype=torch.float64)
    cm = torch.tensor(data["ref"]["c_minus"], dtype=torch.float64)
    s = salt_mean(cp, cm)
    rho = cp - cm
    cpr, cmr = reconstruct_concentrations(s, rho)
    identity_err = max(float((cpr - cp).abs().max()), float((cmr - cm).abs().max()))

    val_idx = data["train_idx"][::6]
    data["val_idx"] = val_idx
    data["fit_idx"] = np.array([i for i in data["train_idx"] if i not in set(val_idx)])

    runs = make_runs(args.seeds)
    trained = {}
    run_records = []
    print(f"[suite] runs={len(runs)} quick={args.quick} "
          f"identity_err={identity_err:.2e}", flush=True)
    for r in runs:
        kind, mode = r["kind"], r["mode"]
        epochs = args.epochs_fno if kind == "fno" else args.epochs_deeponet
        model, info = train_run(kind, mode, r["seed"], r["physics"], epochs,
                                args, data, targets, grid, stats)
        tag = run_tag(r)
        trained[tag] = model
        torch.save({"state_dict": model.state_dict(), "kind": kind, "mode": mode,
                    "seed": r["seed"], "condition": r["condition"],
                    "stats": stats, "size": size_kwargs(kind, args)}, 
                   os.path.join(MODELS, tag + ".pt"))
        rec = dict(r); rec.update(info)
        run_records.append(rec)
        print(f"[run] {tag:46s} val={info['val_loss_best']:.4e} "
              f"t={info['train_time_s']:.1f}s", flush=True)

    # -------- evaluation --------
    splits = {
        "train": data["train_idx"], "interp_test": data["interp_idx"],
        "ood_test": data["ood_idx"], "ood_charge_salt": data["oodA"],
        "ood_geometry_voltage": data["oodB"],
    }
    per_run_metrics = {}
    per_run_global = {}
    per_state_rows = []
    for r in runs:
        tag = run_tag(r)
        model = trained[tag]
        m, g = {}, {}
        for sp, idx in splits.items():
            m[sp], g[sp] = evaluate_model(model, idx, data, targets, grid,
                                          with_consistency=True)
        # benchmark state
        m["benchmark"], g["benchmark"] = evaluate_model(
            model, np.array([data["bench_idx"]]), data, targets, grid, with_consistency=True)
        per_run_metrics[tag] = m
        per_run_global[tag] = g
        for sp in ["interp_test", "ood_test"]:
            idx = splits[sp]
            for j, gi in enumerate(idx):
                per_state_rows.append(dict(
                    run=tag, kind=r["kind"], mode=r["mode"], condition=r["condition"],
                    seed=r["seed"], split=sp, global_index=int(gi),
                    V_mV=float(data["mu"][gi, 0]), C0=float(data["mu"][gi, 1]),
                    q_w=float(data["mu"][gi, 2]), R_nm=float(data["mu"][gi, 3]),
                    lambda_D_over_R=float(data["lambda_D_over_R"][gi]),
                    i_mean_ref_pA=float(m[sp]["i_mean_ref_pA"][j]),
                    rel_l2_psi=float(m[sp]["rel_l2_psi"][j]),
                    rel_l2_c_plus=float(m[sp]["rel_l2_c_plus"][j]),
                    rel_l2_rho=float(m[sp]["rel_l2_rho"][j]),
                    rel_l2_i=float(m[sp]["i_rel_l2"][j]),
                    i_abs_err_pA=float(m[sp]["i_abs_err_pA"][j]),
                    wall_deriv_err=float(m[sp]["wall_deriv_err"][j]),
                ))
        print(f"[eval] {tag} done", flush=True)

    # -------- aggregate across seeds --------
    def agg(kind, mode, condition, split, metric):
        vals = []
        for r in runs:
            if (r["kind"], r["mode"], r["condition"]) == (kind, mode, condition):
                arr = per_run_metrics[run_tag(r)][split].get(metric)
                if arr is not None:
                    vals.append(np.nanmean(arr))
        return summarize(vals)

    def agg_global(kind, mode, condition, split, metric):
        vals = []
        for r in runs:
            if (r["kind"], r["mode"], r["condition"]) == (kind, mode, condition):
                glob = per_run_global[run_tag(r)][split].get(metric)
                if glob is not None:
                    vals.append(glob)
        return summarize(vals)

    metric_names = ["rel_l2_psi", "rel_l2_c_plus", "rel_l2_c_minus", "rel_l2_rho",
                    "rel_l2_sigma_c", "rel_l2_i", "i_abs_err_pA",
                    "rms_psi", "rms_c_plus", "rms_c_minus", "rms_rho",
                    "wall_deriv_err", "wall_flux_residual", "poisson_residual",
                    "continuity_residual"]
    combos_conditions = [(k, mo, "data_plus_physics") for k, mo in COMBOS] + \
                        [("fno", "reparam", "data_only"), ("deeponet", "reparam", "data_only")]
    all_splits = ["train", "interp_test", "ood_test", "ood_charge_salt",
                  "ood_geometry_voltage", "benchmark"]
    summary, summary_global = {}, {}
    for k, mo, cond in combos_conditions:
        key = f"{k}_{mo}_{cond}"
        summary[key] = {sp: {mn: agg(k, mo, cond, sp, mn) for mn in metric_names}
                        for sp in all_splits}
        summary_global[key] = {sp: {mn: agg_global(k, mo, cond, sp, mn)
                                    for mn in per_run_global[run_tag(runs[0])][sp].keys()}
                               for sp in all_splits}

    # -------- H3 stratification (reparam vs direct_c) --------
    h3 = build_h3(runs, per_run_metrics, data)

    # -------- ONNX export (best seed-42 reparam) --------
    onnx_info = build_onnx(trained, data)

    # -------- H4 timing + amortization --------
    h4 = build_h4(trained, run_records, data, grid, args)

    # -------- write statistics.json --------
    statistics = {
        "stage": 4,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args),
        "reparam_identity_max_abs_err": identity_err,
        "prior_dapinn_reference": PRIOR_DAPINN,
        "benchmark_current_reference_pA": BENCHMARK_CURRENT_PA,
        "run_records": run_records,
        "summary_multiseed": summary,
        "summary_multiseed_global_rel_l2": summary_global,
        "h3_conditioning": h3,
        "h4_timing_amortization": h4,
        "onnx_export": onnx_info,
    }
    with open(os.path.join(RESULTS, "statistics.json"), "w") as fh:
        json.dump(statistics, fh, indent=2)

    write_tables(data, summary, summary_global, per_state_rows, h4, onnx_info, run_records)
    print("[suite] DONE", flush=True)


def build_h3(runs, per_run_metrics, data):
    """Net-charge accuracy of reparam vs direct_c, stratified by |q_w| and lamD/R."""
    lam = data["lambda_D_over_R"]
    idx = np.concatenate([data["interp_idx"], data["ood_idx"]])
    out = {}
    for kind in ("fno", "deeponet"):
        for cond in ("data_plus_physics",):
            row = {}
            for mode in ("reparam", "direct_c"):
                seeds = [r for r in runs if r["kind"] == kind and r["mode"] == mode
                         and r["condition"] == cond]
                if not seeds:
                    continue
                vals = np.nanmean([np.concatenate([per_run_metrics[run_tag(r)]["interp_test"]["rel_l2_rho"],
                                                   per_run_metrics[run_tag(r)]["ood_test"]["rel_l2_rho"]])
                                   for r in seeds], axis=0)
                qw = np.abs(data["mu"][idx, 2])
                row[mode] = {
                    "overall_mean": float(np.nanmean(vals)),
                    "by_abs_qw_quartile": _binned(vals, qw),
                    "by_lambdaD_over_R_quartile": _binned(vals, lam[idx]),
                }
            out[f"{kind}_{cond}"] = row
    return out


def _binned(vals, x, edges=(0.0, 0.25, 0.5, 0.75, 1.0)):
    qs = np.quantile(x, edges)
    res = {}
    for a, b in zip(qs[:-1], qs[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() > 0:
            res[f"[{a:.4g},{b:.4g}]"] = float(np.nanmean(vals[m]))
    return res


def build_h4(trained, run_records, data, grid, args):
    import onnxruntime as ort
    inf = {}
    for kind in ("fno", "deeponet"):
        tag = f"{kind}_reparam_data_plus_physics_seed42"
        model = trained.get(tag)
        if model is None:
            continue
        model.eval()
        mu1 = torch.zeros(1, 4, dtype=torch.float32)
        mu48 = torch.zeros(48, 4, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(5):
                model(mu1); model(mu48)
            a0 = time.perf_counter()
            for _ in range(50):
                model(mu1)
            a1 = time.perf_counter()
            for _ in range(50):
                model(mu48)
            a2 = time.perf_counter()
        ort_t1 = ort_t48 = float("nan")
        onnx_path = os.path.join(MODELS, f"{kind}_reparam_seed42.onnx")
        if os.path.exists(onnx_path):
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feed1, feed48 = {"mu": mu1.numpy()}, {"mu": mu48.numpy()}
            for _ in range(5):
                sess.run(None, feed1); sess.run(None, feed48)
            b0 = time.perf_counter()
            for _ in range(50):
                sess.run(None, feed1)
            b1 = time.perf_counter()
            for _ in range(50):
                sess.run(None, feed48)
            b2 = time.perf_counter()
            ort_t1 = (b1 - b0) / 50 * 1e3
            ort_t48 = (b2 - b1) / 50 * 1e3
        inf[kind] = {
            "torch_f32_batch1_ms": (a1 - a0) / 50 * 1e3,
            "torch_f32_batch48_ms": (a2 - a1) / 50 * 1e3,
            "onnx_f32_batch1_ms": ort_t1,
            "onnx_f32_batch48_ms": ort_t48,
        }

    # SG current post-processing time on batch 48
    ref = data["ref"]
    idx = data["interp_idx"][:48]
    cp = torch.tensor(ref["c_plus"][idx], dtype=torch.float32)
    cm = torch.tensor(ref["c_minus"][idx], dtype=torch.float32)
    psi = torch.tensor(ref["psi"][idx], dtype=torch.float32)
    mu_b = torch.tensor(data["mu"][idx], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(5):
            sg_current_pA(cp, cm, psi, mu_b, grid)
        c0 = time.perf_counter()
        for _ in range(20):
            sg_current_pA(cp, cm, psi, mu_b, grid)
        c1 = time.perf_counter()
    inf["sg_current_postprocess_ms_batch48"] = (c1 - c0) / 20 * 1e3

    # amortization
    t_sgfv = {"stage2_median_s": 0.2013, "dataset_mean_s": 0.2391, "prior_xeon_s": 0.44}
    t_dataset_gen = 59.18

    def _train_time(kind):
        for rec in run_records:
            if rec["kind"] == kind and rec["mode"] == "reparam" \
                    and rec["condition"] == "data_plus_physics" and rec["seed"] == 42:
                return rec["train_time_s"]
        return float("nan")

    amort = {}
    for kind, d in inf.items():
        if not isinstance(d, dict):
            continue
        t_inf = d["onnx_f32_batch1_ms"] / 1e3
        t_train = _train_time(kind)
        for ref_name, ts in t_sgfv.items():
            amort[f"{kind}_{ref_name}"] = {
                "t_dataset_gen_s": t_dataset_gen, "t_train_s": t_train,
                "t_infer_s": t_inf, "t_sgfv_s": ts,
                "n_break_even": (t_dataset_gen + t_train) / (ts - t_inf)}
        amort[f"{kind}_vs_dapinn_retrain"] = {
            "t_train_s": t_train, "t_infer_s": t_inf,
            "t_dapinn_retrain_s_per_state": PRIOR_DAPINN["retrain_s_per_state_approx"],
            "n_break_even": (t_dataset_gen + t_train) /
                            (PRIOR_DAPINN["retrain_s_per_state_approx"] - t_inf)}
    return {"inference_ms": inf, "amortization": amort}


def build_onnx(trained, data):
    out = {}
    for kind in ("fno", "deeponet"):
        tag = f"{kind}_reparam_data_plus_physics_seed42"
        model = trained.get(tag)
        if model is None:
            continue
        path = os.path.join(MODELS, f"{kind}_reparam_seed42.onnx")
        export_onnx(model, path)
        idx = np.concatenate([data["interp_idx"], data["ood_idx"]])
        md = onnx_parity(model, path, idx, data["mu"])
        out[kind] = {"path": os.path.relpath(path, ROOT), "max_abs_diff": md,
                     "parity_ok": md < 1e-5, "states_checked": int(len(idx))}
    return out


def write_tables(data, summary, summary_global, per_state_rows, h4, onnx_info, run_records):
    import csv
    # table1
    with open(os.path.join(TABLES, "table1_dataset_and_solver_validation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        w.writerow(["n_states", 240])
        w.writerow(["n_train", 144]); w.writerow(["n_interp_test", 48])
        w.writerow(["n_ood_charge_salt", 24]); w.writerow(["n_ood_geometry_voltage", 24])
        w.writerow(["max_residual", 9.936325096049359e-10])
        w.writerow(["min_c_plus", 0.9999474287033081]); w.writerow(["min_c_minus", 0.8033685088157654])
        w.writerow(["n_continuation_steps", 0])
        w.writerow(["benchmark_current_pA", BENCHMARK_CURRENT_PA])
        w.writerow(["sgfv_median_s", 0.2013]); w.writerow(["sgfv_mean_s", 0.2391])
    # table2 benchmark (global relative L2 vs SG/FV solver; DA-PINN reference row)
    rows = []
    for k, mo, cond in [("fno", "reparam", "data_plus_physics"),
                        ("deeponet", "reparam", "data_plus_physics"),
                        ("fno", "direct_c", "data_plus_physics"),
                        ("deeponet", "direct_c", "data_plus_physics")]:
        key = f"{k}_{mo}_{cond}"
        b = summary_global[key]["benchmark"]
        rows.append(dict(model=f"{k}_{mo}", source="operator(Stage4)",
                         current_mean_pA=round(b["i_mean_pred_pA"]["mean"], 6),
                         current_ref_pA=BENCHMARK_CURRENT_PA,
                         psi_rel_l2=b["grel_l2_psi"]["mean"],
                         c_plus_rel_l2=b["grel_l2_c_plus"]["mean"],
                         c_minus_rel_l2=b["grel_l2_c_minus"]["mean"],
                         rho_rel_l2=b["grel_l2_rho"]["mean"],
                         i_rel_l2=b["grel_l2_i"]["mean"]))
    rows.append(dict(model="prior_DA-PINN(fixed-state)", source="COMPHY-D-26-00857(paper)",
                     current_mean_pA=BENCHMARK_CURRENT_PA, current_ref_pA=BENCHMARK_CURRENT_PA,
                     psi_rel_l2=PRIOR_DAPINN["psi_rel_l2"], c_plus_rel_l2=PRIOR_DAPINN["c_plus_rel_l2"],
                     c_minus_rel_l2=PRIOR_DAPINN["c_minus_rel_l2"], rho_rel_l2=PRIOR_DAPINN["rho_rel_l2"],
                     i_rel_l2=PRIOR_DAPINN["current_profile_rel_l2"]))
    with open(os.path.join(TABLES, "table2_benchmark_state_vs_dapinn.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    # table3 multiseed (global + per-state rel L2), H1/H2/H3 + physics ablation
    with open(os.path.join(TABLES, "table3_multiseed_h1_h2_h3_ablation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "mode", "condition", "split", "metric", "mean", "sd", "median", "iqr"])
        for k, mo, cond in [(kk, mm, "data_plus_physics") for kk, mm in COMBOS] + \
                           [("fno", "reparam", "data_only"), ("deeponet", "reparam", "data_only")]:
            key = f"{k}_{mo}_{cond}"
            for sp in ["train", "interp_test", "ood_test", "ood_charge_salt",
                       "ood_geometry_voltage", "benchmark"]:
                for mn in ["grel_l2_psi", "grel_l2_c_plus", "grel_l2_c_minus",
                           "grel_l2_rho", "grel_l2_sigma_c", "grel_l2_i",
                           "i_abs_err_pA_mean", "wall_deriv_err_mean",
                           "wall_flux_residual_mean", "poisson_residual_mean",
                           "continuity_residual_mean"]:
                    s = summary_global[key][sp][mn]
                    w.writerow([k, mo, cond, sp, mn, s["mean"], s["sd"], s["median"], s["iqr"]])
    # table4 timing
    with open(os.path.join(TABLES, "table4_timing_and_amortization.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "value", "unit"])
        for kind, d in h4["inference_ms"].items():
            if not isinstance(d, dict):
                w.writerow([f"shared.{kind}", d, "ms"])
                continue
            for kk, vv in d.items():
                w.writerow([f"{kind}.{kk}", vv, "ms"])
        for kk, vv in h4["amortization"].items():
            w.writerow([f"breakeven.{kk}", vv["n_break_even"], "states"])
    # per-state
    if per_state_rows:
        with open(os.path.join(TABLES, "per_state_test_metrics.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_state_rows[0].keys()))
            w.writeheader(); [w.writerow(r) for r in per_state_rows]


if __name__ == "__main__":
    main()
