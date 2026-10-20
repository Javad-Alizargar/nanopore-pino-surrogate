"""Stage 11 reviewer-upgrade experiments (deterministic, local, $0 paid spend).

Addresses three priorities of the external peer-review assessment:

  Priority 3 - Exact SG/FV finite-difference Jacobian and Fisher-matrix
      validation (`fisher_sgfv_validation`): centered finite differences
      (relative step 1e-3) of the float64 Scharfetter--Gummel solver are
      compared against the neural-operator Jacobian over 15 held-out states and
      three measurement protocols (single-salt I-V, multi-salt I-V, and the
      dual I-V + near-wall selectivity observable).
  Priority 2 - Four-way hard/soft constraint-scheme benchmark
      (`constraint_schemes_benchmark`): direct_c, a soft charge penalty, a
      linear orthogonal projection, and the hyperbolic reparameterization,
      trained on the identical PI-DeepONet backbone across three seeds.
  Priority 4 - Physics-loss weight sensitivity sweep (`physics_weight_sweep`):
      lambda_phys in {0, 1e-4, 1e-3, 1e-2, 1e-1} at seed 42.

Outputs
  results/stage11_upgrades.json
  results/tables/table_sgfv_fisher_validation.csv
  results/tables/table_constraint_schemes_4way.csv
  results/tables/table_physics_weight_sensitivity.csv
  results/statistics.json  (keys stage11_*)
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
from operators import build_model, feature_stats  # noqa: E402
from operators.losses_and_metrics import (  # noqa: E402
    GridTensors,
    continuity_residual,
    pnp_physics_loss,
    poisson_residual,
    sg_current_pA,
)

# reuse Stage-10 helpers (data loading, ONNX operator, metrics)
sys.path.insert(0, HERE)
import run_stage10_upgrades as S10  # noqa: E402

DATA240 = S10.DATA240
RESULTS = S10.RESULTS
TABLES = S10.TABLES
STATS = S10.STATS
OUT_JSON = os.path.join(RESULTS, "stage11_upgrades.json")
SEEDS = [42, 43, 44]
GRID = S10.GRID
AREA = S10.AREA
NEAR_WALL = S10.NEAR_WALL
FISHER_CACHE = os.path.join(RESULTS, "_stage11_fisher_cache.json")

V_BIASES = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
V_NONZERO = np.array([-50.0, -25.0, 25.0, 50.0])
SALT_MULTI = [10.0, 30.0, 100.0]
FD_EPS = 1e-3

SCHEMES = ["direct_c", "soft_charge_penalty", "linear_ortho_projection",
           "hyperbolic_reparam"]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_scheme(data, targets, train_idx, val_idx, scheme, seed, epochs,
                 physics=True, phys_w=1e-3, charge_w=0.0, lr=2e-3, batch=36,
                 p=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_idx = np.array([i for i in train_idx if i not in set(val_idx.tolist())])
    mode = {"direct_c": "direct_c", "soft_charge_penalty": "direct_c",
            "linear_ortho_projection": "linear_ortho_projection",
            "hyperbolic_reparam": "reparam"}[scheme]
    fields = ["psi", "c_plus", "c_minus"] if mode == "direct_c" else ["psi", "s", "rho"]
    scales = {}
    for f in fields + ["rho"]:
        rms = np.sqrt((targets[f] ** 2).mean(axis=(1, 2)))
        g = math.sqrt((targets[f][fit_idx] ** 2).mean())
        scales[f] = np.maximum(rms, 0.1 * g)

    stats = feature_stats(data["mu"][train_idx])
    model = build_model("deeponet", mode, stats, p=p, use_radial_basis=True)
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
            if charge_w > 0.0:
                rt = torch.tensor(targets["rho"][bidx], dtype=torch.float32)
                sc = torch.tensor(scales["rho"][bidx], dtype=torch.float32).reshape(-1, 1, 1)
                loss = loss + charge_w * (((out["rho"] - rt) / sc) ** 2).mean()
            if physics:
                pl = pnp_physics_loss(out, mu_b, GRID)
                loss = loss + phys_w * (pl["poisson"] + pl["continuity"] + pl["wall"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vl = _val_loss(model, data, targets, val_idx, fields, scales)
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
    wall = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, {"epochs": epochs, "train_time_s": wall, "val_loss_best": best_val}


def _val_loss(model, data, targets, idx, fields, scales):
    model.eval()
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
def predict_full(model, idx, mu):
    model.eval()
    outs = {}
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            bidx = idx[s:s + 64]
            o = model(torch.tensor(mu[bidx], dtype=torch.float32))
            for k in ["psi", "c_plus", "c_minus", "rho", "s", "rho_latent"]:
                outs.setdefault(k, []).append(o[k].numpy().astype(np.float64))
    return {k: np.concatenate(v, 0) for k, v in outs.items()}


def evaluate_scheme(model, idx, data):
    pred = predict_full(model, idx, data["mu"])
    ref = data["ref"]
    res = {}
    for f in ["psi", "c_plus", "c_minus", "rho"]:
        res[f"rel_l2_{f}"] = S10._rel_l2(pred[f], ref[f][idx])
    psi = torch.tensor(pred["psi"], dtype=torch.float32)
    cp = torch.tensor(pred["c_plus"], dtype=torch.float32)
    cm = torch.tensor(pred["c_minus"], dtype=torch.float32)
    mu_b = torch.tensor(data["mu"][idx], dtype=torch.float32)
    with torch.no_grad():
        q_w = mu_b[:, 2].reshape(-1, 1)
        dpsi = (psi[:, -1, :] - psi[:, -2, :]) / GRID.dr
        res["wall_deriv_err"] = (dpsi - q_w).abs().mean(dim=1).numpy()
        ipred = sg_current_pA(cp, cm, psi, mu_b, GRID).numpy().astype(np.float64)
        pres = poisson_residual(psi, cp, cm, mu_b, GRID).abs()
        res["poisson_residual"] = pres.mean(dim=(1, 2)).numpy()
        cres = 0.5 * (continuity_residual(cp, psi, mu_b, GRID, 1).abs()
                      + continuity_residual(cm, psi, mu_b, GRID, -1).abs())
        res["continuity_residual"] = cres.mean(dim=(1, 2)).numpy()
    iref = ref["i_z"][idx]
    res["i_rel_l2"] = (np.sqrt(((ipred - iref) ** 2).mean(axis=1))
                       / np.sqrt((iref ** 2).mean(axis=1)))
    res["consistency_err"] = np.abs(pred["rho"] - pred["rho_latent"]).mean(axis=(1, 2))
    res["min_c"] = np.minimum(pred["c_plus"], pred["c_minus"]).min(axis=(1, 2))
    return res


def run_schemes(epochs):
    data = S10.load_npz(DATA240)
    targets = S10.build_targets(data["ref"])
    val_idx = data["train_idx"][::6]
    rows, summary = [], {}
    for scheme in SCHEMES:
        summary[scheme] = {}
        for seed in SEEDS:
            charge_w = 10.0 if scheme == "soft_charge_penalty" else 0.0
            model, info = train_scheme(data, targets, data["train_idx"], val_idx,
                                       scheme, seed, epochs, physics=True,
                                       charge_w=charge_w)
            interp_rho = None
            for split, idx in [("interp", data["interp_idx"]), ("ood", data["ood_idx"])]:
                m = evaluate_scheme(model, idx, data)
                rec = {"scheme": scheme, "seed": seed, "split": split,
                       **{k: float(np.nanmean(v)) for k, v in m.items()},
                       **{k: v for k, v in info.items() if k != "epochs"}}
                rows.append(rec)
                if split == "interp":
                    interp_rho = float(np.nanmean(m["rel_l2_rho"]))
            print(f"[P2] {scheme:22s} seed{seed} interp_rho={interp_rho:.4f} "
                  f"({info['train_time_s']:.1f}s)", flush=True)
    for scheme in SCHEMES:
        summary[scheme] = {}
        for split in ["interp", "ood"]:
            sub = [r for r in rows if r["scheme"] == scheme and r["split"] == split]
            summary[scheme][split] = {
                k: S10.summarize([r[k] for r in sub])
                for k in sub[0] if k not in ("scheme", "seed", "split")}
    return {"epochs": epochs, "seeds": SEEDS, "rows": rows, "summary": summary}


def run_physics_sweep(epochs, seed=42):
    data = S10.load_npz(DATA240)
    targets = S10.build_targets(data["ref"])
    val_idx = data["train_idx"][::6]
    weights = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    rows = []
    for w in weights:
        model, info = train_scheme(data, targets, data["train_idx"], val_idx,
                                   "hyperbolic_reparam", seed, epochs,
                                   physics=(w > 0), phys_w=w)
        interp = None
        for split, idx in [("interp", data["interp_idx"]), ("ood", data["ood_idx"])]:
            m = evaluate_scheme(model, idx, data)
            rows.append({"lambda_phys": w, "seed": seed, "split": split,
                         **{k: float(np.nanmean(v)) for k, v in m.items()},
                         "train_time_s": info["train_time_s"]})
            if split == "interp":
                interp = m
        print(f"[P4] lambda={w:g} interp_rho={np.nanmean(interp['rel_l2_rho']):.4f} "
              f"wall={np.nanmean(interp['wall_deriv_err']):.3e}", flush=True)
    return {"epochs": epochs, "seed": seed, "weights": weights, "rows": rows}


# --------------------------------------------------------------------------- #
# Priority 3: exact SG/FV Jacobian validation
# --------------------------------------------------------------------------- #
_SG_CACHE = {}


def load_sg_cache():
    if os.path.exists(FISHER_CACHE):
        with open(FISHER_CACHE) as fh:
            for k, v in json.load(fh).items():
                _SG_CACHE[tuple(json.loads(k))] = tuple(v)


def save_sg_cache():
    with open(FISHER_CACHE, "w") as fh:
        json.dump({json.dumps(list(k)): list(v) for k, v in _SG_CACHE.items()}, fh)


def sg_observable(V, C0, q, R):
    """Return (mean axial current pA, near-wall selectivity S) for one state."""
    key = (round(float(V), 6), round(float(C0), 6), round(float(q), 9), round(float(R), 6))
    if key in _SG_CACHE:
        return _SG_CACHE[key]
    sol, cfg, grid, meth, nc = G.solve_state(key[0], key[1], key[2], key[3])
    I = float((G.physical_current(sol, grid, cfg) * 1e12).mean())
    cp, cm = sol.c_plus[-1, :], sol.c_minus[-1, :]
    S = float(np.mean((cp - cm) / np.maximum(cp + cm, 1e-30)))
    _SG_CACHE[key] = (I, S)
    return I, S


def _salts(C0_true, protocol):
    return [C0_true] if protocol == "single" else SALT_MULTI


def sg_obs_vec(C0_true, protocol, q, R):
    rows = []
    for C0 in _salts(C0_true, protocol):
        for V in V_NONZERO:
            I, _ = sg_observable(V, C0, q, R)
            rows.append(np.log(max(abs(I), 1e-12)))
        if protocol == "dual":
            _, S = sg_observable(50.0, C0, q, R)
            rows.append(S)
    return np.array(rows)


def op_obs_vec(op, C0_true, protocol, q, R):
    rows = []
    for C0 in _salts(C0_true, protocol):
        mu = np.column_stack([V_NONZERO, np.full(V_NONZERO.size, C0),
                              np.full(V_NONZERO.size, q), np.full(V_NONZERO.size, R)])
        I = op.current(mu)
        rows.extend(np.log(np.maximum(np.abs(I), 1e-12)).tolist())
        if protocol == "dual":
            S = op.wall_selectivity(np.array([[50.0, C0, q, R]]))[0]
            rows.append(float(S))
    return np.array(rows)


def jac_and_cond(obs_fn, C0_true, protocol, q, R, eps=FD_EPS):
    o0 = obs_fn(q, R)
    oqp = obs_fn(q * (1 + eps), R)
    oqm = obs_fn(q * (1 - eps), R)
    oRp = obs_fn(q, R * (1 + eps))
    oRm = obs_fn(q, R * (1 - eps))
    Jq = (oqp - oqm) / (2 * eps)
    JR = (oRp - oRm) / (2 * eps)
    J = np.column_stack([JR, Jq])
    F = J.T @ J
    ev = np.sort(np.abs(np.linalg.eigvalsh(F)))[::-1]
    cond = float(ev[0] / max(ev[-1], 1e-300))
    return {"J": J, "F": F, "eig": ev, "cond": cond,
            "dlnI_dlnR": float(np.mean(JR)), "dlnI_dlnq": float(np.mean(Jq)),
            "dlnI_dlnR_absmean": float(np.mean(np.abs(JR))),
            "dlnI_dlnq_absmean": float(np.mean(np.abs(Jq)))}


def run_fisher_sgfv(op, n_states=15):
    data = S10.load_npz(DATA240)
    interp = data["interp_idx"]
    lam = data["lambda_D_over_R"][interp]
    order = np.argsort(lam)
    pick = order[np.linspace(0, len(order) - 1, n_states).astype(int)]
    state_ids = interp[pick]

    load_sg_cache()
    records = []
    t0 = time.perf_counter()
    for sid in state_ids:
        C0 = float(data["mu"][sid, 1]); q = float(data["mu"][sid, 2]); R = float(data["mu"][sid, 3])
        rec = {"state_index": int(sid), "C0": C0, "q_w": q, "R_nm": R,
               "lambda_D_over_R": float(data["lambda_D_over_R"][sid])}
        for proto in ["single", "multi", "dual"]:
            js = jac_and_cond(lambda qq, RR: sg_obs_vec(C0, proto, qq, RR), C0, proto, q, R)
            jo = jac_and_cond(lambda qq, RR: op_obs_vec(op, C0, proto, qq, RR), C0, proto, q, R)
            rec[f"sgfv_{proto}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in js.items()}
            rec[f"op_{proto}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in jo.items()}
        dR_s, dR_o = rec["sgfv_single"]["dlnI_dlnR"], rec["op_single"]["dlnI_dlnR"]
        rec["dlnI_dlnR_rel_agreement"] = abs(dR_s - dR_o) / max(abs(dR_s), 1e-30)
        records.append(rec)
        print(f"[P3] state {sid:3d} cond_sgfv(single)={rec['sgfv_single']['cond']:.3e} "
              f"cond_op={rec['op_single']['cond']:.3e} "
              f"dlnR sgfv={dR_s:.4f} op={dR_o:.4f}", flush=True)
    save_sg_cache()
    wall = time.perf_counter() - t0

    def med(path, key):
        return float(np.median([_get(r, path, key) for r in records]))

    summary = {
        "n_states": len(records),
        "wall_s": wall,
        "n_sg_solves": len(_SG_CACHE),
        "median_sgfv_cond_single": med("sgfv_single", "cond"),
        "median_op_cond_single": med("op_single", "cond"),
        "median_sgfv_cond_multi": med("sgfv_multi", "cond"),
        "median_op_cond_multi": med("op_multi", "cond"),
        "median_sgfv_cond_dual": med("sgfv_dual", "cond"),
        "median_op_cond_dual": med("op_dual", "cond"),
        "median_sgfv_dlnI_dlnR": med("sgfv_single", "dlnI_dlnR"),
        "median_op_dlnI_dlnR": med("op_single", "dlnI_dlnR"),
        "median_dlnI_dlnR_rel_agreement": float(np.median(
            [r["dlnI_dlnR_rel_agreement"] for r in records])),
        "median_abs_sgfv_dlnI_dlnq": med("sgfv_single", "dlnI_dlnq_absmean"),
        "median_abs_op_dlnI_dlnq": med("op_single", "dlnI_dlnq_absmean"),
    }
    return {"protocol": {"v_biases_mV": V_NONZERO.tolist(), "salts": SALT_MULTI,
                         "fd_eps": FD_EPS, "n_states": n_states},
            "summary": summary, "records": records}


def _get(rec, path, key):
    return rec[path][key]


# --------------------------------------------------------------------------- #
# tables + statistics
# --------------------------------------------------------------------------- #
def write_tables(schemes, sweep, fisher):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "table_constraint_schemes_4way.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["scheme", "seed", "split", "rel_l2_psi", "rel_l2_c_plus", "rel_l2_c_minus",
                "rel_l2_rho", "i_rel_l2", "wall_deriv_err", "poisson_residual",
                "continuity_residual", "consistency_err", "min_c"]
        w.writerow(cols)
        for r in schemes["rows"]:
            w.writerow([r["scheme"], r["seed"], r["split"]] +
                       [f"{r[c]:.8g}" if np.isfinite(r.get(c, np.nan)) else "nan"
                        for c in cols[3:]])
    with open(os.path.join(TABLES, "table_physics_weight_sensitivity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["lambda_phys", "split", "rel_l2_rho", "rel_l2_psi", "i_rel_l2",
                "wall_deriv_err", "poisson_residual", "continuity_residual"]
        w.writerow(cols)
        for r in sweep["rows"]:
            w.writerow([f"{r['lambda_phys']:g}", r["split"]] +
                       [f"{r[c]:.8g}" for c in cols[2:]])
    with open(os.path.join(TABLES, "table_sgfv_fisher_validation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["state_index", "C0", "q_w", "R_nm", "lambda_D_over_R",
                    "sgfv_dlnI_dlnR", "op_dlnI_dlnR", "dlnI_dlnR_rel_agreement",
                    "sgfv_dlnI_dlnq_absmean", "op_dlnI_dlnq_absmean",
                    "sgfv_cond_single", "op_cond_single",
                    "sgfv_cond_multi", "op_cond_multi",
                    "sgfv_cond_dual", "op_cond_dual"])
        for r in fisher["records"]:
            w.writerow([
                r["state_index"], f"{r['C0']:.6g}", f"{r['q_w']:.6g}", f"{r['R_nm']:.6g}",
                f"{r['lambda_D_over_R']:.6g}",
                f"{r['sgfv_single']['dlnI_dlnR']:.6g}", f"{r['op_single']['dlnI_dlnR']:.6g}",
                f"{r['dlnI_dlnR_rel_agreement']:.6g}",
                f"{r['sgfv_single']['dlnI_dlnq_absmean']:.6g}",
                f"{r['op_single']['dlnI_dlnq_absmean']:.6g}",
                f"{r['sgfv_single']['cond']:.6g}", f"{r['op_single']['cond']:.6g}",
                f"{r['sgfv_multi']['cond']:.6g}", f"{r['op_multi']['cond']:.6g}",
                f"{r['sgfv_dual']['cond']:.6g}", f"{r['op_dual']['cond']:.6g}"])


def merge_statistics(schemes, sweep, fisher):
    with open(STATS) as fh:
        stats = json.load(fh)
    stats["stage11_constraint_schemes_benchmark"] = schemes
    stats["stage11_physics_weight_sweep"] = sweep
    stats["stage11_fisher_sgfv_validation"] = fisher
    stats["stage"] = 11
    with open(STATS, "w") as fh:
        json.dump(stats, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-schemes", type=int, default=250)
    ap.add_argument("--epochs-sweep", type=int, default=250)
    ap.add_argument("--skip-schemes", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-fisher", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs_schemes = 40
        args.epochs_sweep = 40

    def cache_path(tag):
        return os.path.join(RESULTS, f"_stage11_cache_{tag}.json")

    def cached(tag):
        p = cache_path(tag)
        if not args.force and os.path.exists(p):
            with open(p) as fh:
                print(f"[cache] reusing {os.path.basename(p)}", flush=True)
                return json.load(fh)
        return None

    def store(tag, obj):
        with open(cache_path(tag), "w") as fh:
            json.dump(obj, fh, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                      else (int(o) if isinstance(o, np.integer) else float(o)
                            if isinstance(o, np.floating) else str(o)))
        return obj

    out = {"stage": 11, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    schemes = sweep = fisher = None
    if not args.skip_schemes:
        schemes = cached(f"schemes_{args.epochs_schemes}")
        if schemes is None:
            print("=== Priority 2: 4-way constraint-scheme benchmark ===", flush=True)
            schemes = store(f"schemes_{args.epochs_schemes}", run_schemes(args.epochs_schemes))
        out["constraint_schemes_benchmark"] = schemes
    if not args.skip_sweep:
        sweep = cached(f"sweep_{args.epochs_sweep}")
        if sweep is None:
            print("=== Priority 4: physics-weight sensitivity sweep ===", flush=True)
            sweep = store(f"sweep_{args.epochs_sweep}", run_physics_sweep(args.epochs_sweep))
        out["physics_weight_sweep"] = sweep
    if not args.skip_fisher:
        fisher = cached("fisher")
        if fisher is None:
            print("=== Priority 3: exact SG/FV Fisher validation ===", flush=True)
            onnx_path = os.path.join(ROOT, "web_simulator", "models", "deeponet_reparam_seed42.onnx")
            op = S10.OnnxOperator(onnx_path)
            fisher = store("fisher", run_fisher_sgfv(op))
        out["fisher_sgfv_validation"] = fisher

    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if all(x is not None for x in (schemes, sweep, fisher)):
        write_tables(schemes, sweep, fisher)
        merge_statistics(schemes, sweep, fisher)
    print(f"[stage11] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
