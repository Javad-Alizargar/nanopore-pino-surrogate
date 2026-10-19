"""Stage 5 / TASK 5.1: extended-epoch (300) fno_reparam seed-42 verification check.

Trains fno_reparam (seed 42) for 300 epochs, evaluates on interp_test / ood_test /
benchmark, appends the result to results/statistics.json and results/tables/, and
re-exports the ONNX model if the extended run is more accurate than 110 epochs.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "operators"))

import train_and_evaluate_suite as T  # noqa: E402
from operators.losses_and_metrics import GridTensors  # noqa: E402

ROOT = T.ROOT
RESULTS = T.RESULTS
TABLES = T.TABLES
MODELS = T.MODELS


def summarize_arrays(d):
    return {k: T.summarize(v) for k, v in d.items() if isinstance(v, np.ndarray)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_args = argparse.Namespace(
        quick=False, seeds=[args.seed], epochs_fno=args.epochs,
        epochs_deeponet=500, batch=36, lr=2e-3, phys_w=1e-3,
        fno_width=24, fno_modes_r=8, fno_modes_z=12, deeponet_p=64)

    data = T.load_data()
    targets = T.build_targets(data["ref"])
    grid = GridTensors()
    stats = T.feature_stats(data["mu"][data["train_idx"]])
    val_idx = data["train_idx"][::6]
    data["val_idx"] = val_idx
    data["fit_idx"] = np.array([i for i in data["train_idx"] if i not in set(val_idx)])

    t0 = time.perf_counter()
    model, info = T.train_run("fno", "reparam", args.seed, True, args.epochs,
                              run_args, data, targets, grid, stats)
    wall = time.perf_counter() - t0
    print(f"[fno300] trained {args.epochs} ep seed{args.seed}: "
          f"val={info['val_loss_best']:.3e} t={info['train_time_s']:.1f}s", flush=True)

    ckpt = os.path.join(MODELS, f"fno_reparam_seed{args.seed}_{args.epochs}ep.pt")
    torch.save({"state_dict": model.state_dict(), "kind": "fno", "mode": "reparam",
                "seed": args.seed, "condition": "data_plus_physics",
                "epochs": args.epochs, "stats": stats,
                "size": T.size_kwargs("fno", run_args)}, ckpt)

    splits = {"train": data["train_idx"], "interp_test": data["interp_idx"],
              "ood_test": data["ood_idx"], "ood_charge_salt": data["oodA"],
              "ood_geometry_voltage": data["oodB"]}
    per_split = {}
    for sp, idx in splits.items():
        out, glob = T.evaluate_model(model, idx, data, targets, grid, with_consistency=True)
        per_split[sp] = {"global": glob, "per_state": summarize_arrays(out)}
    out, glob = T.evaluate_model(model, np.array([data["bench_idx"]]), data, targets,
                                 grid, with_consistency=True)
    per_split["benchmark"] = {"global": glob, "per_state": summarize_arrays(out)}

    # compare to the 110-epoch 3-seed summary
    with open(os.path.join(RESULTS, "statistics.json")) as fh:
        stat = json.load(fh)
    ref_key = "fno_reparam_data_plus_physics"
    ref_interp = stat["summary_multiseed_global_rel_l2"][ref_key]["interp_test"]
    new_interp_psi = per_split["interp_test"]["global"]["grel_l2_psi"]
    new_interp_I = per_split["interp_test"]["global"]["grel_l2_i"]
    improved = new_interp_psi < ref_interp["grel_l2_psi"]["mean"]
    print(f"[fno300] interp psi 300ep={new_interp_psi:.3e} vs 110ep(3seed)="
          f"{ref_interp['grel_l2_psi']['mean']:.3e} -> improved={improved}", flush=True)

    record = {
        "stage": 5, "task": "5.1_fno_300epoch_check",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed, "epochs": args.epochs, "physics": True,
        "run_record": info, "wall_seconds": wall, "checkpoint": os.path.relpath(ckpt, ROOT),
        "metrics": per_split,
        "comparison_vs_110epoch_3seed": {
            "interp_test_grel_l2_psi_110ep": ref_interp["grel_l2_psi"],
            "interp_test_grel_l2_psi_300ep": new_interp_psi,
            "interp_test_grel_l2_i_110ep": ref_interp["grel_l2_i"],
            "interp_test_grel_l2_i_300ep": new_interp_I,
            "improved": bool(improved),
        },
    }

    # ONNX: re-export if improved, else keep 110-epoch export
    onnx_path = os.path.join(MODELS, f"fno_reparam_seed{args.seed}.onnx")
    idx = np.concatenate([data["interp_idx"], data["ood_idx"]])
    if improved:
        T.export_onnx(model, onnx_path)
        md = T.onnx_parity(model, onnx_path, idx, data["mu"])
        record["onnx"] = {"reexported_300ep": True, "max_abs_diff": md,
                          "parity_ok": bool(md < 1e-5),
                          "path": os.path.relpath(onnx_path, ROOT)}
        stat["onnx_export"]["fno"] = {"path": os.path.relpath(onnx_path, ROOT),
                                      "max_abs_diff": md, "parity_ok": bool(md < 1e-5),
                                      "states_checked": int(len(idx)),
                                      "epochs": args.epochs}
        # also save a dedicated onnx for the 300-epoch model
        T.export_onnx(model, os.path.join(MODELS, f"fno_reparam_seed{args.seed}_300ep.onnx"))
    else:
        record["onnx"] = {"reexported_300ep": False, "note": "110-epoch export retained"}
    print(f"[fno300] onnx {record['onnx']}", flush=True)

    stat["fno_300epoch_check"] = record
    with open(os.path.join(RESULTS, "statistics.json"), "w") as fh:
        json.dump(stat, fh, indent=2)

    # append rows to table3 and write a dedicated comparison table
    t3 = os.path.join(TABLES, "table3_multiseed_h1_h2_h3_ablation.csv")
    rows = list(csv.reader(open(t3)))
    hdr, body = rows[0], rows[1:]
    metric_map = {
        "grel_l2_psi": "grel_l2_psi", "grel_l2_c_plus": "grel_l2_c_plus",
        "grel_l2_c_minus": "grel_l2_c_minus", "grel_l2_rho": "grel_l2_rho",
        "grel_l2_sigma_c": "grel_l2_sigma_c", "grel_l2_i": "grel_l2_i",
        "i_abs_err_pA_mean": "i_abs_err_pA_mean", "wall_deriv_err_mean": "wall_deriv_err_mean",
        "wall_flux_residual_mean": "wall_flux_residual_mean",
        "poisson_residual_mean": "poisson_residual_mean",
        "continuity_residual_mean": "continuity_residual_mean"}
    for sp in ["train", "interp_test", "ood_test", "ood_charge_salt",
               "ood_geometry_voltage", "benchmark"]:
        g = per_split[sp]["global"]
        for mn, key in metric_map.items():
            if key in g:
                body.append(["fno", "reparam", "data_plus_physics_300ep", sp, mn,
                             g[key], 0.0, g[key], 0.0])
    with open(t3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(body)

    with open(os.path.join(TABLES, "table5_fno_epoch_check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "metric", "fno_110ep_3seed_mean", "fno_300ep_seed42"])
        for sp in ["interp_test", "ood_test", "benchmark"]:
            for mn in ["grel_l2_psi", "grel_l2_c_plus", "grel_l2_rho", "grel_l2_i",
                       "i_abs_err_pA_mean", "wall_deriv_err_mean", "poisson_residual_mean"]:
                old = stat["summary_multiseed_global_rel_l2"][ref_key][sp].get(mn, {})
                new = per_split[sp]["global"].get(mn)
                w.writerow([sp, mn, old.get("mean") if isinstance(old, dict) else old, new])

    print("[fno300] DONE", flush=True)


if __name__ == "__main__":
    main()
