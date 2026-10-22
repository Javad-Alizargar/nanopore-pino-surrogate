#!/usr/bin/env python3
"""Recompute every Stage-21 inverse summary directly from the per-realization records.

Reads results/stage21_inverse_records_real.npz (or the resumable checkpoint pickle),
writes results/stage21_inverse_summary_real.json, and emits the markdown body of
Supplementary Table S22 part B to results/table_s22_partB.md.

No hard-coded numbers: all values are medians/fractions over individual records.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NPZ = RES / "stage21_inverse_records_real.npz"
PKL = RES / "checkpoints" / "inverse_records.pkl"
OUT_JSON = RES / "stage21_inverse_summary_real.json"
OUT_MD = RES / "table_s22_partB.md"

BANDS = ["weak", "transitional", "strong"]
PROTOCOLS = ["single", "vrev", "dual"]
NOISES = [0.002, 0.01, 0.02]
PROTO_LABEL = {"single": "single salt", "vrev": "reversal potential", "dual": "dual + selectivity"}


def load_records():
    if NPZ.exists():
        z = np.load(NPZ, allow_pickle=True)
        keys = [k for k in z.files if k not in ("y_obs", "sigmas")]
        n = len(z[keys[0]])
        return [{k: z[k][i].item() if hasattr(z[k][i], "item") else z[k][i]
                 for k in keys} for i in range(n)]
    if PKL.exists():
        with open(PKL, "rb") as fh:
            return pickle.load(fh)["records"]
    raise SystemExit("no inverse records found")


def band_of(sid):
    return BANDS[min(int(sid) // 5, 2)]


def med(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return float(np.median(vals)) if vals.size else float("nan")


def summarize(records):
    for r in records:
        r["band"] = band_of(r["state"])
    out = {"n_records": len(records), "by_condition": {}, "overall_1pct": {}, "validation": {}}

    # Validation: diagnostics must not be all-zero / all-inf.
    val = {}
    for f in ("A_se_lnR", "A_se_lnsigma", "A_se_lnq",
              "B_se_lnR", "B_se_lnsigma", "B_se_lnq",
              "C_se_lnR", "C_se_lnsigma", "C_se_lnq",
              "B_delta_m_inf", "C_delta_m_inf"):
        v = np.asarray([r[f] for r in records], dtype=float)
        val[f] = {"n_finite_positive": int(np.sum(np.isfinite(v) & (v > 0))),
                  "n_total": int(v.size), "median": med(v)}
    out["validation"] = val

    for proto in PROTOCOLS:
        for noise in NOISES:
            sub = [r for r in records if r["protocol"] == proto and abs(r["noise"] - noise) < 1e-12]
            if not sub:
                continue
            key = f"{proto}_{noise:g}"
            row = {"n": len(sub)}
            for est in ("A", "B", "C"):
                for qty, field in (("q", "q_rel"), ("R", "R_rel"), ("sigma", "sigma_rel")):
                    row[f"{est}_{qty}_rel"] = med([r[f"{est}_{qty}_rel"] for r in sub])
                    vals = np.asarray([r[f"{est}_{qty}_rel"] for r in sub], dtype=float)
                    row[f"{est}_{qty}_iqr"] = float(np.quantile(vals, .75) - np.quantile(vals, .25))
                for qty in ("q", "R", "sigma"):
                    row[f"{est}_{qty}_cover"] = float(np.mean([r[f"{est}_{qty}_cover"] for r in sub]))
                row[f"{est}_se_lnq"] = med([r[f"{est}_se_lnq"] for r in sub])
            row["B_converged"] = float(np.mean([r["B_converged"] for r in sub]))
            row["C_converged"] = float(np.mean([r["C_converged"] for r in sub]))
            row["B_boundary_hit"] = float(np.mean([r["B_boundary_hit"] for r in sub]))
            row["B_iterations"] = med([r["B_iterations"] for r in sub])
            row["B_time_s"] = med([r["B_time_s"] for r in sub])
            for band in BANDS:
                bs = [r for r in sub if r["band"] == band]
                row[f"B_q_rel_{band}"] = med([r["B_q_rel"] for r in bs])
                row[f"B_q_cover_{band}"] = float(np.mean([r["B_q_cover"] for r in bs])) if bs else float("nan")
                row[f"B_R_cover_{band}"] = float(np.mean([r["B_R_cover"] for r in bs])) if bs else float("nan")
                row[f"B_sigma_cover_{band}"] = float(np.mean([r["B_sigma_cover"] for r in bs])) if bs else float("nan")
            out["by_condition"][key] = row

    # Headline: dual protocol at 1% noise with estimator B.
    headline = out["by_condition"].get("dual_0.01")
    if headline is not None:
        out["overall_1pct"] = {
            "dual_B_median_q_rel": headline["B_q_rel"],
            "dual_B_median_R_rel": headline["B_R_rel"],
            "dual_B_median_sigma_rel": headline["B_sigma_rel"],
            "dual_B_coverage_q": headline["B_q_cover"],
            "dual_B_coverage_R": headline["B_R_cover"],
            "dual_B_coverage_sigma": headline["B_sigma_cover"],
            "dual_B_coverage_q_weak": headline["B_q_cover_weak"],
            "dual_B_coverage_q_transitional": headline["B_q_cover_transitional"],
            "dual_B_coverage_q_strong": headline["B_q_cover_strong"],
        }
    return out


def table_rows(summary):
    lines = [
        "| Protocol | Noise | median $q_w$ rel. | weak | trans. | strong | median $R$ rel. | median $\\sigma$ rel. | cov. $q_w$ | cov. $R$ | cov. $\\sigma$ | $n$ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for proto in PROTOCOLS:
        for noise in NOISES:
            r = summary["by_condition"].get(f"{proto}_{noise:g}")
            if not r:
                continue
            lines.append(
                f"| {PROTO_LABEL[proto]} | {noise*100:g}% "
                f"| {r['B_q_rel']*100:.2f}% | {r['B_q_rel_weak']*100:.2f}% "
                f"| {r['B_q_rel_transitional']*100:.2f}% | {r['B_q_rel_strong']*100:.2f}% "
                f"| {r['B_R_rel']*100:.2f}% | {r['B_sigma_rel']*100:.2f}% "
                f"| {r['B_q_cover']*100:.0f}% | {r['B_R_cover']*100:.0f}% "
                f"| {r['B_sigma_cover']*100:.0f}% | {r['n']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    records = load_records()
    summary = summarize(records)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    OUT_MD.write_text(table_rows(summary))
    print(f"wrote {OUT_JSON} and {OUT_MD} from {len(records)} records")
    print(json.dumps(summary["overall_1pct"], indent=2))


if __name__ == "__main__":
    main()
