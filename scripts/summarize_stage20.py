#!/usr/bin/env python3
"""Summarize the blind Stage-20 inverse records by protocol, noise, and band."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "results" / "stage20_complete_records.npz"
OUT = ROOT / "results" / "stage20_inverse_by_band.json"
CSV = ROOT / "results" / "tables" / "table_s22_inverse_by_band.csv"
VER = ROOT / "results" / "stage20_hybrid_verification.json"
STATS = ROOT / "results" / "statistics.json"
PROTOCOLS = ["single", "vrev", "dual"]
NOISES = [0.002, 0.01, 0.02]
METHODS = ["A", "B", "C"]
BANDS = ["all", "weak", "transitional", "strong"]


def main():
    d = np.load(NPZ, allow_pickle=True)
    state = d["state"].astype(int)
    protocol = d["protocol"].astype(str)
    noise = d["noise"].astype(float)
    band = np.array([""] * state.size, dtype=object)
    truth = json.loads((ROOT / "results" / "stage20_hybrid_verification.json").read_text())["states"]
    for i, sid in enumerate(state):
        band[i] = truth[int(sid)]["band"]
    rows = {}
    for p in PROTOCOLS:
        for n in NOISES:
            for b in BANDS:
                m = (protocol == p) & np.isclose(noise, n)
                if b != "all":
                    m &= band == b
                for method in METHODS:
                    q = d[f"{method}_q_rel"][m]
                    r = d[f"{method}_R_rel"][m]
                    s = d[f"{method}_sigma_rel"][m]
                    if q.size == 0:
                        continue
                    rows[f"{p}_{method}_noise{n}_{b}"] = {
                        "n": int(q.size),
                        "median_q_rel_err": float(np.median(q)),
                        "q_iqr": float(np.quantile(q, .75) - np.quantile(q, .25)),
                        "q_p95": float(np.quantile(q, .95)),
                        "median_R_rel_err": float(np.median(r)),
                        "R_iqr": float(np.quantile(r, .75) - np.quantile(r, .25)),
                        "R_p95": float(np.quantile(r, .95)),
                        "median_sigma_rel_err": float(np.median(s)),
                        "q_coverage95": float(np.mean(d[f"{method}_q_cover"][m])),
                        "R_coverage95": float(np.mean(d[f"{method}_R_cover"][m])),
                        "sigma_coverage95": float(np.mean(d[f"{method}_sigma_cover"][m])),
                        "converged_fraction": float(np.mean(d[f"{method}_converged"][m])),
                        "boundary_hit_fraction": float(np.mean(d[f"{method}_boundary_hit"][m])),
                        "median_time_s": float(np.median(d[f"{method}_time_s"][m])),
                    }
    OUT.write_text(json.dumps(rows, indent=2))
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w") as fh:
        fh.write("protocol,method,noise,band,n,median_q_rel_err,q_iqr,median_R_rel_err,"
                 "R_iqr,median_sigma_rel_err,q_coverage95,R_coverage95,sigma_coverage95,"
                 "converged_fraction,boundary_hit_fraction,median_time_s\n")
        for key, v in rows.items():
            p, method, nz, b = key.split("_")
            fh.write(f"{p},{method},{nz.replace('noise','')},{b},{v['n']},"
                     f"{v['median_q_rel_err']:.10g},{v['q_iqr']:.10g},"
                     f"{v['median_R_rel_err']:.10g},{v['R_iqr']:.10g},"
                     f"{v['median_sigma_rel_err']:.10g},"
                     f"{v['q_coverage95']:.6g},{v['R_coverage95']:.6g},{v['sigma_coverage95']:.6g},"
                     f"{v['converged_fraction']:.6g},{v['boundary_hit_fraction']:.6g},"
                     f"{v['median_time_s']:.6g}\n")
    ver = json.loads(VER.read_text())
    ver["inverse_summary_by_band"] = rows
    VER.write_text(json.dumps(ver, indent=2))
    stats = json.loads(STATS.read_text())
    stats.setdefault("stage20_hybrid_verification", {})
    stats["stage20_hybrid_verification"]["inverse_by_band"] = rows
    STATS.write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} rows) and {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
