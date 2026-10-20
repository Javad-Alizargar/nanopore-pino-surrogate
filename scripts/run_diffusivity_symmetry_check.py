"""Stage 13.2: diffusivity asymmetry and the conductance flat direction.

Deterministic, local, zero paid spend.  Uses the exact float64 SG/FV solver to
test why wall charge is a near-null direction for a symmetric binary
electrolyte and how unequal ionic diffusivities change the linear sensitivity.

Local ohmic conductivity in the Nernst--Planck flux is

    sigma = (F^2 / R T) (D_+ c_+ + D_- c_-)
          = (F^2 / R T) [Dbar (c_+ + c_-) + (D_+ - D_-)/2 (c_+ - c_-)]

with Dbar = (D_+ + D_-)/2 and rho = c_+ - c_-.  Writing
s = c_+ + c_- = sqrt(4 c_+ c_- + rho^2), a Donnan layer with c_+ c_- ~ 1 has
s = 2 + rho^2/4 + O(rho^4), so for D_+ = D_- the charge enters conductance at
second order only.  For D_+ != D_- the linear term (D_+ - D_-) rho / 2 survives.

Outputs
  results/tables/table_diffusivity_symmetry_breaking.csv
  results/statistics.json  (key diffusivity_symmetry_breaking)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import generate_parametric_dataset as G  # noqa: E402
import run_stage11_reviewer_upgrades as S11  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
STATS = os.path.join(RESULTS, "statistics.json")
CSV_OUT = os.path.join(TABLES, "table_diffusivity_symmetry_breaking.csv")

D_MINUS = 2.03e-9
RATIOS = [1.00, 0.9655, 0.655, 0.20]
LABELS = {1.00: "ideal symmetric", 0.9655: "KCl", 0.655: "NaCl", 0.20: "strongly asymmetric"}
V_APP = 50.0
R_NM = 5.0
Q_W = -0.01
SALTS = [100.0, 20.0]
LABEL_SALT = {100.0: "benchmark_C0100", 20.0: "low_salt_C020"}


def obs_vec(C0, protocol, q, R):
    rows = []
    for V in S11.V_NONZERO:
        I, _ = S11.sg_observable(V, C0, q, R)
        rows.append(np.log(max(abs(I), 1e-12)))
    if protocol == "dual":
        _, S = S11.sg_observable(V_APP, C0, q, R)
        rows.append(S)
    return np.array(rows)


def measure(C0, ratio):
    G.DIFF_CATION = ratio * D_MINUS
    G.DIFF_ANION = D_MINUS
    S11._SG_CACHE.clear()
    t0 = time.perf_counter()
    js = S11.jac_and_cond(lambda qq, RR: obs_vec(C0, "single", qq, RR), C0, "single", Q_W, R_NM)
    jd = S11.jac_and_cond(lambda qq, RR: obs_vec(C0, "dual", qq, RR), C0, "dual", Q_W, R_NM)
    # Local coefficient of the linear charge term at the wall, normalized by Dbar.
    c_wall = 1.0
    rho_wall = 0.01
    linear_frac = abs(ratio - 1.0) / (ratio + 1.0) * rho_wall / (2.0 * c_wall)
    return {
        "C0": C0,
        "D_plus_over_D_minus": ratio,
        "D_plus": ratio * D_MINUS,
        "D_minus": D_MINUS,
        "delta_over_sum": abs(ratio - 1.0) / (ratio + 1.0),
        "linear_charge_fraction": linear_frac,
        "dlnI_dlnq_absmean": js["dlnI_dlnq_absmean"],
        "dlnI_dlnR_absmean": js["dlnI_dlnR_absmean"],
        "kappa_single": js["cond"],
        "kappa_dual": jd["cond"],
        "kappa_reduction": js["cond"] / max(jd["cond"], 1e-300),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    rows = []
    for C0 in SALTS:
        for ratio in RATIOS:
            rec = measure(C0, ratio)
            rec["electrolyte"] = LABELS[ratio]
            rec["state_label"] = LABEL_SALT[C0]
            rows.append(rec)
            print(f"{C0:5.0f} mol/m3  D+/D-={ratio:0.4f}  "
                  f"|dlnI/dlnq|={rec['dlnI_dlnq_absmean']:.3e}  "
                  f"kappa_single={rec['kappa_single']:.3e}  kappa_dual={rec['kappa_dual']:.3e}",
                  flush=True)

    os.makedirs(TABLES, exist_ok=True)
    fields = ["state_label", "electrolyte", "C0", "D_plus_over_D_minus", "D_plus", "D_minus",
              "delta_over_sum", "linear_charge_fraction", "dlnI_dlnq_absmean",
              "dlnI_dlnR_absmean", "kappa_single", "kappa_dual", "kappa_reduction"]
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    stats = json.load(open(STATS)) if os.path.exists(STATS) else {}
    stats["diffusivity_symmetry_breaking"] = {
        "formula": "sigma = (F^2/RT)[Dbar s + (D_+ - D_-) rho / 2], s = sqrt(4 c_+ c_- + rho^2) = 2 + rho^2/4 + O(rho^4)",
        "state": {"V_app_mV": V_APP, "R_nm": R_NM, "q_w": Q_W, "salts": SALTS},
        "ratios": RATIOS,
        "rows": rows,
        "n_sg_solves": len(S11._SG_CACHE),
        "note": "Centered finite differences in ln|q_w| and ln R at relative step 1e-3; kappa is the condition number of F=J^T J with rows ln|I| at four non-zero biases (single) plus S at 50 mV (dual).",
    }
    json.dump(stats, open(STATS, "w"), indent=1)
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
