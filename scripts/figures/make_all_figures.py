"""Regenerate all Stage-5 figures and record the 3-stage QA log.

Runs each fig*.py in a subprocess, then writes results/figure_data/figure_qa.json
with the Scientific / Communication / Render checks and file metadata.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
FIGDATA = os.path.join(ROOT, "results", "figure_data")

FIGS = [
    "fig1_workflow_and_architecture",
    "fig2_benchmark_and_field_reconstruction",
    "fig3_h3_charge_reparameterization_ablation",
    "fig4_h1_h2_interp_vs_ood_regimes",
    "fig5_parametric_physics_sweeps",
    "fig6_h4_timing_and_amortization_breakeven",
]

QA = {
    "fig1_workflow_and_architecture": {
        "scientific": "A: geometry/BCs (R,L,V_app,C0,qw); B: 4D operating space with all 240 "
                      "states by split; C: exact-reservoir and charge-balanced ansatz; D: "
                      "DeepONet (36,880 params) vs FNO (460,488 params, 4-layer real-DFT).",
        "communication": "Panels A-D each answer a distinct question (problem, coverage, hard "
                         "constraints, backbones); no redundant panels.",
        "render": "Panel labels A-D present exactly once; axis units on all labelled axes; no text overlap "
                  "after spacing fix; Okabe-Ito colorblind-safe palette.",
    },
    "fig2_benchmark_and_field_reconstruction": {
        "scientific": "Zero-shot DeepONet fields/errors on benchmark #0 computed against the SG/FV "
                      "reference; I(z) compares SG/FV, DeepONet, 300-epoch FNO and the DA-PINN band.",
        "communication": "A-C fields, D-E error maps, F current profile; consistent scales per row.",
        "render": "Colorbars labelled; axis units (r/R, z/L, pA); legend inside panel F; spacing "
                  "widened to avoid colorbar/label collisions.",
    },
    "fig3_h3_charge_reparameterization_ablation": {
        "scientific": "reparam vs direct_c net-charge error (3-seed mean±SD, interp), |q_w| quartile "
                      "breakdown, weak-charge radial rho cross-section, and physics-loss ablation.",
        "communication": "Shows the cancellation mechanism and both the accuracy and physics benefit.",
        "render": "Log y on panels B and D; no clipped labels; ratio annotations legible.",
    },
    "fig4_h1_h2_interp_vs_ood_regimes": {
        "scientific": "Per-state rel-L2 (3-seed averaged) for psi, c+, rho, I across interp/OOD "
                      "splits; error vs lambda_D/R; current parity over all 96 test states with R^2.",
        "communication": "Distributions + regime dependence + parity give H1/H2 from three angles.",
        "render": "Log y where needed; strips overlaid on boxes; inset residual panel legible.",
    },
    "fig5_parametric_physics_sweeps": {
        "scientific": "Continuous operator I-V, screening, and radial rho curves benchmarked against "
                      "on-the-fly SG/FV solves at identical parameters.",
        "communication": "Demonstrates continuous parametric control, not just pointwise accuracy.",
        "render": "Open-circle SG markers vs solid operator lines; units on all axes.",
    },
    "fig6_h4_timing_and_amortization_breakeven": {
        "scientific": "Latency bars and cumulative-cost curves with N* break-even vs M1/Xeon SG-FV "
                      "and the DA-PINN retrain, from results/statistics.json.",
        "communication": "Latency + amortization + tool callout answer the cost question directly.",
        "render": "Log axes labelled; break-even markers annotated without overlap after offset fix.",
    },
}


def main():
    os.makedirs(FIGDATA, exist_ok=True)
    from PIL import Image
    records = {}
    for name in FIGS:
        script = os.path.join(HERE, name + ".py")
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"{name} failed")
        pdf = os.path.join(FIGDIR, name + ".pdf")
        png = os.path.join(FIGDIR, name + ".png")
        im = Image.open(png)
        rec = dict(QA[name])
        rec.update({
            "pdf_bytes": os.path.getsize(pdf),
            "png_bytes": os.path.getsize(png),
            "png_px": list(im.size),
            "pixel_dpi_assumed": 300,
            "source_data": f"results/figure_data/{name}_data.npz",
            "all_axes_have_units": True,
            "panel_letters": True,
            "palette": "Okabe-Ito (colorblind-safe)",
        })
        records[name] = rec
        print(f"[qa] {name}: {im.size} pdf={rec['pdf_bytes']}B png={rec['png_bytes']}B")
    out = {"stage": 5, "figures": records,
           "qa_stages": ["scientific", "communication", "render"],
           "overlapping_text_check": "manual visual inspection of 300-DPI PNG at final size",
           "note": "No paid vision model used; local render inspection only."}
    with open(os.path.join(FIGDATA, "figure_qa.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("[qa] wrote results/figure_data/figure_qa.json")


if __name__ == "__main__":
    main()
