"""Figure 6: Hypothesis H4 - latency, amortization break-even and the exact
SG/FV vs neural-operator Fisher/Jacobian agreement (Stage 11)."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import common as K

stat = K.load_statistics()
h4 = stat["h4_timing_amortization"]
inf = h4["inference_ms"]
amort = h4["amortization"]

fig = plt.figure(figsize=(10.6, 7.6))
gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.38,
              width_ratios=[1.0, 1.05])

# ---- Panel A: latency bars ---------------------------------------------
axA = fig.add_subplot(gs[0, 0])
items = [
    ("DA-PINN retrain (per state)", 500.0 * 1e3, K.C["grey"]),
    ("SG/FV solver, Xeon", 440.0, K.C["purple"]),
    ("SG/FV solver, Apple M1", 201.0, K.C["orange"]),
    ("PyTorch FNO-2D", inf["fno"]["torch_f32_batch1_ms"], K.C["vermillion"]),
    ("PyTorch PI-DeepONet", inf["deeponet"]["torch_f32_batch1_ms"], K.C["sky"]),
    ("ONNX Runtime FNO-2D", inf["fno"]["onnx_f32_batch1_ms"], K.C["green"]),
    ("ONNX Runtime DeepONet", inf["deeponet"]["onnx_f32_batch1_ms"], K.C["blue"]),
]
labels = [a for a, _, _ in items]
vals = [b for _, b, _ in items]
cols = [c for _, _, c in items]
y = np.arange(len(items))
axA.barh(y, vals, color=cols, edgecolor="0.3", linewidth=0.4)
axA.set_xscale("log")
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=7)
axA.set_xlabel("single-state inference / solve time  (ms, log scale)")
axA.set_title("prediction latency", fontsize=9)
for yi, v in zip(y, vals):
    txt = f"{v:.3g} ms" if v < 1e3 else f"{v/1e3:.0f} s"
    axA.text(v * 1.15, yi, txt, va="center", fontsize=6.4)
axA.set_xlim(0.03, 3e6)
K.panel_label(axA, "A")

# ---- Panel B: cumulative cost vs N queries ------------------------------
axB = fig.add_subplot(gs[0, 1])
N = np.logspace(0, 3.5, 200)
axB.plot(N, 0.2013 * N, color=K.C["orange"], lw=1.4, label="SG/FV, M1 (0.20 s)")
axB.plot(N, 0.44 * N, color=K.C["purple"], lw=1.4, ls="--", label="SG/FV, Xeon (0.44 s)")
axB.plot(N, 500.0 * N, color=K.C["grey"], lw=1.4, ls=":", label="DA-PINN retrain (500 s)")
for kind, col, lab in [("deeponet", K.C["blue"], "PI-DeepONet (ONNX)"),
                       ("fno", K.C["green"], "FNO-2D (ONNX)")]:
    t_train = amort[f"{kind}_stage2_median_s"]["t_train_s"]
    t_inf = amort[f"{kind}_stage2_median_s"]["t_infer_s"]
    c = 59.18 + t_train
    axB.plot(N, c + N * t_inf, color=col, lw=1.7, label=lab)
    nbe = amort[f"{kind}_stage2_median_s"]["n_break_even"]
    if 1 <= nbe <= 10 ** 3.5:
        axB.plot([nbe], [0.2013 * nbe], marker="*", ms=11, color=col,
                 mec="k", mew=0.4, zorder=5)
        off = (7, -13) if kind == "deeponet" else (7, 5)
        axB.annotate(f"$N^*\\!\\approx\\!{nbe:.0f}$", (nbe, 0.2013 * nbe),
                     textcoords="offset points", xytext=off, fontsize=6.6,
                     color=col)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("number of parameter queries  $N$")
axB.set_ylabel("cumulative wall-clock (s, log)")
axB.set_title("amortization break-even", fontsize=9)
axB.legend(frameon=False, fontsize=6.2, loc="upper left")
axB.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
axB.add_patch(FancyBboxPatch((0.36, 0.05), 0.60, 0.23,
              boxstyle="round,pad=0.02", transform=axB.transAxes,
              facecolor="#EAF3F8", edgecolor=K.C["blue"], lw=0.7))
axB.text(0.66, 0.205, "interactive tools", transform=axB.transAxes, ha="center",
         fontsize=6.8, fontweight="bold", color=K.C["blue"])
axB.text(0.66, 0.105, "browser ONNX (0.072 ms)\n+ Gradio live SG/FV compare",
         transform=axB.transAxes, ha="center", fontsize=5.9)
K.panel_label(axB, "B", x=-0.12, y=1.15)

# ---- Panels C/D: exact SG/FV sensitivities and inverse recovery ---------
fp = stat["stage18_fisher_plane_corrected"]
ex = fp["exact_8x8"]

axC = fig.add_subplot(gs[1, 0])
dR_s = np.array([r["exact_dlnI_dlnR"] for r in ex])
dR_o = np.array([r["op_dlnI_dlnR"] for r in ex])
dq_s = np.abs(np.array([r["exact_dlnI_dlnq"] for r in ex]))
dq_o = np.abs(np.array([r["op_dlnI_dlnq"] for r in ex]))
lims = (1e-6, 3.0)
axC.plot(lims, lims, color="0.5", lw=0.9, ls="--")
axC.scatter(dR_s, dR_o, s=24, facecolor=K.C["blue"], marker="o",
            edgecolor="white", lw=0.35,
            label=r"$\partial\ln I/\partial\ln R$")
axC.scatter(dq_s, dq_o, s=24, facecolor=K.C["orange"], marker="s",
            edgecolor="white", lw=0.35,
            label=r"$|\partial\ln I/\partial\ln q_w|$")
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlim(*lims); axC.set_ylim(*lims)
axC.set_xlabel("exact SG/FV finite-difference sensitivity")
axC.set_ylabel("neural-operator sensitivity")
relR = np.median(np.abs(dR_o - dR_s) / np.abs(dR_s))
axC.set_title("noise-whitened sensitivities, 8x8 states\n"
              + rf"$\partial\ln I/\partial\ln R$: median rel. diff {relR:.1e}",
              fontsize=8.0)
axC.legend(frameon=False, fontsize=6.2, loc="upper left")
K.panel_label(axC, "C", x=-0.16, y=1.02)

axD = fig.add_subplot(gs[1, 1])
protos = ["single", "vrev", "dual"]
inv_path = os.path.join(K.RESULTS, "stage21_inverse_records_real.npz")
if os.path.exists(inv_path):
    z = np.load(inv_path, allow_pickle=True)
    proto = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in z["protocol"]])
    noise = z["noise"].astype(float)
    q_med, q_err, r_med, r_err = [], [], [], []
    for p in protos:
        m = (proto == p) & np.isclose(noise, 0.01)
        q = z["B_q_rel"][m].astype(float)
        r = z["B_R_rel"][m].astype(float)
        q_med.append(float(np.median(q)))
        q_err.append(float(np.quantile(q, .75) - np.quantile(q, .25)))
        r_med.append(float(np.median(r)))
        r_err.append(float(np.quantile(r, .75) - np.quantile(r, .25)))
    q_med, q_err, r_med, r_err = map(np.array, (q_med, q_err, r_med, r_err))
    n_rep = int(np.unique(z["realization"]).size)
else:  # fallback for older checkouts
    inv20 = stat["stage20_hybrid_verification"]["inverse_by_band"]
    q_med = np.array([inv20[f"{p}_B_noise0.01_all"]["median_q_rel_err"] for p in protos])
    q_err = np.array([inv20[f"{p}_B_noise0.01_all"]["q_iqr"] for p in protos])
    r_med = np.array([inv20[f"{p}_B_noise0.01_all"]["median_R_rel_err"] for p in protos])
    r_err = np.array([inv20[f"{p}_B_noise0.01_all"]["R_iqr"] for p in protos])
    n_rep = 20
xp = np.arange(len(protos)); w = 0.38
axD.bar(xp - w / 2, q_med, w, yerr=q_err / 2, capsize=2.4,
        color=K.C["orange"], edgecolor="0.35", linewidth=0.3,
        label=r"$q_w$ rel. error")
axD.bar(xp + w / 2, r_med, w, yerr=r_err / 2, capsize=2.4,
        color=K.C["blue"], edgecolor="0.35", linewidth=0.3,
        label=r"$R$ rel. error")
for xi, v in zip(xp, q_med):
    axD.text(xi - w / 2, v * 1.35, f"{v:.2g}", ha="center", fontsize=5.6,
             rotation=90)
for xi, v in zip(xp, r_med):
    axD.text(xi + w / 2, v * 1.35, f"{v:.1e}", ha="center", fontsize=5.6,
             rotation=90)
axD.set_yscale("log"); axD.set_ylim(1e-4, 10.0)
axD.set_xticks(xp)
axD.set_xticklabels(["single\nsalt I-V", "reversal\npotential", "dual\n+S"], fontsize=6.4)
axD.set_ylabel("median relative error")
axD.set_title("blind inversion at 1% noise (Stage 20, B)\n"
              f"15 states x {n_rep} realizations", fontsize=8.0)
axD.legend(frameon=False, fontsize=6.3, loc="upper right")

K.panel_label(axD, "D", x=-0.16, y=1.10)

sdata = {"labels": np.array(labels, dtype=object), "latency_ms": np.array(vals),
         "N": N,
         "sg_m1": 0.2013 * N, "sg_xeon": 0.44 * N, "dapinn": 500.0 * N,
         "break_even": np.array([amort[k]["n_break_even"] for k in amort]),
         "break_even_keys": np.array(list(amort.keys()), dtype=object),
         "sgfv_dlnR": dR_s, "op_dlnR": dR_o,
         "sgfv_dlnq": dq_s, "op_dlnq": dq_o,
         "inverse_q_median": q_med, "inverse_q_iqr": q_err,
         "inverse_R_median": r_med, "inverse_R_iqr": r_err,
         "fisher_protocols": np.array(protos, dtype=object),
         "inverse_stage20": np.array(protos, dtype=object)}
K.save_fig(fig, "fig6_h4_timing_and_amortization_breakeven", sdata)
