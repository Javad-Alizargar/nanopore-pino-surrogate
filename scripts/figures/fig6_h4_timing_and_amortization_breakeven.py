"""Figure 6: Hypothesis H4 - latency, amortization break-even and the exact
SG/FV vs neural-operator Fisher/Jacobian agreement (Stage 11)."""

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

# ---- Panels C/D: exact SG/FV vs operator Fisher/Jacobian agreement ------
fv = stat["stage11_fisher_sgfv_validation"]
frecs = fv["records"]

axC = fig.add_subplot(gs[1, 0])
dR_s = np.array([r["sgfv_single"]["dlnI_dlnR"] for r in frecs])
dR_o = np.array([r["op_single"]["dlnI_dlnR"] for r in frecs])
dq_s = np.array([r["sgfv_single"]["dlnI_dlnq_absmean"] for r in frecs])
dq_o = np.array([r["op_single"]["dlnI_dlnq_absmean"] for r in frecs])
lims = (1e-6, 1e1)
axC.plot(lims, lims, color="0.5", lw=0.9, ls="--")
axC.scatter(dR_s, dR_o, s=26, facecolor=K.C["blue"], marker="o",
            edgecolor="white", lw=0.4, label=r"$\partial\ln I/\partial\ln R$")
axC.scatter(dq_s, dq_o, s=26, facecolor=K.C["orange"], marker="s",
            edgecolor="white", lw=0.4,
            label=r"$|\partial\ln I/\partial\ln q_w|$ (abs.)")
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlim(*lims); axC.set_ylim(*lims)
axC.set_xlabel("exact SG/FV finite-difference sensitivity")
axC.set_ylabel("neural-operator sensitivity")
axC.set_title("Jacobian agreement, 15 states\n"
              + rf"$\partial\ln I/\partial\ln R$: median rel. diff "
              + f"{fv['summary']['median_dlnI_dlnR_rel_agreement']:.1e}", fontsize=8.0)
axC.legend(frameon=False, fontsize=6.2, loc="upper left")
K.panel_label(axC, "C", x=-0.16, y=1.02)

axD = fig.add_subplot(gs[1, 1])
protos = ["single", "multi", "dual"]
sg_med = [fv["summary"][f"median_sgfv_cond_{p}"] for p in protos]
op_med = [fv["summary"][f"median_op_cond_{p}"] for p in protos]
xp = np.arange(len(protos))
axD.bar(xp - 0.19, sg_med, 0.36, color=K.C["purple"], edgecolor="0.35",
        linewidth=0.3, label="exact SG/FV")
axD.bar(xp + 0.19, op_med, 0.36, color=K.C["green"], edgecolor="0.35",
        linewidth=0.3, label="neural operator")
for xi in xp:
    axD.text(xi - 0.19, sg_med[xi] * 1.4, f"{sg_med[xi]:.1e}", ha="center",
             fontsize=5.6, rotation=90)
    axD.text(xi + 0.19, op_med[xi] * 1.4, f"{op_med[xi]:.1e}", ha="center",
             fontsize=5.6, rotation=90)
axD.set_yscale("log")
axD.set_xticks(xp)
axD.set_xticklabels(["single\nsalt I-V", "multi\nsalt I-V", "dual\n+selectivity"],
                    fontsize=7.5)
axD.set_ylabel(r"median  $\kappa(\mathcal{F})$")
axD.set_title("observable choice collapses conditioning", fontsize=9)
axD.legend(frameon=False, fontsize=6.4, loc="upper right")
K.panel_label(axD, "D", x=-0.16, y=1.10)

sdata = {"labels": np.array(labels, dtype=object), "latency_ms": np.array(vals),
         "N": N,
         "sg_m1": 0.2013 * N, "sg_xeon": 0.44 * N, "dapinn": 500.0 * N,
         "break_even": np.array([amort[k]["n_break_even"] for k in amort]),
         "break_even_keys": np.array(list(amort.keys()), dtype=object),
         "fisher_sgfv_dlnR": dR_s, "fisher_op_dlnR": dR_o,
         "fisher_sgfv_dlnq": dq_s, "fisher_op_dlnq": dq_o,
         "fisher_sgfv_cond": np.array(sg_med), "fisher_op_cond": np.array(op_med),
         "fisher_protocols": np.array(protos, dtype=object)}
K.save_fig(fig, "fig6_h4_timing_and_amortization_breakeven", sdata)
