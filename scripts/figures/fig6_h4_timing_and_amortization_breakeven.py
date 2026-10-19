"""Figure 6: Hypothesis H4 - latency and amortization break-even."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import common as K

stat = K.load_statistics()
h4 = stat["h4_timing_amortization"]
inf = h4["inference_ms"]
amort = h4["amortization"]

fig = plt.figure(figsize=(10.6, 4.0))
gs = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.0, 1.05])

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
axB.set_ylabel("cumulative wall-clock time  (s, log scale)")
axB.set_title("amortization break-even", fontsize=9)
axB.legend(frameon=False, fontsize=6.2, loc="upper left")
axB.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
# web-tool callout
axB.add_patch(FancyBboxPatch((0.40, 0.06), 0.56, 0.20,
              boxstyle="round,pad=0.02", transform=axB.transAxes,
              facecolor="#EAF3F8", edgecolor=K.C["blue"], lw=0.7))
axB.text(0.68, 0.185, "interactive tools", transform=axB.transAxes, ha="center",
         fontsize=7, fontweight="bold", color=K.C["blue"])
axB.text(0.68, 0.105, "browser ONNX (0.072 ms) +\nGradio live SG/FV compare",
         transform=axB.transAxes, ha="center", fontsize=6.2)
K.panel_label(axB, "B")

sdata = {"labels": np.array(labels, dtype=object), "latency_ms": np.array(vals),
         "N": N,
         "sg_m1": 0.2013 * N, "sg_xeon": 0.44 * N, "dapinn": 500.0 * N,
         "break_even": np.array([amort[k]["n_break_even"] for k in amort]),
         "break_even_keys": np.array(list(amort.keys()), dtype=object)}
K.save_fig(fig, "fig6_h4_timing_and_amortization_breakeven", sdata)
