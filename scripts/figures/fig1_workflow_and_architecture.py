"""Figure 1: 4D parametric nanopore PNP problem, exact-boundary ansatz, and
parameter-conditioned operator architectures (A-D on a 2 x 2 grid)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Patch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import common as K

stat = K.load_statistics()
dat = K.load_dataset()
mu = dat["mu"]
lam = dat["lambda_D_over_R"]
qw = np.abs(mu[:, 2])

fig = plt.figure(figsize=(15.0, 10.0))
gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.32)

# --------------------------------------------------------------------------- #
# Panel A: axisymmetric charged nanopore geometry and boundary conditions
# --------------------------------------------------------------------------- #
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(-0.30, 1.46)
axA.set_ylim(-0.46, 1.40)
axA.axis("off")
axA.add_patch(Rectangle((0, 0), 1.0, 1.0, facecolor="#EAF3F8", edgecolor="none"))
axA.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=K.C["black"], lw=1.3)
axA.plot([1, 1], [0, 1], color=K.C["blue"], lw=3.6)          # charged wall
for yc in np.linspace(0.14, 0.86, 4):
    axA.text(1.07, yc, r"$-$", color=K.C["blue"], fontsize=13, ha="left", va="center")
axA.text(1.20, 0.5, r"wall charge  $q_w$", color=K.C["blue"], fontsize=10.5,
         rotation=90, va="center")
axA.text(0.5, -0.36, r"reservoir  $C_0,\ \psi=0$", ha="center", fontsize=10)
axA.text(0.5, 1.16, r"reservoir  $C_0,\ \psi=V_\mathrm{app}$", ha="center", fontsize=10)
axA.add_patch(FancyArrowPatch((0, -0.12), (1, -0.12), arrowstyle="<|-|>",
              mutation_scale=11, color=K.C["grey"], lw=1.1))
axA.text(0.5, -0.19, r"radius $R$", ha="center", fontsize=10, color=K.C["grey"])
axA.add_patch(FancyArrowPatch((1.33, 0), (1.33, 1), arrowstyle="<|-|>",
              mutation_scale=11, color=K.C["grey"], lw=1.1))
axA.text(1.40, 0.5, r"length $L$", rotation=90, ha="left", va="center",
         fontsize=10, color=K.C["grey"])
axA.text(0.03, 0.5, r"$r=0$" + "\n" + r"(axis)", fontsize=9.5, va="center",
         color=K.C["grey"])
axA.text(0.5, 0.50, r"Poisson$-$Nernst$-$Planck" + "\n"
         + r"$\nabla\!\cdot\!(\varepsilon\nabla\psi)=-\rho$" + "\n"
         + r"$\nabla\!\cdot\!\mathbf{J}_\pm=0$",
         ha="center", va="center", fontsize=9.6, color="0.25")
axA.set_title(r"axisymmetric charged nanopore", fontsize=11,
              loc="left", fontweight="bold")
K.panel_label(axA, "A")

# --------------------------------------------------------------------------- #
# Panel B: 4D parametric operating space
# --------------------------------------------------------------------------- #
axB = fig.add_subplot(gs[0, 1])
special = 144  # benchmark #0, interp_test[0]
groups = [
    (dat["train_idx"], K.C["grey"], "o", "train ($N{=}144$)"),
    (dat["interp_idx"], K.C["blue"], "o", "interp test ($N{=}48$)"),
    (np.concatenate([dat["oodA"], dat["oodB"]]), K.C["vermillion"], "s",
     "OOD test ($N{=}48$)"),
]
for idx, col, mk, lab in groups:
    idx = np.asarray(idx)
    keep = idx[idx != special]
    axB.scatter(lam[keep], qw[keep], s=34, c=col, marker=mk, label=lab,
                edgecolor="white", lw=0.5, zorder=3)
axB.scatter([lam[special]], [qw[special]], s=190, marker="*",
            c=K.C["orange"], edgecolor="black", lw=0.7, zorder=5,
            label="benchmark #0")
axB.axvspan(0.085, 0.717, color=K.C["grey"], alpha=0.10, zorder=0)
axB.axhline(0.05, color=K.C["grey"], ls="--", lw=0.9, zorder=1)
axB.text(1.17, 0.052, r"train $|q_w|\leq 0.05$", fontsize=8.5, color="0.35",
         ha="right", va="bottom")
axB.set_xlabel(r"Debye ratio  $\lambda_D/R$", fontsize=10)
axB.set_ylabel(r"surface charge magnitude  $|q_w|$", fontsize=10)
axB.set_xlim(0.06, 1.20)
axB.set_ylim(0, 0.145)
axB.tick_params(labelsize=9)
axB.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.90,
           edgecolor="none", fontsize=7.6, columnspacing=1.0,
           handletextpad=0.4)
axB.set_title(r"4D operating space  "
              r"$\boldsymbol{\mu}=(V_\mathrm{app},C_0,q_w,R)$",
              fontsize=11, loc="left", fontweight="bold")
axB.text(0.985, 0.03, r"$N=240$ states", transform=axB.transAxes,
         ha="right", va="bottom", fontsize=9, color="0.3")
K.panel_label(axB, "B")

# --------------------------------------------------------------------------- #
# Panel C: exact-reservoir and charge-balanced ansatz
# --------------------------------------------------------------------------- #
axC = fig.add_subplot(gs[1, 0])
axC.axis("off")
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)
boxes = [
    (0.01, "#EAF3F8", K.C["blue"], "exact reservoir\nDirichlet",
     r"$\psi_\theta=\psi_\mathrm{app}(1-z)$" + "\n"
     r"$+\,z(1-z)\,G_\psi$",
     r"$\psi(r,0)=\psi_\mathrm{app}$" + "\n" + r"$\psi(r,1)=0$"),
    (0.345, "#FBF3E6", K.C["orange"], "charge-balanced\nfields",
     r"$s_\theta=2\sqrt{c_{+\theta}c_{-\theta}}$" + "\n"
     r"$=2\exp\!\left[z(1-z)G_s\right]$" + "\n"
     r"$\rho_\theta=c_{+\theta}-c_{-\theta}$" + "\n"
     r"$=z(1-z)\,G_\rho$",
     r"$s\to2,\ \rho\to0$ at $z\in\{0,1\}$"),
    (0.68, "#E6F2EC", K.C["green"], "hyperbolic\nreconstruction",
     r"$c_{\pm,\theta}=$" + "\n"
     r"$\frac{1}{2}\!\left(\sqrt{s^2+\rho^2}\pm\rho\right)$",
     r"$c_\pm>0$" + "\n" + r"$c_+-c_-\equiv\rho$"),
]
for x0, fc, ec, title, body, foot in boxes:
    axC.add_patch(FancyBboxPatch((x0, 0.10), 0.31, 0.80,
                  boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec,
                  lw=1.0, transform=axC.transAxes))
    axC.text(x0 + 0.155, 0.79, title, ha="center", va="center", fontsize=8.8,
             fontweight="bold", transform=axC.transAxes)
    axC.text(x0 + 0.155, 0.49, body, ha="center", va="center", fontsize=8.8,
             transform=axC.transAxes)
    axC.text(x0 + 0.155, 0.20, foot, ha="center", va="center", fontsize=7.8,
             color="0.30", transform=axC.transAxes)
for xa, xb in ((0.315, 0.345), (0.650, 0.680)):
    axC.add_patch(FancyArrowPatch((xa + 0.005, 0.50), (xb - 0.005, 0.50),
                  arrowstyle="-|>", mutation_scale=13, lw=1.1,
                  color=K.C["grey"], transform=axC.transAxes))
axC.set_title(r"exact-reservoir charge-balanced ansatz",
              fontsize=11, loc="left", fontweight="bold")
axC.text(0.5, 0.035, r"Weyl singular-value check as sensitivity diagnostic; charge identity is exact",
         ha="center", va="center", fontsize=7.1, color="0.30",
         transform=axC.transAxes)
K.panel_label(axC, "C")

# --------------------------------------------------------------------------- #
# Panel D: DeepONet vs FNO
# --------------------------------------------------------------------------- #
axD = fig.add_subplot(gs[1, 1])
axD.axis("off")
axD.set_xlim(0, 1)
axD.set_ylim(0, 1)


def box(ax, x, y, w, h, text, fc, ec, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                 facecolor=fc, edgecolor=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=11, lw=1.0, color=K.C["grey"],
                 shrinkA=1, shrinkB=1))


# DeepONet column
axD.text(0.245, 0.950, "multi-scale radial PI-DeepONet", ha="center",
         va="center", fontsize=9.0, fontweight="bold")
box(axD, 0.02, 0.74, 0.45, 0.15,
    r"branch MLP$(\boldsymbol{\mu})\to 3p$", "#EAF3F8", K.C["blue"], 9.0)
box(axD, 0.02, 0.54, 0.45, 0.15,
    r"trunk MLP$(r,z,e^{-s(1-r)})\to p$", "#FBF3E6", K.C["orange"], 9.0)
box(axD, 0.02, 0.34, 0.45, 0.14, r"inner product $\sum_k b_k t_k$",
    "#F0F0F0", K.C["grey"], 9.4)
box(axD, 0.02, 0.14, 0.45, 0.14, "exact-boundary ansatz",
    "#E6F2EC", K.C["green"], 9.4)
arrow(axD, 0.245, 0.74, 0.245, 0.69)
arrow(axD, 0.245, 0.54, 0.245, 0.49)
arrow(axD, 0.245, 0.34, 0.245, 0.29)
axD.text(0.245, 0.045, "36,880 params  ·  ONNX 0.072 ms",
         ha="center", fontsize=9.0, color=K.C["green"])

# FNO column
axD.text(0.755, 0.950, "2D spectral FNO", ha="center",
         va="center", fontsize=9.4, fontweight="bold")
box(axD, 0.53, 0.70, 0.45, 0.16,
    r"lift: $[r,z]\oplus\boldsymbol{\mu}\to w=24$", "#EAF3F8", K.C["blue"], 9.0)
box(axD, 0.53, 0.48, 0.45, 0.16,
    r"$\times 4$ real-DFT spectral conv" + "\n"
    r"($8\times12$ modes) + FiLM", "#FBF3E6", K.C["orange"], 9.0)
box(axD, 0.53, 0.26, 0.45, 0.16,
    r"project $\to 3$ fields" + "\n" + "exact-boundary ansatz",
    "#E6F2EC", K.C["green"], 9.0)
arrow(axD, 0.755, 0.70, 0.755, 0.65)
arrow(axD, 0.755, 0.48, 0.755, 0.43)
axD.text(0.755, 0.045, "460,488 params  ·  ONNX 1.04 ms",
         ha="center", fontsize=9.0, color=K.C["orange"])
axD.set_title(r"parameter-conditioned operator backbones",
              fontsize=11, loc="left", fontweight="bold")
K.panel_label(axD, "D")

save_data = {
    "lam_over_R": lam,
    "abs_qw": qw,
    "mu": mu,
    "benchmark_index": np.array([special]),
}
K.save_fig(fig, "fig1_workflow_and_architecture", save_data)
