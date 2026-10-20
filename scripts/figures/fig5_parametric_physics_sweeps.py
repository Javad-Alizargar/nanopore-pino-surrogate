"""Figure 5: continuous parametric physics captured by the operator vs SG/FV."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPTS, "sg_baseline", "src"))
import generate_parametric_dataset as G  # noqa: E402

dat = K.load_dataset()
m = K.load_model("deeponet", "reparam", 42)

def op_mean_current(mu_rows):
    pred, ipred = K.predict_mu(m, np.asarray(mu_rows, float))
    return ipred.mean(1)

def sg_mean_current(V, C0, q, R):
    sol, cfg, grid, meth, nc = G.solve_state(V, C0, q, R)
    return float((G.physical_current(sol, grid, cfg) * 1e12).mean())

fig = plt.figure(figsize=(11.2, 3.4))
gs = GridSpec(1, 3, figure=fig, wspace=0.46)

# ---- Panel A: I-V curves for R in {3,5,7} -------------------------------
axA = fig.add_subplot(gs[0, 0])
Vs = np.linspace(-50, 50, 21)
cols = {3: K.C["green"], 5: K.C["blue"], 7: K.C["orange"]}
sg_V = np.array([-50, -25, 0, 25, 50])
for R in (3, 5, 7):
    mus = [[V, 100.0, -0.01, R] for V in Vs]
    axA.plot(Vs, op_mean_current(mus), color=cols[R], lw=1.5,
             label=f"operator $R={R}$ nm")
    axA.plot(sg_V, [sg_mean_current(V, 100.0, -0.01, R) for V in sg_V],
             ls="none", marker="o", ms=4, mfc="white", mec=cols[R], mew=1.1,
             label=f"SG/FV $R={R}$ nm")
axA.set_xlabel(r"$V_\mathrm{app}$  (mV)"); axA.set_ylabel(r"mean current  $\bar I$  (pA)")
axA.set_title(r"$I$–$V$ characteristics  ($C_0{=}100$, $q_w{=}-0.01$)", fontsize=8.8)
axA.legend(frameon=False, fontsize=6.2, ncol=1)
K.panel_label(axA, "A", x=-0.13, y=1.13)

# ---- Panel B: near-wall enrichment vs salt ------------------------------
axB = fig.add_subplot(gs[0, 1])
C0s = np.linspace(10, 200, 25)
mus = [[50.0, c, -0.01, 5.0] for c in C0s]
pred, _ = K.predict_mu(m, np.asarray(mus))
enrich = pred["c_plus"][:, -1, 30]          # near wall r=1, z=0.5
axB.plot(C0s, enrich, color=K.C["blue"], lw=1.5, label="operator $c_+$ (near wall)")
C0sg = [12, 30, 60, 120, 200]
esg = []
for c in C0sg:
    sol, cfg, grid, meth, nc = G.solve_state(50.0, c, -0.01, 5.0)
    esg.append(sol.c_plus[-1, 30])
axB.plot(C0sg, esg, ls="none", marker="o", ms=4, mfc="white",
         mec=K.C["blue"], mew=1.1, label="SG/FV")
axB.set_xscale("log")
axB.set_xlabel(r"bulk salt  $C_0$  (mol m$^{-3}$)")
axB.set_ylabel(r"near-wall $c_+$  (dimensionless)")
axB.set_title("Debye screening transition", fontsize=8.8)
axB.legend(frameon=False, fontsize=6.4)
K.panel_label(axB, "B", x=-0.13, y=1.13)

# ---- Panel C: radial profiles at several lambda_D/R ---------------------
axC = fig.add_subplot(gs[0, 2])
for c, col in [(20, K.C["vermillion"]), (100, K.C["blue"]), (200, K.C["green"])]:
    pred, _ = K.predict_mu(m, np.array([[50.0, c, -0.01, 5.0]]))
    axC.plot(K.R, pred["rho"][0, :, 30], color=col, lw=1.4,
             label=f"operator $C_0={c}$")
    sol, cfg, grid, meth, nc = G.solve_state(50.0, c, -0.01, 5.0)
    axC.plot(grid.r, sol.c_plus[:, 30] - sol.c_minus[:, 30], ls="none", marker="o",
             ms=3.5, mfc="white", mec=col, mew=1.0, label=f"SG/FV $C_0={c}$")
axC.set_xlabel(r"$r/R$"); axC.set_ylabel(r"net charge  $\rho(r,0.5)$")
axC.set_title(r"EDL overlap vs salinity  ($\lambda_D/R$)", fontsize=8.8)
axC.legend(frameon=False, fontsize=5.6, ncol=1)
K.panel_label(axC, "C", x=-0.13, y=1.13)

sdata = {"Vs": Vs, "sg_V": sg_V, "C0s": C0s, "C0sg": np.array(C0sg),
         "enrich_operator": enrich, "enrich_sg": np.array(esg),
         "r": K.R}
K.save_fig(fig, "fig5_parametric_physics_sweeps", sdata)
