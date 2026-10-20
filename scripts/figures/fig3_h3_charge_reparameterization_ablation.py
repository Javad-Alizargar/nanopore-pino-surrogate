"""Figure 3: Hypothesis H3 - four-way constraint-scheme benchmark (Stage 11),
physics-loss weight sweep, physics-loss ablation, trunk ablation and Fisher
degeneracy plane."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K

stat = K.load_statistics()
dat = K.load_dataset()
sgl = stat["summary_multiseed_global_rel_l2"]
schemes = stat["stage11_constraint_schemes_benchmark"]["summary"]
sweep = stat["stage11_physics_weight_sweep"]

ORDER = ["hyperbolic_reparam", "linear_ortho_projection",
         "soft_charge_penalty", "direct_c"]
PRETTY = {"hyperbolic_reparam": "reparam\n(ours)",
          "linear_ortho_projection": "linear\nproj.",
          "soft_charge_penalty": "soft\npenalty",
          "direct_c": "direct $c$"}
SCOL = {"hyperbolic_reparam": K.C["blue"],
        "linear_ortho_projection": K.C["orange"],
        "soft_charge_penalty": K.C["vermillion"],
        "direct_c": K.C["purple"]}

fig = plt.figure(figsize=(13.0, 6.4))
gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.42)
width = 0.36

# ---- Panel A: four-way constraint-scheme accuracy (rho rel L2) ----------
axA = fig.add_subplot(gs[0, 0])
x = np.arange(len(ORDER))
int_m = [schemes[s]["interp"]["rel_l2_rho"]["mean"] for s in ORDER]
int_s = [schemes[s]["interp"]["rel_l2_rho"]["sd"] for s in ORDER]
ood_m = [schemes[s]["ood"]["rel_l2_rho"]["mean"] for s in ORDER]
ood_s = [schemes[s]["ood"]["rel_l2_rho"]["sd"] for s in ORDER]
axA.bar(x - width / 2, int_m, width, yerr=int_s, capsize=2.5,
        color=[SCOL[s] for s in ORDER], label="interp")
axA.bar(x + width / 2, ood_m, width, yerr=ood_s, capsize=2.5,
        color=[SCOL[s] for s in ORDER], alpha=0.45, hatch="//",
        edgecolor="white", linewidth=0.4, label="OOD")
axA.set_xticks(x)
axA.set_xticklabels([PRETTY[s] for s in ORDER], fontsize=7.5)
axA.set_ylabel(r"net-charge rel. $L_2$ error  $\rho$")
axA.set_ylim(0, 1.15)
axA.set_title("4-way constraint schemes, 3-seed", fontsize=9)
axA.legend(frameon=False, fontsize=7)
axA.text(0.02, 0.93, "solid: interp  |  hatched: OOD", transform=axA.transAxes,
         fontsize=6.2, color="0.3")
K.panel_label(axA, "A", x=-0.16, y=1.15)

# ---- Panel B: identity-consistency mismatch + wall-gradient error -------
axB = fig.add_subplot(gs[0, 1])
xb = np.arange(len(ORDER))
cons = [max(schemes[s]["interp"]["consistency_err"]["mean"], 1e-10) for s in ORDER]
wall = [schemes[s]["interp"]["wall_deriv_err"]["mean"] for s in ORDER]
axB.bar(xb - width / 2, cons, width, color=K.C["sky"],
        edgecolor="0.35", linewidth=0.3, label=r"$|\rho-\rho_{\rm latent}|$ mismatch")
axB.bar(xb + width / 2, wall, width, color=K.C["orange"],
        edgecolor="0.35", linewidth=0.3, label=r"wall-gradient error")
for xi, c in enumerate(cons):
    if c <= 1e-9:
        axB.text(xi - width / 2, c, "exact", ha="center", va="bottom", fontsize=5.6,
                 rotation=90, color="0.25")
axB.set_yscale("log")
axB.set_xticks(xb)
axB.set_xticklabels([PRETTY[s] for s in ORDER], fontsize=7.5)
axB.set_ylabel("error (log)")
axB.set_title("identity consistency vs wall accuracy", fontsize=9)
axB.legend(frameon=False, fontsize=5.9)
K.panel_label(axB, "B", x=-0.15, y=1.15)

# ---- Panel C: physics-loss ablation -------------------------------------
axC = fig.add_subplot(gs[0, 2])
metrics = [("wall_deriv_err_mean", r"$|\partial_r\psi|_{r=1}-q_w|$"),
           ("poisson_residual_mean", "interior Poisson resid.")]
xc = np.arange(len(metrics))
ph = {("deeponet", "phys"): [], ("deeponet", "data"): [],
      ("fno", "phys"): [], ("fno", "data"): []}
for m, _ in metrics:
    ph[("deeponet", "phys")].append(sgl["deeponet_reparam_data_plus_physics"]["interp_test"][m]["mean"])
    ph[("deeponet", "data")].append(sgl["deeponet_reparam_data_only"]["interp_test"][m]["mean"])
    ph[("fno", "phys")].append(sgl["fno_reparam_data_plus_physics"]["interp_test"][m]["mean"])
    ph[("fno", "data")].append(sgl["fno_reparam_data_only"]["interp_test"][m]["mean"])
combos = [("deeponet", "phys", K.C["blue"], "DeepONet + physics"),
          ("deeponet", "data", K.C["sky"], "DeepONet data-only"),
          ("fno", "phys", K.C["orange"], "FNO + physics"),
          ("fno", "data", K.C["yellow"], "FNO data-only")]
for i, (kind, cond, col, lab) in enumerate(combos):
    axC.bar(xc + (i - 1.5) * 0.20, ph[(kind, cond)], 0.20, color=col,
            edgecolor="0.4", linewidth=0.3, label=lab)
axC.set_xticks(xc); axC.set_xticklabels([l for _, l in metrics], fontsize=7.5)
axC.set_yscale("log")
axC.set_ylabel("interior residual (log)")
axC.set_title("physics-loss ablation (interp_test)", fontsize=9)
axC.legend(frameon=False, fontsize=6.3)
K.panel_label(axC, "C", x=-0.13, y=1.15)

# ---- Panel D: physics-loss weight sensitivity (Pareto) ------------------
axD = fig.add_subplot(gs[1, 0])
drows = [r for r in sweep["rows"] if r["split"] == "interp"]
dlam = np.array([r["lambda_phys"] for r in drows])
drho = np.array([r["rel_l2_rho"] for r in drows])
dwall = np.array([r["wall_deriv_err"] for r in drows])
dpois = np.array([r["poisson_residual"] for r in drows])
dx = np.where(dlam <= 0, 1e-5, dlam)
axD.plot(dx, drho, "o-", color=K.C["blue"], label=r"$\rho$ rel.-$L_2$ (acc.)")
axD.set_xscale("log")
axD.set_xlabel(r"physics-loss weight  $\lambda_{\rm phys}$")
axD.set_ylabel(r"$\rho$ rel. $L_2$ error", color=K.C["blue"])
axD.tick_params(axis="y", labelcolor=K.C["blue"])
axD.axvline(1e-3, color="0.5", lw=0.8, ls=":", label="_nolegend_")
axD.text(1e-3, axD.get_ylim()[1], " 1e-3", fontsize=6, va="top", color="0.4")
axD2 = axD.twinx()
axD2.plot(dx, dwall, "s--", color=K.C["orange"], label="wall-grad err")
axD2.plot(dx, dpois, "^:", color=K.C["purple"], label="Poisson resid.")
axD2.set_yscale("log")
axD2.set_ylabel("residual (log)", color=K.C["orange"])
axD2.tick_params(axis="y", labelcolor=K.C["orange"])
lines = [l for l in axD.get_lines() + axD2.get_lines()
         if not l.get_label().startswith("_")]
axD.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=6.0,
           loc="upper center")
axD.set_title(r"$\lambda_{\rm phys}$ sensitivity (seed 42)", fontsize=9)
K.panel_label(axD, "D", x=-0.20, y=1.15)

# ---- Panel E: trunk ablation, normalised to the plain MLP trunk ---------
abl = stat["stage10_radial_trunk_ablation"]["summary"]
order = ["mlp_trunk", "radial_k1", "radial_k3", "radial_k10", "radial_k30",
         "radial_multiscale"]
pretty = {"mlp_trunk": "MLP", "radial_k1": r"$k{=}1$", "radial_k3": r"$k{=}3$",
          "radial_k10": r"$k{=}10$", "radial_k30": r"$k{=}30$",
          "radial_multiscale": "multi"}
base_rho = abl["mlp_trunk"]["interp_rel_l2_rho"]["mean"]
base_wall = abl["mlp_trunk"]["interp_wall_deriv_err"]["mean"]
rho_rel = [abl[c]["interp_rel_l2_rho"]["mean"] / base_rho for c in order]
wall_rel = [abl[c]["interp_wall_deriv_err"]["mean"] / base_wall for c in order]
rho_sd = [abl[c]["interp_rel_l2_rho"]["sd"] / base_rho for c in order]
wall_sd = [abl[c]["interp_wall_deriv_err"]["sd"] / base_wall for c in order]
axE = fig.add_subplot(gs[1, 1])
xe = np.arange(len(order))
axE.axhline(1.0, color="0.5", lw=0.8, ls="--", zorder=0)
axE.bar(xe - 0.19, rho_rel, 0.36, yerr=rho_sd, capsize=2.5, color=K.C["blue"],
        label=r"$\rho$ rel.-$L_2$")
axE.bar(xe + 0.19, wall_rel, 0.36, yerr=wall_sd, capsize=2.5, color=K.C["orange"],
        label=r"wall-gradient error")
axE.set_xticks(xe); axE.set_xticklabels([pretty[c] for c in order], fontsize=7.5)
axE.set_xlabel("trunk basis (radial decay $k$)")
axE.set_ylabel("error relative to MLP trunk")
axE.set_title("trunk ablation (500 ep, 3-seed)", fontsize=9)
axE.legend(frameon=False, fontsize=6.6)
K.panel_label(axE, "E", x=-0.18, y=1.15)

# ---- Panel F: Fisher condition-number ridge ----------------------------
fp = stat["stage10_fisher_plane"]
lam = np.array([g["lambda_D_over_R"] for g in fp])
qw = np.array([g["q_w"] for g in fp])
cn = np.array([g["cond"] for g in fp])
dlq = np.array([g["dlnI_dlnq"] for g in fp])
dlr = np.array([g["dlnI_dlnR"] for g in fp])
Lc = np.unique(lam); Qc = np.unique(qw)
Kc = np.full((Lc.size, Qc.size), np.nan)
for g in fp:
    Kc[np.argmin(np.abs(Lc - g["lambda_D_over_R"])),
       np.argmin(np.abs(Qc - g["q_w"]))] = np.log10(max(g["cond"], 1e-6))
axF = fig.add_subplot(gs[1, 2])
mesh = axF.pcolormesh(Lc, Qc * 1e3, Kc.T, cmap="viridis", shading="auto")
axF.contour(Lc, Qc * 1e3, Kc.T, levels=[9.0, 10.0, 11.0],
            colors="white", linewidths=1.2, alpha=0.85)
axF.set_xlabel(r"Debye ratio  $\lambda_D/R$")
axF.set_ylabel(r"surface charge  $|q_w|$  ($\times10^{-3}$)")
axF.set_title(r"Fisher conditioning $\log_{10}\kappa(\mathcal{F})$ (white: $10^{9,10,11}$)"
              "\n" + rf"$\partial\ln I/\partial\ln R\!\approx\!{np.nanmean(dlr):.2f}$, "
              rf"$|\partial\ln I/\partial\ln q_w|\!<\!{np.nanmax(np.abs(dlq)):.1e}$",
              fontsize=8.0)
cb = fig.colorbar(mesh, ax=axF, pad=0.02, fraction=0.046)
cb.set_label(r"$\log_{10}\kappa(\mathcal{F})$", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
K.panel_label(axF, "F", x=-0.18, y=1.15)

sdata = {
    "panelA_schemes": np.array(ORDER, dtype=object),
    "panelA_interp_mean": np.array(int_m), "panelA_interp_sd": np.array(int_s),
    "panelA_ood_mean": np.array(ood_m), "panelA_ood_sd": np.array(ood_s),
    "panelB_schemes": np.array(ORDER, dtype=object),
    "panelB_consistency": np.array(cons), "panelB_wall": np.array(wall),
    "panelC_metrics": np.array([m for m, _ in metrics], dtype=object),
    "panelC_deeponet_phys": np.array(ph[("deeponet", "phys")]),
    "panelC_deeponet_data": np.array(ph[("deeponet", "data")]),
    "panelC_fno_phys": np.array(ph[("fno", "phys")]),
    "panelC_fno_data": np.array(ph[("fno", "data")]),
    "panelD_lambda": dlam, "panelD_rho": drho,
    "panelD_wall": dwall, "panelD_poisson": dpois,
    "panelE_trunk_order": np.array(order, dtype=object),
    "panelE_rho_rel": np.array(rho_rel), "panelE_rho_sd": np.array(rho_sd),
    "panelE_wall_rel": np.array(wall_rel), "panelE_wall_sd": np.array(wall_sd),
    "panelF_lambda": Lc, "panelF_qw": Qc, "panelF_log10_cond": Kc,
    "panelF_dlnI_dlnR": dlr, "panelF_dlnI_dlnq": dlq,
}
K.save_fig(fig, "fig3_h3_charge_reparameterization_ablation", sdata)
