"""Figure 3: Hypothesis H3 - charge-balanced reparameterization vs direct c,
with the Stage-10 trunk ablation and Fisher degeneracy plane."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K

stat = K.load_statistics()
dat = K.load_dataset()
sgl = stat["summary_multiseed_global_rel_l2"]
h3 = stat["h3_conditioning"]

fig = plt.figure(figsize=(13.0, 6.4))
gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.34)
width = 0.36

# ---- Panel A: reparam vs direct_c (rho rel L2, interp_test) -------------
axA = fig.add_subplot(gs[0, 0])
families = ["DeepONet", "FNO-2D"]
rp = [sgl["deeponet_reparam_data_plus_physics"]["interp_test"]["grel_l2_rho"],
      sgl["fno_reparam_data_plus_physics"]["interp_test"]["grel_l2_rho"]]
dc = [sgl["deeponet_direct_c_data_plus_physics"]["interp_test"]["grel_l2_rho"],
      sgl["fno_direct_c_data_plus_physics"]["interp_test"]["grel_l2_rho"]]
x = np.arange(2)
axA.bar(x - width / 2, [v["mean"] for v in rp], width, yerr=[v["sd"] for v in rp],
        capsize=3, color=K.C["blue"], label="reparam (charge-balanced)")
axA.bar(x + width / 2, [v["mean"] for v in dc], width, yerr=[v["sd"] for v in dc],
        capsize=3, color=K.C["vermillion"], label="direct $c$")
for xi, (a, b) in enumerate(zip(rp, dc)):
    axA.text(xi, max(a["mean"], b["mean"]) * 1.04,
             f"{b['mean']/a['mean']:.1f}$\\times$", ha="center", fontsize=8.5,
             color=K.C["black"])
axA.set_xticks(x); axA.set_xticklabels(families)
axA.set_ylabel(r"net-charge rel. $L_2$ error  $\rho$")
axA.set_ylim(0, 1.12)
axA.set_title("interp_test ($N{=}48$), 3-seed mean $\\pm$ SD", fontsize=9)
axA.legend(frameon=False, fontsize=7)
K.panel_label(axA, "A", x=-0.15)

# ---- Panel B: quartile breakdown by |q_w| -------------------------------
axB = fig.add_subplot(gs[0, 1])
qkeys = list(h3["deeponet_data_plus_physics"]["reparam"]["by_abs_qw_quartile"].keys())
xq = np.arange(len(qkeys))
for off, (kind, col, lab) in enumerate([("deeponet", K.C["blue"], "DeepONet reparam"),
                                        ("fno", K.C["orange"], "FNO reparam")]):
    vals = [h3[f"{kind}_data_plus_physics"]["reparam"]["by_abs_qw_quartile"][k] for k in qkeys]
    axB.bar(xq + (off - 0.5) * width, vals, width, color=col, label=lab)
for off, (kind, col, lab) in enumerate([("deeponet", K.C["vermillion"], "DeepONet direct $c$"),
                                        ("fno", K.C["purple"], "FNO direct $c$")]):
    vals = [h3[f"{kind}_data_plus_physics"]["direct_c"]["by_abs_qw_quartile"][k] for k in qkeys]
    axB.bar(xq + (off + 0.5) * width, vals, width, color=col, hatch="//",
            edgecolor="white", linewidth=0.4, label=lab)
short = [k.replace("[", "").replace("]", "").split(",") for k in qkeys]
axB.set_xticks(xq)
axB.set_xticklabels([f"{float(a):.2g}–{float(b):.2g}" for a, b in short],
                    fontsize=7, rotation=18, ha="right")
axB.set_xlabel(r"surface-charge magnitude quartile  $|q_w|$")
axB.set_ylabel(r"$\rho$ rel. $L_2$ error")
axB.set_title("weak-charge cancellation regime", fontsize=9)
axB.legend(frameon=False, fontsize=6.4, ncol=2)
K.panel_label(axB, "B", x=-0.13)

# ---- Panel C: physics-loss ablation -------------------------------------
axC = fig.add_subplot(gs[0, 2])
metrics = [("wall_deriv_err_mean", r"$|\partial_r\psi|_{r=1}-q_w|$"),
           ("poisson_residual_mean", "interior Poisson resid.")]
x = np.arange(len(metrics))
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
    axC.bar(x + (i - 1.5) * 0.20, ph[(kind, cond)], 0.20, color=col,
            edgecolor="0.4", linewidth=0.3, label=lab)
axC.set_xticks(x); axC.set_xticklabels([l for _, l in metrics], fontsize=7.5)
axC.set_yscale("log")
axC.set_ylabel("interior residual (log)")
axC.set_title("physics-loss ablation (interp_test)", fontsize=9)
axC.legend(frameon=False, fontsize=6.3)
K.panel_label(axC, "C", x=-0.13, y=1.10)

# ---- Panel D: radial rho cross-section on a weak-charge test state ------
axD = fig.add_subplot(gs[1, 0])
interp = dat["interp_idx"]
weak = interp[np.argmin(np.abs(dat["mu"][interp, 2]))]
mu_w = dat["mu"][weak:weak + 1]
m_rp = K.load_model("deeponet", "reparam", 42)
m_dc = K.load_model("deeponet", "direct_c", 42)
p_rp, _ = K.predict_mu(m_rp, mu_w)
p_dc, _ = K.predict_mu(m_dc, mu_w)
rz = 30
axD.plot(K.R, dat["ref"]["rho"][weak, :, rz], color="k", lw=2.2, label="SG/FV reference")
axD.plot(K.R, p_rp["rho"][0, :, rz], color=K.C["blue"], lw=1.5, ls="--",
         label="DeepONet reparam")
axD.plot(K.R, p_dc["rho"][0, :, rz], color=K.C["vermillion"], lw=1.3, ls=":",
         label="DeepONet direct $c$")
axD.set_xlabel(r"$r/R$"); axD.set_ylabel(r"$\rho(r,\,z{=}0.5)$")
axD.set_title(r"weak-charge state: $q_w=" + f"{dat['mu'][weak,2]:.4f}" + r"$, $R="
              + f"{dat['mu'][weak,3]:.1f}" + r"$ nm", fontsize=9)
axD.legend(frameon=False, fontsize=7); K.panel_label(axD, "D", x=-0.20, y=1.05)

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
K.panel_label(axE, "E", x=-0.16, y=1.06)

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
K.panel_label(axF, "F", x=-0.16, y=1.06)

sdata = {
    "panelA_families": np.array(families, dtype=object),
    "panelA_reparam_mean": np.array([v["mean"] for v in rp]),
    "panelA_reparam_sd": np.array([v["sd"] for v in rp]),
    "panelA_direct_mean": np.array([v["mean"] for v in dc]),
    "panelA_direct_sd": np.array([v["sd"] for v in dc]),
    "panelB_qw_quartile_labels": np.array(qkeys, dtype=object),
    "panelB_deeponet_reparam": np.array([h3["deeponet_data_plus_physics"]["reparam"]["by_abs_qw_quartile"][k] for k in qkeys]),
    "panelB_deeponet_direct": np.array([h3["deeponet_data_plus_physics"]["direct_c"]["by_abs_qw_quartile"][k] for k in qkeys]),
    "panelB_fno_reparam": np.array([h3["fno_data_plus_physics"]["reparam"]["by_abs_qw_quartile"][k] for k in qkeys]),
    "panelB_fno_direct": np.array([h3["fno_data_plus_physics"]["direct_c"]["by_abs_qw_quartile"][k] for k in qkeys]),
    "panelC_metrics": np.array([m for m, _ in metrics], dtype=object),
    "panelC_deeponet_phys": np.array(ph[("deeponet", "phys")]),
    "panelC_deeponet_data": np.array(ph[("deeponet", "data")]),
    "panelC_fno_phys": np.array(ph[("fno", "phys")]),
    "panelC_fno_data": np.array(ph[("fno", "data")]),
    "panelD_weak_index": weak, "panelD_mu": mu_w[0], "r": K.R,
    "panelD_rho_sg": dat["ref"]["rho"][weak, :, rz],
    "panelD_rho_reparam": p_rp["rho"][0, :, rz],
    "panelD_rho_direct": p_dc["rho"][0, :, rz],
    "panelE_trunk_order": np.array(order, dtype=object),
    "panelE_rho_rel": np.array(rho_rel), "panelE_rho_sd": np.array(rho_sd),
    "panelE_wall_rel": np.array(wall_rel), "panelE_wall_sd": np.array(wall_sd),
    "panelF_lambda": Lc, "panelF_qw": Qc, "panelF_log10_cond": Kc,
    "panelF_dlnI_dlnR": dlr, "panelF_dlnI_dlnq": dlq,
}
K.save_fig(fig, "fig3_h3_charge_reparameterization_ablation", sdata)
