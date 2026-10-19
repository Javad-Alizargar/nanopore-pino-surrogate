"""Figure 3: Hypothesis H3 - charge-balanced reparameterization vs direct c."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K

stat = K.load_statistics()
dat = K.load_dataset()
sgl = stat["summary_multiseed_global_rel_l2"]
h3 = stat["h3_conditioning"]

fig = plt.figure(figsize=(10.4, 6.2))
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.26)
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
axB.set_xticklabels([f"{float(a):.2g}–{float(b):.2g}" for a, b in short], fontsize=7.5)
axB.set_xlabel(r"surface-charge magnitude quartile  $|q_w|$")
axB.set_ylabel(r"$\rho$ rel. $L_2$ error")
axB.set_title("weak-charge cancellation regime", fontsize=9)
axB.legend(frameon=False, fontsize=6.4, ncol=2)
K.panel_label(axB, "B", x=-0.13)

# ---- Panel C: radial rho cross-section on a weak-charge test state ------
axC = fig.add_subplot(gs[1, 0])
interp = dat["interp_idx"]
weak = interp[np.argmin(np.abs(dat["mu"][interp, 2]))]
mu_w = dat["mu"][weak:weak + 1]
m_rp = K.load_model("deeponet", "reparam", 42)
m_dc = K.load_model("deeponet", "direct_c", 42)
p_rp, _ = K.predict_mu(m_rp, mu_w)
p_dc, _ = K.predict_mu(m_dc, mu_w)
rz = 30
axC.plot(K.R, dat["ref"]["rho"][weak, :, rz], color="k", lw=2.2, label="SG/FV reference")
axC.plot(K.R, p_rp["rho"][0, :, rz], color=K.C["blue"], lw=1.5, ls="--",
         label="DeepONet reparam")
axC.plot(K.R, p_dc["rho"][0, :, rz], color=K.C["vermillion"], lw=1.3, ls=":",
         label="DeepONet direct $c$")
axC.set_xlabel(r"$r/R$"); axC.set_ylabel(r"$\rho(r,\,z{=}0.5)$")
axC.set_title(r"weak-charge state: $q_w=" + f"{dat['mu'][weak,2]:.4f}" + r"$, $R="
              + f"{dat['mu'][weak,3]:.1f}" + r"$ nm", fontsize=9)
axC.legend(frameon=False, fontsize=7); K.panel_label(axC, "C", x=-0.20, y=1.05)

# ---- Panel D: physics-loss ablation -------------------------------------
axD = fig.add_subplot(gs[1, 1])
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
    axD.bar(x + (i - 1.5) * 0.20, ph[(kind, cond)], 0.20, color=col,
            edgecolor="0.4", linewidth=0.3, label=lab)
axD.set_xticks(x); axD.set_xticklabels([l for _, l in metrics], fontsize=7.5)
axD.set_yscale("log")
axD.set_ylabel("interior residual (log)")
axD.set_title("physics-loss ablation (interp_test)", fontsize=9)
axD.legend(frameon=False, fontsize=6.3)
K.panel_label(axD, "D", x=-0.13, y=1.10)

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
    "panelC_weak_index": weak, "panelC_mu": mu_w[0], "r": K.R,
    "panelC_rho_sg": dat["ref"]["rho"][weak, :, rz],
    "panelC_rho_reparam": p_rp["rho"][0, :, rz],
    "panelC_rho_direct": p_dc["rho"][0, :, rz],
    "panelD_metrics": np.array([m for m, _ in metrics], dtype=object),
    "panelD_deeponet_phys": np.array(ph[("deeponet", "phys")]),
    "panelD_deeponet_data": np.array(ph[("deeponet", "data")]),
    "panelD_fno_phys": np.array(ph[("fno", "phys")]),
    "panelD_fno_data": np.array(ph[("fno", "data")]),
}
K.save_fig(fig, "fig3_h3_charge_reparameterization_ablation", sdata)
