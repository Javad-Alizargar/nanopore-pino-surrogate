"""Figure 4: Hypotheses H1 & H2 - generalization across the 4D parameter space."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K
from operators.losses_and_metrics import sg_current_pA

dat = K.load_dataset()
T = K.T
GRID = K.GRID

splits = {"interp_test": dat["interp_idx"], "ood_geometry_voltage": dat["oodB"],
          "ood_charge_salt": dat["oodA"]}
split_color = {"interp_test": K.C["blue"], "ood_geometry_voltage": K.C["green"],
               "ood_charge_salt": K.C["vermillion"]}

# per-state errors, averaged over the 3 seeds of the primary model
models = [K.load_model("deeponet", "reparam", s) for s in (42, 43, 44)]

def per_state(idx):
    rel = {k: [] for k in ["psi", "c_plus", "rho", "I"]}
    ipred_all, iref_all = [], []
    for m in models:
        pred, ipred = K.predict_mu(m, dat["mu"][idx])
        ref = dat["ref"]
        for f in ["psi", "c_plus", "rho"]:
            num = np.sqrt(((pred[f] - ref[f][idx]) ** 2 * K.AREA).sum((1, 2)))
            den = np.sqrt((ref[f][idx] ** 2 * K.AREA).sum((1, 2)))
            rel[f].append(num / np.maximum(den, 1e-30))
        iref = ref["i_z"][idx]
        numI = np.sqrt(((ipred - iref) ** 2).mean(1))
        denI = np.sqrt((iref ** 2).mean(1))
        rel["I"].append(numI / np.maximum(denI, 1e-30))
        ipred_all.append(ipred.mean(1)); iref_all.append(iref.mean(1))
    rel = {k: np.nanmean(np.array(v), 0) for k, v in rel.items()}
    return rel, np.nanmean(ipred_all, 0), iref_all[0]

errs, ipred_mean, iref_mean = {}, {}, {}
for sp, idx in splits.items():
    e, ip, ir = per_state(idx)
    errs[sp] = e; ipred_mean[sp] = ip; iref_mean[sp] = ir
lam = dat["lambda_D_over_R"]

fig = plt.figure(figsize=(13.0, 6.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
              width_ratios=[1.25, 1.0, 1.0])

# ---- Panel A: per-state rel-L2 distributions ---------------------------
axA = fig.add_subplot(gs[:, 0])
metrics = ["psi", "c_plus", "rho", "I"]
mlabels = [r"$\psi$", r"$c_+$", r"$\rho$", r"$I(z)$"]
pos = 0; ticks = []; tlabels = []
for mi, (m, lab) in enumerate(zip(metrics, mlabels)):
    for si, sp in enumerate(splits):
        d = errs[sp][m]
        bp = axA.boxplot([d], positions=[pos], widths=0.66, showfliers=False,
                         patch_artist=True, medianprops=dict(color="k", lw=1.1),
                         boxprops=dict(facecolor=split_color[sp], alpha=0.55,
                                       edgecolor="0.3", lw=0.6),
                         whiskerprops=dict(color="0.3", lw=0.6),
                         capprops=dict(color="0.3", lw=0.6))
        jitter = (np.random.RandomState(0 + mi * 7 + si).rand(len(d)) - 0.5) * 0.36
        axA.scatter(pos + jitter, d, s=6, color=split_color[sp], alpha=0.6,
                    edgecolor="none", zorder=3)
        ticks.append(pos); tlabels.append(lab if si == 1 else "")
        pos += 1
    pos += 0.9
axA.set_yscale("log")
axA.set_xticks(ticks)
axA.set_xticklabels([lab for lab in mlabels for _ in splits])
axA.set_ylabel("per-state relative $L_2$ error")
axA.set_title("per-state error distributions (reparam, 3-seed)", fontsize=9)
handles = [plt.Line2D([], [], color=split_color[sp], lw=6, alpha=0.6, label=sp)
           for sp in splits]
axA.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
K.panel_label(axA, "A", x=-0.14, y=1.13)

# ---- Panel B: error vs Debye ratio --------------------------------------
gsB = gs[0, 1].subgridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.95)
axB = fig.add_subplot(gsB[0])
allidx = np.concatenate([splits[sp] for sp in splits])
label_map = {sp: split_color[sp] for sp in splits}
for sp in splits:
    idx = splits[sp]
    axB.scatter(lam[idx], errs[sp]["rho"], s=16, facecolor=label_map[sp],
                edgecolor="white", lw=0.3, label=f"$\\rho$ {sp}")
    axB.scatter(lam[idx], errs[sp]["I"], s=16, marker="^", facecolor="none",
                edgecolor=label_map[sp], lw=0.9)
axB.set_yscale("log")
axB.set_xlabel(r"Debye ratio  $\lambda_D/R$")
axB.set_ylabel("per-state relative $L_2$ error")
axB.set_title("error vs EDL overlap  (filled: $\\rho$, open\ntriangles: $I(z)$)",
              fontsize=8.6)
axB.legend(frameon=False, fontsize=6.2)
K.panel_label(axB, "B", x=-0.08, y=1.45)

# ---- Panel B2: OOD error budget by spatial zone -------------------------
oodstat = K.load_statistics()["ood_mechanistic_breakdown"]["backbones"]
axB2 = fig.add_subplot(gsB[1])
zone_order = ["reservoir_corners", "wall_edl", "core"]
zone_lab = ["corners", "wall EDL", "core"]
zone_col = [K.C["vermillion"], K.C["orange"], K.C["blue"]]
bbars = ["deeponet", "fno"]
for bi, bb in enumerate(bbars):
    for fi, field in enumerate(("psi", "rho")):
        shares = [oodstat[bb]["zones"][z][field]["share_of_sq_error"] for z in zone_order]
        left = 0.0
        for si, (s, zl) in enumerate(zip(shares, zone_lab)):
            axB2.barh(bi * 2 + fi, 100 * s, left=100 * left, height=0.72,
                      color=zone_col[si], alpha=0.85 if bi == 0 else 0.6,
                      edgecolor="white", lw=0.4,
                      hatch=None if bi == 0 else "///")
            left += s
axB2.set_yticks([0, 1, 2, 3])
axB2.set_yticklabels([r"$\psi$ DN", r"$\rho$ DN", r"$\psi$ FNO", r"$\rho$ FNO"],
                     fontsize=6.0)
axB2.set_xlim(0, 100)
axB2.set_xlabel("share of OOD squared error (%)", fontsize=6.2, labelpad=1)
axB2.tick_params(axis="x", labelsize=5.8)
excl = oodstat["deeponet"]["parametric_exclusive"]
lab_map = {"combined_charge_salt": "charge+salt", "charge_only": "charge",
           "salt_only": "salt", "narrow_radius": "narrow $R$",
           "high_voltage": "high $V$", "other": "other"}
txt = "  ".join(f"{lab_map.get(k, k)} {100*v['share_sq_rho']:.0f}%"
                for k, v in excl.items())
axB2.set_title("OOD budget by zone\ncharge+salt 94%, charge 6%", fontsize=5.8, pad=2)



# ---- Panel C: parity of mean axial current ------------------------------
axC = fig.add_subplot(gs[1, 1])
ip = np.concatenate([ipred_mean[sp] for sp in splits])
ir = np.concatenate([iref_mean[sp] for sp in splits])
lo, hi = min(ir.min(), ip.min()) * 1.1, max(ir.max(), ip.max()) * 1.1
axC.plot([lo, hi], [lo, hi], color="0.4", lw=0.9, ls="--", zorder=1)
for sp in splits:
    axC.scatter(iref_mean[sp], ipred_mean[sp], s=22, facecolor=label_map[sp],
                edgecolor="white", lw=0.4, label=sp, zorder=3)
axC.set_xlabel(r"SG/FV mean axial current  $\bar I$  (pA)")
axC.set_ylabel(r"operator  $\bar I_\theta$  (pA)")
r2 = 1 - np.sum((ip - ir) ** 2) / np.sum((ir - ir.mean()) ** 2)
axC.set_title(f"current parity, 96 test states  ($R^2$={r2:.4f})", fontsize=8.8)
axC.legend(frameon=False, fontsize=6.4, loc="upper left")
axC.set_xlim(lo, hi); axC.set_ylim(lo, hi)
# inset residuals
axi = axC.inset_axes([0.60, 0.10, 0.37, 0.30])
axi.axhline(0, color="0.5", lw=0.7)
for sp in splits:
    axi.scatter(iref_mean[sp], ipred_mean[sp] - iref_mean[sp], s=8,
                facecolor=label_map[sp], edgecolor="none")
axi.set_title("residual (pA)", fontsize=6); axi.tick_params(labelsize=5.5)
K.panel_label(axC, "C")

# ---- Panel D: training-set size scaling law -----------------------------
axD = fig.add_subplot(gs[:, 2])
sc = K.load_statistics()["stage10_scaling_law"]
ns = np.array(sc["n_list"], dtype=float)
ssum = sc["summary"]
curves = [
    ("interp_rho", "interp $\\rho$", K.C["blue"], "o", "-"),
    ("ood_rho", "OOD $\\rho$", K.C["vermillion"], "s", "--"),
    ("interp_psi", "interp $\\psi$", K.C["green"], "^", "-."),
]
for key, lab, col, mk, ls in curves:
    ys = np.array([ssum[str(int(n))][key]["mean"] for n in ns])
    sd = np.array([ssum[str(int(n))][key]["sd"] for n in ns])
    axD.errorbar(ns, ys, yerr=sd, color=col, marker=mk, ls=ls, ms=4.5, lw=1.4,
                 capsize=2.5, label=lab)
    fit = sc["power_law_fits"].get(key)
    if fit is not None:
        xx = np.linspace(ns.min(), ns.max(), 50)
        axD.plot(xx, fit["a"] * xx ** (-fit["alpha"]), color=col, lw=0.8, alpha=0.55)
        axD.annotate(rf"$\alpha_{{{key.split('_')[1]}}}={fit['alpha']:.2f}$",
                     (ns[-1], ys[-1]), textcoords="offset points",
                     xytext=(-4, -11 if key == "interp_rho" else 8), fontsize=6.4,
                     color=col, ha="right")
xref = np.array([36.0, 288.0])
axD.plot(xref, 0.35 * (xref / 36.0) ** -0.5, color="0.6", lw=1.0, ls=":",
         label=r"$N^{-1/2}$ reference")
axD.set_xscale("log"); axD.set_yscale("log")
axD.set_xlabel(r"training states  $N_{\mathrm{train}}$")
axD.set_ylabel("relative $L_2$ error")
axD.set_title("data-scaling law (250 ep,\n3-seed mean $\\pm$ SD)", fontsize=9)
axD.legend(frameon=False, fontsize=6.4, loc="lower left")
K.panel_label(axD, "D", x=-0.16, y=1.06)

sdata = {"lam": lam, "metrics": np.array(metrics, dtype=object)}
for sp in splits:
    sdata[f"idx_{sp}"] = splits[sp]
    for m in metrics:
        sdata[f"err_{sp}_{m}"] = errs[sp][m]
    sdata[f"ipred_{sp}"] = ipred_mean[sp]; sdata[f"iref_{sp}"] = iref_mean[sp]
sdata["panelB2_zones"] = np.array(zone_order, dtype=object)
for bb in bbars:
    for field in ("psi", "rho"):
        sdata[f"panelB2_{bb}_{field}_zone_share"] = np.array(
            [oodstat[bb]["zones"][z][field]["share_of_sq_error"] for z in zone_order])
sdata["panelB2_exclusive_categories"] = np.array(list(excl.keys()), dtype=object)
sdata["panelB2_exclusive_share_rho"] = np.array(
    [v["share_sq_rho"] for v in excl.values()])
sdata["panelD_n_train"] = ns
for key, lab, col, mk, ls in curves:
    sdata[f"panelD_{key}_mean"] = np.array([ssum[str(int(n))][key]["mean"] for n in ns])
    sdata[f"panelD_{key}_sd"] = np.array([ssum[str(int(n))][key]["sd"] for n in ns])
sdata["panelD_power_law"] = np.array(
    [[sc["power_law_fits"][k]["alpha"], sc["power_law_fits"][k]["log_a"]]
     for k, *_ in curves])
K.save_fig(fig, "fig4_h1_h2_interp_vs_ood_regimes", sdata)
