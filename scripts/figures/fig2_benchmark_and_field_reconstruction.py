"""Figure 2: out-of-sample operator prediction on held-out benchmark state #0."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common as K

stat = K.load_statistics()
dat = K.load_dataset()
g = dat["bench_idx"]
mu0 = dat["mu"][g:g + 1]
ref = {k: dat["ref"][k][g] for k in ["psi", "c_plus", "c_minus", "rho", "i_z"]}

m_dn = K.load_model("deeponet", "reparam", 42)
m_fno = K.load_model("fno", "reparam", 42, epochs=300)
pred_dn, i_dn = K.predict_mu(m_dn, mu0)
pred_fno, i_fno = K.predict_mu(m_fno, mu0)
psi_dn, cp_dn, rho_dn = pred_dn["psi"][0], pred_dn["c_plus"][0], pred_dn["rho"][0]
psi_fn, rho_fn = pred_fno["psi"][0], pred_fno["rho"][0]

R, Z = K.R, K.Z
extent = [R[0], R[-1], Z[0], Z[-1]]
zm = 0.5 * (Z[:-1] + Z[1:])
i_ref = ref["i_z"]
dapinn_mean = 196.137225
dapinn_err = 1.02e-3 * dapinn_mean

fig = plt.figure(figsize=(11.6, 6.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.60)

def field(ax, C, title, cmap, letter, vmin=None, vmax=None):
    pc = ax.pcolormesh(R, Z, C.T, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$r/R$"); ax.set_ylabel(r"$z/L$")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    K.panel_label(ax, letter, x=-0.25, y=1.16)

ax = fig.add_subplot(gs[0, 0])
field(ax, psi_dn, r"$\psi_\theta(r,z)$  (DeepONet)", "RdBu_r", "A")
ax = fig.add_subplot(gs[0, 1])
field(ax, cp_dn, r"$c_+^\theta(r,z)$  (DeepONet)", "viridis", "B")
ax = fig.add_subplot(gs[0, 2])
field(ax, rho_dn, r"$\rho_\theta(r,z)=c_+^\theta-c_-^\theta$", "RdBu_r", "C")

ax = fig.add_subplot(gs[1, 0])
e_psi = np.abs(psi_dn - ref["psi"])
field(ax, e_psi, r"$|\psi_\theta-\psi_\mathrm{SG}|$", "magma", "D")
ax = fig.add_subplot(gs[1, 1])
e_rho = np.abs(rho_dn - ref["rho"])
field(ax, e_rho, r"$|\rho_\theta-\rho_\mathrm{SG}|$", "magma", "E")

ax = fig.add_subplot(gs[1, 2])
ax.plot(zm, i_ref, color="k", lw=2.2, label="SG/FV (reference)")
ax.plot(zm, i_dn[0], color=K.C["blue"], lw=1.4, ls="--",
        label=f"DeepONet (reparam)  {i_dn[0].mean():.3f} pA")
ax.plot(zm, i_fno[0], color=K.C["orange"], lw=1.4, ls=":",
        label=f"FNO-2D (reparam, 300ep)  {i_fno[0].mean():.3f} pA")
ax.axhspan(dapinn_mean - dapinn_err, dapinn_mean + dapinn_err,
           color=K.C["grey"], alpha=0.35,
           label="prior DA-PINN  $\\pm$ reported err")
ax.set_xlabel(r"$z/L$"); ax.set_ylabel(r"$I(z)$  (pA)", labelpad=2)
ax.set_title(r"axial current profile", fontsize=9)
ax.legend(fontsize=6.4, loc="best"); K.panel_label(ax, "F", x=-0.25, y=1.16)
ax.set_xlim(0, 1)

fig.suptitle(r"Benchmark state #0:  $V_\mathrm{app}=50$ mV, $C_0=100$ mol m$^{-3}$, "
             r"$q_w=-0.01$, $R=5$ nm  (out-of-sample)", fontsize=10, y=0.99)

sdata = {
    "r": R, "z": Z, "zm": zm, "psi_dn": psi_dn, "cp_dn": cp_dn, "rho_dn": rho_dn,
    "psi_fno": psi_fn, "rho_fno": rho_fn, "psi_sg": ref["psi"],
    "rho_sg": ref["rho"], "c_plus_sg": ref["c_plus"], "i_sg": i_ref,
    "i_dn": i_dn[0], "i_fno": i_fno[0], "dapinn_mean": dapinn_mean,
    "dapinn_err": dapinn_err,
}
K.save_fig(fig, "fig2_benchmark_and_field_reconstruction", sdata)
