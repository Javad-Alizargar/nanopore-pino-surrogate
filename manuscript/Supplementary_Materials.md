# Supplementary Materials

**Manuscript:** A parameter-conditioned physics-informed neural operator for axisymmetric Poisson--Nernst--Planck transport in charged nanopores

**Author:** Javad Alizargar

This supplement documents the dataset, constraint derivation, training protocol, extended metrics, out-of-distribution definitions, inverse-recovery protocol, deployment checks, and reproducibility commands. All numerical values are produced by the archived scripts and result files listed in Section S9.

---

## S1. Dataset, ranges, and splits

The dataset contains 240 steady axisymmetric states on a $21\times61$ node-centered grid (radial $\times$ axial). The pore length is $L=30$ nm; the temperature is $T=298.15$ K; the relative permittivity is $\varepsilon_r=78.5$; the ionic diffusivities are $D_+=1.96\times10^{-9}$ m$^2$ s$^{-1}$ and $D_-=2.03\times10^{-9}$ m$^2$ s$^{-1}$.

| Split | Count | Sampling |
|---|---|---|
| Train | 144 | Latin hypercube in the in-distribution box, plus boundary and corner anchors |
| Interpolation test | 48 | Latin hypercube in the in-distribution box; index 144 is the benchmark state |
| OOD charge/salt | 24 | $q_w$ and/or $C_0$ outside the training box |
| OOD geometry/voltage | 24 | $R$ and/or $V_\mathrm{app}$ outside the training box |
| Total | 240 | |

The in-distribution box is $V_\mathrm{app}\in[-50,50]$ mV, $C_0\in[20,200]$ mol m$^{-3}$, $q_w\in[-0.05,0]$, and $R\in[3,8]$ nm. The generated states extend to $C_0=10.23$ mol m$^{-3}$, $R=2.027$ nm, and $q_w=-0.09816$ in the OOD groups. The web simulator exposes wider slider ranges (voltage $\pm75$ mV, $C_0$ 10--200 mol m$^{-3}$, $q_w$ $-0.1$--0, $R$ 2--8 nm) but flags values outside the in-distribution box as OOD.

Acceptance criteria: residual below $10^{-9}$, finite fields, and $c_\pm>0$ everywhere. The realized maximum residual is 9.936e-10, the minimum cation concentration is 0.9999, and the minimum anion concentration is 0.8034. No state required continuation. The benchmark state (index 144) is $V_\mathrm{app}=50$ mV, $C_0=100$ mol m$^{-3}$, $q_w=-0.01$, $R=5$ nm; its reference mean current is 196.1 pA.

Reference solve cost: median 0.2013 s, mean 0.2391 s on the Apple M1 host. Dataset generation wall time: 59.18 s.

---

## S2. Models and training hyperparameters

| Setting | DeepONet | FNO |
|---|---|---|
| Output parameterization | $(s,\rho)$ + exact boundary | $(s,\rho)$ + exact boundary |
| Inner dimension / width | $p=64$ | width 24 |
| Spectral modes | -- | $8\times12$ |
| Spectral layers | -- | 4 |
| Trainable parameters | 36,880 | 460,488 |
| Epochs | 500 | 110 |
| Batch size | 36 | 36 |
| Learning rate | $2\times10^{-3}$ | $2\times10^{-3}$ |
| Physics weight | $10^{-3}$ | $10^{-3}$ |
| Seeds | 42, 43, 44 | 42, 43, 44 |
| Training time, seed 42 | 50.59 s | 108.5 s |

A separate FNO was trained for 300 epochs (seed 42, 302.2 s) to check whether the 110-epoch budget limited accuracy. The 300-epoch results are reported in Table 5 of the main text. A data-only ablation removes the physics term while keeping the architecture, data, and optimizer fixed.

Loss:

$$
\mathcal{L}=\mathcal{L}_\mathrm{data}+10^{-3}\mathcal{L}_\mathrm{phys},
$$

$$
\mathcal{L}_\mathrm{data}=\frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}}
\frac{1}{N_s}\sum_{s=1}^{N_s}
\left[\frac{f_\theta(s)-f_\mathrm{ref}(s)}
{\max\{\mathrm{RMS}_s(f),\,0.1\,\mathrm{RMS}_\mathrm{global}(f)\}}\right]^2,
$$

where $\mathcal{F}=\{\psi,s,\rho\}$ and $\mathrm{RMS}_s(f)$ is the root-mean-square of field $f$ in state $s$. The per-state scale is floored at 10% of the global RMS to avoid amplifying near-zero states. The physics term is

$$
\mathcal{L}_\mathrm{phys}=\overline{|R_\mathrm{Poisson}|}
+\tfrac12\left(\overline{|R_{\mathrm{cont},+}|}+\overline{|R_{\mathrm{cont},-}|}\right)
+\overline{|R_{\mathrm{wall},\psi}|}
+\overline{|R_{\mathrm{wall},J_+}|}
+\overline{|R_{\mathrm{wall},J_-}|},
$$

where the overbar is a mean over collocation or boundary points and the residuals are the dimensionless Poisson, continuity, wall-potential, and zero-flux residuals. The physics term therefore penalizes the Poisson residual, both continuity residuals, and the wall boundary conditions. The current is not used as a training target; it is computed after the forward pass with the SG face-flux formula, so the reported current errors are an independent check on field consistency.

---

## S3. Derivation of the exact-reservoir and charge-balanced ansatz

### S3.1 Dimensionless PNP system

Lengths are scaled by the pore length $L$, the radial coordinate by the pore radius $R$, potentials by the thermal voltage $V_T=k_BT/e$, and concentrations by the bulk concentration $C_0$. With $\tilde r=r/R\in[0,1]$ and $z\in[0,1]$, the steady dimensionless Poisson equation is

$$
\alpha_r\frac{1}{\tilde r}\frac{\partial}{\partial\tilde r}
\left(\tilde r\frac{\partial\tilde\psi}{\partial\tilde r}\right)
+\frac{\partial^2\tilde\psi}{\partial z^2}
=-\beta\,(\tilde c_+-\tilde c_-),
$$

and the two steady continuity equations are

$$
\alpha_r\frac{1}{\tilde r}\frac{\partial}{\partial\tilde r}
\left(\tilde r\tilde J_{\pm,\tilde r}\right)
+\frac{\partial\tilde J_{\pm,z}}{\partial z}=0,
$$

with

$$
\alpha_r=\left(\frac{L}{R}\right)^2,\qquad
\beta=\frac{L^2F C_0}{\varepsilon_0\varepsilon_r V_T}
=\frac{1}{2}\left(\frac{L}{\lambda_D}\right)^2,\qquad
\lambda_D^2=\frac{\varepsilon_0\varepsilon_r V_T}{2F C_0}.
$$

The ionic flux is $\tilde J_\pm=-\nabla\tilde c_\pm\mp\tilde c_\pm\nabla\tilde\psi$. On the finite-volume grid, the axial flux at face $i+\tfrac12$ is

$$
\tilde J_{z,i+1/2}
=-\frac{\tilde c_{i+1}B(-\Delta\tilde\psi_i)-\tilde c_iB(\Delta\tilde\psi_i)}{\Delta z},
\qquad
B(x)=\frac{x}{e^x-1},
$$

and the radial face flux has the same form with $\Delta\tilde r$ and the appropriate $\alpha_r$ factor. The wall conditions are $\tilde J_{\pm,r}=0$ and $\partial_r\tilde\psi|_\mathrm{wall}=q_w$ (the code sign convention, consistent with negative $q_w$ giving a negative wall potential); the reservoir conditions are $\tilde\psi=V_\mathrm{app}/V_T$ at $z=0$, $\tilde\psi=0$ at $z=1$, and $\tilde c_\pm=1$ at both ends.

Let $z\in[0,1]$ be the normalized axial coordinate, with $z=0$ at the high-potential reservoir and $z=1$ at the grounded reservoir. For any smooth $G_\psi$,

$$
\psi_\theta(r,z)=V_\mathrm{app}(1-z)+z(1-z)G_\psi(r,z)
$$

satisfies $\psi_\theta(r,0)=V_\mathrm{app}$ and $\psi_\theta(r,1)=0$ identically, because the correction vanishes at both ends. No penalty or projection is needed.

For the concentrations, define

$$
s_\theta=2\exp\!\big[\mathrm{clip}(z(1-z)G_s,\,-15,\,15)\big],
\qquad
\rho_\theta=z(1-z)G_\rho,
$$

with $s_\theta$ normalized by $C_0$. Reconstruct

$$
c_{\pm,\theta}=\frac{1}{2}\left(\sqrt{s_\theta^2+\rho_\theta^2}\pm\rho_\theta\right).
$$

Then:

1. **Positivity.** $\sqrt{s^2+\rho^2}\ge|\rho|$, so $c_+\ge0$ and $c_-\ge0$ for any real $s,\rho$. The exponential form keeps $s>0$ by construction.
2. **Charge identity.** $c_+-c_-=\rho$ exactly.
3. **Sum identity.** $c_++c_-=\sqrt{s^2+\rho^2}=s$, so the reconstructed sum is positive and equals $s$ exactly.
4. **Reservoir limit.** As $z\to0$ or $z\to1$ the envelope vanishes, so $s\to2$ and $\rho\to0$, hence $c_{\pm}\to1$ in normalized units (i.e., $c_{\pm}\to C_0$). The ansatz therefore enforces both reservoir concentrations exactly, not merely neutrality; the EDL structure is represented by the interior profiles of $G_s$ and $G_\rho$.

The algebraic charge identity was checked numerically: the maximum absolute error over the test states is 1.110e-16, i.e., double-precision round-off.

**Why this matters.** If $c_+$ and $c_-$ are predicted independently, the error in $\rho=c_+-c_-$ is the difference of two errors, each of order $\epsilon c$, while $|\rho|$ can be much smaller than $c$. The relative error in $\rho$ therefore scales like $\epsilon c/|\rho|$ and diverges in weakly charged states. Predicting $s$ and $\rho$ removes this cancellation by construction. The empirical effect is an 8.63-fold (DeepONet) and 10.29-fold (FNO) reduction in interp-test net-charge error.

---

## S4. Extended accuracy metrics

The full three-seed table is `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` (462 data rows; columns: model, mode, condition, split, metric, mean, sd, median, iqr). The main text reports the means for $\psi$, $c_+$, $\rho$, $I$, and the mean absolute current error. The per-state values underlying Figure 4 are in `results/tables/per_state_test_metrics.csv` (1,728 data rows).

Selected additional values:

| Model | Split | $c_-$ rel. $L_2$ | $\sigma_c$ rel. $L_2$ | Wall-derivative error | Poisson residual |
|---|---|---|---|---|---|
| DeepONet reparam. | interp. | 0.00059 | 0.0004484 | 0.004969 | 0.2122 |
| DeepONet reparam. | OOD | 0.02253 | 0.003569 | 0.01579 | 2.543 |
| FNO reparam. | interp. | 0.00159 | 0.001521 | 0.007247 | 0.8617 |
| FNO reparam. | OOD | 0.01311 | 0.005082 | 0.03705 | 9.594 |

All values are three-seed means from `results/statistics.json`. The exact seed-level values for every split are in the 462-row CSV. The current errors are relatively stable across seeds: the interp-test standard deviation of the DeepONet current error is 0.0004264, and the OOD standard deviation is 0.0001388. The potential errors are more variable, with an interp-test standard deviation of 0.0001613 for the DeepONet.

---

## S5. H3 conditioning analysis

Table S5 reports the net-charge error conditioned on the surface-charge magnitude and on the Debye ratio. The values are the means over three seeds for the interpolation split.

**Table S5a.** By $|q_w|$ quartile.

| Model / parameterization | Overall | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| DeepONet reparam. | 0.1567 | 0.143 | 0.08423 | 0.1197 | 0.2801 |
| DeepONet direct $c$ | 0.7795 | 0.9116 | 0.7594 | 0.7635 | 0.6836 |
| FNO reparam. | 0.252 | 0.5316 | 0.112 | 0.1482 | 0.216 |
| FNO direct $c$ | 1.2 | 1.839 | 0.9576 | 0.988 | 1.016 |

**Table S5b.** By $\lambda_D/R$ quartile.

| Model / parameterization | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| DeepONet reparam. | 0.1486 | 0.08436 | 0.1098 | 0.2841 |
| DeepONet direct $c$ | 1.074 | 0.7397 | 0.6677 | 0.6363 |
| FNO reparam. | 0.4925 | 0.1302 | 0.1889 | 0.1962 |
| FNO direct $c$ | 1.811 | 0.9791 | 1.001 | 1.01 |

The reparameterized models are uniformly better, and the advantage is largest in the weak-charge quartile, where the direct models approach unit relative error.

---

## S6. Out-of-distribution definitions

The OOD split is constructed by sampling outside the training box in one or two coordinates while keeping the others in range.

- **OOD charge/salt:** $q_w$ and/or $C_0$ outside the training box.
- **OOD geometry/voltage:** $R$ and/or $V_\mathrm{app}$ outside the training box.

The split is a controlled extrapolation, not an adversarial stress test. It is intended to answer a practical question: if a user moves one slider beyond the calibrated range, does the current prediction degrade gracefully? For current, it does (OOD relative $L_2$ 0.001363 for the DeepONet and 0.009123 for the FNO). For pointwise potential, it does not: the DeepONet OOD potential error is 0.02024, 22.7-fold larger than its interp-test value.

---

## S7. Inverse-recovery protocol and sensitivity

**States.** Five held-out interpolation states: indices 175, 147, 154, 172, and 161. They span $C_0=27.2$--$140.2$ mol m$^{-3}$, $R=3.362$--$7.276$ nm, and $q_w=-0.04557$ to $-0.00206$.

**Measurement.** Exact SG/FV currents at $V_\mathrm{app}=-50,-25,0,25,50$ mV, multiplied by $1+0.01\,\mathcal{N}(0,1)$ with a fixed seed per state.

**Search.** 400 candidate $(q_w,R)$ pairs in the dataset box ($q_w\in[-0.09815884117319931,0]$, $R\in[2.0265975267620897,8]$), evaluated at the five biases (2000 forward evaluations per state). The best three candidates are refined with Nelder--Mead (bounded by clipping, 400 iterations, objective tolerance $10^{-12}$).

**Identifiability diagnostic.** At the recovered radius, the cost is profiled over 41 values of $q_w$ spanning the prior. The 1%-noise floor is $\Delta\mathrm{cost}=10^{-4}$; the set of $q_w$ values within that floor spans the full prior width (0.09816) for all five states. The maximum relative current change across the prior is 0.0009625.

**Interpretation.** Radius is identifiable because the current magnitude and slope constrain the pore cross-section. Fixed charge is not identifiable because the total conductance of a symmetric electrolyte with nearly equal diffusivities is almost invariant to Donnan polarization. This is a property of the measurement, not of the operator. A charge-sensitive observable would break the degeneracy.

**Results.** See Table 6 in the main text and `results/tables/table6_inverse_parameter_recovery.csv`. Median radius error 0.004574; median $q_w$ error 1.154; median surface-charge-density error 1.146. Grid wall time per state 0.1253 s; equivalent SG/FV time 402.6 s; speed-up 3214.

---

## S8. Deployment and parity

| Check | Value |
|---|---|
| ONNX vs PyTorch, 96 states | 3.576e-7 max abs. difference |
| ONNX vs PyTorch, 48-state stage-5 check | 2.384e-7 max abs. difference |
| JavaScript current vs SG/FV | 5.686e-7 max relative difference |
| Browser bundle size | 5,662,351 bytes |
| DOM harness | 29 passed, 0 failed |
| Live SG/FV service | 196.1 pA, 0.2082 s, 4 iterations, residual 2.422e-10 |
| Gradio DeepONet | 196.1 pA, 0.7568 ms |
| Gradio compare, live solve | 0.2057 s; $\psi$ rel. $L_2$ 0.0002126; $I$ rel. $L_2$ 0.0007364 |

The browser bundle contains both ONNX models (`deeponet_reparam_seed42.onnx`, 188,897 bytes; `fno_reparam_seed42.onnx`, 1,912,172 bytes), the model metadata, and the JavaScript engine. The Gradio app imports the same functions used by the UI, so the headless verification exercises the deployed code path.

---

## S9. Reproducibility commands

Run from the project root with the virtual environment activated.

```bash
# 1. Reference dataset (240 states)
.venv/bin/python scripts/generate_parametric_dataset.py

# 2. Three-seed training and evaluation
.venv/bin/python scripts/train_and_evaluate_suite.py --seeds 42 43 44

# 3. FNO 300-epoch check
.venv/bin/python scripts/fno_extended_check.py

# 4. Figures and layout QA
.venv/bin/python scripts/figures/make_all_figures.py
.venv/bin/python scripts/figures/verify_figure_layout.py

# 5. Inverse recovery demo
.venv/bin/python scripts/run_inverse_recovery_demo.py

# 6. Web bundle and DOM test
.venv/bin/python scripts/build_web_bundle.py
node scripts/test_web_simulator_dom.js

# 7. Live services (optional)
.venv/bin/python scripts/serve_interactive.py --host 127.0.0.1 --port 8765
.venv/bin/python scripts/app_gradio.py --host 127.0.0.1 --port 7860
.venv/bin/python scripts/verify_live_servers.py
```

The exact command-line defaults for the training suite are seeds 42 43 44, 110 FNO epochs, 500 DeepONet epochs, batch 36, learning rate $2\times10^{-3}$, physics weight $10^{-3}$, FNO width 24, FNO modes $8\times12$, and DeepONet inner dimension 64.

---

## S10. Figure, table, and claim crosswalk

| Item | Content | Source |
|---|---|---|
| Figure 1 | Geometry, operating space, hard constraints, backbones | `scripts/figures/fig1_workflow_and_architecture.py` |
| Figure 2 | Benchmark fields, errors, current profile | `scripts/figures/fig2_benchmark_and_field_reconstruction.py` |
| Figure 3 | Charge reparameterization and physics ablation | `scripts/figures/fig3_h3_charge_reparameterization_ablation.py` |
| Figure 4 | Per-state distributions, EDL dependence, parity | `scripts/figures/fig4_h1_h2_interp_vs_ood_regimes.py` |
| Figure 5 | Continuous parameter sweeps versus SG/FV | `scripts/figures/fig5_parametric_physics_sweeps.py` |
| Figure 6 | Latency and amortization | `scripts/figures/fig6_h4_timing_and_amortization_breakeven.py` |
| Table 1 | Dataset and solver validation | `results/tables/table1_dataset_and_solver_validation.csv` |
| Table 2 | Benchmark versus DA-PINN | `results/tables/table2_benchmark_state_vs_dapinn.csv` |
| Table 3 | Three-seed metrics | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Table 4 | Timing and break-even | `results/tables/table4_timing_and_amortization.csv` |
| Table 5 | FNO epoch check | `results/tables/table5_fno_epoch_check.csv` |
| Table 6 | Inverse recovery | `results/tables/table6_inverse_parameter_recovery.csv` |
| Claim map | Claim-to-evidence audit | `manuscript/claim_evidence.csv` |
| Statistics | Canonical JSON values | `results/statistics.json` |
| Inverse demo | Full inverse JSON | `results/inverse_recovery_demo.json` |
| Layout QA | 0-issue report | `results/figure_data/figure_layout_report.json` |
| Vision QA | Advisory model report | `results/figure_data/vision_qa_report.json` |

The deterministic layout verifier reports zero issues across all six figures. The paid vision-QA pass was advisory only; its 15 flags were manually adjudicated as false positives, and the layout verifier was used as the binding gate.
