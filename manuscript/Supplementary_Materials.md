# Supplementary Materials

**Manuscript:** Conductance identifiability limits and physics-constrained neural operators for axisymmetric Poisson--Nernst--Planck transport in charged nanopores

**Author:** Javad Alizargar

This supplement documents the dataset, constraint derivation, training protocol, extended metrics, out-of-distribution definitions, inverse-recovery protocol, deployment checks, and reproducibility commands. All numerical values are produced by the archived scripts and result files listed in Section S9. It contains all twenty display tables (Supplementary Tables S1--S20); the main text contains the six multipanel figures only. The released software and data are permanently archived at Zenodo DOI https://doi.org/10.5281/zenodo.23487404.

---

## S0. Supplementary tables S1--S20, summary index and compendium

The twenty display tables are collected here in strict order S1 through S20, matching their first citation order in the main text. The sections that follow document the methods and give extended discussion; each table remains cited from the relevant section.

**Summary index**

- **S1**: Dataset splits and sampling.
- **S2**: Dataset and solver validation.
- **S3**: Radial-trunk ablation and data-scaling law.
- **S4**: Model and training hyperparameters.
- **S5**: Selected three-seed global relative L2 errors.
- **S6**: Selected additional accuracy metrics and residual diagnostics.
- **S7**: Figure, table, and artifact crosswalk.
- **S8**: Deployment and parity checks.
- **S9**: Multi-protocol inverse identifiability.
- **S10**: Benchmark state against the prior fixed-state DA-PINN.
- **S11**: Four-way constraint-scheme benchmark.
- **S12**: Physics-weight sensitivity sweep.
- **S13**: FNO training-budget check.
- **S14**: Out-of-distribution error decomposition.
- **S15**: Net-charge error by wall-charge magnitude quartile.
- **S16**: Net-charge error by Debye-ratio quartile.
- **S17**: Latency and amortization.
- **S18**: Inverse recovery from five-bias noisy current--voltage curves.
- **S19**: Exact SG/FV versus neural-operator Fisher and Jacobian validation.
- **S20**: Exact SG/FV diffusivity-ratio scan.

**Supplementary Table S1.** Dataset splits and sampling.

| Split | Count | Sampling |
|---|---|---|
| Train | 144 | Latin hypercube in the in-distribution box, plus boundary and corner anchors |
| Interpolation test | 48 | Latin hypercube in the in-distribution box; index 144 is the benchmark state |
| OOD charge/salt | 24 | $q_w$ and/or $C_0$ outside the training box |
| OOD geometry/voltage | 24 | $R$ and/or $V_\mathrm{app}$ outside the training box |
| Total | 240 | |

**Supplementary Table S2.** Dataset and solver validation.

| Quantity | Value |
|---|---|
| States | 240 |
| Train / interpolation / OOD | 144 / 48 / 48 |
| Maximum residual | $9.936\times10^{-10}$ |
| Minimum $c_+$ | 0.9999 |
| Minimum $c_-$ | 0.8034 |
| Continuation steps (all states) | 0 |
| Benchmark current | 196.1 pA |
| SG/FV median wall time | 0.2013 s |
| SG/FV mean wall time | 0.2391 s |

**Supplementary Table S3.** Radial-trunk ablation (500 epochs, seeds 42/43/44) and data-scaling law (250 epochs, seeds 42/43/44, nested training subsets of the 384-state expanded dataset). Relative $L_2$ errors are three-seed means; wall-gradient error is the mean absolute deviation $|\partial_r\psi|_{r=1}-q_w|$.

| Trunk basis | Params | $\rho$ interp. | wall-grad. | $I$ interp. (%) | $\rho$ OOD |
|---|---|---|---|---|---|
| plain MLP (no radial) | 33,987 | 0.0914 $\pm$ 0.0011 | $5.197\times10^{-3}$ | 0.364 | 0.2060 |
| radial $k=1$ | 34,051 | 0.1121 $\pm$ 0.0082 | $5.298\times10^{-3}$ | 0.401 | 0.2081 |
| radial $k=3$ | 34,051 | 0.0982 $\pm$ 0.0112 | $4.854\times10^{-3}$ | 0.405 | 0.1991 |
| radial $k=10$ | 34,051 | 0.1105 $\pm$ 0.0131 | $4.997\times10^{-3}$ | 0.381 | 0.2019 |
| radial $k=30$ | 34,051 | 0.1062 $\pm$ 0.0035 | $4.827\times10^{-3}$ | 0.320 | 0.2027 |
| multi-scale (default) | 34,243 | 0.1155 $\pm$ 0.0151 | $4.951\times10^{-3}$ | 0.274 | 0.2053 |

| $N_\mathrm{train}$ | $\rho$ interp. | $\psi$ interp. | $I$ interp. (%) | wall-grad. | $\rho$ OOD |
|---|---|---|---|---|---|
| 36 | 0.3927 $\pm$ 0.0409 | 0.0054 | 0.790 | $8.322\times10^{-3}$ | 0.3170 |
| 72 | 0.3424 $\pm$ 0.0438 | 0.0048 | 0.634 | $8.153\times10^{-3}$ | 0.3203 |
| 144 | 0.2066 $\pm$ 0.0154 | 0.0023 | 0.417 | $6.488\times10^{-3}$ | 0.2575 |
| 288 | 0.1299 $\pm$ 0.0039 | 0.0020 | 0.387 | $6.206\times10^{-3}$ | 0.2527 |

**Supplementary Table S4.** Model and training hyperparameters.

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

**Supplementary Table S5.** Selected three-seed global relative $L_2$ errors (means over seeds; full per-metric table in `results/tables/table3_multiseed_h1_h2_h3_ablation.csv`).

| Model | Split | $\psi$ | $c_+$ | $\rho$ | $I$ | mean abs. $I$ error (pA) |
|---|---|---|---|---|---|---|
| DeepONet reparam. | interp. | 0.0008914 | 0.0007275 | 0.08248 | 0.001093 | 0.08031 |
| DeepONet reparam. | OOD | 0.02024 | 0.0213 | 0.3678 | 0.001363 | 0.1077 |
| FNO reparam. | interp. | 0.005495 | 0.001641 | 0.0931 | 0.008511 | 0.6746 |
| FNO reparam. | OOD | 0.02079 | 0.01487 | 0.2241 | 0.009123 | 0.5338 |
| DeepONet direct $c$ | interp. | 0.003845 | 0.004229 | 0.7116 | 0.002335 | 0.2108 |
| FNO direct $c$ | interp. | 0.00562 | 0.006777 | 0.9578 | 0.007836 | 0.6489 |
| DeepONet data-only | interp. | 0.000928 | 0.000488 | 0.07719 | 0.001687 | 0.1203 |
| FNO data-only | interp. | 0.007958 | 0.002193 | 0.0841 | 0.01393 | 1.126 |

**Supplementary Table S6.** Selected additional accuracy metrics (three-seed means) and residual diagnostics.

| Model | Split | $c_-$ rel. $L_2$ | $\sigma_c$ rel. $L_2$ | Wall-derivative error | Poisson residual |
|---|---|---|---|---|---|
| DeepONet reparam. | interp. | 0.00059 | 0.0004484 | 0.004969 | 0.2122 |
| DeepONet reparam. | OOD | 0.02253 | 0.003569 | 0.01579 | 2.543 |
| FNO reparam. | interp. | 0.00159 | 0.001521 | 0.007247 | 0.8617 |
| FNO reparam. | OOD | 0.01311 | 0.005082 | 0.03705 | 9.594 |

**Supplementary Table S7.** Figure, table, and artifact crosswalk.

| Item | Content | Source |
|---|---|---|
| Figure 1 | Geometry and boundary conditions; operating space; exact constraints; two backbones | `scripts/figures/fig1_workflow_and_architecture.py` |
| Figure 2 | Benchmark fields, error maps, and axial current profile | `scripts/figures/fig2_benchmark_and_field_reconstruction.py` |
| Figure 3 | Constraint-scheme benchmark, physics sweep, ablations, Fisher plane | `scripts/figures/fig3_h3_charge_reparameterization_ablation.py` |
| Figure 4 | Per-state distributions, EDL dependence, current parity | `scripts/figures/fig4_h1_h2_interp_vs_ood_regimes.py` |
| Figure 5 | Continuous parameter sweeps versus SG/FV | `scripts/figures/fig5_parametric_physics_sweeps.py` |
| Figure 6 | Latency, amortization, and exact-versus-operator Fisher validation | `scripts/figures/fig6_h4_timing_and_amortization_breakeven.py` |
| Supplementary Table S1 | Dataset splits and sampling | `data/processed/pnp_parametric_240.npz` |
| Supplementary Table S2 | Dataset and solver validation | `results/tables/table1_dataset_and_solver_validation.csv` |
| Supplementary Table S3 | Radial-trunk ablation and data-scaling law | `results/tables/table_s14_radial_trunk_and_scaling_ablation.csv` |
| Supplementary Table S4 | Model and training hyperparameters | `results/statistics.json` |
| Supplementary Table S5 | Three-seed accuracy and ablation metrics | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Supplementary Table S6 | Additional accuracy and residual metrics | `results/statistics.json` |
| Supplementary Table S10 | Benchmark versus DA-PINN | `results/tables/table2_benchmark_state_vs_dapinn.csv` |
| Supplementary Table S17 | Timing and break-even | `results/tables/table4_timing_and_amortization.csv` |
| Supplementary Table S13 | FNO epoch check | `results/tables/table5_fno_epoch_check.csv` |
| Supplementary Table S18 | Inverse recovery | `results/tables/table6_inverse_parameter_recovery.csv` |
| Supplementary Table S15 | Net-charge error by charge quartile | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Supplementary Table S16 | Net-charge error by Debye-ratio quartile | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Supplementary Table S8 | Deployment and parity checks | `results/stage5b_live_server_verification.json` |
| Supplementary Table S9 | Multi-protocol inverse identifiability | `results/tables/table_s15_fisher_identifiability_protocols.csv` |
| Supplementary Table S11 | Four-way constraint-scheme benchmark | `results/tables/table_constraint_schemes_4way.csv` |
| Supplementary Table S12 | Physics-weight sensitivity sweep | `results/tables/table_physics_weight_sensitivity.csv` |
| Supplementary Table S14 | Out-of-distribution error decomposition | `results/tables/table_ood_mechanistic_breakdown.csv` |
| Supplementary Table S19 | Exact SG/FV versus operator Fisher/Jacobian validation | `results/tables/table_sgfv_fisher_validation.csv` |
| Supplementary Table S20 | Exact SG/FV diffusivity-ratio scan | `results/tables/table_diffusivity_symmetry_breaking.csv` |
| Claim map | Claim-to-evidence audit (C01--C38, L01--L60) | `manuscript/claim_evidence.csv` |
| Statistics | Canonical JSON values | `results/statistics.json` |
| Inverse demo | Full inverse JSON | `results/inverse_recovery_demo.json` |
| Literature verification | API-backed abstract and support audit | `results/literature_claim_verification.json` |
| Layout QA | 0-issue report | `results/figure_data/figure_layout_report.json` |
| Vision QA | Advisory model report | `results/figure_data/vision_qa_report.json` |

**Supplementary Table S8.** Deployment and parity checks.

| Check | Value |
|---|---|
| ONNX vs PyTorch, 96 states | $3.576\times10^{-7}$ max abs. difference |
| ONNX vs PyTorch, 48-state stage-5 check | $2.384\times10^{-7}$ max abs. difference |
| JavaScript current vs SG/FV | $5.686\times10^{-7}$ max relative difference |
| Browser bundle size | 5,662,351 bytes |
| DOM harness | 29 passed, 0 failed |
| Live SG/FV service | 196.1 pA, 0.2082 s, 4 iterations, residual $2.422\times10^{-10}$ |
| Gradio DeepONet | 196.1 pA, 0.7568 ms |
| Gradio compare, live solve | 0.2057 s; $\psi$ rel. $L_2$ 0.0002126; $I$ rel. $L_2$ 0.0007364 |

**Supplementary Table S9.** Multi-protocol inverse identifiability. For 15 held-out states, 1\%/2\%/5\%-proxy Gaussian noise levels (implemented as 0.2\%, 1\%, 2\% relative noise), three measurement protocols: single-salt I--V (one salt), multi-salt I--V (three salts, 10--100 mol $m^{-3}$), and the dual protocol that adds the near-wall Donnan selectivity $S=(c_+-c_-)/(c_++c_-)|_{r=1}$ to the multi-salt I--V data. Recovery uses a $30\times30$ candidate grid in $(q_w,R)$ followed by Nelder--Mead refinement. Errors are medians over the 15 states.

| Protocol | Noise | median $q_w$ rel. err | median $R$ rel. err | $q_w$ identifiable ($<$10\%) | $R$ recovered ($<$2\%) |
|---|---|---|---|---|---|
| single-salt I--V | 0.002 | 1.0000 | 0.0005 | no | yes |
| single-salt I--V | 0.010 | 1.0000 | 0.0025 | no | yes |
| single-salt I--V | 0.020 | 0.7703 | 0.0065 | no | yes |
| multi-salt I--V | 0.002 | 0.8239 | 0.0003 | no | yes |
| multi-salt I--V | 0.010 | 1.0000 | 0.0011 | no | yes |
| multi-salt I--V | 0.020 | 1.0000 | 0.0025 | no | yes |
| dual (I--V + selectivity) | 0.002 | 0.0005 | 0.0003 | yes | yes |
| dual (I--V + selectivity) | 0.010 | 0.0024 | 0.0008 | yes | yes |
| dual (I--V + selectivity) | 0.020 | 0.0028 | 0.0018 | yes | yes |

**Supplementary Table S10.** Benchmark state (50 mV, 100 mol $m^{-3}$, $q_w=-0.01$, $R=5$ nm) against the prior fixed-state DA-PINN.

| Model | Current (pA) | $\psi$ rel. $L_2$ | $c_+$ rel. $L_2$ | $\rho$ rel. $L_2$ | $I$ rel. $L_2$ |
|---|---|---|---|---|---|
| DeepONet (reparam.) | 196.1 | 0.0001485 | 0.0001521 | 0.09363 | 0.0005587 |
| FNO (reparam.) | 196.2 | 0.003298 | 0.001184 | 0.1481 | 0.00634 |
| DeepONet (direct $c$) | 196.1 | 0.0008521 | 0.0006095 | 0.7147 | 0.0006118 |
| FNO (direct $c$) | 195.6 | 0.001352 | 0.003661 | 0.8599 | 0.004085 |
| Prior DA-PINN | 196.1 | 0.000382 | 0.000106 | 0.145 | 0.00102 |

**Supplementary Table S11.** Four-way constraint-scheme benchmark on the identical PI-DeepONet backbone (250 epochs, seeds 42/43/44). Errors are three-seed means on the interpolation-test split; consistency mismatch is $\overline{|\rho-\rho_\mathrm{latent}|}$.

| Scheme | $\rho$ rel. $L_2$ | OOD $\rho$ | wall-grad. | consistency mismatch | min $c$ |
|---|---|---|---|---|---|
| hyperbolic reparam. (ours) | $0.2066\pm0.0154$ | 0.2575 | $6.49\times10^{-3}$ | 0 (exact) | 0.992 |
| linear orthogonal projection | $0.2008\pm0.0206$ | 0.2598 | $6.55\times10^{-3}$ | $4.2\times10^{-8}$ | 0.992 |
| soft charge penalty ($10\times$) | $0.4114\pm0.1871$ | 0.2835 | $8.19\times10^{-3}$ | 0 (exact) | 0.986 |
| direct concentration | $0.9334\pm0.0082$ | 0.8606 | $2.22\times10^{-2}$ | 0 (exact) | 0.999 |

**Supplementary Table S12.** Physics-weight sensitivity sweep at seed 42 (interpolation-test split).

| $\lambda_\mathrm{phys}$ | $\rho$ rel. $L_2$ | $\psi$ rel. $L_2$ | $I$ rel. $L_2$ | wall-grad. | Poisson resid. | continuity resid. |
|---|---|---|---|---|---|---|
| 0 | 0.2175 | 0.0029 | 0.0047 | $1.15\times10^{-2}$ | 1.142 | 0.566 |
| $10^{-4}$ | 0.2063 | 0.0026 | 0.0049 | $6.61\times10^{-3}$ | 0.717 | 0.143 |
| $10^{-3}$ | 0.2125 | 0.0023 | 0.0040 | $5.55\times10^{-3}$ | 0.472 | 0.092 |
| $10^{-2}$ | 0.2866 | 0.0041 | 0.0058 | $7.90\times10^{-3}$ | 0.188 | 0.028 |
| $10^{-1}$ | 0.3056 | 0.0046 | 0.0063 | $8.13\times10^{-3}$ | 0.064 | 0.014 |

**Supplementary Table S13.** FNO training-budget check (seed 42).

| Split | Metric | 110 epochs (3-seed mean) | 300 epochs (seed 42) |
|---|---|---|---|
| interp. | $\psi$ rel. $L_2$ | 0.005495 | 0.002195 |
| interp. | $I$ rel. $L_2$ | 0.008511 | 0.003349 |
| OOD | $\psi$ rel. $L_2$ | 0.02079 | 0.0132 |
| OOD | $I$ rel. $L_2$ | 0.009123 | 0.00234 |
| benchmark | $\psi$ rel. $L_2$ | 0.003298 | 0.001743 |
| benchmark | $I$ rel. $L_2$ | 0.00634 | 0.003669 |

**Supplementary Table S14.** Out-of-distribution error decomposition (48 OOD states; `deeponet_reparam` and `fno_reparam`, seed 42). Spatial rows partition the $21\times61$ domain into the reservoir transitions ($z\in[0,0.10]\cup[0.90,1.0]$), the charged-wall EDL ($r\ge0.80$, $0.10<z<0.90$) and the pore core ($r<0.80$, $0.10<z<0.90$); their squared-error shares sum to 100\% for each field. Parametric rows use mutually exclusive labels with priority combined $>$ charge $>$ salt $>$ narrow $R$ $>$ high $V$. Relative errors are global area-weighted $L_2$ ratios; $I$ is the axial current.

| Backbone | Decomposition | Category | $n$ | $\psi$ share (\%) | $\rho$ share (\%) | $\psi$ rel. $L_2$ | $\rho$ rel. $L_2$ | $I$ rel. $L_2$ |
|---|---|---|---|---|---|---|---|---|
| deeponet | spatial | reservoir transitions | 48 | 9.9 | 8.4 | 0.0114 | 0.4 | 0.00121 |
| deeponet | spatial | wall EDL | 48 | 38.8 | 39.1 | 0.024 | 0.336 | 0.00121 |
| deeponet | spatial | pore core | 48 | 51.3 | 52.4 | 0.0225 | 0.383 | 0.00121 |
| deeponet | parametric | combined charge + low salt | 25 | 94.2 | 93.6 | 0.0388 | 0.372 | 0.0123 |
| deeponet | parametric | charge only | 9 | 5.5 | 6.2 | 0.00816 | 0.304 | 0.000983 |
| deeponet | parametric | low salt only | 1 | 0.0 | 0.0 | 0.000942 | 0.0591 | 0.00149 |
| deeponet | parametric | narrow radius | 7 | 0.2 | 0.2 | 0.0042 | 0.115 | 0.00433 |
| deeponet | parametric | high voltage | 6 | 0.0 | 0.0 | 0.000622 | 0.0687 | 0.000854 |
| fno | spatial | reservoir transitions | 48 | 14.0 | 15.5 | 0.00887 | 0.271 | 0.00234 |
| fno | spatial | wall EDL | 48 | 38.0 | 38.2 | 0.0155 | 0.166 | 0.00234 |
| fno | spatial | pore core | 48 | 48.0 | 46.3 | 0.0142 | 0.18 | 0.00234 |
| fno | parametric | combined charge + low salt | 25 | 94.4 | 93.0 | 0.0254 | 0.185 | 0.00841 |
| fno | parametric | charge only | 9 | 3.0 | 6.6 | 0.00394 | 0.157 | 0.00193 |
| fno | parametric | low salt only | 1 | 0.0 | 0.0 | 0.000597 | 0.0981 | 0.000595 |
| fno | parametric | narrow radius | 7 | 2.1 | 0.4 | 0.00831 | 0.0886 | 0.0112 |
| fno | parametric | high voltage | 6 | 0.6 | 0.0 | 0.00176 | 0.036 | 0.00195 |

**Supplementary Table S15.** Net-charge error by $|q_w|$ quartile.

| Model / parameterization | Overall | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| DeepONet reparam. | 0.1567 | 0.143 | 0.08423 | 0.1197 | 0.2801 |
| DeepONet direct $c$ | 0.7795 | 0.9116 | 0.7594 | 0.7635 | 0.6836 |
| FNO reparam. | 0.252 | 0.5316 | 0.112 | 0.1482 | 0.216 |
| FNO direct $c$ | 1.2 | 1.839 | 0.9576 | 0.988 | 1.016 |

**Supplementary Table S16.** Net-charge error by $\lambda_D/R$ quartile.

| Model / parameterization | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| DeepONet reparam. | 0.1486 | 0.08436 | 0.1098 | 0.2841 |
| DeepONet direct $c$ | 1.074 | 0.7397 | 0.6677 | 0.6363 |
| FNO reparam. | 0.4925 | 0.1302 | 0.1889 | 0.1962 |
| FNO direct $c$ | 1.811 | 0.9791 | 1.001 | 1.01 |

**Supplementary Table S17.** Latency and amortization.

| Item | Value | Unit |
|---|---|---|
| DeepONet ONNX, batch 1 | 0.07165 | ms |
| FNO ONNX, batch 1 | 1.038 | ms |
| DeepONet PyTorch, batch 1 | 1.342 | ms |
| FNO PyTorch, batch 1 | 8.5 | ms |
| DeepONet ONNX, batch 48 | 0.7095 | ms |
| FNO ONNX, batch 48 | 26.86 | ms |
| Current post-processing, batch 48 | 3.788 | ms |
| DeepONet break-even, dataset mean | 459.2 | states |
| FNO break-even, dataset mean | 704.6 | states |
| DeepONet break-even, median | 545.5 | states |
| FNO break-even, median | 837.5 | states |
| DeepONet break-even, prior Xeon | 249.5 | states |
| FNO break-even, prior Xeon | 382.1 | states |
| DeepONet break-even vs DA-PINN retrain | 0.2195 | states |
| FNO break-even vs DA-PINN retrain | 0.3355 | states |

**Supplementary Table S18.** Inverse recovery from five-bias noisy current--voltage curves (1% noise; 400 candidates, 2000 forward evaluations per state). Values are rounded for display; the exact values are in `results/tables/table6_inverse_parameter_recovery.csv`.

| State | $C_0$ (mol $m^{-3}$) | $R$ true (nm) | $R$ recovered (nm) | $R$ rel. error | $q_w$ true | $q_w$ recovered | $q_w$ rel. error |
|---|---|---|---|---|---|---|---|
| 175 | 27.2 | 6.083 | 6.085 | 0.0003 | -0.00206 | -0.09816 | 46.69 |
| 147 | 93.3 | 6.918 | 6.870 | 0.0069 | -0.03770 | -0.07835 | 1.078 |
| 154 | 51.1 | 3.362 | 3.344 | 0.0054 | -0.03711 | -0.09816 | 1.645 |
| 172 | 103.9 | 7.276 | 7.302 | 0.0036 | -0.04557 | -0.09816 | 1.154 |
| 161 | 140.2 | 3.917 | 3.935 | 0.0046 | -0.02875 | -0.00978 | 0.660 |

**Supplementary Table S19.** Exact SG/FV versus neural-operator Fisher and Jacobian validation. Centered finite differences of the float64 Scharfetter--Gummel solver at relative step $10^{-3}$ are compared with the operator Jacobian over 15 held-out states and three protocols (four non-zero biases; salts 10, 30, and 100 mol $m^{-3}$ for the multi-salt and dual protocols). $\kappa(\mathcal{F})$ is the condition number of $\mathcal{F}=J^\top J$ with $J=\partial(\ln|I|,S)/\partial(\ln R,\ln|q_w|)$.

| Quantity | Exact SG/FV | Neural operator |
|---|---|---|
| median $\partial\ln I/\partial\ln R$ | 2.00003 | 1.99997 |
| median relative difference in $\partial\ln I/\partial\ln R$ | $1.5\times10^{-4}$ | -- |
| median $\vert\partial\ln I/\partial\ln q_w\vert$ | $2.1\times10^{-5}$ | $1.2\times10^{-4}$ |
| median $\kappa(\mathcal{F})$, single-salt | $4.9\times10^{15}$ | $8.2\times10^{8}$ |
| median $\kappa(\mathcal{F})$, multi-salt | $8.7\times10^{9}$ | $4.5\times10^{7}$ |
| median $\kappa(\mathcal{F})$, dual | $1.8\times10^{5}$ | $1.8\times10^{5}$ |

**Supplementary Table S20.** Exact SG/FV diffusivity-ratio scan at the benchmark state ($V_\mathrm{app}=50$ mV, $q_w=-0.01$, $R=5$ nm) and at low salt ($C_0=20$ mol $m^{-3}$). The anion diffusivity is fixed at $D_-=2.03\times10^{-9}$ $m^2$ $s^{-1}$ and the cation diffusivity is varied; $\kappa(\mathcal{F})$ is the condition number of $\mathcal{F}=J^\top J$ with $J=\partial(\ln|I|,S)/\partial(\ln R,\ln|q_w|)$, using four non-zero biases for the single-salt protocol and adding $S$ at 50 mV for the dual protocol. Sensitivities are centered finite differences at relative step $10^{-3}$. Condition-number reduction ratios ($\kappa_\mathrm{single}/\kappa_\mathrm{dual}$) are computed from unrounded float64 values before display rounding.

| State | Electrolyte | $D_+/D_-$ | $\vert\partial\ln I/\partial\ln q_w\vert$ | $\kappa_\mathrm{single}$ | $\kappa_\mathrm{dual}$ | Reduction |
|---|---|---|---|---|---|---|
| $C_0=100$ | ideal symmetric | 1.000 | $7.87\times10^{-7}$ | $1.65\times10^{17}$ | $4.04\times10^{6}$ | $4.1\times10^{10}$ |
| $C_0=100$ | KCl | 0.9655 | $1.14\times10^{-5}$ | $1.33\times10^{17}$ | $4.04\times10^{6}$ | $3.3\times10^{10}$ |
| $C_0=100$ | NaCl | 0.655 | $1.43\times10^{-4}$ | $3.02\times10^{16}$ | $4.04\times10^{6}$ | $7.5\times10^{9}$ |
| $C_0=100$ | strongly asymmetric | 0.200 | $4.60\times10^{-4}$ | $5.91\times10^{15}$ | $4.04\times10^{6}$ | $1.5\times10^{9}$ |
| $C_0=20$ | ideal symmetric | 1.000 | $1.06\times10^{-5}$ | $2.42\times10^{14}$ | $6.31\times10^{5}$ | $3.8\times10^{8}$ |
| $C_0=20$ | KCl | 0.9655 | $4.50\times10^{-5}$ | $1.71\times10^{14}$ | $6.31\times10^{5}$ | $2.7\times10^{8}$ |
| $C_0=20$ | NaCl | 0.655 | $6.50\times10^{-4}$ | $2.29\times10^{13}$ | $6.32\times10^{5}$ | $3.6\times10^{7}$ |
| $C_0=20$ | strongly asymmetric | 0.200 | $2.10\times10^{-3}$ | $3.61\times10^{12}$ | $6.35\times10^{5}$ | $5.7\times10^{6}$ |

---

## S1. Dataset, ranges, and splits

The dataset contains 240 steady axisymmetric states on a $21\times61$ node-centered grid (radial $\times$ axial). The pore length is $L=30$ nm; the temperature is $T=298.15$ K; the relative permittivity is $\varepsilon_r=78.5$; the ionic diffusivities are $D_+=1.96\times10^{-9}$ $m^2$ $s^{-1}$ and $D_-=2.03\times10^{-9}$ $m^2$ $s^{-1}$.

The in-distribution box is $V_\mathrm{app}\in[-50,50]$ mV, $C_0\in[20,200]$ mol $m^{-3}$, $q_w\in[-0.05,0]$, and $R\in[3,8]$ nm. The generated states extend to $C_0=10.23$ mol $m^{-3}$, $R=2.027$ nm, and $q_w=-0.09816$ in the OOD groups. The web simulator exposes wider slider ranges (voltage $\pm75$ mV, $C_0$ 10--200 mol $m^{-3}$, $q_w$ $-0.1$--0, $R$ 2--8 nm) but flags values outside the in-distribution box as OOD.

Acceptance criteria: residual below $10^{-9}$, finite fields, and $c_\pm>0$ everywhere. The realized maximum residual is $9.936\times10^{-10}$, the minimum cation concentration is 0.9999, and the minimum anion concentration is 0.8034. No state required continuation. The benchmark state (index 144) is $V_\mathrm{app}=50$ mV, $C_0=100$ mol $m^{-3}$, $q_w=-0.01$, $R=5$ nm; its reference mean current is 196.1 pA.

Reference solve cost: median 0.2013 s, mean 0.2391 s on the Apple M1 host. Dataset generation wall time: 59.18 s.

---

## S2. Models and training hyperparameters

A separate FNO was trained for 300 epochs (seed 42, 302.2 s) to check whether the 110-epoch budget limited accuracy. The 300-epoch results are reported in Supplementary Table S13. A data-only ablation removes the physics term while keeping the architecture, data, and optimizer fixed.

Loss, Eq. (S1):

$$
\mathcal{L}=\mathcal{L}_\mathrm{data}+10^{-3}\mathcal{L}_\mathrm{phys},
$$

The data term, Eq. (S2), is

$$
\mathcal{L}_\mathrm{data}=\frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}}
\frac{1}{N_s}\sum_{s=1}^{N_s}
\left[\frac{f_\theta(s)-f_\mathrm{ref}(s)}
{\max\{\mathrm{RMS}_s(f),\,0.1\,\mathrm{RMS}_\mathrm{global}(f)\}}\right]^2,
$$

where $\mathcal{F}=\{\psi,s,\rho\}$ and $\mathrm{RMS}_s(f)$ is the root-mean-square of field $f$ in state $s$. The per-state scale is floored at 10% of the global RMS to avoid amplifying near-zero states. The physics term, Eq. (S3), is

$$
\mathcal{L}_\mathrm{phys}=\overline{|R_\mathrm{Poisson}|}
+\tfrac12\left(\overline{|R_{\mathrm{cont},+}|}+\overline{|R_{\mathrm{cont},-}|}\right)
+\overline{|R_{\mathrm{wall},\psi}|}
+\overline{|R_{\mathrm{wall},J_+}|}
+\overline{|R_{\mathrm{wall},J_-}|},
$$

where the overbar is a mean over collocation or boundary points and the residuals are the dimensionless Poisson, continuity, wall-potential, and zero-flux residuals. The physics term therefore penalizes the Poisson residual, both continuity residuals, and the wall boundary conditions. The current is not used as a training target; it is computed after the forward pass with the SG face-flux formula, so the reported current errors are an independent check on field consistency.

**Algorithm 1.** End-to-end training and evaluation of the parameter-conditioned PNP operator.

```
Require: parameter box B = [-50,50] mV x [20,200] M x [-0.05,0] x [3,8] nm
Require: grid (nr,nz) = (21,61); seeds {42,43,44}; epochs E_DN = 500, E_FNO = 110
 1: generate 240 states with the SG/FV solver; accept if residual < 1e-9,
    fields finite, and c_pm > 0; split 144/48/48 (train/interp/OOD)
 2: for each backbone in {DeepONet, FNO} and each seed do
 3:   normalize mu with the training-split mean and standard deviation
 4:   build the exact-reservoir ansatz: psi = V(1-z) + z(1-z) G_psi;
       s = 2 exp(clip(z(1-z) G_s)); rho = z(1-z) G_rho
 5:   reconstruct c_pm = (sqrt(s^2+rho^2) +/- rho)/2  (positivity + exact charge)
 6:   for epoch = 1..E do
 7:     sample a minibatch of 36 training states
 8:     evaluate L_data on (psi,s,rho) with per-state scales floored at 0.1 RMS
 9:     evaluate L_phys with the SG face fluxes on the same grid
10:     back-propagate L = L_data + 1e-3 L_phys with AdamW and cosine decay
11:   end for
12:   select the epoch with the lowest validation loss; export to ONNX
13:   report global relative L2 errors and the SG-post-processed current
14: end for
Ensure: released graph web_simulator/models/deeponet_reparam_seed42.onnx
```

**Runtime and dependency pinning.** Reference results were produced on an Apple M1 (arm64) with Python 3.14.7, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1, ONNX 1.23.2, ONNX Runtime 1.31.0, Matplotlib 3.11.2, and Gradio 6.30.0, using one CPU core. Dataset generation takes 59.18 s; a 500-epoch DeepONet takes 50.59 s and a 110-epoch FNO 108.5 s; a 250-epoch scaling model takes about 25 s and a 500-epoch trunk-ablation model about 50 s. The pinned environment used for the archived runs is recorded in `requirements.txt` in the public repository; the Stage-11 experiments add about 8--10 min of single-core time.

---

## S3. Derivation of the exact-reservoir and charge-balanced ansatz

### S3.1 Dimensionless PNP system

Lengths are scaled by the pore length $L$, the radial coordinate by the pore radius $R$, potentials by the thermal voltage $V_T=k_BT/e$, and concentrations by the bulk concentration $C_0$. With $\tilde r=r/R\in[0,1]$ and $z\in[0,1]$, the steady dimensionless Poisson equation, Eq. (S4), is

$$
\alpha_r\frac{1}{\tilde r}\frac{\partial}{\partial\tilde r}
\left(\tilde r\frac{\partial\tilde\psi}{\partial\tilde r}\right)
+\frac{\partial^2\tilde\psi}{\partial z^2}
=-\beta\,(\tilde c_+-\tilde c_-),
$$

and the two steady continuity equations, Eq. (S5), are

$$
\alpha_r\frac{1}{\tilde r}\frac{\partial}{\partial\tilde r}
\left(\tilde r\tilde J_{\pm,\tilde r}\right)
+\frac{\partial\tilde J_{\pm,z}}{\partial z}=0,
$$

with the dimensionless groups, Eq. (S6),

$$
\alpha_r=\left(\frac{L}{R}\right)^2,\qquad
\beta=\frac{L^2F C_0}{\varepsilon_0\varepsilon_r V_T}
=\frac{1}{2}\left(\frac{L}{\lambda_D}\right)^2,\qquad
\lambda_D^2=\frac{\varepsilon_0\varepsilon_r V_T}{2F C_0}.
$$

The ionic flux is $\tilde J_\pm=-\nabla\tilde c_\pm\mp\tilde c_\pm\nabla\tilde\psi$. On the finite-volume grid, the axial flux at face $i+\tfrac12$, Eq. (S7), is

$$
\tilde J_{z,i+1/2}
=-\frac{\tilde c_{i+1}B(-\Delta\tilde\psi_i)-\tilde c_iB(\Delta\tilde\psi_i)}{\Delta z},
\qquad
B(x)=\frac{x}{e^x-1},
$$

and the radial face flux has the same form with $\Delta\tilde r$ and the appropriate $\alpha_r$ factor. The wall conditions are $\tilde J_{\pm,r}=0$ and $\partial_r\tilde\psi|_\mathrm{wall}=q_w$ (the code sign convention, consistent with negative $q_w$ giving a negative wall potential); the reservoir conditions are $\tilde\psi=V_\mathrm{app}/V_T$ at $z=0$, $\tilde\psi=0$ at $z=1$, and $\tilde c_\pm=1$ at both ends.

Let $z\in[0,1]$ be the normalized axial coordinate, with $z=0$ at the high-potential reservoir and $z=1$ at the grounded reservoir. For any smooth $G_\psi$, the potential, Eq. (S8),

$$
\psi_\theta(r,z)=V_\mathrm{app}(1-z)+z(1-z)G_\psi(r,z)
$$

satisfies $\psi_\theta(r,0)=V_\mathrm{app}$ and $\psi_\theta(r,1)=0$ identically, because the correction vanishes at both ends. No penalty or projection is needed.

For the concentrations, Eq. (S9), define

$$
s_\theta=2\exp\!\big[\mathrm{clip}(z(1-z)G_s,\,-15,\,15)\big],
\qquad
\rho_\theta=z(1-z)G_\rho,
$$

with $s_\theta$ the positive geometric salt scale normalized by $C_0$. Reconstruct the concentrations, Eq. (S10),

$$
c_{\pm,\theta}=\frac{1}{2}\left(\sqrt{s_\theta^2+\rho_\theta^2}\pm\rho_\theta\right).
$$

Here $s_\theta=2\sqrt{c_{+,\theta}c_{-,\theta}}>0$ is twice the geometric mean of the two ion concentrations, and

$$
\Sigma c_\theta=c_{+,\theta}+c_{-,\theta}=\sqrt{s_\theta^2+\rho_\theta^2}>0
$$

is their total. The capital $S$ is reserved for the near-wall Donnan selectivity observable. Then:

1. **Positivity.** $\sqrt{s^2+\rho^2}\ge|\rho|$, so $c_+\ge0$ and $c_-\ge0$ for any real $s,\rho$. The exponential form keeps $s>0$ by construction.
2. **Charge identity.** $c_+-c_-=\rho$ exactly.
3. **Total-ion and geometric identities.** $c_++c_-=\sqrt{s^2+\rho^2}=\Sigma c>0$ and $c_+c_-=s^2/4$, so the reconstructed total is positive and $s=2\sqrt{c_+c_-}$ is the positive geometric salt scale.
4. **Reservoir limit.** As $z\to0$ or $z\to1$ the envelope vanishes, so $s\to2$ and $\rho\to0$, hence $c_{\pm}\to1$ in normalized units (i.e., $c_{\pm}\to C_0$). The ansatz therefore enforces both reservoir concentrations exactly, not merely neutrality; the EDL structure is represented by the interior profiles of $G_s$ and $G_\rho$.

The algebraic charge identity was checked numerically: the maximum absolute error over the test states is $1.110\times10^{-16}$, i.e., double-precision round-off.

**Why this matters.** If $c_+$ and $c_-$ are predicted independently, the error in $\rho=c_+-c_-$ is the difference of two errors, each of order $\epsilon c$, while $|\rho|$ can be much smaller than $c$. The relative error in $\rho$ therefore scales like $\epsilon c/|\rho|$ and diverges in weakly charged states. Predicting $s$ and $\rho$ removes this cancellation by construction. The empirical effect is an 8.63-fold (DeepONet) and 10.29-fold (FNO) reduction in interp-test net-charge error.

---

## S4. Extended accuracy metrics

The full three-seed table is `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` (462 data rows; columns: model, mode, condition, split, metric, mean, sd, median, iqr). The main text reports the means for $\psi$, $c_+$, $\rho$, $I$, and the mean absolute current error. The per-state values underlying Figure 4 are in `results/tables/per_state_test_metrics.csv` (1,728 data rows).

Selected additional values are collected in Supplementary Table S6. All values are three-seed means from `results/statistics.json`. The exact seed-level values for every split are in the 462-row CSV. The current errors are relatively stable across seeds: the interp-test standard deviation of the DeepONet current error is 0.0004264, and the OOD standard deviation is 0.0001388. The potential errors are more variable, with an interp-test standard deviation of 0.0001613 for the DeepONet.

---

## S5. H3 conditioning analysis

Supplementary Tables S15, S16 report the net-charge error conditioned on the surface-charge magnitude and on the Debye ratio. The values are the means over three seeds for the interpolation split.

The reparameterized models are uniformly better, and the advantage is largest in the weak-charge quartile, where the direct models approach unit relative error.

---

## S6. Out-of-distribution definitions

The OOD split is constructed by sampling outside the training box in one or two coordinates while keeping the others in range.

- **OOD charge/salt:** $q_w$ and/or $C_0$ outside the training box.
- **OOD geometry/voltage:** $R$ and/or $V_\mathrm{app}$ outside the training box.

The split is a controlled extrapolation, not an adversarial stress test. It is intended to answer a practical question: if a user moves one slider beyond the calibrated range, does the current prediction degrade gracefully? For current, it does (OOD relative $L_2$ 0.001363 for the DeepONet and 0.009123 for the FNO). For pointwise potential, it does not: the DeepONet OOD potential error is 0.02024, 22.7-fold larger than its interp-test value.

---

## S7. Inverse-recovery protocol and sensitivity

**States.** Five held-out interpolation states: indices 175, 147, 154, 172, and 161. They span $C_0=27.2$--$140.2$ mol $m^{-3}$, $R=3.362$--$7.276$ nm, and $q_w=-0.04557$ to $-0.00206$.

**Measurement.** Exact SG/FV currents at $V_\mathrm{app}=-50,-25,0,25,50$ mV, multiplied by $1+0.01\,\mathcal{N}(0,1)$ with a fixed seed per state.

**Search.** 400 candidate $(q_w,R)$ pairs in the dataset box ($q_w\in[-0.09815884117319931,0]$, $R\in[2.0265975267620897,8]$), evaluated at the five biases (2000 forward evaluations per state). The best three candidates are refined with Nelder--Mead (bounded by clipping, 400 iterations, objective tolerance $10^{-12}$).

**Identifiability diagnostic.** At the recovered radius, the cost is profiled over 41 values of $q_w$ spanning the prior. The 1%-noise floor is $\Delta\mathrm{cost}=10^{-4}$; the set of $q_w$ values within that floor spans the full prior width (0.09816) for all five states. The maximum relative current change across the prior is 0.0009625.

**Interpretation.** Radius is identifiable because the current magnitude and slope constrain the pore cross-section. Fixed charge is not identifiable because the total conductance of a symmetric electrolyte with nearly equal diffusivities is almost invariant to Donnan polarization. This is a property of the measurement, not of the operator. A charge-sensitive observable would break the degeneracy.

**Results.** See Supplementary Table S18 and `results/tables/table6_inverse_parameter_recovery.csv`. Median radius error 0.004574; median $q_w$ error 1.154; median surface-charge-density error 1.146. Grid wall time per state 0.1253 s; equivalent SG/FV time 402.6 s; speed-up 3214.

---

## S7b. Stage-10 scaling, trunk, and multi-protocol identifiability

This section documents the three Stage-10 studies. All values are produced by `scripts/run_stage10_upgrades.py` and archived in `results/stage10_upgrades.json`, `results/tables/table_s14_radial_trunk_and_scaling_ablation.csv`, and `results/tables/table_s15_fisher_identifiability_protocols.csv`. The corresponding display tables are S3 and S9 in Section S0.

Fitted power laws $\epsilon=a\,N_\mathrm{train}^{-\alpha}$: $\alpha_\rho=0.552$ and $\alpha_\psi=0.535$ in-distribution, but only $\alpha_\rho=0.130$ and $\alpha_\psi=0.118$ out of distribution and $\alpha_I=0.369$ in-distribution. The interpolation error therefore follows approximately $N^{-1/2}$, whereas OOD error is nearly data-insensitive.

The Fisher-information plane (Figure 3F) spans $\lambda_D/R\in[0.05,0.55]$ at $R=5$ nm and $|q_w|\in[0.005,0.05]$. The dimensionless current sensitivity is $\partial\ln I/\partial\ln R=2.029$ (range 2.028--2.032, i.e.\ the current scales as the pore area) while the median $|\partial\ln I/\partial\ln q_w|=1.9\times10^{-4}$ (maximum $1.6\times10^{-3}$), giving a condition number $\kappa(\mathcal{F})$ between $9.9\times10^{7}$ and $1.7\times10^{12}$ (median $7.5\times10^{8}$); $99.7\%$ of the plane exceeds $10^{8}$. The near-null Fisher direction is the charge axis for every operating point, which is why conductance-only inversion cannot recover $q_w$. The single-salt conductance sensitivity to $q_w$ does grow at low salinity (relative spread of $|I|$ over $q_w\in[-0.05,0]$: $1.3\times10^{-3}$ at $C_0=10$ mol $m^{-3}$ versus $1.5\times10^{-4}$ at $C_0=100$ mol $m^{-3}$), but remains below the measurement noise, so multi-salt I--V alone is still insufficient; the independent near-wall selectivity observable is what breaks the degeneracy.

The expanded dataset used for the scaling study appends 144 new in-distribution Latin-hypercube states (seed 20261020) to the original 240, giving 384 states (`data/processed/pnp_parametric_384.npz`, 6.36 MB) with a maximum residual of $9.86\times10^{-10}$ and strictly positive concentrations.

---

## S7c. Reviewer-upgrade experiments (Stages 11--12)

This section documents the three Stage-11 studies and the Stage-12 OOD decomposition. All values are produced by `scripts/run_stage11_reviewer_upgrades.py` and archived in `results/stage11_upgrades.json` and the matching `results/tables/` CSV files. The corresponding display tables are S11, S12, S14, and S19 in Section S0.

---

## S7d. Diffusivity asymmetry and the conductance flat direction

Supplementary Table S20 tests the algebraic argument in Section 3.9: the local ohmic coefficient is $D_+c_++D_-c_-=\bar D\sqrt{s^2+\rho^2}+\tfrac12(D_+-D_-)\rho$, so equal diffusivities leave only the second-order $\rho^2$ term, whereas a finite $D_+-D_-$ adds a linear charge contribution. Even the strongly asymmetric case keeps $\kappa_\mathrm{single}>3\times10^{12}$, and the near-wall selectivity observable lowers the conditioning by six to ten orders of magnitude at every ratio.

---

## S8. Deployment and parity

The deployment and parity checks are collected in Supplementary Table S8. The browser bundle contains both ONNX models (`deeponet_reparam_seed42.onnx`, 188,897 bytes; `fno_reparam_seed42.onnx`, 1,912,172 bytes), the model metadata, and the JavaScript engine; the full bundle is 5,662,351 bytes. The static page exposes sliders for the four parameters, a split view that overlays the operator and an in-browser SG/FV solve, and an OOD flag that warns when a queried point leaves the calibrated box. The local SG/FV service returns the benchmark current 196.1 pA in 0.2082 s with four iterations and a residual of $2.42\times10^{-10}$; the Gradio application returns a DeepONet mean current of 196.1 pA in 0.757 ms and, when asked to compare against a live solve, a potential relative $L_2$ difference of 0.0002126 in 0.2057 s of solve time. The Gradio app imports the same functions used by the UI, so the headless verification exercises the deployed code path.

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

# 6. Stage-10 studies (trunk ablation, 384-state scaling law, Fisher identifiability)
.venv/bin/python scripts/generate_parametric_dataset.py   # original 240 states
.venv/bin/python scripts/run_stage10_upgrades.py --epochs-ablation 500 --epochs-scaling 250

# 7. Stage-11 reviewer upgrades (4-way constraints, physics sweep, SG/FV Fisher)
.venv/bin/python scripts/run_stage11_reviewer_upgrades.py

# 8. Stage-12 diagnostics (OOD decomposition, selectivity validation, positivity)
.venv/bin/python scripts/run_stage12_diagnostics.py

# 9. Stage-13 diffusivity-asymmetry scan
.venv/bin/python scripts/run_diffusivity_symmetry_check.py

# 8. Web bundle and DOM test
.venv/bin/python scripts/build_web_bundle.py
node scripts/test_web_simulator_dom.js

# 9. Live services (optional)
.venv/bin/python scripts/serve_interactive.py --host 127.0.0.1 --port 8765
.venv/bin/python scripts/app_gradio.py --host 127.0.0.1 --port 7860
.venv/bin/python scripts/verify_live_servers.py
```

The exact command-line defaults for the training suite are seeds 42 43 44, 110 FNO epochs, 500 DeepONet epochs, batch 36, learning rate $2\times10^{-3}$, physics weight $10^{-3}$, FNO width 24, FNO modes $8\times12$, and DeepONet inner dimension 64.

---

## S10. Figure, table, and claim crosswalk

The deterministic layout verifier reports zero issues across all six figures. The paid vision-QA pass was advisory only; its 15 flags were manually adjudicated as false positives, and the layout verifier was used as the binding gate.
