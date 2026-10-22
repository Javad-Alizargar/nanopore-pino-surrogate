# Supplementary Materials

**Manuscript:** Charge-balanced neural operators, conductance identifiability limits, and warm-started finite-volume solvers for nanopore ion transport

**Author:** Javad Alizargar

This supplement documents the dataset, constraint derivation, training protocol, extended metrics, out-of-distribution definitions, inverse-recovery protocol, deployment checks, and reproducibility commands. All numerical values are produced by the archived scripts and result files listed in Section S9. It contains all twenty-two display tables (Supplementary Tables S1--S22); the main text contains the six multipanel figures only. The released software and data are permanently archived at Zenodo DOI https://doi.org/10.5281/zenodo.23487404.

---

## S0. Supplementary tables S1--S22, summary index and compendium

The twenty-two display tables are collected here in strict order S1 through S22, matching their first citation order in the main text. The sections that follow document the methods and give extended discussion; each table remains cited from the relevant section.

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
- **S11**: Five-way constraint-scheme benchmark with matched direct-$c$ control.
- **S12**: Physics-weight sensitivity sweep.
- **S13**: FNO training-budget check.
- **S14**: Out-of-distribution error decomposition.
- **S15**: Net-charge error by wall-charge magnitude quartile.
- **S16**: Net-charge error by Debye-ratio quartile.
- **S17**: Latency and amortization.
- **S18**: Candidate-grid nesting and SG/FV-in-the-loop inverse check.
- **S19**: Exact SG/FV versus neural-operator Fisher and Jacobian validation.
- **S20**: Exact SG/FV diffusivity-ratio scan.
- **S21**: Interpreted baselines and strong-charge benchmark.
- **S22**: Operator-warm-started SG/FV solver and reversal-potential inverse benchmark.

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

**Supplementary Table S4.** Model and training hyperparameters. The parameter count in Table S3 refers to the matched-capacity Stage-10 ablation backbone (250-epoch budget); the production three-seed model in this table is the 500-epoch $p=64$ configuration, so its 36,880 count is not directly comparable with the 34,243 ablation count.

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
| Supplementary Table S18 | Candidate-grid nesting and SG/FV-in-the-loop inverse check | `results/stage18_final_evidence.json` |
| Supplementary Table S15 | Net-charge error by charge quartile | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Supplementary Table S16 | Net-charge error by Debye-ratio quartile | `results/tables/table3_multiseed_h1_h2_h3_ablation.csv` |
| Supplementary Table S8 | Deployment and parity checks | `results/stage5b_live_server_verification.json` |
| Supplementary Table S9 | Multi-protocol inverse identifiability | `results/tables/table_s15_fisher_identifiability_protocols.csv` |
| Supplementary Table S11 | Five-way constraint-scheme benchmark, including matched direct-$c$ control | `results/tables/table_constraint_schemes_4way.csv` |
| Supplementary Table S12 | Physics-weight sensitivity sweep | `results/tables/table_physics_weight_sensitivity.csv` |
| Supplementary Table S14 | Out-of-distribution error decomposition | `results/tables/table_ood_mechanistic_breakdown.csv` |
| Supplementary Table S19 | Exact SG/FV versus operator Fisher/Jacobian validation | `results/tables/table_sgfv_operator_fisher_8x8.csv` |
| Supplementary Table S20 | Exact SG/FV diffusivity-ratio scan | `results/tables/table_diffusivity_symmetry_breaking.csv` |
| Supplementary Table S21 | Interpreted baselines and strong-charge benchmark | `results/tables/table_analytic_pod_baselines.csv`; `results/tables/table_strong_charge_benchmark.csv` |
| Supplementary Table S22 | Operator-warm-started SG/FV solver and reversal-potential inverse benchmark | `results/tables/table_s22_hybrid_controller_and_reversal.csv`; `results/stage19_hybrid_controller.json` |
| Claim map | Claim-to-evidence audit (C01--C44, L01--L61) | `manuscript/claim_evidence.csv` |
| Statistics | Canonical JSON values | `results/statistics.json` |
| Inverse study | Stage-18 inverse, nesting, and SG/FV-in-the-loop results | `results/stage18_final_evidence.json` |
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

**Supplementary Table S9.** Multi-protocol inverse identifiability with exact SG/FV-generated observations. For 15 held-out interpolation states, three protocols, and three relative noise levels (0.2%, 1%, 2%), we use 20 independent realizations per state--protocol--noise cell and a $61\times61$ candidate grid in $(q_w,R)$ followed by quadratic refinement. Medians and interquartile ranges (IQR) are over the 300 realizations per cell. Single-salt uses one bulk concentration; multi-salt uses 10, 30, and 100 mol $m^{-3}$; dual adds the near-wall selectivity $S=(c_+-c_-)/(c_++c_-)|_{r=1}$ to the multi-salt data. Coverage is the fraction of 95% intervals containing the true value.

| Protocol | Noise | median $q_w$ rel. err | $q_w$ IQR | median $R$ rel. err | $R$ IQR | $q_w$ coverage | $R$ coverage |
|---|---|---|---|---|---|---|---|
| single-salt | 0.002 | 1.000 | 4.97 | 0.00262 | 0.00720 | 1.000 | 0.400 |
| single-salt | 0.010 | 1.000 | 1.95 | 0.00357 | 0.00662 | 1.000 | 0.820 |
| single-salt | 0.020 | 1.000 | 1.95 | 0.00440 | 0.00652 | 1.000 | 0.960 |
| multi-salt | 0.002 | 1.087 | 5.78 | 0.00262 | 0.00720 | 0.933 | 0.133 |
| multi-salt | 0.010 | 1.056 | 4.96 | 0.00262 | 0.00670 | 1.000 | 0.610 |
| multi-salt | 0.020 | 1.000 | 1.93 | 0.00357 | 0.00654 | 1.000 | 0.660 |
| dual | 0.002 | 0.0490 | 0.106 | 0.000225 | 0.000284 | 0.690 | 0.890 |
| dual | 0.010 | 0.0483 | 0.116 | 0.000999 | 0.00125 | 0.707 | 0.963 |
| dual | 0.020 | 0.0495 | 0.0897 | 0.00221 | 0.00215 | 0.700 | 0.947 |

**Supplementary Table S10.** Benchmark state (50 mV, 100 mol $m^{-3}$, $q_w=-0.01$, $R=5$ nm) against the prior fixed-state DA-PINN.

| Model | Current (pA) | $\psi$ rel. $L_2$ | $c_+$ rel. $L_2$ | $\rho$ rel. $L_2$ | $I$ rel. $L_2$ |
|---|---|---|---|---|---|
| DeepONet (reparam.) | 196.1 | 0.0001485 | 0.0001521 | 0.09363 | 0.0005587 |
| FNO (reparam.) | 196.2 | 0.003298 | 0.001184 | 0.1481 | 0.00634 |
| DeepONet (direct $c$) | 196.1 | 0.0008521 | 0.0006095 | 0.7147 | 0.0006118 |
| FNO (direct $c$) | 195.6 | 0.001352 | 0.003661 | 0.8599 | 0.004085 |
| Prior DA-PINN | 196.1 | 0.000382 | 0.000106 | 0.145 | 0.00102 |

**Supplementary Table S11.** Five-way constraint-scheme benchmark on the identical PI-DeepONet backbone (250 epochs, seeds 42/43/44). Errors are three-seed means on the interpolation-test split; consistency mismatch is $\overline{|\rho-\rho_\mathrm{latent}|}$. The matched direct-$c$ control uses the same field-normalized data loss as the other schemes, isolating the output parameterization from the optimization pathology of the naive direct loss.

| Scheme | $\rho$ rel. $L_2$ | OOD $\rho$ | $I$ rel. $L_2$ | wall-grad. | consistency mismatch | min $c$ |
|---|---|---|---|---|---|---|
| hyperbolic reparam. (ours) | $0.2066\pm0.0154$ | 0.2575 | 0.00417 | $6.49\times10^{-3}$ | 0 (exact) | 0.992 |
| linear orthogonal projection | $0.2008\pm0.0206$ | 0.2598 | 0.00361 | $6.55\times10^{-3}$ | $4.2\times10^{-8}$ | 0.992 |
| soft charge penalty ($10\times$) | $0.4114\pm0.1871$ | 0.2835 | 0.00690 | $8.19\times10^{-3}$ | 0 (exact) | 0.986 |
| direct concentration, naive | $0.9334\pm0.0082$ | 0.8606 | 0.00991 | $2.22\times10^{-2}$ | 0 (exact) | 0.999 |
| direct concentration, matched | $0.2377\pm0.0571$ | 0.2739 | 0.00514 | $7.34\times10^{-3}$ | 0 (exact) | 0.988 |

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

**Supplementary Table S14.** Out-of-distribution error decomposition (48 OOD states; `deeponet_reparam` and `fno_reparam`, seed 42). Spatial rows partition the $21\times61$ domain into the reservoir transitions ($z\in[0,0.10]\cup[0.90,1.0]$), the charged-wall EDL ($r\ge0.80$, $0.10<z<0.90$) and the pore core ($r<0.80$, $0.10<z<0.90$); their squared-error shares sum to 100% for each field. Parametric rows use mutually exclusive labels with priority combined $>$ charge $>$ salt $>$ narrow $R$ $>$ high $V$. Relative errors are global area-weighted $L_2$ ratios; $I$ is the axial current. For the three spatial rows, I rel. L2 reports the global axial current error across all 48 OOD states because axial current is cross-sectionally integrated rather than partitioned by spatial zone.

| Backbone | Decomposition | Category | $n$ | $\psi$ share (%) | $\rho$ share (%) | $\psi$ rel. $L_2$ | $\rho$ rel. $L_2$ | $I$ rel. $L_2$ |
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

**Supplementary Table S18.** Candidate-grid nesting and SG/FV-in-the-loop inverse check. (A) At 1% noise, median errors on the $61\times61$ and $121\times121$ grids for the same 15 states and 20 realizations; the dual protocol changes by less than 1%, whereas the radius-only protocols have a shallow, grid-sensitive radius optimum. (B) Median errors with the exact SG/FV solver used for every candidate on five states at 1% noise, initialized from the operator estimate; this check validates the protocol ranking but is not a blind estimator.

| Protocol | $q_w$ 61 | $q_w$ 121 | $\Delta q_w$ | $R$ 61 | $R$ 121 | $\Delta R$ |
|---|---|---|---|---|---|---|
| single-salt | 1.000 | 1.000 | 0.0% | 0.00357 | 0.00252 | 29.4% |
| multi-salt | 1.056 | 1.000 | 5.3% | 0.00262 | 0.00208 | 20.6% |
| dual | 0.0483 | 0.0485 | 0.40% | 0.000999 | 0.00101 | 0.91% |

| Protocol | median $q_w$ rel. err | median $R$ rel. err |
|---|---|---|
| single-salt | 1.336 | 0.00440 |
| multi-salt | 1.185 | 0.00272 |
| dual | 0.0871 | 0.00455 |

**Supplementary Table S19.** Exact SG/FV versus neural-operator sensitivities on the $8\times8$ negative-$q_w$ Fisher plane ($R=5$ nm, four non-zero biases). Centered finite differences use relative step $10^{-3}$; the Jacobian changes by at most $3.3\times10^{-5}$ across steps $10^{-3}$, $3\times10^{-3}$, and $10^{-2}$. $\kappa(\mathcal{F})$ is the condition number of $\mathcal{F}=J^\top J/\sigma^2$ with $J=\partial\ln|I|/\partial\ln(R,|q_w|)$ at 1% noise. The exact single-salt matrix is numerically rank-deficient, so its condition number is reported only as a scale; the radius sensitivity is the quantity that is resolved.

| Quantity | Exact SG/FV | Neural operator |
|---|---|---|
| median $\partial\ln I/\partial\ln R$ | 2.00003 | 2.00022 |
| median relative difference in $\partial\ln I/\partial\ln R$ | $1.1\times10^{-4}$ | -- |
| median $\partial\ln I/\partial\ln q_w$ | $-1.64\times10^{-5}$ | $-2.49\times10^{-5}$ |
| median $\kappa(\mathcal{F})$, single-salt | $1.8\times10^{16}$ (rank-deficient) | $9.8\times10^{8}$ |

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

**Supplementary Table S21.** Interpreted baselines and strong-charge benchmark. (A) Relative $L_2$ errors on the 48 interpolation-test and 48 out-of-distribution states. The Ohmic bulk model uses bulk conductivity and the geometric cross-section; the linearized radial Poisson--Boltzmann (PB) model uses a Debye-layer profile calibrated only by $C_0$ and $R$, with no per-state fitting; POD--RBF uses 30 modes per field; DeepONet and FNO are the three-seed canonical models. (B) Strong-charge test: 120 generated states, 96/24 train/test split, $C_0=10$--13 mol $m^{-3}$, $q_w=-0.50$ to $-0.38$, $R=2.5$--5 nm, 600 epochs. The Ohmic row predicts no wall layer, so its net-charge error is unity by construction.

| Split | Method | $\psi$ rel. $L_2$ | $\rho$ rel. $L_2$ | $S$ rel. $L_2$ | $I$ rel. $L_2$ |
|---|---|---|---|---|---|
| interp. | Ohmic bulk | 0.0103 | 1.000 | 1.000 | $4.87\times10^{-5}$ |
| interp. | linearized radial PB | 0.000969 | 0.0939 | 0.0796 | $8.25\times10^{-6}$ |
| interp. | POD--RBF (30 modes) | 0.00102 | 0.433 | 0.0882 | $4.84\times10^{-5}$ |
| interp. | PI-DeepONet | 0.000891 | 0.0825 | 0.0177 | 0.00109 |
| interp. | FNO-2D | 0.00550 | 0.0931 | 0.0264 | 0.00851 |
| OOD | Ohmic bulk | 0.153 | 1.000 | 1.000 | $8.33\times10^{-4}$ |
| OOD | linearized radial PB | 0.0203 | 0.113 | 0.102 | $5.29\times10^{-4}$ |
| OOD | POD--RBF (30 modes) | 0.0463 | 0.333 | 0.309 | 0.00476 |
| OOD | PI-DeepONet | 0.0202 | 0.368 | 0.316 | 0.00136 |
| OOD | FNO-2D | 0.0208 | 0.224 | 0.179 | 0.00912 |

| Method | Seed | $I$ median rel. $L_2$ | $I$ mean rel. $L_2$ | $\psi$ rel. $L_2$ | $\rho$ rel. $L_2$ | $S$ rel. $L_2$ | $S$ abs. error |
|---|---|---|---|---|---|---|---|
| Ohmic bulk | -- | 0.0591 | 0.0797 | 0.469 | 1.000 | 1.000 | -- |
| strong-trained DeepONet | 42 | 0.00259 | 0.00321 | 0.0248 | 0.0453 | 0.0535 | 0.0206 |
| strong-trained DeepONet | 43 | 0.00120 | 0.00257 | 0.0263 | 0.0464 | 0.0552 | 0.0212 |
| strong-trained DeepONet | 44 | 0.00302 | 0.00377 | 0.0276 | 0.0492 | 0.0585 | 0.0223 |



**Supplementary Table S22.** Operator-warm-started SG/FV solver and blind unequal-reservoir reversal-potential benchmark. (A) Median current and reversal-potential errors with median wall time for eight methods on 120 states (40 weak, 40 transitional, 40 strong). The Pino-Controller warm-starts the conservative SG/FV solve from the operator and is reference-free; full SG/FV is the reference. (B) Blind Stage-20 inversion with shared noisy observations, 15 states, and 20 realizations per protocol and noise level; B is the residual-controlled SG/FV Gauss--Newton estimator. Errors are medians over all states and over the weak, transitional, and strong charge bands; coverage is the fraction of 95% intervals containing the truth for the dimensionless wall charge $q_w \propto R\sigma$, reported for all states and by charge band. The dual protocol reaches 1.7% median charge error at 1% noise but its intervals are undercovered (68% for $q_w$), while the reversal potential is strongly state dependent (263% weak-charge versus 7.3% strong-charge). Single-salt conductance remains unidentifiable.

| Method | Median $I$ rel. error | Median $V_\mathrm{rev}$ abs. error (mV) | Median time (s) |
|---|---|---|---|
| Ohmic bulk | $8.06\times10^{-5}$ | 0.03907 | $9.21\times10^{-6}$ |
| Linearized PB | $3.21\times10^{-6}$ | 0.03907 | $5.21\times10^{-4}$ |
| POD--RBF | $3.56\times10^{-4}$ | 0.03907 | $2.22\times10^{-3}$ |
| PI-DeepONet | $1.06\times10^{-3}$ | 0.03838 | $1.42\times10^{-4}$ |
| Ridge-calibrated operator baseline | $8.51\times10^{-4}$ | 0.1384 | $1.09\times10^{-3}$ |
| One-step Gummel | $1.80\times10^{-4}$ | 0.03838 | $7.36\times10^{-3}$ |
| Full SG/FV | 0 | 0 | 0.235 |
| Pino-Controller (warm SG/FV) | $3.03\times10^{-12}$ | 0.03838 | $1.53\times10^{-1}$ |

| Protocol | Noise | $q_w$ all | $q_w$ weak | $q_w$ trans. | $q_w$ strong | $R$ all | $q_w$ cov. all | weak | trans. | strong |
|---|---|---|---|---|---|---|---|---|---|---|
| single salt | 0.2% | 140.71% | 997.51% | 486.04% | 70.48% | 0.34% | 100% | 100% | 100% | 100% |
| single salt | 1% | 140.41% | 1605.31% | 485.20% | 77.31% | 0.30% | 100% | 100% | 100% | 100% |
| single salt | 2% | 138.44% | 940.83% | 206.11% | 67.67% | 0.49% | 100% | 100% | 100% | 100% |
| reversal potential | 0.2% | 31.92% | 94.03% | 72.22% | 6.09% | 0.05% | 62% | 58% | 49% | 78% |
| reversal potential | 1% | 20.67% | 263.41% | 37.65% | 7.29% | 0.20% | 56% | 44% | 46% | 78% |
| reversal potential | 2% | 24.18% | 382.51% | 44.42% | 7.72% | 0.33% | 52% | 35% | 40% | 80% |
| dual + selectivity | 0.2% | 1.47% | 1.41% | 1.36% | 2.62% | 0.03% | 4% | 13% | 0% | 0% |
| dual + selectivity | 1% | 1.72% | 1.38% | 1.40% | 2.41% | 0.17% | 68% | 89% | 70% | 44% |
| dual + selectivity | 2% | 2.29% | 2.55% | 1.80% | 2.74% | 0.39% | 87% | 87% | 91% | 83% |


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
Require: parameter box B = [-50,50] mV x [20,200] mol m⁻³ x [-0.05,0] x [3,8] nm
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

**Why this matters.** If $c_+$ and $c_-$ are predicted independently, the error in $\rho=c_+-c_-$ is the difference of two errors, each of order $\epsilon c$, while $|\rho|$ can be much smaller than $c$. The relative error in $\rho$ therefore scales like $\epsilon c/|\rho|$ and diverges in weakly charged states. Predicting $s$ and $\rho$ removes this cancellation by construction. Relative to a naive direct loss the empirical reduction is 8.63-fold (DeepONet) and 10.29-fold (FNO), but much of that gap is an optimization pathology: with the matched field-normalized loss the direct model reaches $0.238\pm0.057$ net-charge error versus $0.207\pm0.015$ for the hyperbolic map, with 23% higher current error. The robust gain is therefore 13--15% on net charge and 18--23% on current, plus exact positivity and charge identity.

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

**States.** Fifteen held-out interpolation states selected to span the Debye-overlap range. Five of these are also used for the SG/FV-in-the-loop check in Supplementary Table S18.

**Measurements.** Exact SG/FV currents at $V_\mathrm{app}=-50,-25,0,25,50$ mV. Single-salt uses the state's own $C_0$; multi-salt uses 10, 30, and 100 mol $m^{-3}$; dual adds the near-wall Donnan selectivity $S=(c_+-c_-)/(c_++c_-)|_{r=1}$ at the same biases. Each observation is perturbed by independent 0.2%, 1%, or 2% relative Gaussian noise; 20 realizations are used per state--protocol--noise cell.

**Search.** The Stage-18 estimator is a $61\times61$ candidate grid in $(q_w,R)$ over the dataset box, followed by quadratic refinement; the complete study is repeated on a $121\times121$ grid for the nesting check. The Stage-20 estimator is a blind, refreshed-Jacobian Gauss--Newton iteration in $m=(\ln R,\ln|\sigma|)$, with observations generated at $41\times121$ and inverted at $21\times61$. Its stopping rule is $\|\Delta m\|_\infty<10^{-3}$ or at most six iterations, and boundary hits and non-converged realizations are recorded. The final covariance is evaluated at the converged estimate; standard errors for $R$, $\sigma$, and $q_w$ are $\sqrt{C_{00}}$, $\sqrt{C_{11}}$, and $\sqrt{C_{00}+2C_{01}+C_{11}}$, respectively. We report medians, interquartile ranges, and separate 95% coverage for the three parameters over the 300 realizations per cell.

**Identifiability diagnostic.** The Fisher matrix uses $J=\partial\ln|I|/\partial\ln(R,|q_w|)$ at four non-zero biases, whitened by 1% relative noise, and $\mathcal{F}=J^\top J/\sigma^2$. The negative-$q_w$ plane covers $20\le C_0\le200$ mol $m^{-3}$ and $-0.05\le q_w<0$ at $R=5$ nm on an $18\times18$ grid.

**Results.** See Supplementary Table S9 for the protocol summary, S18 for the grid nesting and SG/FV-in-the-loop check, S19 for the $8\times8$ exact sensitivity comparison, and S22 for the Stage-20 blind refreshed-Jacobian results and the three-way bias diagnostic. At 1% noise, single- and multi-salt conductance leave the median $q_w$ error above 100%, while the Stage-18 dual protocol gives 4.83% for $q_w$ and 0.0999% for $R$; the dual $q_w$ and $R$ estimates change by 0.40% and 0.91% between the 61 and 121 grids. An SG/FV-in-the-loop check on five states gives a more conservative 8.7% dual $q_w$ error and 0.45% radius error. The Stage-20 refreshed-Jacobian estimator is reported separately and is not pooled with the Stage-18 grid search.

---

## S7b. Stage-10 scaling, trunk, and multi-protocol identifiability

This section documents the three Stage-10 studies. All values are produced by `scripts/run_stage10_upgrades.py` and archived in `results/stage10_upgrades.json`, `results/tables/table_s14_radial_trunk_and_scaling_ablation.csv`, and `results/tables/table_s15_fisher_identifiability_protocols.csv`. The corresponding display tables are S3 and S9 in Section S0.

Fitted power laws $\epsilon=a\,N_\mathrm{train}^{-\alpha}$: $\alpha_\rho=0.552$ and $\alpha_\psi=0.535$ in-distribution, but only $\alpha_\rho=0.130$ and $\alpha_\psi=0.118$ out of distribution and $\alpha_I=0.369$ in-distribution. The interpolation error therefore follows approximately $N^{-1/2}$, whereas OOD error is nearly data-insensitive.

The Fisher-information plane (Figure 3F) spans $20\le C_0\le200$ mol $m^{-3}$ and $-0.05\le q_w<0$ at $R=5$ nm on an $18\times18$ grid. The dimensionless current sensitivity is $\partial\ln I/\partial\ln R=2.00$ (the current scales as the pore area), while the median $\partial\ln I/\partial\ln q_w$ is $-2.5\times10^{-5}$; the condition number $\kappa(\mathcal{F})$ ranges from $7.8\times10^{7}$ to $1.6\times10^{11}$ (median $1.17\times10^{9}$), and $99.1%$ of the plane exceeds $10^{8}$. The near-null Fisher direction is the charge axis at every operating point, which is why conductance-only inversion cannot recover $q_w$. Multi-salt current--voltage data do not remove the degeneracy; the independent near-wall selectivity observable does, reducing the median 1%-noise charge error to 4.8%.

The expanded dataset used for the scaling study appends 144 new in-distribution Latin-hypercube states (seed 20261020) to the original 240, giving 384 states (`data/processed/pnp_parametric_384.npz`, 6.36 MB) with a maximum residual of $9.86\times10^{-10}$ and strictly positive concentrations.

---

## S7c. Reviewer-upgrade experiments (Stages 11--12)

This section documents the three Stage-11 studies and the Stage-12 OOD decomposition. All values are produced by `scripts/run_stage11_reviewer_upgrades.py` and archived in `results/stage11_upgrades.json` and the matching `results/tables/` CSV files. The corresponding display tables are S11, S12, and S14 in Section S0.

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

# 10. Stage-19 hybrid solver and reversal-potential study
.venv/bin/python scripts/run_stage19_hybrid_solver_and_reversal.py

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
