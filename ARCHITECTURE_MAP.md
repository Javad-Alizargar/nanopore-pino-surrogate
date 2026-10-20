# Architecture map

| Stage | Script | Inputs | Outputs | Figures/Tables |
|---|---|---|---|---|
| Dataset | `scripts/generate_parametric_dataset.py` | `scripts/operators/constants.py`, `scripts/sg_baseline/` | `data/processed/pnp_parametric_240.npz`, `results/dataset_manifest.json` | Table 1 |
| Training/eval | `scripts/train_and_evaluate_suite.py` | dataset, `scripts/operators/` | `results/models/*.pt`, `results/statistics.json`, `results/tables/table3_*.csv` | Tables 2--5, Figures 2--4 |
| FNO budget | `scripts/fno_extended_check.py` | dataset | `results/tables/table5_*.csv` | Table 5 |
| Inverse | `scripts/run_inverse_recovery_demo.py` | dataset, ONNX models | `results/inverse_recovery_demo.json`, `results/tables/table6_*.csv` | Table 6, Figure 6B |
| Timing | `scripts/benchmark_sg_m1.py` | models, solver | `results/tables/table4_*.csv` | Figure 6A |
| Stage-10 upgrades | `scripts/run_stage10_upgrades.py` | 240/384-state datasets, `scripts/operators/` | `results/stage10_upgrades.json`, `results/tables/table_s14_*.csv`, `table_s15_*.csv`, `data/processed/pnp_parametric_384.npz` | Figures 3E,F, 4D; Tables S3, S15 |
| Stage-11 upgrades | `scripts/run_stage11_reviewer_upgrades.py` | 240-state dataset, ONNX operator, SG/FV solver | `results/stage11_upgrades.json`, `results/tables/table_sgfv_fisher_validation.csv`, `table_constraint_schemes_4way.csv`, `table_physics_weight_sensitivity.csv` | Figures 3A--D, 6C,D; Tables S16--S18 |
| Figures | `scripts/figures/*.py` | `results/*` | `figures/*.png`, `pdf` | Figures 1--6 |
| Deployment | `scripts/build_web_simulator.py`, `scripts/build_web_bundle.py` | ONNX models | `web_simulator/` | -- |
| Live apps | `scripts/serve_interactive.py`, `scripts/app_gradio.py` | solver, models | HTTP/Gradio services | -- |

Constraint handling lives in `scripts/operators/boundary_ansatz.py`
(exact reservoir Dirichlet + charge-balanced reconstruction). The SG face
flux and physics residuals live in `scripts/operators/losses_and_metrics.py`.
