# nanopore-pino-surrogate

Parameter-conditioned physics-informed neural operator (PINO) for steady
axisymmetric Poisson--Nernst--Planck (PNP) transport in a charged cylindrical
nanopore. The repository contains the conservative Scharfetter--Gummel
finite-volume reference solver, the 240-state benchmark and a 384-state scaling
expansion, a multi-scale radial physics-informed DeepONet, a real-DFT Fourier
neural operator, the inverse recovery demo, a trunk ablation, a data-scaling
study, a Fisher-information and multi-protocol identifiability analysis, a
four-way constraint-scheme benchmark, an exact SG/FV Fisher validation, a
physics-weight sweep, and a static browser simulator.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/test_operators_smoke.py          # ONNX/PyTorch parity smoke test
python scripts/run_inverse_recovery_demo.py     # five-state inverse demo
python scripts/run_stage10_upgrades.py          # trunk ablation, scaling law, identifiability
python scripts/run_stage11_reviewer_upgrades.py # 4-way constraints, physics sweep, exact Fisher
python scripts/run_stage12_diagnostics.py       # OOD decomposition, selectivity, positivity
python scripts/serve_interactive.py             # live SG/FV + operator service
python scripts/app_gradio.py                    # Gradio comparison app
```

Open `web_simulator/index.html`, or use the hosted portal at
<https://pino.statwhy.com/> (ONNX operator predictions, verified benchmark
cases, and a live SG/FV reference solver).

## Layout

| Path | Contents |
|---|---|
| `scripts/operators/` | DeepONet/FNO, boundary ansatz, SG losses, constants |
| `scripts/sg_baseline/` | conservative SG/FV reference solver |
| `scripts/train_and_evaluate_suite.py` | three-seed training/evaluation |
| `scripts/run_inverse_recovery_demo.py` | radius/charge inverse demo |
| `scripts/run_stage10_upgrades.py` | trunk ablation, 384-state scaling law, Fisher/multi-protocol identifiability |
| `scripts/run_stage11_reviewer_upgrades.py` | four-way constraint benchmark, physics-weight sweep, exact SG/FV Fisher validation |
| `scripts/run_stage12_diagnostics.py` | OOD spatial/parametric decomposition, selectivity validation, positivity stress test |
| `scripts/enforce_sequential_numbering_and_inline_figures.py` | first-appearance reference/table renumbering and inline-figure audit |
| `scripts/figures/` | figure generation and layout verification |
| `results/` | statistics, tables 1--6 and S3, S14--S19, verification JSONs, export models |
| `data/processed/` | 240-state dataset and 384-state scaling expansion |
| `web_simulator/` | static browser simulator (ONNX + JS SG engine) |
| `tests/` | smoke and parity tests |

## Reproducibility notes

- The dataset was generated on an Apple M1 with Python 3.14, seed 20261018.
- `results/statistics.json` and `results/tables/*.csv` hold full-precision
  values; the manuscript reports 3--4 significant figures.
- Fixed physical settings: $T=298.15$ K, $\varepsilon_r=78.5$,
  $D_+=1.96\times10^{-9}$, $D_-=2.03\times10^{-9}$ m$^2$ s$^{-1}$, $L=30$ nm.
- License: MIT. Archived copy to be deposited on Zenodo at acceptance.
