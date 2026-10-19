# USER_RULES

User-approved rules and change history for `nanopore_pino_surrogate`. Only the controller updates this file. Append changes with dates.

## Approved settings (Stage 1, 2026-10-18)
- `PROJECT_NAME`: nanopore_pino_surrogate
- `TASK_MODE`: research_and_manuscript
- `TARGET_JOURNAL`: Journal of Computational Physics (JCP) / Computer Physics Communications (CPC)
- `MANUSCRIPT_LANGUAGE`: English
- `DATA_MODE`: local_only — generate synthetic numerical reference data via the verified SG/FV solver; fetch literature metadata via APIs only.
- `AUTO_PRIVATE_GITHUB_PUSH`: false — prepare repo structure; push only when explicitly commanded.
- `AUTO_PUBLIC_RELEASE`: false.
- Budget: $5.00 project ceiling; $1.00 OpenAI subbudget; $0.50 final-review reserve (inside the ceiling).

## Stage 2 decisions (2026-10-18, user-supplied)
- **Prior solver strategy:** COPY verified v0.007 code (`src/pnp_da_pinn/`, `configs/`, `tests/`, baseline reference outputs) from the sibling archive into `download/baseline_v0007/` as immutable raw provenance with SHA-256 hashes in `download/DATA_MANIFEST.md`; copy working solver modules into `scripts/sg_baseline/`.
- **UI framework:** Gradio for the local Python tool + pure client-side HTML/JS (`onnxruntime-web`) simulator for zero-cost static GitHub Pages hosting.
- **Credential paths:** `apis/chatgpt.txt` (OpenAI) and `apis/GITHUB_fine_grained 2.txt` (GitHub). `AUTO_PRIVATE_GITHUB_PUSH` stays `false` until Stage 7.
- **Disk conservation:** create `.venv` with `--system-site-packages` to reuse system torch/numpy/scipy/pandas/scikit-learn/requests; install only `matplotlib`, `python-docx`, `onnx`, `onnxruntime`, `gradio` inside it.

## Stage 3 decisions (2026-10-18, user-supplied)
- **GitHub token filename:** confirmed as `apis/GITHUB_fine_grained 2.txt` (presence + byte length only). `AUTO_PRIVATE_GITHUB_PUSH` remains `false` until Stage 7.
- **CPC track strategy:** format the manuscript and code archive to be **dual-compatible** with **CPC** (both the *Computer Programs in Physics / CPiP* track **and** the *Computational Physics Papers / Methodological* track) **and** with **JCP**. Concretely: include the standard CPC "Program Summary" metadata block immediately after the Abstract **and** retain full mathematical derivation + ablation rigor. Details in `reports/JOURNAL_REQUIREMENTS.md`.

## Stage 5 decisions (2026-10-18, user-supplied)
- **Primary manuscript architecture:** lock **Multi-Scale Radial PI-DeepONet with charge-balanced reparameterization (`deeponet_reparam`)** as the primary surrogate (≈38× smaller than the FNO, ≈14× faster ONNX inference at 0.072 ms, superior EDL resolution via radial exponential basis functions). **Exact-Boundary FNO-2D (`fno_reparam`)** is presented as the spectral-operator benchmark.
- **Eliminate the "under-trained FNO" reviewer objection:** before final figures, run a single bounded **300-epoch** `fno_reparam` (seed 42, ≈4 min on M1) and report both fixed-time (110-epoch) and extended-epoch (300-epoch) FNO accuracy in `results/statistics.json` and `results/tables/`, to show whether the DeepONet advantage is fundamental (radial EDL basis vs. global spectral modes) rather than an epoch-count artifact.
- **Interactive simulators:** a client-side browser simulator using `onnxruntime-web` (static, GitHub-Pages-ready) and a companion Python Gradio app that can run the live SG/FV solver for side-by-side comparison.

## Standing rules
- Stage 1 scope was strictly setup/accounting/brief; no heavy training or paid worker loops.
- Never print, echo, log, or embed any credential value or substring; report presence + byte length only.
- Prepare the concrete result before asking; ask only for material scientific decisions, missing access/data, or limit increases.
- No GitHub push, public release, deployment, or manuscript submission without explicit command.
- Stop immediately after Stage 1’s `reports/REPORT_FOR_CHATGPT.md` (this stage).

## Change log
- 2026-10-18 — Initial rules created from the Stage 1 user instruction and Section 1 defaults. Config persisted to `config/project_config.json`.
- 2026-10-18 — Stage 2 user decisions added (solver copy strategy, Gradio + ONNX/JS UI, locked credential paths, disk-conserving venv).
- 2026-10-18 — Stage 3 user decisions added (GitHub token filename confirmed; dual-compatible CPC CPiP+Methodological / JCP manuscript strategy).
- 2026-10-18 — Stage 5 user decisions added (deeponet_reparam as primary architecture; 300-epoch FNO verification check; browser ONNX + Python Gradio simulators).
