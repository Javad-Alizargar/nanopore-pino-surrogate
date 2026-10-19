# CAPABILITY REPORT — Stage 1 Preflight

Project: `nanopore_pino_surrogate`
Governing spec: `OPEN INTERPRETER RESEARCH CONTROLLER — MASTER PROMPT v2.0` (`MASTER_PROMPT_ENHANCED.txt`, adopted 2026-10-18)
PROJECT_ROOT: `/Users/javadalizargar/Google drive MAC/PINN to a Neural Operator`
Report date: 2026-10-18
Stage: 1 (preflight, capability & accounting bootstrap, project brief)

---

## 1. Runtime environment

| Item | Value |
|---|---|
| OS | macOS 14.1 (build 23B74), Darwin 23.1.0, arm64 |
| CPU | Apple M1, 8 cores (8 physical) |
| RAM | 8 GB (8,589,934,592 bytes) |
| GPU | Apple M1 GPU, Metal 3 — **no CUDA**, no discrete accelerator |
| Disk (Data volume) | 460 GB total, **~16 GB free (97% used)** — low headroom |
| Shell | zsh |
| Timezone | machine local; author in America/Taipei |

### Toolchain

| Tool | Path | Version |
|---|---|---|
| python3 | `/opt/homebrew/bin/python3` | 3.14.7 |
| pip | module | 26.2.1 |
| git | `/usr/bin/git` | 2.39.3 (Apple Git-146) |
| gh | `/opt/homebrew/bin/gh` | 2.96.0 |
| node | `/opt/homebrew/bin/node` | v26.5.0 |
| npm | `/opt/homebrew/bin/npm` | present |

Note: a bare `python` alias is **not** on PATH; use `python3`.

### Scientific / ML packages

Installed (verified by distribution metadata):

| Package | Version |
|---|---|
| torch | 2.14.1 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| requests | 2.34.2 |

**Not installed (required for later stages):**

| Needed for | Package | Status |
|---|---|---|
| Figures, plots | matplotlib, seaborn | MISSING |
| Manuscript DOCX I/O | python-docx | MISSING |
| Browser/ONNX simulator | onnx, onnxruntime | MISSING |
| Interactive app | gradio **or** streamlit | MISSING |
| Optional SDK | openai (we used `requests`) | MISSING |
| Data containers | h5py | MISSING |
| Image export | Pillow | MISSING |

Environment snapshot (machine-readable): `logs/environment_stage1.json` (52 distributions).

**Capability implications:** PyTorch CPU training is available but limited to 8 GB RAM and integrated GPU (Metal/MPS only; no CUDA). Heavy training must be small-batch and memory-aware. Matplotlib, python-docx, onnx/onnxruntime, and a UI framework must be installed (in a project venv) before figure, manuscript, or simulator stages. Disk headroom (~16 GB) constrains large dataset generation and model checkpoints — plan compact float32/float64 arrays and prune intermediates.

---

## 2. Credential audit (presence only — no secret values read, printed, or logged)

Credential folder: `apis/` (legacy lowercase; the canonical `APIS/` folder does **not** exist).

| Canonical name expected | Present? | Actual file found | Bytes | Verified live? |
|---|---|---|---|---|
| `OPENAI.txt` | no | `apis/chatgpt.txt` | 164 | **yes** (models 200, chat 200) |
| `deepseek.txt` | **yes** | `apis/deepseek.txt` | 35 | **yes** (chat 200) |
| `NCBI.txt` | **yes** | `apis/NCBI.txt` | 36 | **yes** (esearch 200, count 102) |
| `OPENALEX.txt` | **yes** | `apis/OPENALEX.txt` | 22 | **yes** (works 200, count 1302) |
| `GITHUB_fine_grained.txt` | no | `apis/GITHUB_fine_grained 2.txt` | 93 | not tested (no GitHub auth action taken) |

Additional credential-like files present: `apis/Meta API.txt` (64 B), `apis/Open Access API.txt` (32 B).

**Security handling:** all key files were read in-process only. No key content, substring, prefix, hash, or URL containing a key was emitted to any console, file, report, or prompt. The smoke-test script scrubs known secret values from any error text before printing. Values are only ever referenced as booleans and byte lengths here.

**Naming caveat:** the canonical names `OPENAI.txt` and `GITHUB_fine_grained.txt` are absent. The OpenAI key is stored as `chatgpt.txt` (verified functional). The GitHub token is stored as `GITHUB_fine_grained 2.txt` (space + `2`) and was intentionally **not** exercised because `AUTO_PRIVATE_GITHUB_PUSH=false`. Confirm the intended GitHub token file name before any authorized GitHub work.

---

## 3. Existing project assets found

### 3.1 Governing spec
- `MASTER_PROMPT_ENHANCED.txt` (40,924 B) — adopted as the governing specification.

### 3.2 Prior DA-PINN study (fixed-state baseline) — located in sibling directory, **not** copied into PROJECT_ROOT
- Submission dir: `/Users/javadalizargar/Google drive MAC/comphy revisopn by opencode/COMPHY-D-26-00857_revision_2026-10-14/01_submission/`
  - `Manuscript_clean.docx` (12.6 MB) — "Exact-boundary data-assimilated physics-informed neural networks for coupled Poisson–Nernst–Planck transport at charged nanopore biointerfaces" (author Javad Alizargar).
  - `Manuscript_highlighted.docx`, `Supplementary_information.docx`, `Response_to_reviewers.docx`, `Figure_captions.docx`, `Highlights.docx`, `Cover_letter.docx`, `Figures/`.
- Reproducibility package: `.../02_reproducibility/nanopore-da-pinn_v0.007_source_and_results.zip` (22.3 MB):
  - `src/pnp_da_pinn/`: `sg_solver.py` (744 lines, Scharfetter–Gummel finite-volume PNP solver — the verified reference), `pinn.py`, `revision_training.py`, `training.py`, `continuation.py`, `edl.py`, `grid.py`, `metrics.py`, `scales.py`, `conservative.py`, `baseline.py` (prototype, non-conservative, negative comparison), `config.py`.
  - `src/nanopore_da_pinn.py` (455 lines).
  - `scripts/` (16 runners/plotters), `configs/` (case YAMLs), `outputs/revision/` (metrics, weights, figures), `tests/test_conservative.py` (9 tests, all pass).
  - Prior runs: **Windows 10, Intel Xeon Gold 5215, Python 3.14.4, torch 2.14.1+cpu, float64, 2 threads** (`outputs/revision/environment.json`).

### 3.3 Prior DA-PINN key numbers (from retained artifacts)
- Reference case: R = 5 nm, L = 30 nm, grid 21×61, 100 mol m⁻³, 50 mV, wall surface-charge strength −0.01.
- Reference potential fixed-point residual 7.19×10⁻¹⁰; relative axial current range 7.74×10⁻¹⁴.
- DA-PINN (3 seeds) mean relative L2 errors: ψ 0.000382; c₊ 0.000106; c₋ 0.000106; charge density 0.145; current profile 0.00102.
- Benchmark table: empty-FD 0.039 s; bound-FD 88.7 s (2206 iters); bound-DA-PINN inference 0.0054 s.
- Explicitly fixed-state interpolants — no parameter generalization claimed (this is the gap the new project targets).

### 3.4 Literature tooling on disk
- `new paper nanopore/litbank/` — 16-stage literature pipeline (OpenAlex/EuropePMC/NCBI/Crossref retrieval, dedup, classification, novelty, exports).
- `apis/analyze_articles.py`, `apis/search_and_download.py`, `apis/fetch_articles.py`, `apis/download_pdfs.py`, `apis/generate_manuscript.py` (prior helper scripts).
- Raw caches at Google-Drive root: `openalex_works_raw.json` (20.6 MB), `pubmed_articles_raw.json` (54.9 KB).

### 3.5 Not found
- No `Manuscript_clean.docx` or solver/PINN source inside PROJECT_ROOT itself. The prior study lives in the sibling path above and has **not** been copied here (Stage 1 is inspect-only; copying will be a Stage 2 decision).

---

## 4. Budget ledger verification

Implementation: `orchestration/budget_ledger.py` (transactional SQLite, WAL, `BEGIN IMMEDIATE` admission lock), DB at `logs/budget_ledger.sqlite` (git-ignored). Pricing/models: `config/pricing.json`, `config/models.json`. Budget: `config/project_config.json`.

### 4.1 Dry-run tests (zero paid calls) — `orchestration/test_budget_ledger.py`

```
$ python3 -m unittest -v test_budget_ledger
test_two_simultaneous_reservations_near_boundary ... ok
test_per_request_cap_denies_oversized_request     ... ok
test_reconcile_replaces_reservation_with_calculated_cost ... ok
test_outstanding_survives_restart_and_is_not_refunded    ... ok
test_timeout_retains_full_reservation             ... ok
test_unpriced_model_is_denied                     ... ok
Ran 6 tests — OK
```

Required Section-4 scenarios all covered: boundary race (exactly one of two simultaneous reservations admitted; the other denied `project_budget_exceeded`), restart recovery (outstanding reservation survives reopen; not refunded), unknown-usage timeout (full reservation retained as `unresolved`; committed stays 0), unpriced-model denial.

### 4.2 Live smoke test (one 16-token call per provider, gems via free `/models` probe)

| Provider | Model | Chat HTTP | Usage | Calc cost (USD) | Tier |
|---|---|---|---|---|---|
| DeepSeek | `deepseek-flash` | 200 | 37 prompt / 12 completion (10 reasoning) | 0.00001275 | off-peak |
| OpenAI | `gpt-4o-mini` | 200 | 14 prompt / 1 completion | 0.00000270 | standard |

Read-only metadata APIs: OpenAlex `works` HTTP 200 (count 1302, query "Poisson Nernst Planck nanopore"); NCBI E-utilities `esearch` HTTP 200 (PM count 102, query "nanopore Poisson-Nernst-Planck").
Raw results: `logs/smoke_connectivity.json`.

### 4.3 Ledger state after smoke

```
committed      = $0.00001545   (deepseek 0.00001275 + openai 0.00000270)
outstanding    = $0.00000000
total          = $0.00001545
reservations   = 3 attempts (2 reconciled, 1 released)
openai subbudget used = $0.00000270 / $1.00
project remaining     = $4.99998455 / $5.00
warning: false   stop_new_tasks: false
```

One reservation was `released` (not billed): the first OpenAI attempt used a malformed URL (`/v1/v1/...` → HTTP 404) caused by our own endpoint builder; it produced no completion and no tokens. It was corrected and retried exactly once (the successful 200 above). This is recorded in `reports/DECISIONS.md`.

**Spend to date: $0.00001545 (≈ $0.00).**

### 4.4 Accounting limitations (honest disclosure)
- The controller (this OpenCode session) is **not** routed through the ledger; its own model usage is not metered by `budget_ledger.py`. The $5.00 ceiling therefore governs *scripted/gateway* paid calls, not the interactive session. Do not claim a project-wide hard cap.
- No local gateway/proxy is installed. Enforcement is a library-level admission controller; a process that calls a provider SDK directly and bypasses `budget_ledger.py` would not be blocked. All project scripts must import and use the controller.
- Provider-side hard limits and spend alerts were not configured in this stage.

---

## 5. Capability verdict

| Capability | Status |
|---|---|
| Adopt governing spec | DONE |
| Project config persisted | DONE (`config/project_config.json`) |
| Directory tree + `.gitignore` (secrets excluded) | DONE |
| Local git repo initialized (no commit, no push) | DONE |
| Preflight environment audit | DONE |
| Credential presence audit (no secret exposure) | DONE |
| Existing-asset inspection | DONE |
| Budget ledger + admission control | DONE + dry-tested |
| Live provider smoke test | DONE (DeepSeek, OpenAI, OpenAlex, NCBI) |
| Exact-boundary FNO / PI-DeepONet training | NOT STARTED (Stage 2+) |
| Browser simulator / dual repos | NOT STARTED (Stage 4+) |

Readiness for Stage 2: **GO**, subject to (a) installing a project venv with matplotlib, python-docx, onnx/onnxruntime, and a UI framework; (b) deciding where to stage the prior DA-PINN solver (copy vs reference); (c) confirming the GitHub token file name.
