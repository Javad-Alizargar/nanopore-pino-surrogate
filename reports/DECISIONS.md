# DECISIONS

Scientific and implementation decisions, newest first. This is an append-only log.

## D-056 · 2026-10-19 · Stage 7: submission cover letter exported
`scripts/export_cover_letter.py` generates `manuscript/Cover_Letter_CPC_JCP.md` and `manuscript/Cover_Letter_CPC_JCP.docx` addressed to the Editors-in-Chief of *Computer Physics Communications* / *Journal of Computational Physics*. It states the title, author and affiliation, and how the work advances beyond the prior fixed-state DA-PINN study (`COMPHY-D-26-00857`) through five contributions: the 4D parameter-conditioned operator, the exact-boundary charge-balanced hyperbolic reparameterization, the DeepONet-vs-FNO head-to-head comparison, the inverse-sensing demonstration with the radius/surface-charge identifiability boundary, and the live simulator plus MIT reproducibility package. The DOCX reuses the manuscript exporter's inline-math Unicode path; read-back confirms no LaTeX leakage.

## D-055 · 2026-10-19 · Stage 7: private full archive deployed and root project committed
`reproducibility/github/` (full research state, credentials and `.venv` excluded) was committed with `"Complete private research archive v1.0.0 (Stage 1-7 verified)"` and pushed to the **private** repository `Javad-Alizargar/nanopore-pino-surrogate-private` (visibility confirmed `private: true` via the API before pushing). The first push over `gh repo create --push` failed with an HTTP 408 on the 80 MB pack; the retry succeeded with `http.postBuffer` raised and HTTP/1.1 forced. A clean local commit was also made in the project root; `git status --ignored` confirms `apis/`, `logs/`, `download/`, and `.venv/` remain untracked, and the generated `reproducibility/` staging tree was added to `.gitignore`.

## D-054 · 2026-10-19 · Stage 7: public repository released and GitHub Pages activated
`reproducibility/public_repo/` (138 files with the two `.nojekyll` markers, 22.5 MB) was committed with `"Initial public release v1.0.0: ..."`, pushed to the **public** repository `Javad-Alizargar/nanopore-pino-surrogate` (`private: false` confirmed), and tagged `v1.0.0`. GitHub Pages was enabled from branch `main`, path `/`; the build reached `built` and the simulator and its assets (HTML/CSS/JS, embedded bundle, model metadata, both seed-42 ONNX models) all return HTTP 200 at `https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/`.

## D-053 · 2026-10-19 · Stage 7: token never touches disk or remotes
The fine-grained token is read in-memory from `apis/GITHUB_fine_grained 2.txt` and passed only through the `GH_TOKEN` environment to `gh` and via `gh auth git-credential` as a transient `-c credential.helper` argument for `git push` (`scripts/deploy_github_repos.py`). Remote URLs are clean `https://github.com/...` with an in-script assertion that no `@` (and therefore no token) appears in them. The pre-push run of `scripts/audit_release_packages.py` was **20/20 PASS** with zero secret hits, and no token value was printed, echoed, or logged at any point.

## D-052 · 2026-10-19 · `.nojekyll` added so GitHub Pages serves every asset
`scripts/build_reproducibility_packages.py` now writes an empty `.nojekyll` at the public-package root and inside `web_simulator/`, so Pages serves the tree verbatim (no Jekyll processing, no risk of dropping underscore-prefixed paths). This raises the public package from 136 to 138 files; both markers are recorded in `PACKAGE_MANIFEST.json`.

## D-051 · 2026-10-19 · Stage 7 authorized: dual GitHub deployment
On explicit user command, Stage 7 was executed with `AUTO_PRIVATE_GITHUB_PUSH: true` and `AUTO_PUBLIC_RELEASE: true`: the pre-push audit was re-confirmed at **20/20 PASS** before any push, the capability check authenticated as `Javad-Alizargar` (token valid to 2026-11-06), both target repositories were created with the requested visibility, and both releases plus GitHub Pages were verified through the API and by live HTTP requests.

## D-050 · 2026-10-19 · Stage 6B: independent cross-provider review resolved
Ran two independent AI reviewers (`scripts/run_independent_review.py` → `results/independent_review.json`): OpenAI `gpt-4.1-mini` (11 issues, $0.00549) and DeepSeek `deepseek-v4-pro` (12 issues, $0.02112). Both recommended **minor revision**. All 23 issues were adjudicated against the code and validated results (no majority vote) and fixed; every issue is logged in `reports/REVIEW_RESPONSE.md`. The critical finding was the reservoir-limit description: the manuscript and supplement implied `s→0`, but `scripts/operators/boundary_ansatz.py` implements `s=2·exp(clip(z(1−z)G_s,−15,15))`, so `s→2` and `c±→C0` **exactly**. Also added the dimensionless PNP system (Supplement S3.1), the exact data/physics loss, the multi-scale trunk features `κ∈{1,3,10,30}`, CPC metadata fields, DA-PINN and break-even caveats, absolute q_w errors, corrected Figure 5, and the live benchmark current. This is an AI-assisted review, not expert peer review.

## D-049 · 2026-10-19 · Number formatting: 3–4 significant figures in the narrative
Applied `scripts/format_manuscript_numbers.py` to `Manuscript.md` and `Supplementary_Materials.md`: 334 values rounded to 3–4 significant figures (247 in the main text, 87 in the supplement, across three widening passes; the remaining long decimals are DOI fragments in the reference list). Code spans, math, URLs, DOIs, arXiv IDs, the value 298.15 K, and the entire reference list are protected. Exact values remain in `results/statistics.json` and `results/tables/*.csv`; both manuscripts state this.

## D-048 · 2026-10-19 · DOCX export is generated, not hand-edited
`scripts/export_manuscript_docx.py` converts the canonical markdown to `manuscript/Manuscript_PINO_clean.docx` and `manuscript/Supplementary_Materials_PINO.docx`. Display equations are rendered to images with matplotlib mathtext; inline math is converted to Unicode; Tables 1–6 are real Word tables; the six main figures are embedded with captions. The exported text contains no `$$` and no LaTeX commands. Rendered-page inspection is structural only (no Word/PDF viewer available in this environment); this limitation is disclosed.

## D-047 · 2026-10-19 · Dual reproducibility packages and automated secret audit
`scripts/build_reproducibility_packages.py` produces `reproducibility/public_repo/` (136 files, 22.5 MB; strict allowlist, no credentials/reports/governing spec) and `reproducibility/github/` (791 files, 80.2 MB; full research state, credentials and `.venv` excluded). Both carry a SHA-256 `PACKAGE_MANIFEST.json`. `scripts/audit_release_packages.py` scans both packages, the credential folder (`apis/`, values never printed), and all tracked/unignored files for credential values, common key prefixes, and secret file names, and checks Section 14 manuscript consistency. Result: **20/20 PASS** (`reports/FINAL_AUDIT_REPORT.md`). No repository push; `AUTO_PRIVATE_GITHUB_PUSH` remains **false**.

## D-046 · 2026-10-19 · Wall-condition sign reconciled to the solver
The main text wrote `−εrε0 ∂rψ = σ` while the solver uses `∂rψ̃ = q_w` (`scripts/operators/losses_and_metrics.py`). The two texts were reconciled to the code convention, with the sign stated explicitly, and the full dimensionless system including the SG face flux was added to Supplement S3.1. No numerical result changed.

## D-045 · 2026-10-19 · DeepSeek reasoning exhausted the output budget; disabled for review
`deepseek-v4-pro` and `deepseek-flash` returned empty `content` because all output tokens were consumed as `reasoning_tokens`; a probe showed both providers return content when `{"thinking": {"type": "disabled"}}` (or `reasoning_effort: "none"`) is set. The review harness now disables thinking for DeepSeek. Wasted attempts cost about $0.024; the review still finished inside the $0.15 task allowance.

## D-044 · 2026-10-19 · Stage 6A: inverse-recovery demo exposes a real identifiability limit
Built `scripts/run_inverse_recovery_demo.py` and inverted five-bias noisy I–V curves (1% Gaussian, 5 states, 400 candidate grid + Nelder–Mead, 2000 ONNX forward evaluations per state). Radius is identifiable: **median relative error 0.0045737918517061785**. The surface charge is **not** identifiable from conductance: median relative error **1.1539643805520952**, and the 1σ profile spans the full prior. This is not a fitting failure — with a symmetric electrolyte and near-equal diffusivities, the leading conductance is set by the total ionic strength, and the Donnan polarization induced by q_w changes the current by only ~0.005% at C0=100 and ~0.014% at C0=20 over the tested range (exact SG/FV), far below 1% noise. Reported as a scope statement rather than hidden: current-only inversion recovers geometry, not wall charge. Grid inversion costs **0.12526570879854262 s** per state versus **402.6 s** for 2000 SG/FV solves (speed-up **3213.968163046741**); the median state-level current change across the q_w prior is **0.0009625039831069234** (largest state-level change **0.0018995120447014644**), both below the 1% noise floor.

## D-043 · 2026-10-19 · Exact parameter counts replace the spec’s placeholders
The project spec quoted ≈34k (DeepONet) and ≈1.32M (FNO) parameters. Direct counts from the shipped models are **36,880** (DeepONet) and **460,488** (FNO). All manuscript text, Table 1, Figure 1D, the abstract and the supplement now use the exact counts; the spec figures are treated as stale. The FNO/DeepONet parameter ratio is 12.49×, not 38×.

## D-042 · 2026-10-19 · Figure 1 restructured to a 2×2 A–D layout
Figure 1 was rebuilt as a four-panel narrative at `figsize=(15,10)` with `GridSpec(2,2)`: **A** geometry/BCs/PNP, **B** the 4D operating space with split coverage (train 144 / interp 48 / OOD 48, benchmark #0), **C** the three architecture-independent constraints (exact reservoir Dirichlet → charge-balanced fields → hyperbolic reconstruction), **D** the two backbones side by side (DeepONet 36,880 params / 0.072 ms; FNO 460,488 / 1.04 ms). Panel letters are applied once via `K.panel_label`; panel titles no longer carry "(A)" prefixes. All six figures regenerate cleanly and the deterministic layout verifier reports **0 issues**; `figure_qa.json` was regenerated after the change.

## D-041 · 2026-10-19 · Vision-QA flags remain advisory; deterministic layout gate is authoritative
Re-review of the 15 `gpt-4o-mini` flags after the Figure-1 rewrite confirmed all were false positives (legend swatches, axis-label proximity, panel-letter proximity). No further paid vision calls were made. `scripts/figures/verify_figure_layout.py` remains the release gate: 6 px in-box padding, 12 px² text-overlap, 8 px PNG margin → **0 issues** for figs 1–6.

## D-040 · 2026-10-19 · Stage 6A manuscript drafts written against locked numbers
Created `manuscript/claim_evidence.csv` (28 claims C01–C28), `manuscript/Manuscript.md` (title → highlights → 234-word abstract → CPC Program Summary → Sections 1–5 → Tables 1–6 → declarations → 60 references) and `manuscript/Supplementary_Materials.md` (S1–S10). Every high-precision number in both drafts was checked verbatim against `results/statistics.json` or `results/tables/*.csv`; only DOI fragments remain as non-result decimals. Supplementary S4 was corrected to the exact three-seed means (c₋, σ_c, wall-derivative error, Poisson residual) and seed standard deviations from `summary_multiseed_global_rel_l2`. **No paid calls were made in Stage 6A**; ledger unchanged.

## D-039 · 2026-10-19 · Figure layout QA: deterministic verifier is the gate, vision QA is advisory
Text-overflow is now checked **programmatically**: `scripts/figures/verify_figure_layout.py` renders each figure at real DPI and asserts (i) annotation text inside its box/patch with ≥6 px pad, (ii) no two visible texts overlap (>12 px², same-axis ticks excluded), (iii) ≥8 px pure-white margin in the saved PNG; off-view tick labels are filtered by comparing each tick location to the axis view interval. Result: **60 → 0 issues** across figs 1–6 (`results/figure_data/figure_layout_report.json`). A budgeted `gpt-4o-mini` vision pass (`scripts/vision_qa_figures.py`, **$0.019878**, task allowance $0.03) is recorded as **advisory only** — every flag was manually cross-checked and found to be a false positive. Global `savefig.pad_inches` raised 0.03 → 0.08.

## D-038 · 2026-10-19 · Gradio fixes + zero-dependency live SG/FV server
The Gradio app's model `Dropdown` used the wrong choices order `(label, value)` (Gradio expects `(value, label)`), the plot helper returned a `BytesIO` instead of a numpy array, and the **compare** button was never wired — all fixed in `scripts/app_gradio.py` (`--host/--port`, `demo.queue()`). Added `scripts/serve_interactive.py` (stdlib `http.server`, `127.0.0.1:8765`) serving `web_simulator/` plus `GET /api/health` and `POST /api/solve_sg` (live float64 SG/FV). `launch_apps.command`/`launch_apps.sh` start both servers. **Live end-to-end verification** (`scripts/verify_live_servers.py` → `results/stage5b_live_server_verification.json`): interactive `POST /api/solve_sg` → 196.13722469169713 pA (direct, 4 iters, res 2.42e-10, 0.212 s); Gradio `/ui_predict` and `/ui_compare` return valid images (live solve 0.221 s). Both servers left running at 8765/7860.

## D-037 · 2026-10-19 · Web simulator rebuilt as a dual-engine (works over `file://` too)
Stage-5's `fetch()`-based `onnxruntime-web` page failed under `file://` (blocked fetch). Rebuilt `web_simulator/` as **dual-engine**: `scripts/build_web_bundle.py` inlines metadata and base64-embeds both ONNX graphs into `data/embedded_bundle.js` (5,662,351 B), and `engine_js.js` provides a pure-JS forward pass used by default; ORT remains for the HTTP path. Fixed a branch/trunk parameter-count bug → **JS-vs-ONNX parity 7.363e-07** worst case. `scripts/test_web_simulator_dom.js` Node DOM harness: **29/29 pass**.

## D-036 · 2026-10-19 · Interactive-simulator verification and shipped-ONNX provenance
Both interactive tools verified locally (no paid calls) → `results/stage5_simulator_verification.json`. (i) **Static browser simulator** (`web_simulator/`): index/style/app + `model_meta.json` (grid, constants, ranges, 8 verified SG/FV presets); both ONNX operators reload in onnxruntime; the in-browser Scharfetter–Gummel current formula reproduces the stored SG/FV dataset current to **5.7e-7** max relative over the preset states; all assets serve HTTP 200; `node --check` passes. (ii) **Gradio app** (`scripts/app_gradio.py`): headless smoke test runs the same code path as the UI — DeepONet mean I 196.132 pA vs live SG/FV 196.137 pA (I rel-L2 **7.4e-4**), ψ rel-L2 2.13e-4, live solve **0.206 s**. (iii) **ONNX provenance disclosure**: the shipped `fno_reparam_seed42.onnx` (and its web copy) is the **300-epoch** re-export (TASK 5.1, D-034), so its ONNX-vs-PyTorch parity is checked against `fno_reparam_seed42_300ep.pt` → max abs diff **2.38e-7** (DeepONet likewise 2.38e-7). All parities < 1e-5.

## D-035 · 2026-10-19 · Stage 5 presentation deliverables complete (figures + simulators)
Stage 5 presentation work declared **complete**: (i) the bounded **300-epoch FNO check** (D-034); (ii) **six publication-grade multipanel figures** (`manuscript/figures/fig1`–`fig6`, 300-DPI PDF+PNG) built by `scripts/figures/` with 3-stage QA (scientific / communication / render) recorded in `results/figure_data/figure_qa.json` and source data as `*_data.npz`; (iii) the **client-side ONNX browser simulator** (`web_simulator/`) and the **Python Gradio app** (`scripts/app_gradio.py`) with a live SG/FV comparison; (iv) verification record `results/stage5_simulator_verification.json`. Manuscript **drafting** (IMRAD text + CPC Program Summary block) remains the next Stage-5 sub-task.

## D-034 · 2026-10-18 · 300-epoch FNO verification check (user-directed)
Before final figures, run a single bounded extended-epoch check: `fno_reparam`, seed 42, **300 epochs** (vs the 110-epoch fixed-time budget), evaluated on interp_test / ood_test / benchmark and recorded in `results/statistics.json` + `results/tables/`. Purpose: demonstrate whether the DeepONet advantage is fundamental (radial EDL basis vs. global spectral modes) rather than an epoch-count artifact. Re-export `fno_reparam_seed42.onnx` only if the extended run improves test accuracy.

## D-033 · 2026-10-18 · Primary architecture locked (user-directed)
Lock **Multi-Scale Radial PI-DeepONet with charge-balanced reparameterization (`deeponet_reparam`)** as the **primary manuscript surrogate** (≈38× smaller, ≈14× faster ONNX inference at 0.072 ms, superior EDL resolution via exponential radial features), with the **Exact-Boundary FNO-2D (`fno_reparam`)** presented as the spectral-operator benchmark. Rationale anchored in Stage-4 evidence: DeepONet-reparam matches/beats the fixed-state DA-PINN on benchmark #0 (ψ 1.48e-4 vs 3.82e-4; I 5.59e-4 vs 1.02e-3; ρ 9.4e-2 vs 0.145) while being far cheaper.

## D-032 · 2026-10-18 · Stage 4 completion state
Stage 4 declared **COMPLETE**. Delivered the full 18-run multi-seed suite, H1–H4 evaluation, ONNX export of the best seed-42 reparam models, canonical statistics and 5 tables. Findings: H1 (interp) and H2 (OOD) supported with DeepONet-reparam strongest; H3 charge-balanced reparameterization strongly supported (ρ rel-L2 ≈5× lower than direct concentration); H4 amortization break-even ≈460–840 SG/FV solves. Limitations: FNO under-trained relative to DeepONet at the fixed M1 budget; FNO OOD PDE residuals large.

## D-031 · 2026-10-18 · DeepONet-reparam beats the fixed-state DA-PINN on its own benchmark
On the interp #0 benchmark state (50 mV, 100 mol/m³, −0.01, 5 nm) the 3-seed DeepONet-reparam attains ψ rel-L2 = 1.48e-4, current rel-L2 = 5.59e-4, ρ rel-L2 = 9.36e-2, and mean I = 196.120 pA (ref 196.137225) — all *better* than the fixed-state DA-PINN's reported ψ 3.82e-4, I 1.02e-3, ρ 0.145, i.e. a single parameter-conditioned operator matches/exceeds a network specialized to that single state. Recorded in `results/tables/table2_benchmark_state_vs_dapinn.csv`.

## D-030 · 2026-10-18 · Exact reparam targets: s = 2√(c₊c₋) (algebraic identity)
For c_±>0, with ρ=c₊−c₋ and s=2√(c₊c₋): √(s²+ρ²)=√(4c₊c₋+(c₊−c₋)²)=c₊+c₋, hence ½(√(s²+ρ²)±ρ)=c_± exactly. Implemented `salt_mean` / `reconstruct_concentrations` in `boundary_ansatz.py` and used s_target=2√(c₊c₋) in reparam training. Machine-checked identity error over all 240 states = **1.11e-16** (float64) → zero algebraic mismatch at zero loss.

## D-029 · 2026-10-18 · Stage 4 experiment design (18 runs, M1-bounded)
Seeds {42,43,44} (prior DA-PINN's 3-seed protocol). Core matrix 4 model×mode combos with physics losses (12 runs) + physics ablation `data_plus_physics` vs `data_only` on the primary reparam models (6 extra runs). FNO: width 24, modes (8,12), 4 layers, 110 epochs, AdamW lr 2e-3 + cosine, batch 36, physics weight 1e-3. DeepONet: p=64, 500 epochs. Validation subset = every 6th train state (24 states); `interp_test`/`ood_test` never touched in training or selection. Per-state scale-aware target normalization. Total wall ≈ 21 min on M1 CPU.

## D-028 · 2026-10-18 · Stage 4 evaluation protocol
Global relative L2 (volume-weighted with annular quadrature) is the headline H1/H2/H3 metric; per-state rel-L2 (with near-zero-reference masking) also stored. Physical consistency reported via wall potential-derivative error, zero wall-flux residual, and interior FV-Poisson / SG-continuity residuals. H4 = single/batch-48 latency (PyTorch f32 + onnxruntime) + amortization break-even N* vs SG/FV (median 0.2013 s, mean 0.2391 s, prior Xeon 0.44 s) and vs DA-PINN per-state retrain (≈500 s). All in `results/statistics.json` + `results/tables/`.

## D-027 · 2026-10-18 · Stage 3 completion state
Stage 3 declared **COMPLETE_WITH_LIMITATIONS**. Delivered: 240-state 4D parametric SG/FV dataset + manifest, the `scripts/operators/` PyTorch package (exact-boundary ansatz, parameter-conditioned FNO-2D and PI-DeepONet, PNP loss/metrics), and a passing smoke-verification suite. No large-scale training performed (Stage 3 is scaffolding only). Limitations disclosed (smoke-training only on 16 states / 15 epochs; operator accuracy not yet quantified).

## D-026 · 2026-10-18 · ONNX-safe real-valued DFT spectral convolution
Implemented the FNO-2D spectral layers with explicit precomputed cosine/sine basis matrices and real-valued matmuls (no `torch.fft`, no Complex64). This is mathematically identical to a truncated 2D FFT for the retained modes and exports cleanly to ONNX (opset 17, legacy exporter) with max output mismatch ~4.8e-7 (< 1e-5) in onnxruntime — required for the client-side `onnxruntime-web` simulator (D-016).

## D-025 · 2026-10-18 · Verified 240-state parametric SG/FV dataset
Generated `data/processed/pnp_parametric_240.npz` (4.02 MB, SHA-256 `3cf16cf7…6cab`) over mu=(V_app, C0, q_w, R), seed 20261018, splits train/interp_test/ood_test = 144/48/48 (ood = 24 charge-salt + 24 geometry-voltage). All 240 states converged **directly** (0 needed continuation) with residual < 1e-9 (max 9.94e-10), strictly positive concentrations, in 59 s wall. interp #0 is the v0.007 benchmark state (mean I = 196.137225 pA) for direct comparison against the prior fixed-state DA-PINN.

## D-024 · 2026-10-18 · Operator package scope and dual-mode conditioning
`scripts/operators/` implements two conditioning modes — **`reparam`** (charge-balanced: rho=z(1−z)G_rho, s=2exp(clip(z(1−z)G_s)), c_±=½(√(s²+ρ²)±ρ)) and **`direct_c`** (ablation: c_±=exp(clip(z(1−z)G_±)), ρ=c_+−c_−) — to test Hypothesis H3 (net-charge amplification). Both enforce exact reservoir Dirichlet (ψ, c at z∈{0,1}) and strict positivity by construction. Physics/metrics replicate the reference FV/SG discretization (continuity residual on reference solution ~5e-12), so operator predictions are scored with the same physical metrics as the solver.

## D-023 · 2026-10-18 · Dual-compatible CPC/JCP manuscript strategy
Format the manuscript and code archive to be **dual-compatible with CPC (both CPiP and Computational Physics Papers tracks) and JCP**: include the standard CPC "Program Summary" metadata block immediately after the Abstract (satisfying CPiP) while retaining full mathematical derivation, ablation rigor, and JCP-style reproducibility/data-availability content. Resolves the open CPC-track question from D-020 without needing a final track choice at drafting time. Recorded in `instructions/USER_RULES.md`.

## D-022 · 2026-10-18 · GitHub token filename confirmed
User confirmed the GitHub token path is `apis/GITHUB_fine_grained 2.txt` (the earlier "filename mismatch" open item is closed). Only presence + byte length are ever reported; `AUTO_PRIVATE_GITHUB_PUSH` stays `false` until Stage 7.

## D-021 · 2026-10-18 · Stage 2 completion state
Stage 2 declared **COMPLETE_WITH_LIMITATIONS**. Delivered: venv + deps, immutable baseline copy + SHA-256 manifest, staged working solver, 9/9 unit tests, M1 baseline reproduction + timing, verified 60-entry literature DB, and `reports/JOURNAL_REQUIREMENTS.md`. No operator training or experimentation performed. Limitations disclosed (controller session unmetered; disk headroom).

## D-020 · 2026-10-18 · Journal targets: CPC primary, JCP secondary
Confirmed **CPC (Computer Physics Communications)** as primary target and **JCP** as secondary, per user `TARGET_JOURNAL=JCP/CPC`. CPC Track selection (Computational Physics Papers vs Computer Programs in Physics) deferred to manuscript drafting; CPiP would require a Program Summary block, an approved open-source license (MIT/Apache-2.0 planned), and a CPC Program Library archive of source + test-run I/O. Details captured in `reports/JOURNAL_REQUIREMENTS.md`.

## D-019 · 2026-10-18 · Verified literature database construction
Built `manuscript/literature_db.json` from OpenAlex + NCBI E-utilities + Crossref + the prior manuscript. Candidate pool 172 → **60 selected**: 34 prior DA-PINN refs + 26 topical (fno_pino 8, deeponet 6, constraints_conditioning 6, nanopore_edl_design 6). **All 60 Crossref-verified** (DOI-level). Micropublication-value: used free read-only APIs only, $0.00 spend. Also emitted `manuscript/references.bib`.

## D-018 · 2026-10-18 · Baseline reproduced and timing re-measured on M1
Reproduced the v0.007 SG/FV reference from the staged solver on this Apple M1 (float64): converged in 4 Gummel iterations, residual 2.42e-10 (< 1e-9 target), mean axial current 196.1372247 pA (matches 196.137225), relative axial-current range 5.67e-14. Timing: median **0.2013 s**, IQR 0.00298 s (n=10 after warmup) vs the prior Windows/Xeon ≈0.44 s → M1 ≈2.2× faster. 9/9 unit tests pass. Record → `results/stage2_sg_baseline_verification.json`. This establishes the reference solve cost used for the operator amortization/break-even analysis.

## D-017 · 2026-10-18 · Credential paths locked
OpenAI key path locked to `apis/chatgpt.txt`; GitHub token path locked to `apis/GITHUB_fine_grained 2.txt`. `AUTO_PRIVATE_GITHUB_PUSH` remains `false` until Stage 7. Credential values still never printed; presence + byte length only.

## D-016 · 2026-10-18 · UI framework = Gradio + client-side ONNX/JS
Adopted **Gradio** for the local Python interactive tool and a **pure client-side HTML/JS simulator using `onnxruntime-web`** for zero-cost static hosting (GitHub Pages). No server-side inference for the public simulator.

## D-015 · 2026-10-18 · Baseline solver copied for provenance AND working copy
COPY the verified v0.007 package (`src/pnp_da_pinn/`, `configs/`, `tests/`, baseline reference outputs) from the sibling archive into `download/baseline_v0007/` as **immutable raw provenance**, with SHA-256 hashes recorded in `download/DATA_MANIFEST.md`. Copy the working solver modules into `scripts/sg_baseline/` for project use. (Resolves D-004’s copy-vs-reference question in favour of copy.)

## D-014 · 2026-10-18 · Disk-conserving venv strategy
`python3 -m venv --system-site-packages .venv` so the system-installed `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `requests` are reused with no duplicate multi-GB wheels. Inside `.venv`, install ONLY the missing lightweight packages: `matplotlib`, `python-docx`, `onnx`, `onnxruntime`, `gradio`. Rationale: only ~16 GB free disk. Trade-off disclosed: the venv is coupled to the system Python 3.14 site-packages; record exact versions for reproducibility.

## D-013 · 2026-10-18 · Stage 1 completion state
Stage 1 declared **COMPLETE_WITH_LIMITATIONS**. Rationale: all Stage 1 deliverables produced; limitations disclosed (controller session unmetered; deps missing; prior code not yet copied). No Stage 2 work started, per user instruction to stop after `REPORT_FOR_CHATGPT.md`.

## D-012 · 2026-10-18 · Stop after handoff
Per explicit user instruction, execution stops immediately after generating `reports/REPORT_FOR_CHATGPT.md`. No training, no paid worker loops.

## D-011 · 2026-10-18 · Live smoke test policy and the released 404
Ran at most one 16-token paid call per provider, preceded by a free `/models` probe. DeepSeek and OpenAI both returned HTTP 200 and were reconciled. The first OpenAI attempt returned **404** because our endpoint builder produced `/v1/v1/chat/completions` (our bug; base already contained `/v1`). A 404 on a malformed path never reached a billing endpoint and produced no completion. Decision: **release** that reservation (`malformed_url_404_never_billed`) rather than retain it, because the failure was provably pre-send/never-billed and is not a timeout/crash/lost-stream case. The endpoint builder was fixed and the corrected call was made exactly once. No other refunds were issued; timeout/crash cases still retain full reservations.

## D-010 · 2026-10-18 · Ledger honours SPEC’s conservative rules
Admission reserves the **peak + cache-miss** input rate plus full max output, times a 1.15 token-count safety margin for input. Unresolved reservations (timeout/unknown usage) keep the full reserved cost as outstanding spend. Only clean pre-send releases refund. Non-essential tasks are blocked above the 85% stop fraction and may never encroach on the $0.50 final-review reserve; the reserve is reachable only by `essential=True` calls.

## D-009 · 2026-10-18 · Model routing and pricing source
Catalog: DeepSeek `deepseek-flash` (default worker) + `deepseek-v4-pro`; OpenAI `gpt-4o-mini` (default reviewer) + `gpt-4.1-mini`. Prices recorded 2026-10-18 from official DeepSeek and OpenAI pricing pages (secondary cross-checks noted in `config/pricing.json`). Legacy `deepseek-chat`/`deepseek-reasoner` retired 2026-07-24 and are **not** admitted. Unknown/unpriced models are denied (fail-closed).

## D-008 · 2026-10-18 · Secret handling
Credential files read in-process only. No value, substring, or key-bearing URL emitted anywhere. Smoke script scrubs known secrets from error text. Reports state presence + byte length only.

## D-007 · 2026-10-18 · `apis/` is the credential folder
Legacy lowercase `apis/` exists and holds credentials; canonical `APIS/` does not. `project_config.json` records `credential_folder: "apis"` and a name map (`chatgpt.txt`→openai, `GITHUB_fine_grained 2.txt`→github). Both case variants are git-ignored.

## D-006 · 2026-10-18 · Local git repo, no remote
Initialized a local git repo for structure/`.gitignore` hygiene. No commit, no remote, no push (`AUTO_PRIVATE_GITHUB_PUSH=false`).

## D-005 · 2026-10-18 · `.gitignore` first
`.gitignore` written before any other file, excluding `APIS/`, `apis/`, `*.key`, `.env*`, `logs/`, `workers/`, `__pycache__/`, `.venv/`, `venv/`, token/secret patterns, large data, and the master prompt. Verified with `git check-ignore`.

## D-004 · 2026-10-18 · Prior study treated as baseline, not copied
The fixed-state DA-PINN manuscript and v0.007 reproducibility package were **inspected in place** (sibling directory) and are referenced, not copied. Copy vs reference to be decided in Stage 2.

## D-003 · 2026-10-18 · Science framing
Adopted the upgrade from fixed-state DA-PINN to a parameter-conditioned PIO (Exact-Boundary FNO + PI-DeepONet) over $\boldsymbol{\mu}=(V_{\text{app}},C_0,q_w,R)$, with charge-balanced reparameterization as the central conditioning innovation. Full statement in `PROJECT_BRIEF.md`.

## D-002 · 2026-10-18 · Governing spec adoption
`MASTER_PROMPT_ENHANCED.txt` (v2.0) adopted. First read failed because Google Drive had not finished syncing the file; it appeared moments later and was read in full (703 lines).

## D-001 · 2026-10-18 · Config merge
Section 1 defaults retained verbatim; user overrides applied: `PROJECT_NAME=nanopore_pino_surrogate`, `TARGET_JOURNAL=JCP/CPC`, `DATA_MODE=local_only`, GitHub/public release disabled, $5.00 budget with $1.00 OpenAI subbudget and $0.50 reserve.
