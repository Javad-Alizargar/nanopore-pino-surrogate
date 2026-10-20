# NEXT ACTIONS

Ordered plan. Each item is reversible unless flagged. Paid calls must pass the budget ledger.

## Done
- [x] **Stage 1** — config, `.gitignore`, tree, ledger, credential audit, brief.
- [x] **Stage 2** — venv+deps, immutable baseline copy + manifest, staged solver, M1 baseline reproduction + timing, 60-ref literature DB, journal dossier.
- [x] **Stage 3** — 240-state 4D parametric SG/FV dataset + manifest; `scripts/operators/`; smoke verification.
- [x] **Stage 4** — 18-run multi-seed suite (`scripts/train_and_evaluate_suite.py`); H1–H4 evaluation; exact reparam algebra (identity err 1.1e-16); ONNX export of best seed-42 reparam models (parity < 1e-5); `results/statistics.json` + 5 CSV tables.
- [x] **Stage 5 (presentation deliverables)** — 300-epoch FNO verification check (`scripts/fno_extended_check.py`); figures 1–6 (`scripts/figures/`) with 3-stage QA; client-side browser simulator (`web_simulator/`) + Gradio app (`scripts/app_gradio.py`) with live SG/FV comparison; verification record `results/stage5_simulator_verification.json`.
- [x] **Stage 5B (apps hardened + figure layout QA)** — dual-engine web simulator (`file://` + `http://`) with DOM harness 29/29; Gradio fixes + zero-dependency `scripts/serve_interactive.py` (8765) + `launch_apps.command`/`.sh`; **live HTTP verification** of both apps (`results/stage5b_live_server_verification.json`); figure layout verifier `scripts/figures/verify_figure_layout.py` (**0 issues**) + budgeted gpt-4o-mini vision QA ($0.0199). See `REPORT_FOR_CHATGPT_STAGE5B.md`.
- [x] **Stage 6A (figure 1 + inverse demo + manuscripts)** — Figure 1 rebuilt as 2×2 panels A–D (geometry, operating space, constraints, backbones) with exact counts 36,880/460,488; all six figures regenerated, layout verifier **0 issues**, `figure_qa.json` refreshed; `scripts/run_inverse_recovery_demo.py` (R median rel. err **0.0045737918517061785**, q_w **1.1539643805520952** → not identifiable, 0.12526570879854262 s vs 402.6 s SG/FV); `manuscript/claim_evidence.csv` (28 claims), `manuscript/Manuscript.md` (234-word abstract, 5 highlights, 60 refs), `manuscript/Supplementary_Materials.md` (S1–S10). No paid calls. See `REPORT_FOR_CHATGPT.md`.
- [x] **Stage 6B (review + export + packages)** — `scripts/run_independent_review.py` (OpenAI + DeepSeek, ~$0.079 / $0.15); 23 issues resolved in `reports/REVIEW_RESPONSE.md` (critical ansatz `s=2exp[clip(z(1−z)G_s,−15,15)]`; S3.1 dimensionless system; explicit loss; CPC metadata; caveats); `scripts/format_manuscript_numbers.py` (3–4 sig figs); `scripts/export_manuscript_docx.py` → both DOCX; `scripts/build_reproducibility_packages.py` → public 136 files/22.5 MB + private 790 files/80.2 MB; `scripts/audit_release_packages.py` → `reports/FINAL_AUDIT_REPORT.md` **20/20 PASS**. No push.
- [x] **Stage 7 (dual deployment + cover letter)** — authorized by explicit user command (`AUTO_PRIVATE_GITHUB_PUSH: true`, `AUTO_PUBLIC_RELEASE: true`): pre-push audit re-confirmed **20/20 PASS**; `scripts/deploy_github_repos.py` created and pushed `Javad-Alizargar/nanopore-pino-surrogate` (**public**, tag `v1.0.0`) and `Javad-Alizargar/nanopore-pino-surrogate-private` (**private**, verified via API); GitHub Pages enabled from `main` `/` and verified live (simulator + assets HTTP 200); `scripts/export_cover_letter.py` → `manuscript/Cover_Letter_CPC_JCP.{md,docx}`; local project-root commit made. Token passed only via `GH_TOKEN` / transient credential helper (never in `.git/config`).
- [x] **Stage 8 (literature audit + figure-only main text + v1.0.1)** — `scripts/verify_literature_claims_api.py` verified all 60 references (48 abstract-basis, 12 metadata-only; 14 flags adjudicated → 0); `scripts/build_claim_evidence.py` rebuilt the claim map (C01–C28 + L01–L60, all supported); all main-text tables moved to Supplementary Tables S1–S13 and the main text became figure-only with figure order 1→6; `scripts/export_cover_letter.py` now emits CPC/JCP/joint letters; DOCX and packages rebuilt; `scripts/audit_release_packages.py` **22/22 PASS**; both repos redeployed and tagged `v1.0.1`.
- [x] **Stage 9 (native equations, portal, hosted deploy, v1.0.2)** — `scripts/fix_manuscript_typography.py` (number/unit hygiene); `scripts/omml_math.py` + rewritten `scripts/export_manuscript_docx.py` (native OMML display equations, **zero equation images**, inline figures); `web_simulator/` upgraded to a 3-tab portal (DOM harness **33/33**); `scripts/deploy_pino_statwhy.sh` deployed it live to **https://pino.statwhy.com** (isolated nginx vhost + systemd `pino-statwhy.service` + Let's Encrypt); `reports/JOURNAL_SUITABILITY_ASSESSMENT.md`; availability statements point to the hosted portal; `scripts/audit_release_packages.py` **24/24 PASS**; both repos redeployed and tagged `v1.0.2`.
- [x] **Stage 10 (ablation, scaling law, identifiability, v1.1.0)** — `scripts/run_stage10_upgrades.py` (Experiment A trunk ablation, B 384-state scaling law, C Fisher/multi-protocol identifiability; cached); Figures 3/4/6 updated (layout verify **0 issues**); manuscript + supplement reframed (S14–S15, claim map C01–C33); cover letters extended; DOCX and packages rebuilt; **24/24 audit PASS**; portal scope block added and redeployed to `pino.statwhy.com`; both repos force-pushed and tagged `v1.1.0`.
- [x] **Stage 15 (Figure 1/panel labels, StatWhy citation integrity, CPC template + test_run, v1.3.3)** — Figure 1B restored and A–D spaced (all six figures 0 issues; minimum gap 15.0 px); 72 citation flags resolved with individualized attribution; prose square brackets restricted to citations; official CPC 11-field template; `test_run/` prints `CPC TEST RUN PASSED`; **33/33 audit PASS**; packages **165/852 files**; portal redeployed; both repos tagged `v1.3.3`.
- [x] **Stage 14 (CPiP statement, DA-PINN context, §3.3–3.7 polish, v1.3.2)** — explicit CPiP paragraph at the end of §1; Section-1 first citations monotonic 1..60; DA-PINN reference errors and fold improvements added to §3.2; §3.6 compressed and §3.7 closing paragraph tightened to one sentence; **31/31 audit PASS**; packages **161/845 files**; portal redeployed; both repos tagged `v1.3.2`.
- [x] **Stage 13 (Zenodo DOI, diffusivity proof, CPiP polish, v1.3.1)** — `scripts/run_diffusivity_symmetry_check.py` (exact SG/FV, four ratios, two salinities; new S20/§S7d); official DOI `10.5281/zenodo.23487404` in manuscript/supplement/letters/DOCX/README; §3.2–3.6 compressed, §3.9 expanded (analytic channel, Fisher, diffusivity, protocol workflow, boundary), abstract/conclusion boundary statements; Figure 1C typo fixed; strict S1–S20 first-citation order; **31/31 audit PASS**; packages **161/844 files**; portal redeployed; both repos tagged `v1.3.1`.
- [x] **Stage 12 (identifiability reframing, OOD decomposition, v1.3.0)** — `scripts/run_stage12_diagnostics.py` (spatial/parametric OOD decomposition, 45-case selectivity validation, positivity stress; cached in `results/_stage12_selectivity_cache.json`); Figure 3D legend leak fixed and Figure 4 Panel B2 added (layout verify **0 issues**); title/abstract/intro/§3.2/§3.3/§3.4/§3.7/§5 rewritten around conductance identifiability; bibliography refs 33/57 repaired; DOCX references emitted as `[1]--[60]`; supplement S7c reordered S16–S19; enforcement range 1..19; **30/30 audit PASS** (before deploy); packages **159/841 files**; cover letters retitled; portal redeployed; both repos tagged `v1.3.0`.
- [x] **Stage 11 (reviewer upgrades, numbering, v1.2.0)** — `scripts/run_stage11_reviewer_upgrades.py` (four-way constraint benchmark, exact SG/FV Fisher/Jacobian validation, physics-weight sweep; cached); Figures 3A–D and 6C–D updated (layout verify **0 issues**); `scripts/enforce_sequential_numbering_and_inline_figures.py` applied (refs 1–60, tables S1–S18, equation callouts); manuscript §3.2/§3.3/§3.9/§3.10 + Discussion refined; Supplement gains Algorithm 1, dependency pinning, S7c (S16–S18); claim map C01–C38; DOCX + cover letters rebuilt; **28/28 audit PASS** (before deploy); portal redeployed; both repos tagged `v1.2.0`.

## Next (post-Stage-15; project COMPLETE)
1. **Author review** of `manuscript/Manuscript_PINO_clean.docx`, `Supplementary_Materials_PINO.docx`, and the matching cover letter (`Cover_Letter_CPC.docx`, `Cover_Letter_JCP.docx`, or the joint `Cover_Letter_CPC_JCP.docx`) before journal submission; `reports/JOURNAL_SUITABILITY_ASSESSMENT.md` recommends **CPC (primary)** or **JCP (alternative)**.
2. **Submit** via the CPC or JCP editorial system; cite the prior study `COMPHY-D-26-00857`.
3. **Zenodo archive** is live at `10.5281/zenodo.23487404`; confirm the deposited files match tag `v1.3.3` and cite the DOI in the submission system.
4. Optional **human expert review** or a targeted follow-up AI pass on any single unresolved issue (keep the $0.50 final-review reserve).
5. Re-run the verification chain before any future release: `scripts/figures/verify_figure_layout.py` (0 issues) → `scripts/verify_simulators.py` → `scripts/verify_live_servers.py` → `scripts/verify_literature_claims_api.py` (0 flagged) → `scripts/build_reproducibility_packages.py` → `scripts/audit_release_packages.py` (31/31, **before** deploy) → `scripts/deploy_github_repos.py` (idempotent re-push) → `scripts/deploy_pino_statwhy.sh` (idempotent re-deploy of the hosted portal).

## Blocked / needs user decision
- **Compute for a stronger FNO** — M1-only CPU; the 300-epoch check (D-034) shows the DeepONet gap is architectural, so no further FNO compute is planned unless reconsidered.
- **Budget increase** — only if explicitly requested; current cap $5.00 (~$4.80 remaining).
- **Human peer review** — cannot be performed by the controller; the AI-assisted review is disclosed in the cover letter.
- **Zenodo DOI** — resolved: official record `10.5281/zenodo.23487404` is integrated and audited.

## Exact restart steps
```
cd "/Users/javadalizargar/Google drive MAC/PINN to a Neural Operator"
python3 orchestration/budget_ledger.py recover      # reconcile outstanding spend
python3 orchestration/budget_ledger.py status       # check budget
# read reports/STATUS.md, DECISIONS.md, NEXT_ACTIONS.md
# rebuild figures (optional, ~1 min): .venv/bin/python scripts/figures/make_all_figures.py
# verify figure layout (must be 0):  .venv/bin/python scripts/figures/verify_figure_layout.py
# rebuild web bundle:                .venv/bin/python scripts/build_web_bundle.py
# web DOM harness:                   node scripts/test_web_simulator_dom.js
# launch BOTH apps:                  ./launch_apps.sh   (or double-click launch_apps.command)
# re-verify live servers:            .venv/bin/python scripts/verify_live_servers.py
# stop live servers:                 .venv/bin/python scripts/verify_live_servers.py --stop
# round manuscript numbers:          .venv/bin/python scripts/format_manuscript_numbers.py
# number/symbol hygiene:             .venv/bin/python scripts/fix_manuscript_typography.py
# rebuild DOCX:                      .venv/bin/python scripts/export_manuscript_docx.py
# rebuild cover letter:              .venv/bin/python scripts/export_cover_letter.py
# rebuild claim map:                 .venv/bin/python scripts/build_claim_evidence.py
# run Stage-10 experiments:          .venv/bin/python scripts/run_stage10_upgrades.py   (--force to retrain)
# run Stage-11 upgrades:             .venv/bin/python scripts/run_stage11_reviewer_upgrades.py   (cached)
# verify literature claims:          .venv/bin/python scripts/verify_literature_claims_api.py
# rebuild release packages:          .venv/bin/python scripts/build_reproducibility_packages.py
# run Stage-12 diagnostics:          .venv/bin/python scripts/run_stage12_diagnostics.py
# final audit (must be 31/31):       .venv/bin/python scripts/audit_release_packages.py     # run BEFORE deploy
# deploy both repos (idempotent):    .venv/bin/python scripts/deploy_github_repos.py
# deploy one repo only:              .venv/bin/python scripts/deploy_github_repos.py --only public|private
# redeploy hosted portal:            bash scripts/deploy_pino_statwhy.sh   (--dry-run to stage only)
# project is COMPLETE; see reports/REPORT_FOR_CHATGPT.md for final URLs and SHAs
```
