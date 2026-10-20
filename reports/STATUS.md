# STATUS

Project: `nanopore_pino_surrogate`
Updated: 2026-10-20 (Stage 11 — four-way constraint benchmark, exact SG/FV Fisher validation, physics-weight sweep; first-appearance renumbering; Figures 3/6 updated; 28/28 audit; v1.2.0 release)
Completion state: **COMPLETE** (all Stage 1–11 deliverables produced; review is AI-assisted, not expert peer review; DOCX page-rendered via LibreOffice and validated structurally; Zenodo DOI deferred to acceptance)

## Phase
Stage 1 — Preflight, Capability & Accounting Bootstrap, Project Brief. **Done.**
Stage 2 — Environment, baseline reproduction, literature/journal groundwork. **Done.**
Stage 3 — 4D parametric dataset, operator scaffolding, smoke verification. **Done.**
Stage 4 — Multi-seed operator training, H1–H4 evaluation, ONNX export, canonical stats. **Done.**
Stage 5 — Presentation deliverables: 300-epoch FNO check, figures 1–6, interactive simulators. **Done.**
Stage 5B — Harden both apps (live HTTP verification) + eliminate all figure text-overflow. **Done.**
Stage 6A — Figure 1 restructure, inverse-recovery demo, claim map, manuscript + supplement drafts. **Done.**
Stage 6B — Independent cross-provider review, manuscript fixes, DOCX export, release packages + audit. **Done.**
Stage 7 — Dual GitHub deployment, GitHub Pages live, cover letter, final handoff. **Done.**
Stage 8 — Literature verification + claim map, figure-only main text, refreshed cover letters, 22/22 audit, v1.0.1 re-release. **Done.**
Stage 9 — Native editable-equation DOCX, inline figures, number/symbol hygiene, hosted 3-tab portal at `pino.statwhy.com`, journal suitability assessment, 24/24 audit, v1.0.2 release. **Done.**
Stage 10 — Trunk ablation, 384-state data-scaling law, Fisher/multi-protocol identifiability; Figures 3/4/6 and Supplement S14–S15 added; manuscript/cover-letter reframed; 24/24 audit; v1.1.0 release. **Done.**
Stage 11 — Four-way constraint benchmark, exact SG/FV Fisher/Jacobian validation, physics-weight sweep; strict first-appearance reference and 18-table renumbering; equation callouts; Figures 3/6 updated; 28/28 audit; v1.2.0 release. **Done.**

## Progress (Stage 10)
- [x] **10.1 three experiments** — `scripts/run_stage10_upgrades.py` (with per-experiment caching `results/_stage10_cache_*`). (A) **Trunk ablation** (6 variants, matched capacity, 500 epochs, 3 seeds): the plain MLP trunk gives the best interp net-charge error (0.0914) while radial variants (0.098–0.116) reduce the wall-gradient error by ~7% (radial k=30: 4.83e-3 vs 5.20e-3) and give the best current error (multi-scale 0.274%). (B) **Scaling law** on a new 384-state solve expansion: interp net-charge error 0.3927→0.1299 for N=36→288, fitting alpha=0.552 (psi 0.535); OOD net-charge error is nearly flat (0.317→0.253, alpha=0.13). (C) **Identifiability** (15 states, 3 protocols, 0.2/1/2% noise, 30x30 grid + Nelder–Mead; 18x18 Fisher sweep): single/multi-salt I–V leave q_w unidentifiable (median err 0.77–1.00) while recovering R to <0.7%; the dual I–V+selectivity protocol recovers both below 0.3%. Fisher: kappa(F) median 7.46e8 (99.7% > 1e8), dlnI/dlnR=2.029, |dlnI/dlnq_w| <= 1.6e-3.
- [x] **10.2 figures** — Figure 3 rewritten to a 2x3 panel (A reparam, B weak-charge, C physics-loss, D cross-section, E trunk ablation normalized to MLP, F Fisher log10 kappa); Figure 4 gains Panel D (N_train scaling, log-log); Figure 6 gains Panels C/D (q_w and R recovery). `make_all_figures.py` QA updated; `verify_figure_layout.py` reports **0 issues** for all six figures.
- [x] **10.3 manuscript + supplement** — Abstract cut to 244 words; Highlights item 5 replaced; CPC Program Summary running-time extended; §1 rewritten ("several fronts"); §2.4 radial-trunk design note; §2.6 Stage-10 protocol/Fisher paragraph; §3.3 panel fixes + E/F; §3.4 Fig 4D; §3.9 multi-protocol result; Discussion (two scope-bounding results) and Conclusion updated; Data availability notes the 384-state expansion. Supplement header 13→**15** tables, new **Section S7b** (S14 trunk ablation + scaling, S15 identifiability), crosswalk C01–C33, renumbered reproducibility commands. Cover letters add contributions 4–6. `claim_evidence.csv` extended to **C29–C33** (all supported).
- [x] **10.4 rebuild + audit + release** — DOCX rebuilt (main: 155 paragraphs, 0 tables, 6 figures, 233 `m:oMath`; supplement: 16 tables, 0 images); packages rebuilt (public **145 files / 30.2 MB**, private **822 files / 88.5 MB**); **24/24 audit PASS**; portal updated (new "Scope and validation" block) and redeployed to **https://pino.statwhy.com** (HTTP 200, `/api/solve_sg` = 196.137225 pA); both repos force-pushed and tagged **v1.1.0** (`results/stage10_release.json`).

## Progress (Stage 11)
- [x] **11.1 three reviewer-driven experiments** — `scripts/run_stage11_reviewer_upgrades.py` (cached in `results/_stage11_cache_*`). (P2) **Four-way constraint benchmark** on the identical PI-DeepONet backbone, 3 seeds: reparam 0.2066±0.0154 and linear projection 0.2008±0.0206 (interp net-charge) both far below soft penalty 0.4114±0.1871 and direct 0.9334±0.0082; reparam keeps the charge identity exact (1.1e-16) vs 4.2e-8 for the projection. (P3) **Exact SG/FV Fisher/Jacobian validation** (15 states, float64 centered differences, step 1e-3): dlnI/dlnR = 2.00003 (exact) vs 1.99997 (operator), median relative difference 1.5e-4; median |dlnI/dlnq_w| 2.1e-5 vs 1.2e-4; kappa(F) single 4.9e15 / 8.2e8, multi 8.7e9 / 4.5e7, dual 1.8e5 / 1.8e5. (P4) **Physics-weight sweep**: lambda=1e-3 is the Pareto point (rho 0.2125, wall 5.55e-3, Poisson 0.472).
- [x] **11.1 figures** — Figure 3 panels A/B now the four constraint schemes and C/D re-purposed (physics ablation, lambda sweep); Figure 6 panels C/D now the exact-vs-operator Jacobian and conditioning agreement. `verify_figure_layout.py` reports **0 issues**.
- [x] **11.2 numbering** — `scripts/enforce_sequential_numbering_and_inline_figures.py`: references renumbered by first appearance 1..60 (Manuscript, Supplement, `literature_db.json`, L01–L60, verification JSON); supplementary tables renumbered S1–S18 by first citation (labels only; sections preserved, per user decision); equation callouts `Eq. (1)–(8)` and `Eq. (S1)–(S10)` verified sequential; inline figure placement verified in Markdown and DOCX XML.
- [x] **11.3 narrative** — CPC/CPiP framing and DA-PINN context in §3.2; experimental feasibility of the selectivity readout (V_rev/transference, ICR, plasmonic/fluorescence) in the Discussion; §3.10 compressed to one paragraph (UI details moved to Supplement S8); Algorithm 1 plus pinned runtime/dependency versions added to Supplement S2; cover letters gain contribution 7.
- [x] **11.4 release** — audit extended to **28 checks** and passes **28/28**; DOCX rebuilt (main 158 paragraphs, 0 tables, 6 figures, 276 `m:oMath`; supplement 110 paragraphs, 19 table grids, 277 `m:oMath`); cover letters regenerated; packages rebuilt (public **152 files / 30.6 MB**, private **834 files / 89.2 MB**); portal scope block updated and redeployed; both repos tagged **v1.2.0** (`results/stage11_release.json`); project-root commit made.

## Stage 11 key results
```
Constraint schemes (interp rho) : reparam 0.2066 ; linear proj 0.2008 ; soft 0.4114 ; direct 0.9334
Identity consistency            : reparam exact (1.1e-16) ; projection 4.2e-8 ; same accuracy
Physics weight                  : lambda=1e-3 Pareto (rho 0.2125, wall 5.55e-3, Poisson 0.472)
SG/FV vs operator dlnI/dlnR     : 2.00003 vs 1.99997 ; median rel. diff 1.5e-4
Fisher conditioning (single/multi/dual, exact vs op): 4.9e15/8.2e8 ; 8.7e9/4.5e7 ; 1.8e5/1.8e5
Numbering                       : refs 1..60 first-appearance ; tables S1..S18 ; Eqs (1)-(8)/(S1)-(S10)
Audit                           : 28/28 PASS (zero secret hits)
Public repo                     : Javad-Alizargar/nanopore-pino-surrogate ; tag v1.2.0 ; 152 files, 30.6 MB
```

## Stage 10 key results
```
Trunk ablation (interp, 3-seed) : MLP rho 0.0914 (best) ; radial k=30 wall 4.83e-3 (7% lower) ; multi-scale I 0.274%
Scaling law (384 states)        : alpha = 0.552 (rho interp, N^-0.55) ; OOD alpha = 0.130 (nearly flat)
Identifiability (15 states)     : single/multi I-V fail for q_w (err 0.77-1.00) ; dual recovers q_w and R < 0.3%
Fisher                          : kappa(F) median 7.46e8 (99.7% > 1e8) ; dlnI/dlnR 2.029 ; |dlnI/dlnq_w| <= 1.6e-3
Figures                         : 3 (2x3, +E/F), 4 (+D), 6 (+C/D) ; layout verify 0 issues
Audit                           : 24/24 PASS (zero secret hits)
Public repo                     : Javad-Alizargar/nanopore-pino-surrogate (public) ; tag v1.1.0 ; 145 files, 30.2 MB
Private repo                    : Javad-Alizargar/nanopore-pino-surrogate-private (private) ; tag v1.1.0 ; 822 files, 88.5 MB
Portal                          : https://pino.statwhy.com/ (updated Benchmarks scope block) ; live API 196.137225 pA
```

## Progress (Stage 9)
- [x] **9.1 manuscript/a — number & symbol hygiene** — new `scripts/fix_manuscript_typography.py` (idempotent) converts bare prose scientific notation to `$a\times10^{b}$`, caret units to superscripts (`mol m^-3` → `mol $m^{-3}$`), and pulls orphan superscript bases into math (`m$^{-3}$` → `$m^{-3}$`, fixing a placeholder-box render). Both md files pass hygiene (no carets, no bare e-notation, no ASCII `<=`/`>=`).
- [x] **9.1 b — native OMML equations** — new `scripts/omml_math.py` (pandoc LaTeX→OMML with `manuscript/cache/omml_cache.json`); `scripts/export_manuscript_docx.py` rewritten to emit numbered display equations as inline `<m:oMath>` with center/right tabs (**zero equation images**), plus `prose_fix` typography (em/en dash, `≤`/`≥`, true minus).
- [x] **9.1 c — inline figures** — figure blocks moved immediately after their first-citing paragraph; trailing `## Figure captions` section removed; order verified 1→6.
- [x] **9.1 d — audit** — `scripts/audit_release_packages.py` extended to **24/24 PASS**: main DOCX (188 `m:oMath`, 0 equation images, 6 figures before References) and supplement DOCX (204 `m:oMath`, 13 tables, 0 images).
- [x] **9.2 portal** — `web_simulator/` rebuilt as a header + **3-tab** portal (Operator / Verified benchmarks / Live solver); `app.js` uses same-origin `/api/solve_sg` when hosted and a localhost fallback offline; `scripts/test_web_simulator_dom.js` passes **33/33**; `scripts/build_web_bundle.py` regenerated `embedded_bundle.js` (5,662,351 B).
- [x] **9.3 deployment** — `scripts/deploy_pino_statwhy.sh` deployed the portal to **https://pino.statwhy.com** in an isolated layout (static `/var/www/pino.statwhy.com`, systemd `pino-statwhy.service` on `127.0.0.1:18765`, isolated nginx vhost proxying `/api/`, Let's Encrypt TLS). Verified: portal 200, HTTP→HTTPS 301, `/api/solve_sg` = **196.137225 pA**, other sites untouched. Record in `results/stage9_deployment.json`.
- [x] **9.4 assessment + release** — `reports/JOURNAL_SUITABILITY_ASSESSMENT.md`; availability statements point to `https://pino.statwhy.com/`; packages rebuilt (public **138 files / 22.5 MB**, private **810 files / 80.6 MB**); **24/24 audit**; both repos redeployed as **v1.0.2**.

## Stage 9 key results
```
Equations (DOCX)          : native OMML only ; main 188 m:oMath / 0 images ; supplement 204 m:oMath / 13 tables / 0 images
Figures (DOCX)            : 6 inline, placed after first-citing paragraph, before References
Portal                    : https://pino.statwhy.com/ (Operator / Benchmarks / Live solver tabs)
Live SG/FV API            : POST /api/solve_sg -> 196.137225 pA at (50 mV,100 mol m-3,-0.01,5 nm)
DOM test harness          : 33/33 PASS
Audit                     : 24/24 PASS (zero secret hits)
Journal assessment        : reports/JOURNAL_SUITABILITY_ASSESSMENT.md (primary CPC, alternative JCP)
Public repo               : Javad-Alizargar/nanopore-pino-surrogate (public) ; tag v1.0.2 ; 138 files, 22.5 MB
Private repo              : Javad-Alizargar/nanopore-pino-surrogate-private (private) ; tag v1.0.2 ; 810 files, 80.6 MB
GitHub Pages (mirror)     : https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/
```

## Progress (Stage 8)
- [x] **8.1 API-backed literature verification** — `scripts/verify_literature_claims_api.py` fetched canonical metadata/abstracts for all **60** references (OpenAlex, NCBI, Crossref, Semantic Scholar, Europe PMC; cached in `literature/cache/verified_abstracts_60.json`), audited every citing sentence with a budgeted `gpt-4.1-mini` pass, and wrote `results/literature_claim_verification.json`. Basis: **48 abstract-verified, 12 metadata-only**. The LLM flagged 14 sentences; each was adjudicated (cluster-citation strictness, plus the canonical Scharfetter–Gummel 1969 paper `[33]`), and the genuine wording issues found were fixed first (the FNO sentence now cites the physics-informed DeepONet paper for the physics-informed formulation and `[37,40]` for the spectral basis). Final artifact: **0 flagged**, raw verdict plus adjudication note retained per reference.
- [x] **8.1 claim map** — `scripts/build_claim_evidence.py` rebuilt `manuscript/claim_evidence.csv` to a 10-column schema holding **C01–C28** (quantitative, relocated to `Supp. Table S1--S6` and renumbered sections) plus **L01–L60** (one per citation, with basis and excerpt); all **88** rows `supported`.
- [x] **8.2 figure-only main text** — removed all six main-text tables (→ Supplementary Tables S1–S6), relabeled the seven existing supplement tables to S7–S13 with a new S0 section and rebuilt S13 crosswalk, cited every supplement table from the main text, added Figure 3/6 panel callouts, moved the Figure 5 paragraph to a new §3.5 with subsequent renumbering, converted the six-item contribution list to prose, and corrected the Figure 2 caption (index 144). Figure first-citation order is now exactly 1→6.
- [x] **8.3 cover letters** — `scripts/export_cover_letter.py` now emits `Cover_Letter_CPC`, `Cover_Letter_JCP`, and `Cover_Letter_CPC_JCP` (`.md` + `.docx`), each journal-tailored, no LaTeX leakage.
- [x] **8.4 rebuild + audit + re-release** — DOCX rebuilt (main: 0 tables, 6 figures; supplement: 13 tables); `scripts/audit_release_packages.py` extended to **22/22 PASS** (new Checks 21–22); packages rebuilt (public **138 files / 22.5 MB**, private **803 files / 80.6 MB**) and redeployed as **v1.0.1** (public and private both tagged `v1.0.1`; exact commit SHAs in `results/stage8_deployment.json`); Pages `built`, site and simulator HTTP 200.
- [x] **8.5 handoff** — this file, `reports/DECISIONS.md` (D-057–D-063), and `reports/REPORT_FOR_CHATGPT.md` updated.

## Stage 8 key results
```
Literature audit          : 60/60 references ; 48 abstract-verified, 12 metadata-only ; 0 flagged (14 adjudicated)
Claim map                 : manuscript/claim_evidence.csv ; C01-C28 + L01-L60 ; 88 rows, all supported
Main text display items   : 6 figures (panels only) ; 0 tables ; 0 lists ; figure order 1..6
Supplement display items  : 13 tables S1-S13, all cited from the main text
Audit                     : 22/22 PASS (zero secret hits)
Cover letters             : CPC, JCP, CPC_JCP (.md + .docx)
Public repo               : Javad-Alizargar/nanopore-pino-surrogate (public) ; tag v1.0.1 ; 138 files, 22.5 MB
Private repo              : Javad-Alizargar/nanopore-pino-surrogate-private (private) ; tag v1.0.1 ; 803 files, 80.6 MB
GitHub Pages              : https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/ (built; HTTP 200)
Token handling            : GH_TOKEN env + transient credential helper only; clean https remotes; no token in .git/config
```

## Cost
```
Ledger before Stage 6B   : $0.01989345 total ($4.98010655 remaining)
Stage 6B committed       : ~$0.0993 total ledger after review (~$0.079 task spend; ~$0.024 wasted on empty DeepSeek reasoning)
Stage 7 committed        : $0.00 (deployment, Pages, cover letter, reports and commits are local/API only)
Stage 8 committed        : $0.09634 (API-backed literature verification; all other Stage 8 work is local)
Stage 9 committed       : $0.00 (typography, OMML export, portal, deployment, audit and release are all local/API-free)
Stage 10 committed      : $0.00 (trunk ablation, scaling law, identifiability, figures, reframe, audit and v1.1.0 release are all local/API-free)
Remaining                : ~$4.80 of the $5.00 cap
```

## Verified results (unchanged, canonical)
```
H1 interp_test (3-seed mean rel-L2):
  deeponet_reparam  psi 0.0008914  rho 0.08248  I 0.001093  |dI| 0.08031 pA
  fno_reparam       psi 0.005495   rho 0.09310  I 0.008511  |dI| 0.6746 pA
H2 ood_test:
  deeponet_reparam  psi 0.02024  rho 0.3678  I 0.001363  |dI| 0.1077 pA
  fno_reparam       psi 0.02079  rho 0.2241  I 0.009123  |dI| 0.5338 pA
H3 reparameterization rho: DeepONet 8.63x ; FNO 10.29x
H4 ONNX batch-1 (M1): DeepONet 0.072 ms ; FNO 1.04 ms
H4 break-even vs SG/FV (0.2013 s): DeepONet ~546 ; FNO ~838 ; vs prior Xeon 0.44 s: 250 / 382
Benchmark state #0 : reference I 196.137 pA ; live /api/solve_sg 196.137 pA
Inverse demo       : R median rel error 0.004574 ; q_w not identifiable (median abs error 0.0526)
```

## Open limitations
- Independent review was **AI-assisted** (two providers), not expert peer review.
- The DOCX is native OMML and was additionally page-rendered via LibreOffice for visual confirmation; a Word-specific proof was not available in this environment.
- q_w identifiability is a physical limit of conductance-only inversion (confirmed by the Stage-10 Fisher analysis and single/multi-salt protocols); the manuscript shows an independent near-wall selectivity observable resolves it.
- A permanent **Zenodo DOI** has not been minted; deposit is deferred to acceptance.
- The package builder regenerates each package tree, so the archive history was force-pushed for each release (v1.0.0–v1.1.0); the public and private tags through `v1.1.0` are stable.
- The multi-scale radial trunk is retained as the released default (favorable wall-gradient and current behaviour), but its advantage over a plain MLP trunk is modest and is reported honestly as an ablation rather than as a requirement.
- Literature claim support was adjudicated with an API-backed abstract audit plus human-style adjudication of flag clusters; 12 references are metadata-only (no online abstract) and are recorded as such.
