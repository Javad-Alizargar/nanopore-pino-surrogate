# STATUS

Project: `nanopore_pino_surrogate`
Updated: 2026-10-19 (Stage 6B — independent review, manuscript fixes, DOCX export, release packages)
Completion state: **COMPLETE_WITH_LIMITATIONS** (all Stage 6B deliverables produced; review is AI-assisted, not expert peer review; DOCX validated structurally, not page-rendered)

## Phase
Stage 1 — Preflight, Capability & Accounting Bootstrap, Project Brief. **Done.**
Stage 2 — Environment, baseline reproduction, literature/journal groundwork. **Done.**
Stage 3 — 4D parametric dataset, operator scaffolding, smoke verification. **Done.**
Stage 4 — Multi-seed operator training, H1–H4 evaluation, ONNX export, canonical stats. **Done.**
Stage 5 — Presentation deliverables: 300-epoch FNO check, figures 1–6, interactive simulators. **Done.**
Stage 5B — Harden both apps (live HTTP verification) + eliminate all figure text-overflow. **Done.**
Stage 6A — Figure 1 restructure, inverse-recovery demo, claim map, manuscript + supplement drafts. **Done.**
Stage 6B — Independent cross-provider review, manuscript fixes, DOCX export, release packages + audit. **Done.**
Stage 7 — GitHub push / release on explicit command. **Not started** (`AUTO_PRIVATE_GITHUB_PUSH=false`).

## Progress (Stage 6B)
- [x] **6B.1 Independent review** — `scripts/run_independent_review.py` (OpenAI `gpt-4.1-mini` + DeepSeek `deepseek-v4-pro`, thinking disabled) → `results/independent_review.json`. Both recommend **minor revision**; 11 + 12 issues. Task spend ≈ **$0.079** of the $0.15 allowance.
- [x] **6B.2 Review fixes** — every issue adjudicated against code/results in `reports/REVIEW_RESPONSE.md` and applied to `manuscript/Manuscript.md` + `manuscript/Supplementary_Materials.md`:
  - critical: reservoir limit is `s=2·exp(clip(z(1−z)G_s,−15,15))`, `ρ=z(1−z)G_ρ` ⇒ `s→2`, `c±→C0` **exactly** (fixed in §2.3 and S3);
  - added Supplement **S3.1 dimensionless PNP + SG flux**, exact dimensionless data/physics loss (S2), multi-scale trunk features `κ∈{1,3,10,30}`, full CPC metadata, DA-PINN/break-even caveats, absolute q_w errors, corrected Figure 5 description, wall-condition sign reconciliation;
  - **number formatting** with `scripts/format_manuscript_numbers.py` (3–4 sig figs); exact values archived.
- [x] **6B.2 Placeholders resolved** — MIT license; software name `nanopore-pino-surrogate`; repo `https://github.com/Javad-Alizargar/nanopore-pino-surrogate`; simulator `https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/`; no pending DOI marker (Zenodo deposit deferred to acceptance).
- [x] **6B.2 DOCX export** — `scripts/export_manuscript_docx.py` → `manuscript/Manuscript_PINO_clean.docx` (6 tables, 6 embedded figures, 8 equation images) and `manuscript/Supplementary_Materials_PINO.docx` (7 tables, 10 equation images). No `$$` or LaTeX leakage.
- [x] **6B.3 Reproducibility packages** — `scripts/build_reproducibility_packages.py`: `reproducibility/public_repo/` (136 files, 22.5 MB) and `reproducibility/github/` (791 files, 80.2 MB), each with SHA-256 `PACKAGE_MANIFEST.json`.
- [x] **6B.3 Final audit** — `scripts/audit_release_packages.py` → `reports/FINAL_AUDIT_REPORT.md`: **20/20 PASS** (credentials, secret names, key patterns, abstract/highlights, citations, DOIs, placeholders, claim map, DOCX structure, package contents).

## Stage 6B key results
```
Independent review        : OpenAI gpt-4.1-mini (11 issues, minor revision) + DeepSeek v4-pro (12 issues, minor revision)
Task review spend         : ~$0.079 / $0.15 allowance (project ledger ~$0.099 committed after review)
Critical correction       : s = 2 exp[clip(z(1-z)G_s,-15,15)] ; s->2, rho->0, c±->C0 exactly
Number formatting         : 247 values rounded to 3-4 sig figs; exact values archived
DOCX                      : Manuscript 6 tables / 6 figures ; Supplement 7 tables ; 0 LaTeX leaks
Public package            : 136 files, 22.5 MB (strict allowlist)
Private package           : 791 files, 80.2 MB (full state; credentials + .venv excluded)
Final audit               : 20/20 PASS ; no secrets, no placeholders, refs 1-60 all cited with DOIs
GitHub push               : NOT performed (AUTO_PRIVATE_GITHUB_PUSH=false)
```

## Cost
```
Ledger before Stage 6B   : $0.01989345 total ($4.98010655 remaining)
Stage 6B committed       : ~$0.0993 total ledger after review (~$0.079 task spend; ~$0.024 wasted on empty DeepSeek reasoning)
Stage 6B other stages    : $0.00 (manuscript edits, DOCX, packages, audit are local)
Remaining                : ~$4.90 of the $5.00 cap
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
- The DOCX was validated structurally; no page-rendered proof was possible in this environment.
- q_w identifiability is a physical limit of conductance-only inversion, disclosed in the manuscript.
- No repository push, no release, and no DOI have been created.
