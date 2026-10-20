# Final audit report

Generated: 2026-10-20T06:59:35Z

Audited packages:
- Public: 165 files, 31.1 MB
- Private: 851 files, 89.9 MB

## Checklist

| Check | Result |
|---|---|
| No credential values in public package | PASS |
| No credential values in private package | PASS |
| No credential values in tracked/unignored project files | PASS |
| No secret file/dir names in public package | PASS |
| No secret file/dir names in private package | PASS |
| No generic key patterns in public package | PASS |
| No generic key patterns in private package | PASS |
| Abstract <= 250 words | PASS |
| 3-5 highlights, each <= 85 chars | PASS |
| All cited reference numbers resolve | PASS |
| Every reference is cited | PASS |
| Every reference has a DOI | PASS |
| No hidden placeholders remain | PASS |
| Dimensionless system in supplement | PASS |
| Exponential reservoir ansatz documented | PASS |
| Claim-evidence map complete | PASS |
| Main DOCX present, no LaTeX leak | PASS |
| Supplementary DOCX present, no LaTeX leak | PASS |
| Main DOCX: native OMML equations, no equation images, figures inline before References | PASS |
| Supplementary DOCX: native OMML equations, no images | PASS |
| Public package has no private logs/download/literature | PASS |
| Private package contains manuscript and reports | PASS |
| Six figures cited sequentially; no main-text data tables; 20 supplement tables cited | PASS |
| Claim-evidence map spans C01--C38 and L01--L60, all supported | PASS |
| References are numbered strictly by first appearance 1..60 | PASS |
| Supplementary tables and equations are strictly sequential | PASS |
| Figures are placed inline within 1-2 paragraphs of the first mention (Markdown + DOCX) | PASS |
| Stage-11 reviewer upgrades present (tables, JSON, statistics, script) | PASS |
| DOCX bibliography is numbered [1]--[60] with authors and years | PASS |
| Supplementary tables first-seen 1..20; no stray legend children in figures | PASS |
| Official Zenodo DOI present in manuscript, main DOCX, CPC letter, and public README | PASS |
| Zero non-citation square brackets in prose; no 28,200 typo | PASS |
| Figure 1--6 panel labels present/spaced; official CPC template and test_run pass | PASS |
| Supplementary tables physically ordered S1--S20; Word headers repeat and rows cannot split | PASS |
| Stage-16 title, algebra, code filename, and single-absolute Fisher notation | PASS |

## Details

- Credential files inspected: 12 (values never printed).
- Public package secret-value hits: []
- Private package secret-value hits: []
- Tracked/unignored files scanned: 210; hits: []
- Public secret-name hits: []
- Private secret-name hits: []
- Abstract: 248 words; highlights: 5 (max 84 chars).
- References: 60, cited: 60, uncited: [], missing: [], with DOI: 60.
- Claims: 98 (38 quantitative + 60 literature), statuses: ['supported'].
- Display items: figure first-citation order [1, 2, 3, 4, 5, 6]; main-text data tables 0; supplement tables 20; uncited supplement tables [].
- Sequential numbering: references [1, 2, 3, 4, 5]... (n=60); supplementary tables [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; main equations [1, 2, 3, 4, 5, 6, 7, 8]; supplement equations [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
- Stage-11 upgrades: {"tables": true, "json_keys": {"fisher_sgfv_validation": true, "constraint_schemes_benchmark": true, "physics_weight_sweep": true}, "statistics_keys": true, "script": true}.
- Prose bracket integrity: {"ok": true, "bad_brackets": [], "bad_count": 0, "typo_28_200": false}.
- CPC template/test run: {"panels_ok": true, "panel_detail": {"fig1_workflow_and_architecture": {"expected": "ABCD", "found": "ABCD", "set_ok": true, "all_in_bounds": true, "min_gap_px": 24.9, "ok": true}, "fig2_benchmark_and_field_reconstruction": {"expected": "ABCDEF", "found": "ABCDEF", "set_ok": true, "all_in_bounds": true, "min_gap_px": 20.4, "ok": true}, "fig3_h3_charge_reparameterization_ablation": {"expected": "ABCDEF", "found": "ABCDEF", "set_ok": true, "all_in_bounds": true, "min_gap_px": 25.4, "ok": true}, "fig4_h1_h2_interp_vs_ood_regimes": {"expected": "ABCD", "found": "ABCD", "set_ok": true, "all_in_bounds": true, "min_gap_px": 15.0, "ok": true}, "fig5_parametric_physics_sweeps": {"expected": "ABC", "found": "ABC", "set_ok": true, "all_in_bounds": true, "min_gap_px": 17.0, "ok": true}, "fig6_h4_timing_and_amortization_breakeven": {"expected": "ABCD", "found": "ABCD", "set_ok": true, "all_in_bounds": true, "min_gap_px": 18.7, "ok": true}}, "cpc_ok": true, "files_ok": true, "test_ok": true, "test_output": "CPC TEST RUN PASSED \u2014 deeponet_reparam_seed42.onnx, mu=[50.0, 100.0, -0.009999999776482582, 5.0], 5 fields, max |delta| <= 1e-05"}.
- DOCX: {"Manuscript_PINO_clean.docx": {"exists": true, "tables": 0, "omath": 337, "omathpara": 0, "media": 6, "equation_pngs": [], "n_drawings": 6, "figures_before_references": true, "raw_dollar": false, "latex_commands": false, "equation_or_latex_leak": false, "refs_60": true}, "Supplementary_Materials_PINO.docx": {"exists": true, "tables": 21, "omath": 420, "omathpara": 0, "media": 0, "equation_pngs": [], "n_drawings": 0, "figures_before_references": true, "raw_dollar": false, "latex_commands": false, "equation_or_latex_leak": false, "refs_60": false}}.

## Scope and limitations

- The secret scan checks exact credential values, common key prefixes, and
  secret file names. It does not prove the absence of unknown secrets.
- Citation resolution checks numbering and DOI presence, not whether each
  source supports the associated claim; claim support is tracked separately
  in `manuscript/claim_evidence.csv`.
- The DOCX was built from the canonical markdown; rendered-page inspection
  is limited to structural checks (tables, embedded figures, equation images,
  and absence of LaTeX leakage).
- The archived DOI is the official Zenodo record 10.5281/zenodo.23487404,
  checked in the manuscript, main DOCX, CPC cover letter, and public README.
