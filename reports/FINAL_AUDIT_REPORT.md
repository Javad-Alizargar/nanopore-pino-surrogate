# Final audit report

Generated: 2026-10-20T00:03:04Z

Audited packages:
- Public: 152 files, 30.6 MB
- Private: 834 files, 89.2 MB

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
| Six figures cited sequentially; no main-text data tables; 18 supplement tables cited | PASS |
| Claim-evidence map spans C01--C38 and L01--L60, all supported | PASS |
| References are numbered strictly by first appearance 1..60 | PASS |
| Supplementary tables and equations are strictly sequential | PASS |
| Figures are placed inline within 1-2 paragraphs of the first mention (Markdown + DOCX) | PASS |
| Stage-11 reviewer upgrades present (tables, JSON, statistics, script) | PASS |

## Details

- Credential files inspected: 12 (values never printed).
- Public package secret-value hits: []
- Private package secret-value hits: []
- Tracked/unignored files scanned: 195; hits: []
- Public secret-name hits: []
- Private secret-name hits: []
- Abstract: 244 words; highlights: 5 (max 80 chars).
- References: 60, cited: 60, uncited: [], missing: [], with DOI: 60.
- Claims: 98 (38 quantitative + 60 literature), statuses: ['supported'].
- Display items: figure first-citation order [1, 2, 3, 4, 5, 6]; main-text data tables 0; supplement tables 18; uncited supplement tables [].
- Sequential numbering: references [1, 2, 3, 4, 5]... (n=60); supplementary tables [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]; main equations [1, 2, 3, 4, 5, 6, 7, 8]; supplement equations [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
- Stage-11 upgrades: {"tables": true, "json_keys": {"fisher_sgfv_validation": true, "constraint_schemes_benchmark": true, "physics_weight_sweep": true}, "statistics_keys": true, "script": true}.
- DOCX: {"Manuscript_PINO_clean.docx": {"exists": true, "tables": 0, "omath": 266, "omathpara": 0, "media": 6, "equation_pngs": [], "n_drawings": 6, "figures_before_references": true, "raw_dollar": false, "latex_commands": false, "equation_or_latex_leak": false, "refs_60": true}, "Supplementary_Materials_PINO.docx": {"exists": true, "tables": 19, "omath": 331, "omathpara": 0, "media": 0, "equation_pngs": [], "n_drawings": 0, "figures_before_references": true, "raw_dollar": false, "latex_commands": false, "equation_or_latex_leak": false, "refs_60": false}}.

## Scope and limitations

- The secret scan checks exact credential values, common key prefixes, and
  secret file names. It does not prove the absence of unknown secrets.
- Citation resolution checks numbering and DOI presence, not whether each
  source supports the associated claim; claim support is tracked separately
  in `manuscript/claim_evidence.csv`.
- The DOCX was built from the canonical markdown; rendered-page inspection
  is limited to structural checks (tables, embedded figures, equation images,
  and absence of LaTeX leakage).
- No DOI is invented: the data-availability statement references the GitHub
  repository and states that a Zenodo deposit will be created at acceptance.
