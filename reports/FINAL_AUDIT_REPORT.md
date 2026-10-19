# Final audit report

Generated: 2026-10-19T02:04:08Z

Audited packages:
- Public: 138 files, 22.5 MB
- Private: 790 files, 80.1 MB

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
| Main DOCX present, tables + no LaTeX leak | PASS |
| Supplementary DOCX present, no LaTeX leak | PASS |
| Public package has no private logs/download/literature | PASS |
| Private package contains manuscript and reports | PASS |

## Details

- Credential files inspected: 12 (values never printed).
- Public package secret-value hits: []
- Private package secret-value hits: []
- Tracked/unignored files scanned: 846; hits: []
- Public secret-name hits: []
- Private secret-name hits: []
- Abstract: 234 words; highlights: 5 (max 76 chars).
- References: 60, cited: 60, uncited: [], missing: [], with DOI: 60.
- Claims: 28, statuses: ['supported'].
- DOCX: {"Manuscript_PINO_clean.docx": {"exists": true, "tables": 6, "equation_or_latex_leak": false, "refs_60": true}, "Supplementary_Materials_PINO.docx": {"exists": true, "tables": 7, "equation_or_latex_leak": false, "refs_60": false}}.

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
