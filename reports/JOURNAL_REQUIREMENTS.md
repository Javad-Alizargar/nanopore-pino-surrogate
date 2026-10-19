# JOURNAL REQUIREMENTS — CPC (primary) and JCP (secondary)

Project: `nanopore_pino_surrogate`
Compiled: 2026-10-18
Sources (accessed 2026-10-18):
- CPC Guide for Authors: https://www.sciencedirect.com/journal/computer-physics-communications/publish/guide-for-authors (ISSN 0010-4655)
- JCP Guide for Authors: https://www.sciencedirect.com/journal/journal-of-computational-physics/publish/guide-for-authors (ISSN 0021-9991)
- Elsevier artwork & media instructions, research-data policy, GenAI policy (linked from the above)

> Note: requirements are stated as of the access date. Re-check at submission. The excerpts retrieved did not include a journal-mandated rigid IMRAD instruction; both journals follow standard research-article structure (see §4). Where a rule is not stated by the journal, it is marked "not specified".

---

## 1. Aims & scope (fit)

**CPC** — "An International Journal and Program Library for Computational Physics." Publishes:
- **Computer Programs in Physics (CPiP):** full papers describing software archived in the CPC Program Library (Mendeley Data); software must use an **approved open-source license**.
- **Computational Physics Papers (CP):** research papers on mathematical/numerical methods and algorithms, computational models (design/control/analysis of experiments), and algebraic computation. Software implementation and performance details expected; software ideally on GitHub/Zenodo/institutional repo.
- Also: New Version Announcements (NVA), special issues, feature articles.
- Emphasis: novelty + significance clearly articulated to a general audience in the introduction; papers failing this may be desk-rejected.

**JCP** — computational aspects of physical problems; encourages methods spanning math/physics/ML frontiers. When addressing problems previously covered by other approaches, a comparison must be provided. Efficacy, robustness, computational complexity, and reproducibility must be addressed. Short notes ≤4 pages (excl. title page); Letters (no abstract); reviews encouraged. No formal page limit.

**Recommendation for this project:** the parameter-conditioned PIO + verified SG/FV reference + break-even analysis fits **CPC-CP** (methodological) naturally, and could also fit **CPC-CPiP** if we archive the operator + simulator as a documented program. A JCP submission would emphasize numerical analysis, complexity, and reproducibility.

---

## 2. Article types & track-specific metadata

### CPC Computer Programs in Physics (CPiP) — required "Program Summary" immediately after the abstract
The manuscript **must** contain a Program Summary section with at least these fields:
```
PROGRAM SUMMARY / NEW VERSION PROGRAM SUMMARY (delete as appropriate)
Program title:
CPC Library link to program files: (added by Technical Editor)
Developer's repository link: (if available)
Licensing provisions (choose one): MIT / Apache-2.0 / CC BY 4.0 / CC0 1.0 /
    BSD 3-Clause / BSD 2-Clause / GPL-3.0 / GPL-2.0 / LGPL / CC BY NC 3.0 / MPL-2.0
Programming language:
Supplementary material (if any):
Journal Reference of previous version:**
Does the new version supersede the previous version?:**
Reasons for the new version:**
Summary of revisions:**
Nature of problem (approx. 50-250 words):
Solution method (approx. 50-250 words):
Additional comments incl. Restrictions and Unusual features (approx. 50-250 words):
References:
```
**CPiP program submission package** must accompany the manuscript: (i) program source code; (ii) README with file/directory structure and install/run instructions; (iii) sample input and output for at least one comprehensive test run; (iv) user manual where appropriate. Upload as a compressed archive (≤1 GB; contact Technical Editor for larger). LaTeX templates: `cpc-cpip-template.tex`, `cpc-nva-template.tex`.

Approved CPC/Mendeley open-source licenses: MIT, Apache-2.0, BSD-3-Clause, BSD-2-Clause, GPLv3, GPLv2, LGPL, MPL-2.0, CeCILL, CeCILL-B, CC BY 4.0, CC0 1.0, CC BY-NC 3.0.
→ **Implication:** choose an approved license (e.g., MIT or Apache-2.0) and include the Program Summary + test-run data if targeting CPiP.

### CPC Computational Physics Papers (CP)
No Program Summary requirement; standard research paper. Include software implementation and performance details.

### JCP
No Program Summary requirement. Standard research paper. Short-note/letter formats available.

---

## 3. Peer review (both)
Single anonymized (single-blind) review; typically ≥2 independent reviewers; editorial desk assessment first. Appeals: one per submission.

---

## 4. Structure (IMRAD expected)
Both journals publish standard research articles. Adopt IMRAD:
1. Title + Title page (see §5)
2. Abstract (≤250 words; standalone)
3. Keywords (1-7)
4. Highlights (3-5 bullets, ≤85 chars each) — encouraged (both)
5. Graphical abstract (JCP encouraged; CPC not listed) — see §7
6. Introduction (novelty + significance stated explicitly)
7. Methods / Materials and Methods (model equations, units, boundary/initial conditions, numerical schemes, solver settings, hardware, software versions)
8. Results (only executed outputs; statistics/denominators)
9. Discussion (interpretation, limitations, comparison to prior approaches)
10. Conclusions
11. Declarations: funding, competing interests, generative-AI use, data availability
12. References
13. Supplementary material

Short communication variants (JCP): note ≤4 pp (no abstract); letter (no abstract).

---

## 5. Title page (required elements)
- Concise informative title (avoid abbreviations/formulae).
- Author names (given + family), order matching submission system.
- Affiliations (where the work was carried out), full postal address incl. country, lowercase superscript letters.
- Corresponding author with email (published).
- Present/permanent address footnote if affiliation changed.

---

## 6. Abstract, keywords, highlights
- **Abstract:** ≤250 words; states purpose, procedures, main findings, conclusions; standalone; avoid references/undefined abbreviations.
- **Keywords:** 1-7; English; avoid multi-word "and/of" phrases.
- **Highlights:** 3-5 bullets, ≤85 characters each, separate file named with "highlights".

---

## 7. Figures, tables, artwork
**Figures** — supplied as **separate files**; each cited in text; numbered in order; logical names (Figure_1, …); every figure needs a caption (brief title + description, symbols defined).
| Image type | Formats | Requirement |
|---|---|---|
| Vector drawings | EPS, PDF | embed fonts / text as graphics |
| Photographs (color/gray) | TIFF, JPG, PNG | ≥300 dpi (single col ≥1063 px; full page ≥2244 px) |
| Bitmapped line drawings | TIFF, JPG, PNG | ≥1000 dpi (single col ≥3543 px; full page ≥7480 px) |
| Charts / large tables | XLS, XLSX | clearly readable |
| Annotated images | PPT, PPTX | legible; embedded images ≥300 dpi |
Do not submit low-resolution art, disproportionately small text, or multiple graphs combined into one file (except intentional comparison panels). Color accepted (online always; print at cost). Use accessible colors.

**Graphical abstract (JCP encouraged):** separate file; ≥1328 px wide; longest side horizontal; readable at 13 cm; TIFF/EPS/PDF/MS Office.

**Tables** — editable text (not images); cited in text; numbered in order; captions + notes below body; **avoid vertical rules/shading**; use sparingly, no duplication of results.

**Math** — editable text (not images); variables italic; inline vs displayed; number equations consecutively in order of reference; use `exp` for powers of e; solidus for simple fractions.

---

## 8. File formats & submission
- Editable source: `.doc`/`.docx` (Word, **single-column**) or `.tex` (LaTeX; double-column allowed only for LaTeX). **PDF is not an acceptable source file.**
- LaTeX: Elsevier els-cas-templates recommended.
- Remove strikethrough/underline unless scientifically relevant.
- Separate supplementary files; captions for each.
- Video: recommended formats; ≤150 MB/file, ≤1 GB total; provide stills; descriptive text.

---

## 9. Research data & code availability
- **JCP:** Research-data **Option C is REQUIRED** — deposit data in a repository, cite and link the dataset in the article, **or** state why data cannot be shared. Data statement required at submission.
- **CPC:** strong software-availability expectation (GitHub/Zenodo/institutional repo); CPiP requires source in the CPC Program Library with an approved license. Data statement required.
- Both: reproducibility is a core expectation. Provide environment/lock files, seeds, parameters, and a runnable reproduction path.
→ **Implication:** deposit code + reference data on Zenodo (with DOI) and/or GitHub; include an explicit Data/Code Availability statement; keep a public clean reproducibility package.

---

## 10. Ethics & declarations (both)
- Submission declaration (not published/under consideration elsewhere; approved by all authors).
- Authorship criteria; changes only pre-acceptance via the Authorship Change Request form.
- Competing interests via the Elsevier declarations tool.
- Funding statement (or the standard "no funding" sentence).
- **Declaration of generative AI use** — required if AI tools used in manuscript preparation; placed before the reference list; not counted for basic grammar/spelling/reference tools. AI may not be listed as an author. Our workflow must include a truthful AI-use statement.
- Image-manipulation and inclusive-language policies apply.

---

## 11. Consistency check before export (from master spec §14)
- Every numerical claim matches validated results/denominators.
- Citations resolve to bibliography; references exist and claims are separately supported.
- Figures/tables called out in order with accurate legends; symbols/units defined.
- Abstract/methods/results/tables/discussion agree.
- Journal limits, reporting checklist, ethics/consent, funding, conflicts, data availability, AI-use disclosures addressed.
- No secrets, invented DOIs, hidden placeholders, or fabricated outcomes.

---

## 12. Practical targets for this project (planning only — not journal mandates)
| Item | Target |
|---|---|
| Abstract | ≤250 words |
| Keywords | 5-6 |
| Highlights | 4 bullets, ≤85 chars |
| Main display items | ~6 (provisional) + supplementary |
| Figures | vector PDF + ≥300 dpi raster; separate files |
| Data/code | Zenodo DOI + GitHub; explicit availability statement |
| License (if CPiP) | MIT or Apache-2.0 |
| AI-use | declare (OpenCode/LLM-assisted workflow) |
| Structure | IMRAD |
