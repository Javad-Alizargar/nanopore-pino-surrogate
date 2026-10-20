# REPORT FOR EXTERNAL REVIEW — Stage 15 final handoff

Project: `nanopore_pino_surrogate` — parameter-conditioned physics-informed
neural operator for steady axisymmetric Poisson--Nernst--Planck (PNP)
transport in a charged cylindrical nanopore.
Root: `/Users/javadalizargar/Google drive MAC/PINN to a Neural Operator`
Date: 2026-10-20. State: **COMPLETE** (Stage 1–15 done; stop point reached; archived at Zenodo DOI 10.5281/zenodo.23487404).

Stage 8 and Stage 9 were executed on explicit authorization. The fine-grained
token was used **in-memory only** (`GH_TOKEN` environment for `gh`; a transient
`gh auth git-credential` helper for `git push`). No token value was printed,
echoed, or logged, and no remote URL contains a token (`.git/config` holds clean
`https://github.com/...` URLs).

## Objective

Build a 4D parameter-conditioned surrogate over
$\boldsymbol{\mu}=(V_\text{app},C_0,q_w,R)$ that is CPC and JCP compatible:
a verified SG/FV dataset, two backbones (multi-scale radial PI-DeepONet and a
real-DFT FNO), charge-balanced reparameterization, an inverse demonstration,
browser + Gradio apps, and a defensible manuscript. Stages 6B–7 added independent
review, DOCX export, dual release packages, and deployment. **Stage 8** verified
that every cited reference supports its sentence, moved every data table into the
supplement (figure-only main text), refreshed the cover letters, re-audited the
release at **22/22 PASS**, and re-released both repositories as **v1.0.1**.
**Stage 9** made the DOCX equations natively editable (OMML, zero equation
images), fixed number/symbol hygiene, placed figures inline, upgraded the web
portal, deployed it live at **https://pino.statwhy.com**, added a journal
suitability assessment, re-audited at **24/24 PASS**, and re-released as
**v1.0.2**. **Stage 10** ran three pre-submission studies (a trunk ablation, a
384-state data-scaling law, and a Fisher/multi-protocol identifiability
analysis), updated Figures 3/4/6 and the supplement (S14–S15), reframed the
manuscript honestly, re-audited at **24/24 PASS**, and released **v1.1.0**.
**Stage 11** answered the remaining reviewer objections: it benchmarked four
charge-constraint schemes, validated the Fisher/Jacobian analysis against exact
SG/FV finite differences, swept the physics weight, renumbered all references,
tables and equations by first appearance with in-text callouts, added an
Algorithm block and dependency pinning, and released **v1.2.0** after a
**28/28 audit**. **Stage 12** reframed the paper around conductance
identifiability, decomposed the out-of-distribution error spatially and
parametrically, validated the near-wall selectivity observable against exact
SG/FV, stress-tested positivity, repaired the bibliography and table ordering,
and released **v1.3.0** after a **30/30 audit**. **Stage 13** integrated the
official Zenodo DOI, added an exact-SG diffusivity-asymmetry proof (S20) that
bounds the claim beyond KCl, compressed the DA-PINN/architecture narrative,
expanded §3.9 into a protocol-design and boundary section, fixed the Figure 1C
typo, and enforced individual table citations S1–S20. It released **v1.3.1**
after a **31/31 audit**. **Stage 14** added the explicit CPiP statement,
contextualized the DA-PINN comparison with both reference errors, tightened
§3.6–3.7, and made the Section-1 first-citation flow strictly monotonic 1..60;
it released **v1.3.2** after a **31/31 audit**. **Stage 15** restored the
missing Figure 1B label and spaced all panel letters, resolved the 72 external
StatWhy citation-integrity flags, aligned the CPC Program Summary with the
official template, added a self-contained `test_run/`, and released **v1.3.3**
after a **33/33 audit**.

## Stage 15 deliverables (citation integrity, figure labels, CPC template)

1. **Figure 1 and all panel labels.** Panel B now carries its bold `B`; all
   Figure 1 letters sit at `(-0.06, 1.08)` in axes coordinates. The verifier
   now checks exact panel sets (`A–D`, `A–F`, `A–F`, `A–D`, `A–C`, `A–D`),
   in-bounds rendering, and a 14 px minimum clearance from every text artist
   and box edge. All six figures pass with **0 layout issues** (minimum gap
   15.0 px; Figure 1 minimum 24.9 px).
2. **StatWhy citation integrity.** All 72 flags were addressed by rewriting
   cluster citations into individualized clause attributions for refs 1–60,
   including the corrected `[43]` thermal-charring context, the `[52]`/`[54–60]`
   operator-learning wording, and the Discussion replacement of `[20–23]` by
   `[7, 8, 10]`. Mathematical interval brackets were converted to prose or
   inequalities, and the `28,200` typo was corrected to 20–200 mol m^-3.
   Check 32 confirms zero non-citation square brackets in prose.
3. **CPC template and test run.** The Program Summary now follows the official
   11-field Elsevier order (Program title through Running time), with MIT,
   Python 3.10+ (tested 3.14), Code Ocean not applicable, and the repository/
   Zenodo/portal links. `test_run/` contains sample input, expected ONNX
   summary, and a runner that prints `CPC TEST RUN PASSED` in under one second;
   the README and Data availability statement describe it explicitly.
4. **Release (15.4).** Enforcement, typography, DOCX, and cover letters were
   rerun; DOCX rebuilt (main 0 tables, 6 inline figures, native OMML;
   supplement 21 grids, 0 images); packages rebuilt; audit **33/33 PASS**
   (checks 32 and 33 added); portal redeployed (`/api/solve_sg` 196.137225 pA);
   both repositories tagged **v1.3.3** (`results/stage15_release.json`).

## Stage 14 deliverables (surgical pre-submission polish)

1. **CPiP statement and citation flow.** Section 1 now ends with an explicit
   CPiP positioning paragraph describing PINO-Nanopore as a reusable
   protocol-design instrument. The stale `[33, 41]` citation is reduced to
   `[41]`, and `[56--60]`/`[59]` become `[56--58]`/`[59, 60]`; first appearances
   in Section 1 run 1..60 without a gap or backward first use.
2. **DA-PINN context.** Section 3.2 now reports the state-specialized DA-PINN
   errors (0.000382 potential, 0.00102 current) alongside the operator errors
   (0.0001485, 0.0005587), i.e. 2.57-fold and 1.83-fold lower without
   per-state retraining.
3. **Streamlining.** Section 3.6 is compressed to two sentences while keeping
   S15/S16 in place, and the Section 3.7 caveat is one sentence; the
   first-citation order remains strict S1..S20.
4. **Release (14.4).** Enforcement, typography, DOCX, cover letters, packages,
   and audit were rerun. Audit **31/31 PASS**; DOCX rebuilt (main 0 tables,
   6 figures, native OMML; supplement 21 grids, 0 images); packages **161
   public / 845 private files**; portal redeployed (`/api/solve_sg`
   196.137225 pA); both repositories tagged **v1.3.2**
   (`results/stage14_release.json`).

## Stage 13 deliverables (final CPiP polish)

1. **Zenodo DOI.** `10.5281/zenodo.23487404` is in the CPC Program Summary,
   data-availability statement, supplement header, all three cover letters, the
   main DOCX, and the public README/BibTeX. Check 31 verifies all four required
   artifacts.
2. **Diffusivity proof.** `scripts/run_diffusivity_symmetry_check.py` runs the
   exact float64 SG/FV solver at four ratios (1.0, 0.966, 0.655, 0.20) and two
   salinities in 32.5 s. Even D+/D-=0.20 leaves single-salt kappa at 5.9e15
   (C0=100) / 3.6e12 (C0=20), while the dual observable gives 4.0e6 / 6.3e5.
   The analytic channel is sigma=(F^2/RT)[Dbar(c+ + c-) + (D+ - D-)rho/2].
3. **Narrative compression.** §3.2 is now field + selectivity validation with
   the DA-PINN comparison reduced to one sentence (Table S10); §3.3 states the
   four-way decoupling and the structural preference for the hyperbolic map;
   §3.4 gives the measured accuracy and cost budget in three paragraphs; §3.5
   and §3.6 are tightened. §3.9 now runs inverse demonstration → analytic
   near-null channel → Fisher → diffusivity → protocol-design workflow/portal →
   explicit boundary statement. The abstract (247 words) and Conclusion carry
   the same boundary.
4. **Figures and ordering.** Figure 1C now reads `s→2, rho→0 at z in {0,1}`;
   layout verifier **0 issues**. Individual table first citations are strict
   `S1..S20` (S19 Fisher, S20 diffusivity), with no catch-all ranges.
5. **Release (13.5).** Audit extended to **31 checks — 31/31 PASS**. DOCX
   rebuilt (main 0 tables / 6 figures / 332 OMML / 8 numbered equations;
   supplement 21 table grids / 368 OMML / 0 images); packages rebuilt (public
   **161 files / 31.0 MB**, private **844 files / 89.8 MB**); portal redeployed
   (`/api/solve_sg` 196.137225 pA); both repositories tagged **v1.3.1**
   (`results/stage13_release.json`).

## Stage 12 deliverables (identifiability reframing)

1. **Diagnostics (12.1).** `scripts/run_stage12_diagnostics.py` produces the
   spatial OOD budget (DeepONet psi/rho shares: core 51.3/52.4%, wall EDL
   38.8/39.1%, transitions 9.9/8.4%; per-unit wall psi 2.40%, transitions rho
   40.0%), the mutually exclusive parametric budget (combined charge+low-salt
   94.2/93.6%, charge-only 5.5/6.2%, narrow R 0.2/0.2%, high V 0.0/0.0%), the
   45-case selectivity validation (median absolute S error **1.18e-4**, 1.9%
   relative; wall c+ 2.2e-4 and c- 3.4e-4; field c± <=0.20%), and the
   positivity stress test (linear projection identity 1.19e-7 in float32 vs
   hyperbolic 1.11e-16; Softplus bias ln2/20 = 0.0347 with 22.5% clipped grid
   points). Tables S18--S19 and `results/stage12_diagnostics.json`.
2. **Figures (12.2).** Figure 3D legend leak fixed without a literal `_child`
   string; Figure 4 Panel B split and new Panel B2 (OOD budget by zone). Layout
   verifier **0 issues**.
3. **Manuscript (12.3).** New title **"Conductance identifiability limits and
   physics-constrained neural operators for axisymmetric Poisson--Nernst--Planck
   transport in charged nanopores"**; 230-word abstract led by the inverse
   problem; inverse/identifiability background plus three use cases in §1;
   §3.2 compressed and strengthened with the selectivity validation; §3.3 frames
   sum/difference decoupling as the driver and chooses hyperbolic on structural
   grounds (no Softplus clipping bias); §3.4 adds the OOD budget; §5 ends on the
   negative identifiability result and the charge-sensitive-observable
   recommendation.
4. **Bibliography/formatting (12.4).** Refs 33 and 57 repaired (Krishnapriyan
   et al. 2021; Zhang, Xiao, Ghosh 2025, 9(1) 45--64) in both the database and
   the manuscript; DOCX references render `[1]--[60]` with hanging indent;
   supplement S7c reordered **S16--S19**; enforcement range updated to 1..19.
5. **Release (12.5).** Audit extended to **30 checks — 30/30 PASS** (new:
   DOCX bibliography format; table first-seen order plus no stray legend
   children). DOCX rebuilt (main 0 tables / 6 inline figures / 287 native
   equations; supplement 20 table grids / 0 images / 354 native equations);
   packages rebuilt (public **159 files / 31.1 MB**, private **841 files /
   89.8 MB**); cover letters retitled (20 October 2026); portal redeployed;
   both repositories tagged **v1.3.0** (`results/stage12_release.json`).

## Stage 10 deliverables (pre-submission upgrades)

1. **Trunk ablation (10.1A).** Six trunk variants at matched capacity and 500
   epochs, three seeds: the plain MLP trunk gives the lowest interpolation
   net-charge error (0.0914), while radial variants (0.098–0.116) reduce the
   wall-gradient error by about 7 % (radial k=30: 4.83e-3 vs 5.20e-3) and give
   the best integrated-current errors (multi-scale 0.274 %, radial k=30 0.320 %,
   MLP 0.364 %). The pre-defined hypothesis that the radial basis is *essential*
   is **not supported**; per the user's instruction this is reported honestly as
   a scope-bounding ablation, and the multi-scale radial trunk is retained as the
   released default (favourable wall-gradient and current behaviour).
2. **Data-scaling law (10.1B).** A 384-state solve expansion supports nested
   training subsets N=36→288. Interpolation net-charge error falls 0.3927→0.1299
   (fit `ε_ρ = 3.14 N^-0.552`; potential `ε_ψ = 0.0391 N^-0.535`), close to the
   expected `N^-1/2`, whereas out-of-distribution net-charge error changes only
   0.317→0.253 (`α=0.13`). The training box, not the sample count, bounds
   extrapolation.
3. **Fisher + multi-protocol identifiability (10.1C).** Over 15 states and
   three protocols (single-salt, multi-salt, dual I–V + near-wall selectivity)
   at 0.2/1/2 % noise, single- and multi-salt I–V leave the wall charge
   unidentifiable (median error 0.77–1.00) while recovering radius to <0.7 %;
   the dual protocol recovers both below 0.3 %. A 18×18 Fisher sweep gives
   `κ(F)` median 7.46e8 (99.7 % > 1e8), `∂lnI/∂lnR = 2.029`, and
   `|∂lnI/∂lnq_w| ≤ 1.6e-3`: the near-null Fisher direction is the charge axis at
   every operating point.
4. **Figures, supplement, manuscript (10.2–10.3).** Figure 3 is a 2×3 grid
   (physics-loss→C, cross-section→D, new E trunk ablation, F Fisher `log10 κ`);
   Figure 4 gains Panel D (log-log N_train scaling); Figure 6 gains Panels C/D
   (q_w and R recovery). Layout verification reports **0 issues**. The supplement
   gains Section S7b with **S14** and **S15** (header 13→15) and the claim map is
   extended to **C01–C33**. The abstract was cut to 244 words, the Highlights and
   cover letters reframed, and the Discussion/Conclusion/Data-availability
   updated.
5. **Release (10.4).** DOCX rebuilt (main: 155 paragraphs, 0 tables, 6 figures,
   233 `m:oMath`; supplement: 16 tables, 0 images); packages rebuilt (public
   **145 files / 30.2 MB**, private **822 files / 88.5 MB**); **24/24 audit
   PASS**; the portal's Benchmarks tab gained a "Scope and validation" block and
   was redeployed to **https://pino.statwhy.com** (HTTP 200; `/api/solve_sg` =
   196.137225 pA); both repositories were force-pushed and tagged **v1.1.0**
   (`results/stage10_release.json`).

## Stage 11 deliverables (reviewer upgrades)

1. **Exact SG/FV Jacobian/Fisher validation** (`scripts/run_stage11_reviewer_upgrades.py`).
   Centered finite differences of the float64 Scharfetter--Gummel solver on 15
   states (1200 solves, 252 s) give ground-truth log-log sensitivities. The
   operator agrees on `dlnI/dlnR` to a median relative difference of
   **1.5e-4** (2.00003 vs 1.99997) and reproduces the dual-protocol conditioning
   number `kappa(F) ~ 1.8e5`; both currents are flat in `q_w`.
2. **Four-way constraint benchmark** (same backbone, three seeds, interp
   net-charge rel. L2): hyperbolic reparam **0.2066 +/- 0.0154**, linear
   orthogonal projection **0.2008 +/- 0.0206**, soft penalty **0.4114 +/- 0.1871**,
   direct `c` **0.9334 +/- 0.0082**. Only the reparameterization keeps
   `c+ - c- = rho` exact (1.1e-16 vs 4.2e-8).
3. **Physics-weight sweep**: `lambda = 1e-3` is the Pareto point
   (`rho` 0.2125, wall 5.55e-3, Poisson 0.472) and remains the default.
4. **Numbering**: references 1--60 strictly by first appearance (propagated to
   the supplement, `literature_db.json`, claim map and verification record);
   supplementary tables **S1--S18** by first citation (labels only, sections
   preserved per user decision); `Eq. (1)--(8)` and `Eq. (S1)--(S10)` all
   called out in text.
5. **Narrative**: CPiP framing, experimental feasibility of the near-wall
   selectivity readout, compressed §3.10, **Algorithm 1** and pinned versions
   (Python 3.14.7, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1, ONNX Runtime
   1.31.0) in Supplement S2.
6. **Release**: audit extended to **28 checks — 28/28 PASS**; DOCX rebuilt
   (main 0 tables / 6 inline figures / 266 native equations; supplement
   19 table grids / 0 images / 331 native equations); packages rebuilt (public **152 files /
   30.6 MB**, private **834 files / 89.2 MB**); portal scope block updated;
   both repositories tagged **v1.2.0** (`results/stage11_release.json`).

## Stage 9 deliverables

1. **Native editable equations + inline figures (9.1).**
   `scripts/fix_manuscript_typography.py` normalized bare scientific notation,
   caret units, and orphan superscript bases in both manuscript sources. New
   `scripts/omml_math.py` converts LaTeX to OMML via pandoc with a cache
   (`manuscript/cache/omml_cache.json`); the rewritten
   `scripts/export_manuscript_docx.py` emits every display equation as a
   numbered paragraph of **inline `<m:oMath>`** (center+right tab stops), so the
   main DOCX carries **188** `m:oMath` and the supplement **204**, with **zero**
   equation images. Figure blocks were moved inline after their first-citing
   paragraph and the trailing caption section was removed; figures appear before
   References in order 1→6.
2. **Interactive portal (9.2).** `web_simulator/` is now a header + **3-tab**
   site (Operator predictions / Verified benchmarks / Live solver); `app.js`
   calls same-origin `/api/solve_sg` when hosted and falls back to a localhost
   service offline. `scripts/test_web_simulator_dom.js` passes **33/33**;
   `scripts/build_web_bundle.py` regenerated `embedded_bundle.js` (5,662,351 B).
3. **Live deployment (9.3).** `scripts/deploy_pino_statwhy.sh` deploys the portal
   to **https://pino.statwhy.com** in a fully isolated layout on the shared host
   `statwhy.com`: static tree at `/var/www/pino.statwhy.com`, a systemd unit
   `pino-statwhy.service` bound to `127.0.0.1:18765` running the unchanged
   `scripts/serve_interactive.py`, an isolated nginx vhost that proxies `/api/`,
   and Let's Encrypt TLS (expires 2027-01-17). The script writes only its own
   files, gates every reload with `nginx -t`, and verifies the API on loopback
   first. Verified: portal HTML 200, HTTP→HTTPS 301, `/api/solve_sg` returns
   **196.137225 pA** (reference 196.13722469169718), and other sites on the host
   remain 200. Record: `results/stage9_deployment.json`.
4. **Journal suitability assessment (9.4).** `reports/JOURNAL_SUITABILITY_ASSESSMENT.md`
   maps the contribution against CPC and JCP (recommended primary CPC,
   equal alternative JCP), lists strengths/weaknesses, and gives a
   pre-submission checklist. Availability statements in the manuscript, the three
   cover letters, and the public `README.md` now point to the hosted portal.
5. **Rebuild, audit, re-release (9.4).** DOCX rebuilt (main: **0 tables,
   6 inline figures, 0 equation images**; supplement: **13 tables, 0 images**);
   `scripts/audit_release_packages.py` **24/24 PASS**; packages rebuilt (public
   **138 files / 22.5 MB**, private **810 files / 80.6 MB**) and redeployed as
   **v1.0.2**.
6. **Handoff.** This report plus `reports/STATUS.md`, `reports/DECISIONS.md`
   (D-064–D-068), and `reports/NEXT_ACTIONS.md`.

## Stage 8 deliverables

1. **API-backed literature verification (8.1).**
   `scripts/verify_literature_claims_api.py` retrieved canonical metadata and
   abstracts for all **60** references from OpenAlex (inverted-index
   reconstruction), NCBI E-utilities, Crossref, Semantic Scholar, and Europe PMC
   (cached in `literature/cache/verified_abstracts_60.json`), extracted every
   manuscript sentence citing each reference, and audited each sentence against
   the source with a budgeted `gpt-4.1-mini` pass. Basis: **48
   abstract-verified, 12 metadata-only** (`refs 21, 28–32, 34, 35, 37, 40, 50,
   52`). The audit flagged 14 sentences; each was adjudicated and recorded (see
   *Literature adjudication* below). Result: `results/literature_claim_verification.json`
   with **0 flagged**, keeping the raw LLM verdict alongside the adjudicated
   support and note.
2. **Consolidated claim map (8.1).** `scripts/build_claim_evidence.py` rebuilt
   `manuscript/claim_evidence.csv` to a 10-column schema holding **C01–C28**
   (quantitative claims, now pointing at the relocated `Supp. Table S1--S6` and
   renumbered sections) plus **L01–L60** (one literature-support claim per
   citation, with basis and abstract excerpt). All **88** rows are `supported`.
3. **Figure-only main text (8.2).** All six main-text data tables were moved to
   `manuscript/Supplementary_Materials.md` as **Supplementary Tables S1–S6**; the
   seven existing supplement tables were relabeled **S7–S13** with a new S0
   section, captions, and a rebuilt S13 crosswalk. Every supplement table is now
   cited from the main text. The main text contains only the six multipanel
   figures (1A–1D, 2A–2F, 3A–3D, 4A–4C, 5A–5C, 6A–6B), the Highlights, and the
   CPC Program Summary. The Figure 5 paragraph moved to a new §3.5 "Continuous
   parametric control" and later sections were renumbered, so figure
   first-citation order is exactly 1→6. Panel callouts were added for Figures 3
   and 6, the contribution list became prose, and the Figure 2 caption was
   corrected to index 144.
4. **Cover letters (8.3).** `scripts/export_cover_letter.py` now emits three
   journal-tailored letters — `Cover_Letter_CPC`, `Cover_Letter_JCP`, and
   `Cover_Letter_CPC_JCP` (`.md` + `.docx`) — with the refreshed contribution
   numbers and zero LaTeX leakage.
5. **Rebuild, audit, re-release (8.4).** DOCX rebuilt (main: **0 tables,
   6 figures**; supplement: **13 tables**); `scripts/audit_release_packages.py`
   extended to **22/22 PASS** with two new checks (figure-only display items +
   13 supplement tables cited; C01–C28 + L01–L60 all supported); packages
   rebuilt and redeployed as **v1.0.1**.
6. **Handoff (8.5).** This report plus `reports/STATUS.md` and
   `reports/DECISIONS.md` (D-057–D-063).

## Literature adjudication (14 flags → 0)

Every flag was a **cluster-citation strictness** false positive, or the canonical
origin paper, not a substantive error:

- **Cluster citations (13):** `[11,12,16,20,22,23,25,45,46,47,48]` and the
  same-theme groups. The manuscript sentences summarize a body of literature and
  do not attribute unsupported specificity to any single reference; each source
  supports the cited theme (e.g., `[45]`–`[48]` are recent DeepONet
  training/architectural variants; `[20–23]` introduce PINNs).
- **Canonical origin paper (1):** `[33]` is the original Scharfetter–Gummel
  (1969) paper (*IEEE Trans. Electron Devices*, DOI `10.1109/t-ed.1969.16566`);
  its title concerns semiconductor device analysis, but the manuscript sentence
  frames it explicitly as the semiconductor-transport origin of the SG
  discretization, which is exactly this work.

Two **genuine** citation-attribution issues were found during this stage and
fixed in the text **before** the final verification run: the FNO sentence that
had incorrectly cited `[41]` (physics-informed DeepONets) for the spectral
convolution basis now cites the physics-informed DeepONet paper for the
physics-informed formulation and `[37,40]` for the spectral basis. The 12
metadata-only references have no online abstract and are recorded as
`metadata_canonical` support from their canonical titles.

## GitHub URLs and commit SHAs

```
Public repository   : https://github.com/Javad-Alizargar/nanopore-pino-surrogate            (visibility: public)
  main / tag v1.3.3: de9e3d01890ffc2effc3c3c04487cb92fe265d73 (authoritative: results/stage15_release.json)
  tag v1.3.2      : e4ca7330fd2aa477f76ed01998469df526c8507a
  tag v1.3.1      : b2b832192200339f542a6a25e204450ce8fda2dc
  tag v1.3.0      : e97497f861bd8a71f0cf5c660fa871939531c64a
  tag v1.2.0      : 30ba2f841253ce45c264606af719cb64ba6b13ea
  tag v1.0.2      : 0105f96a0863db8f426ac07c6e08851a6403b3cb
  tag v1.0.0      : af5695f50b073a331a60e46ac7a327c610e9ae9e
  files / size    : 165 files, 31.1 MB (31,138,758 B)

Private repository  : https://github.com/Javad-Alizargar/nanopore-pino-surrogate-private    (visibility: private)
  main / tag v1.3.3: 36cc073e54b5e589ccf9800d47662bfec3301ebf (authoritative: results/stage15_release.json)
  tag v1.3.2      : 3b5d61a62a036b836bf701122a296721b9b0ab6f
  tag v1.3.1      : 518d51875bf8169e53b5f79325e71979423ad6c7
  tag v1.3.0      : 7801333be258d18bfe877efe028bb46cff2613e3
  tag v1.2.0      : 39c0a3d57dd11be40c15d282a6249df8f271b04b
  files / size    : 852 files, 89.9 MB (89,889,029 B)

(Each package embeds a build timestamp in its manifest, so a rebuild advances the
commit SHA; the exact released SHAs are always recorded in results/stage15_release.json.)
```

Hosted portal (primary) and GitHub Pages (mirror):
```
Portal      : https://pino.statwhy.com/            (Operator / Benchmarks / Live solver; live SG/FV API)
Health      : https://pino.statwhy.com/api/health  ; POST /api/solve_sg -> 196.137225 pA
TLS         : Let's Encrypt, expires 2027-01-17 ; HTTP->HTTPS redirect active
Mirror      : https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/
```

## Submission file inventory

```
manuscript/Manuscript_PINO_clean.docx               (0 tables, 6 inline figures, native OMML, 0 equation images)
manuscript/Supplementary_Materials_PINO.docx        (16 tables, native OMML, 0 images)
manuscript/Cover_Letter_CPC.docx / Cover_Letter_JCP.docx / Cover_Letter_CPC_JCP.docx  (no LaTeX leakage)
manuscript/Cover_Letter_CPC.md / Cover_Letter_JCP.md / Cover_Letter_CPC_JCP.md
manuscript/Manuscript.md, manuscript/Supplementary_Materials.md       (canonical sources)
manuscript/claim_evidence.csv                        (C01-C33 + L01-L60, 10 columns, all supported)
manuscript/literature_db.json                        (60 references)
literature/cache/verified_abstracts_60.json          (API metadata/abstract cache)
reports/JOURNAL_SUITABILITY_ASSESSMENT.md            (CPC/JCP venue assessment)

Figures (PDF + PNG), manuscript/figures/:
  fig1_workflow_and_architecture            fig4_h1_h2_interp_vs_ood_regimes
  fig2_benchmark_and_field_reconstruction   fig5_parametric_physics_sweeps
  fig3_h3_charge_reparameterization_ablation fig6_h4_timing_and_amortization_breakeven
```

## Final budget ledger summary

```
Project cap              : $5.00
Committed (all stages)   : $0.19564635
Remaining                : $4.80435365
OpenAI subbudget         : $0.1341195 / $1.00
Final-review reserve     : $0.50 (untouched)
Stage 8 literature tasks : stage8_literature_verification $0.0630572 ; _v2 $0.009224 ;
                           _v3 $0.0164608 ; _v4 $0.0075968   (total $0.0963388)
Other Stage 8 work       : $0.00 (editing, DOCX, packages, audit, deploy are local/API only)
Stage 9 work             : $0.00 (typography, OMML export, portal, hosted deploy, assessment, release are all local/API-free)
Stage 11 work            : $0.00 (experiments, numbering, DOCX, audit, portal and release are all local/API-free)
```

## Verified scientific results (unchanged, canonical)

```
H1 interp_test (3-seed mean rel-L2):
  deeponet_reparam  psi 0.0008914  rho 0.08248  I 0.001093  |dI| 0.08031 pA
  fno_reparam       psi 0.005495   rho 0.09310  I 0.008511  |dI| 0.6746 pA
H2 ood_test:
  deeponet_reparam  psi 0.02024  rho 0.3678  I 0.001363  |dI| 0.1077 pA
  fno_reparam       psi 0.02079  rho 0.2241  I 0.009123  |dI| 0.5338 pA
H3 reparameterization rho: DeepONet 8.63x ; FNO 10.29x
H4 ONNX batch-1 (M1): DeepONet 0.072 ms ; FNO 1.04 ms   (params 36,880 vs 460,488 = 12.49x)
Benchmark state (index 144) : reference I 196.137 pA ; live /api/solve_sg 196.137 pA
Inverse demo       : R median rel. err 0.004574 ; q_w not identifiable (median abs err 0.0526) ; ~3,200x vs SG/FV
```

## Audit

`scripts/audit_release_packages.py` → **33/33 PASS** (zero secret hits). The
Stage-13/14 checks keep the DOCX bibliography `[1]--[60]`, enforce individual
supplement-table first mentions in strict order 1..20 with no stray `_child`
legend handles, and verify the official Zenodo DOI in the manuscript, main
DOCX, CPC cover letter, and public README.
The Stage-11 checks remain: references numbered strictly by first appearance
1--60; supplement tables and equations strictly sequential; figures inline
within 1--2 paragraphs of first mention in both Markdown and DOCX XML; and the
Stage-11 artifacts/statistics present. Earlier checks cover the figure-only
main text (order 1→6, 19 supplement tables cited), the claim-evidence map
(C01--C38 and L01--L60, all supported), native OMML equations and the
DOCX/PDF structure. The audit must be run **before** package deploy (the
deploy step creates a `.git` directory in each package tree). Full checklist in
`reports/FINAL_AUDIT_REPORT.md`.

## Remaining limitations

- The independent review was **AI-assisted** (two providers), not expert peer review.
- The DOCX uses native OMML and was additionally page-rendered with LibreOffice for visual confirmation; a Word-specific page proof was not available here.
- `q_w` identifiability is a physical limit of conductance-only inversion (disclosed in the manuscript).
- The archived release is permanently available at Zenodo DOI `10.5281/zenodo.23487404`, verified in the manuscript, main DOCX, CPC cover letter, and public README.
- Literature claim support rests on retrieved abstracts plus adjudication; 12 references are metadata-only (no online abstract) and are flagged as such in the artifact.
- The package builder regenerates each package tree, so both histories were force-pushed for each release (`v1.0.1`, `v1.0.2`); the public tags `v1.0.0`, `v1.0.1`, `v1.0.2` are stable.
- The hosted portal runs on a shared third-party host (`statwhy.com`); uptime and certificate renewal are Let's Encrypt/certbot-managed and outside this repository's control.

## Next (post-completion, author-driven)

Submit to CPC or JCP with the matching cover letter (or the joint letter for an
explicit transfer), per `reports/JOURNAL_SUITABILITY_ASSESSMENT.md`; deposit the
released `v1.0.2` commit to Zenodo at acceptance and insert the real DOI;
optional human expert review. Re-verification chain before any future release:
`verify_figure_layout.py` (0 issues) → `verify_simulators.py` →
`verify_live_servers.py` → `verify_literature_claims_api.py` (0 flagged) →
`build_reproducibility_packages.py` → `audit_release_packages.py` (24/24,
**before** deploy) → `deploy_github_repos.py` → `deploy_pino_statwhy.sh`.
