# Journal suitability assessment

Project: `nanopore_pino_surrogate` — *A parameter-conditioned physics-informed
neural operator for axisymmetric Poisson--Nernst--Planck transport in charged
nanopores*.
Date: 2026-10-19. Prepared at Stage 9.4. Companion letters:
`manuscript/Cover_Letter_CPC.{md,docx}`, `Cover_Letter_JCP.{md,docx}`,
`Cover_Letter_CPC_JCP.{md,docx}`.

## 1. What the paper is

A numerical-methods contribution with a computational-software deliverable.
The core results are:

- a **hard-constrained parameter-conditioned operator ansatz** (multiscale
  radial PI-DeepONet and a real-DFT FNO) that reconstructs charge-balanced
  `(psi, c_+, c_-)` fields, removing the subtractive cancellation of penalty
  PINNs;
- a **charge-reparameterization** study showing a **8.63x (DeepONet)** and
  **10.29x (FNO)** reduction in charge-density error (H3);
- a controlled **accuracy / out-of-distribution / ablation / timing** study
  against a conservative Scharfetter--Gummel finite-volume reference
  (H1--H4, Tables S1--S13);
- an **inverse recovery** demonstration and a **~3,200x** SG/FV speed-up at
  batch-1 ONNX inference (0.072 ms DeepONet, 1.04 ms FNO);
- a **reproducibility package**: MIT public repository, 240-state dataset,
  PyTorch + ONNX models, a browser portal with a live SG/FV solver
  (https://pino.statwhy.com/), and a Gradio app.

## 2. Candidate venues

| Venue | Type | Fit | Notes |
|---|---|---|---|
| **Computer Physics Communications (CPC)** | software + methods | **Strong** | Primary deliverable is a reusable, documented, reproducible package with a Program Summary; CPC explicitly publishes physics software. |
| **Journal of Computational Physics (JCP)** | numerical methods | **Strong** | Core contribution is a numerical-methods result (constrained operator ansatz + SG/FV comparison + OOD study). |
| **Neural Networks / JMLR** | machine learning | Moderate | The novelty is physics/methods, not learning theory; ML venues would want broader benchmarks. |
| **J. Comput. Chem. / Phys. Chem. Chem. Phys.** | application | Weak | The paper is substrate-general, not chemistry-specific; would dilute the methods message. |
| **SoftwareX** | software-only | Moderate | Emphasis is the package, not the science; would under-sell the numerical results. |

## 3. Fit against CPC

CPC criteria, mapped to evidence:

- **Reusable, documented program.** `scripts/` + `README.md` + `ARCH_MAP`;
  ONNX + PyTorch models; browser and Gradio front ends. PASS.
- **Program Summary.** Present in the manuscript (CPC Program Summary block).
- **Novelty over existing software.** No prior parameter-conditioned PNP
  operator surrogate with charge-balanced reparameterization and a
  conservative SG/FV reference is bundled this way.
- **Reproducibility.** Public MIT repo, seed-42 dataset, archived result
  files, deterministic verification scripts, and a documented audit chain.

**Assessment: suitable.** The only CPC friction is the volume of numerical
results; these are already archived in the supplement (S1--S13) and do not
bloat the program summary.

## 4. Fit against JCP

JCP criteria, mapped to evidence:

- **Numerical-methods novelty.** Hard-constrained operator ansatz with exact
  reservoir Dirichlet + charge-balanced reconstruction; ablation isolates each
  component.
- **Rigorous error analysis.** Interp/OOD regimes, three-seed statistics,
  reference agreement to 196.137 pA at the benchmark state, residuals
  ~1e-9.
- **Comparison to a trusted solver.** Conservative SG/FV finite-volume
  discretization with a convergence/budget check (Table S5).

**Assessment: suitable.** JCP reviewers may ask for an explicit *convergence
order* study of the SG/FV reference and a wider parameter sweep; both are
partly covered (Table S5, Figure 6) and can be extended in revision.

## 5. Relative strengths

1. Four-dimensional parameterization `(V_app, C0, q_w, R)` with a genuine
   OOD regime, not a single-parameter toy problem.
2. A physical fix (charge reparameterization) with a **measured** error
   reduction, not a claimed one.
3. Rare for the class: a **live, same-origin interactive tool** plus a
   serverless browser fallback in the same artifact.
4. A complete, independently re-verifiable **claim-evidence map**
   (`manuscript/claim_evidence.csv`, C01--C28 + L01--L60, all supported) and a
   60-reference literature audit (48 abstract-verified, 12 metadata-only, 0
   substantive flags).

## 6. Known weaknesses (disclose, do not hide)

- The independent review was **AI-assisted**, not expert peer review.
- `q_w` is **not identifiable** from conductance only (median abs error
  0.0526); stated as a physical limit of the inverse problem.
- DOCX equations are validated structurally (native OMML) but no
  page-proof was produced by the authors here.
- No permanent **Zenodo DOI** yet; deposit is deferred to acceptance.
- 12 references have no online abstract (metadata-only support, flagged).

## 7. Recommended submission path

1. **Primary: Computer Physics Communications**, with
   `Cover_Letter_CPC.docx` (software-first framing).
2. **Equal alternative: Journal of Computational Physics**, with
   `Cover_Letter_JCP.docx` (methods-first framing).
3. Use `Cover_Letter_CPC_JCP.docx` **only** for an explicit CPC↔JCP transfer
   request, never as a first submission.

Both letters share the same data/code availability statement pointing to the
public MIT repository and the hosted portal at https://pino.statwhy.com/.

## 8. Pre-submission checklist

- [x] Figure-only main text; all 13 supplement tables cited.
- [x] Editable (native OMML) equations in both DOCX files.
- [x] 24/24 release audit PASS; zero secret hits.
- [x] Literature claims verified; 0 flagged.
- [x] Public + private repositories tagged `v1.0.1` (this assessment predates
      the `v1.0.2` re-release that adds the hosted portal).
- [ ] Mint Zenodo DOI at acceptance and insert it in the availability statement.
