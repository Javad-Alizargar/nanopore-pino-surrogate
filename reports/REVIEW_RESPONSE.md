# Independent review response (Stage 6B.1)

Two independent AI reviewers from different providers inspected the Stage 6A
manuscript, supplement, and claim-evidence map. This is an AI-assisted
methodological review, not expert peer review.

- Reviewer 1: OpenAI `gpt-4.1-mini`, 11 issues, recommendation **minor revision**, cost $0.00549.
- Reviewer 2: DeepSeek `deepseek-v4-pro` (thinking disabled), 12 issues, recommendation **minor revision**, cost $0.02112.
- Task allowance: $0.15. Total task spend (including probes and one retried call): ~$0.079.
- Raw structured output: `results/independent_review.json`.

Scientific conflicts were resolved against the code and the validated results,
not by majority vote. Every actionable issue is listed below with severity,
evidence, controller decision, and resolution status.

## Reviewer 2 (DeepSeek) issues

### R1-1 (major) — Raw 15–17 digit floats throughout the prose
- Evidence: `0.0008913628923944445`, `459.2494577697552 states`, `50.59363783302251 s`, `0.07165499962866306 ms`, `9.936325096049359e-10`, `196.137225 pA`, etc.
- Decision: valid.
- Resolution: **fixed**. `scripts/format_manuscript_numbers.py` rounds narrative and table values to 3–4 significant figures (247 placements in the main text, 87 in the supplement, across three widening passes) while protecting code spans, math commands, URLs, DOIs, and the reference list; pure-numeric inline math is rounded, symbol-bearing math is preserved. Exact values remain in `results/statistics.json` and `results/tables/*.csv`; both manuscripts now state this.

### R1-2 (major) — CPC block missing required fields
- Evidence: missing "Program obtainable from", "Distribution format", "Operating system", "RAM", "Number of processors", "Parallelization", "Keywords", "Classification".
- Decision: valid (CPC requirement).
- Resolution: **fixed**. The CPC summary now lists software name, obtainable-from URL, distribution format, computer, OS, RAM, processor count, parallelization, keywords, and classification, and rounds the running-time figures.

### R1-3 (major) — Ansatz incorrectly described as vanishing at the reservoirs
- Evidence: manuscript says `s=z(1-z)G_s` "vanish[es] at both reservoirs and therefore approach the neutral bulk value" while also claiming `s→2C_0`; supplement S3 says the ansatz "does not force c±=C0".
- Decision: valid and critical. `scripts/operators/boundary_ansatz.py` implements `s=2·exp(clip(z(1-z)G_s,−15,15))`, `ρ=z(1-z)G_ρ`, so at the reservoirs `s→2` and `ρ→0`, i.e. `c±→1 (=C0)` **exactly**.
- Resolution: **fixed**. Main text §2.3 and Supplement S3 now state the correct exponential form, the clip, and the exact reservoir limit `c±→C0`; the contradictory sentences were removed.

### R1-4 (minor) — Dimensionless PNP system not written out
- Evidence: only `β=(L/λ_D)²/2` and `α_r=(L/R)²` are stated.
- Decision: valid.
- Resolution: **fixed**. New Supplement §S3.1 gives the dimensionless Poisson and continuity equations, `α_r`, `β`, `λ_D²=ε0εrV_T/(2FC0)`, the SG face flux with `B(x)=x/(e^x−1)`, and the boundary conditions. The `λ_D` convention retains the factor 2 because `scripts/operators/constants.py` defines `λ_D²=ε0εrRT/(2F²C0)`; it is algebraically identical to `ε0εrV_T/(2FC0)`.

### R1-5 (minor) — Physics loss not specified exactly
- Evidence: manuscript only says "penalizes the Poisson residual and the continuity residuals".
- Decision: valid.
- Resolution: **fixed**. Main text §2.4 and Supplement S2 now give `L_data` as a per-state normalized MSE with the state scale floored at 10% of the global RMS, and `L_phys = mean|R_Poisson| + ½(mean|R_cont,+| + mean|R_cont,−|) + mean|R_wall,ψ| + mean|R_wall,J+| + mean|R_wall,J−|`, matching `scripts/operators/losses_and_metrics.py`.

### R1-6 (minor) — DA-PINN comparison is not apples-to-apples
- Evidence: the operator is compared with a DA-PINN retrained on the same single state.
- Decision: valid.
- Resolution: **fixed**. §3.2 now states that the DA-PINN was trained specifically on the benchmark state (500 s retrain per state) while the operator has never seen it, and presents the 2.57-/1.83-fold ratios as an accuracy-per-training-cost observation rather than dominance.

### R1-7 (minor) — Break-even count presented too strongly
- Evidence: 459 states at the dataset-mean time is only ~1.9× the 240-state training set.
- Decision: valid.
- Resolution: **fixed**. §3.6 now notes the 1.9× relation and reserves the "two to three orders of magnitude" claim for sweeps of thousands of states; the conclusion says "about 460 states" and no longer says "a few hundred".

### R1-8 (minor) — Relative q_w error dominated by a near-zero state
- Evidence: median relative q_w error 1.154 is driven by state 175 (`q_w=−0.00206`).
- Decision: valid.
- Resolution: **fixed**. §3.8 and Table 6 now report the median absolute q_w error (0.0526, about half the prior width), the median absolute R error (0.0181 nm), and the state-175 absolute error (0.0961) alongside the relative values.

### R1-9 (editorial) — "3.6e-7 parity" is ambiguous
- Evidence: highlight and conclusion use "parity" without a metric.
- Decision: valid.
- Resolution: **fixed**. The highlight says "ONNX matches PyTorch within 3.6e-7 in browser and Gradio apps"; the conclusion says "match PyTorch to within 3.6×10⁻⁷ maximum absolute difference".

### R1-10 (editorial) — Figure 5 description vague/incorrect
- Evidence: "four radii", unlabeled panels.
- Decision: valid.
- Resolution: **fixed**. §3.3 now states panel A uses three radii (3, 5, 7 nm) with five SG/FV biases, panel B the near-wall c+ versus C0 over 10–200 mol m⁻³, and panel C the radial net charge at three salt concentrations; a full caption was added.

### R1-11 (editorial) — Benchmark current inconsistent
- Evidence: prose `196.137225 pA` versus the live solver `196.13722469169713 pA`.
- Decision: valid.
- Resolution: **fixed**. The value is reported as `196.1 pA` in prose (with the exact value archived), consistent with the live service.

### R1-12 (editorial) — Speed-up reported to 15 digits
- Evidence: `3213.968163046741×`.
- Decision: valid.
- Resolution: **fixed**. Reported as "an approximately 3200-fold speed-up".

## Reviewer 1 (OpenAI) issues

### R1-1 (minor) — Define the latent fields and the ansatz domain
- Evidence: `G_ψ`, `G_s`, `G_ρ` and the range of `z` are not defined.
- Decision: valid.
- Resolution: **fixed**. §2.3 defines the three smooth latent fields and `z∈[0,1]`.

### R1-2 (minor) — Explain how reservoir data are enforced exactly
- Evidence: the reservoir conditions are asserted but not derived.
- Decision: valid.
- Resolution: **fixed**. §2.3 and Supplement S3 show `ψ_θ(r,0)=V_app`, `ψ_θ(r,1)=0`, and `c±→C0` identically from the ansatz, with no penalty.

### R1-3 (editorial) — Number formatting
- Evidence: same as Reviewer 2 R1-1.
- Decision: valid.
- Resolution: **fixed** by the rounding pass (see R2 R1-1).

### R1-4 (minor) — Trunk feature `s` undefined / notation collision
- Evidence: prose uses `e^{-s(1-r)}` with an undefined `s`, colliding with the concentration sum `s`.
- Decision: valid. `scripts/operators/deeponet.py` uses four fixed scales `κ∈{1,3,10,30}` (`_RADIAL_SCALES=(1.0,3.0,10.0,30.0)`), not a learnable parameter.
- Resolution: **fixed**. §2.4 now writes the trunk input as `(r,z,{e^{-κ(1−r)}})` with `κ∈{1,3,10,30}`, removing the collision.

### R1-5 (minor) — Identifiability caveat for the inverse problem
- Evidence: q_w is a flat direction for conductance-only inversion.
- Decision: valid.
- Resolution: **fixed**. §2.6 and §3.8 now state that the conclusion is a physical identifiability limit, that the 1% noise is a conservative proxy, and that it would weaken for asymmetric electrolytes, unequal diffusivities, or a wider charge range.

### R1-6 (minor) — Explain OOD current robustness versus field sensitivity
- Evidence: OOD potential error grows 22.7-fold while current stays below 0.14%.
- Decision: valid.
- Resolution: **fixed**. §3.4 now explains that the current is an integral of the flux and benefits from error cancellation, whereas pointwise fields remain sensitive to local extrapolation error.

### R1-7 (editorial) — Running times to many digits
- Evidence: `50.59363783302251 s`, `108.5470477079507 s`, etc.
- Decision: valid.
- Resolution: **fixed** by the rounding pass.

### R1-8 (minor) — Table captions lack units
- Evidence: current, residuals, and errors without explicit units.
- Decision: valid.
- Resolution: **fixed**. Table 3, 4, and 6 headers/captions state relative-L2, pA, ms, states, nm, and mol m⁻³ as appropriate, and now state that values are rounded for display.

### R1-9 (minor) — Exact finite-volume/SG discretization not written
- Evidence: only a verbal description of the solver.
- Decision: valid.
- Resolution: **fixed**. Supplement S3.1 gives the SG face flux, the Bernoulli function, the conservative flux divergence, and the boundary conditions; §2.2 points to it.

### R1-10 (editorial) — Define the charge-balanced ansatz in the abstract
- Evidence: the term is used without definition.
- Decision: valid.
- Resolution: **fixed**. The abstract states the construction (predict a positive concentration sum and a net charge, recover concentrations hyperbolically) and that reservoir Dirichlet data and charge consistency are imposed exactly.

### R1-11 (minor) — Justify the inverse noise level and optimizer
- Evidence: 1% Gaussian noise and Nelder–Mead refinement are not justified.
- Decision: valid.
- Resolution: **fixed**. §2.6 now states that 1% is a deliberately conservative proxy for a clean nanopore measurement and that the grid search, not the local refinement, determines the reported flat direction.

## Additional fix found during application
- Wall-condition sign: the main text wrote `−ε0εr ∂rψ = σ` while the solver applies `∂rψ̃ = q_w`. The main text and Supplement S3 were reconciled to the code convention (negative `q_w` gives a negative wall potential), with the sign convention stated explicitly. No numerical result changed.

## Outcome
All 23 issues are resolved in `manuscript/Manuscript.md` and
`manuscript/Supplementary_Materials.md`. No fabricated data were introduced;
every corrected value traces to `results/statistics.json`,
`results/tables/*.csv`, `results/inverse_recovery_demo.json`, or the source
code.
