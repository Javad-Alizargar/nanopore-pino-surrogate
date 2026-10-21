# Cover Letter

**Date:** 21 October 2026

**To the Editors-in-Chief**
*Journal of Computational Physics*

**Subject:** Submission of an original research article

Dear Editors,

We submit this manuscript for consideration in the *Journal of Computational Physics*. The software package PINO-Nanopore (archived permanently at Zenodo DOI: 10.5281/zenodo.23487404 and deployed interactively at https://pino.statwhy.com) provides a validated, exact-boundary solution operator for steady axisymmetric Poisson--Nernst--Planck transport, together with a controlled numerical study and an identifiability analysis for nanopore experiments.

I am pleased to submit the original research article entitled **"Sensitivity-controlled physics-constrained neural operators and conductance identifiability limits for Poisson--Nernst--Planck transport in charged nanopores"** for consideration at *Journal of Computational Physics*.

**Author:** Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan (corresponding author: 8javad@ntunhs.edu.tw; ORCID: 0000-0003-2966-1315).

This work is a natural fit for *Journal of Computational Physics* because its core contributions are a numerical-methods result and a validated identifiability ansatz that removes subtractive cancellation in coupled Nernst--Planck fields, a controlled accuracy, ablation, and out-of-distribution study of two operator backbones against a conservative Scharfetter--Gummel reference, and a Fisher analysis validated against exact float64 finite-difference Jacobians.

This work advances decisively beyond my prior fixed-state deep-adaptive PINN study (submitted as COMPHY-D-26-00857), which solved the axisymmetric Poisson--Nernst--Planck (PNP) system for a single operating point at a time and required roughly 500 s of retraining per new state. The present manuscript replaces that per-state workflow with a single parameter-conditioned neural operator, and it contributes the following.

1. **A 4D parameter-conditioned operator.** A multi-scale radial physics-informed DeepONet maps the four physical parameters $(V_{\text{app}}, C_0, q_w, R)$ directly to the coupled potential and ionic-concentration fields, eliminating the ~500 s per-state retraining bottleneck of the prior DA-PINN and enabling instant querying of the whole parameter family from a single trained model.

2. **An exact-boundary, charge-balanced reparameterization.** The ansatz predicts a positive concentration sum $s$ and a net charge $\rho$, with the exponential envelope $s = 2\exp(\mathrm{clip}(z(1-z)G_s))$ and $\rho = z(1-z)G_\rho$, and reconstructs physical concentrations through the hyperbolic identity $c_\pm = (\sqrt{s^2+\rho^2} \pm \rho)/2$. This enforces the reservoir boundary conditions exactly and removes the subtractive cancellation that limits net-charge accuracy, reducing net-charge error 8.63-fold (DeepONet) and 10.29-fold (FNO) relative to a naive direct-concentration output; against a fairly trained direct loss, the improvement is 13% net charge and 18% current.

3. **A head-to-head operator comparison.** The multi-scale radial PI-DeepONet (36,880 parameters, 0.072 ms ONNX latency) is benchmarked against a four-layer real-DFT Fourier neural operator with 12.5-fold more parameters (460,488), across both in-distribution and out-of-distribution Debye-overlap regimes. The DeepONet is 6.16-fold more accurate in potential and 7.79-fold more accurate in current in-distribution, while the FNO is better for net charge and cation concentration out of distribution, isolating where each architecture succeeds.

4. **An inverse-sensing demonstration and an identifiability result.** A gradient-free inverse recovery runs roughly 3,200-fold faster than the finite-volume reference solver. A Fisher-information analysis and a blind 2,700-realization study reveal a physical conductance identifiability boundary: dual current--plus--near-wall-selectivity inversion reaches 1.7% median charge error and 0.17% radius error, but only 68% 95% coverage at 1% noise because a roughly 2% surrogate bias floor remains; neither single-salt nor multi-salt current--voltage data resolve the charge. The negative-$q_w$ Fisher plane is ill-conditioned (median $\kappa(\mathcal{F})=1.2\times10^{9}$; 99.1% above $10^{8}$), with a 1%-noise CRLB of 0.37% for radius but 86 for charge. The conductance-only contrast is a limitation of the measurement, not of the surrogate or solver.

5. **Scope-bounding ablations.** A trunk ablation at matched capacity shows that the multi-scale radial basis is a modest near-wall inductive bias rather than a necessity (about 7% lower wall-gradient error, no bulk-accuracy advantage), and a nested 384-state study gives an in-distribution data-scaling exponent of 0.55 close to the expected N^-1/2, while out-of-distribution error is nearly data-insensitive; a spatial decomposition shows that the residual OOD error is concentrated in the wall EDL and reservoir transitions but cancels in the integrated current.

6. **Reproducibility and an interactive tool.** A verified 240-state benchmark plus a 384-state scaling expansion, trained seed-42 models in PyTorch and ONNX, archived result files, a serverless browser simulator, and an MIT-licensed reproducibility package accompany the manuscript. Every number in the paper is traceable to an archived artifact.

7. **A four-way constraint benchmark and an exact Jacobian validation.** Under an identical PI-DeepONet backbone and three seeds, the hyperbolic reparameterization matches the best projection scheme in accuracy while keeping the charge identity exact to machine precision, and a physics-weight sweep identifies $10^{-3}$ as the Pareto point. A centered finite-difference Jacobian of the float64 SG/FV solver validates the operator's Fisher analysis: the two agree on $\partial\ln I/\partial\ln R$ to $1.1\times10^{-4}$, while both show a near-null single-salt charge direction that the near-wall selectivity observable resolves.

8. **A diffusivity-asymmetry proof and a protocol-design workflow.** We show analytically that the local ohmic coefficient is $\bar D\sqrt{s^2+\rho^2}+\tfrac12(D_+-D_-)\rho$, so equal diffusivities leave only a second-order charge term, and verify with the exact float64 SG/FV solver at four diffusivity ratios ($D_+/D_-=1.0$, 0.966, 0.655, 0.20) and two salinities. Even NaCl leaves the single-salt charge direction poorly conditioned, while adding the near-wall selectivity observable breaks the degeneracy. The released package and portal expose the Jacobian and Fisher construction so that experimentalists can optimize bias, salinity, and observable combinations before fabricating or measuring a pore.

9. **A sensitivity-controlled hybrid solver and a blind inverse benchmark.** A reference-free Weyl-gated controller warm-starts the conservative SG/FV solve from the operator and reproduces the cold-solve current to $3\times10^{-12}$ at 0.165 s per state, 9.5-fold below a full $21\times61$ solve, on a 120-state weak/transitional/strong benchmark. A 2,700-realization blind inverse study with shared noisy observations shows that the dual current-plus-selectivity protocol reaches 1.7% median charge error but only 68% 95% coverage at 1% noise, while the reversal potential is state dependent (263% weak-charge versus 7.3% strong-charge). We therefore report a bias-limited 2% recovery and a strong-exclusion 7% recovery, not calibrated sub-percent charge identification.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission, and the author declares no competing interests.

**Data and code availability.** The reproducibility package is available at https://github.com/Javad-Alizargar/nanopore-pino-surrogate under the MIT License, and the interactive simulator is hosted at https://pino.statwhy.com/. The permanently archived release is available at Zenodo DOI: 10.5281/zenodo.23487404. The repository and archive contain a comprehensive `README.md` covering all files, installation, and execution, together with complete sample test-run input and output files in `test_run/`; `scripts/run_cpc_test_case.py` runs that self-contained CPC verification case.

Thank you for considering this manuscript. I look forward to your response and would be glad to provide any additional information.

Sincerely,

**Javad Alizargar**
Department of Nursing, National Taipei University of Nursing and Health Sciences
8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315
