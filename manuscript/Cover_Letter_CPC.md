**Date:** 23 October 2026

**To the Editors-in-Chief,** *Computer Physics Communications*

**Subject:** Submission of an original research article

Dear Editors,

I submit the original research article entitled **"Charge-balanced neural operators, conductance identifiability limits, and warm-started finite-volume solvers for nanopore ion transport"** for consideration at *Computer Physics Communications*. The submission is under the Computer Programs in Physics (CPiP) category. The software package PINO-Nanopore (archived permanently at Zenodo DOI: 10.5281/zenodo.23487404 and deployed interactively at https://pino.statwhy.com) provides a validated, exact-boundary solution operator for steady axisymmetric Poisson--Nernst--Planck transport, whose primary scientific application is high-throughput Fisher-information and conductance-identifiability mapping for nanopore experiments.

**Author:** Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan (corresponding author: 8javad@ntunhs.edu.tw; ORCID: 0000-0003-2966-1315).

**Relationship to COMPHY-D-26-00857.** An earlier manuscript by the same author, *"Exact-boundary data-assimilated physics-informed neural networks for coupled Poisson--Nernst--Planck transport at charged nanopore biointerfaces"*, was submitted to Computer Physics Communications as COMPHY-D-26-00857 (revision R1); it has not been published. That study is fixed-state and needs about 500 s per new state. The present manuscript is a separate, related study. It reuses the verified SG/FV reference and the DA-PINN benchmark comparison, but its new contributions are the parameter-conditioned operator, the charge-balanced exact-boundary reparameterization with its controlled ablation, and the identifiability analysis with the blind inverse and warm-started solver benchmark. Shared methods and benchmark material are identified explicitly in the manuscript.

**Contributions.** (i) A single multi-scale radial physics-informed DeepONet maps the four parameters (applied voltage, salinity, wall charge, and radius) to the coupled potential and ionic concentrations, with exact reservoir boundary conditions and a charge-balanced hyperbolic reconstruction that removes subtractive cancellation in net charge. It reaches 0.00089/0.00109 relative L2 error for potential/current at 0.072 ms per state after ONNX export. (ii) A Fisher-information analysis, validated against exact float64 SG/FV Jacobians, shows that a conductance-only measurement leaves wall charge practically unidentifiable for symmetric electrolytes with nearly equal diffusivities. A blind 675-realization study with exact SG/FV observations confirms this limit: a dual current-plus-near-wall-selectivity protocol reaches 1.5% median charge error, but with 73% 95% coverage at 1% noise dominated by a 1.3% cross-grid discretization bias; single-salt fits do not converge, so their intervals are reported as provisional local intervals. (iii) A reference-free warm-started SG/FV solver reproduces the cold-solve current to 5 × 10⁻¹⁴ at 0.153 s (1.54× faster than a cold forward solve), and a warm-started reversal root is 1.19× faster than cold Brent. The benchmark, timing file, simulator, and MIT-licensed archive are included.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. I approve this submission and declare no competing interests.

**Data and code availability.** The MIT-licensed package, including the 675 inverse records and the 120-case timing benchmark, is at https://github.com/Javad-Alizargar/nanopore-pino-surrogate. The archived release is at Zenodo DOI: 10.5281/zenodo.23487404, and the simulator is at https://pino.statwhy.com/.

Thank you for considering this manuscript. I look forward to your response and would be glad to provide any additional information.

Sincerely,

**Javad Alizargar**
Department of Nursing, National Taipei University of Nursing and Health Sciences
8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315
