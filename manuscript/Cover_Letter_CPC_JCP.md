# Cover Letter

**Date:** 19 October 2026

**To the Editors-in-Chief**
*Computer Physics Communications* / *Journal of Computational Physics*

**Subject:** Submission of an original research article

Dear Editors,

I am pleased to submit the original research article entitled **"A parameter-conditioned physics-informed neural operator for axisymmetric Poisson--Nernst--Planck transport in charged nanopores"** for consideration at *Computer Physics Communications* / *Journal of Computational Physics*.

**Author:** Javad Alizargar, Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan (corresponding author: 8javad@ntunhs.edu.tw; ORCID: 0000-0003-2966-1315).

This work advances decisively beyond my prior fixed-state deep-adaptive PINN study (submitted as COMPHY-D-26-00857), which solved the axisymmetric Poisson--Nernst--Planck (PNP) system for a single operating point at a time and required roughly 500 s of retraining per new state. The present manuscript replaces that per-state workflow with a single parameter-conditioned neural operator and, in doing so, contributes the following.

1. **A 4D parameter-conditioned operator.** A multi-scale radial physics-informed DeepONet maps the four physical parameters $\boldsymbol{\mu} = (V_{\text{app}}, C_0, q_w, R)$ directly to the coupled potential and ionic-concentration fields, eliminating the ~500 s per-state retraining bottleneck of the prior DA-PINN and enabling instant querying of the whole parameter family from a single trained model.

2. **An exact-boundary, charge-balanced hyperbolic reparameterization.** The ansatz $s = 2\sqrt{c_+ c_-}$ and $\rho = c_+ - c_-$, with $c_\pm = s/2 \pm \rho/2$ and the hyperbolic form $s = 2\exp(\mathrm{clip}(z(1-z)G_s))$, $\rho = z(1-z)G_\rho$, enforces reservoir boundary conditions exactly ($c_\pm \to C_0$) and removes the subtractive cancellation that limits net-charge accuracy. It reduces the net-charge error by approximately 5x--10x relative to an unconstrained output parameterization.

3. **A head-to-head operator comparison.** The Multi-Scale Radial PI-DeepONet (36,880 parameters, 0.072 ms ONNX latency) is benchmarked directly against a 2D Spectral Fourier Neural Operator (FNO) with 12.49x more parameters (460,488), across both in-distribution and out-of-distribution Debye-overlap regimes, isolating where each architecture succeeds and fails.

4. **An inverse-sensing demonstration and an identifiability result.** A gradient-free inverse recovery over the trained operator runs roughly 3,200x faster than the finite-volume/Shell-Gupta reference solver, and reveals a physical conductance identifiability boundary: the pore radius is recoverable from current alone, whereas the surface charge is not (its effect on conductance is <0.2%, well below measurement noise).

5. **Reproducibility and an interactive tool.** A live, serverless browser simulator and a fully open-source MIT-licensed reproducibility package (data, trained seed-42 models in PyTorch and ONNX, scripts, and verification reports) accompany the manuscript.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission, and the author declares no competing interests.

**Data and code availability.** The reproducibility package is available at https://github.com/Javad-Alizargar/nanopore-pino-surrogate under the MIT License, and the interactive simulator is hosted at https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/. A permanent Zenodo DOI will be minted upon acceptance.

Thank you for considering this manuscript. I look forward to your response and would be glad to provide any additional information.

Sincerely,

**Javad Alizargar**
Department of Nursing, National Taipei University of Nursing and Health Sciences
8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315
