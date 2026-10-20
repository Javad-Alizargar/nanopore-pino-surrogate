# Conductance identifiability limits and physics-constrained neural operators for axisymmetric Poisson--Nernst--Planck transport in charged nanopores

**Javad Alizargar**  
Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan  
Corresponding author: 8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315

---

## Highlights

- PINO-Nanopore maps four PNP parameters to fields in 0.072 ms on one CPU.
- Charge-balanced sum/difference outputs cut net-charge error 8.63--10.29-fold.
- Exact SG/FV Jacobians confirm the operator's conductance sensitivities.
- Wall charge is unidentifiable from conductance alone ($\kappa(\mathcal{F})>10^{8}$).
- A near-wall selectivity probe recovers radius and charge below 0.3%.

---

## Abstract

Inferring nanopore surface charge and geometry from ionic current--voltage measurements is a central inverse problem in nanofluidic sensing, yet repeated Poisson--Nernst--Planck (PNP) solves are too slow for exhaustive identifiability mapping and single-state physics-informed networks require retraining per operating point. We address both with PINO-Nanopore, a validated, exact-boundary, charge-balanced operator for steady axisymmetric PNP transport over voltage, salinity, wall charge, and radius, used as a high-throughput instrument to map what conductance can and cannot identify. A conservative Scharfetter--Gummel solver generates a 240-state benchmark (maximum residual $9.936\times10^{-10}$). A multi-scale radial physics-informed DeepONet (36,880 parameters) and a real-DFT Fourier neural operator (460,488 parameters) predict a positive salt sum and net charge and reconstruct concentrations hyperbolically, so reservoir data and the charge identity hold exactly ($1.1\times10^{-16}$). On held-out states the DeepONet reaches relative $L_2$ errors of 0.0008914 for potential and 0.001093 for current, and 0.001363 out of distribution, at 0.072 ms per state on one CPU. Verified against exact float64 SG/FV Jacobians (median agreement $1.5\times10^{-4}$ in $\partial\ln I/\partial\ln R$), a Fisher analysis proves fixed wall charge unidentifiable for symmetric binary electrolytes with nearly equal ionic diffusivities, such as KCl, from conductance alone ($\kappa(\mathcal{F})>10^{8}$), and a near-wall Donnan selectivity observable collapses the conditioning by ten orders, recovering radius and charge below 0.3%. Unequal diffusivities add a linear but still ill-conditioned charge channel. A four-way constraint benchmark, a physics-weight sweep, and an out-of-distribution error decomposition isolate the design choices. The package is deployed in browser and Gradio interfaces with a live reference solver.

**Keywords:** Poisson--Nernst--Planck; nanopore; neural operator; DeepONet; Fourier neural operator; physics-informed machine learning; inverse problem; identifiability; Donnan selectivity

---

## CPC Program Summary

**Program title:** PINO-Nanopore

**Software name:** `nanopore-pino-surrogate`

**Program obtainable from:** https://github.com/Javad-Alizargar/nanopore-pino-surrogate

**Permanent archive DOI:** https://doi.org/10.5281/zenodo.23487404

**CPC Library link to program files:** added by the CPC Technical Editor upon acceptance

**Licensing provisions:** MIT License (Copyright 2026 Javad Alizargar)

**Distribution format:** Source code (Python and JavaScript), ONNX models, CSV/JSON data, and documentation

**Programming language:** Python 3.14; JavaScript ES6

**Computer:** Apple M1 (arm64); any x86-64 or arm64 machine with Python 3.10+ and a modern browser

**Operating system:** macOS 14.1 (tested); Linux and Windows are supported by the dependencies

**RAM:** 8 GB minimum; 16 GB recommended

**Number of processors used:** 1 CPU core (no GPU required)

**Parallelization:** Serial; batched ONNX inference and single-threaded browser inference

**Keywords:** Poisson--Nernst--Planck; nanopore; neural operator; DeepONet; Fourier neural operator; physics-informed machine learning; inverse problem; identifiability; Donnan selectivity

**Classification:** 02.70.-c (computational techniques); 87.16.dj (membranes and bilayers)

**Nature of the problem:** Steady, axisymmetric Poisson--Nernst--Planck transport of a symmetric binary electrolyte in a charged cylindrical nanopore. The solver must represent the potential, both ion concentrations, the net charge density, and the axial current over a four-dimensional operating space: applied voltage, bulk concentration, wall charge strength, and pore radius. Repeated high-fidelity finite-volume solves are accurate but cost about 0.2 s each, which is prohibitive for design sweeps, uncertainty quantification, and inverse parameter estimation. A second problem is scientific rather than computational: it is not known which of these parameters a conductance measurement can actually identify, because geometric access and wall-charge Donnan enrichment enter the current in a correlated way.

**Solution method:** A conservative Scharfetter--Gummel finite-volume reference solver generates a verified 240-state dataset. Two parameter-conditioned operator backbones are trained: a multi-scale radial physics-informed DeepONet and a parameter-conditioned two-dimensional Fourier neural operator. Both use an exact-reservoir Dirichlet ansatz and a charge-balanced field parameterization that guarantees positive concentrations and an exact algebraic charge identity. The operator is then used as a computational instrument: its Fisher information is validated against exact float64 finite-difference Jacobians of the reference solver, and the resulting identifiability map separates radius from wall charge and identifies the near-wall Donnan selectivity observable that breaks the conductance degeneracy. Models are exported to ONNX and served by a static browser simulator (JavaScript) with a live reference solve and a Gradio application.

**Restrictions and unusual features:** The model is two-dimensional and axisymmetric; it neglects steric effects, electro-osmotic convection, surface chemistry, and pore-wall roughness. Temperature, permittivity, and ionic diffusivities are fixed. The operator is trained on in-distribution ranges of $-50$ to 50 mV, 20--200 mol $m^{-3}$, $q_w\in[-0.05,0]$, and 3--8 nm radius; extrapolation beyond these ranges is not certified. The package includes a dual-engine browser simulator that can switch between the operator and an in-browser SG/FV solve.

**Running time:** Dataset generation 59.2 s; DeepONet training 50.6 s for 500 epochs; FNO training 108.5 s for 110 epochs; single-state inference 0.072 ms (DeepONet) and 1.04 ms (FNO) on an Apple M1 CPU. The Stage-10 studies add 29.6 s to generate the 144-state expansion, 50 s per 500-epoch trunk-ablation model, and 25 s per 250-epoch scaling model. The Stage-11 constraint benchmark (four schemes, three seeds), exact SG/FV Jacobian validation (1200 reference solves), and Stage-12 diagnostics add about four minutes on one CPU.

---

## 1. Introduction

Estimating the surface charge and geometry of a nanopore from its ionic conductance is a central inverse problem in nanofluidic sensing, yet the measurement is fast while the physics that connects it to the parameters is slow to evaluate and ambiguous to invert. Nanopores and nanofluidic channels convert molecular-scale charge and geometry into measurable ionic currents, and they underpin single-molecule sensing and sequencing [1--6]. In confined geometries the electrical double layer (EDL) can overlap, so ion transport departs from bulk conductance [7--11]. Surface charge, salt concentration, pore radius, and applied voltage jointly set selectivity and rectification [12--16], and the PNP system remains the standard continuum description of these effects [17--19]. EDL overlap, charge regulation, and electrokinetic dispersion further modify conduction in narrow pores [20--23], and the same physics underlies emerging iontronic and nanofluidic devices [24, 25].

Estimating surface charge and pore geometry from a current--voltage curve is therefore a long-standing inverse problem [7--16]. Conductance convolves the geometric access resistance with the Donnan enrichment or depletion of ions at the charged wall, so a change in wall charge can mimic a change in radius [12--16, 20--23]. Rectification in asymmetric or conical pores adds a second charge-sensitive signature, but it also mixes charge with geometry [14, 15]. Separating the two from conductance alone remains an open experimental and theoretical challenge, and the usual remedy---fitting many curves with repeated PNP solves---is slow and statistically opaque. Three concrete uses follow for a fast, validated operator: pre-experiment protocol design, in which bias and salinity combinations are chosen to maximize the Fisher information for the parameters of interest; rapid design sweeps of radius, charge, and operating point; and real-time digital-twin calibration, in which millisecond operator evaluations are compared continuously with a measured current. Each use requires not only speed but a calibrated statement of what the measurement can resolve.

Numerical solution of PNP is well understood. Scharfetter--Gummel (SG) discretizations, introduced for semiconductor carrier transport [26], and conservative finite-volume formulations [27] provide stable, accurate currents on moderate grids. A single axisymmetric solve on a 21 x 61 grid takes about 0.2 s on a modern laptop, which is acceptable for one operating point but not for the thousands of evaluations demanded by parameter sweeps, sensitivity analysis, design optimization, or inverse estimation. The cost is compounded when a neural surrogate is trained per state.

Physics-informed neural networks (PINNs) [28--31] remove the need for labeled data but are typically trained for one parameter set at a time, and they are sensitive to gradient-flow pathologies and failure modes [32, 33]. Hard-constraint formulations improve conditioning and enforce boundary data exactly [34--40], and extended or domain-decomposed variants enlarge the admissible geometry [33, 41]. For PNP specifically, enriched and physics-informed variants have been reported [42, 43], operator variants such as CF-DeepONet and T-phPINN have been applied to adjacent transport problems [44, 45], and physics-informed surrogates have been used for battery-health prediction [46]. A previously reported fixed-state discontinuous-Galerkin DA-PINN for nanopore transport (COMPHY-D-26-00857) reached a potential relative L2 error of $3.82\times10^{-4}$ but required roughly 500 s of retraining per state. That trade-off motivates the present work: if the parameter dependence can be learned once, the per-query cost falls by orders of magnitude.

Operator learning provides the required map from parameters to fields. DeepONet approximates nonlinear operators through a branch--trunk inner product [47], with error estimates in infinite dimensions [48], physics-informed DeepONet formulations [49], and several recent architectural and training variants [50--53]. Fourier neural operators (FNOs) learn global convolutions in a spectral basis [54, 55] and have been extended to parametric and physics-informed settings [56--60]. Neural Green's operators offer another route to parametric PDE solution [59]. The central difficulty for transport in charged pores is not representational capacity but constraint satisfaction: potentials must satisfy reservoir Dirichlet data, concentrations must remain positive, and the predicted charge must equal the difference of the predicted concentrations. Unconstrained outputs violate these requirements, and the resulting errors are largest precisely where the net charge is small.

This paper makes two contributions. First, we deliver PINO-Nanopore, a validated computational-physics software instrument: a verified 240-state four-dimensional benchmark (maximum residual $9.936\times10^{-10}$; minimum anion concentration 0.8034), an exact-reservoir charge-balanced parameterization (charge identity $1.110\times10^{-16}$), two parameter-conditioned backbones (multi-scale radial PI-DeepONet, 36,880 parameters; real-DFT FNO, 460,488 parameters) trained with three seeds each, and a browser and Gradio deployment with 0.072 ms ONNX inference. A controlled four-way benchmark shows that the accuracy gain comes from decoupling the $O(1)$ salt sum from the small net-charge difference, a mechanism that the hyperbolic map and the linear projection share ($0.207\pm0.015$ versus $0.201\pm0.021$ interpolation-test net-charge error), and that the hyperbolic map is preferred because it guarantees $c_\pm>0$ and an exact algebraic charge identity without a Softplus clipping bias; the physics weight is set at its Pareto value $10^{-3}$. A trunk ablation and a nested 384-state study bound the design: the radial basis lowers the wall-gradient error by about 7% without a bulk-accuracy advantage, in-distribution error scales as $N^{-0.55}$, and out-of-distribution error is nearly data-insensitive (exponent 0.13). Second, we use the operator to resolve the identifiability question that motivated it. Its conductance sensitivities agree with exact float64 SG/FV finite-difference Jacobians to $1.5\times10^{-4}$ in $\partial\ln I/\partial\ln R$; the exact Fisher matrix confirms that fixed wall charge is a near-null direction at every operating point ($\kappa(\mathcal{F})>10^{8}$), that single- and multi-salt current--voltage protocols leave it unidentifiable at 1% noise, and that a near-wall Donnan selectivity observable collapses the conditioning by ten orders of magnitude and recovers radius and charge below 0.3%. Finally, a spatial and parametric decomposition of the out-of-distribution error shows that the residual error is largest per unit reference in the wall EDL and at the reservoir transitions, yet it cancels in the cross-sectionally integrated current, which stays below 0.23%.

The remainder of the paper describes the model and reference solver (Section 2), reports accuracy, ablation, timing, deployment, and inversion results (Section 3), and discusses scope and limitations (Section 4).

---

## 2. Methods

### 2.1 Physical model and dimensionless form

We consider a rigid, straight cylindrical pore of radius $R$ and length $L=30$ nm connecting two reservoirs at temperature $T=298.15$ K. The pore is filled with a symmetric binary electrolyte of bulk concentration $C_0$; the cation and anion diffusivities are $D_+=1.96\times10^{-9}$ $m^2$ $s^{-1}$ and $D_-=2.03\times10^{-9}$ $m^2$ $s^{-1}$. The relative permittivity is $\varepsilon_r=78.5$. A uniform surface charge of strength $q_w$ is imposed on the wall, and an axial voltage $V_\mathrm{app}$ is applied between the reservoirs. The unknowns are the electrostatic potential $\psi$, the cation and anion concentrations $c_\pm$, and the net charge density $\rho=c_+-c_-$.

The steady PNP system, comprising the Poisson equation and the continuity statement in Eq. (1) and the Nernst--Planck fluxes in Eq. (2), is

$$
-\nabla\cdot\left(\varepsilon_r\varepsilon_0\nabla\psi\right)=F\left(c_+-c_-\right),
\qquad
\nabla\cdot\mathbf{J}_\pm=0,
$$

$$
\mathbf{J}_\pm=-D_\pm\left(\nabla c_\pm\pm\frac{F}{RT}c_\pm\nabla\psi\right),
$$

where $F$ is Faraday's constant and $R$ the gas constant. At the reservoir faces, $\psi=V_\mathrm{app}$ and $\psi=0$, and $c_\pm=C_0$. On the charged wall, $\varepsilon_r\varepsilon_0\,\partial_r\psi|_\mathrm{wall}=\sigma$ and $\mathbf{J}_\pm\cdot\mathbf{n}=0$; on the axis, symmetry conditions apply. The wall charge strength is $q_w=\sigma R/(\varepsilon_r\varepsilon_0 V_T)$, with $V_T=RT/F$; this sign convention makes a negative $q_w$ produce the negative wall potential and cation enrichment seen in the reference solver.

We non-dimensionalize lengths by $R$ radially and by $L$ axially, concentrations by $C_0$, and potential by $V_T$. The Poisson coupling becomes $\beta=(L/\lambda_D)^2/2$, with $\lambda_D=\sqrt{\varepsilon_r\varepsilon_0 V_T/(F C_0)}$, and the geometric anisotropy is $\alpha_r=(L/R)^2$. The operating point, Eq. (3), is the four-vector

$$
\boldsymbol{\mu}=\left(V_\mathrm{app},\,C_0,\,q_w,\,R\right).
$$

The in-distribution training box is $V_\mathrm{app}\in[-50,50]$ mV, $C_0\in[28, 200]$ mol $m^{-3}$, $q_w\in[-0.05,0]$, and $R\in[3, 8]$ nm. The full dimensionless system, including the SG face fluxes used in the physics loss, is written out in Supplementary Section S3. The generated dataset also contains deliberate out-of-distribution charge/salt and geometry/voltage states, as described below.

### 2.2 Conservative SG/FV reference solver and dataset

The reference solver uses a node-centered axisymmetric finite-volume discretization with Scharfetter--Gummel face fluxes for the Nernst--Planck currents and a conservative Poisson update [26, 27]. The production grid is $21\times61$ (radial $\times$ axial). Each state is first attempted directly; if the Newton/Gummel iteration does not converge, a deterministic continuation ramps $q_w$ at zero voltage and then ramps the voltage at the target charge, carrying the potential forward as the initial guess. A state is accepted only if the residual is below $10^{-9}$, all fields are finite, and both concentrations are strictly positive. The face flux uses the Bernoulli function $B(x)=x/(e^x-1)$: $J_{z,i+1/2}=-[c_{i+1}B(-\Delta\psi_i)-c_iB(\Delta\psi_i)]/\Delta z$, with the radial expression analogous; the Poisson update uses the conservative five-point stencil. The discrete equations are repeated in Supplementary Section S3.

The dataset contains $N=240$ states: 144 training states sampled by Latin hypercube inside the in-distribution box with boundary and corner anchors, 48 interpolation-test states, and 48 out-of-distribution states (24 with charge/salt outside the training box, 24 with geometry/voltage outside the box). The exact split composition, sampled ranges, and per-split counts are listed in Supplementary Table S1. Dataset construction took 59.18 s in total. The maximum residual over all states is $9.936\times10^{-10}$, the minimum cation concentration is 0.9999, and the minimum anion concentration is 0.8034; the full validation summary is tabulated in Supplementary Table S2. No state required continuation. The median SG/FV wall time is 0.2013 s and the mean is 0.2391 s.

The benchmark state (index 144, interpolation split) is $V_\mathrm{app}=50$ mV, $C_0=100$ mol $m^{-3}$, $q_w=-0.01$, and $R=5$ nm. Its reference mean axial current is 196.1 pA. This state is used for the field and current comparisons in Section 3.2.

### 2.3 Exact-reservoir and charge-balanced parameterization

The operator does not output $c_+$ and $c_-$ directly. It predicts three smooth latent fields $G_\psi$, $G_s$, and $G_\rho$ on the fixed grid, where $z\in[0, 1]$ is the normalized axial coordinate. The potential, Eq. (4), is

$$
\psi_\theta(r,z)=V_\mathrm{app}(1-z)+z(1-z)\,G_\psi(r,z),
$$

so that $\psi_\theta(r,0)=V_\mathrm{app}$ and $\psi_\theta(r,1)=0$ hold exactly for any $G_\psi$. The concentration sum and net charge, Eq. (5), are

$$
s_\theta(r,z)=2\exp\!\big[z(1-z)\,G_s(r,z)\big],\qquad
\rho_\theta(r,z)=z(1-z)\,G_\rho(r,z),
$$

with the exponent clipped to $[-15,15]$ for numerical safety. At both reservoirs the envelope $z(1-z)$ vanishes, so $s_\theta\to2$ (i.e., $2C_0$ in dimensional units) and $\rho_\theta\to0$, which gives $c_{\pm,\theta}\to1$ (i.e., $C_0$) exactly. Physical concentrations are recovered from the hyperbolic identity, Eq. (6),

$$
c_{\pm,\theta}=\frac{1}{2}\left(\sqrt{s_\theta^2+\rho_\theta^2}\pm\rho_\theta\right).
$$

Because $\sqrt{s^2+\rho^2}\ge|\rho|$, both concentrations are non-negative for any real $s$ and $\rho$, and the algebraic identity $c_{+,\theta}-c_{-,\theta}=\rho_\theta$ holds exactly. In the implementation the residual of this identity is $1.110\times10^{-16}$, i.e., machine precision. The same parameterization is applied to both backbones, so architecture comparisons are not confounded by different constraint handling.

This construction should be distinguished from soft penalty-based constraint enforcement [34--40]. No boundary or positivity penalty is required; the constraints are structural. The physics loss is used only to penalize violations of the differential equations in the interior.

### 2.4 Operator backbones

**Multi-scale radial PI-DeepONet.** The branch network maps the four parameters to $3p$ latent coefficients. The trunk network maps the coordinates and four fixed radial boundary-layer features, $(r,z,\{e^{-\kappa(1-r)}\}_{\kappa\in\{1,3,10,30\}})$, to $p$ basis functions; the multi-scale exponential features resolve the thin EDL near $r=1$ without increasing the trunk width. The two latent vectors are combined by an inner product, and the resulting three fields are passed through the exact-boundary and charge-balanced reconstruction. The model has $p=64$ and 36,880 trainable parameters. The radial basis is a design choice rather than a requirement: Section 3.3 and Supplementary Table S3 compare it against a plain MLP trunk and single-scale radial variants at matched capacity and training budget, using the configuration key `use_radial_basis`/`radial_scales` in `scripts/operators/deeponet.py`.

**Parameter-conditioned two-dimensional FNO.** The FNO lifts the coordinate pair and the four parameters to a width of 24 channels, applies four spectral convolution blocks with $8\times12$ modes and FiLM parameter conditioning, and projects to three fields. The spectral convolutions use a real discrete Fourier transform so that the exported graph contains no complex-valued operators. The model has 460,488 trainable parameters.

Both backbones are trained with the same data and physics objective. The data term is a per-state normalized mean-squared error on the three latent fields, with the scale of each state floored at 10% of the global RMS. The physics term is the sum of the mean absolute Poisson residual, the mean absolute continuity residual for each ion, and the mean absolute wall potential and zero-flux residuals (Supplementary Section S2). The total loss, Eq. (7), is

$$
\mathcal{L}=\mathcal{L}_\mathrm{data}+10^{-3}\,\mathcal{L}_\mathrm{phys}.
$$

The complete procedure---data generation, constraint construction, joint training, and evaluation---is summarized as Algorithm 1 in Supplementary Section S2, together with pinned dependency versions and wall-clock budgets.

### 2.5 Training, splits, and metrics

Models are trained with seeds 42, 43, and 44. The DeepONet is trained for 500 epochs and the FNO for 110 epochs, with batch size 36, learning rate $2\times10^{-3}$, FNO width 24, FNO modes $8\times12$, and DeepONet inner dimension 64. Training times for seed 42 are 50.59 s (DeepONet) and 108.5 s (FNO). A separate 300-epoch FNO is trained to test whether the 110-epoch budget is the limiting factor. The complete optimizer, schedule, architecture, and seed settings are tabulated in Supplementary Table S4.

Accuracy is reported as the global relative $L_2$ error, Eq. (8),

$$
\epsilon_\mathrm{rel}=\frac{\lVert f_\theta-f_\mathrm{ref}\rVert_2}{\lVert f_\mathrm{ref}\rVert_2}
$$

for $\psi$, $c_+$, $c_-$, $\rho$, and the axial current $I(z)$, together with the mean absolute current error in pA. The current is computed from the predicted fields with the same Scharfetter--Gummel face-flux formula used by the reference solver, so current errors include both field errors and post-processing consistency. All reported means are averages over the three seeds, and spreads are reported as standard deviations in Supplementary Table S5 and in the per-state error distributions presented in Section 3.4. Extended accuracy metrics and residual diagnostics are collected in Supplementary Table S6. A cross-reference index (Supplementary Table S7) maps every figure, table, and archived artifact to its source file. The manuscript reports 3--4 significant figures in the narrative and in the supplementary tables; full machine-precision values are archived in `results/statistics.json` and `results/tables/*.csv`.

### 2.6 Deployment and inverse recovery

The trained DeepONet and FNO are exported to ONNX. Parity is checked against PyTorch on 96 states; the maximum absolute difference is $3.576\times10^{-7}$. The DeepONet is served (i) in a static browser simulator with a JavaScript evaluator and an optional in-browser SG/FV solve and (ii) in a Gradio application with operator-versus-reference comparison. Supplementary Table S8 lists the deployment and parity checks, including the browser-bundle size, the JavaScript-versus-reference agreement, and the DOM test results.

For the inverse demonstration, five held-out interpolation states are selected to span the charge and radius ranges. At each state, a five-bias current--voltage curve is generated with the exact SG/FV solver at $V_\mathrm{app}=-50,-25,0,25,50$ mV, and 1% Gaussian noise is added. The operator then evaluates 400 candidate $(q_w,R)$ pairs at the same five biases (2000 forward evaluations) and refines the best candidates with Nelder--Mead. Radius, surface charge density, and their relative errors are recorded. The flat direction of the objective in $q_w$ is quantified by profiling the current mismatch over the full charge prior at the recovered radius. The 1% noise level is a deliberately conservative proxy for a clean nanopore measurement, and the grid search, rather than the local Nelder--Mead refinement, determines the reported flat direction. The identifiability conclusion is expected to weaken for asymmetric electrolytes, unequal diffusivities, or a much wider charge range; these cases are not covered by the present benchmark.

To separate a measurement limitation from a surrogate limitation, the inverse study is repeated as a controlled identifiability experiment over 15 held-out states and three measurement protocols. Protocol 1 (single-salt) uses one bulk concentration; Protocol 2 (multi-salt) uses three concentrations spanning 10--100 mol $m^{-3}$ to probe the EDL-overlap regime; Protocol 3 (dual) adds a near-wall Donnan selectivity observable $S=(c_+-c_-)/(c_++c_-)$ evaluated at $r=1$ to the multi-salt current--voltage data. For each protocol and each relative noise level (0.2%, 1%, 2%) the operator inverts a $30\times30$ candidate grid in $(q_w,R)$ followed by Nelder--Mead refinement, and the median relative errors over the 15 states are reported. In parallel, a $18\times18$ Fisher-information sweep over $(\lambda_D/R,|q_w|)$ at $R=5$ nm computes the dimensionless sensitivity Jacobian $J=\partial\ln|I|/\partial\ln(R,q_w)$ and the condition number $\kappa(\mathcal{F})$ of $\mathcal{F}=J^\top J$; the non-zero bias points are used because the $V_\mathrm{app}=0$ point carries no current information. The results are reported in Section 3.9 and Supplementary Table S9, and visualized in the identifiability and inverse-recovery panels of Figures 3 and 6.

---

## 3. Results

### 3.1 Reference dataset and solver validation

All 240 states converged directly, with a maximum residual of $9.936\times10^{-10}$ and strictly positive concentrations (Supplementary Table S2). The smallest anion concentration, 0.8034, occurs in the strongest-overlap states, where the EDL occupies a large fraction of the pore cross-section. The benchmark current is 196.1 pA, and the SG/FV median and mean wall times are 0.2013 s and 0.2391 s. These values define the cost baseline for the amortization analysis in Section 3.7.

Figure 1 summarizes the problem. Panel A shows the axisymmetric geometry, reservoirs, wall charge, and the PNP equations. Panel B places all 240 states in the $(\lambda_D/R,|q_w|)$ plane, colored by split, with the benchmark state marked; the in-distribution band and the training charge limit are indicated. Panel C shows the exact-reservoir and charge-balanced construction. Panel D compares the two backbones, including parameter counts and ONNX latencies.

![Figure 1](figures/fig1_workflow_and_architecture.png)

**Figure 1.** Problem definition and architecture. (A) Axisymmetric charged nanopore of radius $R$ and length $L$, with fixed wall charge $q_w$, reservoirs at $z=0$ and $z=1$, and applied bias $V_\mathrm{app}$. (B) The 240-state operating space in the four parameters $(V_\mathrm{app},C_0,q_w,R)$, showing the 144 training states, 48 interpolation-test states, and 48 out-of-distribution states. (C) Hard constraints imposed by the ansatz: exact reservoir potential and concentration values, zero wall ion flux, wall charge condition, and positive concentrations. (D) The two parameter-conditioned backbones: a DeepONet with a multi-scale radial trunk and a 2D Fourier neural operator, both trained with the same data and physics objective.

### 3.2 Field reconstruction and selectivity observable validation

Figure 2 resolves the held-out benchmark state (index 144) panel by panel. The operator reconstructs the potential $\psi_\theta$ (Figure 2A) and cation concentration $c_+^\theta$ (Figure 2B) smoothly and symmetrically about the axis, with the potential dropping along the pore and the cation layer accumulating against the negatively charged wall. The net charge $\rho_\theta$ (Figure 2C) is depleted in the core and enriched in the wall layer. The absolute errors in potential and net charge (Figure 2D,E) remain localized to the reservoir transitions and the wall layer rather than accumulating in the interior, and the axial current profile (Figure 2F) follows the reference to within the line width over most of the pore. A full comparison with a state-specialized DA-PINN, which requires roughly 500 s of retraining per state, is archived in Supplementary Table S10 for context; on this unseen state the operator reaches 0.0001485 potential and 0.0005587 current relative $L_2$ error.

The same fields carry the observable used for identifiability. At the 15 inversion operating points and three salt concentrations, the operator reproduces the exact SG/FV near-wall Donnan selectivity $S=(c_+-c_-)/(c_++c_-)|_{r=1}$ with a median absolute error of $1.2\times10^{-4}$ (1.9% relative, rising only where $|S|\to0$), the wall concentrations $c_{\pm,\theta}|_{r=1}$ to 0.02--0.03% (1.1% worst case), and the full $c_\pm$ fields to better than 0.2% relative $L_2$; the log-sensitivity $\partial S/\partial\ln|q_w|$ agrees to 3%. These numbers validate the concentration field used for protocol design, not only the current.

![Figure 2](figures/fig2_benchmark_and_field_reconstruction.png)

**Figure 2.** Benchmark state (index 144, a held-out interpolation-test state; $V_\mathrm{app}=50$ mV, $C_0=100$ mol $m^{-3}$, $q_w=-0.01$, $R=5$ nm). (A) DeepONet potential $\psi_\theta(r,z)$. (B) DeepONet cation concentration $c_+^\theta(r,z)$. (C) DeepONet net charge $\rho_\theta=c_+^\theta-c_-^\theta$. (D) Absolute potential error against the SG/FV reference. (E) Absolute net-charge error. (F) Axial current profile $I(z)$ for the SG/FV reference, the DeepONet, the FNO, and the prior fixed-state DA-PINN (band shows its reported error). All fields are normalized by their reference scales and plotted on the physical grid.

### 3.3 In-distribution accuracy and the value of charge reparameterization

Averaged over three seeds and 48 interpolation states, the DeepONet attains relative $L_2$ errors of 0.0008914 for $\psi$, 0.0007275 for $c_+$, 0.08248 for $\rho$, and 0.001093 for $I$ (mean absolute current error 0.08031 pA). The FNO attains 0.005495, 0.001641, 0.0931, and 0.008511 (0.6746 pA), so the DeepONet is 6.16-fold, 2.26-fold, 1.13-fold and 7.79-fold more accurate, respectively.

The net-charge metric is the most sensitive to the output parameterization. Direct concentration outputs give interp-test $\rho$ errors of 0.7116 (DeepONet) and 0.9578 (FNO); the charge-balanced parameterization reduces them to 0.08248 and 0.0931, i.e. 8.63-fold and 10.29-fold. The reason is subtractive cancellation: independent $c_+$ and $c_-$ errors are amplified in their small difference, whereas predicting $s$ and $\rho$ removes that cancellation. The four-way benchmark (Figure 3A,B; Supplementary Table S11) shows that this decoupling, shared by the algebraic projection and the hyperbolic map, is the dominant driver: both reduce the error about 4.5-fold relative to direct outputs, and their mutual difference ($0.201\pm0.021$ versus $0.207\pm0.015$) is within seed noise. Because they are statistically indistinguishable, the choice is structural. The linear projection leaves a charge-identity mismatch of $4.2\times10^{-8}$ and its raw reconstruction can become negative when co-ion exclusion is strong (22% of a physical latent grid would be clipped), with a Softplus bias $O(\beta^{-1}\ln2)=0.035$ near $c_-\to0$; the hyperbolic map is parameter-free, exact to machine precision ($1.1\times10^{-16}$) and guarantees $c_+c_-=(s/2)^2>0$ at the same accuracy. Figure 3C,D examines the physics-loss and physics-weight ablations (Supplementary Table S12); $\lambda_\mathrm{phys}=10^{-3}$ is the Pareto point. Panels E,F report the trunk and Fisher studies discussed in Sections 3.6 and 3.9.

![Figure 3](figures/fig3_h3_charge_reparameterization_ablation.png)

**Figure 3.** Constraint-scheme benchmark, physics-weight sweep, physics-loss and trunk ablation, and Fisher conditioning. (A) Interpolation-test (solid) and OOD (hatched) net-charge relative $L_2$ error for four constraint schemes on the identical PI-DeepONet backbone; bars are three-seed mean $\pm$ SD. (B) Charge-identity mismatch $|\rho-\rho_\mathrm{latent}|$ and wall-gradient error for the four schemes; the hyperbolic reconstruction and the direct scheme are exact by construction, whereas the linear projection leaves a finite mismatch. (C) Physics-loss ablation on the interpolation-test split for the wall-gradient and interior Poisson residuals. (D) Physics-weight sensitivity at seed 42: net-charge error (left axis) and wall-gradient and Poisson residuals (right axis) versus $\lambda_\mathrm{phys}$. (E) Trunk ablation at matched capacity and 500 epochs: interp-test net-charge and wall-gradient errors of the plain MLP trunk and five radial variants, normalized to the MLP trunk (dashed line). (F) Fisher condition number $\log_{10}\kappa(\mathcal{F})$ over the $(\lambda_D/R,|q_w|)$ plane with white contours at $\kappa=10^{9,10,11}$; the current is insensitive to wall charge at every operating point.

A 300-epoch FNO lowers the interp-test potential error to 0.002195 and the current error to 0.003349 but does not change the ranking (Supplementary Table S13).

### 3.4 Out-of-distribution behavior

The OOD split combines charge/salt and geometry/voltage extrapolation. For the DeepONet the relative $L_2$ errors are 0.02024 for $\psi$, 0.0213 for $c_+$, 0.3678 for $\rho$, and 0.001363 for $I$ (0.1077 pA); for the FNO they are 0.02079, 0.01487, 0.2241, and 0.009123 (0.5338 pA). The two are indistinguishable for potential, while the FNO is better for cation concentration (1.43-fold) and net charge (1.64-fold) and the DeepONet is better for current (6.69-fold). Current is far more robust than the pointwise fields because it integrates the flux and benefits from error cancellation; field-level OOD predictions should therefore be interpreted more cautiously than current--voltage predictions.

Figure 4A shows the per-state error distributions, Figure 4B resolves the net-charge and current errors against $\lambda_D/R$ with a lower spatial budget, and Figure 4C gives the current parity over all 96 test states (slope unity, $R^2=1$ to four decimals, no systematic residual trend). Figure 4B (lower) and Supplementary Table S14 decompose the OOD squared error. Of the total potential error, 51% lies in the pore core, 39% in the wall EDL, and 10% at the reservoir transitions; net charge follows the same pattern (52/39/8%). Per unit reference, however, the wall EDL carries the largest potential error (2.4% versus 2.25% in the core and 1.1% at the transitions) and the transitions carry the largest net-charge error (40%). Combined charge and low-salt extrapolation accounts for 94% of the squared error for both fields, charge-only for about 6%, and narrow radius for 0.2--2%; high-voltage states alone are benign. The current survives because the corner and wall errors cancel: the area-weighted cross-section mean of the potential error is only 1.3% of its RMS, and current continuity integrates the residual flux to 0.12--0.23%.

Figure 4D adds a data-scaling study on the 384-state dataset (Supplementary Table S3). Nested subsets $N_\mathrm{train}=36,72,144,288$ reduce interp-test net-charge error from 0.3927 to 0.1299 and potential error from 0.0054 to 0.0020, fitting $\epsilon_\rho=3.14\,N_\mathrm{train}^{-0.552}$ and $\epsilon_\psi=0.0391\,N_\mathrm{train}^{-0.535}$; the out-of-distribution net-charge error changes only from 0.317 to 0.253 ($\alpha=0.13$). The training box, not the sample count, is the binding constraint on extrapolation.

![Figure 4](figures/fig4_h1_h2_interp_vs_ood_regimes.png)

**Figure 4.** In-distribution versus out-of-distribution behavior and data scaling. (A) Per-state relative $L_2$ error distributions for potential, net charge, and current for the DeepONet and FNO on the interpolation-test and OOD splits; boxes show the median and interquartile range and whiskers extend to the full state range. (B) Top: error versus EDL overlap parameter for net charge (filled symbols) and axial current (open triangles). Bottom: share of the total OOD squared error in the reservoir transitions, wall EDL, and pore core for $\psi$ and $\rho$, for both backbones; the annotation gives the mutually exclusive parametric shares. (C) Current parity for the 96 test states, with the residual (prediction minus reference, in pA) shown in the lower subpanel; the dashed line is the identity. (D) Data-scaling law on the 384-state dataset: interp-test and OOD net-charge error and interp-test potential error versus $N_\mathrm{train}$, with fitted power laws and an $N^{-1/2}$ reference.

### 3.5 Continuous parametric control

Figure 5 tests the operator as a function rather than a lookup table. Figure 5A sweeps the applied voltage from $-50$ to 50 mV at $C_0=100$ mol $m^{-3}$ and $q_w=-0.01$ for three radii and overlays SG/FV solves; the curves overlap across the full range and order monotonically with radius. Figure 5B shows the near-wall cation concentration versus bulk salt from 10 to 200 mol $m^{-3}$, and Figure 5C the radial net-charge profile at $C_0=20$, 100, and 200 mol $m^{-3}$, moving from overlap-dominated to wall-localized as salinity increases. Every curve is an interpolation at a parameter combination that is not a training state.

![Figure 5](figures/fig5_parametric_physics_sweeps.png)

**Figure 5.** Continuous parametric control. (A) Current--voltage curves from $-50$ to 50 mV at $C_0=100$ mol $m^{-3}$ and $q_w=-0.01$ for $R=3$, 5, and 7 nm; symbols are SG/FV solves at five biases and lines are operator predictions. (B) Near-wall cation concentration versus bulk concentration from 10 to 200 mol $m^{-3}$ at $R=5$ nm and $q_w=-0.01$. (C) Radial net-charge profiles at $C_0=20$, 100, and 200 mol $m^{-3}$ for $R=5$ nm, showing the transition from EDL overlap to wall-localized charge. These parameter values are interpolations, not training states.

### 3.6 Charge reparameterization as a structural fix

The ablation in Figure 3 and Supplementary Table S5 isolates the output parameterization. The reparameterized models remove the heavy tail of the direct-charge models: in the weakest-charge quartile the direct models approach unit relative error while the reparameterized models remain below 0.15, and the same ordering holds for the FNO. Supplementary Tables S15, S16 give the breakdown by $|q_w|$ quartile and by $\lambda_D/R$. The reconstruction is algebraic, so the improvement costs nothing at inference; the identity $c_+-c_-=\rho$ holds to machine precision and the downstream current post-processing receives a physically consistent charge field by construction.

### 3.7 Timing, amortization, and break-even

Supplementary Table S17 and Figure 6 summarize latency and amortization. On an Apple M1 CPU, the DeepONet exports to ONNX at 0.07165 ms per state, 18.7-fold faster than its PyTorch batch-1 latency of 1.342 ms; the FNO exports at 1.038 ms, 8.19-fold faster than 8.5 ms. At batch 48, the ONNX latencies are 0.7095 ms and 26.86 ms, and the shared current post-processing adds 3.788 ms. Figure 6A presents these batch-1 and batch-48 latencies for both backbones in PyTorch and ONNX on a common axis, and shows that the DeepONet retains a roughly an order-of-magnitude latency advantage at both batch sizes.

The break-even count is the number of parameter states at which the one-time cost of dataset generation and training equals the cost of solving those states with SG/FV. Using the dataset-mean SG/FV time of 0.2391 s, the DeepONet breaks even at 459.2 states and the FNO at 704.6 states. Using the median time of 0.2013 s, the counts are 545.5 and 837.5. On the slower prior Xeon timing of 0.44 s per solve, the counts fall to 249.5 and 382.1. Against per-state DA-PINN retraining at 500 s per state, the break-even is 0.2195 states for the DeepONet and 0.3355 states for the FNO: a single query already amortizes the operator. Figure 6B shows these cumulative-cost curves, plotting the operator cost and the SG/FV cost against the number of forward states and marking the break-even counts; the prior-Xeon and DA-PINN-retrain references are drawn for context.

![Figure 6](figures/fig6_h4_timing_and_amortization_breakeven.png)

**Figure 6.** Latency, amortization, and exact-versus-operator Fisher validation. (A) Batch-1 and batch-48 prediction latency for the DeepONet and FNO in PyTorch and ONNX on an Apple M1 CPU, including the shared current post-processing at batch 48. (B) Cumulative cost versus number of forward states for the operator and for SG/FV solves at the dataset-mean and median solve times, with the break-even counts marked; the prior-Xeon and DA-PINN-retrain reference lines are included for context. (C) Exact SG/FV finite-difference sensitivities against the neural-operator sensitivities for 15 held-out states: $\partial\ln I/\partial\ln R$ (circles) and the absolute charge sensitivity $|\partial\ln I/\partial\ln q_w|$ (squares); the dashed line is the identity. (D) Median Fisher condition number $\kappa(\mathcal{F})$ for the single-salt, multi-salt, and dual protocols, comparing the exact SG/FV Jacobian with the operator; both collapse by ten orders of magnitude once the near-wall selectivity observable is added.

These counts are conservative in one respect and optimistic in another: they charge the full dataset-generation and training cost to the operator but not the engineering time of building it, and they assume that the SG/FV baseline is run serially. The break-even count is only about 1.9 times the 240-state training set, so the operator is most advantageous for sweeps of thousands of states; against per-state DA-PINN retraining, a single query already amortizes it.

### 3.8 Physics loss and data-only ablation

Removing the physics term degrades the current more than the fields. For the DeepONet, the interp-test current error rises from 0.001093 to 0.001687 (35.2% higher), while the potential error changes only from 0.0008914 to 0.000928 (3.95% higher). For the FNO, the interp-test current error rises from 0.008511 to 0.01393 (38.9% higher), and the OOD current error rises from 0.009123 to 0.01815 (49.7% higher). The OOD potential error of the FNO also rises from 0.02079 to 0.03013 (31.0% higher). Figure 3C summarizes this ablation through the near-wall derivative residual $|\partial_r\psi|_{r=1}-q_w|$ and the interior Poisson residual on the interpolation split, where the data-only bars are the ones that move most.

The interpretation is that the physics residual constrains the flux balance, which the current integral senses even when the pointwise fields are visually similar. The data-only DeepONet is competitive for in-distribution potential and cation concentration, but its current error is 54.3% higher, and its OOD current error is 30.5% higher. The physics term is therefore not cosmetic; it is the main driver of current accuracy in the sparse-data regime.

### 3.9 Conductance identifiability limits, Fisher spectrum, and experimental protocol design

The inverse demonstration asks what can be inferred from a noisy current--voltage curve alone. For five held-out states the operator recovers the pore radius with a median relative error of 0.004574 (0.46%), because the current scales with the pore cross-section and the five-bias slope fixes the conductance scale. Surface charge is not identifiable at the same noise. Across the full charge prior, the median state-level change in predicted current at the recovered radius is 0.0009625 (0.096%) and the largest is 0.0019 (0.19%), both below the 1% noise floor; the objective is therefore flat in $q_w$. The median relative $q_w$ error is 1.154 (absolute 0.0526, about half the prior width), and the median absolute radius error is 0.0181 nm; Supplementary Table S18 reports relative and per-state values.

This flat direction follows from the structure of the binary-electrolyte flux. The local ohmic drift conductivity is $\sigma=(F^2/RT)(D_+c_++D_-c_-)=(F^2/RT)[\bar D\,(c_++c_-)+\tfrac12(D_+-D_-)(c_+-c_-)]$, with $\bar D=(D_++D_-)/2$. Writing $s=c_++c_-=\sqrt{4c_+c_-+\rho^2}$, a Donnan layer with $c_+c_-\approx1$ has $s=2+\rho^2/4+O(\rho^4)$. Hence for $D_+=D_-$ the charge enters the conductance only at second order in $\rho=O(q_w)$, and $\partial\ln I/\partial\ln|q_w|=O(\rho^2)$ vanishes at zero charge. A finite diffusivity difference adds the linear term $(D_+-D_-)\rho/2$, with relative coefficient $|D_+-D_-|/(D_++D_-)$: 0.0175 for KCl and 0.21 even for NaCl. The first term is the salt sum fixed by bulk concentration and geometry; the second is the only linear charge channel, and it is small.

The multi-protocol experiment confirms this interpretation and shows how to break the degeneracy (Supplementary Table S9). Over 15 held-out states, single-salt current--voltage inversion leaves the median $q_w$ error at 0.77--1.00 at all three noise levels, and the multi-salt protocol improves the low-salinity charge sensitivity roughly tenfold (relative spread of $|I|$ over the charge prior: $1.3\times10^{-3}$ at $C_0=10$ versus $1.5\times10^{-4}$ at $C_0=100$ mol $m^{-3}$) but still leaves the median $q_w$ error at 0.82--1.00, because the gain stays below the noise floor. Both protocols recover radius to better than 0.7%. Adding $S$ recovers $q_w$ with median relative error 0.0005--0.0028 and $R$ with 0.0003--0.0018 across the same noise levels, i.e. both parameters below 0.3%. The exact SG/FV Jacobian validation closes the loop (Figure 6C,D; Supplementary Table S19): the two sensitivities agree on $\partial\ln I/\partial\ln R$ to a median relative difference of $1.5\times10^{-4}$ (both 2.000), the exact single-salt $\kappa$ is $4.9\times10^{15}$ versus $8.2\times10^{8}$ for the operator, and the dual value agrees to within 1% ($1.8\times10^{5}$ versus $1.76\times10^{5}$). Both analyses agree that conductance-only conditioning exceeds $10^{8}$ and that the dual observable collapses it by ten orders of magnitude.

Supplementary Table S20 verifies this picture with the exact float64 SG/FV solver at the benchmark state and at low salt. For ideal symmetric diffusivities, $|\partial\ln I/\partial\ln|q_w||$ is $7.9\times10^{-7}$ at $C_0=100$ mol $m^{-3}$ and the single-salt condition number is $1.7\times10^{17}$. KCl ($D_+/D_-=0.966$) raises the sensitivity only to $1.1\times10^{-5}$ and leaves $\kappa=1.3\times10^{17}$; NaCl (0.655) gives $1.4\times10^{-4}$ and $3.0\times10^{16}$; even the strongly asymmetric ratio 0.20 gives $4.6\times10^{-4}$ and $5.9\times10^{15}$. At $C_0=20$ mol $m^{-3}$ the sensitivities are larger but $\kappa$ stays above $3.6\times10^{12}$. Adding the near-wall selectivity $S$ collapses $\kappa$ to $4.0\times10^{6}$ at $C_0=100$ and $6.3\times10^{5}$ at $C_0=20$ for every ratio, a reduction of six to ten orders of magnitude. Diffusivity asymmetry therefore changes the prefactor of the linear channel but does not remove the identifiability limit at 1% noise; the non-monotonic $\kappa$ (a larger diffusivity difference gives a smaller $\kappa$) reflects the larger charge sensitivity, not a better-conditioned inverse.

These results turn the operator into a protocol-design tool. Before fabricating or measuring a pore, a user can query the live portal (https://pino.statwhy.com) or the Python API to evaluate $J=\partial(\ln|I|,S)/\partial(\ln R,\ln|q_w|)$ and $\kappa(\mathcal F)$ for candidate bias sets, salinities, salinity gradients, and observable combinations, and choose the protocol that minimizes the charge-direction variance subject to experimental constraints. The package exposes the operator Jacobian and the Fisher construction, and the portal reports the same sensitivities interactively; the diffusive-ratio scan of Supplementary Table S20 is reproducible with `scripts/run_diffusivity_symmetry_check.py`. For a conductance-only experiment the calculation predicts a flat charge direction and should redirect effort to a charge-sensitive readout---reversal potential or transference number, streaming potential, ionic current rectification, or field-resolved near-wall sensing---rather than to additional conductance data.

The boundary of the claim is explicit. The near-null result is established for symmetric binary electrolytes with nearly equal ionic diffusivities, such as KCl, and for a conductance-only protocol. Unequal diffusivities add the linear term above but leave $\kappa>10^{15}$ at 1% noise (Supplementary Table S20); asymmetric valence or a non-binary electrolyte changes the flux structure and lies outside the present proof. Within that boundary the limit is a property of the measurement, not of the surrogate or the solver.

### 3.10 Interactive deployment

The exported models run without a Python server: a single ONNX graph serves a static browser simulator (with an optional in-browser SG/FV solve) and a Gradio application, so the same network can be used in teaching and in experimental design without installing a solver. The JavaScript current formula agrees with the SG/FV reference to a maximum relative difference of $5.686\times10^{-7}$ over the eight verification states, the DOM test harness passes 29 of 29 checks, and the Gradio comparison against a live solve reports a potential relative $L_2$ difference of 0.0002126. The dual-engine design keeps the reference solver available for validation rather than replacing it. Interface details, bundle sizes, and the full deployment and parity record are collected in Supplementary Section S8 and Supplementary Table S8.

---

## 4. Discussion

The results support a simple division of labor. The DeepONet is the better default surrogate: it is 6.16-fold more accurate for potential and 7.79-fold more accurate for current than the FNO in-distribution, with 12.5-fold fewer parameters and 14.5-fold lower ONNX latency. The FNO is not redundant, however. Its spectral basis captures the cation and net-charge fields better in the out-of-distribution split (1.43-fold and 1.64-fold, respectively), and its errors grow more gracefully when the Debye ratio leaves the training band. A practical strategy is to use the DeepONet for current-based design and the FNO as a cross-check or as a backbone when field-level OOD accuracy is the priority.

The charge-balanced parameterization is the most transferable result. The improvement is large (8.63-fold for DeepONet and 10.29-fold for FNO), it costs nothing at inference, and it is architecture-agnostic. The underlying mechanism is general: whenever a physical quantity is a small difference of two large positive fields, direct regression suffers from cancellation, and a reparameterization in terms of the sum and the difference removes the ill-conditioning. The same idea should apply to other Nernst--Planck and drift--diffusion systems in which charge or mole fractions are inferred from separate species predictions.

The physics loss plays a complementary role. It does not dramatically improve the pointwise potential in-distribution, but it reduces current error by 35.2% for the DeepONet and 49.7% for the FNO out of distribution. Because current is the quantity that experiments measure, this is the relevant metric, and it justifies retaining the residual term even when labeled data are available.

The inverse experiment sharpens the scope of the method. Radius is identifiable from a five-bias current--voltage curve to 0.46%, and the operator makes the inversion cheap (0.1253 s per curve versus 402.6 s of SG/FV work). Surface charge is not identifiable under the same protocol: the median state-level current change is 0.096% across the full charge prior (largest state-level change 0.19%), both below the 1% noise. This is a physical identifiability limit, not a surrogate error. It follows from the near-invariance of the concentration sum for a symmetric electrolyte with nearly equal diffusivities, and the Fisher analysis shows it holds at every operating point, with a near-null charge direction and condition numbers above $10^{8}$. The multi-protocol study sharpens the remedy: multi-salt current--voltage data increase the charge sensitivity by roughly an order of magnitude at low salinity but remain below the noise floor, and it is the independent near-wall selectivity observable that recovers wall charge to below 0.3%. Closing the gap in a real experiment therefore requires a genuinely charge-sensitive measurement, such as streaming potential, field-resolved sensing, or a near-wall selectivity readout, rather than more current--voltage data at different salinities. The operator is well suited to such protocols because it can evaluate thousands of candidate parameter combinations per second.

The selectivity observable that breaks the degeneracy is not a purely theoretical construct. In a symmetric binary electrolyte the near-wall selectivity is, to leading order, the transference-number imbalance $S\simeq2t_+-1$, so a concentration-cell reversal potential $V_\mathrm{rev}$ (or the streaming potential under a pressure gradient) provides a direct, calibration-based estimate of $t_+$ and hence of $S$ [20--23]. Current-rectification ratios (ICR) in asymmetric or conical pores carry the same charge information and are routinely measured, although they convolve charge with geometry and require the operator to separate the two. Optical routes--- plasmonic nanopores, surface-enhanced Raman scattering, or wall-localized fluorescent ion indicators--- can in principle resolve the near-wall concentration difference directly, at the cost of labeling and photostability constraints. None of these measurements is as routine as a current--voltage sweep, which is precisely why the identifiability analysis matters: it tells experimentalists that a charge-sensitive readout, not more conductance data, is the measurement to invest in, and the operator can be used to design the bias and salinity protocol that maximizes the information gain before the experiment is built.

Two further results bound the method rather than extend it. The trunk ablation shows that the multi-scale radial basis, although physically motivated, is not essential: a plain MLP trunk matches it on the net-charge field and is slightly better in-distribution, and the radial features only improve the wall gradient by about 7%. We retain the radial trunk because the wall gradient and the integrated current are the operationally important outputs, but we report the comparison rather than presenting the radial basis as a necessary ingredient. The scaling study shows that in-distribution accuracy follows the expected $N^{-1/2}$ trend, so more in-distribution data translate predictably into higher accuracy, whereas out-of-distribution error is almost flat in $N_\mathrm{train}$; the training box, not the sample count, sets the extrapolation limit. Together these results indicate that the main lever for improving this class of surrogate is not a more expressive trunk but a wider calibration range and charge-sensitive observables.

Several limitations should be stated plainly. The model is two-dimensional and axisymmetric; it does not resolve azimuthal roughness, pore mouths, or non-circular cross-sections. It neglects steric effects, finite ion size, electro-osmotic convection, and surface chemistry, so it should not be used to interpret pH-dependent or highly concentrated experiments without extension. Temperature, permittivity, and diffusivities are fixed. The training box is modest ($-50$ to 50 mV, 20--200 mol $m^{-3}$, $|q_w|\le0.05$, 3--8 nm), and the OOD split is a controlled extrapolation rather than a stress test at experimental extremes. The reference solver itself is a continuum model; agreement with it demonstrates numerical consistency, not agreement with a particular experiment. Finally, the inverse demonstration uses synthetic noise and a known bulk concentration; in an experiment, concentration and temperature uncertainty would widen the posterior and could break the degeneracy between radius and other geometric parameters.

The deployment results suggest where the work can go next. The browser simulator makes the operator accessible without a solver installation, and the Gradio comparison keeps a reference solve one click away. A natural extension is to add a charge-sensitive observable to the inverse workflow, and to train a second operator that maps current--voltage curves to parameters directly, using the forward operator to generate training pairs. The same dataset and constraint machinery can be reused for that task.

---

## 5. Conclusion

A charge-balanced, exact-reservoir physics-informed operator learns steady axisymmetric PNP transport over a four-dimensional parameter space, and the same operator serves as a computational instrument for identifiability. The DeepONet achieves 0.0008914 potential and 0.001093 current relative $L_2$ error on held-out in-distribution states, remains below 0.23% current error out of distribution, and runs in 0.072 ms per state after ONNX export. The charge-balanced parameterization reduces net-charge error by an order of magnitude relative to direct concentration outputs, and the physics loss reduces current error by 35--50%. A trunk ablation shows that the multi-scale radial basis is a modest near-wall inductive bias rather than a necessity, and a nested 384-state study gives an in-distribution error scaling as $N^{-0.55}$ while out-of-distribution error is nearly data-insensitive. A spatial and parametric decomposition shows that the residual out-of-distribution error is concentrated in the wall EDL and reservoir transitions per unit reference, but cancels in the cross-sectionally integrated current. The operator breaks even against the reference solver after about 460 states at the dataset-mean solve time, and is deployed in a browser simulator and a Gradio application that match PyTorch to within $3.6\times10^{-7}$ maximum absolute difference. The central physical result is a limit, not a speed-up: for symmetric binary electrolytes with nearly equal ionic diffusivities, such as KCl, and a conductance-only protocol, fixed wall charge is fundamentally unidentifiable at 1% noise regardless of surrogate or solver accuracy, as confirmed by the exact SG/FV Fisher matrix ($\kappa(\mathcal{F})>10^{8}$). Single- and multi-salt current--voltage sweeps cannot remove this degeneracy. Differential diffusivity ($D_+\neq D_-$) adds a linear charge channel of relative size $|D_+-D_-|/(D_++D_-)$, but the single-salt conditioning remains above $10^{15}$ even for NaCl and a strongly asymmetric macro-ion electrolyte (Supplementary Table S20). Experimentalists should therefore invest in charge-sensitive observables---a concentration-cell reversal potential or transference number, a streaming potential, or field-resolved near-wall sensing---rather than in additional conductance data, and the operator is intended to design those protocols before the experiment is built.

---

## CRediT authorship contribution statement

**Javad Alizargar:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing -- original draft, Writing -- review and editing, Visualization, Supervision, Project administration.

---

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Data and code availability

The dataset, trained models, figure-generation scripts, and analysis code are available at https://github.com/Javad-Alizargar/nanopore-pino-surrogate under the MIT License. The interactive browser simulator, including a live SG/FV reference solver, is served at https://pino.statwhy.com/. The permanently archived release is available at Zenodo DOI https://doi.org/10.5281/zenodo.23487404. The repository contains the SG/FV reference solver, the 240-state dataset and the 384-state expansion used for the scaling study, the training and evaluation scripts, the Stage-10 ablation, scaling, and identifiability scripts, the Stage-11 constraint and Fisher-validation scripts, the Stage-12 diagnostics, the ONNX models, the browser simulator, and the Gradio application. The raw literature metadata used to build the reference list is included in `manuscript/literature_db.json`.

---

## Generative AI declaration

During the preparation of this work the author used large language model tools for drafting assistance, code generation, and figure-quality review. All numerical results were produced by the local code and verified against the archived result files; all references were checked against Crossref metadata. The author reviewed and edited the content and takes full responsibility for the published article.

---

## Acknowledgements

The author thanks the developers of the open-source scientific Python stack used in this work (NumPy, SciPy, PyTorch, ONNX Runtime, Matplotlib, Gradio) and the maintainers of the Crossref and PubMed metadata services.

---

## References

1. Kasianowicz, John J., Brandin, Eric, Branton, Daniel, et al. (1996). Characterization of individual polynucleotide molecules using a membrane channel *Proceedings of the National Academy of Sciences*. doi:10.1073/pnas.93.24.13770
2. Dekker, Cees (2007). Solid-state nanopores *Nature Nanotechnology*. doi:10.1038/nnano.2007.27
3. Branton, Daniel, Deamer, David W, Marziali, Andre, et al. (2008). The potential and challenges of nanopore sequencing *Nature Biotechnology*. doi:10.1038/nbt.1495
4. Howorka, Stefan, Siwy, Zuzanna (2009). Nanopore analytics: sensing of single molecules *Chemical Society Reviews*. doi:10.1039/b813796j
5. Venkatesan, Bala Murali, Bashir, Rashid (2011). Nanopore sensors for nucleic acid analysis *Nature Nanotechnology*. doi:10.1038/nnano.2011.129
6. Wanunu, Meni (2012). Nanopores: A journey towards DNA sequencing *Physics of Life Reviews*. doi:10.1016/j.plrev.2012.05.010
7. Stein, Derek, Kruithof, Maarten, Dekker, Cees (2004). Surface-Charge-Governed Ion Transport in Nanofluidic Channels *Physical Review Letters*. doi:10.1103/physrevlett.93.035901
8. Schoch, Reto B., Han, Jongyoon, Renaud, Philippe (2008). Transport phenomena in nanofluidics *Reviews of Modern Physics*. doi:10.1103/revmodphys.80.839
9. Bocquet, Lydéric, Charlaix, Elisabeth (2010). Nanofluidics, from bulk to interfaces *Chem. Soc. Rev.*. doi:10.1039/b909366b
10. Daiguji, Hirofumi (2010). Ion transport in nanofluidic channels *Chem. Soc. Rev.*. doi:10.1039/b820556f
11. Tagliazucchi, Mario, Szleifer, Igal (2015). Transport mechanisms in nanopores and nanochannels: can we mimic nature? *Materials Today*. doi:10.1016/j.mattod.2014.10.020
12. Karnik, Rohit, Fan, Rong, Yue, Min, et al. (2005). Electrostatic Control of Ions and Molecules in Nanofluidic Transistors *Nano Letters*. doi:10.1021/nl050493b
13. Smeets, Ralph M. M., Keyser, Ulrich F., Krapf, Diego, et al. (2005). Salt Dependence of Ion Transport and DNA Translocation through Solid-State Nanopores *Nano Letters*. doi:10.1021/nl052107w
14. Kosińska, I. D., Goychuk, I., Kostur, M., et al. (2008). Rectification in synthetic conical nanopores: A one-dimensional Poisson-Nernst-Planck model *Physical Review E*. doi:10.1103/physreve.77.031131
15. Valiskó, Mónika, Matejczyk, Bartłomiej, Ható, Zoltán, et al. (2019). Multiscale analysis of the effect of surface charge pattern on a nanopore’s rectification and selectivity properties: From all-atom model to Poisson-Nernst-Planck *The Journal of Chemical Physics*. doi:10.1063/1.5091789
16. Ma, Pengfei, Zheng, Jianxiang, Zhao, Danting, et al. (2021). The Selective Transport of Ions in Charged Nanopore with Combined Multi-Physics Fields *Materials*. doi:10.3390/ma14227012
17. Coalson, R.D., Kurnikova, M.G. (2005). Poisson–Nernst–Planck Theory Approach to the Calculation of Current Through Biological Ion Channels *IEEE Transactions on Nanobioscience*. doi:10.1109/tnb.2004.842495
18. Eisenberg, Bob, Liu, Weishi (2007). Poisson–Nernst–Planck Systems for Ion Channels with Permanent Charges *SIAM Journal on Mathematical Analysis*. doi:10.1137/060657480
19. Zheng, Qiong, Chen, Duan, Wei, Guo-Wei (2011). Second-order Poisson–Nernst–Planck solver for ion transport *Journal of Computational Physics*. doi:10.1016/j.jcp.2011.03.020
20. Zhuorui Song (2021). Electrokinetic Flow in a Nanochannel with an Overlapped Electrical Double Layer *Digital Commons - USU (Utah State University)*. doi:10.26076/a748-e85a
21. Haosen Xu, Jianbo Zhang, Michael Hermann Eikerling, et al. (2024). Pure Water Splitting Driven by Overlapping Electric Double Layers *Journal of the American Chemical Society*. doi:10.1021/jacs.4c01070
22. Cody L. Ritt, J. Pedro de Souza, Michelle G. Barsukov, et al. (2022). Thermodynamics of Charge Regulation during Ion Transport through Silica Nanochannels *ACS Nano*. doi:10.1021/acsnano.2c06633
23. Baisakhi Pal, Poulami Dutta, Sourav Chowdhury, et al. (2026). Electroosmotic Dispersion of Uncharged Solutes through a Nanochannel *Industrial & Engineering Chemistry Research*. doi:10.1021/acs.iecr.6c00231
24. Tim M. Kamsma, Jaehyun Kim, Kyungjun Kim, et al. (2024). Brain-inspired computing with fluidic iontronic nanochannels *Proceedings of the National Academy of Sciences*. doi:10.1073/pnas.2320242121
25. Olga I. Vinogradova, Elena F. Silkina (2023). Electrophoresis of ions and electrolyte conductivity: From bulk to nanochannels *The Journal of Chemical Physics*. doi:10.1063/5.0168557
26. Scharfetter, D.L., Gummel, H.K. (1969). Large-signal analysis of a silicon Read diode oscillator *IEEE Transactions on Electron Devices*. doi:10.1109/t-ed.1969.16566
27. Miller, J. J. H., Wang, Song (1994). An analysis of the Scharfetter-Gummel box method for the stationary semiconductor device equations *ESAIM: Mathematical Modelling and Numerical Analysis*. doi:10.1051/m2an/1994280201231
28. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations *Journal of Computational Physics*. doi:10.1016/j.jcp.2018.10.045
29. Karniadakis, George Em, Kevrekidis, Ioannis G., Lu, Lu, et al. (2021). Physics-informed machine learning *Nature Reviews Physics*. doi:10.1038/s42254-021-00314-5
30. Lu, Lu, Meng, Xuhui, Mao, Zhiping, et al. (2021). DeepXDE: A Deep Learning Library for Solving Differential Equations *SIAM Review*. doi:10.1137/19m1274067
31. Lagaris, I.E., Likas, A., Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations *IEEE Transactions on Neural Networks*. doi:10.1109/72.712178
32. Wang, Sifan, Teng, Yujun, Perdikaris, Paris (2021). Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks *SIAM Journal on Scientific Computing*. doi:10.1137/20m1318043
33. Krishnapriyan, Aditi S., Gholami, Amir, Zhe, Shandian, et al. (2021). Characterizing possible failure modes in physics-informed neural networks *Advances in Neural Information Processing Systems*. doi:10.48550/arxiv.2109.01050
34. Sukumar, N., Srivastava, Ankit (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2021.114333
35. Shaikhah Alkhadhr, Mohamed Khaled Almekkawy (2023). Wave Equation Modeling via Physics-Informed Neural Networks: Models of Soft and Hard Constraints for Initial and Boundary Conditions *Sensors*. doi:10.3390/s23052792
36. Gang Jing, Huan Wang, Yuan Jiu, et al. (2026). A unified hard-constraint physics-informed neural network approach for sparse-observation airflow-field reconstruction under complex boundary conditions *Engineering Applications of Artificial Intelligence*. doi:10.1016/j.engappai.2026.116169
37. Mohammad Sheikholeslami, Saeed Salehi, Wengang Mao, et al. (2025). Physics-informed neural networks with hard and soft boundary conditions for linear free surface waves *Physics of Fluids*. doi:10.1063/5.0277421
38. Peng Lan, Jing-jing Su, Xin-yan Ma, et al. (2023). Application of improved physics-informed neural networks for nonlinear consolidation problems with continuous drainage boundary conditions *Acta Geotechnica*. doi:10.1007/s11440-023-01899-0
39. Stefano Berrone, Claudio Canuto, Moreno Pintore, et al. (2023). Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks *Heliyon*. doi:10.1016/j.heliyon.2023.e18820
40. Xinquan Huang, Tariq A. Alkhalifah (2024). Microseismic Source Imaging Using Physics-Informed Neural Networks With Hard Constraints *IEEE Transactions on Geoscience and Remote Sensing*. doi:10.1109/tgrs.2024.3366449
41. D. Jagtap, Ameya, Em Karniadakis, George (2020). Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations *Communications in Computational Physics*. doi:10.4208/cicp.oa-2020-0164
42. Huang, Xujia, Wang, Fajie, Zhang, Benrong, et al. (2025). Enriched physics-informed neural networks for dynamic Poisson-Nernst-Planck systems *Mathematics and Computers in Simulation*. doi:10.1016/j.matcom.2025.04.037
43. Zheng, Jinglai, Huang, Jie, Huang, Haiming (2025). Physics-informed neural networks for solving thermal response of charring materials *Acta Astronautica*. doi:10.1016/j.actaastro.2025.07.010
44. Zheng, Jinglai, Hu, Hanying, Huang, Jie, et al. (2025). CF-DeepONet: Deep operator neural networks for solving compressible flows *Aerospace Science and Technology*. doi:10.1016/j.ast.2025.110329
45. Zheng, Jinglai, Li, Fan, Huang, Haiming (2024). T-phPINN: Physics-informed neural networks for solving 2D non-Fourier heat conduction equations *International Journal of Heat and Mass Transfer*. doi:10.1016/j.ijheatmasstransfer.2024.126216
46. Wei, Congxin, Pang, Haikuo, Huang, Teng, et al. (2026). Predicting lithium-ion battery health using attention mechanism with Kolmogorov-Arnold and physics-informed neural networks *Expert Systems with Applications*. doi:10.1016/j.eswa.2025.128969
47. Lu Lu, Pengzhan Jin, Guofei Pang, et al. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators *Nature Machine Intelligence*. doi:10.1038/s42256-021-00302-5
48. Samuel Lanthaler, Siddhartha Mishra, George Em Karniadakis (2022). Error estimates for DeepONets: a deep learning framework in infinite dimensions *Transactions of Mathematics and Its Applications*. doi:10.1093/imatrm/tnac001
49. Sifan Wang, Hanwen Wang, Paris G. Perdikaris (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets *Science Advances*. doi:10.1126/sciadv.abi8605
50. Hwijae Son (2025). ELM-DeepONets: Backpropagation-Free Training of Deep Operator Networks via Extreme Learning Machines *IEEE Access*. doi:10.1109/access.2025.3570502
51. Qiuying Ma, Xinghan Jiang, Xinlong Guo, et al. (2025). Layer-Interaction DeepONet for Modeling Ultrafast Nonlinear Dynamics in Optical Fibers *Journal of Lightwave Technology*. doi:10.1109/jlt.2025.3598027
52. Samuel Lanthaler, Zongyi Li, Andrew M. Stuart (2025). Nonlocality and Nonlinearity Implies Universality in Operator Learning *Constructive Approximation*. doi:10.1007/s00365-025-09718-3
53. Yongjin Choi, Chenying Liu, Jorge Macedo (2026). Operator learning for consolidation: An architectural comparison for DeepONet variants *Computers and Geotechnics*. doi:10.1016/j.compgeo.2026.108017
54. Hailong Sheng, Yulin Xie, Chao Yang (2025). XFNO: Extended Fourier neural operator for solving parametric partial differential equations on general geometries *Computer Physics Communications*. doi:10.1016/j.cpc.2025.109774
55. Kai Qi, Jian Sun (2024). Gabor-Filtered Fourier Neural Operator for solving Partial Differential Equations *Computers & Fluids*. doi:10.1016/j.compfluid.2024.106239
56. Kangjie Li, Wenjing Ye (2025). D-FNO: A decomposed Fourier neural operator for large-scale parametric partial differential equations *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2025.117732
57. Zhang, Tao, Xiao, Hui, Ghosh, Debdulal (2025). Physics-informed fourier neural operators: A machine learning method for parametric partial differential equations *Journal of Nonlinear and Variational Analysis* 9(1), 45--64. doi:10.23952/jnva.9.2025.1.04
58. Zongyi Li, Hongkai Zheng, Nikola Kovachki, et al. (2024). Physics-Informed Neural Operator for Learning Partial Differential Equations *ACM / IMS Journal of Data Science*. doi:10.1145/3648506
59. Hugo Melchers, Joost Prins, Michael Abdelmalik (2026). Neural Green’s operators for parametric partial differential equations *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2026.118893
60. Muhammad Tahir Rafiq, Ghazala Rafiq, Ho-Youl Jung, et al. (2022). SSNO: Spatio-Spectral Neural Operator for Functional Space Learning of Partial Differential Equations *IEEE Access*. doi:10.1109/access.2022.3148401
