# A parameter-conditioned physics-informed neural operator for axisymmetric Poisson--Nernst--Planck transport in charged nanopores

**Javad Alizargar**  
Department of Nursing, National Taipei University of Nursing and Health Sciences, No. 365, Mingde Rd, Beitou District, Taipei City 112, Taiwan  
Corresponding author: 8javad@ntunhs.edu.tw | ORCID: 0000-0003-2966-1315

---

## Highlights

- A physics-informed DeepONet maps four PNP parameters to fields in 0.072 ms.
- Charge-balanced reparameterization cuts net-charge error 8.63--10.29-fold.
- Break-even versus Scharfetter--Gummel solving is 459 parameter states.
- ONNX matches PyTorch within 3.6e-7 in browser and Gradio apps.
- Current-only inversion recovers pore radius to 0.46% but not surface charge.

---

## Abstract

Physics-informed neural networks (PINNs) solve the Poisson--Nernst--Planck (PNP) equations one operating point at a time, so design sweeps and inverse problems require repeated retraining. We instead learn the solution operator of the steady axisymmetric PNP equations over a four-dimensional parameter space of applied voltage, bulk concentration, wall charge, and pore radius. A 240-state benchmark is generated with a conservative Scharfetter--Gummel finite-volume solver (maximum residual 9.936e-10). Two parameter-conditioned backbones are compared: a multi-scale radial physics-informed DeepONet (36,880 parameters) and a four-layer real-DFT Fourier neural operator (460,488 parameters). Both use a charge-balanced reparameterization in which a positive concentration sum and a net charge are predicted, and physical concentrations are recovered hyperbolically; reservoir Dirichlet data and charge consistency are imposed exactly. On held-out in-distribution states, the DeepONet attains a mean relative L2 error of 0.0008914 for potential and 0.001093 for current, improving on the FNO by 6.16-fold and 7.79-fold, respectively. Under out-of-distribution conditions the current error remains 0.001363, although potential errors grow 22.7-fold. Replacing the direct concentration output with the charge-balanced parameterization reduces net-charge error 8.63-fold (DeepONet) and 10.29-fold (FNO). The exported ONNX model runs in 0.07165 ms per state, breaks even against the reference solver after 459.2 states at the dataset-mean solve time, and is deployed in a browser simulator and a Gradio interface. A noisy five-bias inversion recovers pore radius to 0.46% but reveals that fixed surface charge is a flat direction for conductance-only inversion.

**Keywords:** Poisson--Nernst--Planck; nanopore; neural operator; DeepONet; Fourier neural operator; physics-informed machine learning; inverse problem

---

## CPC Program Summary

**Program title:** PINO-Nanopore

**Software name:** `nanopore-pino-surrogate`

**Program obtainable from:** https://github.com/Javad-Alizargar/nanopore-pino-surrogate

**CPC Library link to program files:** added by the CPC Technical Editor upon acceptance

**Licensing provisions:** MIT License (Copyright 2026 Javad Alizargar)

**Distribution format:** Source code (Python and JavaScript), ONNX models, CSV/JSON data, and documentation

**Programming language:** Python 3.14; JavaScript ES6

**Computer:** Apple M1 (arm64); any x86-64 or arm64 machine with Python 3.10+ and a modern browser

**Operating system:** macOS 14.1 (tested); Linux and Windows are supported by the dependencies

**RAM:** 8 GB minimum; 16 GB recommended

**Number of processors used:** 1 CPU core (no GPU required)

**Parallelization:** Serial; batched ONNX inference and single-threaded browser inference

**Keywords:** Poisson--Nernst--Planck; nanopore; neural operator; DeepONet; Fourier neural operator; physics-informed machine learning; inverse problem

**Classification:** 02.70.-c (computational techniques); 87.16.dj (membranes and bilayers)

**Nature of the problem:** Steady, axisymmetric Poisson--Nernst--Planck transport of a symmetric binary electrolyte in a charged cylindrical nanopore. The solver must represent the potential, both ion concentrations, the net charge density, and the axial current over a four-dimensional operating space: applied voltage, bulk concentration, wall charge strength, and pore radius. Repeated high-fidelity finite-volume solves are accurate but cost about 0.2 s each, which is prohibitive for design sweeps, uncertainty quantification, and inverse parameter estimation.

**Solution method:** A conservative Scharfetter--Gummel finite-volume reference solver generates a verified 240-state dataset. Two parameter-conditioned operator backbones are trained: a multi-scale radial physics-informed DeepONet and a parameter-conditioned two-dimensional Fourier neural operator. Both use an exact-reservoir Dirichlet ansatz and a charge-balanced field parameterization that guarantees positive concentrations and an exact algebraic charge identity. Models are exported to ONNX and served by a static browser simulator (JavaScript) and a Gradio application.

**Restrictions and unusual features:** The model is two-dimensional and axisymmetric; it neglects steric effects, electro-osmotic convection, surface chemistry, and pore-wall roughness. Temperature, permittivity, and ionic diffusivities are fixed. The operator is trained on in-distribution ranges of -50 to 50 mV, 20--200 mol m^-3, -0.05 <= q_w <= 0, and 3--8 nm radius; extrapolation beyond these ranges is not certified. The package includes a dual-engine browser simulator that can switch between the operator and an in-browser SG/FV solve.

**Running time:** Dataset generation 59.2 s; DeepONet training 50.6 s for 500 epochs; FNO training 108.5 s for 110 epochs; single-state inference 0.072 ms (DeepONet) and 1.04 ms (FNO) on an Apple M1 CPU.

---

## 1. Introduction

Nanopores and nanofluidic channels convert molecular-scale charge and geometry into measurable ionic currents, and they underpin single-molecule sensing, sequencing, and ionic circuit elements [1--6]. In confined geometries the electrical double layer (EDL) can overlap, so transport is governed by the coupled Poisson--Nernst--Planck (PNP) equations rather than by bulk conductance alone [7--12]. Surface charge, salt concentration, pore radius, and applied voltage jointly set selectivity and rectification [13--16], and the PNP system remains the standard continuum description of these effects [17--19]. EDL overlap, charge regulation, and electrokinetic dispersion modify conduction in narrow pores [55--58], and the same physics underlies emerging iontronic and nanofluidic devices [59,60].

Numerical solution of PNP is well understood. Scharfetter--Gummel (SG) discretizations [33] and conservative finite-volume formulations [34] provide stable, accurate currents on moderate grids. A single axisymmetric solve on a 21 x 61 grid takes about 0.2 s on a modern laptop, which is acceptable for one operating point but not for the thousands of evaluations demanded by parameter sweeps, sensitivity analysis, design optimization, or inverse estimation. The cost is compounded when a neural surrogate is trained per state.

Physics-informed neural networks (PINNs) [20--23] remove the need for labeled data but are trained for a single parameter set, and they are sensitive to gradient-flow pathologies and failure modes [25,26]. Hard-constraint formulations improve conditioning and enforce boundary data exactly [24,49--54], and extended or domain-decomposed variants enlarge the admissible geometry [26,27]. For PNP specifically, enriched and physics-informed variants have been reported [28,29], operator variants such as CF-DeepONet and T-phPINN have been applied to adjacent transport problems [30,31], and physics-informed surrogates have been used for battery-health prediction [32]. A previously reported fixed-state discontinuous-Galerkin DA-PINN for nanopore transport (COMPHY-D-26-00857) reached a potential relative L2 error of 3.82e-4 but required roughly 500 s of retraining per state. That trade-off motivates the present work: if the parameter dependence can be learned once, the per-query cost falls by orders of magnitude.

Operator learning provides the required map from parameters to fields. DeepONet approximates nonlinear operators through a branch--trunk inner product [43], with error estimates in infinite dimensions [44] and several recent architectural improvements [45--48]. Fourier neural operators (FNOs) learn global convolutions in a spectral basis [41] and have been extended to parametric and physics-informed settings [35--40,42]. Neural Green's operators offer another route to parametric PDE solution [39]. The central difficulty for transport in charged pores is not representational capacity but constraint satisfaction: potentials must satisfy reservoir Dirichlet data, concentrations must remain positive, and the predicted charge must equal the difference of the predicted concentrations. Unconstrained outputs violate these requirements, and the resulting errors are largest precisely where the net charge is small.

This paper makes six contributions.

1. A verified 240-state, four-dimensional PNP benchmark with deterministic train/interpolation/out-of-distribution splits and strict acceptance criteria (maximum residual 9.936e-10; minimum anion concentration 0.8034).
2. A charge-balanced, exact-reservoir parameterization that predicts a positive concentration sum and a net charge and reconstructs physical concentrations hyperbolically; the algebraic charge identity is satisfied to 1.110e-16.
3. Two parameter-conditioned operator backbones, a multi-scale radial PI-DeepONet (36,880 parameters) and a four-layer real-DFT FNO (460,488 parameters), trained with three seeds and evaluated in- and out-of-distribution.
4. A controlled ablation showing that charge reparameterization reduces net-charge error by 8.63-fold (DeepONet) and 10.29-fold (FNO), and that the physics loss reduces current error by 35.2% (DeepONet, interpolation) and 49.7% (FNO, OOD).
5. A deployment and amortization analysis: ONNX inference at 0.07165 ms, break-even after 459.2 dataset-mean states, and a browser/Gradio stack that matches PyTorch to 3.6e-7.
6. An inverse-recovery demonstration that separates an identifiable geometric parameter (radius, 0.46% median error) from a non-identifiable charge parameter, quantifying why conductance-only inversion cannot recover fixed surface charge.

The remainder of the paper describes the model and reference solver (Section 2), reports accuracy, ablation, timing, deployment, and inversion results (Section 3), and discusses scope and limitations (Section 4).

---

## 2. Methods

### 2.1 Physical model and dimensionless form

We consider a rigid, straight cylindrical pore of radius $R$ and length $L=30$ nm connecting two reservoirs at temperature $T=298.15$ K. The pore is filled with a symmetric binary electrolyte of bulk concentration $C_0$; the cation and anion diffusivities are $D_+=1.96\times10^{-9}$ m$^2$ s$^{-1}$ and $D_-=2.03\times10^{-9}$ m$^2$ s$^{-1}$. The relative permittivity is $\varepsilon_r=78.5$. A uniform surface charge of strength $q_w$ is imposed on the wall, and an axial voltage $V_\mathrm{app}$ is applied between the reservoirs. The unknowns are the electrostatic potential $\psi$, the cation and anion concentrations $c_\pm$, and the net charge density $\rho=c_+-c_-$.

The steady PNP system is

$$
-\nabla\cdot\left(\varepsilon_r\varepsilon_0\nabla\psi\right)=F\left(c_+-c_-\right),
\qquad
\nabla\cdot\mathbf{J}_\pm=0,
$$

$$
\mathbf{J}_\pm=-D_\pm\left(\nabla c_\pm\pm\frac{F}{RT}c_\pm\nabla\psi\right),
$$

where $F$ is Faraday's constant and $R$ the gas constant. At the reservoir faces, $\psi=V_\mathrm{app}$ and $\psi=0$, and $c_\pm=C_0$. On the charged wall, $\varepsilon_r\varepsilon_0\,\partial_r\psi|_\mathrm{wall}=\sigma$ and $\mathbf{J}_\pm\cdot\mathbf{n}=0$; on the axis, symmetry conditions apply. The wall charge strength is $q_w=\sigma R/(\varepsilon_r\varepsilon_0 V_T)$, with $V_T=RT/F$; this sign convention makes a negative $q_w$ produce the negative wall potential and cation enrichment seen in the reference solver.

We non-dimensionalize lengths by $R$ radially and by $L$ axially, concentrations by $C_0$, and potential by $V_T$. The Poisson coupling becomes $\beta=(L/\lambda_D)^2/2$, with $\lambda_D=\sqrt{\varepsilon_r\varepsilon_0 V_T/(F C_0)}$, and the geometric anisotropy is $\alpha_r=(L/R)^2$. The operating point is the four-vector

$$
\boldsymbol{\mu}=\left(V_\mathrm{app},\,C_0,\,q_w,\,R\right).
$$

The in-distribution training box is $V_\mathrm{app}\in[-50,50]$ mV, $C_0\in[20,200]$ mol m$^{-3}$, $q_w\in[-0.05,0]$, and $R\in[3,8]$ nm. The full dimensionless system, including the SG face fluxes used in the physics loss, is written out in Supplementary Section S3. The generated dataset also contains deliberate out-of-distribution charge/salt and geometry/voltage states, as described below.

### 2.2 Conservative SG/FV reference solver and dataset

The reference solver uses a node-centered axisymmetric finite-volume discretization with Scharfetter--Gummel face fluxes for the Nernst--Planck currents and a conservative Poisson update [33,34]. The production grid is $21\times61$ (radial $\times$ axial). Each state is first attempted directly; if the Newton/Gummel iteration does not converge, a deterministic continuation ramps $q_w$ at zero voltage and then ramps the voltage at the target charge, carrying the potential forward as the initial guess. A state is accepted only if the residual is below $10^{-9}$, all fields are finite, and both concentrations are strictly positive. The face flux uses the Bernoulli function $B(x)=x/(e^x-1)$: $J_{z,i+1/2}=-[c_{i+1}B(-\Delta\psi_i)-c_iB(\Delta\psi_i)]/\Delta z$, with the radial expression analogous; the Poisson update uses the conservative five-point stencil. The discrete equations are repeated in Supplementary Section S3.

The dataset contains $N=240$ states: 144 training states sampled by Latin hypercube inside the in-distribution box with boundary and corner anchors, 48 interpolation-test states, and 48 out-of-distribution states (24 with charge/salt outside the training box, 24 with geometry/voltage outside the box). Dataset construction took 59.18 s in total. The maximum residual over all states is 9.936e-10, the minimum cation concentration is 0.9999, and the minimum anion concentration is 0.8034 (Table 1). No state required continuation. The median SG/FV wall time is 0.2013 s and the mean is 0.2391 s.

The benchmark state (index 144, interpolation split) is $V_\mathrm{app}=50$ mV, $C_0=100$ mol m$^{-3}$, $q_w=-0.01$, and $R=5$ nm. Its reference mean axial current is 196.1 pA. This state is used for the field and current comparisons in Section 3.2 and in Figure 2.

**Table 1.** Dataset and solver validation.

| Quantity | Value |
|---|---|
| States | 240 |
| Train / interpolation / OOD | 144 / 48 / 48 |
| Maximum residual | 9.936e-10 |
| Minimum $c_+$ | 0.9999 |
| Minimum $c_-$ | 0.8034 |
| Continuation steps (all states) | 0 |
| Benchmark current | 196.1 pA |
| SG/FV median wall time | 0.2013 s |
| SG/FV mean wall time | 0.2391 s |

### 2.3 Exact-reservoir and charge-balanced parameterization

The operator does not output $c_+$ and $c_-$ directly. It predicts three smooth latent fields $G_\psi$, $G_s$, and $G_\rho$ on the fixed grid, where $z\in[0,1]$ is the normalized axial coordinate. The potential is

$$
\psi_\theta(r,z)=V_\mathrm{app}(1-z)+z(1-z)\,G_\psi(r,z),
$$

so that $\psi_\theta(r,0)=V_\mathrm{app}$ and $\psi_\theta(r,1)=0$ hold exactly for any $G_\psi$. The concentration sum and net charge are

$$
s_\theta(r,z)=2\exp\!\big[z(1-z)\,G_s(r,z)\big],\qquad
\rho_\theta(r,z)=z(1-z)\,G_\rho(r,z),
$$

with the exponent clipped to $[-15,15]$ for numerical safety. At both reservoirs the envelope $z(1-z)$ vanishes, so $s_\theta\to2$ (i.e., $2C_0$ in dimensional units) and $\rho_\theta\to0$, which gives $c_{\pm,\theta}\to1$ (i.e., $C_0$) exactly. Physical concentrations are recovered from the hyperbolic identity

$$
c_{\pm,\theta}=\frac{1}{2}\left(\sqrt{s_\theta^2+\rho_\theta^2}\pm\rho_\theta\right).
$$

Because $\sqrt{s^2+\rho^2}\ge|\rho|$, both concentrations are non-negative for any real $s$ and $\rho$, and the algebraic identity $c_{+,\theta}-c_{-,\theta}=\rho_\theta$ holds exactly. In the implementation the residual of this identity is 1.110e-16, i.e., machine precision. The same parameterization is applied to both backbones, so architecture comparisons are not confounded by different constraint handling.

This construction should be distinguished from soft penalty-based constraint enforcement [24,49--54]. No boundary or positivity penalty is required; the constraints are structural. The physics loss is used only to penalize violations of the differential equations in the interior.

### 2.4 Operator backbones

**Multi-scale radial PI-DeepONet.** The branch network maps the four parameters to $3p$ latent coefficients. The trunk network maps the coordinates and four fixed radial boundary-layer features, $(r,z,\{e^{-\kappa(1-r)}\}_{\kappa\in\{1,3,10,30\}})$, to $p$ basis functions; the multi-scale exponential features resolve the thin EDL near $r=1$ without increasing the trunk width. The two latent vectors are combined by an inner product, and the resulting three fields are passed through the exact-boundary and charge-balanced reconstruction. The model has $p=64$ and 36,880 trainable parameters.

**Parameter-conditioned two-dimensional FNO.** The FNO lifts the coordinate pair and the four parameters to a width of 24 channels, applies four spectral convolution blocks with $8\times12$ modes and FiLM parameter conditioning, and projects to three fields. The spectral convolutions use a real discrete Fourier transform so that the exported graph contains no complex-valued operators. The model has 460,488 trainable parameters.

Both backbones are trained with the same data and physics objective. The data term is a per-state normalized mean-squared error on the three latent fields, with the scale of each state floored at 10% of the global RMS. The physics term is the sum of the mean absolute Poisson residual, the mean absolute continuity residual for each ion, and the mean absolute wall potential and zero-flux residuals (Supplementary Section S2). The total loss is

$$
\mathcal{L}=\mathcal{L}_\mathrm{data}+10^{-3}\,\mathcal{L}_\mathrm{phys}.
$$

### 2.5 Training, splits, and metrics

Models are trained with seeds 42, 43, and 44. The DeepONet is trained for 500 epochs and the FNO for 110 epochs, with batch size 36, learning rate $2\times10^{-3}$, FNO width 24, FNO modes $8\times12$, and DeepONet inner dimension 64. Training times for seed 42 are 50.59 s (DeepONet) and 108.5 s (FNO). A separate 300-epoch FNO is trained to test whether the 110-epoch budget is the limiting factor.

Accuracy is reported as the global relative $L_2$ error

$$
\epsilon_\mathrm{rel}=\frac{\lVert f_\theta-f_\mathrm{ref}\rVert_2}{\lVert f_\mathrm{ref}\rVert_2}
$$

for $\psi$, $c_+$, $c_-$, $\rho$, and the axial current $I(z)$, together with the mean absolute current error in pA. The current is computed from the predicted fields with the same Scharfetter--Gummel face-flux formula used by the reference solver, so current errors include both field errors and post-processing consistency. All reported means are averages over the three seeds; spreads are reported as standard deviations in Table 3 and in the per-state distributions of Figure 4. The manuscript reports 3--4 significant figures in the narrative and tables; full machine-precision values are archived in `results/statistics.json` and `results/tables/*.csv`.

### 2.6 Deployment and inverse recovery

The trained DeepONet and FNO are exported to ONNX. Parity is checked against PyTorch on 96 states; the maximum absolute difference is 3.576e-7. The DeepONet is served (i) in a static browser simulator with a JavaScript evaluator and an optional in-browser SG/FV solve and (ii) in a Gradio application with operator-versus-reference comparison.

For the inverse demonstration, five held-out interpolation states are selected to span the charge and radius ranges. At each state, a five-bias current--voltage curve is generated with the exact SG/FV solver at $V_\mathrm{app}=-50,-25,0,25,50$ mV, and 1% Gaussian noise is added. The operator then evaluates 400 candidate $(q_w,R)$ pairs at the same five biases (2000 forward evaluations) and refines the best candidates with Nelder--Mead. Radius, surface charge density, and their relative errors are recorded. The flat direction of the objective in $q_w$ is quantified by profiling the current mismatch over the full charge prior at the recovered radius. The 1% noise level is a deliberately conservative proxy for a clean nanopore measurement, and the grid search, rather than the local Nelder--Mead refinement, determines the reported flat direction. The identifiability conclusion is expected to weaken for asymmetric electrolytes, unequal diffusivities, or a much wider charge range; these cases are not covered by the present benchmark.

---

## 3. Results

### 3.1 Reference dataset and solver validation

All 240 states converged directly, with a maximum residual of 9.936e-10 and strictly positive concentrations (Table 1). The smallest anion concentration, 0.8034, occurs in the strongest-overlap states, where the EDL occupies a large fraction of the pore cross-section. The benchmark current is 196.1 pA, and the SG/FV median and mean wall times are 0.2013 s and 0.2391 s. These values define the cost baseline for the amortization analysis in Section 3.6.

Figure 1 summarizes the problem. Panel A shows the axisymmetric geometry, reservoirs, wall charge, and the PNP equations. Panel B places all 240 states in the $(\lambda_D/R,|q_w|)$ plane, colored by split, with the benchmark state marked; the in-distribution band and the training charge limit are indicated. Panel C shows the exact-reservoir and charge-balanced construction. Panel D compares the two backbones, including parameter counts and ONNX latencies.

### 3.2 Benchmark state versus a fixed-state DA-PINN

Table 2 compares the operators with the previously reported fixed-state discontinuous-Galerkin DA-PINN (COMPHY-D-26-00857) on the benchmark state. The DeepONet reaches a potential relative $L_2$ error of 0.0001485 and a current relative $L_2$ error of 0.0005587, compared with 0.000382 and 0.00102 for the DA-PINN. This is comparable or better accuracy without any per-state retraining, but the comparison is not apples-to-apples: the DA-PINN was trained specifically on this state, whereas the operator has never seen it. We therefore report the ratio (2.57-fold and 1.83-fold) as an accuracy-per-training-cost observation, not as evidence that the operator dominates a state-specialized model. The DeepONet net-charge error is 0.09363 versus 0.145, a 1.55-fold reduction. The direct cation error is slightly larger than that of the DA-PINN (0.0001521 versus 0.000106, i.e., 1.44-fold higher), which is expected because the DA-PINN was trained specifically on this state while the operator must generalize across the parameter space.

The FNO is less accurate on this state: potential error 0.003298 and current error 0.00634. Its current mean is 196.2 pA versus the reference 196.1 pA. The DeepONet current mean is 196.1 pA. Figure 2 shows the DeepONet fields, the error maps, and the axial current profile for the benchmark state; the current profile agrees with the SG/FV reference to within the line width over most of the pore, with the largest deviations near the reservoir transitions.

**Table 2.** Benchmark state (50 mV, 100 mol m$^{-3}$, $q_w=-0.01$, $R=5$ nm) against the prior fixed-state DA-PINN.

| Model | Current (pA) | $\psi$ rel. $L_2$ | $c_+$ rel. $L_2$ | $\rho$ rel. $L_2$ | $I$ rel. $L_2$ |
|---|---|---|---|---|---|
| DeepONet (reparam.) | 196.1 | 0.0001485 | 0.0001521 | 0.09363 | 0.0005587 |
| FNO (reparam.) | 196.2 | 0.003298 | 0.001184 | 0.1481 | 0.00634 |
| DeepONet (direct $c$) | 196.1 | 0.0008521 | 0.0006095 | 0.7147 | 0.0006118 |
| FNO (direct $c$) | 195.6 | 0.001352 | 0.003661 | 0.8599 | 0.004085 |
| Prior DA-PINN | 196.1 | 0.000382 | 0.000106 | 0.145 | 0.00102 |

### 3.3 In-distribution accuracy and the value of charge reparameterization

Averaged over three seeds and 48 interpolation states, the DeepONet attains relative $L_2$ errors of 0.0008914 for $\psi$, 0.0007275 for $c_+$, 0.08248 for $\rho$, and 0.001093 for $I$. The mean absolute current error is 0.08031 pA. The FNO attains 0.005495, 0.001641, 0.0931, and 0.008511, respectively, with a mean absolute current error of 0.6746 pA. The DeepONet is therefore 6.16-fold more accurate for potential, 2.26-fold for cation concentration, 1.13-fold for net charge, and 7.79-fold for current.

The net-charge metric is the most sensitive to the output parameterization. With direct concentration outputs, the interp-test net-charge error rises to 0.7116 for the DeepONet and 0.9578 for the FNO; the charge-balanced parameterization reduces these to 0.08248 and 0.0931, i.e., by 8.63-fold and 10.29-fold. This is the cancellation mechanism: when $c_+$ and $c_-$ are predicted independently, small relative errors in each field produce a large relative error in their difference, especially in the weakly charged states where $\rho$ is small. Predicting $s$ and $\rho$ and reconstructing $c_\pm$ removes that cancellation. Figure 3A quantifies the effect on the benchmark interpolation split, and Figure 3C shows the radial net-charge profile in a weak-charge state, where the direct model departs from the reference near the wall while the reparameterized model follows it.

A 300-epoch FNO reduces the interp-test potential error from 0.005495 to 0.002195 (2.50-fold) and the current error from 0.008511 to 0.003349 (2.54-fold), with smaller gains out of distribution (Table 5). The 110-epoch FNO is therefore not fully converged in-distribution, but its ranking against the DeepONet is unchanged; the DeepONet remains more accurate at a fraction of the parameter count.

Figure 5 checks continuous parametric control rather than pointwise accuracy. Panel A sweeps the applied voltage from $-50$ to 50 mV at $C_0=100$ mol m$^{-3}$ and $q_w=-0.01$ for three radii (3, 5, and 7 nm) and compares the operator current--voltage curves with on-the-fly SG/FV solves at five biases; the curves overlap across the full range. Panel B plots the near-wall cation concentration against bulk salt from 10 to 200 mol m$^{-3}$ at $R=5$ nm, showing the transition from EDL-dominated enrichment at low salt to bulk-like values at high salt. Panel C shows the radial net-charge profile at $C_0=20$, 100, and 200 mol m$^{-3}$, where the profile changes from overlap-dominated to wall-localized as salinity increases. These curves are interpolations at parameter values that are not training states, so they test the operator as a function rather than as a lookup table.

### 3.4 Out-of-distribution behavior

The OOD split combines charge/salt extrapolation and geometry/voltage extrapolation. For the DeepONet, the relative $L_2$ errors are 0.02024 for $\psi$, 0.0213 for $c_+$, 0.3678 for $\rho$, and 0.001363 for $I$; the mean absolute current error is 0.1077 pA. For the FNO the corresponding values are 0.02079, 0.01487, 0.2241, and 0.009123, with a mean absolute current error of 0.5338 pA. The two models are statistically indistinguishable for potential (FNO/DeepONet ratio 1.03), while the FNO is better for cation concentration (1.43-fold) and net charge (1.64-fold), and the DeepONet is better for current (6.69-fold).

The regime shift is visible in the potential error: the DeepONet OOD potential error is 22.7-fold larger than its interp-test value, whereas the FNO grows only 3.78-fold because it starts from a larger in-distribution error. Current is far more robust than the pointwise fields: both models keep OOD current errors below 1%, because the current is an integral of the flux and benefits from error cancellation, whereas pointwise concentrations and potentials remain sensitive to local extrapolation error. Field-level OOD predictions should therefore be interpreted more cautiously than current--voltage predictions. This distinction matters for applications: if the target is a measured current--voltage curve, the operator transfers better than the field-level metrics suggest, but if the target is a local concentration or charge profile, OOD accuracy must be checked explicitly.

Figure 4A shows the per-state distributions for both splits, Figure 4B resolves the error against the Debye ratio, and Figure 4C shows the current parity over all 96 test states. The parity slope is close to unity and the coefficient of determination is 1 to four decimal places; the residual inset shows no systematic bias with current magnitude.

### 3.5 Charge reparameterization as a structural fix

The ablation in Figure 3 and Table 3 isolates the effect of the output parameterization. The reparameterized models are not merely better on average; they remove the heavy tail of the direct-charge models. In the weakest-charge quartile, the direct models produce relative net-charge errors near unity, whereas the reparameterized models remain below 0.15. The same ordering holds for the FNO, so the benefit is architectural rather than specific to the DeepONet.

Because the reconstruction is algebraic, the improvement costs nothing at inference: the additional square root and division are negligible compared with the network evaluation, and they do not introduce tunable penalties. The identity $c_+-c_-=\rho$ holds to machine precision, which means the downstream current post-processing receives a physically consistent charge field by construction.

### 3.6 Timing, amortization, and break-even

Table 4 and Figure 6 summarize latency and amortization. On an Apple M1 CPU, the DeepONet exports to ONNX at 0.07165 ms per state, 18.7-fold faster than its PyTorch batch-1 latency of 1.342 ms; the FNO exports at 1.038 ms, 8.19-fold faster than 8.5 ms. At batch 48, the ONNX latencies are 0.7095 ms and 26.86 ms, and the shared current post-processing adds 3.788 ms.

The break-even count is the number of parameter states at which the one-time cost of dataset generation and training equals the cost of solving those states with SG/FV. Using the dataset-mean SG/FV time of 0.2391 s, the DeepONet breaks even at 459.2 states and the FNO at 704.6 states. Using the median time of 0.2013 s, the counts are 545.5 and 837.5. On the slower prior Xeon timing of 0.44 s per solve, the counts fall to 249.5 and 382.1. Against per-state DA-PINN retraining at 500 s per state, the break-even is 0.2195 states for the DeepONet and 0.3355 states for the FNO: a single query already amortizes the operator.

These counts are conservative in one respect and optimistic in another. They charge the full dataset-generation and training cost to the operator, but they do not charge the engineering time of building the surrogate; conversely, they assume that the SG/FV baseline is run serially, whereas it could be parallelized across cores. The break-even count is only about 1.9 times the 240-state training set, so the operator is most advantageous for sweeps of thousands of states rather than for a few hundred queries. The qualitative conclusion remains robust: for the thousands of queries typical of a design sweep, the operator is two to three orders of magnitude cheaper per query.

### 3.7 Physics loss and data-only ablation

Removing the physics term degrades the current more than the fields. For the DeepONet, the interp-test current error rises from 0.001093 to 0.001687 (35.2% higher), while the potential error changes only from 0.0008914 to 0.000928 (3.95% higher). For the FNO, the interp-test current error rises from 0.008511 to 0.01393 (38.9% higher), and the OOD current error rises from 0.009123 to 0.01815 (49.7% higher). The OOD potential error of the FNO also rises from 0.02079 to 0.03013 (31.0% higher).

The interpretation is that the physics residual constrains the flux balance, which the current integral senses even when the pointwise fields are visually similar. The data-only DeepONet is competitive for in-distribution potential and cation concentration, but its current error is 54.3% higher, and its OOD current error is 30.5% higher. The physics term is therefore not cosmetic; it is the main driver of current accuracy in the sparse-data regime.

### 3.8 Inverse parameter recovery

The inverse demonstration asks what can be inferred from a noisy current--voltage curve alone. For five held-out states, the operator recovers the pore radius with a median relative error of 0.004574 (0.46%). The radius is identifiable because the current scales approximately with the pore cross-section and the five-bias slope fixes the conductance scale.

Surface charge is not identifiable at the same noise level. Across the full charge prior, the median state-level change in the predicted current at the recovered radius is 0.0009625 (0.096%), and the largest state-level change is 0.0019 (0.19%); both are below the 1% measurement noise. The objective is therefore flat in $q_w$: the recovered charge can move anywhere in the prior without a detectable change in the residual. The median relative error in $q_w$ is 1.154, and the median relative error in the derived surface charge density is 1.146. Relative error is misleading for the near-zero charge state: state 175 has a true $q_w=-0.00206$, so a small absolute error of 0.0961 produces a relative error of 46.7. The median absolute error in $q_w$ is 0.0526 (about half the prior width), and the median absolute error in $R$ is 0.0181 nm. Table 6 reports both relative errors and the per-state values. The flat direction is not a failure of the operator; it is a property of the measurement. For a symmetric electrolyte with nearly equal ionic diffusivities, the leading conductance is set by the total ionic strength, and over the accessible charge range the Donnan polarization induced by the wall changes that conductance by less than 0.2%. The total conductance is therefore nearly independent of fixed wall charge at the 1% noise level. Recovering $q_w$ requires a charge-sensitive observable, such as the streaming potential, the radial potential profile, or a multi-salt nonlinear response, rather than conductance alone.

The timing makes the amortization concrete. Each inverse problem evaluates 2000 forward states (400 candidates at five biases); the operator does so in 0.1253 s, whereas the same number of SG/FV solves would take 402.6 s, an approximately 3200-fold speed-up. Table 6 reports the per-state results. The same arithmetic as Figure 6B then applies: 2000 forward evaluations cost 0.125 s with the operator and 402.6 s with SG/FV, so a single five-state inversion already sits far past the forward break-even point.

**Table 3.** Selected three-seed global relative $L_2$ errors (means over seeds; full per-metric table in `results/tables/table3_multiseed_h1_h2_h3_ablation.csv`).

| Model | Split | $\psi$ | $c_+$ | $\rho$ | $I$ | mean abs. $I$ error (pA) |
|---|---|---|---|---|---|---|
| DeepONet reparam. | interp. | 0.0008914 | 0.0007275 | 0.08248 | 0.001093 | 0.08031 |
| DeepONet reparam. | OOD | 0.02024 | 0.0213 | 0.3678 | 0.001363 | 0.1077 |
| FNO reparam. | interp. | 0.005495 | 0.001641 | 0.0931 | 0.008511 | 0.6746 |
| FNO reparam. | OOD | 0.02079 | 0.01487 | 0.2241 | 0.009123 | 0.5338 |
| DeepONet direct $c$ | interp. | 0.003845 | 0.004229 | 0.7116 | 0.002335 | 0.2108 |
| FNO direct $c$ | interp. | 0.00562 | 0.006777 | 0.9578 | 0.007836 | 0.6489 |
| DeepONet data-only | interp. | 0.000928 | 0.000488 | 0.07719 | 0.001687 | 0.1203 |
| FNO data-only | interp. | 0.007958 | 0.002193 | 0.0841 | 0.01393 | 1.126 |

**Table 4.** Latency and amortization.

| Item | Value | Unit |
|---|---|---|
| DeepONet ONNX, batch 1 | 0.07165 | ms |
| FNO ONNX, batch 1 | 1.038 | ms |
| DeepONet PyTorch, batch 1 | 1.342 | ms |
| FNO PyTorch, batch 1 | 8.5 | ms |
| DeepONet ONNX, batch 48 | 0.7095 | ms |
| FNO ONNX, batch 48 | 26.86 | ms |
| Current post-processing, batch 48 | 3.788 | ms |
| DeepONet break-even, dataset mean | 459.2 | states |
| FNO break-even, dataset mean | 704.6 | states |
| DeepONet break-even, median | 545.5 | states |
| FNO break-even, median | 837.5 | states |
| DeepONet break-even, prior Xeon | 249.5 | states |
| FNO break-even, prior Xeon | 382.1 | states |
| DeepONet break-even vs DA-PINN retrain | 0.2195 | states |
| FNO break-even vs DA-PINN retrain | 0.3355 | states |

**Table 5.** FNO training-budget check (seed 42).

| Split | Metric | 110 epochs (3-seed mean) | 300 epochs (seed 42) |
|---|---|---|---|
| interp. | $\psi$ rel. $L_2$ | 0.005495 | 0.002195 |
| interp. | $I$ rel. $L_2$ | 0.008511 | 0.003349 |
| OOD | $\psi$ rel. $L_2$ | 0.02079 | 0.0132 |
| OOD | $I$ rel. $L_2$ | 0.009123 | 0.00234 |
| benchmark | $\psi$ rel. $L_2$ | 0.003298 | 0.001743 |
| benchmark | $I$ rel. $L_2$ | 0.00634 | 0.003669 |

**Table 6.** Inverse recovery from five-bias noisy current--voltage curves (1% noise; 400 candidates, 2000 forward evaluations per state). Values are rounded for display; the exact values are in `results/tables/table6_inverse_parameter_recovery.csv`.

| State | $C_0$ (mol m$^{-3}$) | $R$ true (nm) | $R$ recovered (nm) | $R$ rel. error | $q_w$ true | $q_w$ recovered | $q_w$ rel. error |
|---|---|---|---|---|---|---|---|
| 175 | 27.2 | 6.083 | 6.085 | 0.0003 | -0.00206 | -0.09816 | 46.69 |
| 147 | 93.3 | 6.918 | 6.870 | 0.0069 | -0.03770 | -0.07835 | 1.078 |
| 154 | 51.1 | 3.362 | 3.344 | 0.0054 | -0.03711 | -0.09816 | 1.645 |
| 172 | 103.9 | 7.276 | 7.302 | 0.0036 | -0.04557 | -0.09816 | 1.154 |
| 161 | 140.2 | 3.917 | 3.935 | 0.0046 | -0.02875 | -0.00978 | 0.660 |

### 3.9 Interactive deployment

The exported models run without a Python server. The browser bundle is 5,662,351 bytes and includes both ONNX models, the metadata, and a JavaScript evaluator. The JavaScript current formula agrees with the SG/FV reference to a maximum relative difference of 5.686e-7 over the eight verification states, and the DOM test harness passes 29 of 29 checks. The local SG/FV service returns the benchmark current 196.1 pA in 0.2082 s with four iterations and a residual of 2.42e-10. The Gradio application returns a DeepONet mean current of 196.1 pA in 0.757 ms and, when asked to compare against a live solve, reports a potential relative $L_2$ difference of 0.0002126 in 0.2057 s of solve time.

This deployment matters for the intended use. A browser simulator can be used in teaching and in experimental design without installing a solver, and the same ONNX graph that runs in the browser also runs in the Gradio app on a laptop. The dual-engine design keeps the reference solver available for validation rather than replacing it.

---

## 4. Discussion

The results support a simple division of labor. The DeepONet is the better default surrogate: it is 6.16-fold more accurate for potential and 7.79-fold more accurate for current than the FNO in-distribution, with 12.5-fold fewer parameters and 14.5-fold lower ONNX latency. The FNO is not redundant, however. Its spectral basis captures the cation and net-charge fields better in the out-of-distribution split (1.43-fold and 1.64-fold, respectively), and its errors grow more gracefully when the Debye ratio leaves the training band. A practical strategy is to use the DeepONet for current-based design and the FNO as a cross-check or as a backbone when field-level OOD accuracy is the priority.

The charge-balanced parameterization is the most transferable result. The improvement is large (8.63-fold for DeepONet and 10.29-fold for FNO), it costs nothing at inference, and it is architecture-agnostic. The underlying mechanism is general: whenever a physical quantity is a small difference of two large positive fields, direct regression suffers from cancellation, and a reparameterization in terms of the sum and the difference removes the ill-conditioning. The same idea should apply to other Nernst--Planck and drift--diffusion systems in which charge or mole fractions are inferred from separate species predictions.

The physics loss plays a complementary role. It does not dramatically improve the pointwise potential in-distribution, but it reduces current error by 35.2% for the DeepONet and 49.7% for the FNO out of distribution. Because current is the quantity that experiments measure, this is the relevant metric, and it justifies retaining the residual term even when labeled data are available.

The inverse experiment sharpens the scope of the method. Radius is identifiable from a five-bias current--voltage curve to 0.46%, and the operator makes the inversion cheap (0.1253 s per curve versus 402.6 s of SG/FV work). Surface charge is not identifiable under the same protocol: the median state-level current change is 0.096% across the full charge prior (largest state-level change 0.19%), both below the 1% noise. This is a physical identifiability limit, not a surrogate error. It follows from the near-invariance of the concentration sum for a symmetric electrolyte with nearly equal diffusivities, and it implies that conductance-only inversion cannot recover fixed charge even with a perfect forward model. Closing this gap requires a different measurement: streaming potential, field-resolved sensing, or a multi-salt protocol that probes the EDL overlap nonlinearity. The operator is well suited to such protocols because it can evaluate thousands of candidate parameter combinations per second.

Several limitations should be stated plainly. The model is two-dimensional and axisymmetric; it does not resolve azimuthal roughness, pore mouths, or non-circular cross-sections. It neglects steric effects, finite ion size, electro-osmotic convection, and surface chemistry, so it should not be used to interpret pH-dependent or highly concentrated experiments without extension. Temperature, permittivity, and diffusivities are fixed. The training box is modest ($-50$ to 50 mV, 20--200 mol m^-3, $|q_w|\le0.05$, 3--8 nm), and the OOD split is a controlled extrapolation rather than a stress test at experimental extremes. The reference solver itself is a continuum model; agreement with it demonstrates numerical consistency, not agreement with a particular experiment. Finally, the inverse demonstration uses synthetic noise and a known bulk concentration; in an experiment, concentration and temperature uncertainty would widen the posterior and could break the degeneracy between radius and other geometric parameters.

The deployment results suggest where the work can go next. The browser simulator makes the operator accessible without a solver installation, and the Gradio comparison keeps a reference solve one click away. A natural extension is to add a charge-sensitive observable to the inverse workflow, and to train a second operator that maps current--voltage curves to parameters directly, using the forward operator to generate training pairs. The same dataset and constraint machinery can be reused for that task.

---

## 5. Conclusion

A charge-balanced, exact-reservoir physics-informed operator learns steady axisymmetric PNP transport over a four-dimensional parameter space. The DeepONet achieves 0.0008914 potential and 0.001093 current relative $L_2$ error on held-out in-distribution states, remains below 0.14% current error out of distribution, and runs in 0.072 ms per state after ONNX export. The charge-balanced parameterization reduces net-charge error by an order of magnitude relative to direct concentration outputs, and the physics loss reduces current error by 35--50%. The operator breaks even against the reference solver after about 460 states at the dataset-mean solve time, and is deployed in a browser simulator and a Gradio application that match PyTorch to within $3.6\times10^{-7}$ maximum absolute difference. A noisy inverse experiment recovers pore radius to 0.46% but shows that surface charge is a flat direction for conductance-only inversion, which defines the measurement requirements for future charge-sensitive workflows.

---

## CRediT authorship contribution statement

**Javad Alizargar:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing -- original draft, Writing -- review and editing, Visualization, Supervision, Project administration.

---

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Data and code availability

The dataset, trained models, figure-generation scripts, and analysis code are available at https://github.com/Javad-Alizargar/nanopore-pino-surrogate under the MIT License. The interactive browser simulator is served at https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/. A permanent archived copy will be deposited on Zenodo upon acceptance. The repository contains the SG/FV reference solver, the 240-state dataset, the training and evaluation scripts, the ONNX models, the browser simulator, and the Gradio application. The raw literature metadata used to build the reference list is included in `manuscript/literature_db.json`.

---

## Generative AI declaration

During the preparation of this work the author used large language model tools for drafting assistance, code generation, and figure-quality review. All numerical results were produced by the local code and verified against the archived result files; all references were checked against Crossref metadata. The author reviewed and edited the content and takes full responsibility for the published article.

---

## Acknowledgements

The author thanks the developers of the open-source scientific Python stack used in this work (NumPy, SciPy, PyTorch, ONNX Runtime, Matplotlib, Gradio) and the maintainers of the Crossref and PubMed metadata services.

---

## Figure captions

**Figure 1.** Problem definition and architecture. (A) Axisymmetric charged nanopore of radius $R$ and length $L$, with fixed wall charge $q_w$, reservoirs at $z=0$ and $z=1$, and applied bias $V_\mathrm{app}$. (B) The 240-state operating space in the four parameters $(V_\mathrm{app},C_0,q_w,R)$, showing the 144 training states, 48 interpolation-test states, and 48 out-of-distribution states. (C) Hard constraints imposed by the ansatz: exact reservoir potential and concentration values, zero wall ion flux, wall charge condition, and positive concentrations. (D) The two parameter-conditioned backbones: a DeepONet with a multi-scale radial trunk and a 2D Fourier neural operator, both trained with the same data and physics objective.

**Figure 2.** Benchmark state #0 ($V_\mathrm{app}=50$ mV, $C_0=100$ mol m$^{-3}$, $q_w=-0.01$, $R=5$ nm), which is out-of-sample. (A) DeepONet potential $\psi_\theta(r,z)$. (B) DeepONet cation concentration $c_+^\theta(r,z)$. (C) DeepONet net charge $\rho_\theta=c_+^\theta-c_-^\theta$. (D) Absolute potential error against the SG/FV reference. (E) Absolute net-charge error. (F) Axial current profile $I(z)$ for the SG/FV reference, the DeepONet, the FNO, and the prior fixed-state DA-PINN (band shows its reported error). All fields are normalized by their reference scales and plotted on the physical grid.

**Figure 3.** Charge-balanced reparameterization and physics-loss ablation. (A) Three-seed mean $\pm$ SD of global relative $L_2$ errors for direct-concentration and reparameterized DeepONet and FNO models on the interpolation-test split. (B) The weak-charge cancellation regime: relative net-charge error versus $|q_w|$ for direct and reparameterized outputs. (C) Radial net-charge profile for a weak-charge state, comparing the reference, direct, and reparameterized predictions. (D) Physics-loss ablation on the interpolation-test split for potential, net charge, and current errors.

**Figure 4.** In-distribution versus out-of-distribution behavior. (A) Per-state relative $L_2$ error distributions for potential, net charge, and current for the DeepONet and FNO on the interpolation-test and OOD splits; boxes show the median and interquartile range and whiskers extend to the full state range. (B) Error versus EDL overlap parameter for net charge (filled symbols) and axial current (open triangles). (C) Current parity for the 96 test states, with the residual (prediction minus reference, in pA) shown in the lower subpanel; the dashed line is the identity.

**Figure 5.** Continuous parametric control. (A) Current--voltage curves from $-50$ to 50 mV at $C_0=100$ mol m$^{-3}$ and $q_w=-0.01$ for $R=3$, 5, and 7 nm; symbols are SG/FV solves at five biases and lines are operator predictions. (B) Near-wall cation concentration versus bulk concentration from 10 to 200 mol m$^{-3}$ at $R=5$ nm and $q_w=-0.01$. (C) Radial net-charge profiles at $C_0=20$, 100, and 200 mol m$^{-3}$ for $R=5$ nm, showing the transition from EDL overlap to wall-localized charge. These parameter values are interpolations, not training states.

**Figure 6.** Latency and amortization. (A) Batch-1 and batch-48 prediction latency for the DeepONet and FNO in PyTorch and ONNX on an Apple M1 CPU, including the shared current post-processing at batch 48. (B) Cumulative cost versus number of forward states for the operator and for SG/FV solves at the dataset-mean and median solve times, with the break-even counts marked. The prior-Xeon and DA-PINN-retrain reference lines are included for context.

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
11. Karnik, Rohit, Fan, Rong, Yue, Min, et al. (2005). Electrostatic Control of Ions and Molecules in Nanofluidic Transistors *Nano Letters*. doi:10.1021/nl050493b
12. Tagliazucchi, Mario, Szleifer, Igal (2015). Transport mechanisms in nanopores and nanochannels: can we mimic nature? *Materials Today*. doi:10.1016/j.mattod.2014.10.020
13. Smeets, Ralph M. M., Keyser, Ulrich F., Krapf, Diego, et al. (2005). Salt Dependence of Ion Transport and DNA Translocation through Solid-State Nanopores *Nano Letters*. doi:10.1021/nl052107w
14. Kosińska, I. D., Goychuk, I., Kostur, M., et al. (2008). Rectification in synthetic conical nanopores: A one-dimensional Poisson-Nernst-Planck model *Physical Review E*. doi:10.1103/physreve.77.031131
15. Valiskó, Mónika, Matejczyk, Bartłomiej, Ható, Zoltán, et al. (2019). Multiscale analysis of the effect of surface charge pattern on a nanopore’s rectification and selectivity properties: From all-atom model to Poisson-Nernst-Planck *The Journal of Chemical Physics*. doi:10.1063/1.5091789
16. Ma, Pengfei, Zheng, Jianxiang, Zhao, Danting, et al. (2021). The Selective Transport of Ions in Charged Nanopore with Combined Multi-Physics Fields *Materials*. doi:10.3390/ma14227012
17. Coalson, R.D., Kurnikova, M.G. (2005). Poisson–Nernst–Planck Theory Approach to the Calculation of Current Through Biological Ion Channels *IEEE Transactions on Nanobioscience*. doi:10.1109/tnb.2004.842495
18. Eisenberg, Bob, Liu, Weishi (2007). Poisson–Nernst–Planck Systems for Ion Channels with Permanent Charges *SIAM Journal on Mathematical Analysis*. doi:10.1137/060657480
19. Zheng, Qiong, Chen, Duan, Wei, Guo-Wei (2011). Second-order Poisson–Nernst–Planck solver for ion transport *Journal of Computational Physics*. doi:10.1016/j.jcp.2011.03.020
20. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations *Journal of Computational Physics*. doi:10.1016/j.jcp.2018.10.045
21. Karniadakis, George Em, Kevrekidis, Ioannis G., Lu, Lu, et al. (2021). Physics-informed machine learning *Nature Reviews Physics*. doi:10.1038/s42254-021-00314-5
22. Lu, Lu, Meng, Xuhui, Mao, Zhiping, et al. (2021). DeepXDE: A Deep Learning Library for Solving Differential Equations *SIAM Review*. doi:10.1137/19m1274067
23. Lagaris, I.E., Likas, A., Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations *IEEE Transactions on Neural Networks*. doi:10.1109/72.712178
24. Sukumar, N., Srivastava, Ankit (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2021.114333
25. Wang, Sifan, Teng, Yujun, Perdikaris, Paris (2021). Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks *SIAM Journal on Scientific Computing*. doi:10.1137/20m1318043
26.  (2021). Characterizing possible failure modes in physics-informed neural networks *arXiv (Cornell University)*. doi:10.48550/arxiv.2109.01050
27. D. Jagtap, Ameya, Em Karniadakis, George (2020). Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations *Communications in Computational Physics*. doi:10.4208/cicp.oa-2020-0164
28. Huang, Xujia, Wang, Fajie, Zhang, Benrong, et al. (2025). Enriched physics-informed neural networks for dynamic Poisson-Nernst-Planck systems *Mathematics and Computers in Simulation*. doi:10.1016/j.matcom.2025.04.037
29. Zheng, Jinglai, Huang, Jie, Huang, Haiming (2025). Physics-informed neural networks for solving thermal response of charring materials *Acta Astronautica*. doi:10.1016/j.actaastro.2025.07.010
30. Zheng, Jinglai, Hu, Hanying, Huang, Jie, et al. (2025). CF-DeepONet: Deep operator neural networks for solving compressible flows *Aerospace Science and Technology*. doi:10.1016/j.ast.2025.110329
31. Zheng, Jinglai, Li, Fan, Huang, Haiming (2024). T-phPINN: Physics-informed neural networks for solving 2D non-Fourier heat conduction equations *International Journal of Heat and Mass Transfer*. doi:10.1016/j.ijheatmasstransfer.2024.126216
32. Wei, Congxin, Pang, Haikuo, Huang, Teng, et al. (2026). Predicting lithium-ion battery health using attention mechanism with Kolmogorov-Arnold and physics-informed neural networks *Expert Systems with Applications*. doi:10.1016/j.eswa.2025.128969
33. Scharfetter, D.L., Gummel, H.K. (1969). Large-signal analysis of a silicon Read diode oscillator *IEEE Transactions on Electron Devices*. doi:10.1109/t-ed.1969.16566
34. Miller, J. J. H., Wang, Song (1994). An analysis of the Scharfetter-Gummel box method for the stationary semiconductor device equations *ESAIM: Mathematical Modelling and Numerical Analysis*. doi:10.1051/m2an/1994280201231
35. Kangjie Li, Wenjing Ye (2025). D-FNO: A decomposed Fourier neural operator for large-scale parametric partial differential equations *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2025.117732
36.  (2024). Physics-informed fourier neural operators: A machine learning method for parametric partial differential equations *Journal of Nonlinear and Variational Analysis*. doi:10.23952/jnva.9.2025.1.04
37. Hailong Sheng, Yulin Xie, Chao Yang (2025). XFNO: Extended Fourier neural operator for solving parametric partial differential equations on general geometries *Computer Physics Communications*. doi:10.1016/j.cpc.2025.109774
38. Zongyi Li, Hongkai Zheng, Nikola Kovachki, et al. (2024). Physics-Informed Neural Operator for Learning Partial Differential Equations *ACM / IMS Journal of Data Science*. doi:10.1145/3648506
39. Hugo Melchers, Joost Prins, Michael Abdelmalik (2026). Neural Green’s operators for parametric partial differential equations *Computer Methods in Applied Mechanics and Engineering*. doi:10.1016/j.cma.2026.118893
40. Kai Qi, Jian Sun (2024). Gabor-Filtered Fourier Neural Operator for solving Partial Differential Equations *Computers & Fluids*. doi:10.1016/j.compfluid.2024.106239
41. Sifan Wang, Hanwen Wang, Paris G. Perdikaris (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets *Science Advances*. doi:10.1126/sciadv.abi8605
42. Muhammad Tahir Rafiq, Ghazala Rafiq, Ho-Youl Jung, et al. (2022). SSNO: Spatio-Spectral Neural Operator for Functional Space Learning of Partial Differential Equations *IEEE Access*. doi:10.1109/access.2022.3148401
43. Lu Lu, Pengzhan Jin, Guofei Pang, et al. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators *Nature Machine Intelligence*. doi:10.1038/s42256-021-00302-5
44. Samuel Lanthaler, Siddhartha Mishra, George Em Karniadakis (2022). Error estimates for DeepONets: a deep learning framework in infinite dimensions *Transactions of Mathematics and Its Applications*. doi:10.1093/imatrm/tnac001
45. Hwijae Son (2025). ELM-DeepONets: Backpropagation-Free Training of Deep Operator Networks via Extreme Learning Machines *IEEE Access*. doi:10.1109/access.2025.3570502
46. Qiuying Ma, Xinghan Jiang, Xinlong Guo, et al. (2025). Layer-Interaction DeepONet for Modeling Ultrafast Nonlinear Dynamics in Optical Fibers *Journal of Lightwave Technology*. doi:10.1109/jlt.2025.3598027
47. Samuel Lanthaler, Zongyi Li, Andrew M. Stuart (2025). Nonlocality and Nonlinearity Implies Universality in Operator Learning *Constructive Approximation*. doi:10.1007/s00365-025-09718-3
48. Yongjin Choi, Chenying Liu, Jorge Macedo (2026). Operator learning for consolidation: An architectural comparison for DeepONet variants *Computers and Geotechnics*. doi:10.1016/j.compgeo.2026.108017
49. Shaikhah Alkhadhr, Mohamed Khaled Almekkawy (2023). Wave Equation Modeling via Physics-Informed Neural Networks: Models of Soft and Hard Constraints for Initial and Boundary Conditions *Sensors*. doi:10.3390/s23052792
50. Gang Jing, Huan Wang, Yuan Jiu, et al. (2026). A unified hard-constraint physics-informed neural network approach for sparse-observation airflow-field reconstruction under complex boundary conditions *Engineering Applications of Artificial Intelligence*. doi:10.1016/j.engappai.2026.116169
51. Mohammad Sheikholeslami, Saeed Salehi, Wengang Mao, et al. (2025). Physics-informed neural networks with hard and soft boundary conditions for linear free surface waves *Physics of Fluids*. doi:10.1063/5.0277421
52. Peng Lan, Jing-jing Su, Xin-yan Ma, et al. (2023). Application of improved physics-informed neural networks for nonlinear consolidation problems with continuous drainage boundary conditions *Acta Geotechnica*. doi:10.1007/s11440-023-01899-0
53. Stefano Berrone, Claudio Canuto, Moreno Pintore, et al. (2023). Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks *Heliyon*. doi:10.1016/j.heliyon.2023.e18820
54. Xinquan Huang, Tariq A. Alkhalifah (2024). Microseismic Source Imaging Using Physics-Informed Neural Networks With Hard Constraints *IEEE Transactions on Geoscience and Remote Sensing*. doi:10.1109/tgrs.2024.3366449
55. Zhuorui Song (2021). Electrokinetic Flow in a Nanochannel with an Overlapped Electrical Double Layer *Digital Commons - USU (Utah State University)*. doi:10.26076/a748-e85a
56. Haosen Xu, Jianbo Zhang, Michael Hermann Eikerling, et al. (2024). Pure Water Splitting Driven by Overlapping Electric Double Layers *Journal of the American Chemical Society*. doi:10.1021/jacs.4c01070
57. Cody L. Ritt, J. Pedro de Souza, Michelle G. Barsukov, et al. (2022). Thermodynamics of Charge Regulation during Ion Transport through Silica Nanochannels *ACS Nano*. doi:10.1021/acsnano.2c06633
58. Baisakhi Pal, Poulami Dutta, Sourav Chowdhury, et al. (2026). Electroosmotic Dispersion of Uncharged Solutes through a Nanochannel *Industrial & Engineering Chemistry Research*. doi:10.1021/acs.iecr.6c00231
59. Tim M. Kamsma, Jaehyun Kim, Kyungjun Kim, et al. (2024). Brain-inspired computing with fluidic iontronic nanochannels *Proceedings of the National Academy of Sciences*. doi:10.1073/pnas.2320242121
60. Olga I. Vinogradova, Elena F. Silkina (2023). Electrophoresis of ions and electrolyte conductivity: From bulk to nanochannels *The Journal of Chemical Physics*. doi:10.1063/5.0168557