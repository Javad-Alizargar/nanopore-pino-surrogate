# Numerical Solver Notes

## Current status

Two baselines exist:

1. `baseline.py`: prototype relaxed Gummel/Boltzmann solver. This is retained only as a negative comparison and plumbing check. It does not conserve current acceptably.
2. `sg_solver.py`: finite-volume/Scharfetter-Gummel Nernst-Planck solver coupled to a linear Poisson solve. This is the path toward paper-grade baseline data.

The SG solver now includes:

- numerically clipped Bernoulli function,
- fixed-charge continuation,
- adaptive under-relaxation within each continuation stage,
- matrix-rank failure detection.
- geometry-aware radial operators using `(L/R)^2` for a normalized 2D axisymmetric grid,
- optional Debye-Huckel radial EDL initialization for charged-wall continuation,
- Anderson acceleration for pure surface-charge stages,
- Scharfetter-Gummel zero electrochemical-flux ion boundary at the charged wall.
- annular finite-volume current integration for axisymmetric current diagnostics.

## Smoke results

Neutral mild smoke case (`configs/smoke.yaml`):

- SG converged in 1 iteration.
- Dimensionless current is constant across z to numerical precision.
- SG flux conservation error: approximately 3e-14.
- After adding geometry-aware radial operators, neutral current conservation remains at numerical precision.
- Prototype conservation error: approximately 1.0.

DA-PINN smoke training against the SG baseline:

- psi relative L2: approximately 0.035
- c_plus relative L2: approximately 0.313 after 20 epochs
- c_minus relative L2: approximately 0.317 after 20 epochs
- inference: approximately 0.0012 s

This confirms the code path but is not yet a publishable physical regime because neutral PNP with equal reservoir concentrations is close to a trivial concentration field.

## Charged-wall development cases

`configs/charged_wall_low_salt.yaml` is stable enough for development. Current conservation is below 1e-4, but convergence is slow and concentration perturbations are small.

`configs/charged_wall.yaml` remains a stress test. At 100 mM and charged-wall forcing, the current SG/Gummel continuation develops large potential excursions and poor current conservation. This case is not usable for paper figures yet.

## Required production upgrade

Before generating manuscript figures, implement at least one of the following:

1. stronger continuation in fixed charge, voltage, and bulk concentration,
2. adaptive Gummel damping based on actual nonlinear residual decrease,
3. damped Newton coupling for Poisson and SG NP solves,
4. calibrated surface-charge boundary condition instead of a volume charge shell,
5. independent validation against a known PNP/nanofluidic benchmark.

## Acceptance criteria for production baseline

Before training final DA-PINN models:

1. Gummel/Newton iterations converge for at least 10 mM and 100 mM cases.
2. Charged-wall or biomolecular-interface cases show nontrivial cation/anion redistribution.
3. Axial current conservation error is below 1-5% in the pore interior.
4. Concentrations remain positive without clipping dominating the solution.
5. Grid-refinement study changes current by less than 2-5%.
6. DA-PINN is trained against the conservative SG/FV baseline, not the prototype solver.

## Continuation diagnostics

A bulk-voltage-charge continuation driver was added in `src/pnp_da_pinn/continuation.py` and `scripts/run_continuation.py`.

For `charged_wall.yaml` with target 100 mM, 50 mV, and fixed-charge shell strength -1e-6:

- voltage continuation at low salt succeeds,
- bulk continuation from 1 to 100 mol/m3 succeeds while charge is zero,
- fixed-charge continuation fails between approximately -1.3e-8 and -2.5e-8 in the current volume-charge shell formulation,
- failure mode is large potential excursion and loss of current conservation.

Interpretation: the current volume fixed-charge shell is not a robust production representation of charged nanopore walls at high salt. The next solver-development step should be either a true surface-charge Neumann boundary condition or damped Newton coupling. Continuing to tune scalar damping is unlikely to produce a defensible full-article baseline.

## Surface-charge boundary update

A dimensionless wall surface-charge Neumann condition was added to the SG Poisson solve:

psi_wall - psi_inner = q_wall * dr

Ion concentrations now use a Scharfetter-Gummel zero electrochemical-flux wall condition. This is more physically appropriate for charged nanopore walls than either the earlier thin volume-charge shell or a zero concentration-gradient wall.

For the normalized radial coordinate, the approximate physical conversion is:

sigma = -q_wall * epsilon * V_T / R_pore

For the 5 nm pore radius used here, q_wall = -0.28 corresponds to about +1.0 mC/m2.

Diagnostics:

- Neutral smoke remains stable and current-conserving.
- Very weak charged-wall cases remain stable, but they are not physically compelling.
- The 0.036 mC/m2 scout (`q_wall=-0.01`) now reaches later continuation stages with Anderson acceleration, but current conservation is still not paper-grade for the charged-wall case.
- The 0.36 mC/m2 and 1.0 mC/m2 calibrated cases are currently nonlinear stress tests, not production figure sources.
- Anderson no-convergence candidates are now sanity-gated before fallback because ill-conditioned accelerated states can be worse than the stable previous iterate.
- A bounded pseudo-transient stage solver was added for charged-wall states. It preserves low axial current-conservation error much better than plain damped Picard.
- Development diagnostics now distinguish strict convergence from `accepted_for_continuation`, so parameter-envelope scouting can proceed without mislabeling quasi-steady stages as fully converged.
- Micro-scout result for `q_wall=-0.001` (approximately 0.0036 mC/m2): reaches 100 mM with residual about 3.1e-2 and current conservation about 1.2e-3.
- Micro-scout result for `q_wall=-0.01` (approximately 0.036 mC/m2): reaches 31.6 mM in accepted scout mode; at 100 mM, current conservation remains about 1.1e-2 but residual rises to about 3.1e-1.
- A reduced potential-only Krylov solve was added after pseudo-transient stabilization. Boundary degrees of freedom are eliminated from the nonlinear unknown vector before solving the self-consistent SG/Poisson fixed point.
- With this reduced Krylov stage, `q_wall=-0.01` at 100 mM now reaches strict convergence with residual about 5.9e-4 and current-conservation error about 3e-6.
- The low-salt surface-charge ramp stages are still marked as `accepted_for_continuation` rather than strict convergence, but the final 100 mM target state is strict.

Decision: the boundary representation and geometry are now much more defensible, but the production blocker is still the nonlinear coupled solve in realistic charged-wall regimes. Next implementation should focus on robust nonlinear acceleration and a grid-refinement/current-conservation benchmark before generating full article figures.

## Next numerical decision

The highest-return next step is to replace the current black-box Anderson call with a bounded pseudo-transient continuation method:

1. use smaller surface-charge increments,
2. accept a stage only when both nonlinear residual and axial current conservation decrease,
3. adapt the pseudo-time step/relaxation instead of relying on a fixed scalar damping,
4. store the best residual/current-conservation state for each stage.

This is more publishable than tuning the existing fixed-point loop because it creates a reproducible convergence protocol that can be reported in the Methods section.

## DA-PINN development benchmark

`scripts/run_charged_dev_benchmark.py` now runs the current low-charge development case (`surface_wall_0p0036mC.yaml`) through:

1. micro surface-charge/bulk continuation,
2. DA-PINN training with exact potential and concentration ansatz,
3. charge-aware sparse data assimilation,
4. field/current metric export,
5. diagnostic figure export.

Important PINN correction: the concentration ansatz now uses an exponential envelope so reservoir concentration boundaries remain exactly equal to 1. The previous softplus transform violated this boundary because `softplus(1) != 1`.

Current charged-development metrics:

- potential relative L2: about 5e-4,
- cation and anion relative L2: about 2e-4,
- current relative L2: about 9e-4,
- DA-PINN inference: about 1.6 ms,
- SG current-conservation error: about 1.2e-3,
- SG residual: about 3.1e-2, accepted for continuation but not strict convergence,
- net charge-density relative L2: about 0.39 because the low-charge signal is extremely small.

Decision: this is a good reproducible pipeline benchmark and figure-plumbing case, but not a headline charged-biointerface result. The article's novelty still depends on improving the stronger charged-wall solver envelope or adding a second, independently validated benchmark.

## Stronger charged-wall benchmark

`surface_wall_0p036mC.yaml` is now the preferred candidate for the next DA-PINN benchmark:

- final 100 mM SG residual: about 5.9e-4,
- final SG current-conservation error: about 3e-6,
- DA-PINN potential relative L2: about 6.9e-3,
- DA-PINN cation/anion relative L2: about 2.0e-3,
- DA-PINN current relative L2: about 1.0e-2,
- DA-PINN inference: about 1.5 ms.

Remaining limitation: the DA-PINN still under-resolves the detailed net charge-density pattern for the stronger surface-charge case. This should be addressed before using charge-density maps as a headline result. Potential, concentration, and current predictions are already promising.

## Grid-refinement check

`scripts/run_grid_refinement.py` was added to compare the final current across grids for the stronger charged-wall case.

Quick refinement result for `surface_wall_0p036mC.yaml`:

- 15 x 45 grid: residual about 5.6e-4, current-conservation error about 4e-6,
- 21 x 61 grid: residual about 5.9e-4, current-conservation error about 3e-6,
- relative current change from 15 x 45 to 21 x 61: about 5e-8.
- 31 x 91 grid: residual about 5.9e-4, current-conservation error about 2e-6,
- relative current change from 21 x 61 to 31 x 91: about 3e-9.

Interpretation: the final current is extremely stable across three grids. This strongly supports using the reduced-Krylov SG/FV solution as the reference for current and potential benchmarks, and removes a major desk-rejection risk.
