import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pnp_da_pinn.config import load_config
from pnp_da_pinn.grid import make_grid
from pnp_da_pinn.sg_solver import sg_face_flux_z
from pnp_da_pinn.conservative import ConservativePNP, solve_reference, physical_current
from pnp_da_pinn.sg_solver import annular_control_volume_weights
from pnp_da_pinn.baseline import PnpSolution
from pnp_da_pinn.scales import FARADAY


class PhysicalBenchmarks(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/surface_wall_0p036mC.yaml")
        self.config = replace(self.config, geometry=replace(self.config.geometry, nr=11, nz=25))
        self.grid = make_grid(self.config.geometry)

    def test_boltzmann_equilibrium_flux(self):
        psi = 0.2 * np.sin(np.pi * self.grid.zz)
        for sign in (-1, 1):
            flux = sg_face_flux_z(np.exp(-sign * psi), psi, self.grid, sign)
            self.assertLess(np.max(np.abs(flux)), 1e-12)

    def test_neutral_ohmic_solution(self):
        c = replace(self.config, physics=replace(self.config.physics, wall_surface_charge_strength=0))
        s = solve_reference(c, self.grid)
        np.testing.assert_allclose(s.c_plus, 1, atol=1e-10, rtol=0)
        np.testing.assert_allclose(s.c_minus, 1, atol=1e-10, rtol=0)
        voltage = c.physics.applied_voltage_V
        expected = (FARADAY * c.physics.bulk_concentration_mol_m3
                    * (c.physics.diffusion_cation_m2_s + c.physics.diffusion_anion_m2_s)
                    * np.pi * (c.geometry.radius_nm * 1e-9)**2
                    / (c.geometry.length_nm * 1e-9) * voltage / c.scales.thermal_voltage_V)
        np.testing.assert_allclose(physical_current(s, self.grid, c), expected, rtol=1e-9, atol=0)

    def test_charged_equilibrium_and_conservation(self):
        c = replace(self.config, physics=replace(self.config.physics, applied_voltage_V=0))
        s = solve_reference(c, self.grid)
        np.testing.assert_allclose(s.c_plus, np.exp(-s.psi), atol=1e-9, rtol=0)
        np.testing.assert_allclose(s.c_minus, np.exp(s.psi), atol=1e-9, rtol=0)
        self.assertLess(s.psi[-1, self.grid.z.size // 2], -1e-4)
        self.assertGreater(s.c_plus[-1, self.grid.z.size // 2], 1.0001)
        self.assertLess(np.max(np.abs(physical_current(s, self.grid, c))), 1e-20)
        s = solve_reference(self.config, self.grid)
        current = physical_current(s, self.grid, self.config)
        self.assertLess(np.ptp(current) / np.max(np.abs(current)), 1e-9)

    def test_manufactured_radial_poisson(self):
        g = self.grid
        a = -0.13
        beta = self.config.scales.poisson_coupling
        rho = np.full(g.shape, -4 * a * g.radial_metric / beta)
        problem = ConservativePNP(g, beta, 0, 2*a, fixed_charge=rho)
        exact = a * g.rr**2
        problem.load[:, 0] = exact[:, 0]
        problem.load[:, -1] = exact[:, -1]
        solved = problem.poisson_lu.solve(problem.load.ravel()).reshape(g.shape)
        np.testing.assert_allclose(solved, exact, rtol=0, atol=1e-11)

    def test_production_ionic_equilibrium_all_faces(self):
        from pnp_da_pinn.sg_solver import bernoulli
        g = self.grid
        psi = -.2 * g.rr**2 * g.zz * (1-g.zz)
        problem = ConservativePNP(g,1,0,0)
        for sign in (-1,1):
            c = problem.ions(psi,sign)
            np.testing.assert_allclose(c,np.exp(-sign*psi),rtol=0,atol=1e-11)
            delta = sign*(psi.ravel()[problem.col]-psi.ravel()[problem.row])
            face = bernoulli(delta)*c.ravel()[problem.row]-bernoulli(-delta)*c.ravel()[problem.col]
            self.assertLess(np.max(np.abs(face)),1e-12)

    def test_gauss_balance_and_species_conservation(self):
        g, c = self.grid, self.config
        s = solve_reference(c,g,tolerance=1e-11)
        w = annular_control_volume_weights(g)
        axial_out = np.sum(w * ((s.psi[:,0]-s.psi[:,1]) + (s.psi[:,-1]-s.psi[:,-2]))) / g.dz
        wall_out = g.radial_metric*c.physics.wall_surface_charge_strength*g.dz*(g.z.size-2)
        charge = c.scales.poisson_coupling * np.sum(w[:,None]*(s.c_plus-s.c_minus)[:,1:-1])*g.dz
        self.assertLess(abs(axial_out + wall_out + charge),1e-8)
        for sign,field in [(1,s.c_plus),(-1,s.c_minus)]:
            flux = np.sum(sg_face_flux_z(field,s.psi,g,sign)*w[:,None],axis=0)
            self.assertLess(np.ptp(flux)/np.max(np.abs(flux)),1e-9)

    def test_species_diffusivity_scaling(self):
        g,c = self.grid,self.config
        for species in ('plus','minus'):
            cp = 1+g.zz if species=='plus' else np.ones(g.shape)
            cm = 1+g.zz if species=='minus' else np.ones(g.shape)
            s=PnpSolution(np.zeros(g.shape),cp,cm,0,0,0,True)
            diffusion = c.physics.diffusion_cation_m2_s if species=='plus' else c.physics.diffusion_anion_m2_s
            sign = -1 if species=='plus' else 1
            expected = sign * FARADAY * c.physics.bulk_concentration_mol_m3 * diffusion * np.pi * (c.geometry.radius_nm*1e-9)**2 / (c.geometry.length_nm*1e-9)
            np.testing.assert_allclose(physical_current(s,g,c),expected,rtol=1e-12,atol=0)

    def test_salt_scaling_is_recomputed(self):
        c=replace(self.config,physics=replace(self.config.physics,bulk_concentration_mol_m3=10))
        self.assertAlmostEqual(c.scales.poisson_coupling/self.config.scales.poisson_coupling,.1,places=12)

    def test_axis_divergence_limit(self):
        import torch
        from pnp_da_pinn.pinn import axisymmetric_divergence
        r=torch.tensor([[0.],[.25]],dtype=torch.float64,requires_grad=True)
        z=torch.tensor([[.2],[.7]],dtype=torch.float64,requires_grad=True)
        result=axisymmetric_divergence(2*r,3*z,r,z,4)
        np.testing.assert_allclose(result.detach().numpy(),19,atol=1e-12,rtol=0)


if __name__ == "__main__":
    unittest.main()
