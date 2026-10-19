from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn

from .baseline import PnpSolution
from .config import ExperimentConfig
from .grid import AxisymmetricGrid
from .pinn import AnsatzConfig, PnpPINN, pnp_residuals, predict_fields
from .scales import voltage_to_psi


@dataclass
class TrainingResult:
    model: PnpPINN
    history: list[dict[str, float]]
    runtime_s: float


def sample_data_from_solution(
    solution: PnpSolution,
    grid: AxisymmetricGrid,
    n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx_r = rng.integers(0, len(grid.r), size=n)
    idx_z = rng.integers(0, len(grid.z), size=n)
    r = torch.tensor(grid.r[idx_r].reshape(-1, 1), dtype=torch.float32)
    z = torch.tensor(grid.z[idx_z].reshape(-1, 1), dtype=torch.float32)
    fields = np.stack(
        [solution.psi[idx_r, idx_z], solution.c_plus[idx_r, idx_z], solution.c_minus[idx_r, idx_z]],
        axis=1,
    )
    y = torch.tensor(fields, dtype=torch.float32)
    return r, z, y


def sample_training_collocation(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((n, 1), generator=generator), torch.rand((n, 1), generator=generator)


def train_da_pinn(
    solution: PnpSolution,
    grid: AxisymmetricGrid,
    config: ExperimentConfig,
    use_data: bool = True,
    exact_ansatz: bool = True,
) -> TrainingResult:
    torch.manual_seed(config.pinn.seed)
    model = PnpPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    mse = nn.MSELoss()

    beta = config.scales.poisson_coupling
    left_psi = voltage_to_psi(config.physics.applied_voltage_V, config.scales)
    ansatz = AnsatzConfig(left_psi=left_psi)
    r_col, z_col = sample_training_collocation(config.pinn.collocation_points, config.pinn.seed)
    r_data, z_data, y_data = sample_data_from_solution(
        solution, grid, config.pinn.data_points, config.pinn.seed + 1
    )
    data_scale = torch.std(y_data, dim=0, keepdim=True).clamp_min(1e-3)
    charge_data = y_data[:, 1:2] - y_data[:, 2:3]
    charge_scale = torch.std(charge_data, dim=0, keepdim=True).clamp_min(1e-5)

    start = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(config.pinn.epochs + 1):
        optimizer.zero_grad()
        residual = pnp_residuals(model, r_col, z_col, ansatz, beta, grid.radial_metric)
        loss_poisson = mse(residual["poisson"], torch.zeros_like(residual["poisson"]))
        loss_np_plus = mse(residual["np_plus"], torch.zeros_like(residual["np_plus"]))
        loss_np_minus = mse(residual["np_minus"], torch.zeros_like(residual["np_minus"]))
        loss = loss_poisson + loss_np_plus + loss_np_minus

        if use_data:
            psi_d, cp_d, cm_d = predict_fields(model, r_data, z_data, ansatz)
            pred = torch.cat([psi_d, cp_d, cm_d], dim=1)
            loss_data = mse((pred - y_data) / data_scale, torch.zeros_like(pred))
            pred_charge = cp_d - cm_d
            loss_charge_data = mse(
                (pred_charge - charge_data) / charge_scale,
                torch.zeros_like(pred_charge),
            )
            loss = loss + 10.0 * loss_data + 20.0 * loss_charge_data
        else:
            loss_data = torch.tensor(0.0)
            loss_charge_data = torch.tensor(0.0)

        # Placeholder boundary penalty for future standard-PINN comparison.
        if not exact_ansatz:
            raise NotImplementedError("Standard PINN boundary-penalty mode is planned but not enabled.")

        loss.backward()
        optimizer.step()

        if epoch % max(1, config.pinn.epochs // 5) == 0 or epoch == config.pinn.epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "loss_poisson": float(loss_poisson.detach()),
                    "loss_np_plus": float(loss_np_plus.detach()),
                    "loss_np_minus": float(loss_np_minus.detach()),
                    "loss_data": float(loss_data.detach()),
                    "loss_charge_data": float(loss_charge_data.detach()),
                }
            )

    return TrainingResult(model=model, history=history, runtime_s=time.perf_counter() - start)


def predict_on_grid(model: PnpPINN, grid: AxisymmetricGrid, config: ExperimentConfig):
    left_psi = voltage_to_psi(config.physics.applied_voltage_V, config.scales)
    ansatz = AnsatzConfig(left_psi=left_psi)
    r = torch.tensor(grid.rr.reshape(-1, 1), dtype=torch.float32)
    z = torch.tensor(grid.zz.reshape(-1, 1), dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        psi, cp, cm = predict_fields(model, r, z, ansatz)
    runtime_s = time.perf_counter() - start
    return (
        psi.numpy().reshape(grid.shape),
        cp.numpy().reshape(grid.shape),
        cm.numpy().reshape(grid.shape),
        runtime_s,
    )
