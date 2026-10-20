"""Reproducible training with reservoir, axis and blocking-wall conditions."""
import time
import numpy as np
import torch

from .pinn import PnpPINN, AnsatzConfig, predict_fields, pnp_residuals, grad
from .training import TrainingResult
from .scales import voltage_to_psi


def train_reference_surrogate(solution, grid, config, use_data=True, seed=42,
                              steps=500, wall_adaptive=False):
    torch.manual_seed(seed)
    dtype = torch.float64
    model = PnpPINN().to(dtype=dtype)
    with torch.no_grad():
        model.net[-1].weight.mul_(0.01)
        model.net[-1].bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    n = config.pinn.collocation_points
    rc = rng.random((n, 1))
    if wall_adaptive:
        rc[:n // 2] = 0.8 + 0.2 * rng.random((n // 2, 1))
    r = torch.tensor(rc, dtype=dtype)
    z = torch.tensor(rng.random((n, 1)), dtype=dtype)
    data_rng = np.random.default_rng(seed + 1)
    # Sampling without replacement makes the held-out set explicit.
    indices = data_rng.choice(grid.rr.size, config.pinn.data_points, replace=False)
    rd = torch.tensor(grid.rr.ravel()[indices, None], dtype=dtype)
    zd = torch.tensor(grid.zz.ravel()[indices, None], dtype=dtype)
    truth = np.stack((solution.psi, solution.c_plus, solution.c_minus), axis=-1).reshape(-1, 3)
    y = torch.tensor(truth[indices], dtype=dtype)
    scale = y.std(dim=0, keepdim=True).clamp_min(1e-3)
    charge = y[:, 1:2] - y[:, 2:3]
    charge_scale = charge.std().clamp_min(1e-5)
    ansatz = AnsatzConfig(voltage_to_psi(config.physics.applied_voltage_V, config.scales))
    beta = config.scales.poisson_coupling
    q = config.physics.wall_surface_charge_strength
    zb = torch.linspace(0.01, 0.99, 128, dtype=dtype).reshape(-1, 1)
    history = []
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad()
        res = pnp_residuals(model, r, z, ansatz, beta, grid.radial_metric)
        physics = (res['poisson'] / max(beta, 1)).square().mean()
        physics += (res['np_plus'] / max(grid.radial_metric, 1)).square().mean()
        physics += (res['np_minus'] / max(grid.radial_metric, 1)).square().mean()
        boundary = torch.zeros((), dtype=dtype)
        for location in (0., 1.):
            rb = torch.full_like(zb, location, requires_grad=True)
            pb, cpb, cmb = predict_fields(model, rb, zb, ansatz)
            dp = grad(pb, rb)
            boundary += ((dp - (q if location else 0)) / max(abs(q), 0.01)).square().mean()
            boundary += ((grad(cpb, rb) + cpb * dp) / max(abs(q), 0.01)).square().mean()
            boundary += ((grad(cmb, rb) - cmb * dp) / max(abs(q), 0.01)).square().mean()
        data_loss = torch.zeros((), dtype=dtype)
        charge_loss = torch.zeros((), dtype=dtype)
        if use_data:
            pred = torch.cat(predict_fields(model, rd, zd, ansatz), dim=1)
            data_loss = ((pred - y) / scale).square().mean()
            charge_loss = ((pred[:, 1:2] - pred[:, 2:3] - charge) / charge_scale).square().mean()
        loss = physics + boundary + 10 * data_loss + 20 * charge_loss
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append(dict(step=step + 1, loss=float(loss.detach()), physics=float(physics.detach()),
                                boundary=float(boundary.detach()), data=float(data_loss.detach()),
                                charge=float(charge_loss.detach())))
    return TrainingResult(model, history, time.perf_counter() - start), indices


def predict(model, grid, config):
    r = torch.tensor(grid.rr.reshape(-1, 1), dtype=torch.float64)
    z = torch.tensor(grid.zz.reshape(-1, 1), dtype=torch.float64)
    ansatz = AnsatzConfig(voltage_to_psi(config.physics.applied_voltage_V, config.scales))
    with torch.no_grad():
        arrays = predict_fields(model, r, z, ansatz)
    return tuple(x.numpy().reshape(grid.shape) for x in arrays)
