import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


torch.set_num_threads(20)


class Params:
    radius_nm = 5.0
    length_nm = 30.0
    obstacle_r_nm = 0.0
    obstacle_z_nm = 15.0
    obstacle_radius_nm = 2.0
    obstacle_potential_v = -0.025
    conductivity_s_m = 0.5
    fd_omega = 1.85
    fd_tolerance = 1e-6
    fd_max_iter = 15000


class NanoporePINN(nn.Module):
    def __init__(self, inputs=2, width=64, depth=4):
        super().__init__()
        layers = []
        in_features = inputs
        for _ in range(depth):
            layers.extend([nn.Linear(in_features, width), nn.SiLU()])
            in_features = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_grid(dr_nm, dz_nm):
    r_vec = np.arange(0, Params.radius_nm + dr_nm / 2, dr_nm)
    z_vec = np.arange(0, Params.length_nm + dz_nm / 2, dz_nm)
    z_mesh, r_mesh = np.meshgrid(z_vec, r_vec)
    obstacle = (
        (r_mesh - Params.obstacle_r_nm) ** 2
        + (z_mesh - Params.obstacle_z_nm) ** 2
        <= Params.obstacle_radius_nm**2
    )
    return r_vec, z_vec, r_mesh, z_mesh, obstacle


def solve_finite_difference(voltage_v, r_vec, z_vec, dr_nm, dz_nm, obstacle=True):
    start = time.perf_counter()
    nr, nz = len(r_vec), len(z_vec)
    z_mesh, r_mesh = np.meshgrid(z_vec, r_vec)
    obstacle_mask = (
        (r_mesh - Params.obstacle_r_nm) ** 2
        + (z_mesh - Params.obstacle_z_nm) ** 2
        <= Params.obstacle_radius_nm**2
    )
    if not obstacle:
        obstacle_mask = np.zeros_like(obstacle_mask, dtype=bool)

    phi = np.zeros((nr, nz))
    for j, z_nm in enumerate(z_vec):
        phi[:, j] = voltage_v * (1.0 - z_nm / Params.length_nm)

    iterations = Params.fd_max_iter
    for it in range(Params.fd_max_iter):
        old = phi.copy()
        for i in range(1, nr - 1):
            radial_term = 1.0 / (2.0 * r_vec[i] * dr_nm)
            for j in range(1, nz - 1):
                if obstacle_mask[i, j]:
                    continue
                update = (
                    (phi[i + 1, j] + phi[i - 1, j]) / dr_nm**2
                    + (phi[i + 1, j] - phi[i - 1, j]) * radial_term
                    + (phi[i, j + 1] + phi[i, j - 1]) / dz_nm**2
                ) / (2.0 / dr_nm**2 + 2.0 / dz_nm**2)
                phi[i, j] = (1.0 - Params.fd_omega) * phi[i, j] + Params.fd_omega * update
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = voltage_v
        phi[:, -1] = 0.0
        if obstacle:
            phi[obstacle_mask] = Params.obstacle_potential_v
        if np.max(np.abs(phi - old)) < Params.fd_tolerance:
            iterations = it + 1
            break
    return phi, time.perf_counter() - start, iterations, obstacle_mask


def current_pa(phi, r_vec, dz_nm, z_index):
    ez = -(phi[:, z_index + 1] - phi[:, z_index - 1]) / (2.0 * dz_nm * 1e-9)
    rings_m2 = 2.0 * np.pi * (r_vec[1:] * 1e-9) * ((r_vec[1] - r_vec[0]) * 1e-9)
    return float(np.sum(Params.conductivity_s_m * ez[1:] * rings_m2) * 1e12)


def ansatz_2d(model, r_nm, z_nm, voltage_v):
    phi_bg = voltage_v * (1.0 - z_nm / Params.length_nm)
    envelope = (z_nm / Params.length_nm) * (1.0 - z_nm / Params.length_nm)
    return phi_bg + model(torch.cat([r_nm / Params.radius_nm, z_nm / Params.length_nm], dim=1)) * envelope


def ansatz_v(model, r_nm, z_nm, voltage_v):
    phi_bg = voltage_v * (1.0 - z_nm / Params.length_nm)
    envelope = (z_nm / Params.length_nm) * (1.0 - z_nm / Params.length_nm)
    v_scaled = voltage_v / 0.2
    x = torch.cat([r_nm / Params.radius_nm, z_nm / Params.length_nm, v_scaled], dim=1)
    return phi_bg + model(x) * envelope


def pde_residual(predict_fn, model, r_nm, z_nm, voltage_v):
    r_nm.requires_grad_(True)
    z_nm.requires_grad_(True)
    phi = predict_fn(model, r_nm, z_nm, voltage_v)
    dphi_dr = torch.autograd.grad(phi, r_nm, torch.ones_like(phi), create_graph=True)[0]
    dphi_dz = torch.autograd.grad(phi, z_nm, torch.ones_like(phi), create_graph=True)[0]
    d2phi_dr2 = torch.autograd.grad(dphi_dr, r_nm, torch.ones_like(dphi_dr), create_graph=True)[0]
    d2phi_dz2 = torch.autograd.grad(dphi_dz, z_nm, torch.ones_like(dphi_dz), create_graph=True)[0]
    r_safe = torch.where(r_nm == 0, torch.full_like(r_nm, 1e-4), r_nm)
    return d2phi_dr2 + (1.0 / r_safe) * dphi_dr + d2phi_dz2


def sample_training_points(rng, r_vec, z_vec, phi, n_data, voltage_v):
    nr, nz = len(r_vec), len(z_vec)
    idx_r = rng.integers(0, nr, n_data)
    idx_z = rng.integers(0, nz, n_data)
    return (
        torch.tensor(r_vec[idx_r].reshape(-1, 1), dtype=torch.float32),
        torch.tensor(z_vec[idx_z].reshape(-1, 1), dtype=torch.float32),
        torch.full((n_data, 1), float(voltage_v), dtype=torch.float32),
        torch.tensor(phi[idx_r, idx_z].reshape(-1, 1), dtype=torch.float32),
    )


def train_single_voltage(phi_fd, r_vec, z_vec, obstacle_mask, voltage_v, epochs, seed):
    rng = np.random.default_rng(seed)
    model = NanoporePINN(inputs=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    mse = nn.MSELoss()

    r_col = rng.uniform(0, Params.radius_nm, (8000, 1))
    z_col = rng.uniform(0, Params.length_nm, (8000, 1))
    outside = (
        (r_col - Params.obstacle_r_nm) ** 2 + (z_col - Params.obstacle_z_nm) ** 2
        > Params.obstacle_radius_nm**2
    ).reshape(-1)
    r_col_t = torch.tensor(r_col[outside].reshape(-1, 1), dtype=torch.float32)
    z_col_t = torch.tensor(z_col[outside].reshape(-1, 1), dtype=torch.float32)
    v_col_t = torch.full_like(r_col_t, float(voltage_v))

    obs_r = rng.uniform(0, Params.obstacle_radius_nm, (2500, 1))
    obs_z = rng.uniform(
        Params.obstacle_z_nm - Params.obstacle_radius_nm,
        Params.obstacle_z_nm + Params.obstacle_radius_nm,
        (2500, 1),
    )
    obs_inside = (
        (obs_r - Params.obstacle_r_nm) ** 2 + (obs_z - Params.obstacle_z_nm) ** 2
        <= Params.obstacle_radius_nm**2
    ).reshape(-1)
    obs_r_t = torch.tensor(obs_r[obs_inside].reshape(-1, 1), dtype=torch.float32)
    obs_z_t = torch.tensor(obs_z[obs_inside].reshape(-1, 1), dtype=torch.float32)
    obs_v_t = torch.full_like(obs_r_t, float(voltage_v))

    wall_r = torch.full((1000, 1), Params.radius_nm, dtype=torch.float32, requires_grad=True)
    wall_z = torch.tensor(rng.uniform(0, Params.length_nm, (1000, 1)), dtype=torch.float32)
    wall_v = torch.full_like(wall_r, float(voltage_v))
    sym_r = torch.zeros((1000, 1), dtype=torch.float32, requires_grad=True)
    sym_z = torch.tensor(rng.uniform(0, Params.length_nm, (1000, 1)), dtype=torch.float32)
    sym_v = torch.full_like(sym_r, float(voltage_v))

    r_data, z_data, v_data, phi_data = sample_training_points(rng, r_vec, z_vec, phi_fd, 1500, voltage_v)

    start = time.perf_counter()
    history = []
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        res = pde_residual(ansatz_2d, model, r_col_t, z_col_t, v_col_t)
        loss_pde = mse(res, torch.zeros_like(res))
        loss_obs = mse(ansatz_2d(model, obs_r_t, obs_z_t, obs_v_t), torch.full_like(obs_r_t, Params.obstacle_potential_v))
        loss_data = mse(ansatz_2d(model, r_data, z_data, v_data), phi_data)

        phi_wall = ansatz_2d(model, wall_r, wall_z, wall_v)
        dphi_dr_wall = torch.autograd.grad(phi_wall, wall_r, torch.ones_like(phi_wall), create_graph=True)[0]
        phi_sym = ansatz_2d(model, sym_r, sym_z, sym_v)
        dphi_dr_sym = torch.autograd.grad(phi_sym, sym_r, torch.ones_like(phi_sym), create_graph=True)[0]
        loss_neumann = mse(dphi_dr_wall, torch.zeros_like(dphi_dr_wall)) + mse(dphi_dr_sym, torch.zeros_like(dphi_dr_sym))

        loss = loss_pde + 100.0 * loss_obs + 50.0 * loss_neumann + 500.0 * loss_data
        loss.backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": float(loss.detach())})

    r_mesh, z_mesh = np.meshgrid(r_vec, z_vec, indexing="ij")
    r_flat = torch.tensor(r_mesh.reshape(-1, 1), dtype=torch.float32)
    z_flat = torch.tensor(z_mesh.reshape(-1, 1), dtype=torch.float32)
    v_flat = torch.full_like(r_flat, float(voltage_v))
    infer_start = time.perf_counter()
    with torch.no_grad():
        phi_pred = ansatz_2d(model, r_flat, z_flat, v_flat).numpy().reshape(len(r_vec), len(z_vec))
    inference_time = time.perf_counter() - infer_start
    phi_pred[obstacle_mask] = Params.obstacle_potential_v
    return model, phi_pred, time.perf_counter() - start, inference_time, pd.DataFrame(history)


def train_voltage_conditioned(fd_solutions, r_vec, z_vec, epochs, seed):
    rng = np.random.default_rng(seed)
    model = NanoporePINN(inputs=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    mse = nn.MSELoss()

    data_chunks = [sample_training_points(rng, r_vec, z_vec, phi, 600, v) for v, phi in fd_solutions.items()]
    r_data = torch.cat([chunk[0] for chunk in data_chunks])
    z_data = torch.cat([chunk[1] for chunk in data_chunks])
    v_data = torch.cat([chunk[2] for chunk in data_chunks])
    phi_data = torch.cat([chunk[3] for chunk in data_chunks])

    n_col = 9000
    r_col = rng.uniform(0, Params.radius_nm, (n_col, 1))
    z_col = rng.uniform(0, Params.length_nm, (n_col, 1))
    v_col = rng.choice(np.array(list(fd_solutions.keys())), size=(n_col, 1))
    outside = (
        (r_col - Params.obstacle_r_nm) ** 2 + (z_col - Params.obstacle_z_nm) ** 2
        > Params.obstacle_radius_nm**2
    ).reshape(-1)
    r_col_t = torch.tensor(r_col[outside].reshape(-1, 1), dtype=torch.float32)
    z_col_t = torch.tensor(z_col[outside].reshape(-1, 1), dtype=torch.float32)
    v_col_t = torch.tensor(v_col[outside].reshape(-1, 1), dtype=torch.float32)

    obs_r = rng.uniform(0, Params.obstacle_radius_nm, (2500, 1))
    obs_z = rng.uniform(
        Params.obstacle_z_nm - Params.obstacle_radius_nm,
        Params.obstacle_z_nm + Params.obstacle_radius_nm,
        (2500, 1),
    )
    obs_inside = (
        (obs_r - Params.obstacle_r_nm) ** 2 + (obs_z - Params.obstacle_z_nm) ** 2
        <= Params.obstacle_radius_nm**2
    ).reshape(-1)
    obs_r_t = torch.tensor(obs_r[obs_inside].reshape(-1, 1), dtype=torch.float32)
    obs_z_t = torch.tensor(obs_z[obs_inside].reshape(-1, 1), dtype=torch.float32)
    obs_v_t = torch.tensor(rng.choice(np.array(list(fd_solutions.keys())), size=(len(obs_r_t), 1)), dtype=torch.float32)

    start = time.perf_counter()
    history = []
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        res = pde_residual(ansatz_v, model, r_col_t, z_col_t, v_col_t)
        loss_pde = mse(res, torch.zeros_like(res))
        loss_obs = mse(ansatz_v(model, obs_r_t, obs_z_t, obs_v_t), torch.full_like(obs_r_t, Params.obstacle_potential_v))
        loss_data = mse(ansatz_v(model, r_data, z_data, v_data), phi_data)
        loss = loss_pde + 100.0 * loss_obs + 600.0 * loss_data
        loss.backward()
        optimizer.step()
        if epoch % 250 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": float(loss.detach())})
    return model, time.perf_counter() - start, pd.DataFrame(history)


def predict_grid_voltage(model, r_vec, z_vec, voltage_v, obstacle_mask):
    r_mesh, z_mesh = np.meshgrid(r_vec, z_vec, indexing="ij")
    r_flat = torch.tensor(r_mesh.reshape(-1, 1), dtype=torch.float32)
    z_flat = torch.tensor(z_mesh.reshape(-1, 1), dtype=torch.float32)
    v_flat = torch.full_like(r_flat, float(voltage_v))
    start = time.perf_counter()
    with torch.no_grad():
        phi = ansatz_v(model, r_flat, z_flat, v_flat).numpy().reshape(len(r_vec), len(z_vec))
    elapsed = time.perf_counter() - start
    phi[obstacle_mask] = Params.obstacle_potential_v
    return phi, elapsed


def make_figures(r_vec, z_vec, phi_fd, phi_pinn, obstacle_mask, summary, iv_table, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    z_mesh, r_mesh = np.meshgrid(z_vec, r_vec)
    ez_fd, er_fd = np.gradient(-phi_fd, z_vec[1] - z_vec[0], r_vec[1] - r_vec[0])
    ez_pinn, er_pinn = np.gradient(-phi_pinn, z_vec[1] - z_vec[0], r_vec[1] - r_vec[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    levels = np.linspace(min(phi_fd.min(), phi_pinn.min()), max(phi_fd.max(), phi_pinn.max()), 60)
    for ax, phi, ez, er, title in [
        (axes[0, 0], phi_fd, ez_fd, er_fd, "A. Finite-difference ground truth"),
        (axes[0, 1], phi_pinn, ez_pinn, er_pinn, "B. DA-PINN prediction"),
    ]:
        c = ax.contourf(z_mesh, r_mesh, phi, levels=levels, cmap="magma")
        ax.streamplot(z_vec, r_vec, ez, er, color="white", linewidth=0.45, density=1.0, arrowsize=0.7)
        ax.add_patch(plt.Circle((Params.obstacle_z_nm, Params.obstacle_r_nm), Params.obstacle_radius_nm, color="cyan", fill=False, lw=1.6))
        ax.set_title(title)
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("r (nm)")
        ax.set_xlim(0, Params.length_nm)
        ax.set_ylim(0, Params.radius_nm)
        fig.colorbar(c, ax=ax, label="Potential (V)")

    err = np.abs(phi_fd - phi_pinn)
    err[obstacle_mask] = 0.0
    c = axes[1, 0].contourf(z_mesh, r_mesh, err, levels=60, cmap="inferno")
    axes[1, 0].add_patch(plt.Circle((Params.obstacle_z_nm, Params.obstacle_r_nm), Params.obstacle_radius_nm, color="white", fill=False, lw=1.2, ls="--"))
    axes[1, 0].set_title("C. Absolute error")
    axes[1, 0].set_xlabel("z (nm)")
    axes[1, 0].set_ylabel("r (nm)")
    axes[1, 0].set_xlim(0, Params.length_nm)
    axes[1, 0].set_ylim(0, Params.radius_nm)
    fig.colorbar(c, ax=axes[1, 0], label="|error| (V)")

    axes[1, 1].plot(z_vec, phi_fd[0, :], color="black", lw=2, label="FD, r=0 nm")
    axes[1, 1].plot(z_vec, phi_pinn[0, :], color="crimson", lw=1.8, ls="--", label="DA-PINN, r=0 nm")
    wall_index = np.argmin(np.abs(r_vec - 4.0))
    axes[1, 1].plot(z_vec, phi_fd[wall_index, :], color="0.45", lw=2, label="FD, r=4 nm")
    axes[1, 1].plot(z_vec, phi_pinn[wall_index, :], color="royalblue", lw=1.8, ls="--", label="DA-PINN, r=4 nm")
    axes[1, 1].set_title("D. Axial potential sections")
    axes[1, 1].set_xlabel("z (nm)")
    axes[1, 1].set_ylabel("Potential (V)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "field_comparison_analysis.png", dpi=500)
    fig.savefig(out_dir / "field_comparison_analysis.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    compare = summary[summary["analysis"].isin(["empty_fd", "bound_fd", "bound_da_pinn"])]
    axes[0].bar(compare["label"], compare["current_pA"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_ylabel("Current (pA)")
    axes[0].set_title("A. Empty vs bound nanopore")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].plot(iv_table["voltage_mV"], iv_table["fd_current_pA"], "o-", color="black", label="FD")
    axes[1].plot(iv_table["voltage_mV"], iv_table["da_pinn_current_pA"], "s--", color="crimson", label="Voltage-conditioned DA-PINN")
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].axvline(0, color="0.7", lw=0.8)
    axes[1].set_xlabel("Applied voltage (mV)")
    axes[1].set_ylabel("Current (pA)")
    axes[1].set_title("B. I-V response")
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "current_validation_analysis.png", dpi=500)
    fig.savefig(out_dir / "current_validation_analysis.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dr_nm = dz_nm = 0.2 if args.fast else 0.1
    epochs = min(args.epochs, 300) if args.fast else args.epochs
    r_vec, z_vec, _, _, obstacle_mask = make_grid(dr_nm, dz_nm)
    z_eval_idx = int(round(5.0 / dz_nm))

    fd_bound, fd_bound_time, fd_bound_iter, obstacle_mask = solve_finite_difference(0.1, r_vec, z_vec, dr_nm, dz_nm, obstacle=True)
    fd_empty, fd_empty_time, fd_empty_iter, _ = solve_finite_difference(0.1, r_vec, z_vec, dr_nm, dz_nm, obstacle=False)

    model_2d, phi_pinn, train_time, infer_time, hist_2d = train_single_voltage(
        fd_bound, r_vec, z_vec, obstacle_mask, 0.1, epochs, args.seed
    )

    bound_fd_current = current_pa(fd_bound, r_vec, dz_nm, z_eval_idx)
    empty_fd_current = current_pa(fd_empty, r_vec, dz_nm, z_eval_idx)
    bound_pinn_current = current_pa(phi_pinn, r_vec, dz_nm, z_eval_idx)
    current_error = abs(bound_fd_current - bound_pinn_current) / abs(bound_fd_current) * 100.0
    delta_i = bound_fd_current - empty_fd_current

    iv_voltages = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    fd_iv = {}
    fd_iv_rows = []
    for voltage in iv_voltages:
        phi, elapsed, iterations, _ = solve_finite_difference(float(voltage), r_vec, z_vec, dr_nm, dz_nm, obstacle=True)
        fd_iv[float(voltage)] = phi
        fd_iv_rows.append(
            {
                "voltage_mV": voltage * 1000.0,
                "fd_current_pA": current_pa(phi, r_vec, dz_nm, z_eval_idx),
                "fd_time_s": elapsed,
                "fd_iterations": iterations,
            }
        )

    v_model, v_train_time, hist_v = train_voltage_conditioned(fd_iv, r_vec, z_vec, epochs, args.seed + 1)
    iv_rows = []
    for row in fd_iv_rows:
        phi_v, v_infer = predict_grid_voltage(v_model, r_vec, z_vec, row["voltage_mV"] / 1000.0, obstacle_mask)
        da_current = current_pa(phi_v, r_vec, dz_nm, z_eval_idx)
        iv_result = dict(row)
        iv_result["da_pinn_current_pA"] = da_current
        iv_result["da_pinn_inference_s"] = v_infer
        iv_result["absolute_error_percent"] = (
            abs(row["fd_current_pA"] - da_current) / max(abs(row["fd_current_pA"]), 1e-12) * 100.0
        )
        iv_rows.append(iv_result)

    summary = pd.DataFrame(
        [
            {
                "analysis": "empty_fd",
                "label": "Empty FD",
                "current_pA": empty_fd_current,
                "runtime_s": fd_empty_time,
                "iterations": fd_empty_iter,
                "absolute_error_percent": np.nan,
            },
            {
                "analysis": "bound_fd",
                "label": "Bound FD",
                "current_pA": bound_fd_current,
                "runtime_s": fd_bound_time,
                "iterations": fd_bound_iter,
                "absolute_error_percent": np.nan,
            },
            {
                "analysis": "bound_da_pinn",
                "label": "Bound DA-PINN",
                "current_pA": bound_pinn_current,
                "runtime_s": infer_time,
                "iterations": np.nan,
                "absolute_error_percent": current_error,
            },
        ]
    )
    summary["delta_bound_minus_empty_pA"] = delta_i
    summary["single_voltage_training_s"] = train_time
    summary["voltage_conditioned_training_s"] = v_train_time
    summary.to_csv(data_dir / "benchmark_summary.csv", index=False)

    iv_table = pd.DataFrame(iv_rows)
    iv_table.to_csv(data_dir / "iv_curve_validation.csv", index=False)
    hist_2d.to_csv(data_dir / "training_history_single_voltage.csv", index=False)
    hist_v.to_csv(data_dir / "training_history_voltage_conditioned.csv", index=False)
    np.save(data_dir / "phi_fd_bound_100mV.npy", fd_bound)
    np.save(data_dir / "phi_da_pinn_bound_100mV.npy", phi_pinn)

    make_figures(r_vec, z_vec, fd_bound, phi_pinn, obstacle_mask, summary, iv_table, fig_dir)

    print(summary.to_string(index=False))
    print(iv_table.to_string(index=False))
    print(f"Outputs written to {root}")


if __name__ == "__main__":
    main()
