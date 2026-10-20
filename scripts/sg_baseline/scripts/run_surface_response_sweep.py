from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.baseline import PnpSolution, current_scale_A  # noqa: E402
from pnp_da_pinn.config import ExperimentConfig, load_config  # noqa: E402
from pnp_da_pinn.continuation import run_bulk_voltage_charge_continuation  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.sg_solver import sg_current_profile, sg_flux_conservation_error  # noqa: E402


CASE = "surface_wall_0p036mC.yaml"
OUT_DIR = ROOT / "outputs" / "surface_response"
FIG_DIR = ROOT / "manuscript" / "figures"


def run_reference(config: ExperimentConfig, mode: str) -> tuple[PnpSolution, object, float]:
    grid = make_grid(config.geometry)
    if mode == "quick":
        bulk_stages = 3
        voltage_stages = 2
        charge_stages = 2
        max_iter = 10
        tolerance = 2e-3
    else:
        bulk_stages = 5
        voltage_stages = 3
        charge_stages = 3
        max_iter = 18
        tolerance = 1e-3
    result = run_bulk_voltage_charge_continuation(
        config=config,
        grid=grid,
        start_bulk_mol_m3=1.0,
        bulk_stages=bulk_stages,
        voltage_stages=voltage_stages,
        charge_stages=charge_stages,
        max_iter_per_stage=max_iter,
        tolerance=tolerance,
        relaxation=0.02,
        schedule_mode="surface_then_bulk",
    )
    runtime = float(sum(row["runtime_s"] for row in result.diagnostics))
    return result.solution, grid, runtime


def panel_label(ax, label: str) -> None:
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
    )


def summarize_solution(
    label: str,
    voltage_v: float,
    wall_charge: float,
    config: ExperimentConfig,
    solution: PnpSolution,
    grid,
    runtime_s: float,
) -> dict[str, float | str | bool]:
    current_dimless = sg_current_profile(solution, grid)
    current_a = current_dimless * current_scale_A(config)
    wall_mask = grid.r >= 0.75
    return {
        "case": label,
        "voltage_mV": voltage_v * 1000.0,
        "wall_surface_charge": wall_charge,
        "current_mean_pA": float(current_a.mean() * 1e12),
        "current_min_pA": float(current_a.min() * 1e12),
        "current_max_pA": float(current_a.max() * 1e12),
        "conductance_nS": float((current_a.mean() / voltage_v) * 1e9) if abs(voltage_v) > 1e-12 else np.nan,
        "residual": float(solution.residual),
        "current_conservation_error": sg_flux_conservation_error(solution, grid),
        "wall_c_plus_mean": float(solution.c_plus[wall_mask, :].mean()),
        "wall_c_minus_mean": float(solution.c_minus[wall_mask, :].mean()),
        "wall_charge_density_mean": float((solution.c_plus[wall_mask, :] - solution.c_minus[wall_mask, :]).mean()),
        "runtime_s": runtime_s,
        "converged": bool(solution.converged),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    base = load_config(ROOT / "configs" / CASE)
    if mode == "quick":
        base = replace(base, geometry=replace(base.geometry, nr=15, nz=45))

    iv_voltages = [-0.05, -0.025, 0.0, 0.025, 0.05]
    wall_charges = [0.0, -0.001, -0.005, -0.01]
    rows: list[dict[str, float | str | bool]] = []
    saved_fields: dict[str, np.ndarray] = {}
    saved_grid = None

    for voltage in iv_voltages:
        config = replace(base, physics=replace(base.physics, applied_voltage_V=voltage, wall_surface_charge_strength=-0.01))
        solution, grid, runtime = run_reference(config, mode)
        rows.append(summarize_solution("iv", voltage, -0.01, config, solution, grid, runtime))
        pd.DataFrame(rows).to_csv(OUT_DIR / "surface_response_sweep.csv", index=False)

    for charge in wall_charges:
        config = replace(base, physics=replace(base.physics, applied_voltage_V=0.05, wall_surface_charge_strength=charge))
        solution, grid, runtime = run_reference(config, mode)
        rows.append(summarize_solution("surface_charge", 0.05, charge, config, solution, grid, runtime))
        pd.DataFrame(rows).to_csv(OUT_DIR / "surface_response_sweep.csv", index=False)
        if abs(charge) < 1e-15:
            saved_fields["neutral_c_plus"] = solution.c_plus
            saved_fields["neutral_c_minus"] = solution.c_minus
            saved_fields["neutral_psi"] = solution.psi
            saved_grid = grid
        if abs(charge + 0.01) < 1e-15:
            saved_fields["charged_c_plus"] = solution.c_plus
            saved_fields["charged_c_minus"] = solution.c_minus
            saved_fields["charged_psi"] = solution.psi
            saved_grid = grid

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "surface_response_sweep.csv"
    df.to_csv(csv_path, index=False)
    if saved_grid is not None and saved_fields:
        np.savez_compressed(
            OUT_DIR / "surface_response_fields.npz",
            r=saved_grid.r,
            z=saved_grid.z,
            **saved_fields,
        )

    plot_figure(df, saved_grid, saved_fields)
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")


def plot_figure(df: pd.DataFrame, grid, fields: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    iv = df[df["case"] == "iv"].sort_values("voltage_mV")
    axes[0, 0].plot(iv["voltage_mV"], iv["current_mean_pA"], marker="o", color="black", linewidth=2)
    axes[0, 0].axhline(0, color="0.7", linewidth=1)
    axes[0, 0].axvline(0, color="0.7", linewidth=1)
    axes[0, 0].set_xlabel("Applied voltage (mV)")
    axes[0, 0].set_ylabel("Mean current (pA)")
    axes[0, 0].set_title("Current-voltage response")
    panel_label(axes[0, 0], "A")

    charge = df[df["case"] == "surface_charge"].sort_values("wall_surface_charge")
    charge_plot = charge.copy()
    charge_plot["surface_charge_label"] = charge_plot["wall_surface_charge"].abs()
    neutral_wall_cp = float(charge_plot.loc[charge_plot["surface_charge_label"] == 0.0, "wall_c_plus_mean"].iloc[0])
    axes[0, 1].plot(
        charge_plot["surface_charge_label"],
        charge_plot["wall_c_plus_mean"] - neutral_wall_cp,
        marker="o",
        color="#4c78a8",
        linewidth=2,
    )
    axes[0, 1].set_xlabel(r"$|q_w|$ (dimensionless)")
    axes[0, 1].set_ylabel(r"Near-wall cation enrichment, $\Delta\langle c_+\rangle$")
    axes[0, 1].set_title("Surface-charge ion redistribution")
    panel_label(axes[0, 1], "B")

    if grid is not None and {"charged_c_plus", "neutral_c_plus"}.issubset(fields):
        delta_cp = fields["charged_c_plus"] - fields["neutral_c_plus"]
        im = axes[1, 0].contourf(grid.z, grid.r, delta_cp, levels=48, cmap="coolwarm")
        fig.colorbar(im, ax=axes[1, 0], label=r"$\Delta c_+$")
        axes[1, 0].set_xlabel("z/L")
        axes[1, 0].set_ylabel("r/R")
        axes[1, 0].set_title("Cation redistribution by wall charge")
    else:
        axes[1, 0].text(0.5, 0.5, "Field comparison unavailable", ha="center", va="center")
    panel_label(axes[1, 0], "C")

    width = 0.35
    x = np.arange(len(charge))
    axes[1, 1].bar(x - width / 2, charge["residual"], width, label="residual", color="#59a14f")
    axes[1, 1].bar(
        x + width / 2,
        charge["current_conservation_error"],
        width,
        label="current conservation",
        color="#e15759",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{abs(v):.3g}" for v in charge["wall_surface_charge"]])
    axes[1, 1].set_xlabel(r"$|q_w|$ (dimensionless)")
    axes[1, 1].set_ylabel("Error metric")
    axes[1, 1].set_title("Solver quality across sweep")
    axes[1, 1].legend(fontsize=8)
    panel_label(axes[1, 1], "D")

    fig.savefig(FIG_DIR / "Figure_5_surface_response.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
