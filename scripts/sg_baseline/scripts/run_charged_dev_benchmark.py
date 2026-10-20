from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.continuation import run_bulk_voltage_charge_continuation  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.metrics import max_abs, relative_l2  # noqa: E402
from pnp_da_pinn.baseline import PnpSolution  # noqa: E402
from pnp_da_pinn.sg_solver import sg_current_profile, sg_flux_conservation_error  # noqa: E402
from pnp_da_pinn.training import predict_on_grid, train_da_pinn  # noqa: E402


def save_npz(out_dir: Path, solution, psi_p: np.ndarray, cp_p: np.ndarray, cm_p: np.ndarray) -> None:
    np.savez_compressed(
        out_dir / "charged_dev_fields.npz",
        psi_sg=solution.psi,
        c_plus_sg=solution.c_plus,
        c_minus_sg=solution.c_minus,
        psi_pinn=psi_p,
        c_plus_pinn=cp_p,
        c_minus_pinn=cm_p,
    )


def save_figure(out_dir: Path, grid, solution, psi_p: np.ndarray, cp_p: np.ndarray, cm_p: np.ndarray) -> None:
    current = sg_current_profile(solution, grid)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    panels = [
        (solution.psi, "SG potential", r"$\psi$"),
        (psi_p, "DA-PINN potential", r"$\psi$"),
        (np.abs(psi_p - solution.psi), "Potential abs. error", r"$|\Delta\psi|$"),
        (solution.c_plus - solution.c_minus, "SG charge density", r"$c_+ - c_-$"),
        (cp_p - cm_p, "DA-PINN charge density", r"$c_+ - c_-$"),
        (np.abs((cp_p - cm_p) - (solution.c_plus - solution.c_minus)), "Charge-density abs. error", "error"),
    ]
    for ax, (field, title, label) in zip(axes.ravel(), panels):
        image = ax.contourf(grid.z, grid.r, field, levels=40, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("z/L")
        ax.set_ylabel("r/R")
        fig.colorbar(image, ax=ax, label=label)

    fig.savefig(out_dir / "charged_dev_field_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(0.5 * (grid.z[:-1] + grid.z[1:]), current, color="black", linewidth=2)
    ax.set_xlabel("z/L")
    ax.set_ylabel("Dimensionless current")
    ax.set_title("Finite-volume current conservation")
    fig.savefig(out_dir / "charged_dev_current_profile.png", dpi=300)
    plt.close(fig)


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall_0p0036mC.yaml"
    config = load_config(ROOT / "configs" / config_name)
    grid = make_grid(config.geometry)
    out_dir = ROOT / "outputs" / "charged_dev" / Path(config_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    continuation = run_bulk_voltage_charge_continuation(
        config=config,
        grid=grid,
        start_bulk_mol_m3=1.0,
        bulk_stages=5,
        voltage_stages=3,
        charge_stages=3,
        max_iter_per_stage=18,
        tolerance=1e-3,
        relaxation=0.02,
        schedule_mode="surface_then_bulk",
    )
    pd.DataFrame(continuation.diagnostics).to_csv(out_dir / "charged_dev_continuation.csv", index=False)
    solution = continuation.solution
    conservation = sg_flux_conservation_error(solution, grid)

    training = train_da_pinn(solution, grid, config, use_data=True)
    psi_p, cp_p, cm_p, infer_s = predict_on_grid(training.model, grid, config)
    pred_solution = PnpSolution(
        psi=psi_p,
        c_plus=cp_p,
        c_minus=cm_p,
        iterations=0,
        residual=np.nan,
        runtime_s=infer_s,
        converged=True,
    )
    current_sg = sg_current_profile(solution, grid)
    current_pinn = sg_current_profile(pred_solution, grid)
    charge_sg = solution.c_plus - solution.c_minus
    charge_pinn = cp_p - cm_p
    current_rel_l2 = relative_l2(current_pinn, current_sg)
    current_max_abs = max_abs(current_pinn, current_sg)

    rows = [
        {
            "quantity": "psi",
            "relative_l2": relative_l2(psi_p, solution.psi),
            "max_abs": max_abs(psi_p, solution.psi),
            "inference_s": infer_s,
            "training_s": training.runtime_s,
            "sg_flux_conservation_error": conservation,
            "sg_residual": solution.residual,
            "sg_converged": solution.converged,
            "current_relative_l2": current_rel_l2,
            "current_max_abs": current_max_abs,
        },
        {
            "quantity": "c_plus",
            "relative_l2": relative_l2(cp_p, solution.c_plus),
            "max_abs": max_abs(cp_p, solution.c_plus),
            "inference_s": infer_s,
            "training_s": training.runtime_s,
            "sg_flux_conservation_error": conservation,
            "sg_residual": solution.residual,
            "sg_converged": solution.converged,
            "current_relative_l2": current_rel_l2,
            "current_max_abs": current_max_abs,
        },
        {
            "quantity": "c_minus",
            "relative_l2": relative_l2(cm_p, solution.c_minus),
            "max_abs": max_abs(cm_p, solution.c_minus),
            "inference_s": infer_s,
            "training_s": training.runtime_s,
            "sg_flux_conservation_error": conservation,
            "sg_residual": solution.residual,
            "sg_converged": solution.converged,
            "current_relative_l2": current_rel_l2,
            "current_max_abs": current_max_abs,
        },
        {
            "quantity": "charge_density",
            "relative_l2": relative_l2(charge_pinn, charge_sg),
            "max_abs": max_abs(charge_pinn, charge_sg),
            "inference_s": infer_s,
            "training_s": training.runtime_s,
            "sg_flux_conservation_error": conservation,
            "sg_residual": solution.residual,
            "sg_converged": solution.converged,
            "current_relative_l2": current_rel_l2,
            "current_max_abs": current_max_abs,
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "charged_dev_metrics.csv", index=False)
    pd.DataFrame(training.history).to_csv(out_dir / "charged_dev_training_history.csv", index=False)
    save_npz(out_dir, solution, psi_p, cp_p, cm_p)
    save_figure(out_dir, grid, solution, psi_p, cp_p, cm_p)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
