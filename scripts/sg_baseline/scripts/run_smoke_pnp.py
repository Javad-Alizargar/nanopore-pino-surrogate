from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.baseline import (  # noqa: E402
    flux_conservation_error,
    physical_current_profile_A,
    solve_pnp_gummel,
)
from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.metrics import relative_l2  # noqa: E402
from pnp_da_pinn.sg_solver import (  # noqa: E402
    sg_current_profile,
    sg_flux_conservation_error,
    solve_pnp_sg,
)
from pnp_da_pinn.training import predict_on_grid, train_da_pinn  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    grid = make_grid(config.geometry)

    print("Solving prototype Gummel PNP baseline...")
    prototype = solve_pnp_gummel(config, grid)
    prototype_current_A = physical_current_profile_A(prototype, grid, config)
    prototype_conservation = flux_conservation_error(prototype, grid)
    print(
        f"Prototype converged={prototype.converged} iterations={prototype.iterations} "
        f"residual={prototype.residual:.3e} runtime={prototype.runtime_s:.2f}s"
    )
    print(
        f"Prototype current range: {prototype_current_A.min() * 1e12:.3f} to "
        f"{prototype_current_A.max() * 1e12:.3f} pA; conservation={prototype_conservation:.3e}"
    )

    print("Solving Scharfetter-Gummel finite-volume PNP baseline...")
    solution = solve_pnp_sg(config, grid)
    sg_current = sg_current_profile(solution, grid)
    conservation = sg_flux_conservation_error(solution, grid)
    print(
        f"SG converged={solution.converged} iterations={solution.iterations} "
        f"residual={solution.residual:.3e} runtime={solution.runtime_s:.2f}s"
    )
    print(
        f"SG dimensionless current range: {sg_current.min():.3e} to {sg_current.max():.3e}; "
        f"conservation={conservation:.3e}"
    )

    print("Training first DA-PINN smoke model on SG baseline...")
    result = train_da_pinn(solution, grid, config, use_data=True)
    psi_p, cp_p, cm_p, infer_s = predict_on_grid(result.model, grid, config)

    rows = [
        {
            "quantity": "psi",
            "relative_l2": relative_l2(psi_p, solution.psi),
            "inference_s": infer_s,
            "training_s": result.runtime_s,
            "sg_flux_conservation_error": conservation,
        },
        {
            "quantity": "c_plus",
            "relative_l2": relative_l2(cp_p, solution.c_plus),
            "inference_s": infer_s,
            "training_s": result.runtime_s,
            "sg_flux_conservation_error": conservation,
        },
        {
            "quantity": "c_minus",
            "relative_l2": relative_l2(cm_p, solution.c_minus),
            "inference_s": infer_s,
            "training_s": result.runtime_s,
            "sg_flux_conservation_error": conservation,
        },
    ]
    out_dir = ROOT / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "smoke_metrics.csv", index=False)
    pd.DataFrame(result.history).to_csv(out_dir / "smoke_training_history.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
