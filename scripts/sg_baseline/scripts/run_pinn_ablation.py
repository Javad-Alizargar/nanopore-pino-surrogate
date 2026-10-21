from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.baseline import PnpSolution  # noqa: E402
from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.metrics import max_abs, relative_l2  # noqa: E402
from pnp_da_pinn.sg_solver import sg_current_profile  # noqa: E402
from pnp_da_pinn.training import predict_on_grid, train_da_pinn  # noqa: E402


def metric_rows(label: str, solution: PnpSolution, pred: PnpSolution, training_s: float) -> list[dict[str, float | str]]:
    current_ref = sg_current_profile(solution, grid)
    current_pred = sg_current_profile(pred, grid)
    charge_ref = solution.c_plus - solution.c_minus
    charge_pred = pred.c_plus - pred.c_minus
    fields = [
        ("psi", pred.psi, solution.psi),
        ("c_plus", pred.c_plus, solution.c_plus),
        ("c_minus", pred.c_minus, solution.c_minus),
        ("charge_density", charge_pred, charge_ref),
        ("current_profile", current_pred, current_ref),
    ]
    return [
        {
            "model": label,
            "quantity": quantity,
            "relative_l2": relative_l2(predicted, reference),
            "max_abs": max_abs(predicted, reference),
            "inference_s": pred.runtime_s,
            "training_s": training_s,
        }
        for quantity, predicted, reference in fields
    ]


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall_0p036mC.yaml"
    case_name = Path(config_name).stem
    config = load_config(ROOT / "configs" / config_name)
    global grid
    grid = make_grid(config.geometry)
    case_dir = ROOT / "outputs" / "charged_dev" / case_name
    fields = np.load(case_dir / "charged_dev_fields.npz")
    reference = PnpSolution(
        psi=fields["psi_sg"],
        c_plus=fields["c_plus_sg"],
        c_minus=fields["c_minus_sg"],
        iterations=0,
        residual=np.nan,
        runtime_s=np.nan,
        converged=True,
    )

    out_dir = ROOT / "outputs" / "ablation" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for label, use_data in [("physics_only_exact_boundary", False), ("data_assimilated_exact_boundary", True)]:
        training = train_da_pinn(reference, grid, config, use_data=use_data)
        psi_p, cp_p, cm_p, infer_s = predict_on_grid(training.model, grid, config)
        pred = PnpSolution(
            psi=psi_p,
            c_plus=cp_p,
            c_minus=cm_p,
            iterations=0,
            residual=np.nan,
            runtime_s=infer_s,
            converged=True,
        )
        rows.extend(metric_rows(label, reference, pred, training.runtime_s))
        pd.DataFrame(training.history).to_csv(out_dir / f"{label}_history.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pinn_ablation_metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
