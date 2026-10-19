from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.continuation import run_bulk_voltage_charge_continuation  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.sg_solver import sg_current_profile, sg_flux_conservation_error  # noqa: E402


def run_case(config_path: Path, nr: int, nz: int) -> dict[str, float | int | bool]:
    config = load_config(config_path)
    config = replace(config, geometry=replace(config.geometry, nr=nr, nz=nz))
    grid = make_grid(config.geometry)
    result = run_bulk_voltage_charge_continuation(
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
    solution = result.solution
    current = sg_current_profile(solution, grid)
    final_diag = result.diagnostics[-1]
    return {
        "nr": nr,
        "nz": nz,
        "nodes": nr * nz,
        "final_converged": bool(solution.converged),
        "final_accepted": bool(final_diag["accepted_for_continuation"]),
        "residual": float(solution.residual),
        "current_mean": float(current.mean()),
        "current_min": float(current.min()),
        "current_max": float(current.max()),
        "current_conservation_error": sg_flux_conservation_error(solution, grid),
        "runtime_s": float(sum(row["runtime_s"] for row in result.diagnostics)),
    }


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall_0p036mC.yaml"
    mode = sys.argv[2] if len(sys.argv) > 2 else "quick"
    grids = [(15, 45), (21, 61)] if mode == "quick" else [(15, 45), (21, 61), (31, 91)]
    config_path = ROOT / "configs" / config_name
    rows = [run_case(config_path, nr, nz) for nr, nz in grids]
    for idx, row in enumerate(rows):
        if idx == 0:
            row["relative_current_change_vs_previous"] = 0.0
        else:
            previous = rows[idx - 1]["current_mean"]
            row["relative_current_change_vs_previous"] = abs((row["current_mean"] - previous) / previous)
    out_dir = ROOT / "outputs" / "grid_refinement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_name).stem}_{mode}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
