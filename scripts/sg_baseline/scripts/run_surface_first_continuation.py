from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.continuation import run_bulk_voltage_charge_continuation  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall.yaml"
    config = load_config(ROOT / "configs" / config_name)
    grid = make_grid(config.geometry)
    result = run_bulk_voltage_charge_continuation(
        config=config,
        grid=grid,
        start_bulk_mol_m3=1.0,
        bulk_stages=8,
        voltage_stages=6,
        charge_stages=24,
        max_iter_per_stage=config.solver.max_gummel_iter,
        tolerance=max(config.solver.tolerance, 1e-5),
        relaxation=min(config.solver.relaxation, 0.005),
        schedule_mode="surface_then_bulk",
    )
    df = pd.DataFrame(result.diagnostics)
    out_dir = ROOT / "outputs" / "continuation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_name).stem}_surface_first_diagnostics.csv"
    df.to_csv(out_path, index=False)
    print(df.tail(min(16, len(df))).to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
