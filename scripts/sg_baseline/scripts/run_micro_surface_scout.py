from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.continuation import run_bulk_voltage_charge_continuation  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall_0p036mC.yaml"
    config = load_config(ROOT / "configs" / config_name)
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
    df = pd.DataFrame(result.diagnostics)
    out_dir = ROOT / "outputs" / "continuation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_name).stem}_micro_surface_scout.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
