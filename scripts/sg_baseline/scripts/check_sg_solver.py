from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.sg_solver import (  # noqa: E402
    sg_current_profile,
    sg_flux_conservation_error,
    solve_pnp_sg,
)


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "smoke.yaml"
    config = load_config(ROOT / "configs" / config_name)
    grid = make_grid(config.geometry)
    solution = solve_pnp_sg(config, grid)
    current = sg_current_profile(solution, grid)
    conservation = sg_flux_conservation_error(solution, grid)
    print(f"config={config_name}")
    print(
        f"converged={solution.converged} iterations={solution.iterations} "
        f"residual={solution.residual:.3e} runtime={solution.runtime_s:.2f}s"
    )
    print(f"psi range={solution.psi.min():.3e} to {solution.psi.max():.3e}")
    print(f"c_plus range={solution.c_plus.min():.3e} to {solution.c_plus.max():.3e}")
    print(f"c_minus range={solution.c_minus.min():.3e} to {solution.c_minus.max():.3e}")
    print(f"current range={current.min():.6e} to {current.max():.6e}")
    print(f"conservation={conservation:.3e}")


if __name__ == "__main__":
    main()
