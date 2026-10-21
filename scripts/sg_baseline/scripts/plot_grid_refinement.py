from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    csv_name = sys.argv[1] if len(sys.argv) > 1 else "surface_wall_0p036mC_full.csv"
    csv_path = ROOT / "outputs" / "grid_refinement" / csv_name
    df = pd.read_csv(csv_path)
    out_path = csv_path.with_suffix(".png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(df["nodes"], df["current_mean"], marker="o", color="black", linewidth=2)
    axes[0].set_xlabel("Grid nodes")
    axes[0].set_ylabel("Mean dimensionless current")
    axes[0].set_title("Current grid stability")

    axes[1].semilogy(
        df["nodes"],
        df["current_conservation_error"],
        marker="o",
        color="#2f6f8f",
        linewidth=2,
    )
    axes[1].set_xlabel("Grid nodes")
    axes[1].set_ylabel("Current conservation error")
    axes[1].set_title("Axial current conservation")

    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
