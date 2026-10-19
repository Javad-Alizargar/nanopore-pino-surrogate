from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "outputs" / "charged_dev" / "surface_wall_0p036mC"
FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
    )


def load_fields():
    fields = np.load(CASE_DIR / "charged_dev_fields.npz")
    nr, nz = fields["psi_sg"].shape
    r = np.linspace(0.0, 1.0, nr)
    z = np.linspace(0.0, 1.0, nz)
    return fields, r, z


def plot_reference_fields() -> None:
    fields, r, z = load_fields()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    panels = [
        (fields["psi_sg"], r"Potential, $\psi$", "A"),
        (fields["c_plus_sg"], r"Cation concentration, $c_+$", "B"),
        (fields["c_minus_sg"], r"Anion concentration, $c_-$", "C"),
    ]
    for ax, (data, title, label) in zip(axes, panels):
        im = ax.contourf(z, r, data, levels=48, cmap="viridis")
        panel_label(ax, label)
        ax.set_title(title)
        ax.set_xlabel("z/L")
        ax.set_ylabel("r/R")
        fig.colorbar(im, ax=ax)
    fig.savefig(FIG_DIR / "Figure_2_reference_fields.png", dpi=300)
    plt.close(fig)


def plot_surrogate_performance() -> None:
    fields, r, z = load_fields()
    psi_err = np.abs(fields["psi_pinn"] - fields["psi_sg"])
    cp_err = np.abs(fields["c_plus_pinn"] - fields["c_plus_sg"])
    cm_err = np.abs(fields["c_minus_pinn"] - fields["c_minus_sg"])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1), constrained_layout=True)
    panels = [
        (fields["psi_pinn"], r"DA-PINN $\psi$", "A", "viridis"),
        (fields["c_plus_pinn"], r"DA-PINN $c_+$", "B", "viridis"),
        (fields["c_minus_pinn"], r"DA-PINN anion concentration, $c_-$", "C", "viridis"),
        (psi_err, r"$|\Delta\psi|$", "D", "magma"),
        (cp_err, r"$|\Delta c_+|$", "E", "magma"),
        (cm_err, r"$|\Delta c_-|$", "F", "magma"),
    ]
    for ax, (data, title, label, cmap) in zip(axes.ravel(), panels):
        im = ax.contourf(z, r, data, levels=48, cmap=cmap)
        panel_label(ax, label)
        ax.set_title(title)
        ax.set_xlabel("z/L")
        ax.set_ylabel("r/R")
        fig.colorbar(im, ax=ax)
    fig.savefig(FIG_DIR / "Figure_3_surrogate_errors.png", dpi=300)
    plt.close(fig)


def plot_training_and_metrics() -> None:
    history = pd.read_csv(CASE_DIR / "charged_dev_training_history.csv")
    metrics = pd.read_csv(CASE_DIR / "charged_dev_metrics.csv")
    ablation_path = ROOT / "outputs" / "ablation" / "surface_wall_0p036mC" / "pinn_ablation_metrics.csv"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4), constrained_layout=True)
    axes[0].semilogy(history["epoch"], history["loss"], color="black", linewidth=2, label="total")
    for key, color in [
        ("loss_poisson", "#1f77b4"),
        ("loss_np_plus", "#2ca02c"),
        ("loss_np_minus", "#ff7f0e"),
        ("loss_data", "#9467bd"),
        ("loss_charge_data", "#d62728"),
    ]:
        if key in history:
            axes[0].semilogy(history["epoch"], history[key], linewidth=1.4, label=key.replace("loss_", ""))
    panel_label(axes[0], "A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training history")
    axes[0].legend(fontsize=7)

    current_error = float(metrics["current_relative_l2"].iloc[0])
    summary = pd.concat(
        [
            metrics[metrics["quantity"].isin(["psi", "c_plus", "c_minus", "charge_density"])][
                ["quantity", "relative_l2"]
            ],
            pd.DataFrame([{"quantity": "current_profile", "relative_l2": current_error}]),
        ],
        ignore_index=True,
    )
    labels = [
        q.replace("c_plus", r"$c_+$")
        .replace("c_minus", r"$c_-$")
        .replace("psi", r"$\psi$")
        .replace("charge_density", r"$c_+-c_-$")
        .replace("current_profile", "current")
        for q in summary["quantity"]
    ]
    axes[1].bar(labels, summary["relative_l2"], color=["#4c78a8", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"])
    axes[1].set_yscale("log")
    panel_label(axes[1], "B")
    axes[1].set_ylabel(r"Relative $L_2$ error")
    axes[1].set_title("DA-PINN prediction error")
    axes[1].tick_params(axis="x", rotation=30)

    panel_label(axes[2], "C")
    axes[2].set_title("Ablation: sparse data assimilation")
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        subset = ablation[ablation["quantity"].isin(["c_plus", "c_minus", "charge_density"])]
        pivot = subset.pivot(index="quantity", columns="model", values="relative_l2").loc[
            ["c_plus", "c_minus", "charge_density"]
        ]
        x = np.arange(len(pivot.index))
        width = 0.36
        axes[2].bar(
            x - width / 2,
            pivot["physics_only_exact_boundary"],
            width,
            label="physics-only",
            color="#bab0ab",
        )
        axes[2].bar(
            x + width / 2,
            pivot["data_assimilated_exact_boundary"],
            width,
            label="DA-PINN",
            color="#4c78a8",
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([r"$c_+$", r"$c_-$", r"$c_+-c_-$"])
        axes[2].set_yscale("log")
        axes[2].set_ylabel(r"Relative $L_2$ error")
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "Ablation metrics not generated", ha="center", va="center")
        axes[2].set_axis_off()
    fig.savefig(FIG_DIR / "Figure_5_training_metrics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_reference_fields()
    plot_surrogate_performance()
    plot_training_and_metrics()
    print(f"Wrote manuscript figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
