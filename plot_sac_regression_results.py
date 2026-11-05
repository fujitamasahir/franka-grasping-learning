"""Generate publication-ready figures for the controlled SAC study."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("bc_results")
FIG = OUT / "sac_regdiag_figures"
FIG.mkdir(exist_ok=True)
SEEDS = (907, 911, 919, 929, 937)
COLORS = {"A": "#2878B5", "B": "#D9534F"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def training_figure():
    rows = read_csv(OUT / "sac_regdiag_training_diagnostics.csv")
    steps = np.array([10, 20, 30])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("entropy_coefficient", "Entropy coefficient"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for arm in ("A", "B"):
            matrix = np.array([
                [float(next(r[metric] for r in rows if int(r["sac_seed"]) == seed and r["arm"] == arm and int(r["timesteps"]) == step * 1000)) for step in steps]
                for seed in SEEDS
            ])
            for values in matrix:
                ax.plot(steps, values, color=COLORS[arm], alpha=0.25, linewidth=1)
            mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
            ax.plot(steps, mean, "o-", color=COLORS[arm], linewidth=2.5, label=f"Arm {arm} mean")
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[arm], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Training steps (thousands)")
        ax.set_xticks(steps)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("TensorBoard scalar")
    axes[0].legend(frameon=False)
    fig.suptitle("SAC training diagnostics (thin lines: seeds; band: ±1 SD)")
    save(fig, "sac_training_diagnostics")


def residual_figure():
    rows = read_csv(OUT / "sac_regdiag_training_diagnostics.csv")
    steps = [10, 20, 30]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, ylabel in (
        (axes[0], "mean_residual_norm_mm", "Mean residual norm (mm)"),
        (axes[1], "action_saturation_frequency", "Action saturation frequency"),
    ):
        for arm in ("A", "B"):
            matrix = np.array([
                [float(next(r[metric] for r in rows if int(r["sac_seed"]) == seed and r["arm"] == arm and int(r["timesteps"]) == step * 1000)) for step in steps]
                for seed in SEEDS
            ])
            mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
            ax.plot(steps, mean, "o-", color=COLORS[arm], linewidth=2, label=f"Arm {arm} (coef {0.005 if arm == 'A' else 0.020:g})")
            ax.fill_between(steps, mean - sd, mean + sd, color=COLORS[arm], alpha=0.16)
            for values in matrix:
                ax.plot(steps, values, color=COLORS[arm], alpha=0.18, linewidth=0.8)
        ax.set_xlabel("Training steps (thousands)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(steps)
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Residual action diagnostics during training (mean ± SD across five seeds)")
    save(fig, "residual_action_diagnostics")


def rescue_regression_figure():
    rows = read_csv(OUT / "sac_regdiag_primary_seed_comparisons.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    max_value = max(max(int(r[f"arm_{arm}_{key}"]) for r in rows) for arm in ("a", "b") for key in ("rescue_count", "regression_count"))
    ax.plot([0, max_value + 8], [0, max_value + 8], "--", color="gray", linewidth=1, label="Equal rescues and regressions")
    for arm, marker in (("a", "o"), ("b", "s")):
        for r in rows:
            ax.scatter(int(r[f"arm_{arm}_regression_count"]), int(r[f"arm_{arm}_rescue_count"]),
                       color=COLORS[arm.upper()], marker=marker, s=58, alpha=0.9)
            ax.annotate(f"{r['sac_seed']} {arm.upper()}",
                        (int(r[f"arm_{arm}_regression_count"]), int(r[f"arm_{arm}_rescue_count"])),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Regression count (BC success → SAC failure)")
    ax.set_ylabel("Rescue count (BC failure → SAC success)")
    ax.set_title("Rescues vs regressions on Final positions (30k checkpoints)")
    ax.grid(alpha=0.25)
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["A"], label="Arm A"),
                       plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["B"], label="Arm B"),
                       plt.Line2D([0], [0], linestyle="--", color="gray", label="Equal")], frameon=False)
    save(fig, "rescue_vs_regression")


def final_success_figure():
    rows = read_csv(OUT / "sac_regdiag_final_checkpoint_metrics.csv")
    baseline = sum(int(r["success"]) for r in read_csv(OUT / "sac_regdiag_bc_final2000.csv")) / 20
    categories = ["BC Only", "Arm A 30k", "Arm B 30k", "Arm A selected", "Arm B selected"]
    vals = {k: [] for k in categories[1:]}
    for seed in SEEDS:
        vals["Arm A 30k"].append(100 * float(next(r["success_rate"] for r in rows if int(r["sac_seed"]) == seed and r["arm"] == "A" and int(r["timesteps"]) == 30000)))
        vals["Arm B 30k"].append(100 * float(next(r["success_rate"] for r in rows if int(r["sac_seed"]) == seed and r["arm"] == "B" and int(r["timesteps"]) == 30000)))
    selection = read_csv(OUT / "sac_regdiag_selected_seed_comparisons.csv")
    for arm, key in (("A", "Arm A selected"), ("B", "Arm B selected")):
        vals[key] = [100 * float(r[f"arm_{arm.lower()}_success_rate"]) for r in selection]
    means = [100 * baseline] + [np.mean(vals[k]) for k in categories[1:]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(categories))
    ax.bar(x, means, color=["#555555", COLORS["A"], COLORS["B"], "#78A9D1", "#E98B86"], alpha=0.85)
    ax.scatter(np.zeros(1), [100 * baseline], color="black", zorder=3, s=28)
    for i, key in enumerate(categories[1:], 1):
        jitter = np.linspace(-0.13, 0.13, len(SEEDS))
        ax.scatter(i + jitter, vals[key], color="#222222", s=24, zorder=3)
    ax.axhline(100 * baseline, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(x, categories)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(75, 100)
    ax.set_title("Final success across five SAC training seeds (2,000 shared positions)")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "final_success_rate")


def validation_steps_figure():
    rows = read_csv(OUT / "sac_regdiag_validation_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, arm in zip(axes, ("A", "B")):
        for seed in SEEDS:
            subset = [r for r in rows if int(r["sac_seed"]) == seed and r["arm"] == arm]
            subset.sort(key=lambda r: int(r["timesteps"]))
            ax.plot([int(r["timesteps"]) / 1000 for r in subset], [100 * float(r["success_rate"]) for r in subset],
                    marker="o", color=COLORS[arm], alpha=0.28, linewidth=1)
        means = [100 * np.mean([float(r["success_rate"]) for r in rows if r["arm"] == arm and int(r["timesteps"]) == step]) for step in (10000, 20000, 30000)]
        ax.plot([10, 20, 30], means, "o-", color=COLORS[arm], linewidth=2.8, label="Validation B mean")
        ax.set_title(f"Arm {arm} (penalty coef {0.005 if arm == 'A' else 0.020:g})")
        ax.set_xlabel("Training steps (thousands)")
        ax.set_xticks([10, 20, 30])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation B success (%)")
    fig.suptitle("Validation B checkpoint-selection trends (500 positions per candidate)")
    save(fig, "validation_checkpoint_selection")


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    training_figure()
    residual_figure()
    rescue_regression_figure()
    final_success_figure()
    validation_steps_figure()
    print(f"Saved five figures (PNG and SVG) in {FIG}")


if __name__ == "__main__":
    main()
