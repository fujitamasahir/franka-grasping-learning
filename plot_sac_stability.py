"""Create portfolio-quality plots from sac_stability_analysis_summary.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY = Path("bc_results/sac_stability_analysis_summary.json")
OUT = Path("bc_results/sac_stability_figures")
SEED_COLORS = {
    811: "#0072B2", 823: "#D55E00", 827: "#009E73",
    829: "#CC79A7", 839: "#E69F00",
}
STEPS_K = [10, 20, 30]


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = data["checkpoint_rows"]
    selections = data["selection_rows"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 1. Learning curves with all model trajectories and the cross-seed mean.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    values_by_step = []
    for seed in sorted(SEED_COLORS):
        subset = sorted((r for r in rows if r["sac_seed"] == seed), key=lambda r: r["timesteps"])
        values = [100 * r["final_success_rate"] for r in subset]
        values_by_step.append(values)
        ax.plot(STEPS_K, values, marker="o", linewidth=1.8, markersize=6,
                color=SEED_COLORS[seed], label=f"SAC seed {seed}")
    values = np.asarray(values_by_step)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    ax.errorbar(STEPS_K, mean, yerr=sd, color="#222222", marker="D", markersize=6,
                linewidth=2.5, capsize=4, label="Across-seed mean ± SD", zorder=5)
    ax.set(title="Residual SAC Learning Curves", xlabel="Training steps (thousands)",
           ylabel="Final success rate (%)", xticks=STEPS_K)
    ax.set_ylim(max(0, np.floor((values.min() - 1) * 2) / 2), 100)
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, loc="lower left")
    save(fig, "sac_stability_learning_curves")

    # 2. Validation-to-test relationship for 40- and 500-episode validation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, key, title in [
        (axes[0], "validation_a_rate", "Validation A (40 episodes)"),
        (axes[1], "validation_b_rate", "Validation B (500 episodes)"),
    ]:
        for seed in sorted(SEED_COLORS):
            subset = [r for r in rows if r["sac_seed"] == seed]
            ax.scatter([100 * r[key] for r in subset], [100 * r["final_success_rate"] for r in subset],
                       color=SEED_COLORS[seed], s=48, alpha=.9, label=f"Seed {seed}")
            for r in subset:
                ax.annotate(f"{r['timesteps']//1000}k", (100*r[key],100*r["final_success_rate"]),
                            xytext=(4, 3), textcoords="offset points", fontsize=7, color=SEED_COLORS[seed])
        ax.plot([0, 100], [0, 100], linestyle="--", color="#777777", linewidth=1)
        ax.set(title=title, xlabel="Validation success rate (%)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Final evaluation success rate (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Validation Performance vs Held-Out Final Evaluation", y=1.02, fontsize=14)
    save(fig, "sac_stability_validation_vs_final")

    # 3. Final success of checkpoints selected by each validation-set size.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(selections))
    width = .34
    for i, row in enumerate(selections):
        ax.plot([x[i] - width/2, x[i] + width/2],
                [100*row["selection_a_success_rate"], 100*row["selection_b_success_rate"]],
                color="#999999", linewidth=1.2, zorder=1)
    a = [100*r["selection_a_success_rate"] for r in selections]
    b = [100*r["selection_b_success_rate"] for r in selections]
    ax.bar(x - width/2, a, width, color="#56B4E9", label="Selected by 40 episodes (A)")
    ax.bar(x + width/2, b, width, color="#0072B2", label="Selected by 500 episodes (B)")
    for i, row in enumerate(selections):
        ax.text(x[i]-width/2, a[i]+.08, f"{row['selection_a_step']//1000}k", ha="center", fontsize=8)
        ax.text(x[i]+width/2, b[i]+.08, f"{row['selection_b_step']//1000}k", ha="center", fontsize=8)
    ax.set_xticks(x, [str(r["sac_seed"]) for r in selections])
    ax.set(title="Checkpoint Selection: Held-Out Final Performance", xlabel="SAC training seed",
           ylabel="Final evaluation success rate (%)")
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="lower right")
    save(fig, "sac_stability_checkpoint_selection")

    # 4. Paired rescue/regression outcomes, with equal-impact diagonal.
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    max_count = max(max(r["rescue_count"], r["regression_count"]) for r in rows)
    lim = max(5, int(np.ceil(max_count / 5) * 5))
    ax.plot([0, lim], [0, lim], color="#555555", linestyle="--", linewidth=1.4,
            label="Rescues = regressions (no net change)")
    zero_rows = [r for r in rows if not r["rescue_count"] and not r["regression_count"]]
    for r in rows:
        seed = r["sac_seed"]
        ax.scatter(r["regression_count"], r["rescue_count"], s=70,
                   color=SEED_COLORS[seed], edgecolor="white", linewidth=.7)
        if r["rescue_count"] or r["regression_count"]:
            ax.annotate(f"{seed}, {r['timesteps']//1000}k",
                        (r["regression_count"], r["rescue_count"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=7)
    if zero_rows:
        labels = ", ".join(f"{r['sac_seed']}/{r['timesteps']//1000}k" for r in zero_rows)
        ax.annotate(f"0 rescue / 0 regression: {labels}", (0, 0),
                    xytext=(15, 16), textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .8})
    ax.set(xlim=(-.8, lim), ylim=(-.8, lim),
           title="Paired SAC Effect Relative to BC",
           xlabel="Regression count (BC success → SAC failure)",
           ylabel="Rescue count (BC failure → SAC success)")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "sac_stability_rescue_vs_regression")
    print(f"Saved four SVG/PNG figure pairs to {OUT}")


if __name__ == "__main__":
    main()
