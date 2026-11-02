
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("bc_results")
CONDITIONS = [
    "narrow_expert",
    "narrow_dagger",
    "wide_expert",
    "wide_dagger",
]
SEEDS = [42, 123, 456]
RANGES = [0.02, 0.06, 0.12, 0.20]

all_data = []

for condition in CONDITIONS:
    for train_seed in SEEDS:
        path = RESULTS / f"final_{condition}_seed{train_seed}.csv"
        df = pd.read_csv(path)

        required = {
            "policy", "seed", "cube_x", "cube_y",
            "success", "steps", "max_lift",
            "final_lift", "position_range",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        df["condition"] = condition
        df["train_seed"] = train_seed
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Check the expected number of evaluation episodes.
expected = len(SEEDS) * len(CONDITIONS) * len(RANGES) * 200 * 2
assert len(data) == expected, (
    f"Expected {expected} rows, got {len(data)}"
)

# Verify every condition has 200 BC and 200 expert trials.
counts = data.groupby(
    ["condition", "train_seed", "position_range", "policy"]
).size()
assert (counts == 200).all(), "Unexpected episode counts"

bc = data[data["policy"] == "bc"].copy()

summary = (
    bc.groupby(["condition", "train_seed", "position_range"])
    .agg(
        trials=("success", "size"),
        successes=("success", "sum"),
        success_rate=("success", "mean"),
        mean_steps_success=(
            "steps",
            lambda x: x.loc[bc.loc[x.index, "success"] == 1].mean()
        ),
        mean_max_lift=("max_lift", "mean"),
    )
    .reset_index()
)

summary["success_pct"] = summary["success_rate"] * 100

print("\n===== SUCCESS RATE BY MODEL =====")
pivot = summary.pivot_table(
    index=["condition", "train_seed"],
    columns="position_range",
    values="success_pct",
)
print(pivot.round(1).to_string())

print("\n===== MEAN SUCCESS RATE ACROSS TRAINING SEEDS =====")
mean_summary = (
    summary.groupby(["condition", "position_range"])
    .agg(
        mean_pct=("success_pct", "mean"),
        std_pct=("success_pct", "std"),
        min_pct=("success_pct", "min"),
        max_pct=("success_pct", "max"),
    )
    .reset_index()
)

for r in RANGES:
    print(f"\nPosition range: ±{r:.2f} m")
    subset = mean_summary[
        np.isclose(mean_summary["position_range"], r)
    ]
    for _, row in subset.iterrows():
        print(
            f"  {row['condition']:15s} "
            f"{row['mean_pct']:5.1f}% ± "
            f"{row['std_pct']:4.1f} pp "
            f"(min={row['min_pct']:.1f}%, "
            f"max={row['max_pct']:.1f}%)"
        )

# Calculate DAgger effects within each training range.
print("\n===== DAGGER EFFECT (percentage points) =====")
for r in RANGES:
    s = mean_summary[
        np.isclose(mean_summary["position_range"], r)
    ].set_index("condition")["mean_pct"]

    narrow = s["narrow_dagger"] - s["narrow_expert"]
    wide = s["wide_dagger"] - s["wide_expert"]

    print(
        f"±{r:.2f}m | "
        f"Narrow: {narrow:+.1f} pp | "
        f"Wide: {wide:+.1f} pp | "
        f"Interaction: {wide - narrow:+.1f} pp"
    )

# Compare matched success/failure outcomes for each seed.
print("\n===== PAIRED SUCCESS / FAILURE COMPARISON =====")
for train_seed in SEEDS:
    for r in RANGES:
        sub = bc[
            (bc["train_seed"] == train_seed)
            & np.isclose(bc["position_range"], r)
        ]

        def paired_counts(a, b):
            left = sub[sub["condition"] == a][
                ["seed", "success"]
            ].rename(columns={"success": "a_success"})
            right = sub[sub["condition"] == b][
                ["seed", "success"]
            ].rename(columns={"success": "b_success"})

            pair = left.merge(
                right, on="seed", validate="one_to_one"
            )
            if len(pair) != 200:
                raise ValueError("Missing matched evaluation seeds")

            a_only = int(
                ((pair["a_success"] == 1)
                 & (pair["b_success"] == 0)).sum()
            )
            b_only = int(
                ((pair["a_success"] == 0)
                 & (pair["b_success"] == 1)).sum()
            )
            return a_only, b_only

        ne, nd = paired_counts(
            "narrow_expert", "narrow_dagger"
        )
        we, wd = paired_counts(
            "wide_expert", "wide_dagger"
        )

        print(
            f"seed={train_seed} ±{r:.2f}m | "
            f"Narrow E-only={ne}, D-only={nd} | "
            f"Wide E-only={we}, D-only={wd}"
        )

RESULTS.mkdir(exist_ok=True)
data.to_csv(RESULTS / "all_fair_results.csv", index=False)
summary.to_csv(RESULTS / "fair_model_summary.csv", index=False)
mean_summary.to_csv(
    RESULTS / "fair_mean_summary.csv", index=False
)

print("\n===== SAVED =====")
print("bc_results/all_fair_results.csv")
print("bc_results/fair_model_summary.csv")
print("bc_results/fair_mean_summary.csv")
