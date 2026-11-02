
from pathlib import Path
import numpy as np

ROOT = Path("demonstrations")
SEED = 20261101

SOURCES = {
    "narrow_expert": (
        "franka_expert_narrow.npz", False
    ),
    "narrow_dagger": (
        "franka_dagger_narrow.npz", True
    ),
    "wide_expert": (
        "franka_expert_wide.npz", False
    ),
    "wide_dagger": (
        "franka_dagger1.npz", True
    ),
}

with np.load(ROOT / "franka_expert_train.npz") as f:
    base_obs = f["observations"].copy()
    base_act = f["actions"].copy()
    base_ids = f["episode_ids"].copy()
    max_move = float(f["max_move"])
    max_time = float(f["max_time"])

n_base = len(base_obs)
base_max_id = int(base_ids.max())
additional = {}

for name, (filename, combined) in SOURCES.items():
    with np.load(ROOT / filename) as f:
        obs = f["observations"].copy()
        act = f["actions"].copy()
        ids = f["episode_ids"].copy()

        if not np.isclose(
            float(f["max_move"]), max_move
        ):
            raise ValueError(f"{name}: max_move mismatch")

        if not np.isclose(
            float(f["max_time"]), max_time
        ):
            raise ValueError(f"{name}: max_time mismatch")

    if combined:
        # DAgger files already include the original data.
        if len(obs) <= n_base:
            raise ValueError(f"{name}: no added data")

        if not np.array_equal(
            obs[:n_base], base_obs
        ):
            raise ValueError(
                f"{name}: original observations differ"
            )

        if not np.array_equal(
            act[:n_base], base_act
        ):
            raise ValueError(
                f"{name}: original actions differ"
            )

        obs = obs[n_base:]
        act = act[n_base:]
        ids = ids[n_base:]

    if len(obs) != len(act) or len(obs) != len(ids):
        raise ValueError(f"{name}: array length mismatch")

    unique_ids = np.unique(ids)

    if len(unique_ids) != 100:
        raise ValueError(
            f"{name}: expected 100 added episodes, "
            f"found {len(unique_ids)}"
        )

    additional[name] = (obs, act, ids)

target = min(
    len(item[0]) for item in additional.values()
)

print("===== DATASET PREPARATION =====")
print("Common original samples:", n_base)
print("Matched additional samples:", target)

for i, (name, (obs, act, ids)) in enumerate(
    additional.items()
):
    rng = np.random.default_rng(SEED + i)
    unique_ids = np.unique(ids)

    # Preserve at least one observation per episode.
    mandatory = np.array([
        rng.choice(np.flatnonzero(ids == eid))
        for eid in unique_ids
    ], dtype=np.int64)

    mask = np.ones(len(obs), dtype=bool)
    mask[mandatory] = False
    remaining = np.flatnonzero(mask)

    extra = rng.choice(
        remaining,
        size=target - len(mandatory),
        replace=False,
    )

    chosen = np.sort(
        np.concatenate([mandatory, extra])
    )

    sampled_obs = obs[chosen]
    sampled_act = act[chosen]
    sampled_ids = ids[chosen]

    # Identical episode ID sets across all four arms.
    id_map = {
        old: base_max_id + 1 + j
        for j, old in enumerate(unique_ids)
    }

    new_ids = np.array(
        [id_map[x] for x in sampled_ids],
        dtype=base_ids.dtype,
    )

    output = ROOT / f"fair_{name}.npz"

    np.savez_compressed(
        output,
        observations=np.concatenate([
            base_obs, sampled_obs
        ]),
        actions=np.concatenate([
            base_act, sampled_act
        ]),
        episode_ids=np.concatenate([
            base_ids, new_ids
        ]),
        max_move=np.float32(max_move),
        max_time=np.float32(max_time),
        position_range=np.float32(
            0.02 if name.startswith("narrow") else 0.12
        ),
    )

    print(
        f"{name:15s} | "
        f"original={n_base} | "
        f"added={len(sampled_obs)} | "
        f"total={n_base + len(sampled_obs)}"
    )
    print(" Saved:", output)

print("All four datasets have equal sample counts.")
