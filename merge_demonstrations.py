
import numpy as np

original_path = "demonstrations/franka_expert_train.npz"
wide_path = "demonstrations/franka_expert_wide.npz"
output_path = "demonstrations/franka_bc_wide.npz"

with np.load(original_path) as a, np.load(wide_path) as b:
    obs_a = a["observations"].copy()
    act_a = a["actions"].copy()
    ids_a = a["episode_ids"].copy()

    obs_b = b["observations"].copy()
    act_b = b["actions"].copy()
    ids_b = b["episode_ids"].copy()

    max_move_a = float(a["max_move"])
    max_move_b = float(b["max_move"])

    max_time_a = float(a["max_time"])
    max_time_b = float(b["max_time"])

if not np.isclose(max_move_a, max_move_b):
    raise ValueError("max_move mismatch")

if not np.isclose(max_time_a, max_time_b):
    raise ValueError("max_time mismatch")

# Ensure episode IDs remain unique after merging.
ids_b = ids_b - ids_b.min() + ids_a.max() + 1

observations = np.concatenate([obs_a, obs_b])
actions = np.concatenate([act_a, act_b])
episode_ids = np.concatenate([ids_a, ids_b])

np.savez_compressed(
    output_path,
    observations=observations,
    actions=actions,
    episode_ids=episode_ids,
    max_move=np.float32(max_move_a),
    max_time=np.float32(max_time_a),
    position_range=np.float32(0.12),
)

print("===== MERGE RESULT =====")
print("Original samples:", len(obs_a))
print("Wide samples:", len(obs_b))
print("Combined samples:", len(observations))
print("Saved:", output_path)
