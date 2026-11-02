
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.xyz_head = nn.Linear(128, 3)
        self.grip_head = nn.Linear(128, 1)

    def forward(self, obs):
        features = self.backbone(obs)

        xyz = torch.tanh(self.xyz_head(features))
        grip_logits = self.grip_head(features)

        return xyz, grip_logits


def compute_loss(model, obs, actions):
    xyz_pred, grip_logits = model(obs)

    xyz_target = actions[:, :3]
    grip_target = (actions[:, 3:4] > 0).float()

    xyz_loss = nn.functional.smooth_l1_loss(
        xyz_pred, xyz_target
    )

    grip_loss = nn.functional.binary_cross_entropy_with_logits(
        grip_logits, grip_target
    )

    loss = xyz_loss + 0.25 * grip_loss

    return loss, xyz_loss, grip_loss


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CPU is a good default for small MLPs on Apple Silicon.
    torch.set_num_threads(4)
    device = torch.device("cpu")

    with np.load(args.data) as data:
        observations = data["observations"].copy()
        actions = data["actions"].copy()
        episode_ids = data["episode_ids"].copy()
        max_move = float(data["max_move"])
        max_time = float(data["max_time"])
        position_range = float(data["position_range"])

    unique_episodes = np.unique(episode_ids)

    if len(unique_episodes) < 5:
        raise ValueError(
            "Need at least five successful episodes."
        )

    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_episodes)

    n_val = max(1, round(len(unique_episodes) * 0.1))

    val_ids = unique_episodes[:n_val]
    train_ids = unique_episodes[n_val:]

    train_mask = np.isin(episode_ids, train_ids)
    val_mask = np.isin(episode_ids, val_ids)

    x_train = observations[train_mask]
    y_train = actions[train_mask]

    x_val = observations[val_mask]
    y_val = actions[val_mask]

    # Normalize using training observations ONLY.
    obs_mean = x_train.mean(axis=0).astype(np.float32)
    obs_std = x_train.std(axis=0).astype(np.float32)

    # Prevent division by very small values.
    obs_std = np.maximum(obs_std, 1e-4)

    x_train = (x_train - obs_mean) / obs_std
    x_val = (x_val - obs_mean) / obs_std

    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(x_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )

    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    model = BCPolicy().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    print("===== BEHAVIOR CLONING TRAINING =====")
    print(f"Device: {device}")
    print(f"Training episodes: {len(train_ids)}")
    print(f"Validation episodes: {len(val_ids)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Max move: {max_move}")
    print(f"Max time: {max_time}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0

        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            optimizer.zero_grad()

            loss, _, _ = compute_loss(
                model, obs_batch, action_batch
            )

            loss.backward()
            optimizer.step()

            train_total += loss.item() * len(obs_batch)

        train_loss = train_total / len(train_dataset)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                loss, _, _ = compute_loss(
                    model, obs_batch, action_batch
                )

                val_total += loss.item() * len(obs_batch)

        val_loss = val_total / len(val_dataset)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_mean": torch.from_numpy(obs_mean),
            "obs_std": torch.from_numpy(obs_std),
            "max_move": max_move,
            "max_time": max_time,
            "position_range": position_range,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "seed": args.seed,
        },
        output,
    )

    print("\n===== TRAINING RESULT =====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="demonstrations/franka_expert_train.npz",
    )
    parser.add_argument(
        "--output",
        default="bc_models/franka_bc.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
