import imageio_ffmpeg
import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from franka_autonomous_env import FrankaAutonomousEnv
from evaluate_behavior_cloning import load_policy, predict_action


def get_font(size):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


class VideoWriter:
    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height

        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error",
            "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium",
            "-crf", "22", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {code}")
        print(f"Saved video: {self.path}")


def make_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = [0.55, 0.0, 0.52]
    camera.distance = 1.35
    camera.azimuth = 135
    camera.elevation = -25
    return camera


class BCPlayer:
    def __init__(self, model_path, position_range, seed, width, height):
        self.model, self.mean, self.std, ckpt = load_policy(model_path)

        self.env = FrankaAutonomousEnv(position_range=position_range)
        self.env.max_move = float(ckpt["max_move"])
        self.env.max_time = float(ckpt["max_time"])

        self.env.model.vis.global_.offwidth = max(
            self.env.model.vis.global_.offwidth, width
        )
        self.env.model.vis.global_.offheight = max(
            self.env.model.vis.global_.offheight, height
        )

        self.renderer = mujoco.Renderer(
            self.env.model, height=height, width=width
        )
        self.camera = make_camera()
        self.reset(seed)

    def reset(self, seed):
        self.obs, _ = self.env.reset(seed=seed)
        self.seed = seed
        self.done = False
        self.success = False
        self.steps = 0
        self.max_lift = 0.0
        self.cube_x = float(self.env.cube_x)
        self.cube_y = float(self.env.cube_y)

    def step(self):
        if self.done:
            return

        action = predict_action(
            self.model, self.obs, self.mean, self.std
        )

        self.obs, _, terminated, truncated, info = (
            self.env.step(action)
        )

        self.steps += 1
        self.max_lift = max(
            self.max_lift, float(info["lift_height"])
        )
        self.done = bool(terminated or truncated)
        self.success = bool(info["is_success"])

    def render(self):
        self.renderer.update_scene(
            self.env.data, camera=self.camera
        )
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()
        self.env.close()


def add_label(frame, title, status=None):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    width = image.width
    draw.rectangle((0, 0, width, 55), fill=(20, 24, 31))

    font = get_font(21)
    small_font = get_font(16)

    draw.text(
        (14, 8), title,
        font=font, fill=(255, 255, 255)
    )

    if status:
        draw.text(
            (14, 34), status,
            font=small_font, fill=(205, 215, 225)
        )

    return np.asarray(image)


def csv_rows(condition, train_seed, position_range):
    path = Path(
        f"bc_results/final_{condition}_seed{train_seed}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    return df[
        (df["policy"] == "bc")
        & np.isclose(df["position_range"], position_range)
    ].copy()


def choose_comparison_seed(train_seed, position_range):
    narrow = csv_rows(
        "narrow_expert", train_seed, position_range
    )
    wide = csv_rows(
        "wide_expert", train_seed, position_range
    )

    paired = narrow.merge(
        wide, on="seed",
        suffixes=("_narrow", "_wide"),
        validate="one_to_one",
    )

    if not np.allclose(
        paired["cube_x_narrow"], paired["cube_x_wide"]
    ) or not np.allclose(
        paired["cube_y_narrow"], paired["cube_y_wide"]
    ):
        raise ValueError("CSV initial positions do not match")

    candidates = paired[
        (paired["success_narrow"] == 0)
        & (paired["success_wide"] == 1)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No Narrow-failure / Wide-success pair found"
        )

    # Prefer a clearly different maximum lifting height.
    candidates["lift_gap"] = (
        candidates["max_lift_wide"]
        - candidates["max_lift_narrow"]
    )

    best = candidates.sort_values(
        "lift_gap", ascending=False
    ).iloc[0]

    print(
        f"Selected comparison seed={int(best['seed'])}, "
        f"cube=({best['cube_x_narrow']:.4f}, "
        f"{best['cube_y_narrow']:.4f})"
    )
    return int(best["seed"])


def choose_main_seed(train_seed, position_range):
    df = csv_rows(
        "wide_expert", train_seed, position_range
    )
    successes = df[df["success"] == 1].copy()

    if successes.empty:
        raise ValueError("No successful Wide Expert trial found")

    # Prefer a moderately displaced cube for a clear demonstration.
    successes["distance"] = np.hypot(
        successes["cube_x"] - 0.55,
        successes["cube_y"],
    )
    successes = successes.sort_values(
        "distance", ascending=False
    )
    return int(successes.iloc[0]["seed"])


def model_path(condition, train_seed):
    return (
        f"bc_models/fair_{condition}_seed{train_seed}.pt"
    )


def print_result(name, player):
    print(
        f"{name}: success={player.success}, "
        f"steps={player.steps}, "
        f"max_lift={player.max_lift:.3f}m, "
        f"cube=({player.cube_x:.4f}, {player.cube_y:.4f})"
    )


def record_main(args, seed, output):
    player = BCPlayer(
        model_path("wide_expert", args.train_seed),
        args.position_range, seed, args.width, args.height,
    )

    writer = VideoWriter(
        output, args.width, args.height, args.fps
    )

    try:
        frame = add_label(
            player.render(), "Wide Expert BC",
            f"Cube range: +/-{args.position_range:.2f} m",
        )
        writer.write(frame)

        while not player.done:
            player.step()
            frame = add_label(
                player.render(), "Wide Expert BC",
                f"Step {player.steps} | "
                f"Lift {player.max_lift:.3f} m",
            )
            writer.write(frame)

        final_status = (
            "SUCCESS" if player.success else "FAILURE"
        )
        frame = add_label(
            player.render(), "Wide Expert BC",
            f"{final_status} | {player.steps} steps",
        )

        for _ in range(args.fps):
            writer.write(frame)

        print_result("Wide Expert", player)
    finally:
        writer.close()
        player.close()


def record_comparison(args, seed, output):
    narrow = BCPlayer(
        model_path("narrow_expert", args.train_seed),
        args.position_range, seed, args.width, args.height,
    )
    wide = BCPlayer(
        model_path("wide_expert", args.train_seed),
        args.position_range, seed, args.width, args.height,
    )

    if not (
        np.isclose(narrow.cube_x, wide.cube_x)
        and np.isclose(narrow.cube_y, wide.cube_y)
    ):
        raise ValueError("The two cube positions do not match")

    writer = VideoWriter(
        output, args.width * 2, args.height, args.fps
    )

    def make_frame():
        left = add_label(
            narrow.render(), "Narrow Expert BC",
            f"Trained on +/-2 cm | Step {narrow.steps}",
        )
        right = add_label(
            wide.render(), "Wide Expert BC",
            f"Trained on +/-12 cm | Step {wide.steps}",
        )
        return np.concatenate([left, right], axis=1)

    try:
        writer.write(make_frame())

        while not (narrow.done and wide.done):
            narrow.step()
            wide.step()
            writer.write(make_frame())

        left_status = (
            "SUCCESS" if narrow.success else "FAILURE"
        )
        right_status = (
            "SUCCESS" if wide.success else "FAILURE"
        )

        final_frame = np.concatenate([
            add_label(
                narrow.render(), "Narrow Expert BC",
                f"{left_status} | {narrow.steps} steps",
            ),
            add_label(
                wide.render(), "Wide Expert BC",
                f"{right_status} | {wide.steps} steps",
            ),
        ], axis=1)

        for _ in range(args.fps * 2):
            writer.write(final_frame)

        print_result("Narrow Expert", narrow)
        print_result("Wide Expert", wide)
    finally:
        writer.close()
        narrow.close()
        wide.close()


def record_multiple(args, output):
    df = csv_rows(
        "wide_expert", args.train_seed, args.position_range
    )
    df = df[df["success"] == 1].copy()

    if len(df) < 3:
        raise ValueError("Fewer than 3 successful trials")

    # Pick successful trials at distinct cube positions.
    choices = [
        df.loc[df["cube_x"].idxmin()],
        df.loc[df["cube_x"].idxmax()],
        df.loc[df["cube_y"].abs().idxmax()],
    ]

    selected = []
    for row in choices:
        if int(row["seed"]) not in selected:
            selected.append(int(row["seed"]))

    if len(selected) < 3:
        for seed in df["seed"]:
            if int(seed) not in selected:
                selected.append(int(seed))
            if len(selected) == 3:
                break

    player = BCPlayer(
        model_path("wide_expert", args.train_seed),
        args.position_range, selected[0],
        args.width, args.height,
    )
    writer = VideoWriter(
        output, args.width, args.height, args.fps
    )

    try:
        for index, seed in enumerate(selected, 1):
            player.reset(seed)

            title = f"Wide Expert BC | Trial {index}/3"
            subtitle = (
                f"Cube ({player.cube_x:.3f}, "
                f"{player.cube_y:.3f})"
            )

            for _ in range(args.fps // 2):
                writer.write(add_label(
                    player.render(), title, subtitle
                ))

            while not player.done:
                player.step()
                writer.write(add_label(
                    player.render(), title, subtitle
                ))

            print_result(f"Trial {index}", player)

            status = (
                "SUCCESS" if player.success else "FAILURE"
            )
            for _ in range(args.fps):
                writer.write(add_label(
                    player.render(), title, status
                ))
    finally:
        writer.close()
        player.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["main", "comparison", "multiple"],
        required=True,
    )
    parser.add_argument(
        "--train-seed", type=int, default=42
    )
    parser.add_argument(
        "--seed", type=int, default=None
    )
    parser.add_argument(
        "--range", dest="position_range",
        type=float, default=None,
    )
    parser.add_argument(
        "--width", type=int, default=640
    )
    parser.add_argument(
        "--height", type=int, default=480
    )
    parser.add_argument(
        "--fps", type=int, default=10
    )
    parser.add_argument(
        "--output", default=None
    )
    args = parser.parse_args()

    if args.position_range is None:
        args.position_range = (
            0.12 if args.mode == "main" else 0.20
        )

    if args.width % 2 or args.height % 2:
        raise ValueError("Width and height must be even")

    output = args.output or (
        f"assets/franka/franka-{args.mode}.mp4"
    )

    if args.mode == "main":
        seed = (
            args.seed if args.seed is not None
            else choose_main_seed(
                args.train_seed, args.position_range
            )
        )
        record_main(args, seed, output)

    elif args.mode == "comparison":
        seed = (
            args.seed if args.seed is not None
            else choose_comparison_seed(
                args.train_seed, args.position_range
            )
        )
        record_comparison(args, seed, output)

    else:
        record_multiple(args, output)


if __name__ == "__main__":
    main()
