
import argparse
import subprocess
from pathlib import Path
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--width", type=int, default=640)
args = parser.parse_args()

Path(args.output).parent.mkdir(
    parents=True, exist_ok=True
)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

filters = (
    f"[0:v]fps={args.fps},"
    f"scale={args.width}:-1:flags=lanczos,"
    "split[a][b];"
    "[a]palettegen[p];"
    "[b][p]paletteuse"
)

subprocess.run(
    [
        ffmpeg, "-y",
        "-i", args.input,
        "-filter_complex", filters,
        "-loop", "0",
        args.output,
    ],
    check=True,
)

print(f"Saved GIF: {args.output}")
