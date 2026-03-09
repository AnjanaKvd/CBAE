import os
import argparse
import imageio
from tqdm import tqdm

from generation.synthetic import (
    generate_sequence,
    generate_base_character,
    generate_character_variant,
)
from generation.motion_functions import breathing_motion, eye_blink, gentle_sway
from generation.noise_schedule import NOISE_CLEAN, NOISE_ROBUSTNESS, NOISE_BRIDGE
from rendering.rasterizer import rasterize


def render_demo_video(
    output_path: str = "output/demo.mp4", noise_level: str = "clean", seed: int = 42
):
    """
    Generates a CRF sequence, rasterizes 192 frames, and encodes them to an MP4 video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating synthetic sequence (seed={seed}, noise={noise_level})...")

    # Select noise config
    noise_config = NOISE_CLEAN
    if noise_level == "robustness":
        noise_config = NOISE_ROBUSTNESS
    elif noise_level == "bridge":
        noise_config = NOISE_BRIDGE

    def make_char():
        base = generate_base_character(style="robe")
        return generate_character_variant(base, seed=seed)

    seq = generate_sequence(
        character_fn=make_char,
        motion_fns=[breathing_motion, gentle_sway, eye_blink],
        noise_config=noise_config,
        n_frames=192,
        fps=24,
    )

    print("Rasterizing frames...")
    frames_rgb = []

    for frame in tqdm(seq.frames, desc="Rasterizing"):
        # Rasterize directly returns (512, 512, 3) uint8 numpy array matching spec
        img = rasterize(frame, width=512, height=512)
        frames_rgb.append(img)

    print(f"Encoding video to {output_path}...")

    # Uses imageio with the ffmpeg plugin
    # macro_block_size=None prevents imageio from resizing to multiples of 16 if not needed
    # (though 512 is obviously a multiple of 16 anyway).
    imageio.mimwrite(
        output_path,
        frames_rgb,
        fps=24,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    print("Done! Video rendered successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a CBAE demo animation video.")
    parser.add_argument("--output", default="output/demo.mp4", help="Output video path")
    parser.add_argument(
        "--noise",
        choices=["clean", "robustness", "bridge"],
        default="clean",
        help="Noise intensity schedule",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for character variation"
    )
    args = parser.parse_args()

    render_demo_video(output_path=args.output, noise_level=args.noise, seed=args.seed)
