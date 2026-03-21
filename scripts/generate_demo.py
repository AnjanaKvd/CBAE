#!/usr/bin/env python
# scripts/generate_demo.py
"""
Generate demo videos from a trained CBAE checkpoint.

Loads a trained model, runs inference on text prompts, re-rasterises
the predicted topology at high resolution using the Cairo rasterizer,
and outputs GIF + MP4 files.

Usage:
    python scripts/generate_demo.py \
        --checkpoint checkpoints/stage3_pretrain/model_epoch_0100.pt \
        --output_dir demo_output/ \
        --prompts "a blue character breathing" "a red figure standing"

    python scripts/generate_demo.py \
        --checkpoint checkpoints/stage3_pretrain/model_epoch_0100.pt \
        --render_size 256 --fps 8
"""

import argparse
import logging
import os
import time

import imageio
import numpy as np
import torch

from models.cbae_model import CBAE_EndToEnd
from core.crf_tensor import CRFTensor
from rendering.rasterizer import rasterize

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("CBAE-demo")

# ── Default prompts ─────────────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    "a blue character standing and breathing slowly",
    "a red figure with arms raised",
    "a green character swaying gently",
    "a character blinking its eyes",
    "a purple figure waving hello",
]


def topology_to_frames(
    topology: dict,
    video_tensor: torch.Tensor,
    render_size: int = 256,
    batch_idx: int = 0,
) -> list:
    """
    Re-rasterise model topology at high resolution using Cairo.

    Takes the predicted P, colors, aliveness, alpha, z, csg from model output
    and renders each frame at render_size × render_size.

    Args:
        topology:     dict from CBAE_EndToEnd.forward()
        video_tensor: (batch, T, H, W, 3) — used only for T dimension
        render_size:  output resolution (width = height)
        batch_idx:    which batch item to render

    Returns:
        List of (H, W, 3) uint8 numpy arrays
    """
    P_seq = topology["P"][batch_idx]            # (T, 128, 12, 2)
    aliveness_seq = topology["aliveness"][batch_idx]  # (T, 128)
    colors = topology["colors"][batch_idx]      # (128, 3)
    alpha = topology["alpha"]                   # (128,)
    z = topology["z"]                           # (128,)
    csg = topology["csg"]                       # (128,)

    T = P_seq.shape[0]
    frames = []

    for t in range(T):
        # Build a CRFTensor for this frame
        crf = CRFTensor()

        # Copy predicted values
        crf.P = P_seq[t].detach().cpu().numpy().astype(np.float16)
        crf.c = colors.detach().cpu().numpy().astype(np.float16)
        crf.alpha = alpha.detach().cpu().numpy().astype(np.float16)
        crf.z = z.detach().cpu().numpy().astype(np.int8)
        crf.csg = csg.detach().cpu().numpy().astype(bool)

        # Set aliveness from model prediction
        crf.alive = aliveness_seq[t].detach().cpu().numpy().astype(np.float16)

        # Smooth alpha: alpha * sigmoid(alive) — match model forward
        alive_f32 = crf.alive.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-alive_f32))
        crf.alpha = (crf.alpha.astype(np.float32) * sig).astype(np.float16)

        # Rasterise at high resolution
        rgb = rasterize(crf, width=render_size, height=render_size)
        frames.append(rgb)

    return frames


def save_gif(frames: list, path: str, fps: int = 8):
    """Save frames as animated GIF."""
    imageio.mimsave(path, frames, fps=fps, loop=0)
    log.info("GIF saved → %s (%.1f MB)", path, os.path.getsize(path) / 1e6)


def save_mp4(frames: list, path: str, fps: int = 8):
    """Save frames as MP4 video."""
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    log.info("MP4 saved → %s (%.1f MB)", path, os.path.getsize(path) / 1e6)


def create_comparison_grid(all_frames: dict, render_size: int) -> list:
    """
    Create a side-by-side comparison grid from multiple prompts.
    Returns list of composite frames.
    """
    prompts = list(all_frames.keys())
    n_prompts = len(prompts)

    if n_prompts == 0:
        return []

    T = len(all_frames[prompts[0]])

    # Layout: up to 3 per row
    cols = min(n_prompts, 3)
    rows = (n_prompts + cols - 1) // cols

    grid_w = cols * render_size
    grid_h = rows * render_size

    composite_frames = []
    for t in range(T):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, prompt in enumerate(prompts):
            r, c = divmod(i, cols)
            y0 = r * render_size
            x0 = c * render_size

            if t < len(all_frames[prompt]):
                canvas[y0:y0 + render_size, x0:x0 + render_size] = \
                    all_frames[prompt][t]

        composite_frames.append(canvas)

    return composite_frames


def main():
    parser = argparse.ArgumentParser(description="Generate CBAE demo videos")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Output directory for videos")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Text prompts for generation")
    parser.add_argument("--render_size", type=int, default=256,
                        help="Render resolution (default: 256)")
    parser.add_argument("--model_render_size", type=int, default=32,
                        help="Model internal render size (must match training)")
    parser.add_argument("--max_frames", type=int, default=4,
                        help="Frames per sequence (must match training)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Output video FPS")
    parser.add_argument("--no_grid", action="store_true",
                        help="Skip comparison grid generation")
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")
    log.info("Device: %s", device)

    # ── Load Model ─────────────────────────────────────────────────────────
    log.info("Building model …")
    model = CBAE_EndToEnd(
        render_width=args.model_render_size,
        render_height=args.model_render_size,
        n_steps=args.max_frames,
    ).to(device)

    log.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stage = ckpt.get("stage", "unknown")
    epoch = ckpt.get("epoch", "?")
    log.info("Loaded: stage=%s, epoch=%s", stage, epoch)

    # ── Generate ───────────────────────────────────────────────────────────
    all_frames = {}
    dummy_audio = torch.zeros(1, 8 * 16000)  # silent audio, batch=1

    for i, prompt in enumerate(prompts):
        log.info("[%d/%d] Generating: \"%s\"", i + 1, len(prompts), prompt)

        t0 = time.time()
        with torch.no_grad():
            video_tensor, topology = model(prompt, dummy_audio)
        inference_time = time.time() - t0
        log.info("  Inference: %.1fs", inference_time)

        # Re-rasterise at high resolution
        log.info("  Rasterising at %dx%d …", args.render_size, args.render_size)
        frames = topology_to_frames(
            topology, video_tensor,
            render_size=args.render_size,
            batch_idx=0,
        )
        all_frames[prompt] = frames
        log.info("  Got %d frames", len(frames))

        # Save individual GIF
        safe_name = prompt.replace(" ", "_")[:40]
        gif_path = os.path.join(args.output_dir, f"{i:02d}_{safe_name}.gif")
        save_gif(frames, gif_path, fps=args.fps)

        # Save individual MP4
        try:
            mp4_path = os.path.join(args.output_dir, f"{i:02d}_{safe_name}.mp4")
            save_mp4(frames, mp4_path, fps=args.fps)
        except Exception as e:
            log.warning("  MP4 save failed (ffmpeg missing?): %s", e)

    # ── Comparison Grid ────────────────────────────────────────────────────
    if not args.no_grid and len(all_frames) > 1:
        log.info("Creating comparison grid …")
        grid_frames = create_comparison_grid(all_frames, args.render_size)
        grid_gif = os.path.join(args.output_dir, "comparison_grid.gif")
        save_gif(grid_frames, grid_gif, fps=args.fps)

        try:
            grid_mp4 = os.path.join(args.output_dir, "comparison_grid.mp4")
            save_mp4(grid_frames, grid_mp4, fps=args.fps)
        except Exception as e:
            log.warning("Grid MP4 failed: %s", e)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("═" * 50)
    log.info("Demo generation complete!")
    log.info("  Checkpoint: %s (stage=%s, epoch=%s)", args.checkpoint, stage, epoch)
    log.info("  Prompts: %d", len(prompts))
    log.info("  Resolution: %dx%d", args.render_size, args.render_size)
    log.info("  Output: %s", os.path.abspath(args.output_dir))
    log.info("═" * 50)


if __name__ == "__main__":
    main()
