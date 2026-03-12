"""
evaluation/benchmark.py — CBAE Evaluation & Benchmarking Pipeline

Loads a trained checkpoint, runs inference on a set of prompts,
computes topological + perceptual metrics, and writes results to JSON.

Usage:
    python evaluation/benchmark.py --checkpoint checkpoints/cbae_epoch0050.pt
    python evaluation/benchmark.py --checkpoint ckpt.pt --prompts prompts.txt
    python evaluation/benchmark.py --checkpoint ckpt.pt --output-dir evaluation/
"""

import argparse
import json
import logging
import os
import time
import tracemalloc
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.cbae_model import CBAE_EndToEnd
from models.encoders import CLIPEncoder
from rendering.diff_rasterizer import bezier_to_polyline_torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("CBAE-eval")

# ---------------------------------------------------------------------------
# Default prompts (fallback when none supplied)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS: List[str] = [
    "a gentle blue character breathing slowly",
    "a red block jumping up and down",
    "a green circle swaying side to side",
    "a yellow triangle blinking its eyes",
    "a purple oval standing still",
    "a pink character waving its arms",
    "an orange figure dancing rhythmically",
    "a white ghost-like shape hovering",
    "a dark blue wizard casting a spell",
    "a small brown creature walking forward",
]


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def compute_bcs(topology: Dict[str, torch.Tensor]) -> float:
    """
    Boundary Constraint Score — measures temporal smoothness of shape curvature.

    Args:
        topology: dict with 'P' (batch, T, slots, 12, 2) and 'aliveness' (batch, T, slots)

    Returns:
        Scalar BCS value (lower = smoother boundaries).
    """
    P = topology["P"]
    aliveness = topology["aliveness"]
    batch, T, slots, _, _ = P.shape

    P_flat = P.reshape(batch * T, slots, 12, 2)
    polylines = bezier_to_polyline_torch(P_flat, n_samples=30)

    # 1st and 2nd derivatives along curve parameter u
    dP_du = polylines[:, :, 1:, :] - polylines[:, :, :-1, :]
    d2P_du2 = dP_du[:, :, 1:, :] - dP_du[:, :, :-1, :]
    dP_du_t = dP_du[:, :, :-1, :]

    cross = torch.abs(
        dP_du_t[..., 0] * d2P_du2[..., 1] - dP_du_t[..., 1] * d2P_du2[..., 0]
    )
    denom = torch.pow(torch.norm(dP_du_t, dim=-1), 3) + 1e-6
    kappa = cross / denom
    K = kappa.mean(dim=-1).view(batch, T, slots)

    dK = torch.abs(K[:, 1:, :] - K[:, :-1, :])
    alive_w = torch.sigmoid(aliveness[:, 1:, :])
    return (dK * alive_w).mean().item()


def compute_crs(topology: Dict[str, torch.Tensor]) -> float:
    """
    Content Retention Score — measures temporal colour consistency.

    Args:
        topology: dict with 'colors' (batch, slots, 3) or (batch, T, slots, 3)
                  and 'aliveness' (batch, T, slots)

    Returns:
        Scalar CRS value (lower = more consistent colours).
    """
    colors = topology["colors"]
    if colors.dim() == 3:
        # Static colours ⇒ variance is trivially 0.
        return 0.0

    aliveness = topology["aliveness"]
    c_mean = colors.mean(dim=1, keepdim=True)
    mse = (colors - c_mean).pow(2).sum(dim=-1)
    alive_w = torch.sigmoid(aliveness)
    return (mse * alive_w).mean().item()


def compute_clip_score(
    video_tensor: torch.Tensor,
    prompt: str,
    clip_encoder: CLIPEncoder,
) -> float:
    """
    CLIP cosine similarity between the mean rendered frame and the text prompt.

    Args:
        video_tensor: (1, T, H, W, 3)
        prompt: text string
        clip_encoder: frozen CLIPEncoder instance

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    with torch.no_grad():
        text_emb = clip_encoder.encode_text(prompt)  # (1, 512)

        # Average across time to get a single representative frame
        mean_frame = video_tensor[0].mean(dim=0)  # (H, W, 3)

        # Flatten frame into a pseudo-embedding (projection placeholder)
        frame_flat = mean_frame.reshape(-1)
        # Project to 512 via simple linear interpolation (placeholder for a
        # real CLIP ViT image encoder; sufficient for the benchmark scaffold)
        if frame_flat.shape[0] != 512:
            frame_emb = F.adaptive_avg_pool1d(
                frame_flat.unsqueeze(0).unsqueeze(0), 512
            ).squeeze()
        else:
            frame_emb = frame_flat

        frame_emb = F.normalize(frame_emb.unsqueeze(0), dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        sim = F.cosine_similarity(frame_emb, text_emb).item()

    return sim


def compute_hei(video_tensor: torch.Tensor) -> float:
    """
    Human Eval Index proxy — lightweight perceptual-quality heuristic.

    Measures overall image "energy" (mean gradient magnitude across frames)
    as a proxy for visual sharpness / non-blankness.

    Args:
        video_tensor: (1, T, H, W, 3)

    Returns:
        Scalar HEI score (higher = sharper / more detailed).
    """
    frames = video_tensor[0]  # (T, H, W, 3)
    # Sobel-like gradient approximation
    dx = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean()
    dy = (frames[:, 1:, :, :] - frames[:, :-1, :, :]).abs().mean()
    return (dx + dy).item()


def profile_resources(
    model: CBAE_EndToEnd,
    prompt: str,
    audio: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Measures peak RAM and wall-clock inference time for one forward pass.

    Returns:
        dict with 'ram_mb' and 'inference_time_s'.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    with torch.no_grad():
        _ = model(prompt, audio)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "ram_mb": peak / (1024 * 1024),
        "inference_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Top-level evaluation driver
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    prompts: List[str],
    device: torch.device,
) -> Dict:
    """
    Loads a model from *checkpoint_path*, runs inference on each prompt,
    and returns aggregated metrics.
    """
    log.info("Loading checkpoint: %s", checkpoint_path)

    model = CBAE_EndToEnd(render_width=64, render_height=64).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log.info("Loaded weights from epoch %d", ckpt.get("epoch", -1))
    else:
        log.warning("Checkpoint not found — running with random weights.")

    model.eval()

    clip_encoder = CLIPEncoder()

    # Dummy audio tensor (8 s silence @ 16 kHz)
    audio = torch.zeros(1, 8 * 16_000, device=device)

    all_results: List[Dict] = []

    for i, prompt in enumerate(prompts):
        log.info("Prompt %d/%d: %s", i + 1, len(prompts), prompt)

        with torch.no_grad():
            video_tensor, topology = model(prompt, audio)

        bcs  = compute_bcs(topology)
        crs  = compute_crs(topology)
        clip = compute_clip_score(video_tensor, prompt, clip_encoder)
        hei  = compute_hei(video_tensor)

        res = profile_resources(model, prompt, audio, device)

        entry = {
            "prompt": prompt,
            "bcs": bcs,
            "crs": crs,
            "clip_score": clip,
            "hei": hei,
            "ram_mb": res["ram_mb"],
            "inference_time_s": res["inference_time_s"],
        }
        log.info(
            "  bcs=%.4f  crs=%.4f  clip=%.4f  hei=%.4f  ram=%.1fMB  time=%.2fs",
            bcs, crs, clip, hei, res["ram_mb"], res["inference_time_s"],
        )
        all_results.append(entry)

    # Aggregate across prompts
    agg = {}
    for key in ["bcs", "crs", "clip_score", "hei", "ram_mb", "inference_time_s"]:
        vals = [r[key] for r in all_results]
        agg[key] = float(np.mean(vals))

    return {"per_prompt": all_results, "aggregate": agg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBAE Evaluation Benchmark")
    p.add_argument(
        "--checkpoint", type=str, default="checkpoints/cbae_epoch0050.pt",
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--prompts", type=str, nargs="*", default=None,
        help="Inline prompts or path to a .txt file (one per line)",
    )
    p.add_argument(
        "--output-dir", type=str, default="evaluation",
        help="Directory for results.json output",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    log.info("Device: %s", device)

    # Resolve prompts
    prompts: Optional[List[str]] = None

    if args.prompts:
        # If a single .txt file is given, read lines from it
        if len(args.prompts) == 1 and args.prompts[0].endswith(".txt"):
            with open(args.prompts[0], "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = args.prompts

    if not prompts:
        log.info("No prompts supplied — using 10 built-in defaults.")
        prompts = DEFAULT_PROMPTS

    results = evaluate_checkpoint(args.checkpoint, prompts, device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s", out_path)

    agg = results["aggregate"]
    log.info(
        "=== Aggregate ===  bcs=%.4f  crs=%.4f  clip=%.4f  hei=%.4f  "
        "ram=%.1fMB  time=%.2fs",
        agg["bcs"], agg["crs"], agg["clip_score"],
        agg["hei"], agg["ram_mb"], agg["inference_time_s"],
    )


if __name__ == "__main__":
    main()
