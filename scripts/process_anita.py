#!/usr/bin/env python
# scripts/process_anita.py
"""
Converts Anita Dataset PNG frames into CBAE CRFSequence HDF5 files.

Features:
    - Auto-discovers scene folders recursively (no input list needed)
    - Sliding-window chunking: extracts overlapping subsequences from long scenes
    - Flat-color pre-filter: skips heavily shaded/photorealistic frames
    - Accepts sequences >= 6 frames to capture short animation clips

Usage:
    python scripts/process_anita.py \
        --input_dir data/real/anita_files/ \
        --output_dir data/real/processed_anita/ \
        --sequence_length 24

    python scripts/process_anita.py \
        --input_dir data/real/anita_files/ \
        --output_dir data/real/processed_anita/ \
        --sequence_length 12 --min_frames 6 --flat_threshold 0.70
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core.constants import N_SLOTS, CANVAS_SIZE
from core.crf_tensor import CRFTensor, CRFSequence
from data.pipeline import (
    quantize_colors,
    extract_regions,
    fit_bezier_boundaries,
    assign_slots_heuristic,
    quality_filter,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("anita-proc")


# ── Helpers ─────────────────────────────────────────────────────────────────

def flat_color_score(frame: np.ndarray) -> float:
    """
    Measures how 'flat-color' a frame is.
    Returns fraction of pixels with low local variance (higher = flatter).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(float)
    mean = cv2.blur(gray, (7, 7))
    sq_mean = cv2.blur(gray ** 2, (7, 7))
    variance = sq_mean - mean ** 2
    return float((variance < 50).mean())


def discover_scenes(input_dir: str) -> list:
    """
    Auto-discover all scene folders containing .png files.
    Returns list of (scene_path, series_name, scene_name, n_frames).
    """
    root = Path(input_dir)
    scenes = []

    for top_dir in sorted(root.iterdir()):
        if not top_dir.is_dir():
            continue
        series = top_dir.name

        # Check if top_dir itself has PNGs (flat structure)
        direct_pngs = sorted(top_dir.glob("*.png"))
        if direct_pngs:
            scenes.append((str(top_dir), series, series, len(direct_pngs)))
            continue

        # Otherwise check subdirectories
        for sub in sorted(top_dir.iterdir()):
            if not sub.is_dir():
                continue
            pngs = sorted(sub.rglob("*.png"))
            if pngs:
                scenes.append((str(sub), series, sub.name, len(pngs)))

    return scenes


def load_scene_frames(scene_dir: str, max_frames: int = 96) -> list:
    """Load consecutive PNG frames from a scene directory, resized to CANVAS_SIZE."""
    frame_files = sorted(Path(scene_dir).rglob("*.png"))[:max_frames]
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, CANVAS_SIZE)
        frames.append(img)
    return frames


def frame_to_crf(frame: np.ndarray, colors: np.ndarray,
                 assigned: dict) -> CRFTensor:
    """Build a CRFTensor from pipeline output."""
    crf = CRFTensor()
    for slot_idx, shape_data in assigned.items():
        crf.P[slot_idx] = shape_data["control_points"]
        color_idx = shape_data["color_idx"]
        if colors is not None and color_idx < len(colors):
            crf.c[slot_idx] = np.clip(colors[color_idx], 0, 1).astype(np.float16)
        crf.alpha[slot_idx] = shape_data.get("alpha", 1.0)
        crf.alive[slot_idx] = 5.0
        crf.csg[slot_idx] = shape_data.get("csg", False)
        crf.z[slot_idx] = slot_idx
    return crf


def compute_velocities(crf_frames: list) -> np.ndarray:
    """Finite-difference velocities from CRF frame sequence."""
    dp_dts = []
    n = len(crf_frames)
    for i in range(n):
        if i == 0:
            v = crf_frames[1].P.astype(np.float32) - crf_frames[0].P.astype(np.float32)
        elif i == n - 1:
            v = crf_frames[-1].P.astype(np.float32) - crf_frames[-2].P.astype(np.float32)
        else:
            v = (crf_frames[i + 1].P.astype(np.float32) -
                 crf_frames[i - 1].P.astype(np.float32)) / 2.0
        dp_dts.append(v.astype(np.float16))
    return np.stack(dp_dts, axis=0)


# ── Main Processing ─────────────────────────────────────────────────────────

def process_frames_to_crfs(frames: list, flat_threshold: float = 0.70) -> list:
    """
    Run the full pipeline on a list of RGB frames.
    Returns list of valid CRFTensors (frames that failed quality filter are skipped).
    """
    crf_frames = []

    for frame in frames:
        # Pre-filter: skip non-flat frames
        score = flat_color_score(frame)
        if score < flat_threshold:
            continue

        try:
            palette_img, colors = quantize_colors(frame, k=16)
            regions = extract_regions(palette_img, min_area_fraction=0.001)
            shapes = fit_bezier_boundaries(regions, frame.shape, epsilon=2.0)
            assigned = assign_slots_heuristic(shapes, canvas_size=CANVAS_SIZE)

            ok, _ = quality_filter(
                shapes=assigned,
                min_shapes=2,    # real frames can have just character + bg
                max_shapes=N_SLOTS,
                max_fit_error=50.0,  # real contours are complex (15-25px typical)
            )
            if not ok:
                continue

            crf = frame_to_crf(frame, colors, assigned)
            crf_frames.append(crf)

        except Exception as e:
            # Skip frames that cause any pipeline error
            continue

    return crf_frames


def extract_sequences(crf_frames: list, sequence_length: int,
                      min_frames: int, stride: int = None) -> list:
    """
    Extract sub-sequences using sliding window.
    For short scenes, returns the whole scene as one sequence.
    For long scenes, returns overlapping chunks.
    """
    n = len(crf_frames)
    if n < min_frames:
        return []

    if stride is None:
        stride = max(1, sequence_length // 2)

    sequences = []

    if n <= sequence_length:
        # Short scene — return as-is
        dp_dt = compute_velocities(crf_frames)
        sequences.append(CRFSequence(crf_list=crf_frames, dp_dt=dp_dt))
    else:
        # Sliding window for long scenes
        start = 0
        while start + min_frames <= n:
            end = min(start + sequence_length, n)
            chunk = crf_frames[start:end]
            dp_dt = compute_velocities(chunk)
            sequences.append(CRFSequence(crf_list=chunk, dp_dt=dp_dt))
            start += stride
            if end == n:
                break

    return sequences


def process_anita(input_dir: str, output_dir: str, sequence_length: int,
                  min_frames: int, flat_threshold: float, max_frames_load: int):
    """Main processing loop."""
    os.makedirs(output_dir, exist_ok=True)

    scenes = discover_scenes(input_dir)
    log.info("Discovered %d scenes in %s", len(scenes), input_dir)

    # Per-series stats
    series_stats = {}
    saved_total = 0
    skipped_total = 0

    for scene_path, series, scene_name, n_raw in tqdm(scenes, desc="Processing"):
        frames = load_scene_frames(scene_path, max_frames=max_frames_load)
        if len(frames) < min_frames:
            skipped_total += 1
            continue

        crf_frames = process_frames_to_crfs(frames, flat_threshold=flat_threshold)

        if len(crf_frames) < min_frames:
            skipped_total += 1
            series_stats.setdefault(series, {"saved": 0, "skipped": 0, "seqs": 0})
            series_stats[series]["skipped"] += 1
            continue

        sequences = extract_sequences(
            crf_frames, sequence_length, min_frames,
        )

        series_stats.setdefault(series, {"saved": 0, "skipped": 0, "seqs": 0})

        for seq in sequences:
            out_path = os.path.join(output_dir, f"seq_{saved_total:04d}.h5")
            seq.to_hdf5(out_path)
            saved_total += 1
            series_stats[series]["seqs"] += 1

        series_stats[series]["saved"] += 1

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Processing complete!")
    log.info("  Total scenes: %d", len(scenes))
    log.info("  Skipped: %d", skipped_total)
    log.info("  Sequences saved: %d", saved_total)
    log.info("  Output: %s", os.path.abspath(output_dir))
    log.info("")
    log.info("Per-series breakdown:")
    for series in sorted(series_stats):
        s = series_stats[series]
        log.info("  %-25s %3d scenes OK, %3d skipped, %3d sequences",
                 series, s["saved"], s["skipped"], s["seqs"])
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Anita Dataset to CBAE HDF5 sequences"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root directory of Anita PNG scenes")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .h5 files")
    parser.add_argument("--sequence_length", type=int, default=24,
                        help="Target frames per sequence (default: 24)")
    parser.add_argument("--min_frames", type=int, default=6,
                        help="Minimum frames to accept a scene (default: 6)")
    parser.add_argument("--flat_threshold", type=float, default=0.70,
                        help="Flat-color score threshold (default: 0.70)")
    parser.add_argument("--max_frames_load", type=int, default=96,
                        help="Max frames to load per scene (default: 96)")
    args = parser.parse_args()
    process_anita(**vars(args))
