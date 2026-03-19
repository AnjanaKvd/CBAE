#!/usr/bin/env python
# scripts/process_anita.py
"""
Converts Anita Dataset PNG frames into CBAE CRFSequence HDF5 files.

Usage:
    python scripts/process_anita.py \
        --input_list data/real/anita_color_files.txt \
        --output_dir data/real/processed_anita/ \
        --sequence_length 48 \
        --target_sequences 112
"""

import argparse
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


def load_scene_frames(scene_dir: str, max_frames: int = 48):
    """Load consecutive PNG frames from one Anita scene directory."""
    frame_files = sorted(Path(scene_dir).glob("*.png"))[:max_frames]
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, CANVAS_SIZE)
        frames.append(img)
    return frames


def frames_to_crf_sequence(frames: list, colors_lookup: np.ndarray = None,
                           scene_name: str = ""):
    """
    Convert a list of RGB frames to a CRFSequence.
    Returns None if quality filter rejects the sequence.
    """
    crf_frames = []

    for i, frame in enumerate(frames):
        # Step 1: Color quantization
        palette_img, colors = quantize_colors(frame, k=16)

        # Step 2: Region extraction
        regions = extract_regions(palette_img, min_area_fraction=0.001)

        # Step 3: Bezier boundary fitting
        shapes = fit_bezier_boundaries(regions, frame.shape, epsilon=2.0)

        # Step 4: Slot assignment
        assigned = assign_slots_heuristic(shapes, canvas_size=CANVAS_SIZE)

        # Step 5: Quality filter
        ok, reason = quality_filter(
            shapes=assigned,
            min_shapes=8,
            max_shapes=N_SLOTS,
            max_fit_error=8.0,
        )
        if not ok:
            continue

        # Build CRFTensor
        crf = CRFTensor()
        for slot_idx, shape_data in assigned.items():
            crf.P[slot_idx] = shape_data["control_points"]
            # Look up actual RGB colour from palette
            color_idx = shape_data["color_idx"]
            if colors is not None and color_idx < len(colors):
                crf.c[slot_idx] = np.clip(colors[color_idx], 0, 1).astype(np.float16)
            crf.alpha[slot_idx] = shape_data.get("alpha", 1.0)
            crf.alive[slot_idx] = 5.0  # activated
            crf.csg[slot_idx] = shape_data.get("csg", False)
            crf.z[slot_idx] = slot_idx  # z-order = slot index

        crf_frames.append(crf)

    if len(crf_frames) < 24:
        print(f"  REJECTED {scene_name}: only {len(crf_frames)} valid frames")
        return None

    # Compute velocities via finite differences
    dp_dts = []
    n_crf = len(crf_frames)
    for i in range(n_crf):
        if n_crf == 1:
            # Only one frame — zero velocity
            v = np.zeros_like(crf_frames[0].P, dtype=np.float32)
        elif i == 0:
            v = crf_frames[1].P.astype(np.float32) - crf_frames[0].P.astype(np.float32)
        elif i == n_crf - 1:
            v = crf_frames[-1].P.astype(np.float32) - crf_frames[-2].P.astype(np.float32)
        else:
            v = (crf_frames[i + 1].P.astype(np.float32) - crf_frames[i - 1].P.astype(np.float32)) / 2.0
        dp_dts.append(v.astype(np.float16))

    return CRFSequence(crf_list=crf_frames, dp_dt=np.stack(dp_dts, axis=0))


def process_anita(input_list: str, output_dir: str,
                  sequence_length: int, target_sequences: int):
    """Main processing loop."""
    os.makedirs(output_dir, exist_ok=True)

    # Resolve paths relative to the project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent

    # Group files by scene directory
    with open(input_list) as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # Normalise each path: if relative, resolve against project root
    all_files = []
    for line in raw_lines:
        p = Path(line)
        if not p.is_absolute():
            p = project_root / p
        all_files.append(str(p))

    scenes = {}
    for fpath in all_files:
        scene_dir = str(Path(fpath).parent)
        scenes.setdefault(scene_dir, []).append(fpath)

    print(f"Found {len(scenes)} scenes from {len(all_files)} files")

    saved = 0
    rejected = 0
    for scene_dir, frame_files in tqdm(scenes.items(), desc="Processing scenes"):
        if saved >= target_sequences:
            break

        if len(frame_files) < 12:
            continue

        frames = load_scene_frames(scene_dir, max_frames=sequence_length)
        if len(frames) < 12:
            continue

        seq = frames_to_crf_sequence(
            frames, scene_name=os.path.basename(scene_dir)
        )
        if seq is None:
            rejected += 1
            continue

        out_path = os.path.join(output_dir, f"seq_{saved:04d}.h5")
        seq.to_hdf5(out_path)
        saved += 1

    print(f"\nDone. Saved {saved} sequences to {output_dir}")
    print(f"Rejected: {rejected} scenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Anita Dataset to CBAE HDF5 sequences"
    )
    parser.add_argument("--input_list", required=True,
                        help="Text file listing PNG paths (one per line)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .h5 files")
    parser.add_argument("--sequence_length", type=int, default=48,
                        help="Max frames per scene (default: 48)")
    parser.add_argument("--target_sequences", type=int, default=500,
                        help="Target number of output sequences (default: 500)")
    args = parser.parse_args()
    process_anita(**vars(args))
