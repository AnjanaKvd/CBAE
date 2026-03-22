import cv2
import numpy as np
from pathlib import Path

from data.pipeline import (
    quantize_colors,
    extract_regions,
    fit_bezier_boundaries,
    assign_slots_heuristic,
    quality_filter,
)
from core.constants import N_SLOTS, CANVAS_SIZE

# Pick a frame from a good series
scenes = [
    "data/real/anita_files/dogmatism/188_a",
    "data/real/anita_files/hero/212_a",
    "data/real/anita_files/hope/15_a",
]

for scene_dir in scenes:
    pngs = sorted(Path(scene_dir).glob("*.png"))
    if not pngs:
        print(f"{scene_dir}: no PNGs")
        continue

    img = cv2.imread(str(pngs[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, CANVAS_SIZE)

    print(f"\n=== {scene_dir} (frame: {pngs[0].name}) ===")
    print(f"  Shape: {img.shape}")

    # Step 1: quantize
    try:
        labels, colors = quantize_colors(img, k=16)
        n_clusters = len(np.unique(labels))
        print(f"  Quantize: {n_clusters} clusters, {len(colors)} palette entries")
    except Exception as e:
        print(f"  Quantize FAILED: {e}")
        continue

    # Step 2: regions
    regions = extract_regions(labels, min_area_fraction=0.001)
    print(f"  Regions: {len(regions)} found")

    if regions:
        areas = [r["area_frac"] for r in regions]
        print(f"    Area range: {min(areas):.4f} - {max(areas):.4f}")

    # Step 3: bezier fit
    shapes = fit_bezier_boundaries(regions, img.shape, epsilon=2.0)
    print(f"  Shapes: {len(shapes)} fitted")

    if shapes:
        errors = [s["fit_error"] for s in shapes]
        print(f"    Fit error range: {min(errors):.2f} - {max(errors):.2f}")

    # Step 4: assign slots
    assigned = assign_slots_heuristic(shapes, canvas_size=CANVAS_SIZE)
    print(f"  Assigned: {len(assigned)} slots")

    # Step 5: quality filter
    ok, reason = quality_filter(
        assigned, min_shapes=4, max_shapes=N_SLOTS, max_fit_error=5.0
    )
    print(f"  Quality: {'PASS' if ok else 'FAIL'} ({reason})")
