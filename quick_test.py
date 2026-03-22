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
        continue

    img = cv2.imread(str(pngs[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, CANVAS_SIZE)

    print(f"\n=== {scene_dir} (frame: {pngs[0].name}) ===")

    try:
        labels, colors = quantize_colors(img, k=16)
        regions = extract_regions(labels, min_area_fraction=0.001)
        shapes = fit_bezier_boundaries(regions, img.shape, epsilon=2.0)
        assigned = assign_slots_heuristic(shapes, canvas_size=CANVAS_SIZE)

        ok, reason = quality_filter(
            assigned, min_shapes=2, max_shapes=N_SLOTS, max_fit_error=50.0
        )

        print(f"  Quantize: {len(np.unique(labels))} clusters")
        print(f"  Regions: {len(regions)}")
        print(f"  Shapes: {len(shapes)}")
        print(f"  Assigned: {len(assigned)}")

        if shapes:
            errors = [s["fit_error"] for s in shapes]
            print(f"  Max fit error: {max(errors):.2f}")

        print(f"  Quality Filter: {'PASS' if ok else 'FAIL'} ({reason})")

    except Exception as e:
        print(f"  ERROR: {e}")
