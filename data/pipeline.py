# data/pipeline.py
"""
Real-data preprocessing pipeline for CBAE.
Converts raw RGB frames into CRFTensor-compatible slot assignments.

Functions:
    quantize_colors      — K-means in LAB → color palette + labels
    extract_regions      — connected components → region dicts
    fit_bezier_boundaries — contour → 12-point bezier control points
    assign_slots_heuristic — rule-based centroid/area → slot index
    quality_filter        — pass/fail gate on shape count & fit error
"""

import cv2
import numpy as np
from scipy import interpolate
from sklearn.cluster import KMeans

from core.constants import (
    N_CTRL_PTS,
    SLOT_BG_STATIC,
    SLOT_BG_DYNAMIC,
    SLOT_BODY,
    SLOT_FACE,
    SLOT_MOUTH,
    SLOT_SECONDARY,
    SLOT_DYNAMIC,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Color Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_colors(frame: np.ndarray, k: int = 16):
    """
    K-means color quantization in CIELAB space.

    Args:
        frame: (H, W, 3) uint8 RGB image.
        k:     number of color clusters.

    Returns:
        labels: (H, W) int32 — cluster index per pixel.
        colors: (k, 3) float64 — RGB palette in [0, 1].
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

    # Clamp k to the number of actual unique colors (downsampled for speed)
    small = cv2.resize(frame, (64, 64))
    n_unique = len(np.unique(small.reshape(-1, 3), axis=0))
    k = min(k, max(n_unique, 2))  # at least 2 clusters

    best_km, best_inertia = None, float("inf")
    for seed in [0, 42, 99]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=50)
        km.fit(lab)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_km = km

    labels = best_km.labels_.reshape(frame.shape[:2])

    # Convert LAB cluster centres back to RGB [0, 1]
    centres_lab = best_km.cluster_centers_.astype(np.uint8)
    colours_rgb = []
    for c in centres_lab:
        rgb = cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_LAB2RGB)
        colours_rgb.append(rgb.reshape(3).astype(float) / 255.0)

    return labels, np.array(colours_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Region Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_regions(palette_img: np.ndarray, min_area_fraction: float = 0.001):
    """
    Connected-component region extraction from a quantized palette image.

    Args:
        palette_img:       (H, W) int — cluster labels.
        min_area_fraction: minimum region area as fraction of total pixels.

    Returns:
        List of dicts with keys:
            color_idx, mask, area, centroid (normalised), area_frac
    """
    h, w = palette_img.shape
    min_area = int(min_area_fraction * h * w)
    regions = []

    for label in np.unique(palette_img):
        binary = (palette_img == label).astype(np.uint8)
        n_comp, comp_map, stats, centroids = cv2.connectedComponentsWithStats(binary)

        for comp_idx in range(1, n_comp):  # skip background label 0
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            mask = comp_map == comp_idx
            cx, cy = centroids[comp_idx]
            regions.append({
                "color_idx": int(label),
                "mask":      mask,
                "area":      area,
                "centroid":  (cx / w, cy / h),
                "area_frac": area / (h * w),
            })

    return regions


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bezier Boundary Fitting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fit_error(ctrl_pts: np.ndarray, original_pts: np.ndarray,
                       w: int, h: int) -> float:
    """Mean pixel distance between original contour points and fitted curve."""
    from generation.synthetic import oval_bezier  # reuse bezier math

    # Evaluate the fitted curve at many t values
    n_pts = ctrl_pts.shape[0]
    n_samples = 200
    t_vals = np.linspace(0, 1, n_samples)

    # Simple piecewise-linear interpolation of control points for error calc
    t_orig = np.linspace(0, 1, n_pts)
    curve = np.zeros((n_samples, 2))
    for dim in range(2):
        f = interpolate.interp1d(t_orig, ctrl_pts[:, dim], kind="linear")
        curve[:, dim] = f(t_vals)

    original_px = original_pts * np.array([w, h])
    curve_px = curve * np.array([w, h])

    min_dists = []
    for pt in original_px:
        dists = np.linalg.norm(curve_px - pt, axis=1)
        min_dists.append(dists.min())

    return float(np.mean(min_dists))


def fit_bezier_boundaries(regions: list, frame_shape: tuple,
                          epsilon: float = 2.0):
    """
    Fit cubic Bézier control points to each region's boundary contour.

    Process: contour → Douglas-Peucker simplification → cubic interpolation
    to exactly N_CTRL_PTS (12) points normalised to [0, 1].

    Args:
        regions:     list of region dicts from extract_regions().
        frame_shape: (H, W, ...) of the source frame.
        epsilon:     Douglas-Peucker approximation tolerance in pixels.

    Returns:
        List of shape dicts with keys:
            control_points (12, 2), color_idx, area_frac, centroid,
            fit_error, alpha, csg
    """
    h, w = frame_shape[:2]
    shapes = []

    for region in regions:
        mask_uint8 = region["mask"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 4:
            continue

        # Simplify
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        pts = simplified.squeeze().astype(float)
        if len(pts.shape) < 2 or pts.shape[0] < 4:
            continue

        # Normalise to [0, 1]
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0, 1)

        # Resample to exactly N_CTRL_PTS via cubic interpolation
        n_pts = len(pts)
        t_orig = np.linspace(0, 1, n_pts)
        t_new = np.linspace(0, 1, N_CTRL_PTS)

        ctrl_pts = np.zeros((N_CTRL_PTS, 2))
        for dim in range(2):
            f = interpolate.interp1d(
                t_orig, pts[:, dim],
                kind="cubic", bounds_error=False,
                fill_value=(pts[0, dim], pts[-1, dim]),
            )
            ctrl_pts[:, dim] = np.clip(f(t_new), 0, 1)

        fit_error = _compute_fit_error(ctrl_pts, pts, w, h)

        shapes.append({
            "control_points": ctrl_pts.astype(np.float16),
            "color_idx":      region["color_idx"],
            "area_frac":      region["area_frac"],
            "centroid":       region["centroid"],
            "fit_error":      fit_error,
            "alpha":          1.0,
            "csg":            False,
        })

    return shapes


# ─────────────────────────────────────────────────────────────────────────────
# 4. Slot Assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_slots_heuristic(shapes: list, canvas_size: tuple):
    """
    Assign shapes to semantic slot indices using rule-based heuristics.
    Uses centroid position and area to choose the slot block.

    Args:
        shapes:      list of shape dicts from fit_bezier_boundaries().
        canvas_size: (W, H) of the canvas (unused directly; centroids are normalised).

    Returns:
        Dict mapping slot_index → shape_data.
    """
    assigned = {}
    shapes_sorted = sorted(shapes, key=lambda s: s["area_frac"], reverse=True)

    counters = {
        "bg_static":  SLOT_BG_STATIC[0],
        "bg_dynamic": SLOT_BG_DYNAMIC[0],
        "body":       SLOT_BODY[0],
        "face":       SLOT_FACE[0],
        "mouth":      SLOT_MOUTH[0],
        "secondary":  SLOT_SECONDARY[0],
        "dynamic":    SLOT_DYNAMIC[0],
    }
    limits = {
        "bg_static":  SLOT_BG_STATIC[1],
        "bg_dynamic": SLOT_BG_DYNAMIC[1],
        "body":       SLOT_BODY[1],
        "face":       SLOT_FACE[1],
        "mouth":      SLOT_MOUTH[1],
        "secondary":  SLOT_SECONDARY[1],
        "dynamic":    SLOT_DYNAMIC[1],
    }

    for shape in shapes_sorted:
        cx, cy = shape["centroid"]
        area = shape["area_frac"]

        # Heuristic rules (centroid + area → block)
        if area > 0.15 and cy > 0.5:
            block = "bg_static"
        elif area > 0.08:
            block = "body"
        elif cy < 0.45 and area > 0.02:
            block = "face"
        elif cy < 0.55 and area < 0.02 and 0.3 < cx < 0.7:
            block = "mouth"
        elif area > 0.05:
            block = "secondary"
        else:
            block = "dynamic"

        # Overflow into dynamic if block is full
        if counters[block] > limits[block]:
            block = "dynamic"
            if counters["dynamic"] > limits["dynamic"]:
                continue  # no space — skip

        slot_idx = counters[block]
        counters[block] += 1
        assigned[slot_idx] = shape

    return assigned


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

def quality_filter(shapes, min_shapes: int, max_shapes: int,
                   max_fit_error: float):
    """
    Gate that accepts or rejects a frame's shape extraction.

    Args:
        shapes:        dict (from assign_slots_heuristic) or list.
        min_shapes:    minimum number of shapes required.
        max_shapes:    maximum number of shapes allowed.
        max_fit_error: maximum acceptable bezier fit error in pixels.

    Returns:
        (True, None) if passes, (False, reason_string) if rejected.
    """
    if isinstance(shapes, dict):
        n = len(shapes)
        shape_list = list(shapes.values())
    else:
        n = len(shapes)
        shape_list = shapes

    if n < min_shapes:
        return False, f"too few shapes: {n} < {min_shapes}"
    if n > max_shapes:
        return False, f"too many shapes: {n} > {max_shapes}"

    errors = [s.get("fit_error", 0) for s in shape_list]
    if errors and max(errors) > max_fit_error:
        return False, f"high fit error: {max(errors):.2f}px > {max_fit_error}px"

    return True, None
