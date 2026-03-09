import numpy as np
import cairo
from typing import List

from core.crf_tensor import CRFTensor, CRFSequence


def bezier_to_polyline(control_points: np.ndarray, n_samples: int = 50) -> np.ndarray:
    """
    Evaluates 4 connected cubic Bézier curves forming a closed shape.
    control_points: (12, 2) array
    Returns: (n_samples, 2) polyline points array
    """
    # Evaluate t smoothly across 4 segments from 0 to 4
    t_global = np.linspace(0, 4, n_samples, endpoint=False)

    # Determine the segment index (0, 1, 2, 3) for each point
    seg_idx = np.clip(t_global.astype(int), 0, 3)

    # Local t within the segment (0 to 1)
    t = t_global - seg_idx
    t_inv = 1.0 - t

    # Cubic Bézier basis functions
    b0 = t_inv**3
    b1 = 3 * (t_inv**2) * t
    b2 = 3 * t_inv * (t**2)
    b3 = t**3

    # Extract control points (P0, P1, P2, P3) for each sample based on its segment
    P0 = control_points[seg_idx * 3]
    P1 = control_points[seg_idx * 3 + 1]
    P2 = control_points[seg_idx * 3 + 2]

    # Ensure P3 wraps back correctly to index 0 at the end of the 4th segment
    P3_idx = (seg_idx * 3 + 3) % 12
    P3 = control_points[P3_idx]

    # Calculate the coordinates
    points = b0[:, None] * P0 + b1[:, None] * P1 + b2[:, None] * P2 + b3[:, None] * P3
    return points


def render_shape(
    ctx: cairo.Context, control_points: np.ndarray, color: np.ndarray, alpha: float
):
    """Draws one filled shape on a Cairo context using an approximated polyline."""
    if alpha < 1e-3:
        return

    poly = bezier_to_polyline(control_points)

    ctx.move_to(poly[0, 0], poly[0, 1])
    for i in range(1, len(poly)):
        ctx.line_to(poly[i, 0], poly[i, 1])
    ctx.close_path()

    ctx.set_source_rgba(color[0], color[1], color[2], alpha)
    ctx.fill()


def render_csg_shape(ctx: cairo.Context, control_points: np.ndarray):
    """Draws subtractive shape using clip/even-odd rule."""
    poly = bezier_to_polyline(control_points)

    ctx.move_to(poly[0, 0], poly[0, 1])
    for i in range(1, len(poly)):
        ctx.line_to(poly[i, 0], poly[i, 1])
    ctx.close_path()

    # Configure the context for subtractive clipping
    orig_fill_rule = ctx.get_fill_rule()
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

    orig_op = ctx.get_operator()
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)

    # A fully opaque source removes the path region entirely
    ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    ctx.fill()

    # Restore prior states
    ctx.set_fill_rule(orig_fill_rule)
    ctx.set_operator(orig_op)


def rasterize(crf_tensor: CRFTensor, width: int = 512, height: int = 512) -> np.ndarray:
    """Composite crf limits and renders a 512x512 numpy image array."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    active_indices = crf_tensor.active_slots()

    # Sort active shapes by z-order ascending (render low z to high z)
    z_sorted_indices = sorted(active_indices.tolist(), key=lambda i: crf_tensor.z[i])

    # Map normalized control points `[0, 1]` to pixel dimensions
    scale = np.array([width, height])

    for idx in z_sorted_indices:
        P_scaled = crf_tensor.P[idx] * scale
        c = crf_tensor.c[idx]
        alpha = crf_tensor.alpha[idx]
        csg = crf_tensor.csg[idx]

        if csg:
            render_csg_shape(ctx, P_scaled)
        else:
            render_shape(ctx, P_scaled, c, alpha)

    # Serialize cairo memory out to raw numpy buffer
    buf = surface.get_data()
    image = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)

    rgb = np.empty((height, width, 3), dtype=np.uint8)

    # Extract native-endian cairo channels (Little Endian ARGB32 => BGRA structure: offset 2 is R, 1 is G, 0 is B)
    rgb[:, :, 0] = image[:, :, 2]
    rgb[:, :, 1] = image[:, :, 1]
    rgb[:, :, 2] = image[:, :, 0]

    # Explicit unpremultiply colors affected by alpha
    alpha_channel = image[:, :, 3].astype(np.float32) / 255.0
    mask = alpha_channel > 0

    for channel_i in range(3):
        channel = rgb[:, :, channel_i].astype(np.float32)
        channel[mask] = np.clip(channel[mask] / alpha_channel[mask], 0, 255)
        rgb[:, :, channel_i] = channel.astype(np.uint8)

    return rgb


def rasterize_sequence(
    crf_sequence: CRFSequence, width: int = 512, height: int = 512
) -> List[np.ndarray]:
    """Rasterizes an entire sequence sequentially."""
    return [rasterize(frame, width, height) for frame in crf_sequence.frames]
