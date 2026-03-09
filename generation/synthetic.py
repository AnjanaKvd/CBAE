import numpy as np


def oval_bezier(cx, cy, rx, ry, n_ctrl=12):
    """Ellipse approximation using 4 cubic Béziers."""
    kappa = 0.552284749831
    kx = rx * kappa
    ky = ry * kappa

    # Standard 12 points starting from right-midpoint, going clockwise
    # (assuming y goes down, cy-ky is "up" visually but lower value)
    pts = [
        [cx + rx, cy],  # 0: Right endpoint
        [cx + rx, cy - ky],  # 1: CP going up
        [cx + kx, cy - ry],  # 2: CP arriving at top
        [cx, cy - ry],  # 3: Top endpoint
        [cx - kx, cy - ry],  # 4: CP going left
        [cx - rx, cy - ky],  # 5: CP arriving at left
        [cx - rx, cy],  # 6: Left endpoint
        [cx - rx, cy + ky],  # 7: CP going down
        [cx - kx, cy + ry],  # 8: CP arriving at bottom
        [cx, cy + ry],  # 9: Bottom endpoint
        [cx + kx, cy + ry],  # 10: CP going right
        [cx + rx, cy + ky],  # 11: CP arriving at right
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def circle_bezier(cx, cy, r, n_ctrl=12):
    """Approximate circle with 12 Bézier control points."""
    return oval_bezier(cx, cy, r, r, n_ctrl)


def trapezoid_bezier(cx, cy, top_w, bot_w, h, n_ctrl=12):
    """Trapezoid for robe/torso."""
    TR = np.array([cx + top_w / 2, cy - h / 2])
    TL = np.array([cx - top_w / 2, cy - h / 2])
    BL = np.array([cx - bot_w / 2, cy + h / 2])
    BR = np.array([cx + bot_w / 2, cy + h / 2])

    # 4 straight segments interpolated over 12 control points
    pts = [
        TR,
        TR + (TL - TR) * 0.333333,
        TR + (TL - TR) * 0.666667,
        TL,
        TL + (BL - TL) * 0.333333,
        TL + (BL - TL) * 0.666667,
        BL,
        BL + (BR - BL) * 0.333333,
        BL + (BR - BL) * 0.666667,
        BR,
        BR + (TR - BR) * 0.333333,
        BR + (TR - BR) * 0.666667,
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def rounded_rect_bezier(cx, cy, w, h, corner_r, n_ctrl=12):
    """Rounded rectangle approximation."""
    kappa = 0.552284749831
    # Interpolation distance handles the squircle offset dynamically
    dx = w / 2 - corner_r * (1 - kappa)
    dy = h / 2 - corner_r * (1 - kappa)

    pts = [
        [cx + w / 2, cy],  # 0: RM
        [cx + w / 2, cy - dy],  # 1: CP up
        [cx + dx, cy - h / 2],  # 2: CP arriving at TM
        [cx, cy - h / 2],  # 3: TM
        [cx - dx, cy - h / 2],  # 4: CP left
        [cx - w / 2, cy - dy],  # 5: CP arriving at LM
        [cx - w / 2, cy],  # 6: LM
        [cx - w / 2, cy + dy],  # 7: CP down
        [cx - dx, cy + h / 2],  # 8: CP arriving at BM
        [cx, cy + h / 2],  # 9: BM
        [cx + dx, cy + h / 2],  # 10: CP right
        [cx + w / 2, cy + dy],  # 11: CP arriving at RM
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def _get_normal(p1, p2):
    """Helper to get 2D normal vector to a line segment."""
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-5:
        return np.array([0.0, 1.0])
    return np.array([-d[1], d[0]]) / length


def arm_bezier(shoulder_xy, elbow_xy, wrist_xy, width, n_ctrl=12):
    """Tapered arm shape from shoulder to wrist via elbow."""
    S = np.array(shoulder_xy)
    E = np.array(elbow_xy)
    W = np.array(wrist_xy)

    n_SE = _get_normal(S, E)
    n_EW = _get_normal(E, W)

    # Elbow bisector normal
    n_E = n_SE + n_EW
    n_E_norm = np.linalg.norm(n_E)
    if n_E_norm > 1e-5:
        n_E = n_E / n_E_norm
    else:
        n_E = n_SE

    w_S = width
    w_E = width * 0.8
    w_W = width * 0.5

    out_S = S + n_SE * (w_S / 2)
    inn_S = S - n_SE * (w_S / 2)

    out_E = E + n_E * (w_E / 2)
    inn_E = E - n_E * (w_E / 2)

    out_W = W + n_EW * (w_W / 2)
    inn_W = W - n_EW * (w_W / 2)

    # Push control points outwards so the curve (t=0.5) reaches the target elbow width exactly
    cp_out = out_E * 1.333333 - out_S * 0.166667 - out_W * 0.166667
    cp_inn = inn_E * 1.333333 - inn_S * 0.166667 - inn_W * 0.166667

    pts = [
        out_S,
        cp_out,
        cp_out,
        out_W,
        # Linear flat/round wrist cap
        out_W + (inn_W - out_W) * 0.333333,
        out_W + (inn_W - out_W) * 0.666667,
        inn_W,
        cp_inn,
        cp_inn,
        inn_S,
        # Linear flat/round shoulder cap
        inn_S + (out_S - inn_S) * 0.333333,
        inn_S + (out_S - inn_S) * 0.666667,
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)
