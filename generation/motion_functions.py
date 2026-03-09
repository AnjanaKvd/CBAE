import numpy as np
from typing import List, Callable
from core.crf_tensor import CRFTensor


def breathing_motion(crf: CRFTensor, t: float) -> CRFTensor:
    """
    Applies a vertical sinusoidal breathing scale to the main body shape (slot 26).
    0.25 Hz, 0.5% amplitude.
    """
    out = crf.clone()
    if out.alive[26] < 0:  # Not active
        return out

    scale_y = 1.0 + 0.005 * np.sin(2 * np.pi * 0.25 * t)

    # Calculate centroid to scale relative to it
    pts = out.P[26]
    cy = np.mean(pts[:, 1])

    # Apply vertical scale only
    pts[:, 1] = cy + (pts[:, 1] - cy) * scale_y

    # Clamp
    out.P[26] = np.clip(pts, 0.0, 1.0).astype(np.float16)
    return out


def eye_blink(
    crf: CRFTensor, t: float, blink_times: List[float] = [1.5, 4.0, 6.5]
) -> CRFTensor:
    """
    Applies a Gaussian dip to scale down the height of the eyes (slots 47, 48) at specific times.
    """
    out = crf.clone()

    scale = 1.0
    for t_b in blink_times:
        if abs(t - t_b) < 0.083:  # roughly 2 frames at 24fps
            # Gaussian dip: at peak (t=t_b), scale goes to 1 - exp(0) = 0
            # To ensure it doesn't invert and just closes to ~5%, we max it with 0.05
            dip = 1.0 - np.exp(-((t - t_b) ** 2) / 0.002)
            scale = min(scale, max(0.05, dip))

    if scale >= 0.999:
        return out

    for slot in [47, 48]:
        if out.alive[slot] > 0:
            pts = out.P[slot]
            cy = np.mean(pts[:, 1])
            pts[:, 1] = cy + (pts[:, 1] - cy) * scale
            out.P[slot] = np.clip(pts, 0.0, 1.0).astype(np.float16)

    return out


def gentle_sway(crf: CRFTensor, t: float) -> CRFTensor:
    """
    Applies a horizontal translation to body + arms (slots 26, 27, 28).
    0.1 Hz, 1% canvas width amplitude.
    """
    out = crf.clone()

    dx = 0.01 * np.sin(2 * np.pi * 0.1 * t)

    for slot in [26, 27, 28]:
        if out.alive[slot] > 0:
            out.P[slot, :, 0] += dx
            out.P[slot] = np.clip(out.P[slot], 0.0, 1.0).astype(np.float16)

    return out


def compose_motions(crf: CRFTensor, t: float, motion_fns: List[Callable]) -> CRFTensor:
    """
    Sequentially applies a list of motion functions.
    """
    current_crf = crf
    for fn in motion_fns:
        current_crf = fn(current_crf, t)
    return current_crf


def compute_velocity_gt(
    motion_fn: Callable, crf_base: CRFTensor, t: float, dt: float = 0.001
) -> np.ndarray:
    """
    Computes numerical derivative (velocity) of control points using central difference.
    dP/dt ≈ (P(t+dt) - P(t-dt)) / (2*dt)

    Returns: (n_slots, 12, 2) shaped float32 array
    """
    crf_forward = motion_fn(crf_base, t + dt)
    crf_backward = motion_fn(crf_base, t - dt)

    # Calculate finite difference
    dp_dt = (crf_forward.P.astype(np.float32) - crf_backward.P.astype(np.float32)) / (
        2.0 * dt
    )

    return dp_dt
