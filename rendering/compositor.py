import numpy as np
from typing import List, Tuple


def alpha_over_composite(bg_rgba: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """
    Performs Porter-Duff alpha-over compositing of a foreground layer over a background layer.

    Args:
        bg_rgba: Background image (H, W, 4) in [0.0, 1.0] float32
        fg_rgba: Foreground image (H, W, 4) in [0.0, 1.0] float32

    Returns:
        Composited image (H, W, 4) in [0.0, 1.0] float32
    """
    # Extract alpha channels
    alpha_fg = fg_rgba[..., 3:4]
    alpha_bg = bg_rgba[..., 3:4]

    # Extract RGB channels
    color_fg = fg_rgba[..., :3]
    color_bg = bg_rgba[..., :3]

    # Compute output alpha: out_a = fg_a + bg_a * (1 - fg_a)
    alpha_out = alpha_fg + alpha_bg * (1.0 - alpha_fg)

    # To avoid division by zero where output alpha is zero
    safe_alpha_out = np.where(alpha_out > 0.0, alpha_out, 1.0)

    # Compute output color
    # C_out = (C_fg * alpha_fg + C_bg * alpha_bg * (1 - alpha_fg)) / alpha_out
    color_out = (
        color_fg * alpha_fg + color_bg * alpha_bg * (1.0 - alpha_fg)
    ) / safe_alpha_out

    # For pixels where alpha_out is exactly zero, color should be 0 (black/transparent)
    color_out = np.where(alpha_out > 0.0, color_out, 0.0)

    # Concatenate color and alpha channels back
    out_rgba = np.concatenate([color_out, alpha_out], axis=-1)

    return np.clip(out_rgba, 0.0, 1.0)


def csg_subtract(base_rgba: np.ndarray, mask_alpha: np.ndarray) -> np.ndarray:
    """
    Performs CSG subtractive mask. Pixels in the base layer where the mask is active
    (high alpha) are made transparent.

    Args:
        base_rgba: The layer to be carved out, (H, W, 4) in [0.0, 1.0] float32
        mask_alpha: The mask alpha channel, (H, W, 1) or (H, W) in [0.0, 1.0] float32

    Returns:
        The carved image (H, W, 4) in [0.0, 1.0] float32
    """
    if mask_alpha.ndim == 2:
        mask_alpha = mask_alpha[..., np.newaxis]

    # The mask opacity defines how much of the base to REMOVE.
    # Therefore, we keep (1.0 - mask_alpha) of the base alpha.
    new_alpha = base_rgba[..., 3:4] * (1.0 - mask_alpha)

    out_rgba = base_rgba.copy()
    out_rgba[..., 3:4] = np.clip(new_alpha, 0.0, 1.0)

    # Technically, we don't strictly need to zero out the RGB channels where alpha is 0,
    # because they will be ignored in subsequent alpha-over compositing, but it's cleaner.
    out_rgba[..., :3] = np.where(new_alpha > 0.0, out_rgba[..., :3], 0.0)

    return out_rgba


def layer_stack_composite(layers: List[Tuple[np.ndarray, bool]]) -> np.ndarray:
    """
    Compositor taking an ordered list of properties and returning final flattened RGB image.
    Expects layers from back to front order (low Z to high Z).

    Args:
        layers: List of tuples (shape_rgba, is_csg).
                shape_rgba is (H, W, 4) array in uint8 [0, 255] or float [0.0, 1.0].
                is_csg indicates if the layer subtracts from elements below it.

    Returns:
        Final RGB composite (H, W, 3) in uint8 [0, 255]
    """
    if not layers:
        raise ValueError("Cannot composite an empty layer stack.")

    shape_0, _ = layers[0]
    H, W = shape_0.shape[:2]

    # Initialize an empty, fully transparent canvas to accumulate on
    canvas_rgba = np.zeros((H, W, 4), dtype=np.float32)

    for shape_data, is_csg in layers:
        # Normalize incoming layers to [0.0, 1.0] float32
        if shape_data.dtype == np.uint8:
            layer_float = shape_data.astype(np.float32) / 255.0
        else:
            layer_float = shape_data.astype(np.float32)

        if is_csg:
            # Subtractive mask removes alpha from the aggregated canvas underneath
            mask_alpha = layer_float[..., 3:4]
            canvas_rgba = csg_subtract(canvas_rgba, mask_alpha)
        else:
            # Normal filled polygon, alpha_over on top of the current canvas
            canvas_rgba = alpha_over_composite(canvas_rgba, layer_float)

    # Finally drop the alpha channel (or implicitly composite over a black background)
    # The spec just says return RGB image. Since canvas_rgba might still have alpha < 1,
    # we simulate an implicit black background.
    final_rgb = canvas_rgba[..., :3] * canvas_rgba[..., 3:4]

    # Convert and cast to 8-bit output
    final_rgb_uint8 = np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)
    return final_rgb_uint8
