import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math

# Try to import diffvg, but have a robust fallback ready
try:
    import pydiffvg

    HAS_DIFFVG = True
except ImportError:
    HAS_DIFFVG = False


def bezier_to_polyline_torch(
    control_points: torch.Tensor, n_samples: int = 50
) -> torch.Tensor:
    """
    Evaluates 4 connected cubic Bézier curves forming a closed shape in PyTorch.
    Args:
        control_points: (n_slots, 12, 2) tensor
    Returns:
        (n_slots, n_samples, 2) polyline points tensor
    """
    device = control_points.device
    n_slots = control_points.shape[0]

    t_global = torch.linspace(0, 4, n_samples, device=device)[
        :-1
    ]  # Remove endpoint to match numpy linspace(endpoint=False)
    # n_samples might be off by 1 because of removing the endpoint, let's just use exact points
    t_global = torch.linspace(0, 4 - (4 / n_samples), n_samples, device=device)

    seg_idx = torch.clamp(t_global.floor().long(), 0, 3)
    t = t_global - seg_idx
    t_inv = 1.0 - t

    b0 = t_inv**3
    b1 = 3 * (t_inv**2) * t
    b2 = 3 * t_inv * (t**2)
    b3 = t**3

    # Shape: (n_samples, 3)
    basis = torch.stack([b0, b1, b2, b3], dim=-1)

    points = torch.zeros(
        (n_slots, n_samples, 2), device=device, dtype=control_points.dtype
    )

    for i in range(n_samples):
        s_idx = seg_idx[i]
        P0 = control_points[:, s_idx * 3]
        P1 = control_points[:, s_idx * 3 + 1]
        P2 = control_points[:, s_idx * 3 + 2]
        P3_idx = (s_idx * 3 + 3) % 12
        P3 = control_points[:, P3_idx]

        # basis[i] has 4 weights
        points[:, i, :] = (
            basis[i, 0] * P0 + basis[i, 1] * P1 + basis[i, 2] * P2 + basis[i, 3] * P3
        )

    return points


def point_to_line_segment_distance(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Computes squared combined distance from point p to line segment a-b.
    p: (H, W, 2)
    a: (2,)
    b: (2,)
    Returns: (H, W) squared distances
    """
    ab = b - a
    ap = p - a

    # Project ap onto ab, clamped between 0 and 1
    ab_len_sq = torch.sum(ab * ab) + 1e-8
    t = torch.sum(ap * ab, dim=-1) / ab_len_sq
    t = torch.clamp(t, 0.0, 1.0)

    # Find closest point on segment
    closest = a + t.unsqueeze(-1) * ab

    # Return distance squared
    diff = p - closest
    return torch.sum(diff * diff, dim=-1)


def signed_distance_field(
    points_hw: torch.Tensor, polyline: torch.Tensor
) -> torch.Tensor:
    """
    Computes an approximate signed distance field for a closed polygon.
    points_hw: (H, W, 2) grid of points
    polyline: (N, 2) polygon vertices

    Returns: (H, W) signed distances (negative inside, positive outside)
    """
    H, W = points_hw.shape[:2]
    N = polyline.shape[0]

    min_dist_sq = torch.full((H, W), float("inf"), device=points_hw.device)
    winding_number = torch.zeros((H, W), dtype=torch.int32, device=points_hw.device)

    for i in range(N):
        a = polyline[i]
        b = polyline[(i + 1) % N]

        # Distance to segment
        dist_sq = point_to_line_segment_distance(points_hw, a, b)
        min_dist_sq = torch.min(min_dist_sq, dist_sq)

        # Winding number calculation (ray casting in +x direction)
        # Check if point's y is between a.y and b.y
        y_cond1 = a[1] <= points_hw[..., 1]
        y_cond2 = points_hw[..., 1] < b[1]

        # Edge crossing upwards
        up_cross = y_cond1 & y_cond2

        # Edge crossing downwards
        down_cross = (~y_cond1) & (~y_cond2)

        # Check x intersection
        # x_intersect = a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x)
        # Simplified: (b.x - a.x) * (point.y - a.y) - (point.x - a.x) * (b.y - a.y) > 0 for point to be on the left

        is_left = (
            (b[0] - a[0]) * (points_hw[..., 1] - a[1])
            - (points_hw[..., 0] - a[0]) * (b[1] - a[1])
        ) > 0

        winding_number[up_cross & is_left] += 1
        winding_number[down_cross & (~is_left)] -= 1

    # Distance is sqrt of squared distance
    dist = torch.sqrt(min_dist_sq + 1e-8)

    # Negative if inside (winding number != 0), positive if outside
    sign = torch.where(winding_number != 0, -1.0, 1.0)

    return dist * sign


def _soft_rasterize_single_shape(
    polyline: torch.Tensor,
    color: torch.Tensor,
    alpha: torch.Tensor,
    grid: torch.Tensor,
    softness: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterizes a single shape down to an RGBA mask.
    polyline: (N, 2) vertices in [0, 1] normalized space
    color: (3,) RGB
    alpha: (1,) scalar opacity
    grid: (H, W, 2) pixel coordinates in [0, 1] normalized space
    """
    sdf = signed_distance_field(grid, polyline)

    # Coverage = sigmoid(-SDF / softness)
    # When SDF is deeply negative (inside), -SDF is deeply positive, sigmoid goes to 1
    # When SDF is deeply positive (outside), -SDF is deeply negative, sigmoid goes to 0
    coverage = torch.sigmoid(-sdf / softness)

    # Multiply by explicit shape alpha
    final_alpha = coverage * alpha

    # Shape color map (H, W, 3)
    color_map = color.view(1, 1, 3).expand(grid.shape[0], grid.shape[1], 3)

    return color_map, final_alpha.unsqueeze(-1)


def soft_rasterize_approximation(
    P: torch.Tensor,
    c: torch.Tensor,
    shape_alphas: torch.Tensor,
    shape_zs: torch.Tensor,
    shape_csgs: torch.Tensor,
    width: int = 512,
    height: int = 512,
    softness: float = 0.01,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch fallback differentiable rasterizer.
    P: (N, 12, 2) control points
    c: (N, 3) colors
    shape_alphas: (N,) alpha values
    shape_zs: (N,) z-orders
    shape_csgs: (N,) booleans indicating subtractive shapes
    """
    device = P.device
    N = P.shape[0]

    if active_mask is None:
        active_mask = torch.ones(N, dtype=torch.bool, device=device)

    polylines = bezier_to_polyline_torch(P, n_samples=30)  # Reduce samples for perf

    # Create normalized coordinate grid
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    # Initialize background as transparent black
    canvas_rgb = torch.zeros((height, width, 3), device=device)
    canvas_alpha = torch.zeros((height, width, 1), device=device)

    # Sort by z-order, but keep original indices to fetch data
    valid_indices = torch.nonzero(active_mask).squeeze(-1)
    if valid_indices.numel() == 0:
        return canvas_rgb

    # Get active z values and sort them
    active_zs = shape_zs[valid_indices]
    sorted_idx = torch.argsort(active_zs)
    render_order = valid_indices[sorted_idx]

    for idx in render_order:
        layer_color, layer_alpha = _soft_rasterize_single_shape(
            polylines[idx], c[idx], shape_alphas[idx], grid, softness
        )

        if shape_csgs[idx]:
            # CSG Subtraction: remove alpha proportional to layer_alpha
            canvas_alpha = canvas_alpha * (1.0 - layer_alpha)
            canvas_rgb = canvas_rgb * (canvas_alpha > 0.0).float()
        else:
            # Alpha over compositing
            out_alpha = layer_alpha + canvas_alpha * (1.0 - layer_alpha)

            # Avoid division by zero
            safe_out_alpha = torch.where(
                out_alpha > 0.0, out_alpha, torch.ones_like(out_alpha)
            )

            out_rgb = (
                layer_color * layer_alpha
                + canvas_rgb * canvas_alpha * (1.0 - layer_alpha)
            ) / safe_out_alpha
            out_rgb = torch.where(out_alpha > 0.0, out_rgb, torch.zeros_like(out_rgb))

            canvas_rgb = out_rgb
            canvas_alpha = out_alpha

    # Implicit black background composite at the end
    final_rgb = canvas_rgb * canvas_alpha
    return torch.clamp(final_rgb, 0.0, 1.0)


def diffvg_rasterize(
    P: torch.Tensor,
    c: torch.Tensor,
    shape_alphas: torch.Tensor,
    shape_zs: torch.Tensor,
    shape_csgs: torch.Tensor,
    width: int = 512,
    height: int = 512,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Uses DiffVG for exact gradient flow."""
    if not HAS_DIFFVG:
        raise RuntimeError(
            "pydiffvg is not installed. Use soft_rasterize_approximation instead."
        )

    device = P.device
    N = P.shape[0]

    if active_mask is None:
        active_mask = torch.ones(N, dtype=torch.bool, device=device)

    # Convert points from [0, 1] bounds to raw pixel bounds for DiffVG
    P_pixels = P * torch.tensor([width, height], device=device).view(1, 1, 2)

    shapes = []
    shape_groups = []

    valid_indices = torch.nonzero(active_mask).squeeze(-1)
    if valid_indices.numel() == 0:
        return torch.zeros((height, width, 3), device=device)

    active_zs = shape_zs[valid_indices]
    sorted_idx = torch.argsort(active_zs)
    render_order = valid_indices[sorted_idx]

    for idx in render_order:
        pts = P_pixels[idx]

        # Create DiffVG path (12 control points = 4 cubic segments)
        num_control_points = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=pts,
            is_closed=True,
            stroke_width=torch.tensor(0.0),
        )
        shapes.append(path)

        if shape_csgs[idx]:
            # DiffVG does not natively support CSG destructively in a single pass easily
            # without complex compositing ops.
            raise NotImplementedError(
                "CSG is only implemented in soft_ras fallback currently."
            )
        else:
            color_rgba = torch.cat([c[idx], shape_alphas[idx : idx + 1]])
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]), fill_color=color_rgba
            )
            shape_groups.append(group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply

    # We provide a dummy background color image
    bg = torch.zeros((height, width, 4), device=device)
    img = render(width, height, 2, 2, 0, bg, *scene_args)

    return img[..., :3]


# Main interface
class DiffRasterizer(nn.Module):
    def __init__(self, use_diffvg: bool = False, fallback_softness: float = 0.01):
        super().__init__()
        self.use_diffvg = use_diffvg and HAS_DIFFVG
        self.softness = fallback_softness

    def forward(
        self,
        P: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        alive: torch.Tensor,
        z: torch.Tensor,
        csg: torch.Tensor,
        width: int = 512,
        height: int = 512,
        alive_threshold: float = -2.197,  # sigmoid(-2.197) ≈ 0.1
    ) -> torch.Tensor:
        """
        P: (N, 12, 2)
        c: (N, 3)
        alpha: (N,)
        alive: (N,) logits
        z: (N,)
        csg: (N,)
        """
        # Active mask calculation natively retaining gradients where applicable
        active_mask = torch.sigmoid(alive) > 0.1

        if self.use_diffvg:
            return diffvg_rasterize(
                P, c, alpha, z, csg, width=width, height=height, active_mask=active_mask
            )
        else:
            return soft_rasterize_approximation(
                P,
                c,
                alpha,
                z,
                csg,
                width=width,
                height=height,
                softness=self.softness,
                active_mask=active_mask,
            )
