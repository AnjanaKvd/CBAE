import torch
import torch.nn as nn
from typing import Tuple, Optional

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
    Evaluates 4 connected cubic Bézier curves forming a closed shape.
    Fully vectorized — no Python loops.

    Args:
        control_points: (n_slots, 12, 2) tensor
    Returns:
        (n_slots, n_samples, 2) polyline points tensor
    """
    device = control_points.device
    n_slots = control_points.shape[0]

    # Global parameter values in [0, 4) for n_samples points
    t_global = torch.linspace(0, 4 - (4 / n_samples), n_samples, device=device)

    seg_idx = torch.clamp(t_global.floor().long(), 0, 3)  # (n_samples,)
    t = t_global - seg_idx.float()  # local parameter in [0, 1)

    # Bernstein basis weights: (n_samples, 4)
    t_inv = 1.0 - t
    basis = torch.stack([
        t_inv ** 3,
        3 * (t_inv ** 2) * t,
        3 * t_inv * (t ** 2),
        t ** 3
    ], dim=-1)  # (n_samples, 4)

    # Build control point indices: P0=3s, P1=3s+1, P2=3s+2, P3=(3s+3)%12
    cp_indices = torch.stack([
        seg_idx * 3,
        seg_idx * 3 + 1,
        seg_idx * 3 + 2,
        (seg_idx * 3 + 3) % 12,
    ], dim=-1)  # (n_samples, 4)

    # Gather control points for all samples at once
    # Flatten (n_samples, 4) -> (n_samples*4,) indices, expand for n_slots
    flat_idx = cp_indices.reshape(-1)  # (n_samples*4,)
    flat_idx = flat_idx.unsqueeze(0).unsqueeze(-1).expand(n_slots, -1, 2)  # (n_slots, n_samples*4, 2)
    gathered = torch.gather(control_points, 1, flat_idx)  # (n_slots, n_samples*4, 2)
    gathered = gathered.reshape(n_slots, n_samples, 4, 2)  # (n_slots, n_samples, 4, 2)

    # Weighted sum: basis (n_samples, 4) with gathered (n_slots, n_samples, 4, 2)
    # einsum: sum over k (the 4 basis functions)
    points = torch.einsum('sk,nsk d->ns d', basis, gathered)
    # Simplify: (n_slots, n_samples, 2)
    points = (basis.unsqueeze(0).unsqueeze(-1) * gathered).sum(dim=2)

    return points


def signed_distance_field_batched(
    grid: torch.Tensor, polylines: torch.Tensor
) -> torch.Tensor:
    """
    Batched signed distance field for multiple polygons simultaneously.
    Fully vectorized — no Python loops over segments.

    Args:
        grid: (H, W, 2) grid of points in [0, 1]
        polylines: (M, N, 2) M polygons, each with N vertices

    Returns:
        (M, H, W) signed distances (negative inside, positive outside)
    """
    H, W = grid.shape[:2]
    M, N, _ = polylines.shape

    # Build segment endpoints: a[i] -> b[i] = a[(i+1)%N]
    a = polylines                                          # (M, N, 2)
    b = torch.roll(polylines, -1, dims=1)                  # (M, N, 2)

    # Expand grid for broadcasting: (1, 1, H, W, 2)
    p = grid.unsqueeze(0).unsqueeze(0)                     # (1, 1, H, W, 2)

    # Expand segments: (M, N, 1, 1, 2)
    a_exp = a.unsqueeze(2).unsqueeze(3)                    # (M, N, 1, 1, 2)
    b_exp = b.unsqueeze(2).unsqueeze(3)                    # (M, N, 1, 1, 2)

    # --- Distance to segments (vectorized) ---
    ab = b_exp - a_exp                                     # (M, N, 1, 1, 2)
    ap = p - a_exp                                         # (M, N, H, W, 2)

    ab_len_sq = (ab * ab).sum(dim=-1) + 1e-8               # (M, N, 1, 1)
    t_proj = (ap * ab).sum(dim=-1) / ab_len_sq             # (M, N, H, W)
    t_proj = torch.clamp(t_proj, 0.0, 1.0)

    closest = a_exp + t_proj.unsqueeze(-1) * ab            # (M, N, H, W, 2)
    diff = p - closest                                     # (M, N, H, W, 2)
    dist_sq = (diff * diff).sum(dim=-1)                    # (M, N, H, W)

    # Min distance across all segments
    min_dist_sq, _ = dist_sq.min(dim=1)                    # (M, H, W)

    # --- Winding number (vectorized) ---
    ay = a_exp[..., 1]                                     # (M, N, 1, 1)
    by = b_exp[..., 1]                                     # (M, N, 1, 1)
    ax = a_exp[..., 0]                                     # (M, N, 1, 1)
    bx = b_exp[..., 0]                                     # (M, N, 1, 1)
    py = p[..., 1]                                         # (1, 1, H, W)
    px = p[..., 0]                                         # (1, 1, H, W)

    y_cond1 = ay <= py                                     # (M, N, H, W)
    y_cond2 = py < by                                      # (M, N, H, W)

    up_cross = y_cond1 & y_cond2
    down_cross = (~y_cond1) & (~y_cond2)

    is_left = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) > 0

    # +1 for up-crossing where left, -1 for down-crossing where not-left
    winding_contrib = torch.zeros_like(dist_sq, dtype=torch.float32)
    winding_contrib[up_cross & is_left] = 1.0
    winding_contrib[down_cross & (~is_left)] = -1.0

    winding_number = winding_contrib.sum(dim=1)            # (M, H, W)

    dist = torch.sqrt(min_dist_sq + 1e-8)
    sign = torch.where(winding_number != 0, -1.0, 1.0)

    return dist * sign


def signed_distance_field(
    points_hw: torch.Tensor, polyline: torch.Tensor
) -> torch.Tensor:
    """
    Single-polygon SDF (wraps batched version for backward compat).
    points_hw: (H, W, 2), polyline: (N, 2)
    Returns: (H, W) signed distances
    """
    return signed_distance_field_batched(points_hw, polyline.unsqueeze(0)).squeeze(0)


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

    coverage = torch.sigmoid(-sdf / softness)
    final_alpha = coverage * alpha

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
    Optimized differentiable soft rasterizer.
    Batches SDF computation across all active shapes, then composites sequentially.

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

    # Sort and filter active shapes
    valid_indices = torch.nonzero(active_mask).squeeze(-1)
    if valid_indices.numel() == 0:
        return torch.zeros((height, width, 3), device=device)

    active_zs = shape_zs[valid_indices]
    sorted_idx = torch.argsort(active_zs)
    render_order = valid_indices[sorted_idx]
    M = render_order.numel()

    # --- BATCH: Compute all polylines at once ---
    active_P = P[render_order]                              # (M, 12, 2)
    polylines = bezier_to_polyline_torch(active_P, n_samples=30)  # (M, 30, 2)

    # --- BATCH: Compute all SDFs at once ---
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)            # (H, W, 2)

    all_sdf = signed_distance_field_batched(grid, polylines)  # (M, H, W)

    # --- BATCH: Compute all coverages at once ---
    all_coverage = torch.sigmoid(-all_sdf / softness)        # (M, H, W)

    active_alphas = shape_alphas[render_order]               # (M,)
    active_colors = c[render_order]                          # (M, 3)
    active_csgs = shape_csgs[render_order]                   # (M,)

    # Per-shape alpha: coverage * shape_alpha -> (M, H, W, 1)
    all_layer_alpha = (all_coverage * active_alphas.view(M, 1, 1)).unsqueeze(-1)

    # Per-shape color maps: (M, 1, 1, 3) -> broadcast to (M, H, W, 3)
    all_layer_color = active_colors.view(M, 1, 1, 3).expand(M, height, width, 3)

    # --- Sequential compositing (must be ordered for correct alpha-over) ---
    canvas_rgb = torch.zeros((height, width, 3), device=device)
    canvas_alpha = torch.zeros((height, width, 1), device=device)

    for i in range(M):
        layer_alpha = all_layer_alpha[i]                     # (H, W, 1)
        layer_color = all_layer_color[i]                     # (H, W, 3)

        if active_csgs[i]:
            canvas_alpha = canvas_alpha * (1.0 - layer_alpha)
            canvas_rgb = canvas_rgb * (canvas_alpha > 0.0).float()
        else:
            out_alpha = layer_alpha + canvas_alpha * (1.0 - layer_alpha)
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

        num_control_points = torch.tensor([2, 2, 2, 2], dtype=torch.int32)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=pts,
            is_closed=True,
            stroke_width=torch.tensor(0.0),
        )
        shapes.append(path)

        if shape_csgs[idx]:
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
