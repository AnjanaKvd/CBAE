import pytest
import numpy as np
import torch
from core.crf_tensor import CRFTensor
from rendering.rasterizer import rasterize
from rendering.diff_rasterizer import DiffRasterizer, HAS_DIFFVG


def _make_square_control_points(cx, cy, size):
    # Rough approximation of a square using bezier points
    pts = np.zeros((12, 2))
    r = size / 2.0

    # Top edge
    pts[0] = [cx - r, cy - r]
    pts[1] = [cx, cy - r]
    pts[2] = [cx + r, cy - r]

    # Right edge
    pts[3] = [cx + r, cy - r]
    pts[4] = [cx + r, cy]
    pts[5] = [cx + r, cy + r]

    # Bottom edge
    pts[6] = [cx + r, cy + r]
    pts[7] = [cx, cy + r]
    pts[8] = [cx - r, cy + r]

    # Left edge
    pts[9] = [cx - r, cy + r]
    pts[10] = [cx - r, cy]
    pts[11] = [cx - r, cy - r]
    return pts


def test_rasterizer_single_shape():
    crf = CRFTensor()
    crf.P[0] = _make_square_control_points(0.5, 0.5, 0.5)
    crf.c[0] = [1.0, 0.0, 0.0]
    crf.alpha[0] = 1.0
    crf.activate(0)

    img = rasterize(crf, width=64, height=64)
    assert img.shape == (64, 64, 3)

    # Check that center pixel is roughly red
    center_color = img[32, 32]
    assert center_color[0] > 200  # Red channel high
    assert center_color[1] < 50
    assert center_color[2] < 50


def test_rasterizer_z_order():
    crf = CRFTensor()

    # Large red square at z=0
    crf.P[0] = _make_square_control_points(0.5, 0.5, 0.8)
    crf.c[0] = [1.0, 0.0, 0.0]
    crf.alpha[0] = 1.0
    crf.z[0] = 0.0
    crf.activate(0)

    # Small green square at z=1, should be on top
    crf.P[1] = _make_square_control_points(0.5, 0.5, 0.2)
    crf.c[1] = [0.0, 1.0, 0.0]
    crf.alpha[1] = 1.0
    crf.z[1] = 1.0
    crf.activate(1)

    img = rasterize(crf, width=64, height=64)
    center_color = img[32, 32]
    # Should be green
    assert center_color[1] > 200
    assert center_color[0] < 50


def test_csg_creates_hole():
    crf = CRFTensor()

    # Large red square
    crf.P[0] = _make_square_control_points(0.5, 0.5, 0.8)
    crf.c[0] = [1.0, 0.0, 0.0]
    crf.alpha[0] = 1.0
    crf.z[0] = 0.0
    crf.activate(0)

    # Small subtractive (csg) square in the middle
    crf.P[1] = _make_square_control_points(0.5, 0.5, 0.4)
    crf.csg[1] = True
    crf.z[1] = 1.0
    crf.activate(1)

    img = rasterize(crf, width=64, height=64)
    center_color = img[32, 32]
    # Should be background color (black or transparent, unpacked as (0,0,0) due to 0-alpha)
    assert center_color[0] == 0
    assert center_color[1] == 0
    assert center_color[2] == 0

    # Edge should still be red
    edge_color = img[10, 32]
    assert edge_color[0] > 200


def test_diff_rasterizer_gradients():
    dr = DiffRasterizer(use_diffvg=False)  # Test fallback specifically

    P = torch.tensor(
        _make_square_control_points(0.5, 0.5, 0.5), dtype=torch.float32
    ).unsqueeze(0)
    P.requires_grad = True

    c = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    alpha = torch.tensor([1.0], dtype=torch.float32)
    alive = torch.tensor([5.0], dtype=torch.float32)  # active
    z = torch.tensor([0.0], dtype=torch.float32)
    csg = torch.tensor([False], dtype=torch.bool)

    img = dr(P, c, alpha, alive, z, csg, width=16, height=16)

    target = torch.ones_like(img)
    loss = torch.nn.functional.mse_loss(img, target)
    loss.backward()

    assert P.grad is not None
    # Check that gradients are finite
    assert torch.isfinite(P.grad).all()
    # Check that gradients are not all zero
    assert torch.any(P.grad != 0.0)
