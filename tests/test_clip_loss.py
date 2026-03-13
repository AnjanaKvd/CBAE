"""
tests/test_clip_loss.py — Tests for the real CLIP contrastive loss (D1 debt resolution).

Three tests:
 - test_clip_loss_backward_compatible: no clip_model → clip=0.0
 - test_clip_loss_returns_scalar: with real CLIP model → scalar > 0
 - test_clip_loss_gradient_flows: backward pass through CLIP loss works
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def clip_model_and_emb(device):
    """Load open_clip ViT-B/32 for use across tests (loaded once)."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(["a blue robed character"])
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = F.normalize(text_emb.float(), dim=-1)  # (1, 512)

    return model, text_emb


def _make_video_tensor(batch=1, T=8, H=32, W=32, device="cpu", requires_grad=False):
    """Small synthetic video tensor in [0, 1]."""
    v = torch.rand(batch, T, H, W, 3, device=device)
    if requires_grad:
        v = v.requires_grad_(True)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Backward compatibility — no clip_model → returns 0.0
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_loss_backward_compatible(device):
    """CBAELossWrapper() without clip_model must return clip=0.0 exactly."""
    from training.loss import CBAELossWrapper

    loss_fn = CBAELossWrapper()  # no clip_model passed
    video = _make_video_tensor(device=device)
    prompt_emb = torch.randn(1, 512, device=device)

    clip_loss = loss_fn.compute_clip_loss(video, prompt_emb)

    assert clip_loss.item() == 0.0, (
        f"Expected 0.0 without clip_model, got {clip_loss.item()}"
    )
    assert clip_loss.ndim == 0, "clip loss must be a scalar tensor"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Real CLIP model → returns a plausible scalar > 0
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_loss_returns_scalar(device, clip_model_and_emb):
    """With a real CLIP model, compute_clip_loss must return a finite scalar > 0."""
    from training.loss import CBAELossWrapper

    clip_model, text_emb = clip_model_and_emb

    loss_fn = CBAELossWrapper(clip_model=clip_model)
    video = _make_video_tensor(batch=1, T=8, H=32, W=32, device=device)

    with torch.no_grad():
        clip_loss = loss_fn.compute_clip_loss(video, text_emb)

    assert clip_loss.ndim == 0, "clip loss must be a scalar"
    assert torch.isfinite(clip_loss), f"clip loss is not finite: {clip_loss.item()}"
    # 1 - cos_sim in [-1,1] → loss in [0, 2]; result should be > 0 for random frames
    assert clip_loss.item() > 0.0, (
        f"Expected loss > 0 for random frames vs text, got {clip_loss.item()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Gradient flows through CLIP loss
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_loss_gradient_flows(device, clip_model_and_emb):
    """Backward pass through CLIP loss must not raise and must produce non-None grads."""
    from training.loss import CBAELossWrapper

    clip_model, text_emb = clip_model_and_emb

    loss_fn = CBAELossWrapper(clip_model=clip_model)

    # video requires grad so we can check gradient flows back through the resize/encode path
    video = _make_video_tensor(batch=1, T=8, H=32, W=32, device=device, requires_grad=True)

    clip_loss = loss_fn.compute_clip_loss(video, text_emb)

    assert clip_loss.ndim == 0, "clip loss must be scalar before backward"
    assert torch.isfinite(clip_loss), f"clip loss is not finite before backward: {clip_loss.item()}"

    # Backward must not raise
    clip_loss.backward()

    assert video.grad is not None, "No gradient flowed back to video tensor"
    assert torch.isfinite(video.grad).all(), "Gradients contain NaN or Inf"
    assert video.grad.abs().sum() > 0, "Gradients are all zero — no signal"
