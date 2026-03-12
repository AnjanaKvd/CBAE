import pytest
import torch
from models.cbae_model import CBAE_EndToEnd
from training.loss import CBAELossWrapper


@pytest.fixture(scope="module")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="module")
def model_and_loss(device):
    model = CBAE_EndToEnd(render_width=32, render_height=32).to(device)
    loss_fn = CBAELossWrapper().to(device)
    return model, loss_fn


@pytest.fixture(scope="module")
def forward_outputs(device, model_and_loss):
    model, loss_fn = model_and_loss
    batch_size = 2
    prompt = ["A gentle blue character breathing", "A red block jumping"]
    audio = torch.randn(batch_size, 16000 * 8).to(device)  # 8 seconds of audio
    gt_video_mock = torch.rand(batch_size, 192, 32, 32, 3).to(device)

    with torch.no_grad():
        video_tensor, topology = model(prompt, audio)

    return video_tensor, topology, gt_video_mock, loss_fn


def test_video_tensor_shape(forward_outputs):
    """Model output video tensor should be (batch, T, H, W, 3)."""
    video_tensor, topology, gt_video_mock, _ = forward_outputs
    assert video_tensor.shape == gt_video_mock.shape, (
        f"Expected video shape {gt_video_mock.shape}, got {video_tensor.shape}"
    )


def test_loss_forward_returns_scalar_and_metrics(forward_outputs):
    """CBAELossWrapper.forward should return a scalar loss and a metrics dict."""
    video_tensor, topology, gt_video_mock, loss_fn = forward_outputs
    loss_total, metrics = loss_fn((video_tensor, topology), gt_video_mock)

    assert loss_total.ndim == 0, "Total loss must be a scalar tensor"
    assert isinstance(metrics, dict), "Metrics must be a dict"
    for key in ('bcs', 'crs', 'temp', 'render', 'clip'):
        assert key in metrics, f"Missing metric key: '{key}'"


def test_loss_values_are_finite(forward_outputs):
    """All loss components should be finite (no NaN / Inf)."""
    video_tensor, topology, gt_video_mock, loss_fn = forward_outputs
    loss_total, metrics = loss_fn((video_tensor, topology), gt_video_mock)

    assert torch.isfinite(loss_total), "Total loss is not finite"
    for k, v in metrics.items():
        assert torch.isfinite(torch.tensor(v)), f"Metric '{k}' is not finite: {v}"


def test_loss_is_non_negative(forward_outputs):
    """Total loss should be >= 0."""
    video_tensor, topology, gt_video_mock, loss_fn = forward_outputs
    loss_total, _ = loss_fn((video_tensor, topology), gt_video_mock)

    assert loss_total.item() >= 0.0, f"Total loss is negative: {loss_total.item()}"


def test_backward_pass(device):
    """Backward pass through the loss should not raise errors."""
    from unittest.mock import patch
    import numpy as np
    from core.crf_tensor import CRFTensor
    from core.constants import N_SLOTS

    # Use smaller dimensions to avoid OOM on CPU
    model = CBAE_EndToEnd(render_width=16, render_height=16).to(device)
    loss_fn = CBAELossWrapper().to(device)
    batch_size = 1
    time_steps = 2  # Only 2 steps to keep memory manageable

    # Build mock outputs with requires_grad so backward can flow
    mock_trajectory = torch.randn(time_steps, batch_size, 3200, requires_grad=True)
    mock_slot_embs = torch.randn(batch_size, N_SLOTS, 64, requires_grad=True)
    mock_text_emb = torch.randn(batch_size, 512)
    mock_base_crf = CRFTensor()
    mock_base_crf.alpha = np.ones(N_SLOTS, dtype=np.float32)
    mock_base_crf.z = np.zeros(N_SLOTS, dtype=np.float32)
    mock_base_crf.csg = np.zeros(N_SLOTS, dtype=bool)

    gt_video_mock = torch.rand(batch_size, time_steps, 16, 16, 3).to(device)

    with patch.object(model.seq_model, 'forward',
                      return_value=(mock_trajectory, mock_slot_embs, mock_text_emb, mock_base_crf)):
        video_tensor, topology = model(["A gentle blue character breathing"],
                                       torch.randn(batch_size, 16000))
        loss_total, _ = loss_fn((video_tensor, topology), gt_video_mock)
        loss_total.backward()  # Should not raise

