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


def test_backward_pass(device, model_and_loss):
    """Backward pass through the loss should not raise errors."""
    model, loss_fn = model_and_loss
    batch_size = 2
    prompt = ["A gentle blue character breathing", "A red block jumping"]
    audio = torch.randn(batch_size, 16000 * 8).to(device)
    gt_video_mock = torch.rand(batch_size, 192, 32, 32, 3).to(device)

    video_tensor, topology = model(prompt, audio)
    loss_total, _ = loss_fn((video_tensor, topology), gt_video_mock)
    loss_total.backward()  # Should not raise

