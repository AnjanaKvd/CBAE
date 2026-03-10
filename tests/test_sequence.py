import pytest
import torch

from models.sequence import SequenceModel

def test_sequence_model_forward_shape():
    model = SequenceModel()
    
    batch_size = 2
    prompt = ["A bouncing red ball"] * batch_size
    # mock audio tensor (batch, 48000) -> 3 seconds of 16kHz audio
    audio = torch.randn(batch_size, 48000)
    
    # We should avoid doing actual ODE integration for a full 192 steps if it's too slow in a simple shape test, 
    # but the prompt requires validating the output tuple or tensor represents the evaluated trajectory.
    # The ODE runs automatically in forward(). 
    # We can rely on torchdiffeq's `odeint` for a quick integration if the model isn't massive.
    
    # By default, SequenceModel has no external dependencies for just the forward shape test.
    # It will use the initialized random weights.
    
    trajectory = model(prompt, audio)
    
    # Expected shape: (192 time steps, 2 batch, 3200 state_dim)
    assert trajectory.shape == (192, batch_size, 3200)
    assert trajectory.dtype == torch.float32
    
    # Just to be sure, ODE output has finite values
    assert torch.isfinite(trajectory).all()
