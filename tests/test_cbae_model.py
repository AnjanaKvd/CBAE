import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from models.cbae_model import CBAE_EndToEnd
from core.crf_tensor import CRFTensor
from core.constants import N_SLOTS

def test_cbae_end_to_end_forward():
    # Instantiate the model
    model = CBAE_EndToEnd(render_width=256, render_height=256, use_diffvg=False)
    
    # We will mock the SequenceModel's forward pass to avoid the expensive ODE integration 
    # of 192 time steps during a unit test, opting for 2 steps.
    
    batch_size = 1
    time_steps = 2
    
    # SequenceModel is expected (based on cbae_model.py) to return:
    # trajectory, slot_embs, text_emb, base_crf
    
    # 1. Trajectory: (time_steps, batch_size, 3200)
    mock_trajectory = torch.randn(time_steps, batch_size, 3200)
    
    # 2. Slot embeddings: (batch_size, 128, 64)
    mock_slot_embs = torch.randn(batch_size, N_SLOTS, 64)
    
    # 3. Text embedding: (batch_size, 512)
    mock_text_emb = torch.randn(batch_size, 512)
    
    # 4. Base CRFTensor: Needs alpha, z_index, is_csg_subtract numpy arrays
    mock_base_crf = CRFTensor()
    mock_base_crf.alpha = np.ones(N_SLOTS, dtype=np.float32)
    mock_base_crf.z_index = np.zeros(N_SLOTS, dtype=np.float32)
    mock_base_crf.is_csg_subtract = np.zeros(N_SLOTS, dtype=bool)
    
    with patch.object(model.seq_model, 'forward', return_value=(mock_trajectory, mock_slot_embs, mock_text_emb, mock_base_crf)):
    
        # Provide mock inputs to CBAE_EndToEnd
        prompt = 'A red ball'
        audio = torch.randn(batch_size, 48000)
        
        # Run the forward pass
        video_tensor = model(prompt, audio)
        
        # Assert output shape is (1, 2, 256, 256, 3) representing (batch, frames, H, W, channels)
        assert video_tensor.shape == (batch_size, time_steps, 256, 256, 3)
        assert video_tensor.dtype == torch.float32
        
        # Assert no NaNs or Infs
        assert torch.isfinite(video_tensor).all()
