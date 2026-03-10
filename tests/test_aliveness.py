import pytest
import torch
import numpy as np
from core.crf_tensor import CRFTensor
from core.constants import N_SLOTS, N_CTRL_PTS
from models.aliveness import AlivenessMLP, InitialStateCombiner

def test_aliveness_mlp_shape():
    mlp = AlivenessMLP()
    batch_size = 2
    # mock conditional slot embeddings (batch, N_SLOTS, 64)
    slot_embs = torch.randn(batch_size, N_SLOTS, 64)
    
    out = mlp(slot_embs)
    
    # Check output is float tensor (batch, N_SLOTS)
    assert out.shape == (batch_size, N_SLOTS)
    assert out.dtype == torch.float32

def test_initial_state_combiner_shape():
    combiner = InitialStateCombiner()
    batch_size = 2
    
    # Mock CRFTensor
    crf = CRFTensor()
    crf.P = np.random.rand(N_SLOTS, N_CTRL_PTS, 2).astype(np.float32)
    
    # Mock aliveness logits
    aliveness_logits = torch.randn(batch_size, N_SLOTS)
    
    flat_state = combiner(crf, aliveness_logits)
    
    # 1. Validate output shape (batch_size, 3200)
    assert flat_state.shape == (batch_size, 3200)
    
    # 2. Check first 3072 elements reconstruct to P correctly
    P_flat = flat_state[:, :3072]
    P_reconstructed = P_flat.view(batch_size, N_SLOTS, N_CTRL_PTS, 2)
    
    # The CRFTensor was duplicated to match batch size internal to combiner
    P_tensor_expected = torch.from_numpy(crf.P).float().unsqueeze(0).expand(batch_size, -1, -1, -1)
    assert torch.allclose(P_reconstructed, P_tensor_expected)
    
    # 3. Check last 128 elements match logits exactly
    A_flat = flat_state[:, 3072:]
    assert torch.allclose(A_flat, aliveness_logits)
