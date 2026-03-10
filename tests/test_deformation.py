import pytest
import torch
import numpy as np

from models.deformation_mlp import TemplateLibrary, DeformationMLP, initialize_crf
from core.crf_tensor import CRFTensor
from core.slot_blocks import get_delta_max
from core.constants import N_SLOTS, N_CTRL_PTS

def test_template_library_fallback():
    # Pass a non-existent directory to trigger the fallback logic
    library = TemplateLibrary(templates_dir="/this_dir_definitely_does_not_exist_12345")
    
    # Should have at least 1 template generated
    assert len(library.templates) >= 1
    assert isinstance(library.templates[0], CRFTensor)
    
    # Should have a dummy embedding of shape (1, 512)
    assert library.embeddings is not None
    assert library.embeddings.shape == (1, 512)

def test_deformation_mlp_shape():
    mlp = DeformationMLP()
    
    batch_size = 2
    # Mock text embeddings: (Batch, 512)
    text_emb = torch.randn(batch_size, 512)
    
    # Mock slot embeddings: (Batch, N_SLOTS, 64)
    slot_embs = torch.randn(batch_size, N_SLOTS, 64)
    
    # Forward pass
    delta_P = mlp(text_emb, slot_embs)
    
    # Output shape should be (Batch, N_SLOTS, 12, 2)
    assert delta_P.shape == (batch_size, N_SLOTS, N_CTRL_PTS, 2)
    
    # Delta max for slot 0 (body base) is 0.0 in CBAE (assuming it's a completely static background slot)
    # Actually wait - get_delta_max(0) which is bg_static[0] is 0.0
    # Let's verify min/max is bounded by delta_max
    
    # Check that outputs for slot 0 are bounded by its max delta
    limit = get_delta_max(0)
    assert torch.all(torch.abs(delta_P[:, 0, :, :]) <= limit)

def test_initialize_crf():
    # Setup components
    library = TemplateLibrary(templates_dir="/this_dir_definitely_does_not_exist_12345")
    mlp = DeformationMLP()
    
    text_emb = torch.randn(1, 512)
    slot_embs = torch.randn(1, N_SLOTS, 64)
    
    # Run initialize_crf
    crf = initialize_crf(library, mlp, text_emb, slot_embs)
    
    assert isinstance(crf, CRFTensor)
    
    # Verify bounds [0, 1] on control points
    assert np.max(crf.P) <= 1.0
    assert np.min(crf.P) >= 0.0
