import pytest
import torch

from models.audio_alignment import AudioAlignmentLayer

def test_audio_alignment_shape():
    layer = AudioAlignmentLayer()
    
    # Mock input: (Batch, Seq, Features) -> (2 batches, ~8s audio at 50Hz, 384 dim)
    # 8 * 50 = 400
    batch_size = 2
    seq_len = 400
    features = 384
    mock_audio_emb = torch.randn(batch_size, seq_len, features)
    
    output = layer(mock_audio_emb)
    
    # Output should be downsampled by 2 (stride=2 in Conv1d)
    # -> (2, 200, 384)
    assert output.shape == (2, 200, 384)
    assert output.dtype == torch.float32

def test_get_frame_embedding():
    layer = AudioAlignmentLayer()
    
    aligned_emb = torch.randn(2, 200, 384)
    
    # Test valid index
    t_valid = 10
    emb1 = layer.get_frame_embedding(aligned_emb, t=t_valid)
    assert emb1.shape == (2, 384)
    assert torch.allclose(emb1, aligned_emb[:, t_valid, :])
    
    # Test out of bounds index (safety clamp)
    t_oob = 500
    emb2 = layer.get_frame_embedding(aligned_emb, t=t_oob)
    assert emb2.shape == (2, 384)
    assert torch.allclose(emb2, aligned_emb[:, 199, :])  # Clamped to last valid index 199
