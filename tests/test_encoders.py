import pytest
import torch
import numpy as np

from models.encoders import CLIPEncoder, WhisperEncoder

def test_clip_encoder_shape():
    encoder = CLIPEncoder(device="cpu")
    
    # Test single string
    emb_single = encoder.encode_text("hello")
    assert emb_single.shape == (1, 512)
    assert emb_single.dtype == torch.float32
    
    # Test list of strings
    emb_list = encoder.encode_text(["hello", "world"])
    assert emb_list.shape == (2, 512)
    assert emb_list.dtype == torch.float32

def test_whisper_encoder_shape():
    encoder = WhisperEncoder(device="cpu")
    
    # Simulate 1 second of audio at 16kHz
    audio_array = np.zeros(16000, dtype=np.float32)
    
    emb = encoder.encode_audio(audio_array, sr=16000)
    
    # Shape should be (Batch, T, 384) for whisper-tiny
    # Batch = 1.
    assert len(emb.shape) == 3
    assert emb.shape[0] == 1
    assert emb.shape[2] == 384
    assert emb.dtype == torch.float32

def test_encoders_frozen():
    clip_encoder = CLIPEncoder(device="cpu")
    for param in clip_encoder.model.parameters():
        assert not param.requires_grad
        
    whisper_encoder = WhisperEncoder(device="cpu")
    for param in whisper_encoder.encoder.parameters():
        assert not param.requires_grad
