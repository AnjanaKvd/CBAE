import torch
from models.slot_conditioning import SlotConditioner
from core.constants import N_SLOTS

def test_slot_conditioner():
    conditioner = SlotConditioner()
    # Mock inputs
    batch_size = 2
    text_emb = torch.randn(batch_size, 512)
    audio_emb = torch.randn(batch_size, 384)
    
    # Forward
    out = conditioner(text_emb, audio_emb)
    
    assert out.shape == (batch_size, 128, 64), f"Wrong shape: {out.shape}"
    print("SlotConditioner shape verification passed! Shape:", out.shape)
    
if __name__ == "__main__":
    test_slot_conditioner()
