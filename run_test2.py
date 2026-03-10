import torch
from models.encoders import WhisperEncoder
from models.audio_alignment import AudioAlignmentLayer
from models.slot_conditioning import SlotConditioner

def test():
    whisper = WhisperEncoder(device="cpu")
    audio = torch.randn(2, 48000)
    audio_raw_emb = whisper.encode_audio(audio)
    print("audio_raw_emb shape:", audio_raw_emb.shape)
    
    audio_align = AudioAlignmentLayer()
    audio_aligned = audio_align(audio_raw_emb)
    print("audio_aligned shape:", audio_aligned.shape)
    
    audio_0 = audio_align.get_frame_embedding(audio_aligned, 0)
    print("audio_0 shape:", audio_0.shape)

test()
