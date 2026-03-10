import torch

slot_emb = torch.randn(2, 20, 64)
text_emb = torch.randn(2, 512)
audio_emb = torch.randn(2, 1, 384)

print("init text_emb:", text_emb.shape)
print("init audio_emb:", audio_emb.shape)

if text_emb.dim() < slot_emb.dim():
    text_emb = text_emb.unsqueeze(1)
text_emb = text_emb.expand(-1, slot_emb.size(1), -1)
    
if audio_emb.dim() < slot_emb.dim():
    audio_emb = audio_emb.unsqueeze(1)
audio_emb = audio_emb.expand(-1, slot_emb.size(1), -1)

print("exp text_emb:", text_emb.shape)
print("exp audio_emb:", audio_emb.shape)

x = torch.cat([slot_emb, text_emb, audio_emb], dim=-1)
print("cat x:", x.shape)
