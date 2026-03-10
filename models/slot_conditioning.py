import torch
import torch.nn as nn
from core.constants import N_SLOTS

class SlotEmbeddingTable(nn.Module):
    def __init__(self, num_embeddings=N_SLOTS, embedding_dim=64):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_idx: Int tensor of slot indices (batch_size, num_slots)
        Returns:
            slot_emb: Float tensor of embeddings (batch_size, num_slots, 64)
        """
        return self.embeddings(slot_idx)

class MotionConditioningMLP(nn.Module):
    def __init__(self, slot_dim=64, text_dim=512, hidden_dim=256, out_dim=64):
        super().__init__()
        in_dim = slot_dim + text_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, slot_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_emb: (..., 64)
            text_emb: (..., 512) -> might need broadcasting depending on shape
        Returns:
            e_i: (..., 64)
        """
        # Ensure text_emb is broadcast correctly if necessary
        # Usually text_emb is (batch, 512) and slot_emb is (batch, num_slots, 64)
        if text_emb.dim() < slot_emb.dim():
            text_emb = text_emb.unsqueeze(1).expand(-1, slot_emb.size(1), -1)
            
        x = torch.cat([slot_emb, text_emb], dim=-1)
        return self.net(x)

class MouthConditioningMLP(nn.Module):
    def __init__(self, slot_dim=64, text_dim=512, audio_dim=384, hidden_dim=256, out_dim=64):
        super().__init__()
        in_dim = slot_dim + text_dim + audio_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, slot_emb: torch.Tensor, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_emb: (..., 64)
            text_emb: (..., 512)
            audio_emb: (..., 384)
        Returns:
            e_i: (..., 64)
        """
        # Broadcast text and audio to match slot_emb shape if they are (batch, dim)
        if text_emb.dim() < slot_emb.dim():
            text_emb = text_emb.unsqueeze(1).expand(-1, slot_emb.size(1), -1)
        if audio_emb.dim() < slot_emb.dim():
            audio_emb = audio_emb.unsqueeze(1).expand(-1, slot_emb.size(1), -1)
            
        x = torch.cat([slot_emb, text_emb, audio_emb], dim=-1)
        return self.net(x)

class SlotConditioner(nn.Module):
    def __init__(self, num_slots=N_SLOTS, slot_dim=64, text_dim=512, audio_dim=384):
        super().__init__()
        self.num_slots = num_slots
        self.slot_embeddings = SlotEmbeddingTable(num_slots, slot_dim)
        
        # Shared motion MLP for all slots except mouth (0-70, 91-127)
        self.motion_mlp = MotionConditioningMLP(slot_dim, text_dim)
        
        # Specific mouth MLP (71-90)
        self.mouth_mlp = MouthConditioningMLP(slot_dim, text_dim, audio_dim)
        
        # Pre-compute mouth slot mask for quick routing
        from core.constants import SLOT_MOUTH
        mouth_mask = torch.zeros(num_slots, dtype=torch.bool)
        mouth_mask[SLOT_MOUTH[0]:SLOT_MOUTH[1]+1] = True
        self.register_buffer('mouth_mask', mouth_mask)
        
    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Computes conditional embeddings for every slot.
        
        Args:
            text_emb: (batch_size, 512)
            audio_emb: (batch_size, 384) - Can be None if no audio provided (or if evaluating non-mouth only)
            
        Returns:
            (batch_size, 128, 64) tensor of motion embeddings.
        """
        batch_size = text_emb.size(0)
        device = text_emb.device
        
        # Generate slot indices: (batch_size, 128)
        slot_indices = torch.arange(self.num_slots, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get base slot embeddings: (batch_size, 128, 64)
        base_slot_embs = self.slot_embeddings(slot_indices)
        
        # Default all slots to non-mouth motion embeddings
        # This computes it for all 128 slots safely
        out_embs = self.motion_mlp(base_slot_embs, text_emb)
        
        # If we have audio, overwrite the mouth slots (71-90) with the mouth MLP output
        if audio_emb is not None:
            mouth_indices = torch.nonzero(self.mouth_mask).squeeze()
            
            # Extract only the mouth slots from the base embeddings: (batch, n_mouth_slots, 64)
            mouth_base_embs = base_slot_embs[:, mouth_indices, :]
            
            # Pass through mouth MLP
            mouth_out_embs = self.mouth_mlp(mouth_base_embs, text_emb, audio_emb)
            
            # Scatter/overwrite the specific output slots
            # out_embs is (batch, 128, 64), mouth_out_embs is (batch, 20, 64)
            out_embs[:, mouth_indices, :] = mouth_out_embs
            
        return out_embs
