import torch
import torch.nn as nn
from core.constants import N_SLOTS

class ColorPredictionMLP(nn.Module):
    def __init__(self, slot_dim=512, text_dim=512, hidden_dim=256, num_slots=N_SLOTS):
        """
        Predicts robust (batch_size, 128, 3) colors based on text prompt + structural primitives.
        """
        super().__init__()
        in_dim = slot_dim + text_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3), # 3 logical RGB channels natively representing properties
            nn.Sigmoid()  # Restrict strictly mathematically between [0.0, 1.0]
        )
        
    def forward(self, slot_embs: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Computes dynamic [0.0, 1.0] bounded color mappings for shape parameters natively mapping.
        
        Args:
            slot_embs: Conditional embedding states (batch_size, num_slots, 64)
            text_emb: Prompt representations (batch_size, 512)
            
        Returns:
            rgb_colors: Explicit float arrays mapped reliably (batch_size, num_slots, 3)
        """
        # Ensure text_emb correctly expands along num_slots dim for computation dynamically
        # text_emb is typically (batch_size, 512)
        if text_emb.dim() == 2:
            # (batch_size, 1, 512) --> (batch_size, num_slots, 512)
            text_expanded = text_emb.unsqueeze(1).expand(-1, slot_embs.size(1), -1)
        else:
            text_expanded = text_emb
            
        # Concat the conditions -> (batch_size, num_slots, 576)
        x = torch.cat([slot_embs, text_expanded], dim=-1)
        
        # Predict Colors -> (batch_size, num_slots, 3)
        colors = self.net(x)
        
        return colors
