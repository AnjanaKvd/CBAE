import torch
import torch.nn as nn
from core.crf_tensor import CRFTensor
from core.constants import N_SLOTS, N_CTRL_PTS

class AlivenessMLP(nn.Module):
    def __init__(self, slot_dim=64, hidden_dim=32):
        """
        Computes the initial aliveness logit for each slot.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, slot_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_embs: (batch_size, N_SLOTS, 64) conditional embeddings
            
        Returns:
            a_i: (batch_size, N_SLOTS) logits representing initial alive state probability
        """
        # (batch, N_SLOTS, 1)
        raw_logits = self.net(slot_embs)
        # Sequence squeeze -> (batch, N_SLOTS)
        return raw_logits.squeeze(-1)


class InitialStateCombiner(nn.Module):
    """
    Combines initialized control points and aliveness logits into a flat vector for torchdiffeq.
    """
    def __init__(self):
        super().__init__()
        # Size assertions as per specification
        self.p_size = N_SLOTS * N_CTRL_PTS * 2  # 128 * 12 * 2 = 3072
        self.a_size = N_SLOTS                   # 128
        self.out_size = self.p_size + self.a_size # 3200
        
    def forward(self, crf_tensor: CRFTensor, aliveness_logits: torch.Tensor) -> torch.Tensor:
        """
        Creates the flat (batch_size, 3200) state vector.
        
        Args:
            crf_tensor: The starting CRFTensor (usually from initialize_crf). 
                        It is assumed to be a single CRFTensor for batch_size=1, 
                        or we map its properties natively.
            aliveness_logits: float tensor of shape (batch, 128)
            
        Returns:
            flat_state: tensor of shape (batch, 3200)
        """
        # Ensure we can extract batch dimension from the logits
        batch_size = aliveness_logits.size(0)
        device = aliveness_logits.device
        
        # 1. Extract control points
        # crf_tensor.P is normally a numpy array of shape (128, 12, 2). 
        # Convert to tensor and broadcast to batch if necessary.
        P_tensor = torch.from_numpy(crf_tensor.P).float().to(device)
        
        if P_tensor.dim() == 3: # (128, 12, 2)
            P_tensor = P_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
        # Flatten P: (batch, 128, 12, 2) -> (batch, 3072)
        P_flat = P_tensor.reshape(batch_size, self.p_size)
        
        # 2. Concatenate with aliveness logits -> (batch, 3200)
        flat_state = torch.cat([P_flat, aliveness_logits], dim=1)
        
        return flat_state

def build_initial_state(crf_tensor: CRFTensor, aliveness_logits: torch.Tensor) -> torch.Tensor:
    """Convenience function wrapper for InitialStateCombiner."""
    combiner = InitialStateCombiner()
    return combiner(crf_tensor, aliveness_logits)
