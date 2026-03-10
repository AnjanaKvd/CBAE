import torch
import torch.nn as nn
from core.constants import N_SLOTS, N_CTRL_PTS

class ODEFx(nn.Module):
    def __init__(self, num_slots=N_SLOTS, hidden_dim=128):
        """
        The continuous-time derivative ODE function for the CBAE solver.
        Utilizes self-attention to relate slot states and predict their velocity.
        """
        super().__init__()
        self.num_slots = num_slots
        
        # 24 control point dimensions + 1 aliveness logit = 25 dims per slot
        self.slot_dim = (N_CTRL_PTS * 2) + 1
        
        # Self-attention directly correlates intersections and occlusion logic between active slots
        self.attention = nn.MultiheadAttention(
            embed_dim=self.slot_dim, 
            num_heads=5, 
            batch_first=True
        )
        
        # Geometrical velocity mapping MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.slot_dim),
            nn.Tanh()  # Tanh bounds the predicted velocities preventing runaway integrations
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Signature matches `torchdiffeq` natively: f(t, y)
        
        Args:
            t: A scalar time tensor in [0, 1].
            state: The continuously integrated topology array shaped (batch_size, 3200).
                   [0:3072] -> Control Points | [3072:3200] -> Aliveness
                   
        Returns:
            dstate_dt: Same (batch_size, 3200) shape encoding the gradients of movement matching delta targets.
        """
        batch_size = state.size(0)
        
        # 1. Unflatten explicitly into slot domains
        # Control points: (batch_size, 3072) -> (batch_size, 128, 24)
        P_flat = state[:, :3072].view(batch_size, self.num_slots, 24)
        
        # Aliveness logits: (batch_size, 128) -> (batch_size, 128, 1)
        A_flat = state[:, 3072:].view(batch_size, self.num_slots, 1)
        
        # 2. Re-combine state purely on a per-slot basis
        # x shape: (batch_size, 128, 25)
        x = torch.cat([P_flat, A_flat], dim=-1)
        
        # 3. Message passing across structural primitives
        # Self-attention relates slots to one another (e.g., eyes align relative to face)
        attn_out, _ = self.attention(x, x, x)
        
        # Residual projection
        x = x + attn_out
        
        # 4. Predict time derivatives dynamically (dP/dt, dA/dt)
        dx = self.mlp(x)
        
        # 5. Segment and re-flatten to exact memory spec
        dP_dt = dx[:, :, :24].reshape(batch_size, 3072)
        dA_dt = dx[:, :, 24:].reshape(batch_size, 128)
        
        dstate_dt = torch.cat([dP_dt, dA_dt], dim=1)
        
        return dstate_dt
