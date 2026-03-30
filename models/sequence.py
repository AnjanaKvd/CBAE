import torch
import torch.nn as nn
from torchdiffeq import odeint

from models.vae import TopologicalVAE
from models.deformation_mlp import TemplateLibrary, DeformationMLP, initialize_crf
from models.aliveness import AlivenessMLP, InitialStateCombiner
from models.neural_ode import ODEFx

class SequenceModel(nn.Module):
    """
    Top-level PyTorch Module assembling the complete CBAE Neural ODE Sequence pipeline.
    Now operates as an unconditional Topological VAE-ODE generator.
    """
    def __init__(self, n_steps=192):
        super().__init__()
        self.n_steps = n_steps
        
        # 1. Topological VAE (replaces CLIP/Whisper)
        self.vae = TopologicalVAE()
        
        # 2. Deformation & Initial State
        self.template_lib = TemplateLibrary()
        self.deformation_mlp = DeformationMLP()
        self.aliveness_mlp = AlivenessMLP()
        self.state_combiner = InitialStateCombiner()
        
        # 3. Neural ODE
        self.ode_fx = ODEFx()

    def forward(self, P=None, colors=None, alive=None, z=None):
        """
        Executes the full forward pass from a latent seed mapped into continuous ODE sequences.
        
        Args:
            P, colors, alive: Frame 0 topology (used during training)
            z: Latent random seed (used during generation)
            
        Returns:
            trajectory: Torch tensor shaped (192, batch_size, 3200) representing 192 temporal frames
        """
        device = next(self.ode_fx.parameters()).device
        
        if z is None:
            # Training: Encode Frame 0 topology to latent distribution
            slot_embs, mu, logvar, z = self.vae(P, colors, alive)
        else:
            # Generation: Decode directly from random latent seed
            slot_embs = self.vae.decoder(z)
            mu, logvar = None, None
        
        # 4. Get ΔP_i (via DeformationMLP) -> generating base geometry CRFTensor
        # Using latent z as the global conditioning parameter (replacing text_emb)
        base_crf = initialize_crf(
            self.template_lib,
            self.deformation_mlp,
            z,
            slot_embs
        )
        
        # 5. Get Aliveness logits (via AlivenessMLP) -> Output: (batch_size, 128)
        aliveness_logits = self.aliveness_mlp(slot_embs)
        
        # 6. Flatten geometry & aliveness (via InitialStateCombiner) -> Output (batch_size, 3200)
        initial_state = self.state_combiner(base_crf, aliveness_logits)
        initial_state = initial_state.to(device)
        
        # 7. Use torchdiffeq.odeint with RK4 to integrate ODEFx from t=0 to t=8.0
        t_span = torch.linspace(0.0, 8.0, self.n_steps, device=device)
        
        trajectory = odeint(
            func=self.ode_fx,
            y0=initial_state,
            t=t_span,
            method='rk4'
        )
        
        # trajectory shape: (192, batch_size, 3200)
        return trajectory, slot_embs, z, base_crf, mu, logvar
