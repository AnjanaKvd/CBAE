import torch
import torch.nn as nn
from models.sequence import SequenceModel
from models.color_mlp import ColorPredictionMLP
from rendering.diff_rasterizer import DiffRasterizer

class CBAE_EndToEnd(nn.Module):
    def __init__(self, render_width=256, render_height=256, use_diffvg=False, n_steps=192):
        """
        The top-level overarching architecture connecting encoders, ODE logic, and rasterization rendering logically seamlessly propagating end-to-end backprop pipelines cleanly.
        """
        super().__init__()
        self.seq_model = SequenceModel(n_steps=n_steps)
        self.color_mlp = ColorPredictionMLP()
        self.rasterizer = DiffRasterizer(use_diffvg=use_diffvg, fallback_softness=0.01)
        self.width = render_width
        self.height = render_height
        
    def forward(self, P=None, colors=None, alive=None, z=None) -> tuple:
        """
        Full CBAE VAE-ODE forward pass: encode topology → ODE integrate → rasterize.

        Args:
            P, colors, alive: Frame 0 ground truth topology (for training)
            z: Latent random seed (for generation)

        Returns:
            video_tensor: (batch_size, T, height, width, 3)
            topology: dict with P, aliveness, colors, alpha, z, csg, mu, logvar
        """
        device = next(self.parameters()).device

        # 1. Integrate geometries via Neural ODE
        trajectory, slot_embs, latent_z, base_crf, mu, logvar = self.seq_model(P, colors, alive, z)

        # 2. Extract static properties from base CRF (shared across all timesteps)
        alpha = torch.from_numpy(base_crf.alpha).float().to(device)
        z_depth = torch.from_numpy(base_crf.z).float().to(device)
        csg = torch.from_numpy(base_crf.csg).bool().to(device)

        # 3. Predict colors (time-independent)
        predicted_colors = self.color_mlp(slot_embs, latent_z)  # (batch, 128, 3)

        time_steps, batch_size, _ = trajectory.shape

        # 4. Pre-compute all P and aliveness for all timesteps at once
        # trajectory: (T, batch, 3200) -> unflatten once
        P_all = trajectory[:, :, :3072].reshape(time_steps, batch_size, 128, 12, 2)
        aliveness_all = trajectory[:, :, 3072:].reshape(time_steps, batch_size, 128)

        # Pre-compute smooth alpha for all timesteps and batches: (T, batch, 128)
        smooth_alpha_all = alpha.unsqueeze(0).unsqueeze(0) * torch.sigmoid(aliveness_all)

        # 5. Rasterize — iterate over timesteps, batch items handled individually
        # (compositing is inherently sequential per-frame)
        video_frames = []
        for t in range(time_steps):
            batch_frames = []
            for b in range(batch_size):
                frame = self.rasterizer(
                    P=P_all[t, b],
                    c=predicted_colors[b],
                    alpha=smooth_alpha_all[t, b],
                    alive=aliveness_all[t, b],
                    z=z_depth,
                    csg=csg,
                    width=self.width,
                    height=self.height
                )
                batch_frames.append(frame)
            video_frames.append(torch.stack(batch_frames, dim=0))

        # 6. Stack: list of (batch, H, W, 3) -> (T, batch, H, W, 3) -> (batch, T, H, W, 3)
        video_tensor = torch.stack(video_frames, dim=0).transpose(0, 1)

        # 7. Topology dict for loss computation
        P_seq = P_all.permute(1, 0, 2, 3, 4)       # (batch, T, 128, 12, 2)
        aliveness_seq = aliveness_all.permute(1, 0, 2)  # (batch, T, 128)

        topology = {
            'P': P_seq,
            'aliveness': aliveness_seq,
            'colors': predicted_colors,
            'alpha': alpha,
            'z': z_depth,
            'csg': csg,
            'latent_z': latent_z,
            'mu': mu,
            'logvar': logvar
        }

        return video_tensor, topology
