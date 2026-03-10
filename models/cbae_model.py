import torch
import torch.nn as nn
from models.sequence import SequenceModel
from models.color_mlp import ColorPredictionMLP
from rendering.diff_rasterizer import DiffRasterizer

class CBAE_EndToEnd(nn.Module):
    def __init__(self, render_width=512, render_height=512, use_diffvg=False):
        """
        The top-level overarching architecture connecting encoders, ODE logic, and rasterization rendering logically seamlessly propagating end-to-end backprop pipelines cleanly.
        """
        super().__init__()
        self.seq_model = SequenceModel()
        self.color_mlp = ColorPredictionMLP()
        self.rasterizer = DiffRasterizer(use_diffvg=use_diffvg, fallback_softness=0.01)
        self.width = render_width
        self.height = render_height
        
    def forward(self, prompt: str, audio: torch.Tensor) -> torch.Tensor:
        """
        Executes full pipeline computing explicit geometric state topologies temporally mapping them across 192 images natively handling differentiable render loops.
        
        Args:
            prompt: Text conditional bounds
            audio: Sequence conditional bound
            
        Returns:
            video_tensor: Differentiable array sequence natively sized (batch_size, 192, height, width, 3)
        """
        device = next(self.parameters()).device
        
        # 1. Integrate Explicit Geometries (RK4 Neural ODE mapped shapes)
        trajectory, slot_embs, text_emb, base_crf = self.seq_model(prompt, audio)
        
        # 2. Extract explicit configuration arrays logically independent from time (e.g global scale or Z indices)
        alpha = torch.from_numpy(base_crf.alpha).float().to(device)
        z = torch.from_numpy(base_crf.z_index).float().to(device)
        csg = torch.from_numpy(base_crf.is_csg_subtract).bool().to(device)
        
        # 3. Retrieve explicitly bounded RGB predictions conditionally
        colors = self.color_mlp(slot_embs, text_emb) # (batch, 128, 3)
        
        time_steps, batch_size, _ = trajectory.shape
        video_frames = []
        
        # 4. Rasterization Compositing Sequence
        # DiffRasterizer executes 2D composites natively on individual configurations sequentially
        for t in range(time_steps):
            state_t = trajectory[t] # (batch_size, 3200)
            
            # Unflatten explicit geometries
            P_t = state_t[:, :3072].view(batch_size, 128, 12, 2)
            aliveness_t = state_t[:, 3072:].view(batch_size, 128)
            
            batch_frames = []
            for b in range(batch_size):
                frame = self.rasterizer(
                    P=P_t[b], 
                    c=colors[b], 
                    alpha=alpha, 
                    alive=aliveness_t[b], 
                    z=z, 
                    csg=csg, 
                    width=self.width, 
                    height=self.height
                )
                batch_frames.append(frame)
                
            # Stack batch into (batch_size, H, W, 3)
            batch_frames = torch.stack(batch_frames, dim=0)
            video_frames.append(batch_frames)
            
        # 5. Native PyTorch video representations
        # Stacking across lists gives (192, batch_size, H, W, 3) 
        # Then transpose to match specs explicitly: (batch_size, 192, H, W, 3)
        video_tensor = torch.stack(video_frames, dim=0).transpose(0, 1)
        
        return video_tensor
