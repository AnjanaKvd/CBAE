import torch
import torch.nn as nn
from core.crf_tensor import CRFTensor
from rendering.diff_rasterizer import bezier_to_polyline_torch

class CBAELossWrapper(nn.Module):
    def __init__(self, w_render=1.0, w_bcs=0.1, w_crs=0.1, w_temp=0.5):
        """
        Computes the objective function scaling topological logic and raster representations natively into gradients over parameters predicting CBAE trajectories mathematically.
        """
        super().__init__()
        self.w_render = w_render
        self.w_bcs = w_bcs
        self.w_crs = w_crs
        self.w_temp = w_temp
        
        # Placeholder for LPIPS or similar perceptual spatial pixel boundaries locally executing gradient maps
        self.l1_loss = nn.L1Loss()
        
    def compute_bcs(self, P: torch.Tensor, aliveness: torch.Tensor) -> torch.Tensor:
        """
        Computes the Boundary Constraint Score mapping curvature smoothly filtering extreme changes temporally.
        P: (batch, T, slots, 12, 2)
        aliveness: (batch, T, slots) - raw logits
        """
        batch, T, slots, _, _ = P.shape
        # Flatten time into batch mapping shapes reliably evaluating the polylines cleanly
        P_flat = P.view(batch * T, slots, 12, 2)
        
        # We sample 30 discrete segments mapping curve trajectories explicitly approximating curvature derivatives mathematically
        polylines = bezier_to_polyline_torch(P_flat, n_samples=30) # (batch*T, slots, 30, 2)
        
        # Approximate curvature logically mathematically tracking diffs
        # 1st derivative (velocity along curve u parameter)
        dP_du = polylines[:, :, 1:, :] - polylines[:, :, :-1, :] # (..., 29, 2)
        
        # 2nd derivative (acceleration along curve u parameter)
        d2P_du2 = dP_du[:, :, 1:, :] - dP_du[:, :, :-1, :] # (..., 28, 2)
        
        # Pad dP_du natively matching sizes to compute cross products
        dP_du_trunc = dP_du[:, :, :-1, :] # (..., 28, 2)
        
        # Cross product in 2D: |x1*y2 - y1*x2|
        cross_prod = torch.abs(dP_du_trunc[..., 0] * d2P_du2[..., 1] - dP_du_trunc[..., 1] * d2P_du2[..., 0])
        
        # Magnitude cubed
        norm_v = torch.norm(dP_du_trunc, dim=-1)
        denom = torch.pow(norm_v, 3) + 1e-6
        
        # Approximate curvature
        kappa = cross_prod / denom
        
        # Mean absolute curvature mathematically extracting bounds over the curve logic directly 
        K = kappa.mean(dim=-1) # (batch*T, slots)
        
        # Reshape to time steps cleanly extracting matrices
        K_time = K.view(batch, T, slots)
        
        # Temporal differences 
        dK = torch.abs(K_time[:, 1:, :] - K_time[:, :-1, :]) # (batch, T-1, slots)
        
        # Weight by aliveness explicitly masking non-active geometries scaling values mathematically smoothly
        alive_mask = torch.sigmoid(aliveness[:, 1:, :]) # Use the state predicting the change metric bounds natively
        
        # We calculate the final weighted structural logic
        bcs_loss = (dK * alive_mask).mean()
        return bcs_loss

    def compute_crs(self, colors: torch.Tensor, aliveness: torch.Tensor) -> torch.Tensor:
        """
        Computes the Content Retention Score reducing temporal fluctuations in RGB assignments correctly mapping structural allocations statically scaling mathematically towards 0 natively if shapes remain temporally invariant.
        colors: (batch, T, slots, 3) OR (batch, slots, 3)
        aliveness: (batch, T, slots) - raw logits
        """
        # If the model explicitly limits parameters dynamically mapping (batch, slots, 3) naturally the temporal loss structurally equates to exactly 0 protecting matrices
        if colors.dim() == 3:
            return torch.tensor(0.0, device=colors.device)
            
        # colors: (batch, T, slots, 3)
        # Compute mean structurally natively
        c_mean = colors.mean(dim=1, keepdim=True)
        
        # MSE mapping
        mse = (colors - c_mean).pow(2).sum(dim=-1) # (batch, T, slots)
        
        # Scale logically explicitly mapping aliveness filtering logically structurally
        alive_mask = torch.sigmoid(aliveness)
        
        crs_loss = (mse * alive_mask).mean()
        return crs_loss

    def compute_temporal_coherence(self, P: torch.Tensor, aliveness: torch.Tensor) -> torch.Tensor:
        """
        Constrains velocity logic evaluating the structural interpolation matrices matching physics natively extracting arrays smoothing parameters across explicit parameters.
        P: (batch, T, slots, 12, 2)
        aliveness: (batch, T, slots) - raw logits
        """
        # 1. Aliveness Temporal Coherence (first order explicitly tracking diffs filtering flickering)
        alive_probs = torch.sigmoid(aliveness)
        alive_diffs = torch.diff(alive_probs, dim=1) # (batch, T-1, slots)
        loss_alive = alive_diffs.pow(2).mean()
        
        # 2. Acceleration bounds
        # P[:, 2:] - 2*P[:, 1:-1] + P[:, :-2]
        acc_P = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2] # (batch, T-2, slots, 12, 2)
        
        # Square over coordinate blocks
        acc_P_sq = acc_P.pow(2).sum(dim=-1).sum(dim=-1) # (batch, T-2, slots)
        
        # Condition dynamically filtering
        # We match time `t` evaluating structurally mathematically scaling matrices
        alive_mask_inner = alive_probs[:, 1:-1, :]
        
        loss_P = (acc_P_sq * alive_mask_inner).mean()
        
        return loss_alive + loss_P

    def forward(self, model_outputs: tuple, gt_video: torch.Tensor) -> tuple:
        """
        Calculates gradients executing constraints logic seamlessly passing parameters backward updating states directly.
        
        Args:
            model_outputs: (video_tensor, topology_dict)
            gt_video: target ground truth array configuration natively shaped identical mathematically to outputs
            
        Returns:
            loss_total, metrics_dict
        """
        video_tensor, topology = model_outputs
        
        # 1. Rasterized Spatial Render Matrix Bounds Error
        loss_render = self.l1_loss(video_tensor, gt_video)
        
        # 2. Topological Space Physics Penalties
        P = topology['P']                 
        aliveness = topology['aliveness'] 
        colors = topology['colors']       
        
        loss_bcs = self.compute_bcs(P, aliveness)
        loss_crs = self.compute_crs(colors, aliveness)
        loss_temp = self.compute_temporal_coherence(P, aliveness)
        
        # Accumulate metrics
        loss_total = (self.w_render * loss_render) + \
                     (self.w_bcs * loss_bcs) + \
                     (self.w_crs * loss_crs) + \
                     (self.w_temp * loss_temp)
                     
        return loss_total, {
            'bcs': loss_bcs.item(),
            'crs': loss_crs.item(),
            'temp': loss_temp.item(),
            'render': loss_render.item()
        }
