import torch
import torch.nn as nn
from rendering.diff_rasterizer import bezier_to_polyline_torch



class CBAELossWrapper(nn.Module):
    def __init__(self, w_render=1.0, w_bcs=0.1, w_crs=0.1, w_temp=0.5, w_kl=1e-4, w_topo=10.0):
        """
        Combined loss for CBAE VAE-ODE training: render L1 + BCS + CRS + temporal coherence + KL Divergence + Topology.
        """
        super().__init__()
        self.w_render = w_render
        self.w_bcs = w_bcs
        self.w_crs = w_crs
        self.w_temp = w_temp
        self.w_kl = w_kl
        self.w_topo = w_topo

        self.l1_loss = nn.L1Loss()
        
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence loss for the VAE to regularize the latent space.
        D_KL(N(mu, sigma) || N(0, 1))
        """
        if mu is None or logvar is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # KL loss: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def compute_bcs(self, P: torch.Tensor, aliveness: torch.Tensor) -> torch.Tensor:
        """
        Computes the Boundary Constraint Score mapping curvature smoothly filtering extreme changes temporally.
        P: (batch, T, slots, 12, 2)
        aliveness: (batch, T, slots) - raw logits
        """
        batch, T, slots, _, _ = P.shape
        # Flatten batch, time, and slots into a single dimension for bezier_to_polyline_torch
        # which expects (n_slots, 12, 2) — a 3D tensor
        P_flat = P.reshape(batch * T * slots, 12, 2)
        
        # We sample 30 discrete segments mapping curve trajectories explicitly approximating curvature derivatives mathematically
        polylines_flat = bezier_to_polyline_torch(P_flat, n_samples=30) # (batch*T*slots, 30, 2)
        polylines = polylines_flat.reshape(batch * T, slots, 30, 2)
        
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
        K_time = K.reshape(batch, T, slots)
        
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

    def forward(self, model_outputs: tuple, gt_video: torch.Tensor, gt_topology_0: dict = None) -> tuple:
        """
        Calculates gradients executing constraints logic seamlessly passing parameters backward updating states directly.
        
        Args:
            model_outputs: (video_tensor, topology_dict)
            gt_video: target ground truth array configuration natively shaped identical mathematically to outputs
            gt_topology_0: target initial frame topology dictionary. If passed, guides VAE encoder reconstruction.
            
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
        
        # 3. VAE Latent Regularization
        loss_kl = self.compute_kl_loss(topology.get('mu'), topology.get('logvar'))
        
        # 4. Topological Initial State Reconstruction (fixes zero-overlap gradient collapse)
        loss_topo = torch.tensor(0.0, device=video_tensor.device)
        if gt_topology_0 is not None:
            P_pred_0 = P[:, 0] # (batch, 128, 12, 2)
            loss_P = nn.functional.mse_loss(P_pred_0, gt_topology_0['P'].to(video_tensor.device))
            loss_c = nn.functional.mse_loss(colors, gt_topology_0['colors'].to(video_tensor.device))
            loss_a = nn.functional.mse_loss(aliveness[:, 0], gt_topology_0['alive'].to(video_tensor.device))
            loss_topo = loss_P + loss_c + loss_a
            
        # Accumulate metrics
        loss_total = (self.w_render * loss_render) + \
                     (self.w_bcs * loss_bcs) + \
                     (self.w_crs * loss_crs) + \
                     (self.w_temp * loss_temp) + \
                     (self.w_kl * loss_kl) + \
                     (self.w_topo * loss_topo)
                     
        return loss_total, {
            'bcs': loss_bcs.item(),
            'crs': loss_crs.item(),
            'temp': loss_temp.item(),
            'render': loss_render.item(),
            'kl': loss_kl.item(),
            'topo': loss_topo.item()
        }
