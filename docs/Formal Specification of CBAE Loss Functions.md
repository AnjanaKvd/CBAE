# Formal Specification of CBAE Loss Functions (`training/loss.py`)

This document provides the formal mathematical and implementation specifications for the loss functions of the CBAE model, specifically addressing the Boundary Constraint Score (BCS), Content Retention Score (CRS), Temporal Coherence, and the bridging of Phase 1 topological metrics with rasterized outputs.

## User Review Required
> [!IMPORTANT]
> **Model Return Signature Adjustment:** Currently, `models/cbae_model.py`'s `forward` method only returns the `video_tensor` (the rasterized frames). To compute topological metrics like BCS and Temporal Coherence efficiently without breaking differentiability, the training unroll needs direct access to `trajectory` (containing P and aliveness) and `colors`. I propose updating `CBAE_EndToEnd.forward` to return a tuple `(video_tensor, topological_state)` where `topological_state` is a dictionary containing `trajectory`, `colors`, and base structural parameters (`alpha`, `z`, `csg`). Please confirm if this signature update is approved for Task 4.1.2.
>
> **CRS Definition vs Model Output:** The current `ColorPredictionMLP` predicts static colors per slot (`colors[b, i]`). The standard CRS (variance across frames) will trivially be 0 unless the colors become time-dependent. The formulation below supports time-dependent colors natively in case the design evolves.

## 1. Boundary Constraint Score (BCS) Loss

**Objective:** Enforce geometric smoothness of the boundaries over time, reducing flickering or erratic contour changes.

**Mathematical Formulation:**
Let $P_i(t)$ be the set of control points for the $i$-th active shape at frame $t$. The boundary curve $B_i(u, t)$ is defined by the cubic Bézier function over $u \in [0, 1]$.
The curvature $\kappa_i(u, t)$ of the curve is given by:
$$ \kappa_i(u, t) = \frac{| B_i'(u, t) \times B_i''(u, t) |}{| B_i'(u, t) |^3} $$
where derivatives are w.r.t $u$.
The mean absolute curvature for the shape is $K_i(t) = \int_0^1 \kappa_i(u, t) du$.
The Boundary Constraint Score (BCS) penalty is the absolute temporal difference in mean curvature:
$$ \mathcal{L}_{BCS} = \frac{1}{N_{active} (T-1)} \sum_{t=1}^{T-1} \sum_{i \in \text{active}} \Big| K_i(t+1) - K_i(t) \Big| $$

**Implementation Specification (PyTorch):**
1. **Differentiable Curvature:** Instead of analytical continuous integration, sample $N_{samples}$ points along the Bézier curve using batched matrix multiplications for the parameter $u$.
2. Compute $B'$ and $B''$ analytically from the control points (which is a linear operation and fully differentiable).
3. Compute $\kappa$ for each sample point and average them to get $K_i(t)$.
4. Compute the L1 difference $| K_i(t+1) - K_i(t) |$.
5. Modulate the contribution of shape $i$ by its smoothed aliveness $\sigma(a_i(t))$.

## 2. Content Retention Score (CRS) Loss

**Objective:** Ensure color region stability and consistency throughout the sequence for identical topological elements.

**Mathematical Formulation:**
Let $c_i(t)$ be the RGB color for shape $i$ at frame $t$. The mean color over time is $\bar{c}_i = \frac{1}{T} \sum_{t} c_i(t)$.
The content retention score computes the variance (fluctuation) of the color:
$$ \mathcal{L}_{CRS} = \frac{1}{N_{active} T} \sum_{i \in \text{active}} \sum_{t=1}^{T} \sigma(a_i(t)) || c_i(t) - \bar{c}_i ||_2^2 $$
*(Note: If the color prediction remains static across time $c_i(t) = c_i$, this loss naturally falls to 0. It serves as a guardrail against instability in future iterations where color is dynamically conditioned on audio/time).*

**Implementation Specification (PyTorch):**
1. If $c$ varies over time, compute the temporal mean `c_mean = c.mean(dim=1, keepdim=True)`.
2. Compute the MSE `(c - c_mean).pow(2)`.
3. Weight the errors by the sigmoid of the aliveness logit.

## 3. Temporal Coherence Loss (Derivatives over Time)

**Objective:** Regularize the gradients (velocities and accelerations) of both control points and aliveness states to mandate smooth interpolations.

**Mathematical Formulation:**
The objective consists of a first-order penalty on aliveness changes (to prevent flickering) and a second-order penalty (acceleration) on control points (to enforce momentum-like smoothness).
$$ \mathcal{L}_{temp\_alive} = \frac{1}{(T-1) N_{slot}} \sum_t \sum_i \Big( \sigma(a_i(t+1)) - \sigma(a_i(t)) \Big)^2 $$
$$ \mathcal{L}_{temp\_P} = \frac{1}{(T-2) N_{slot}} \sum_t \sum_i \sigma(a_i(t)) \Big|\Big| P_i(t+1) - 2P_i(t) + P_i(t-1) \Big|\Big|_2^2 $$

**Implementation Specification (PyTorch):**
1. **Aliveness Gradient:** Compute `diff_a = torch.diff(torch.sigmoid(aliveness), dim=1)` (along the time dimension). The loss is `diff_a.pow(2).mean()`.
2. **Control Point Acceleration:** Compute `acc_P = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]`. The loss is `acc_P.pow(2).sum(dim=-1).mean()`.

## 4. Integration Logic: Bridging Topological Metrics with Rasterized Outputs

To successfully train the end-to-end model, the overall scalar loss is a linear combination of spatial/raster metrics and the topological metrics outlined above.

**Bridging Formulation:**
$$ \mathcal{L}_{total} = \lambda_{render} \mathcal{L}_{render}(I_{pred}, I_{gt}) + \lambda_{BCS} \mathcal{L}_{BCS} + \lambda_{CRS} \mathcal{L}_{CRS} + \lambda_{temp} ( \mathcal{L}_{temp\_alive} + \mathcal{L}_{temp\_P} ) $$

**Integration Architecture Flow:**
1. **Rasterized Loss (`L_render`):**
   - Executed against the `video_tensor` generated by the differentiable rasterizer. Typically LPIPS + MSE on pixel space.
2. **Topological Loss (`L_BCS`, `L_CRS`, `L_temp`):**
   - Executed against the intermediate explicit graph topologies: `P_t` (control points), `aliveness_t` (active logit), and `colors` arrays predicted during the Neural ODE and MLP stages.
3. **Training Iteration Bridge:**
   - The loss module takes the tuple `(video_tensor, topological_state)` and the `ground_truth_video`.
   - The topological state gradients automatically flow through the control points (modifying Neural ODE dynamics) and aliveness logit.
   - The rasterized gradients flow from the pixels through the DiffVG/Soft-Ras backend, distributing analytical gradients into the topological states concurrently. PyTorch natively accumulates these gradients (`topological_grad + raster_grad`) on the `trajectory` variables, enabling simultaneous optimization of structural integrity (Phase 1 metrics) and visual accuracy (raster outputs).

## Proposed `training/loss.py` Skeleton

```python
import torch
import torch.nn as nn

class CBAELoss(nn.Module):
    def __init__(self, w_render=1.0, w_bcs=0.1, w_crs=0.1, w_temp=0.5):
        super().__init__()
        self.w_render = w_render
        self.w_bcs = w_bcs
        self.w_crs = w_crs
        self.w_temp = w_temp
        # self.lpips = LPIPS(net='alex')

    def compute_bcs(self, P, aliveness):
        # Implementation of discrete Bézier curvature variance
        pass

    def compute_crs(self, colors, aliveness):
        # Implementation of color variance over time, if colors are batched sequentially
        pass

    def compute_temporal_coherence(self, P, aliveness):
        # Implementation of 2nd order motion derivatives and 1st order aliveness diff
        pass

    def forward(self, model_outputs, gt_video):
        video_tensor, topology = model_outputs
        
        # 1. Rasterized Pixel Space Loss
        # loss_render = self.lpips(video_tensor, gt_video)
        loss_render = torch.tensor(0.0)
        
        # 2. Topological Space Losses
        P = topology['P']                 # (batch, T, slots, 12, 2)
        aliveness = topology['aliveness'] # (batch, T, slots)
        colors = topology['colors']       # (batch, slots, 3) or (batch, T, slots, 3)
        
        loss_bcs = self.compute_bcs(P, aliveness)
        loss_crs = self.compute_crs(colors, aliveness)
        loss_temp = self.compute_temporal_coherence(P, aliveness)
        
        # 3. Accumulated Objective
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
```
