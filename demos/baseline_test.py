import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from generation.synthetic import generate_base_character
from rendering.rasterizer import rasterize
from rendering.diff_rasterizer import DiffRasterizer

def run_baseline_optimization(output_dir="output/baseline_test", steps=2000, lr=0.01):
    """
    Directly optimizes the control points and colors of a random CRFTensor to match 
    a target rendered image via differentiable rasterization.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu') # Spec insists on CPU constraint
    
    # ---------------------------------------------------------
    # 1. Generate Target Image using Non-Differentiable Rasterizer
    # ---------------------------------------------------------
    print("Generating target ground truth...")
    target_crf = generate_base_character(style='robe')
    target_img_np = rasterize(target_crf, width=512, height=512)
    
    # Save target for reference
    Image.fromarray(target_img_np).save(os.path.join(output_dir, "target.png"))
    
    # Convert numpy target to torch [0, 1] tensor (H, W, 3)
    target_tensor = torch.from_numpy(target_img_np).float() / 255.0
    target_tensor = target_tensor.to(device)

    # ---------------------------------------------------------
    # 2. Initialize Random Starting State
    # ---------------------------------------------------------
    print("Initializing random starting state parameters...")
    
    # Randomly initialize properties matching the active slots of the target exactly
    active_mask = target_crf.active_slots()
    
    # Starting values (randomized within constrained space for control points and color)
    P_init = torch.rand((128, 12, 2), device=device)
    c_init = torch.rand((128, 3), device=device)
    
    # Other properties remain fixed (alpha=1 for active, alive, z, csg imported natively)
    alpha_tensor = torch.from_numpy(target_crf.alpha.astype(np.float32)).to(device)
    alive_tensor = torch.from_numpy(target_crf.alive.astype(np.float32)).to(device)
    z_tensor = torch.from_numpy(target_crf.z.astype(np.int32)).to(device)
    csg_tensor = torch.from_numpy(target_crf.csg.astype(bool)).to(device)
    
    # Overwrite the actual optimized variables so only active shapes matter properly
    P = P_init.clone().requires_grad_(True)
    c = c_init.clone().requires_grad_(True)

    # Setup Differentiable Rasterizer natively handling SDF poly approximation gracefully
    diff_ras = DiffRasterizer(use_diffvg=False, fallback_softness=0.005).to(device)
    
    # Save initial state view
    with torch.no_grad():
        initial_render = diff_ras(P, c, alpha_tensor, alive_tensor, z_tensor, csg_tensor)
        init_img = (initial_render.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(init_img).save(os.path.join(output_dir, "step_0000.png"))
        
    optimizer = optim.Adam([P, c], lr=lr)

    # ---------------------------------------------------------
    # 3. Optimization Loop
    # ---------------------------------------------------------
    print(f"Starting Adam optimization loop over {steps} steps...")
    loss_history = []
    
    # Only optimizing MSE for now since LPIPS requires external library setup (Task spec allows MSE fallback)
    # L1 and L2 combination (smooth L1) often gradients better for shapes natively
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        
        # Forward render mapping
        rendered = diff_ras(P, c, alpha_tensor, alive_tensor, z_tensor, csg_tensor)
        
        # Loss (MSE) - comparing (H, W, 3) pixel matrices
        loss = F.mse_loss(rendered, target_tensor)
        
        loss.backward()
        
        # Optional: Print grad norms occasionally to ensure they are finite and non-zero
        if step == 1 or step % 500 == 0:
            p_grad_norm = P.grad.norm().item() if P.grad is not None else 0.0
            print(f"Step {step:04d} | Loss: {loss.item():.6f} | P.grad_norm: {p_grad_norm:.6f}")
        
        optimizer.step()
        
        # ---------------------------------------------------------
        # Strict constraints: Keep control points and colors valid
        # ---------------------------------------------------------
        with torch.no_grad():
            P.data.clamp_(0.0, 1.0)
            c.data.clamp_(0.0, 1.0)
            
        loss_history.append(loss.item())
        
        # Periodically save images
        if step % 200 == 0:
            with torch.no_grad():
                out_img = (rendered.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(out_img).save(os.path.join(output_dir, f"step_{step:04d}.png"))
                
    # ---------------------------------------------------------
    # 4. Save Loss Plot Log
    # ---------------------------------------------------------
    print("Optimization finished. Generating loss curve plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="MSE Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Baseline Differentiable Rendering Optimization")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CBAE differentiable baseline optimization.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument("--out", type=str, default="output/baseline_test", help="Output directory")
    args = parser.parse_args()
    
    run_baseline_optimization(output_dir=args.out, steps=args.steps, lr=args.lr)
