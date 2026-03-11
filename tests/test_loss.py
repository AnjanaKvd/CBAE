import torch
from models.cbae_model import CBAE_EndToEnd
from training.loss import CBAELossWrapper

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBAE_EndToEnd(render_width=32, render_height=32).to(device)
    loss_fn = CBAELossWrapper().to(device)
    
    # Mock inputs
    batch_size = 2
    prompt = ["A gentle blue character breathing", "A red block jumping"]
    audio = torch.randn(batch_size, 16000 * 8).to(device) # 8 seconds of audio
    gt_video_mock = torch.rand(batch_size, 192, 32, 32, 3).to(device)
    
    # Execute forward pass natively mapping sequences properly
    video_tensor, topology = model(prompt, audio)
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Check physics bounds tracking
    loss_total, metrics = loss_fn((video_tensor, topology), gt_video_mock)
    
    print(f"Total Loss: {loss_total.item():.4f}")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
    
    loss_total.backward()
    print("Backward pass validated smoothly!")
