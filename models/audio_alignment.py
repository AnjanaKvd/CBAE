import torch
import torch.nn as nn

class AudioAlignmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1D Convolution downsampling Whisper's ~50Hz output to our 24Hz ODE steps
        self.conv = nn.Conv1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
    def forward(self, raw_audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Aligns the temporal dimension of the raw audio embedding via convolution.
        
        Args:
            raw_audio_emb: Float tensor of shape (batch_size, T_audio, 384)
            
        Returns:
            aligned_emb: Float tensor of shape (batch, T_frames, 384) aligned to 24fps
        """
        # Linear layers (like encoder outputs) use (Batch, Seq, Features)
        # Conv1d expects (Batch, Channels, Seq).
        # We treat 384 as Channels and T_audio as Seq.
        x = raw_audio_emb.transpose(1, 2)
        
        x = self.conv(x)
        
        # Transpose back to (Batch, T_frames, 384)
        aligned_emb = x.transpose(1, 2)
        
        return aligned_emb
        
    def get_frame_embedding(self, aligned_emb: torch.Tensor, t: int) -> torch.Tensor:
        """
        Extracts the specific audio embedding map for frame index t.
        
        Args:
            aligned_emb: The full pre-calculated sequence from forward()
            t: The frame integer index to query
            
        Returns:
            (batch_size, 384) tensor for that specific timestep.
        """
        # Standard clamp to safely avoid bounds errors on edge sequences
        # e.g. if the audio cuts off before 8 seconds
        max_idx = max(0, aligned_emb.size(1) - 1)
        safe_t = min(t, max_idx)
        
        return aligned_emb[:, safe_t, :]
