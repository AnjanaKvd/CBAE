import torch
import torch.nn as nn
from torchdiffeq import odeint

from models.encoders import CLIPEncoder, WhisperEncoder
from models.audio_alignment import AudioAlignmentLayer
from models.slot_conditioning import SlotConditioner, SlotEmbeddingTable
from models.deformation_mlp import TemplateLibrary, DeformationMLP, initialize_crf
from models.aliveness import AlivenessMLP, InitialStateCombiner
from models.neural_ode import ODEFx

class SequenceModel(nn.Module):
    """
    Top-level PyTorch Module assembling the complete CBAE Neural ODE Sequence pipeline.
    """
    def __init__(self, n_steps=192):
        super().__init__()
        self.n_steps = n_steps
        # 1. Encoders
        self.clip = CLIPEncoder()
        self.whisper = WhisperEncoder()
        
        # 2. Audio Alignment
        self.audio_align = AudioAlignmentLayer()
        
        # 3. Slot Conditioning
        self.slot_embedding_table = SlotEmbeddingTable()
        self.slot_conditioner = SlotConditioner()
        
        # 4. Deformation & Initial State
        self.template_lib = TemplateLibrary()
        self.deformation_mlp = DeformationMLP()
        self.aliveness_mlp = AlivenessMLP()
        self.state_combiner = InitialStateCombiner()
        
        # 5. Neural ODE
        self.ode_fx = ODEFx()

    def forward(self, prompt: str, audio: torch.Tensor) -> torch.Tensor:
        """
        Executes the full forward pass from conditionals mapping initial conditions natively 
        into continuous ODE sequences.
        
        Args:
            prompt: Text describing layout/character parameters
            audio: Unprocessed raw audio array map correctly scaling matching whispers sample rate
            
        Returns:
            trajectory: Torch tensor shaped (192, batch_size, 3200) representing 192 temporal frames
        """
        device = next(self.ode_fx.parameters()).device
        
        # Normalize prompt to a list matching the batch size
        batch_size = audio.size(0)
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif len(prompt) == 1 and batch_size > 1:
            prompt = prompt * batch_size
        
        # 1. Encode Text & Audio
        text_emb = self.clip.encode_text(prompt).to(device)
        audio_raw_emb = self.whisper.encode_audio(audio)
        audio_raw_emb = audio_raw_emb.to(device)
        
        # 2. Extract 24fps aligned Audio features
        audio_aligned = self.audio_align(audio_raw_emb)
        
        # 3. Get conditional slot embeddings (via SlotConditioner)
        # Assuming frame 0 defines explicit start state logic dependencies
        audio_0 = self.audio_align.get_frame_embedding(audio_aligned, 0)
        slot_embs = self.slot_conditioner(text_emb, audio_0)
        
        # 4. Get ΔP_i (via DeformationMLP) -> generating base geometry CRFTensor
        base_crf = initialize_crf(
            self.template_lib,
            self.deformation_mlp,
            text_emb,
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
        return trajectory, slot_embs, text_emb, base_crf
