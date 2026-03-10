import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from core.crf_tensor import CRFTensor
from core.slot_blocks import get_delta_max
from core.constants import N_SLOTS, N_CTRL_PTS

class TemplateLibrary:
    def __init__(self, templates_dir: str = "data/templates"):
        """
        Loads pre-computed base poses and their CLIP embeddings.
        Falls back to generating a default dynamic template if no files are found.
        """
        self.templates_dir = templates_dir
        self.base_poses_dir = os.path.join(templates_dir, "base_poses")
        self.embeddings_path = os.path.join(templates_dir, "embeddings.npy")
        
        self.templates = []
        self.embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_library()
        
    def _load_library(self):
        # Graceful fallback if files don't exist yet before Task 4 script creates them
        if not os.path.exists(self.embeddings_path) or not os.path.exists(self.base_poses_dir):
            print("TemplateLibrary: No templates found. Generating ad-hoc fallback.")
            from generation.synthetic import generate_base_character
            self.templates.append(generate_base_character(style='robe'))
            # Dummy normalized embedding matching CLIP's 512-dim
            dummy_emb = torch.randn(1, 512)
            dummy_emb = dummy_emb / dummy_emb.norm(dim=-1, keepdim=True)
            self.embeddings = dummy_emb.to(self.device)
            return

        # Load embeddings
        emb_np = np.load(self.embeddings_path)
        self.embeddings = torch.from_numpy(emb_np).float().to(self.device)
        
        # Load templates (assuming alphabetical/numerical parity with embeddings array)
        files = sorted(os.listdir(self.base_poses_dir))
        for f in files:
            if f.endswith('.json'):
                path = os.path.join(self.base_poses_dir, f)
                with open(path, 'r') as file:
                    data = json.load(file)
                    crf = CRFTensor()
                    crf.from_json(data)
                    self.templates.append(crf)
                    
        assert len(self.templates) == self.embeddings.size(0), "Templates & embeddings size mismatch!"
        
    def retrieve(self, text_emb: torch.Tensor) -> Tuple[CRFTensor, float]:
        """
        Finds the nearest template matching the provided text embedding.
        
        Args:
            text_emb: (512,) or (1, 512) tensor
            
        Returns:
            Nearest CRFTensor copy and the float similarity score.
        """
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
            
        # Cosine similarity: (1, 512) @ (N, 512)^T -> (1, N)
        # Assumes both are unit normalized
        text_emb = text_emb.to(self.device)
        sim = F.cosine_similarity(text_emb, self.embeddings, dim=-1)
        
        best_idx = torch.argmax(sim).item()
        best_score = sim[best_idx].item()
        
        return self.templates[best_idx].clone(), best_score


class DeformationMLP(nn.Module):
    def __init__(self, text_dim=512, slot_dim=64, hidden1=256, hidden2=128, ctrl_pts=N_CTRL_PTS):
        super().__init__()
        in_dim = text_dim + slot_dim
        out_dim = ctrl_pts * 2  # 12 control points * 2 coords = 24
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, out_dim)
        )
        
        # Cache max deltas for scaling outputs per slot
        delta_max_cache = torch.zeros(N_SLOTS)
        for i in range(N_SLOTS):
            delta_max_cache[i] = get_delta_max(i)
        self.register_buffer('delta_max', delta_max_cache)
        
    def forward(self, text_emb: torch.Tensor, slot_embs: torch.Tensor) -> torch.Tensor:
        """
        Computes the geometrical deltas mapping shapes for initialization.
        
        Args:
            text_emb: (batch_size, 512)
            slot_embs: (batch_size, N_SLOTS, 64) conditional embeddings from SlotConditioner
            
        Returns:
            delta_P: (batch_size, N_SLOTS, 12, 2)
        """
        batch_size = text_emb.size(0)
        
        # Broadcast text_emb: (batch_size, 1, 512) -> (batch_size, N_SLOTS, 512)
        text_expanded = text_emb.unsqueeze(1).expand(-1, N_SLOTS, -1)
        
        # Concat inputs -> (batch_size, N_SLOTS, 576)
        x = torch.cat([text_expanded, slot_embs], dim=-1)
        
        # MLP output -> (batch_size, N_SLOTS, 24)
        raw_out = self.net(x)
        
        # We clamp via Tanh to strictly bound standard deviation outputs into physical bounds
        # raw_out is clamped to [-1, 1], then multiplied by physical constraint boundaries
        # Reshape delta_max: (N_SLOTS,) -> (1, N_SLOTS, 1)
        d_max = self.delta_max.unsqueeze(0).unsqueeze(-1)
        
        scaled_out = torch.tanh(raw_out) * d_max
        
        # Reshape exactly to (batch_size, N_SLOTS, 12, 2) coordinates
        delta_P = scaled_out.view(batch_size, N_SLOTS, N_CTRL_PTS, 2)
        
        return delta_P


def initialize_crf(template_library: TemplateLibrary, 
                   deformation_mlp: DeformationMLP, 
                   text_emb: torch.Tensor, 
                   slot_embeddings: torch.Tensor) -> CRFTensor:
    """
    Creates the starting CRFTensor for inference integrating templates and MLP predictions.
    Currently operates on batch_size=1 semantics for explicit single sequence generation.
    """
    # 1. Retrieve the nearest semantic template shape state
    template_crf, score = template_library.retrieve(text_emb)
    
    # Ensure batched format for MLP
    if text_emb.dim() == 1:
        text_emb = text_emb.unsqueeze(0)
    if slot_embeddings.dim() == 2:
        slot_embeddings = slot_embeddings.unsqueeze(0)
        
    device = next(deformation_mlp.parameters()).device
    text_emb = text_emb.to(device)
    slot_embeddings = slot_embeddings.to(device)
    
    # 2. Compute explicit physics bounded control point deltas
    with torch.no_grad():
        delta_P_batch = deformation_mlp(text_emb, slot_embeddings)
        
    # Extract element (assuming batch 1 inference workflow)
    delta_P = delta_P_batch[0].cpu().numpy()
    
    # 3. Apply only to actively marked shapes maintaining zeroed properties seamlessly everywhere else
    active_slots = template_crf.active_slots()
    
    for slot_idx in active_slots:
        template_crf.P[slot_idx] += delta_P[slot_idx]
        
    # Enforce strictly valid canvas constraints avoiding topological intersections
    template_crf.P = np.clip(template_crf.P, 0.0, 1.0).astype(np.float16)
    
    return template_crf
