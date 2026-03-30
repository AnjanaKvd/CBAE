import torch
import torch.nn as nn

class TopologicalEncoder(nn.Module):
    def __init__(self, n_slots=128, latent_dim=512):
        """
        Encodes Frame 0 topology into a global latent distribution (mean, logvar).
        Input is the flat representation of (P, colors, aliveness).
        """
        super().__init__()
        self.n_slots = n_slots
        self.latent_dim = latent_dim
        
        # P: 12x2 = 24. Colors: 3. Aliveness: 1. Total per slot = 28.
        self.slot_dim = 28
        
        self.slot_encoder = nn.Sequential(
            nn.Linear(self.slot_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256)
        )
        
        # We aggregate over slots using a globally aware pool + MLP
        self.global_encoder = nn.Sequential(
            nn.Linear(256 * n_slots, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU()
        )
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, P, colors, alive):
        """
        Args:
            P: (batch, slots, 12, 2)
            colors: (batch, slots, 3) 
            alive: (batch, slots)
        Returns:
            mu, logvar
        """
        batch_size = P.size(0)
        
        # Flatten P per slot
        P_flat = P.view(batch_size, self.n_slots, 24)
        alive_unsqueezed = alive.unsqueeze(-1)
        
        # Concat features per slot: (batch, slots, 28)
        slot_features = torch.cat([P_flat, colors, alive_unsqueezed], dim=-1)
        
        # Encode each slot independently
        encoded_slots = self.slot_encoder(slot_features)  # (batch, slots, 256)
        
        # Flatten across all slots for global context
        encoded_flat = encoded_slots.view(batch_size, -1)  # (batch, slots * 256)
        
        # Global compression
        global_feat = self.global_encoder(encoded_flat)
        
        mu = self.fc_mu(global_feat)
        logvar = self.fc_logvar(global_feat)
        
        return mu, logvar

class TopologicalDecoder(nn.Module):
    def __init__(self, n_slots=128, latent_dim=512, slot_emb_dim=512):
        """
        Decodes a global latent seed z into individual slot_embs.
        """
        super().__init__()
        self.n_slots = n_slots
        self.latent_dim = latent_dim
        self.slot_emb_dim = slot_emb_dim
        
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, n_slots * 256),
            nn.GELU()
        )
        
        # Project each slot's 256-dim feature to the full 512-dim slot_emb
        self.slot_projector = nn.Sequential(
            nn.Linear(256, slot_emb_dim),
            nn.LayerNorm(slot_emb_dim),
            nn.GELU(),
            nn.Linear(slot_emb_dim, slot_emb_dim)
        )

    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            slot_embs: (batch, n_slots, slot_emb_dim)
        """
        batch_size = z.size(0)
        
        # Decode global to slot-wise features
        x = self.decoder_mlp(z)  # (batch, n_slots * 256)
        x = x.view(batch_size, self.n_slots, 256)
        
        # Project to final slot embeddings
        slot_embs = self.slot_projector(x)
        
        return slot_embs

class TopologicalVAE(nn.Module):
    def __init__(self, n_slots=128, latent_dim=512, slot_emb_dim=512):
        """
        Wraps the Encoder and Decoder with the reparameterization trick.
        """
        super().__init__()
        self.encoder = TopologicalEncoder(n_slots, latent_dim)
        self.decoder = TopologicalDecoder(n_slots, latent_dim, slot_emb_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, P, colors, alive):
        mu, logvar = self.encoder(P, colors, alive)
        z = self.reparameterize(mu, logvar)
        slot_embs = self.decoder(z)
        return slot_embs, mu, logvar, z
