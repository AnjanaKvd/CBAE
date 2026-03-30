# Architectural Pivot: Unconditional Topological Video Generator

The user requested an unconditional generative model that learns the physical dynamics of animation directly from topological data (without text or audio prompts). 

## New Architecture Overview: The CBAE VAE-ODE

We will pivot the model from a conditional feed-forward network (Text/Audio-to-Video) to a **Variational Auto-Encoder + Neural ODE** (Seed-to-Video).

### 1. The Encoder: `q(z | Frame 0)`
During training, we need to map the first frame of the animation into a latent mathematical space so the model knows *what* it is animating without text.
- Input: `Frame 0` topology (Control Points, Colors, Aliveness)
- Network: A lightweight 1D Convolutional or MLP network.
- Output: Latent mean $\mu$ and variance $\sigma$.
- Sample: $z \sim \mathcal{N}(\mu, \sigma)$

### 2. The Decoder (Initial State Generator): `Frame 0 = f(z)`
- Input: Latent vector $z$ (sampled from encoder during training, or purely random noise during generation).
- Output: `slot_embs` (the internal mathematical representation of the shapes and colors).
- This replaces the `CLIPEncoder` and `SlotConditioner`. 

### 3. The Dynamics Engine ( Neural ODE )
- Remains largely exactly the same!
- It takes the generated `slot_embs` and unrolls them forward in time using standard physics laws learned from the dataset.

### 4. Training Loss Updates
- We remove the `clip_loss` entirely.
- We add a `KL-Divergence Loss` to force the latent space $z$ to be organized logically (like a standard Gaussian distribution). This prevents the mode-collapse mathematically.
- We keep the `render` L1 loss and the topological `bcs`/`crs` constraints.

---

## Action Plan

### Task 1: Clean Up `models/sequence.py`
- [MODIFY] Remove `CLIPEncoder` and `WhisperEncoder`.
- [MODIFY] Remove `AudioAlignmentLayer`.
- [MODIFY] Replace input arguments `(prompt, audio)` with `(gt_topology_0)` during training, and `(z_seed)` during generation.

### Task 2: Implement Topological VAE
- [NEW] `models/vae.py`: Implement the Encoder (mapping `P` and `colors` to $\mu$, $\sigma$) and the Decoder (mapping $z$ to `slot_embs` and `text_emb_equivalent`).

### Task 3: Adjust `training/trainer.py` and `training/loss.py`
- [MODIFY] Modify the data loader to pass the ground-truth topology (which we can get directly from the HDF5 sequences, bypassing the need to rasterise GT frames every time for the initial state).
- [MODIFY] Replace `compute_clip_loss` with `compute_kl_loss`.

### Task 4: Testing & Demo Generation
- [MODIFY] Update `scripts/generate_demo.py` to sample random `z` vectors instead of passing text prompts.
