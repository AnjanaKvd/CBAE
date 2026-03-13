# Technical Debt Resolution

After completing Phase 4, the following technical debt items were identified. This plan addresses the actionable ones while documenting items that are deferred by design.

## Debt Items Identified

| # | File | Issue | Severity |
|---|------|-------|----------|
| D1 | [training/loss.py](file:///f:/Github/CBAE/training/loss.py) | [compute_clip_loss](file:///f:/Github/CBAE/training/loss.py#122-133) returns `0.0` — no actual CLIP image encoding | High |
| D2 | [evaluation/benchmark.py](file:///f:/Github/CBAE/evaluation/benchmark.py) | [compute_clip_score](file:///f:/Github/CBAE/evaluation/benchmark.py#121-160) uses `adaptive_avg_pool1d` as a fake projection instead of real CLIP ViT image encoder | High |
| D3 | [models/cbae_model.py](file:///f:/Github/CBAE/models/cbae_model.py) | Nested Python `for t` + `for b` loops over 192 timesteps × batch — extremely slow (~6400s inference) | Medium |
| D4 | [models/encoders.py](file:///f:/Github/CBAE/models/encoders.py) | `WhisperEncoder.encode_audio` uses `[x for x in audio_array]` list comprehension for batch conversion | Low |

## User Review Required

> [!IMPORTANT]
> **D3 (loop vectorization)** is medium-severity because the nested Python loops over 192×batch are the primary cause of the ~6400s inference time. However, fully vectorizing the rasterizer is a significant architectural change to [DiffRasterizer](file:///f:/Github/CBAE/rendering/diff_rasterizer.py#332-377) (which currently processes one slot configuration at a time). I recommend deferring D3 to a dedicated optimization pass and focusing on D1 + D2 now, which are functionally incorrect placeholders.

> [!NOTE]
> **D4 (WhisperEncoder batch hack)** works correctly and is only a minor code style issue. The `[x for x in array]` pattern is required by `WhisperFeatureExtractor`'s API. No fix needed — documenting as acceptable.

## Proposed Changes

### D1: Real CLIP Contrastive Loss

#### [MODIFY] [loss.py](file:///f:/Github/CBAE/training/loss.py)

Replace the [compute_clip_loss](file:///f:/Github/CBAE/training/loss.py#122-133) placeholder with a real implementation that:
1. Samples K representative frames from the video (e.g., every 24th frame = 8 frames from 192)
2. Resizes frames to 224×224 (CLIP ViT input size)
3. Encodes each frame through the CLIP image encoder (`CLIPEncoder.model.encode_image`)
4. Computes mean cosine similarity between frame embeddings and `prompt_emb`
5. Returns `1 - mean_similarity` as the loss (lower loss = better alignment)

The `CBAELossWrapper.__init__` will accept an optional `clip_model` parameter (the shared `open_clip` model from [CLIPEncoder](file:///f:/Github/CBAE/models/encoders.py#6-41)). If not provided, CLIP loss remains 0.0 (backward-compatible).

---

### D2: Real CLIP Score in Evaluation

#### [MODIFY] [benchmark.py](file:///f:/Github/CBAE/evaluation/benchmark.py)

Replace the fake `adaptive_avg_pool1d` projection in [compute_clip_score](file:///f:/Github/CBAE/evaluation/benchmark.py#121-160) with real CLIP ViT image encoding:
1. Sample K frames (same strategy as D1)
2. Resize + normalize frames using CLIP's preprocessing pipeline
3. Encode through `clip_encoder.model.encode_image`
4. Compute cosine similarity with text embedding
5. Return mean similarity

---

### D3: Rasterization Loop Optimization (DEFERRED)

> [!NOTE]
> Deferring to a dedicated optimization pass. The current per-frame loop works correctly but is slow. Optimization would require either batching the [DiffRasterizer](file:///f:/Github/CBAE/rendering/diff_rasterizer.py#332-377) or using `torch.vmap`. Logging as v2 improvement.

---

### D4: WhisperEncoder Batch Conversion (NO ACTION)

The `[x for x in audio_array]` pattern is the correct way to pass batched inputs to `WhisperFeatureExtractor`. No change needed.

---

## Verification Plan

### Automated Tests

**Existing tests that must continue to pass:**
```
python -m pytest -v tests/test_loss.py
```
All 5 existing tests (shape, scalar, finite, non-negative, backward) must still pass.

**New test to add — `tests/test_clip_loss.py`:**
```
python -m pytest -v tests/test_clip_loss.py
```
- `test_clip_loss_returns_scalar`: Verify [compute_clip_loss](file:///f:/Github/CBAE/training/loss.py#122-133) returns a scalar > 0 when given a real CLIP model
- `test_clip_loss_backward_compatible`: Verify [CBAELossWrapper()](file:///f:/Github/CBAE/training/loss.py#7-179) without `clip_model` still returns `clip=0.0`
- `test_clip_loss_gradient_flows`: Verify backward pass works through the CLIP loss

**Benchmark sanity check:**
```
python -m evaluation.benchmark --checkpoint ckpt.pt --prompts "a blue character"
```
Verify [clip_score](file:///f:/Github/CBAE/evaluation/benchmark.py#121-160) is no longer a near-zero fake value but a plausible cosine similarity.
