"""
training/train.py — CBAE Training Entry Point

Implements the outer training loop that runs the end-to-end CBAE model 
and propagates gradients through Physics→ODE→Rasterizer pipeline.

Usage:
    python training/train.py [--epochs N] [--batch-size B] [--lr LR] [--dry-run]
"""

import argparse
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.cbae_model import CBAE_EndToEnd
from training.loss import CBAELossWrapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("CBAE-train")


# ---------------------------------------------------------------------------
# Placeholder / dummy dataloader
# ---------------------------------------------------------------------------

def build_dummy_dataloader(batch_size: int, n_batches: int = 8, n_steps: int = 192, render_size: int = 64) -> DataLoader:
    """
    Returns a minimal DataLoader emitting dummy tensors so the training loop
    can be exercised before a real HDF5 dataset is wired up (Task 4.x).

    Each item is a tuple:
        (audio, gt_video)
        audio    : (8*16000,)  — 8 s of silence
        gt_video : (192, H, W, 3) — random pixels
    """
    H = W = render_size
    n = batch_size * n_batches
    T = n_steps if n_steps else 192

    audio_data    = torch.zeros(n, 8 * 16_000)
    gt_video_data = torch.rand(n, T, H, W, 3)

    ds = TensorDataset(audio_data, gt_video_data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Single-epoch training function
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CBAE_EndToEnd,
    dataloader: DataLoader,
    loss_fn: CBAELossWrapper,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dry_run: bool = False,
    prompt: str = "a synthetic animated character",
) -> Dict[str, float]:
    """
    Iterates over batches, runs the CBAE_EndToEnd forward pass, computes the
    combined loss via CBAELossWrapper, calls backward, clips gradients, and
    steps the optimizer.

    Args:
        model:      Instantiated CBAE_EndToEnd module.
        dataloader: Yields (audio, gt_video) batches.
        loss_fn:    CBAELossWrapper instance.
        optimizer:  AdamW or compatible optimizer.
        device:     torch.device.
        dry_run:    If True, stop after the first batch.
        prompt:     Text prompt shared across the batch (simplification for
                    the synthetic-data phase; replace with per-sample prompts
                    when real captions are added).

    Returns:
        Dict of averaged per-metric losses for the epoch.
    """
    model.train()

    epoch_metrics: Dict[str, float] = {
        "total": 0.0,
        "render": 0.0,
        "bcs": 0.0,
        "crs": 0.0,
        "temp": 0.0,
        "clip": 0.0,
    }
    n_batches = 0

    for audio, gt_video in dataloader:
        audio    = audio.to(device)       # (B, 8*16000)
        gt_video = gt_video.to(device)    # (B, 192, H, W, 3)

        # ── Forward ────────────────────────────────────────────────────────
        model_outputs = model(prompt, audio)
        # model_outputs = (video_tensor, topology_dict)

        # ── Loss ───────────────────────────────────────────────────────────
        loss_total, metrics = loss_fn(model_outputs, gt_video)

        # ── Backward ───────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss_total.backward()

        # Gradient clipping (prevents exploding gradients in Neural ODE)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Accumulate ─────────────────────────────────────────────────────
        epoch_metrics["total"]  += loss_total.item()
        epoch_metrics["render"] += metrics["render"]
        epoch_metrics["bcs"]    += metrics["bcs"]
        epoch_metrics["crs"]    += metrics["crs"]
        epoch_metrics["temp"]   += metrics["temp"]
        epoch_metrics["clip"]   += metrics["clip"]
        n_batches += 1

        log.debug(
            "  batch %d | total=%.4f render=%.4f bcs=%.4f crs=%.4f "
            "temp=%.4f clip=%.4f",
            n_batches, loss_total.item(),
            metrics["render"], metrics["bcs"],
            metrics["crs"], metrics["temp"], metrics["clip"],
        )

        if dry_run:
            log.info("--dry-run: stopping after 1 batch.")
            break

    # Average over batches
    if n_batches > 0:
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

    return epoch_metrics


# ---------------------------------------------------------------------------
# main / CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CBAE End-to-End Training Script"
    )
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int,   default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--n-batches",  type=int,   default=8,
                        help="Dummy-loader batches per epoch (default: 8)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoints (default: checkpoints/)")
    parser.add_argument("--save-every", type=int,   default=5,
                        help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 1 batch only, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Instantiate components ─────────────────────────────────────────────
    log.info("Building CBAE_EndToEnd model …")
    # Use a smaller canvas on CPU to avoid OOM with full gradient tracking;
    # increase to 128+ when training on GPU.
    render_size = 64 if device.type == "cuda" else 32
    n_steps = 192 if device.type == "cuda" else 4
    model   = CBAE_EndToEnd(render_width=render_size, render_height=render_size, n_steps=n_steps).to(device)
    loss_fn = CBAELossWrapper().to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Checkpoint directory ───────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building dummy dataloader (replace with real dataset for full training)…")
    # Limit batch size on CPU; full batch sizes require a GPU.
    effective_batch = args.batch_size if device.type == "cuda" else 1
    dataloader = build_dummy_dataloader(effective_batch, n_batches=args.n_batches, n_steps=n_steps, render_size=render_size)

    # ── Training loop ─────────────────────────────────────────────────────
    max_epochs = 1 if args.dry_run else args.epochs
    log.info("Starting training for %d epoch(s).", max_epochs)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        metrics = train_one_epoch(
            model, dataloader, loss_fn, optimizer, device,
            dry_run=args.dry_run,
        )

        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "Epoch %03d/%03d | %.1fs | "
            "total=%.4f render=%.4f bcs=%.4f crs=%.4f temp=%.4f clip=%.4f",
            epoch, max_epochs, elapsed,
            metrics["total"], metrics["render"],
            metrics["bcs"],   metrics["crs"],
            metrics["temp"],  metrics["clip"],
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % args.save_every == 0 or args.dry_run:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"cbae_epoch{epoch:04d}.pt"
            )
            torch.save(
                {
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "metrics":    metrics,
                },
                ckpt_path,
            )
            log.info("Checkpoint saved → %s", ckpt_path)

        if args.dry_run:
            log.info("--dry-run complete. Exiting.")
            break

    log.info("Training finished.")


if __name__ == "__main__":
    main()
