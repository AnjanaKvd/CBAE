# training/trainer.py
"""
Multi-stage CBAE trainer following CBAE_TRAINING_GUIDE.md.

Supports staged training progression:
    clean → robustness → bridge → mixed (real data)

Usage:
    python -m training.trainer --stage clean --data_dir data/synthetic/clean/ --epochs 50
    python -m training.trainer --stage robustness --resume checkpoints/stage1/model_epoch_50.pt --epochs 30
    python -m training.trainer --stage mixed --resume ckpt.pt --loss_velocity_weight 0.0 --lr 5e-5
"""

import argparse
import csv
import logging
import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.cbae_model import CBAE_EndToEnd
from training.loss import CBAELossWrapper
from training.config import (
    BATCH_SIZE,
    N_EPOCHS_CLEAN,
    N_EPOCHS_ROBUSTNESS,
    N_EPOCHS_BRIDGE,
    LR_INIT,
    LR_MIN,
    GRAD_CLIP,
    WEIGHT_DECAY,
    DATA_DIR_CLEAN,
    DATA_DIR_ROBUST,
    DATA_DIR_BRIDGE,
    CHECKPOINT_DIR,
    LOG_FILE,
    DEVICE,
    LOSS_WEIGHTS,
)
from data.dataset import CBAEDataset

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("CBAE-trainer")

# ── Stage → default config ─────────────────────────────────────────────────
STAGE_DEFAULTS = {
    "clean":      {"data_dir": DATA_DIR_CLEAN,   "epochs": N_EPOCHS_CLEAN,      "lr": LR_INIT},
    "robustness": {"data_dir": DATA_DIR_ROBUST,  "epochs": N_EPOCHS_ROBUSTNESS, "lr": 1e-4},
    "bridge":     {"data_dir": DATA_DIR_BRIDGE,  "epochs": N_EPOCHS_BRIDGE,     "lr": 1e-4},
    "mixed":      {"data_dir": "data/mixed/",    "epochs": 30,                  "lr": 5e-5},
    "mixed_30":   {"data_dir": "data/mixed/",    "epochs": 20,                  "lr": 2e-5},
    "mixed_50":   {"data_dir": "data/mixed/",    "epochs": 20,                  "lr": 1e-5},
}


# ── CSV Logger ──────────────────────────────────────────────────────────────

class CSVLogger:
    """Append-mode CSV logger for training metrics."""

    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._write_header = not os.path.exists(filepath)

    def log(self, row: dict):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(row)


# ── Training Loop ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: CBAE_EndToEnd,
    dataloader: DataLoader,
    loss_fn: CBAELossWrapper,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prompt: str = "a synthetic animated character",
) -> Dict[str, float]:
    """Run one epoch of training. Returns averaged metrics."""
    model.train()

    epoch_metrics = {
        "total": 0.0, "render": 0.0, "bcs": 0.0,
        "crs": 0.0, "temp": 0.0, "clip": 0.0,
    }
    n_batches = 0

    for audio, gt_video in dataloader:
        audio = audio.to(device)
        gt_video = gt_video.to(device)

        # Forward
        model_outputs = model(prompt, audio)

        # Loss
        loss_total, metrics = loss_fn(model_outputs, gt_video)

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        # Accumulate
        epoch_metrics["total"] += loss_total.item()
        for k in ["render", "bcs", "crs", "temp", "clip"]:
            epoch_metrics[k] += metrics.get(k, 0.0)
        n_batches += 1

    # Average
    if n_batches > 0:
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

    return epoch_metrics


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBAE Multi-Stage Trainer")
    p.add_argument("--stage", type=str, required=True,
                   choices=list(STAGE_DEFAULTS.keys()),
                   help="Training stage (clean/robustness/bridge/mixed)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Data directory (overrides stage default)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs (overrides stage default)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (overrides stage default)")
    p.add_argument("--batch_size", type=int, default=None,
                   help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Checkpoint output directory")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--loss_velocity_weight", type=float, default=None,
                   help="Override velocity loss weight (0.0 for real data)")
    p.add_argument("--render_size", type=int, default=32,
                   help="Render resolution for GT frames (default: 32 for CPU)")
    p.add_argument("--max_frames", type=int, default=4,
                   help="Max frames per sequence (default: 4 for CPU)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage_cfg = STAGE_DEFAULTS[args.stage]

    # Resolve config (CLI overrides → stage defaults → global defaults)
    data_dir = args.data_dir or stage_cfg["data_dir"]
    epochs = args.epochs or stage_cfg["epochs"]
    lr = args.lr or stage_cfg["lr"]
    batch_size = args.batch_size or BATCH_SIZE
    ckpt_dir = args.checkpoint_dir or os.path.join(CHECKPOINT_DIR, args.stage)
    render_size = args.render_size
    max_frames = args.max_frames

    device = torch.device(DEVICE)
    log.info("Device: %s | Stage: %s", device, args.stage)
    log.info("Data: %s | Epochs: %d | LR: %g | Batch: %d",
             data_dir, epochs, lr, batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    log.info("Building CBAE_EndToEnd model …")
    model = CBAE_EndToEnd(
        render_width=render_size,
        render_height=render_size,
        n_steps=max_frames,
    ).to(device)
    
    loss_fn = CBAELossWrapper(
        w_render=LOSS_WEIGHTS['render'],
        w_bcs=LOSS_WEIGHTS['smooth'],
        w_crs=LOSS_WEIGHTS['alive'],
        w_temp=LOSS_WEIGHTS['smooth'], # roughly maps to temp
        w_clip=LOSS_WEIGHTS['semantic'],
        clip_model=model.seq_model.clip.model,
        clip_preprocess=model.seq_model.clip.preprocess,
    ).to(device)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed at epoch %d", start_epoch)
    elif args.resume:
        log.warning("Checkpoint not found: %s — starting fresh", args.resume)

    # ── Optimizer + Scheduler ──────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR_MIN)

    # ── Data ───────────────────────────────────────────────────────────────
    log.info("Loading dataset from %s …", data_dir)
    dataset = CBAEDataset(
        data_dir=data_dir,
        render_size=render_size,
        max_frames=max_frames,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True,
    )
    log.info("Dataset: %d sequences, %d batches/epoch",
             len(dataset), len(dataloader))

    # ── Directories ────────────────────────────────────────────────────────
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── CSV logger ─────────────────────────────────────────────────────────
    csv_path = os.path.join(ckpt_dir, LOG_FILE)
    csv_logger = CSVLogger(csv_path, [
        "epoch", "total", "render", "bcs", "crs", "temp", "clip", "lr", "time_s",
    ])

    # ── Training loop ──────────────────────────────────────────────────────
    log.info("Starting training for %d epoch(s) (from epoch %d).",
             epochs, start_epoch)

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()

        metrics = train_one_epoch(
            model, dataloader, loss_fn, optimizer, device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "Epoch %03d | %.1fs | total=%.4f render=%.4f bcs=%.4f "
            "crs=%.4f temp=%.4f clip=%.4f | lr=%.2e",
            epoch, elapsed,
            metrics["total"], metrics["render"],
            metrics["bcs"], metrics["crs"],
            metrics["temp"], metrics["clip"],
            current_lr,
        )

        # CSV log
        csv_logger.log({
            "epoch": epoch,
            "total": f"{metrics['total']:.6f}",
            "render": f"{metrics['render']:.6f}",
            "bcs": f"{metrics['bcs']:.6f}",
            "crs": f"{metrics['crs']:.6f}",
            "temp": f"{metrics['temp']:.6f}",
            "clip": f"{metrics['clip']:.6f}",
            "lr": f"{current_lr:.2e}",
            "time_s": f"{elapsed:.1f}",
        })

        # Checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
                "stage": args.stage,
            }, ckpt_path)
            log.info("Checkpoint saved → %s", ckpt_path)

    log.info("Training finished. Stage: %s | Final loss: %.4f",
             args.stage, metrics["total"])


if __name__ == "__main__":
    main()
