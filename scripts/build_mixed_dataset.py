#!/usr/bin/env python
# scripts/build_mixed_dataset.py
"""
Combine synthetic bridge dataset with processed real Anita data.
Supports progressive real-fraction increases (10% → 30% → 50%).

Usage:
    python scripts/build_mixed_dataset.py --real_fraction 0.10
    python scripts/build_mixed_dataset.py --real_fraction 0.30 --total 1000
    python scripts/build_mixed_dataset.py --real_fraction 0.50
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("mixed-builder")


def build_mixed(
    synthetic_dir: str = "data/synthetic/bridge/",
    real_dir: str = "data/real/processed_anita/",
    output_dir: str = "data/mixed/",
    real_fraction: float = 0.10,
    total_sequences: int = 1000,
    seed: int = 42,
):
    """Build a mixed dataset by copying files from synthetic + real dirs."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing mixed data
    existing = list(Path(output_dir).glob("*.h5"))
    if existing:
        log.info("Clearing %d existing files in %s", len(existing), output_dir)
        for f in existing:
            f.unlink()

    synthetic_files = sorted(Path(synthetic_dir).glob("*.h5"))
    real_files = sorted(Path(real_dir).glob("*.h5"))

    log.info("Available: %d synthetic, %d real", len(synthetic_files), len(real_files))

    if not real_files:
        log.warning("No real data found in %s — using synthetic only", real_dir)
        real_fraction = 0.0

    n_real = int(total_sequences * real_fraction)
    n_syn = total_sequences - n_real

    # Cap to available
    n_real = min(n_real, len(real_files))
    n_syn = min(n_syn, len(synthetic_files))

    # If not enough synthetic, fill with real (and vice versa)
    if n_syn < total_sequences - n_real and len(real_files) > n_real:
        extra = min(total_sequences - n_real - n_syn, len(real_files) - n_real)
        n_real += extra

    # Sample
    selected_syn = random.sample(synthetic_files, n_syn) if n_syn > 0 else []
    selected_real = random.sample(real_files, n_real) if n_real > 0 else []

    # Copy to output
    idx = 0
    for f in selected_syn:
        dst = os.path.join(output_dir, f"syn_{idx:04d}.h5")
        shutil.copy2(str(f), dst)
        idx += 1

    for f in selected_real:
        dst = os.path.join(output_dir, f"real_{idx:04d}.h5")
        shutil.copy2(str(f), dst)
        idx += 1

    actual_total = n_syn + n_real
    actual_frac = n_real / actual_total * 100 if actual_total > 0 else 0

    log.info("Mixed dataset built:")
    log.info("  Synthetic: %d", n_syn)
    log.info("  Real:      %d", n_real)
    log.info("  Total:     %d", actual_total)
    log.info("  Real %%:    %.1f%%", actual_frac)
    log.info("  Output:    %s", os.path.abspath(output_dir))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build mixed synthetic+real dataset")
    p.add_argument("--synthetic_dir", default="data/synthetic/bridge/",
                   help="Synthetic bridge data directory")
    p.add_argument("--real_dir", default="data/real/processed_anita/",
                   help="Processed real data directory")
    p.add_argument("--output_dir", default="data/mixed/",
                   help="Output directory for mixed dataset")
    p.add_argument("--real_fraction", type=float, default=0.10,
                   help="Fraction of real data (0.0-1.0)")
    p.add_argument("--total", type=int, default=1000,
                   help="Total sequences in mixed dataset")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    args = p.parse_args()

    build_mixed(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_dir,
        output_dir=args.output_dir,
        real_fraction=args.real_fraction,
        total_sequences=args.total,
        seed=args.seed,
    )
