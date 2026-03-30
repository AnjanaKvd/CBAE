# data/dataset.py
"""
CBAEDataset — PyTorch Dataset that loads HDF5 CRF sequences
and rasterises ground-truth video frames for training.

Usage:
    from data.dataset import CBAEDataset
    ds = CBAEDataset('data/synthetic/clean/', render_size=64, max_frames=192)
    audio, gt_video = ds[0]
"""

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from core.crf_tensor import CRFTensor


class CBAEDataset(Dataset):
    """
    Loads HDF5 CRF sequences and produces (audio, gt_video) tensors.

    Each item returns:
        audio:    (8*16000,) float32 — silent placeholder (synthetic has no audio)
        gt_video: (T, H, W, 3) float32 in [0, 1] — rasterised ground truth
    """

    def __init__(
        self,
        data_dir: str,
        render_size: int = 64,
        max_frames: Optional[int] = None,
        audio_seconds: int = 8,
        sample_rate: int = 16_000,
        cache_renders: bool = True,
    ):
        """
        Args:
            data_dir:      directory containing .h5 sequence files.
            render_size:   width=height for rasterised GT frames.
            max_frames:    cap on number of frames per sequence (None = use all).
            audio_seconds: length of dummy audio in seconds.
            sample_rate:   audio sample rate (16 kHz for Whisper).
            cache_renders: if True, cache rendered video tensors after first load.
        """
        self.data_dir = data_dir
        self.render_size = render_size
        self.max_frames = max_frames
        self.audio_len = audio_seconds * sample_rate
        self.cache_renders = cache_renders

        # Discover all .h5 files
        self.h5_files = sorted(
            [str(p) for p in Path(data_dir).glob("*.h5")]
        )
        if len(self.h5_files) == 0:
            raise FileNotFoundError(
                f"No .h5 files found in {data_dir}"
            )

        # Render cache (path → tensor)
        self._cache = {}

    def __len__(self) -> int:
        return len(self.h5_files)

    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        h5_path = self.h5_files[idx]

        # Load topology 0 from HDF5
        with h5py.File(h5_path, "r") as f:
            if "P" not in f or f["P"].size == 0:
                topology_0 = {
                    'P': torch.zeros(128, 12, 2, dtype=torch.float32),
                    'colors': torch.zeros(128, 3, dtype=torch.float32),
                    'alive': torch.zeros(128, dtype=torch.float32)
                }
            else:
                topology_0 = {
                    'P': torch.from_numpy(f["P"][0]).float(),
                    'colors': torch.from_numpy(f["c"][0]).float(),
                    'alive': torch.from_numpy(f["alive"][0]).float()
                }

        # Check cache
        if self.cache_renders and h5_path in self._cache:
            gt_video = self._cache[h5_path]
        else:
            gt_video = self._load_and_render(h5_path)
            if self.cache_renders:
                self._cache[h5_path] = gt_video

        return topology_0, gt_video

    def _load_and_render(self, h5_path: str) -> torch.Tensor:
        """Load CRF sequence from HDF5 and rasterise each frame."""
        # Lazy import to avoid circular deps and Cairo at module load time
        from rendering.rasterizer import rasterize

        with h5py.File(h5_path, "r") as f:
            if "P" not in f or f["P"].size == 0:
                # Empty sequence — return blank video
                T = self.max_frames or 192
                return torch.zeros(
                    T, self.render_size, self.render_size, 3,
                    dtype=torch.float32,
                )

            P_all = f["P"][:]          # (T, 128, 12, 2)
            c_all = f["c"][:]          # (T, 128, 3)
            alpha_all = f["alpha"][:]  # (T, 128)
            alive_all = f["alive"][:]  # (T, 128)
            csg_all = f["csg"][:]      # (T, 128)
            z_all = f["z"][:]          # (T, 128)

        n_frames = P_all.shape[0]
        if self.max_frames is not None:
            n_frames = min(n_frames, self.max_frames)

        frames = []
        for i in range(n_frames):
            crf = CRFTensor(n_slots=P_all.shape[1], n_ctrl_pts=P_all.shape[2])
            crf.P = P_all[i]
            crf.c = c_all[i]
            crf.alpha = alpha_all[i]
            crf.alive = alive_all[i]
            crf.csg = csg_all[i]
            crf.z = z_all[i]

            # Rasterise to uint8 RGB then normalise to [0, 1]
            rgb = rasterize(crf, width=self.render_size, height=self.render_size)
            frames.append(
                torch.from_numpy(rgb.astype(np.float32) / 255.0)
            )

        # (T, H, W, 3)
        return torch.stack(frames, dim=0)
