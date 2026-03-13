# CBAE Training Guide
## From Phase 4 completion → Trained Model
## Complete step-by-step execution guide

---

## The Training Sequence

You do not jump straight to real data. The training follows a strict three-stage progression. Each stage gates the next. Skipping stages will produce a model that learns nothing.

```
Stage 0 — Validate synthetic pipeline works        (1-2 days)
Stage 1 — Train on clean synthetic data            (3-5 days)
Stage 2 — Fine-tune on noise-injected synthetic    (1-2 days)
Stage 3 — Bridge fine-tune on real animation data  (3-7 days)
```

---

## STAGE 0 — Generate Your Synthetic Training Data

Before any real data, your Phase 2 synthetic engine must produce the training dataset. If you skipped actually running it after building it, do this first.

### Step 0.1 — Generate 1,000 synthetic sequences

```bash
cd your_cbae_project/
python -c "
from generation.synthetic import generate_dataset
from generation.noise_schedule import NOISE_CLEAN

generate_dataset(
    n_sequences=1000,
    noise_config=NOISE_CLEAN,
    output_path='data/synthetic/clean/',
    seed=42
)
"
```

This will take 20–60 minutes on a laptop CPU. Expected output:
```
data/synthetic/clean/
├── sequence_0000.h5
├── sequence_0001.h5
├── ...
└── sequence_0999.h5
```

Each HDF5 file contains:
- `frames/` — 192 CRF tensors (one per frame)
- `velocities/` — 192 ground-truth dP/dt tensors
- `metadata` — sequence parameters

### Step 0.2 — Verify dataset integrity

```python
# run this as: python scripts/verify_dataset.py

import h5py
import numpy as np
import os

data_dir = 'data/synthetic/clean/'
files = sorted(os.listdir(data_dir))
errors = []

for fname in files[:10]:  # check first 10
    path = os.path.join(data_dir, fname)
    with h5py.File(path, 'r') as f:
        P = f['frames/P'][:]          # (192, 128, 12, 2)
        V = f['velocities/dP_dt'][:]  # (192, 128, 12, 2)

        if np.any(np.isnan(P)):
            errors.append(f"{fname}: NaN in P")
        if np.any(np.isnan(V)):
            errors.append(f"{fname}: NaN in V")
        if P.shape != (192, 128, 12, 2):
            errors.append(f"{fname}: wrong P shape {P.shape}")
        if np.any(P > 1.0) or np.any(P < 0.0):
            errors.append(f"{fname}: P out of [0,1] range")

if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(f"  {e}")
else:
    print(f"OK: {len(files)} files verified")
    print(f"    P range: [{P.min():.3f}, {P.max():.3f}]")
    print(f"    V magnitude mean: {np.abs(V).mean():.6f}")
```

### Step 0.3 — Generate template library

```bash
python -c "
from generation.synthetic import generate_template_library
generate_template_library(output_path='data/templates/')
print('Templates generated')
"
```

Verify you have 50 files in `data/templates/base_poses/` and `data/templates/embeddings.npy`.

---

## STAGE 1 — Train on Clean Synthetic Data

### Step 1.1 — Verify training configuration

Edit `training/config.py` to match exactly:

```python
# training/config.py

BATCH_SIZE         = 2         # DO NOT increase — checkpointed adjoint limits this
N_EPOCHS_CLEAN     = 50
N_EPOCHS_ROBUSTNESS= 30
N_EPOCHS_BRIDGE    = 20

LR_INIT            = 1e-3
LR_MIN             = 1e-5
GRAD_CLIP          = 1.0
WEIGHT_DECAY       = 1e-4

LOSS_WEIGHTS = {
    'render':   1.0,
    'smooth':   0.1,
    'alive':    0.5,
    'semantic': 0.1,
    'velocity': 1.0,   # set to 0.0 when ground truth not available
}

DATA_DIR_CLEAN      = 'data/synthetic/clean/'
DATA_DIR_ROBUST     = 'data/synthetic/robustness/'
DATA_DIR_BRIDGE     = 'data/synthetic/bridge/'
CHECKPOINT_DIR      = 'checkpoints/'
SAMPLE_OUTPUT_DIR   = 'training_samples/'
LOG_FILE            = 'training_log.csv'

CANVAS_SIZE         = (512, 512)
SEQUENCE_LENGTH     = 192      # frames per sequence
DEVICE              = 'cpu'    # force CPU — matches our hardware target
```

### Step 1.2 — Run the training loop

```bash
python -m training.trainer \
    --stage clean \
    --data_dir data/synthetic/clean/ \
    --epochs 50 \
    --checkpoint_dir checkpoints/stage1/
```

Or if your trainer is a script:

```bash
python training/trainer.py --stage clean
```

### What to watch during training

Monitor the log file in real time:
```bash
# In a second terminal:
tail -f training_log.csv | python -c "
import sys
for line in sys.stdin:
    parts = line.strip().split(',')
    if len(parts) >= 4:
        print(f'Epoch {parts[0]:>3} | Loss: {parts[1]:>8} | Render: {parts[2]:>8} | Velocity: {parts[3]:>8}')
"
```

### What healthy training looks like

```
Epoch   1 | Loss:   2.8400 | Render:   1.2100 | Velocity: 0.8300
Epoch   5 | Loss:   1.9200 | Render:   0.8400 | Velocity: 0.5100
Epoch  10 | Loss:   1.2100 | Render:   0.4900 | Velocity: 0.2800
Epoch  20 | Loss:   0.7400 | Render:   0.2800 | Velocity: 0.1200
Epoch  50 | Loss:   0.3100 | Render:   0.1200 | Velocity: 0.0300
```

Loss should decrease monotonically for the first 10 epochs. If it oscillates wildly or stays flat:

```
Flat loss (no change):
  → Learning rate too low: try LR_INIT = 1e-2
  → Gradient clipping too aggressive: try GRAD_CLIP = 5.0
  → Check MLP_ode is receiving non-zero e_i (motion embeddings)

Oscillating loss:
  → Learning rate too high: try LR_INIT = 1e-4
  → Batch size too small causing high variance: keep at 2, try gradient accumulation steps=4

NaN loss:
  → Gradient explosion: reduce GRAD_CLIP to 0.5
  → Check for division by zero in loss computation
  → Verify canvas coordinates are normalized [0,1]
```

### Step 1.3 — Visual checkpoint check (every 10 epochs)

After epoch 10 and every 10 epochs after, run:

```bash
python -c "
from models.cbae_model import CBAEModel
from rendering.rasterizer import rasterize_sequence
import torch, imageio, numpy as np

# load checkpoint
model = CBAEModel()
model.load_state_dict(torch.load('checkpoints/stage1/model_epoch_10.pt'))
model.eval()

# generate sample
with torch.no_grad():
    crf_seq = model.forward('a character standing and breathing')

# render first 24 frames (1 second)
frames = rasterize_sequence(crf_seq, width=256, height=256)[:24]
imageio.mimsave('training_samples/epoch_10_sample.gif', frames, fps=8)
print('Saved sample to training_samples/epoch_10_sample.gif')
"
```

At epoch 10 you should see rough, jittery shapes in the correct general positions.
At epoch 30 the shapes should be recognizable and mostly in their slots.
At epoch 50 the character should be visible with smooth breathing motion.

### Stage 1 completion criterion

Stage 1 is complete when ALL of:
- 50 epochs completed without NaN
- Total loss at epoch 50 is below 0.5
- Visual sample at epoch 50 shows recognizable character with visible motion
- `training_log.csv` shows consistent downward trend after epoch 5

---

## STAGE 2 — Fine-tune on Noise-Injected Synthetic

### Step 2.1 — Generate noisy synthetic datasets

```bash
# Generate robustness dataset (medium noise)
python -c "
from generation.synthetic import generate_dataset
from generation.noise_schedule import NOISE_ROBUSTNESS
generate_dataset(1000, NOISE_ROBUSTNESS, 'data/synthetic/robustness/')
"

# Generate bridge dataset (higher noise — approaching real-data noise floor)
python -c "
from generation.synthetic import generate_dataset
from generation.noise_schedule import NOISE_BRIDGE
generate_dataset(1000, NOISE_BRIDGE, 'data/synthetic/bridge/')
"
```

### Step 2.2 — Fine-tune from Stage 1 checkpoint

```bash
python training/trainer.py \
    --stage robustness \
    --resume checkpoints/stage1/model_epoch_50.pt \
    --data_dir data/synthetic/robustness/ \
    --epochs 30 \
    --checkpoint_dir checkpoints/stage2/
```

Then bridge fine-tune:

```bash
python training/trainer.py \
    --stage bridge \
    --resume checkpoints/stage2/model_epoch_30.pt \
    --data_dir data/synthetic/bridge/ \
    --epochs 20 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/stage3_pretrain/
```

Lower the learning rate to `1e-4` for fine-tuning — you are refining, not relearning.

After Stage 2 your model should handle imperfect CRF inputs without geometric breakdown. This is the checkpoint you will use as the starting point for real data.

---

## STAGE 3 — Real Animation Data

### WHERE TO GET THE DATA

You have four real datasets available. Use them in the order listed — start with the most structured, move to the noisier ones.

---

### Dataset 1 — Anita Dataset (START HERE)
**Best for CBAE. 16,000+ 1080p flat-color keyframes from professional 2D animation.**

```
License:     CC BY-NC-SA 4.0 (academic/research use OK)
Size:        7.14 GB
Resolution:  1080p
Format:      PNG frames, organized by scene
Content:     Professional hand-drawn, flat-color animation
Why first:   High quality, structured, flat-color (matches your CRF assumptions)
```

Download:
```bash
git clone https://github.com/zhenglinpan/AnitaDataset.git
cd AnitaDataset
# Follow the download instructions in their README
# Data is in: Anita_Dataset/Mirror/scene_*/color/*.png
```

Filter for flat-color frames only (skip sketch frames):
```bash
# Only use files in the 'color' subdirectories, not 'sketch'
find AnitaDataset/Anita_Dataset -path "*/color/*.png" > data/real/anita_color_files.txt
wc -l data/real/anita_color_files.txt  # should be ~16,000
```

---

### Dataset 2 — AnimeRun
**Animation sequences with visual correspondence annotations. Good for optical flow ground truth.**

```
License:     Research use
Size:        ~15 GB
Content:     3D-rendered anime characters with flow annotations
Why useful:  Has ground-truth correspondence — reduces RAFT noise
```

```bash
# Download from paper page:
# https://github.com/lisiyao21/AnimeRun
git clone https://github.com/lisiyao21/AnimeRun
# Follow dataset download instructions
```

---

### Dataset 3 — Sakuga-42M (SUBSET ONLY)
**42 million keyframes. Do NOT download all of it. Download a filtered subset.**

```
License:     CC BY-NC-SA 4.0 (academic use only)
Full size:   Terabytes
Your subset: ~5,000 clips = ~20 GB
Filter for:  High dynamic score, flat-color taxonomy tag, 24-96 keyframes
```

```bash
git clone https://github.com/KytraScript/SakugaDataset
cd SakugaDataset
conda activate sakuga
pip install -r requirement.txt

# Fill access form at their repo to get parquet files
# Then filter parquet for ONLY flat-color clips:

python -c "
import pandas as pd
df = pd.read_parquet('download/parquet/metadata.parquet')

# Filter criteria for CBAE-compatible clips:
filtered = df[
    (df['dynamic_score'] > 0.3) &          # has motion
    (df['aesthetic_score'] > 0.5) &         # reasonable quality
    (df['clip_duration'] >= 24) &           # at least 24 frames
    (df['clip_duration'] <= 96) &           # max 4 seconds at 24fps
    (df['has_text'] == False) &             # no subtitle overlay
    (df['safety_rating'] == 'general')      # safe content only
]

# Further filter: keep only clips tagged as flat-color style
# Sakuga taxonomy tags include 'genga' (rough), 'douga' (in-between/flat-color)
# We want 'douga' style
if 'tags' in df.columns:
    filtered = filtered[filtered['tags'].str.contains('douga', na=False)]

print(f'Filtered: {len(filtered)} clips from {len(df)} total')
filtered.head(5000).to_parquet('download/parquet/cbae_subset.parquet')
print('Saved subset of 5000 clips')
"

# Download only the filtered subset:
python download/download.py --parquet download/parquet/cbae_subset.parquet
```

---

### Dataset 4 — LinkTo-Anime
**395 video sequences with ground-truth optical flow. Use for validating your RAFT extraction.**

```
License:     Research use (check their terms)
Size:        ~9 GB
Content:     3D-rendered anime character motion with exact flow annotations
Why useful:  Can validate your RAFT+least-squares projection accuracy
             by comparing your extracted velocities to ground-truth flow
```

```bash
# Available at: https://arxiv.org/abs/2506.02733
# Download link in their paper/repo — check HuggingFace:
# huggingface.co — search "LinkTo-Anime"
```

---

### HOW TO APPLY DATA TO YOUR PROJECT

This is the complete pipeline from raw video/images to HDF5 training data.

### Step 3.1 — Convert Anita frames to CRF sequences

Create `scripts/process_anita.py`:

```python
"""
Processes Anita Dataset PNG frames into CBAE CRFSequence HDF5 files.
This is your main real-data preprocessing script.

Usage:
    python scripts/process_anita.py \
        --input_list data/real/anita_color_files.txt \
        --output_dir data/real/processed_anita/ \
        --sequence_length 48 \
        --target_sequences 500
"""

import argparse
import os
import cv2
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Your CBAE imports
from core.crf_tensor import CRFTensor, CRFSequence
from core.constants import N_SLOTS, CANVAS_SIZE
from data.pipeline import (
    quantize_colors,
    extract_regions,
    fit_bezier_boundaries,
    assign_slots_heuristic,
    quality_filter
)


def load_scene_frames(scene_dir: str, max_frames: int = 48):
    """Load consecutive frames from one Anita scene directory."""
    frame_files = sorted(Path(scene_dir).glob("*.png"))[:max_frames]
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, CANVAS_SIZE)
        frames.append(img)
    return frames


def frames_to_crf_sequence(frames: list, scene_name: str) -> CRFSequence | None:
    """
    Convert a list of RGB frames to a CRFSequence.
    Returns None if quality filter rejects the sequence.
    """
    crf_frames = []
    reject_reasons = []

    for i, frame in enumerate(frames):
        # Step 1: Color quantization (K-means in LAB)
        palette_img, colors = quantize_colors(frame, k=16)

        # Step 2: Region extraction (connected components)
        regions = extract_regions(palette_img, min_area_fraction=0.001)

        # Step 3: Boundary fitting (Douglas-Peucker + Bezier)
        shapes = fit_bezier_boundaries(regions, frame.shape, epsilon=2.0)

        # Step 4: Slot assignment (heuristic)
        assigned = assign_slots_heuristic(shapes, canvas_size=CANVAS_SIZE)

        # Step 5: Quality filter
        ok, reason = quality_filter(
            shapes=assigned,
            min_shapes=8,
            max_shapes=N_SLOTS,
            max_fit_error=3.0
        )
        if not ok:
            reject_reasons.append(f"frame {i}: {reason}")
            continue

        # Build CRFTensor
        crf = CRFTensor()
        for slot_idx, shape_data in assigned.items():
            crf.P[slot_idx]     = shape_data['control_points']
            crf.c[slot_idx]     = shape_data['color']
            crf.alpha[slot_idx] = shape_data['alpha']
            crf.alive[slot_idx] = 5.0  # activated
            crf.csg[slot_idx]   = shape_data.get('csg', False)

        crf_frames.append(crf)

    if len(crf_frames) < 24:  # need at least 24 frames (1 second)
        print(f"  REJECTED {scene_name}: only {len(crf_frames)} valid frames")
        return None

    # Compute velocities via finite difference between consecutive frames
    velocities = []
    for i in range(len(crf_frames)):
        if i == 0:
            v = crf_frames[1].P - crf_frames[0].P  # forward diff for first frame
        elif i == len(crf_frames) - 1:
            v = crf_frames[-1].P - crf_frames[-2].P  # backward diff for last frame
        else:
            v = (crf_frames[i+1].P - crf_frames[i-1].P) / 2.0  # central diff
        velocities.append(v.astype(np.float16))

    return CRFSequence(frames=crf_frames, velocities=velocities)


def process_anita(input_list: str, output_dir: str,
                  sequence_length: int, target_sequences: int):

    os.makedirs(output_dir, exist_ok=True)

    # Group files by scene directory
    with open(input_list) as f:
        all_files = [line.strip() for line in f]

    scenes = {}
    for fpath in all_files:
        scene_dir = str(Path(fpath).parent)
        scenes.setdefault(scene_dir, []).append(fpath)

    print(f"Found {len(scenes)} scenes")

    saved = 0
    for scene_dir, frame_files in tqdm(scenes.items(), desc="Processing scenes"):
        if saved >= target_sequences:
            break

        if len(frame_files) < 12:  # too short
            continue

        frames = load_scene_frames(scene_dir, max_frames=sequence_length)
        if len(frames) < 12:
            continue

        seq = frames_to_crf_sequence(frames, scene_name=os.path.basename(scene_dir))
        if seq is None:
            continue

        out_path = os.path.join(output_dir, f"seq_{saved:04d}.h5")
        seq.to_hdf5(out_path)
        saved += 1

    print(f"\nDone. Saved {saved} sequences to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_list', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sequence_length', type=int, default=48)
    parser.add_argument('--target_sequences', type=int, default=500)
    args = parser.parse_args()
    process_anita(**vars(args))
```

Run it:
```bash
python scripts/process_anita.py \
    --input_list data/real/anita_color_files.txt \
    --output_dir data/real/processed_anita/ \
    --sequence_length 48 \
    --target_sequences 500
```

Expected output: 300–500 HDF5 files (some scenes will be rejected by quality filter — that's normal).

---

### Step 3.2 — Process Sakuga-42M clips (after Anita is working)

Create `scripts/process_sakuga.py`:

```python
"""
Process downloaded Sakuga video clips into CRFSequences.
Clips are .mp4 files. We extract frames, then apply the same pipeline.

Usage:
    python scripts/process_sakuga.py \
        --video_dir download/download/ \
        --output_dir data/real/processed_sakuga/ \
        --target_sequences 2000
"""

import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from core.crf_tensor import CRFTensor, CRFSequence
from core.constants import CANVAS_SIZE
from data.pipeline import (
    quantize_colors, extract_regions, fit_bezier_boundaries,
    assign_slots_heuristic, quality_filter
)


def extract_frames_from_video(video_path: str, max_frames=96) -> list:
    """Extract frames from .mp4 file, resize to CANVAS_SIZE."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, CANVAS_SIZE)
        frames.append(frame)
    cap.release()
    return frames


def flat_color_score(frame: np.ndarray) -> float:
    """
    Score how 'flat-color' a frame is.
    Flat-color animation has low intra-region color variance.
    Score = fraction of pixels whose local variance is below threshold.
    Higher = more flat-color (better for CBAE).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(float)
    kernel_size = 7
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = sq_mean - mean**2
    flat_fraction = (variance < 50).mean()  # threshold: adjust if needed
    return float(flat_fraction)


def process_sakuga(video_dir: str, output_dir: str, target_sequences: int):
    os.makedirs(output_dir, exist_ok=True)

    video_files = list(Path(video_dir).rglob("*.mp4"))
    print(f"Found {len(video_files)} video files")

    saved = 0
    rejected_flat = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        if saved >= target_sequences:
            break

        frames = extract_frames_from_video(str(video_path), max_frames=96)
        if len(frames) < 24:
            continue

        # Pre-filter: reject photorealistic or heavily shaded frames
        sample_frame = frames[len(frames)//2]
        score = flat_color_score(sample_frame)
        if score < 0.55:  # less than 55% flat-color pixels → skip
            rejected_flat += 1
            continue

        # Same pipeline as Anita
        crf_frames = []
        for frame in frames[:48]:  # cap at 48 frames = 2 seconds
            palette_img, colors = quantize_colors(frame, k=16)
            regions = extract_regions(palette_img, min_area_fraction=0.001)
            shapes = fit_bezier_boundaries(regions, frame.shape)
            assigned = assign_slots_heuristic(shapes, CANVAS_SIZE)
            ok, _ = quality_filter(assigned, 8, 128, 3.0)
            if not ok:
                continue

            crf = CRFTensor()
            for slot_idx, sd in assigned.items():
                crf.P[slot_idx]     = sd['control_points']
                crf.c[slot_idx]     = sd['color']
                crf.alpha[slot_idx] = sd['alpha']
                crf.alive[slot_idx] = 5.0
            crf_frames.append(crf)

        if len(crf_frames) < 24:
            continue

        velocities = []
        for i in range(len(crf_frames)):
            if i == 0:
                v = crf_frames[1].P - crf_frames[0].P
            elif i == len(crf_frames)-1:
                v = crf_frames[-1].P - crf_frames[-2].P
            else:
                v = (crf_frames[i+1].P - crf_frames[i-1].P) / 2.0
            velocities.append(v.astype(np.float16))

        seq = CRFSequence(frames=crf_frames, velocities=velocities)
        seq.to_hdf5(os.path.join(output_dir, f"seq_{saved:04d}.h5"))
        saved += 1

    print(f"\nSaved {saved} sequences | Rejected (not flat-color): {rejected_flat}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--target_sequences', type=int, default=2000)
    process_sakuga(**vars(p.parse_args()))
```

---

### Step 3.3 — Implement the data pipeline functions

These go in `data/pipeline.py`. These are the functions called by both processing scripts above:

```python
# data/pipeline.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import interpolate
from core.constants import (
    N_SLOTS, N_CTRL_PTS, CANVAS_SIZE,
    SLOT_BG_STATIC, SLOT_BG_DYNAMIC, SLOT_BODY,
    SLOT_FACE, SLOT_MOUTH, SLOT_SECONDARY, SLOT_DYNAMIC
)


def quantize_colors(frame: np.ndarray, k: int = 16):
    """
    K-means color quantization in CIELAB space.
    Returns: palette_img (H,W) int — cluster labels, colors (k,3) float [0,1]
    """
    # Convert to LAB for perceptual uniformity
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)

    # Run K-means 3 times, pick best (lowest inertia)
    best_km, best_inertia = None, float('inf')
    for seed in [0, 42, 99]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=50)
        km.fit(lab)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_km = km

    labels = best_km.labels_.reshape(frame.shape[:2])
    # Convert cluster centers back to RGB [0,1]
    centers_lab = best_km.cluster_centers_.astype(np.uint8)
    centers_rgb = []
    for c in centers_lab:
        rgb = cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_LAB2RGB)
        centers_rgb.append(rgb.reshape(3).astype(float) / 255.0)

    return labels, np.array(centers_rgb)


def extract_regions(palette_img: np.ndarray, min_area_fraction: float = 0.001):
    """
    Find connected component regions for each color cluster.
    Returns list of dicts: {label, mask, area, centroid, color_idx}
    """
    h, w = palette_img.shape
    min_area = int(min_area_fraction * h * w)
    regions = []

    for label in np.unique(palette_img):
        binary = (palette_img == label).astype(np.uint8)
        n_comp, comp_map, stats, centroids = cv2.connectedComponentsWithStats(binary)

        for comp_idx in range(1, n_comp):  # skip background (0)
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            mask = (comp_map == comp_idx)
            cx, cy = centroids[comp_idx]
            regions.append({
                'color_idx':  label,
                'mask':       mask,
                'area':       area,
                'centroid':   (cx / w, cy / h),  # normalized [0,1]
                'area_frac':  area / (h * w)
            })

    return regions


def fit_bezier_boundaries(regions: list, frame_shape: tuple, epsilon: float = 2.0):
    """
    Fit cubic Bezier curves to region boundaries.
    Returns list of dicts with control_points (12,2) normalized [0,1].
    """
    h, w = frame_shape[:2]
    shapes = []

    for region in regions:
        # Find contour
        mask_uint8 = region['mask'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 4:
            continue

        # Douglas-Peucker simplification
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        pts = simplified.squeeze().astype(float)
        if len(pts.shape) < 2 or pts.shape[0] < 4:
            continue

        # Normalize to [0,1]
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0, 1)

        # Resample to exactly N_CTRL_PTS points via interpolation
        n_pts = len(pts)
        t_orig = np.linspace(0, 1, n_pts)
        t_new  = np.linspace(0, 1, N_CTRL_PTS)

        ctrl_pts = np.zeros((N_CTRL_PTS, 2))
        for dim in range(2):
            f = interpolate.interp1d(t_orig, pts[:, dim],
                                      kind='cubic', bounds_error=False,
                                      fill_value=(pts[0, dim], pts[-1, dim]))
            ctrl_pts[:, dim] = np.clip(f(t_new), 0, 1)

        # Compute fit error for quality filter
        fit_error = _compute_fit_error(ctrl_pts, pts, w, h)

        shapes.append({
            'control_points': ctrl_pts.astype(np.float16),
            'color_idx':      region['color_idx'],
            'area_frac':      region['area_frac'],
            'centroid':       region['centroid'],
            'fit_error':      fit_error,
            'alpha':          1.0,
            'csg':            False
        })

    return shapes


def _compute_fit_error(ctrl_pts, original_pts, w, h):
    """Mean distance between original contour points and fitted curve (in pixels)."""
    from generation.synthetic import bezier_to_polyline  # reuse your implementation
    curve = bezier_to_polyline(ctrl_pts, n_samples=200)
    original_px = original_pts * np.array([w, h])
    min_dists = []
    for pt in original_px:
        dists = np.linalg.norm(curve * np.array([w, h]) - pt, axis=1)
        min_dists.append(dists.min())
    return np.mean(min_dists)


def assign_slots_heuristic(shapes: list, canvas_size: tuple):
    """
    Assign shapes to semantic slot indices using rule-based heuristics.
    Uses centroid position and area to determine slot block.
    Returns dict: {slot_index: shape_data}
    """
    w, h = canvas_size
    assigned = {}

    # Sort by area descending (largest shapes first)
    shapes_sorted = sorted(shapes, key=lambda s: s['area_frac'], reverse=True)

    # Counters per block
    counters = {
        'bg_static':  SLOT_BG_STATIC[0],
        'bg_dynamic': SLOT_BG_DYNAMIC[0],
        'body':       SLOT_BODY[0],
        'face':       SLOT_FACE[0],
        'mouth':      SLOT_MOUTH[0],
        'secondary':  SLOT_SECONDARY[0],
        'dynamic':    SLOT_DYNAMIC[0]
    }
    limits = {
        'bg_static':  SLOT_BG_STATIC[1],
        'bg_dynamic': SLOT_BG_DYNAMIC[1],
        'body':       SLOT_BODY[1],
        'face':       SLOT_FACE[1],
        'mouth':      SLOT_MOUTH[1],
        'secondary':  SLOT_SECONDARY[1],
        'dynamic':    SLOT_DYNAMIC[1]
    }

    for shape in shapes_sorted:
        cx, cy = shape['centroid']  # normalized [0,1]
        area = shape['area_frac']

        # Heuristic rules:
        if area > 0.15 and cy > 0.5:
            block = 'bg_static'    # Large area, lower half → background
        elif area > 0.08:
            block = 'body'         # Medium-large → body/torso
        elif cy < 0.45 and area > 0.02:
            block = 'face'         # Upper area, moderate size → face
        elif cy < 0.55 and area < 0.02 and cx > 0.3 and cx < 0.7:
            block = 'mouth'        # Small, center, upper-mid → mouth
        elif area > 0.05:
            block = 'secondary'    # Medium misc → secondary
        else:
            block = 'dynamic'      # Small misc → dynamic

        # Check if slot block is full
        if counters[block] > limits[block]:
            # Overflow into dynamic
            block = 'dynamic'
            if counters['dynamic'] > limits['dynamic']:
                continue  # Truly no space — skip this shape

        slot_idx = counters[block]
        counters[block] += 1
        assigned[slot_idx] = shape

    return assigned


def quality_filter(shapes: dict, min_shapes: int, max_shapes: int,
                   max_fit_error: float):
    """
    Returns (True, None) if passes, (False, reason_string) if rejected.
    """
    n = len(shapes)
    if n < min_shapes:
        return False, f"too few shapes: {n} < {min_shapes}"
    if n > max_shapes:
        return False, f"too many shapes: {n} > {max_shapes}"

    errors = [s.get('fit_error', 0) for s in shapes.values()]
    if errors and max(errors) > max_fit_error:
        return False, f"high fit error: {max(errors):.2f}px > {max_fit_error}px"

    return True, None
```

---

### Step 3.4 — Combine synthetic + real into mixed dataset

```python
# scripts/build_mixed_dataset.py
"""
Combine synthetic bridge dataset with processed real data.
Start with 90% synthetic / 10% real, then adjust.
"""

import os, shutil
from pathlib import Path
import random

def build_mixed(
    synthetic_dir='data/synthetic/bridge/',
    real_dir='data/real/processed_anita/',
    output_dir='data/mixed/',
    real_fraction=0.10,
    total_sequences=1000
):
    os.makedirs(output_dir, exist_ok=True)

    synthetic_files = list(Path(synthetic_dir).glob("*.h5"))
    real_files      = list(Path(real_dir).glob("*.h5"))

    n_real = int(total_sequences * real_fraction)
    n_syn  = total_sequences - n_real

    n_real = min(n_real, len(real_files))
    n_syn  = min(n_syn,  len(synthetic_files))

    selected_real = random.sample(real_files, n_real)
    selected_syn  = random.sample(synthetic_files, n_syn)

    for i, f in enumerate(selected_syn):
        shutil.copy(f, output_dir + f"syn_{i:04d}.h5")
    for i, f in enumerate(selected_real):
        shutil.copy(f, output_dir + f"real_{i:04d}.h5")

    print(f"Mixed dataset: {n_syn} synthetic + {n_real} real = {n_syn+n_real} total")
    print(f"Real fraction: {n_real/(n_syn+n_real)*100:.1f}%")

build_mixed()
```

---

### Step 3.5 — Fine-tune on mixed dataset

```bash
# Start from Stage 2 final checkpoint
python training/trainer.py \
    --stage mixed \
    --resume checkpoints/stage3_pretrain/model_epoch_20.pt \
    --data_dir data/mixed/ \
    --epochs 30 \
    --lr 5e-5 \
    --loss_velocity_weight 0.0 \
    --checkpoint_dir checkpoints/stage4_mixed/
```

Notice `--loss_velocity_weight 0.0` — real data does not have exact dP/dt ground truth (only finite-difference approximations), so we turn off the velocity loss and rely on LPIPS + smoothness only.

---

### Step 3.6 — Gradually increase real data fraction

After 10 epochs at 10% real, rebuild the mixed dataset at 30% real and continue:

```bash
python scripts/build_mixed_dataset.py --real_fraction 0.30
python training/trainer.py \
    --stage mixed_30 \
    --resume checkpoints/stage4_mixed/model_epoch_10.pt \
    --data_dir data/mixed/ \
    --epochs 20 \
    --lr 2e-5 \
    --checkpoint_dir checkpoints/stage4_mixed_30/
```

After 10 more epochs at 30%, move to 50%:

```bash
python scripts/build_mixed_dataset.py --real_fraction 0.50
python training/trainer.py \
    --stage mixed_50 \
    --resume checkpoints/stage4_mixed_30/model_epoch_10.pt \
    --data_dir data/mixed/ \
    --epochs 20 \
    --lr 1e-5 \
    --checkpoint_dir checkpoints/final/
```

---

## MONITORING TRAINING HEALTH

### Signs that real data is hurting the model

Run this after every 10 epochs on the mixed dataset:

```bash
python evaluation/benchmark.py \
    --checkpoint checkpoints/stage4_mixed/model_epoch_10.pt \
    --test_prompts "a character standing" "a figure breathing" \
    --compare_against checkpoints/stage3_pretrain/model_epoch_20.pt
```

If BCS (Boundary Coherence Score) INCREASES after adding real data, the real data is introducing noise. Reduce the real fraction and retrain.

---

## FINAL TRAINING SUMMARY — ALL COMMANDS IN ORDER

```bash
# === STAGE 0: Generate data ===
python -c "from generation.synthetic import generate_dataset, generate_template_library; \
           from generation.noise_schedule import *; \
           generate_dataset(1000, NOISE_CLEAN, 'data/synthetic/clean/'); \
           generate_template_library('data/templates/')"

# === STAGE 1: Train clean ===
python training/trainer.py --stage clean --epochs 50 \
    --checkpoint_dir checkpoints/stage1/

# === STAGE 2: Robustness fine-tune ===
python -c "from generation.synthetic import generate_dataset; \
           from generation.noise_schedule import *; \
           generate_dataset(1000, NOISE_ROBUSTNESS, 'data/synthetic/robustness/'); \
           generate_dataset(1000, NOISE_BRIDGE, 'data/synthetic/bridge/')"

python training/trainer.py --stage robustness \
    --resume checkpoints/stage1/model_epoch_50.pt \
    --epochs 30 --lr 1e-4 --checkpoint_dir checkpoints/stage2/

python training/trainer.py --stage bridge \
    --resume checkpoints/stage2/model_epoch_30.pt \
    --epochs 20 --lr 1e-4 --checkpoint_dir checkpoints/stage3_pretrain/

# === STAGE 3: Real data ===
# Download datasets (see sections above), then:
python scripts/process_anita.py \
    --input_list data/real/anita_color_files.txt \
    --output_dir data/real/processed_anita/ \
    --target_sequences 500

python scripts/build_mixed_dataset.py --real_fraction 0.10
python training/trainer.py --stage mixed \
    --resume checkpoints/stage3_pretrain/model_epoch_20.pt \
    --loss_velocity_weight 0.0 --lr 5e-5 --epochs 30 \
    --checkpoint_dir checkpoints/stage4_mixed/

python scripts/build_mixed_dataset.py --real_fraction 0.50
python training/trainer.py --stage mixed_50 \
    --resume checkpoints/stage4_mixed/model_epoch_30.pt \
    --lr 1e-5 --epochs 20 \
    --checkpoint_dir checkpoints/final/

# === FINAL EVALUATION ===
python evaluation/benchmark.py \
    --checkpoint checkpoints/final/model_epoch_20.pt \
    --output evaluation/final_results.json
```

---

## EXPECTED TOTAL TRAINING TIME

| Stage | Sequences | Epochs | Est. Time (laptop CPU) |
|---|---|---|---|
| Stage 1 clean | 1,000 | 50 | 12–24 hours |
| Stage 2 robustness | 1,000 | 30 | 7–14 hours |
| Stage 2 bridge | 1,000 | 20 | 5–10 hours |
| Stage 3 mixed 10% | ~1,000 | 30 | 7–14 hours |
| Stage 3 mixed 50% | ~1,000 | 20 | 5–10 hours |
| **Total** | | **150 epochs** | **~36–72 hours** |

Train while you sleep. Each stage is independent — you can stop, resume from checkpoint. The only cost of stopping is losing the current epoch's progress.

If 72 hours is too long, use Google Colab (free T4 GPU) for Stages 1–2. The model is small enough to fit. Estimated time on T4: ~4–6 hours total for all stages.

---

*CBAE Training Guide v1.0 — follows CBAE_PLAN.md Phase 3 & Phase 2+ Real Data sections*
