"""
Constants and Configuration for CBAE (Color Boundary Animation Engine).
"""

from typing import Dict, Tuple

# Total number of region slots per frame
N_SLOTS: int = 128

# Number of control points per shape (Bézier curves)
N_CTRL_PTS: int = 12

# Slot indices for static background elements
SLOT_BG_STATIC: Tuple[int, int] = (0, 15)

# Slot indices for dynamic background elements (clouds, water, etc.)
SLOT_BG_DYNAMIC: Tuple[int, int] = (16, 25)

# Slot indices for primary character body
SLOT_BODY: Tuple[int, int] = (26, 45)

# Slot indices for primary character face (eyes, brows, etc.)
SLOT_FACE: Tuple[int, int] = (46, 70)

# Slot indices for mouth region (audio-conditioned)
SLOT_MOUTH: Tuple[int, int] = (71, 90)

# Slot indices for secondary characters or large props
SLOT_SECONDARY: Tuple[int, int] = (91, 110)

# Slot indices for dynamic entities (effects, particles, etc.)
SLOT_DYNAMIC: Tuple[int, int] = (111, 127)

# Maximum deformation allowed per slot block (in canvas units)
DELTA_MAX: Dict[str, float] = {
    "background": 0.02,
    "body": 0.35,
    "face": 0.15,
    "mouth": 0.08,
    "secondary": 0.25,
    "dynamic": 0.40,
}

# Shapes with sigmoid(alive) < threshold are skipped
ALIVENESS_THRESHOLD: float = 0.1

# Default dimensions for the rendering canvas (width, height)
CANVAS_SIZE: Tuple[int, int] = (512, 512)

# Output frames per second
FPS: int = 24

# Internal integration steps per second (for RK4 stability)
FPS_INTERNAL: int = 48

# Total duration of an animation sequence in seconds
SEQUENCE_SECONDS: int = 8

# Total number of generated output frames per sequence
N_FRAMES: int = FPS * SEQUENCE_SECONDS  # = 192

# Total number of internal integration steps per sequence
N_INTERNAL_STEPS: int = FPS_INTERNAL * SEQUENCE_SECONDS  # = 384
