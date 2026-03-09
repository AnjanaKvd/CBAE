"""
Constants and Configuration for the CBAE System.

This module contains the definition of system-wide constants, hyper-parameters,
and configurations as specified in the Phase 1 Foundation plan.
"""

from typing import Dict, Tuple

# Number of fixed-size slots per frame (power of 2 for attention efficiency)
N_SLOTS: int = 128

# Number of cubic Bézier control points per shape
N_CTRL_PTS: int = 12

# Slot indices for static background elements (sky, walls, floors)
SLOT_BG_STATIC: Tuple[int, int] = (0, 15)

# Slot indices for dynamic background elements (clouds, water, wind effects)
SLOT_BG_DYNAMIC: Tuple[int, int] = (16, 25)

# Slot indices for the primary character's body (torso, limbs, clothing/robe)
SLOT_BODY: Tuple[int, int] = (26, 45)

# Slot indices for the primary character's face (eyes, brows, nose, ears, head)
SLOT_FACE: Tuple[int, int] = (46, 70)

# Slot indices for the mouth region (lips, teeth, tongue), which are audio-conditioned
SLOT_MOUTH: Tuple[int, int] = (71, 90)

# Slot indices for secondary characters or large props
SLOT_SECONDARY: Tuple[int, int] = (91, 110)

# Slot indices for dynamic entities (effects, particles, small objects)
SLOT_DYNAMIC: Tuple[int, int] = (111, 127)

# Maximum canvas unit deformation clamped per slot block semantic category
DELTA_MAX: Dict[str, float] = {
    'background': 0.02,
    'body':       0.35,
    'face':       0.15,
    'mouth':      0.08,
    'secondary':  0.25,
    'dynamic':    0.40,
}

# Threshold above which a shape is considered alive (sigmoid(alive) >= threshold)
ALIVENESS_THRESHOLD: float = 0.1

# Default resolution (width, height) of the rendered canvas in pixels
CANVAS_SIZE: Tuple[int, int] = (512, 512)

# Frames per second for the output animation video
FPS: int = 24

# Internal simulation steps per second for the RK4 ODE solver
FPS_INTERNAL: int = 48

# Total duration of the generated sequence in seconds
SEQUENCE_SECONDS: int = 8

# Total number of generated output frames for the sequence
N_FRAMES: int = FPS * SEQUENCE_SECONDS  # 192

# Total number of internal ODE solver steps for the sequence
N_INTERNAL_STEPS: int = FPS_INTERNAL * SEQUENCE_SECONDS  # 384
