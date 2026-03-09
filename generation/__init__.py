from .synthetic import (
    circle_bezier,
    oval_bezier,
    trapezoid_bezier,
    rounded_rect_bezier,
    arm_bezier,
    generate_base_character,
    generate_character_variant,
    PALETTE,
)

from .motion_functions import (
    breathing_motion,
    eye_blink,
    gentle_sway,
    compose_motions,
    compute_velocity_gt,
)

from .noise_schedule import (
    NoiseConfig,
    NOISE_CLEAN,
    NOISE_ROBUSTNESS,
    NOISE_BRIDGE,
    apply_noise,
)
