import numpy as np
from dataclasses import dataclass
from core.crf_tensor import CRFTensor
from core.slot_blocks import is_mouth_slot


@dataclass
class NoiseConfig:
    control_point_jitter: float  # std dev in canvas units
    color_noise: float  # std dev in [0,1] color space
    slot_assignment_error_rate: float  # fraction of slots randomly reassigned
    stage: str  # 'clean', 'robustness', 'bridge'


# Define standard configurations for training curriculum
NOISE_CLEAN = NoiseConfig(0.0, 0.0, 0.0, "clean")
NOISE_ROBUSTNESS = NoiseConfig(0.002, 0.01, 0.02, "robustness")
NOISE_BRIDGE = NoiseConfig(0.005, 0.03, 0.05, "bridge")


def apply_noise(
    crf: CRFTensor, config: NoiseConfig, rng: np.random.Generator
) -> CRFTensor:
    """
    Applies Gaussian noise to control points and colors based on config,
    and stochastically swaps active slots with inactive slots to build classification robustness.
    """
    if config.stage == "clean" or (
        config.control_point_jitter == 0
        and config.color_noise == 0
        and config.slot_assignment_error_rate == 0
    ):
        return crf.clone()

    out = crf.clone()
    active_indices = out.active_slots()

    if len(active_indices) == 0:
        return out

    n_active = len(active_indices)
    n_slots = out.P.shape[0]

    # -----------------------------------------------------
    # 1. Spatial and Color Noise Injection
    # -----------------------------------------------------
    if config.control_point_jitter > 0:
        spatial_noise = rng.normal(
            0, config.control_point_jitter, size=(n_active, out.P.shape[1], 2)
        )

        # We must index row-wise carefully into out.P
        for i, idx in enumerate(active_indices):
            out.P[idx] += spatial_noise[i]

        out.P = np.clip(out.P, 0.0, 1.0).astype(np.float16)

    if config.color_noise > 0:
        color_noise = rng.normal(0, config.color_noise, size=(n_active, 3))

        for i, idx in enumerate(active_indices):
            out.c[idx] += color_noise[i]

        out.c = np.clip(out.c, 0.0, 1.0).astype(np.float16)

    # -----------------------------------------------------
    # 2. Slot Assignment Error Injection
    # -----------------------------------------------------
    if config.slot_assignment_error_rate > 0:
        # Determine how many slots to perturb
        # We usually perturb non-mouth, non-CSG slots to avoid trivial visual explosions
        # for shapes that are physically subtractive or tied directly to audio.

        valid_swap_candidates = [
            idx for idx in active_indices if not out.csg[idx] and not is_mouth_slot(idx)
        ]

        n_swaps = int(len(valid_swap_candidates) * config.slot_assignment_error_rate)

        if n_swaps > 0:
            # Pick which active slots to move
            slots_to_move = rng.choice(
                valid_swap_candidates, size=n_swaps, replace=False
            )

            # Find all currently inactive slots
            inactive_mask = out.alive < 0
            inactive_indices = np.where(inactive_mask)[0]

            # If we don't have enough inactive slots, we can't swap
            if len(inactive_indices) >= n_swaps:
                target_slots = rng.choice(inactive_indices, size=n_swaps, replace=False)

                for src_idx, dst_idx in zip(slots_to_move, target_slots):
                    # Copy shape data
                    out.set_shape(
                        dst_idx,
                        out.P[src_idx],
                        out.c[src_idx],
                        out.alpha[src_idx],
                        out.csg[src_idx],
                    )
                    out.z[dst_idx] = out.z[src_idx]  # preserve z-order visibly

                    # Activate target, deactivate source
                    out.activate(dst_idx)
                    out.deactivate(src_idx)

    return out
