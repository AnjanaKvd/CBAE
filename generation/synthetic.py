import numpy as np
import os
import json
from tqdm import tqdm
from core.crf_tensor import CRFTensor, CRFSequence
from generation.noise_schedule import NoiseConfig, apply_noise
from generation.motion_functions import compute_velocity_gt


def oval_bezier(cx, cy, rx, ry, n_ctrl=12):
    """Ellipse approximation using 4 cubic Béziers."""
    kappa = 0.552284749831
    kx = rx * kappa
    ky = ry * kappa

    # Standard 12 points starting from right-midpoint, going clockwise
    # (assuming y goes down, cy-ky is "up" visually but lower value)
    pts = [
        [cx + rx, cy],  # 0: Right endpoint
        [cx + rx, cy - ky],  # 1: CP going up
        [cx + kx, cy - ry],  # 2: CP arriving at top
        [cx, cy - ry],  # 3: Top endpoint
        [cx - kx, cy - ry],  # 4: CP going left
        [cx - rx, cy - ky],  # 5: CP arriving at left
        [cx - rx, cy],  # 6: Left endpoint
        [cx - rx, cy + ky],  # 7: CP going down
        [cx - kx, cy + ry],  # 8: CP arriving at bottom
        [cx, cy + ry],  # 9: Bottom endpoint
        [cx + kx, cy + ry],  # 10: CP going right
        [cx + rx, cy + ky],  # 11: CP arriving at right
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def circle_bezier(cx, cy, r, n_ctrl=12):
    """Approximate circle with 12 Bézier control points."""
    return oval_bezier(cx, cy, r, r, n_ctrl)


def trapezoid_bezier(cx, cy, top_w, bot_w, h, n_ctrl=12):
    """Trapezoid for robe/torso."""
    TR = np.array([cx + top_w / 2, cy - h / 2])
    TL = np.array([cx - top_w / 2, cy - h / 2])
    BL = np.array([cx - bot_w / 2, cy + h / 2])
    BR = np.array([cx + bot_w / 2, cy + h / 2])

    # 4 straight segments interpolated over 12 control points
    pts = [
        TR,
        TR + (TL - TR) * 0.333333,
        TR + (TL - TR) * 0.666667,
        TL,
        TL + (BL - TL) * 0.333333,
        TL + (BL - TL) * 0.666667,
        BL,
        BL + (BR - BL) * 0.333333,
        BL + (BR - BL) * 0.666667,
        BR,
        BR + (TR - BR) * 0.333333,
        BR + (TR - BR) * 0.666667,
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def rounded_rect_bezier(cx, cy, w, h, corner_r, n_ctrl=12):
    """Rounded rectangle approximation."""
    kappa = 0.552284749831
    # Interpolation distance handles the squircle offset dynamically
    dx = w / 2 - corner_r * (1 - kappa)
    dy = h / 2 - corner_r * (1 - kappa)

    pts = [
        [cx + w / 2, cy],  # 0: RM
        [cx + w / 2, cy - dy],  # 1: CP up
        [cx + dx, cy - h / 2],  # 2: CP arriving at TM
        [cx, cy - h / 2],  # 3: TM
        [cx - dx, cy - h / 2],  # 4: CP left
        [cx - w / 2, cy - dy],  # 5: CP arriving at LM
        [cx - w / 2, cy],  # 6: LM
        [cx - w / 2, cy + dy],  # 7: CP down
        [cx - dx, cy + h / 2],  # 8: CP arriving at BM
        [cx, cy + h / 2],  # 9: BM
        [cx + dx, cy + h / 2],  # 10: CP right
        [cx + w / 2, cy + dy],  # 11: CP arriving at RM
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


def _get_normal(p1, p2):
    """Helper to get 2D normal vector to a line segment."""
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-5:
        return np.array([0.0, 1.0])
    return np.array([-d[1], d[0]]) / length


def arm_bezier(shoulder_xy, elbow_xy, wrist_xy, width, n_ctrl=12):
    """Tapered arm shape from shoulder to wrist via elbow."""
    S = np.array(shoulder_xy)
    E = np.array(elbow_xy)
    W = np.array(wrist_xy)

    n_SE = _get_normal(S, E)
    n_EW = _get_normal(E, W)

    # Elbow bisector normal
    n_E = n_SE + n_EW
    n_E_norm = np.linalg.norm(n_E)
    if n_E_norm > 1e-5:
        n_E = n_E / n_E_norm
    else:
        n_E = n_SE

    w_S = width
    w_E = width * 0.8
    w_W = width * 0.5

    out_S = S + n_SE * (w_S / 2)
    inn_S = S - n_SE * (w_S / 2)

    out_E = E + n_E * (w_E / 2)
    inn_E = E - n_E * (w_E / 2)

    out_W = W + n_EW * (w_W / 2)
    inn_W = W - n_EW * (w_W / 2)

    # Push control points outwards so the curve (t=0.5) reaches the target elbow width exactly
    cp_out = out_E * 1.333333 - out_S * 0.166667 - out_W * 0.166667
    cp_inn = inn_E * 1.333333 - inn_S * 0.166667 - inn_W * 0.166667

    pts = [
        out_S,
        cp_out,
        cp_out,
        out_W,
        # Linear flat/round wrist cap
        out_W + (inn_W - out_W) * 0.333333,
        out_W + (inn_W - out_W) * 0.666667,
        inn_W,
        cp_inn,
        cp_inn,
        inn_S,
        # Linear flat/round shoulder cap
        inn_S + (out_S - inn_S) * 0.333333,
        inn_S + (out_S - inn_S) * 0.666667,
    ]
    return np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)


# Predefined palettes to assign deterministic color variants
PALETTE = {
    "skin": [
        np.array([0.98, 0.88, 0.77]),  # pale
        np.array([0.89, 0.71, 0.54]),  # pale/tan
        np.array([0.76, 0.52, 0.33]),  # tan
        np.array([0.55, 0.34, 0.20]),  # brown
        np.array([0.31, 0.16, 0.08]),  # dark brown
    ],
    "robe_blue": [
        np.array([0.2, 0.4, 0.8]),  # standard blue
        np.array([0.1, 0.3, 0.6]),  # dark blue
        np.array([0.4, 0.6, 0.9]),  # light blue
        np.array([0.1, 0.2, 0.5]),  # navy blue
        np.array([0.2, 0.5, 0.85]),  # azure blue
    ],
    "robe_red": [
        np.array([0.8, 0.2, 0.2]),
        np.array([0.6, 0.1, 0.1]),
        np.array([0.9, 0.4, 0.4]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.85, 0.3, 0.2]),
    ],
    "robe_green": [
        np.array([0.2, 0.7, 0.3]),
        np.array([0.1, 0.5, 0.2]),
        np.array([0.4, 0.8, 0.5]),
        np.array([0.05, 0.4, 0.1]),
        np.array([0.3, 0.6, 0.3]),
    ],
    "background": [
        np.array([0.9, 0.95, 1.0]),  # light sky
        np.array([0.85, 0.9, 0.95]),  # slightly cloudy sky
        np.array([0.95, 0.9, 0.85]),  # sunset tint
        np.array([0.8, 0.85, 0.9]),  # darker sky
        np.array([0.95, 0.95, 0.95]),  # neutral grey-white
    ],
}


def generate_base_character(style="robe", canvas_size=(512, 512)) -> CRFTensor:
    """Generates the neutral base pose CRFTensor mapping explicit shapes to slots."""
    crf = CRFTensor()

    cw, ch = canvas_size
    # We normalized everything from 0 to 1, but build relative to internal layout bounds initially
    # Let's say center is (0.5, 0.5)

    # Base palette colors (using index 0 of style)
    c_bg = PALETTE["background"][0]
    c_skin = PALETTE["skin"][0]
    c_robe = PALETTE.get(
        f"robe_{style.split('_')[-1] if '_' in style else 'blue'}", PALETTE["robe_blue"]
    )[0]
    c_eye = np.array([0.1, 0.1, 0.1])
    c_mouth = np.array([0.3, 0.1, 0.1])

    # ---------------------------------------------------------
    # Background
    # ---------------------------------------------------------
    # Slot 0 (BG static): full-canvas background rectangle
    P_bg = rounded_rect_bezier(0.5, 0.5, 1.0, 1.0, 0.0)
    crf.set_shape(0, P_bg, c_bg, alpha=1.0, csg=False)
    crf.activate(0)
    crf.z[0] = 0

    # ---------------------------------------------------------
    # Body (Slots 26-28)
    # ---------------------------------------------------------
    # Slot 26: Robe trapezoid, center-lower canvas
    P_robe = trapezoid_bezier(cx=0.5, cy=0.75, top_w=0.2, bot_w=0.4, h=0.45)
    crf.set_shape(26, P_robe, c_robe, alpha=1.0, csg=False)
    crf.activate(26)
    crf.z[26] = 26

    # Slot 27: Left arm elongated oval (mirrored right later)
    P_l_arm = oval_bezier(cx=0.35, cy=0.75, rx=0.08, ry=0.2)
    crf.set_shape(27, P_l_arm, c_robe * 0.9, alpha=1.0, csg=False)
    crf.activate(27)
    crf.z[27] = 27

    # Slot 28: Right arm elongated oval
    P_r_arm = oval_bezier(cx=0.65, cy=0.75, rx=0.08, ry=0.2)
    crf.set_shape(28, P_r_arm, c_robe * 0.9, alpha=1.0, csg=False)
    crf.activate(28)
    crf.z[28] = 28

    # ---------------------------------------------------------
    # Face (Slots 46-50)
    # ---------------------------------------------------------
    # Slot 46: Head circle, center-upper canvas
    P_head = circle_bezier(cx=0.5, cy=0.35, r=0.15)
    crf.set_shape(46, P_head, c_skin, alpha=1.0, csg=False)
    crf.activate(46)
    crf.z[46] = 46

    # Slot 47: Left eye — small circle
    P_l_eye = circle_bezier(cx=0.44, cy=0.32, r=0.02)
    crf.set_shape(47, P_l_eye, c_eye, alpha=1.0, csg=False)
    crf.activate(47)
    crf.z[47] = 47

    # Slot 48: Right eye — small circle
    P_r_eye = circle_bezier(cx=0.56, cy=0.32, r=0.02)
    crf.set_shape(48, P_r_eye, c_eye, alpha=1.0, csg=False)
    crf.activate(48)
    crf.z[48] = 48

    # Slot 49: Left eyebrow — thin elongated oval, angled
    # Simple straight interpolation representing an angled thin oval
    # Angle it up slightly
    P_l_brow = oval_bezier(cx=0.44, cy=0.27, rx=0.03, ry=0.008)
    # Fake simple rotation perturbation on control points for angle
    for i in range(12):
        x, y = P_l_brow[i]
        dx, dy = x - 0.44, y - 0.27
        P_l_brow[i] = [
            0.44 + dx * 0.966 - dy * 0.259,
            0.27 + dx * 0.259 + dy * 0.966,
        ]  # ~15 deg rot
    crf.set_shape(49, P_l_brow, c_eye, alpha=1.0, csg=False)
    crf.activate(49)
    crf.z[49] = 49

    # Slot 50: Right eyebrow — mirror of left
    P_r_brow = oval_bezier(cx=0.56, cy=0.27, rx=0.03, ry=0.008)
    for i in range(12):
        x, y = P_r_brow[i]
        dx, dy = x - 0.56, y - 0.27
        P_r_brow[i] = [
            0.56 + dx * 0.966 + dy * 0.259,
            0.27 - dx * 0.259 + dy * 0.966,
        ]  # ~-15 deg rot
    crf.set_shape(50, P_r_brow, c_eye, alpha=1.0, csg=False)
    crf.activate(50)
    crf.z[50] = 50

    # ---------------------------------------------------------
    # Mouth (Slots 71-73)
    # ---------------------------------------------------------
    # Slot 71: Mouth base oval, small horizontal
    P_mouth_base = oval_bezier(cx=0.5, cy=0.42, rx=0.05, ry=0.02)
    crf.set_shape(71, P_mouth_base, c_skin * 0.85, alpha=1.0, csg=False)
    crf.activate(71)
    crf.z[71] = 71

    # Slot 72: Upper lip curve (just another thin oval above the base for outline)
    P_upper_lip = oval_bezier(cx=0.5, cy=0.41, rx=0.04, ry=0.01)
    crf.set_shape(72, P_upper_lip, c_mouth, alpha=1.0, csg=False)
    crf.activate(72)
    crf.z[72] = 72

    # Slot 73 (mouth, csg=True): Mouth cavity hole (subtractive)
    # scaled near-zero at rest
    P_mouth_hole = oval_bezier(cx=0.5, cy=0.42, rx=0.001, ry=0.001)
    crf.set_shape(73, P_mouth_hole, np.zeros(3), alpha=1.0, csg=True)
    crf.activate(73)
    crf.z[73] = 73

    return crf


def generate_character_variant(base: CRFTensor, seed: int) -> CRFTensor:
    """Creates a deterministic variation of the base character mapping using random seeds."""
    np.random.seed(seed)

    variant = base.clone()
    active = variant.active_slots()

    # Randomly pick palette indices
    skin_idx = np.random.randint(0, len(PALETTE["skin"]))
    bg_idx = np.random.randint(0, len(PALETTE["background"]))

    # Determine the robe color block used
    robe_color_key = np.random.choice(["robe_blue", "robe_red", "robe_green"])
    robe_idx = np.random.randint(0, len(PALETTE[robe_color_key]))

    # Apply colors to slots
    if 0 in active:
        variant.set_shape(
            0, variant.P[0], PALETTE["background"][bg_idx], variant.alpha[0], csg=False
        )

    for slot in [26, 27, 28]:
        if slot in active:
            shade = 1.0 if slot == 26 else 0.9  # arms slightly darker
            variant.set_shape(
                slot,
                variant.P[slot],
                PALETTE[robe_color_key][robe_idx] * shade,
                variant.alpha[slot],
                csg=False,
            )

    for slot in [46, 71]:  # head, mouth base
        if slot in active:
            shade = 1.0 if slot == 46 else 0.85
            variant.set_shape(
                slot,
                variant.P[slot],
                PALETTE["skin"][skin_idx] * shade,
                variant.alpha[slot],
                csg=False,
            )

    # Apply tiny physical perturbations to control points (+- 0.5% canvas scale randomly)
    noise_idx = np.random.normal(0, 0.005, size=variant.P.shape)

    # Keep shapes locally structurally valid: apply noise primarily as translation + minor vertex jitter
    for slot in active:
        # Translative noise
        tx, ty = np.random.normal(0, 0.01, size=2)

        # Symmetrical face constraints (keep eyes aligned horizontally)
        if slot in [47, 48, 49, 50, 71, 72, 73]:
            # Scale down face perturbations to keep features intact
            tx *= 0.2
            ty *= 0.2
            noise_idx[slot] *= 0.3

        variant.P[slot, :, 0] += tx + noise_idx[slot, :, 0]
        variant.P[slot, :, 1] += ty + noise_idx[slot, :, 1]

    # Enforce strict domain clamp
    variant.P = np.clip(variant.P, 0.0, 1.0).astype(np.float16)

    return variant


def generate_sequence(
    character_fn, motion_fns, noise_config: NoiseConfig, n_frames=192, fps=24
) -> CRFSequence:
    """Generates a complete sequence with truth velocities."""
    base_crf = character_fn()
    frames = []
    dp_dts = []

    rng = np.random.default_rng()

    # Create composed motion wrapper locally to pass to velocity_gt
    # (Avoiding circular import inside function by doing inline)
    def composed_motion(crf, t_val):
        from .motion_functions import compose_motions

        return compose_motions(crf, t_val, motion_fns)

    for i in range(n_frames):
        t = i / fps
        clean_crf = composed_motion(base_crf, t)
        noisy_crf = apply_noise(clean_crf, noise_config, rng)

        frames.append(noisy_crf)

        v = compute_velocity_gt(composed_motion, base_crf, t, dt=0.1)
        dp_dts.append(v)

    return CRFSequence(frames, dp_dt=np.stack(dp_dts, axis=0))


def generate_dataset(n_sequences: int, noise_config: NoiseConfig, output_path: str):
    """Generates an entire dataset to HDF5."""
    os.makedirs(output_path, exist_ok=True)
    from .motion_functions import breathing_motion, eye_blink, gentle_sway

    print(f"Generating {n_sequences} sequences for {noise_config.stage} stage...")
    for i in tqdm(range(n_sequences)):
        seed = i

        def make_char():
            base = generate_base_character(style="robe")
            return generate_character_variant(base, seed)

        seq = generate_sequence(
            character_fn=make_char,
            motion_fns=[breathing_motion, gentle_sway, eye_blink],
            noise_config=noise_config,
            n_frames=192,
            fps=24,
        )
        filepath = os.path.join(output_path, f"seq_{i:04d}_{noise_config.stage}.h5")
        seq.to_hdf5(filepath)


def generate_template_library() -> dict:
    """Generate 50 base poses for the dictionary initialization."""
    out_dir = os.path.join("data", "templates", "base_poses")
    os.makedirs(out_dir, exist_ok=True)
    templates = {}

    base = generate_base_character(style="robe")
    for i in range(50):
        variant = generate_character_variant(base, seed=1000 + i)
        filename = f"template_{i:03d}.json"
        filepath = os.path.join(out_dir, filename)

        with open(filepath, "w") as f:
            json.dump(variant.to_json(), f, indent=2)

        templates[filename] = f"Template pose {i}"

    return templates
