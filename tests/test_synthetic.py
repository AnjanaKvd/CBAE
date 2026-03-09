import pytest
import numpy as np
import os
import tempfile

from core.crf_tensor import CRFTensor, CRFSequence
from generation.synthetic import generate_base_character, generate_sequence
from generation.motion_functions import (
    compute_velocity_gt,
    gentle_sway,
    breathing_motion,
)
from generation.noise_schedule import apply_noise, NoiseConfig, NOISE_CLEAN


def test_character_slot_assignments():
    crf = generate_base_character(style="robe")
    active = crf.active_slots()

    # Check that main body slot is active (slot 26 usually for robe body)
    assert 26 in active
    # Face base should be active (slot 46)
    assert 46 in active
    # Mouth should be active (slot 71)
    assert 71 in active


def test_motion_continuity():
    base = generate_base_character()

    crf1 = gentle_sway(base, 0.0)
    crf2 = gentle_sway(base, 0.01)

    # Check max delta is small between very small time steps
    max_delta = np.max(np.abs(crf1.P - crf2.P))
    assert max_delta < 0.05

    crf3 = breathing_motion(base, 0.0)
    crf4 = breathing_motion(base, 0.01)
    max_delta2 = np.max(np.abs(crf3.P - crf4.P))
    assert max_delta2 < 0.05


def test_velocity_accuracy():
    base = generate_base_character()
    t = 1.0

    # Compute numerical velocity
    v_num = compute_velocity_gt(gentle_sway, base, t, dt=0.1)

    # Analytical derivative of gentle_sway:
    # dx(t) = 0.01 * sin(2 * pi * 0.1 * t)
    # v_x(t) = d(dx)/dt = 0.01 * 2 * pi * 0.1 * cos(2 * pi * 0.1 * t)
    v_analytic_x = 0.01 * 2 * np.pi * 0.1 * np.cos(2 * np.pi * 0.1 * t)

    # Gentle sway applies horizontal translation to slots 26, 27, 28.
    # Let's check one active control point from slot 26
    v_num_x = v_num[26, 0, 0]  # slot 26, point 0, x-coord

    err = abs(v_num_x - v_analytic_x)
    assert err < 1e-3


def test_noise_injection():
    base = generate_base_character()
    rng = np.random.default_rng(42)

    config = NoiseConfig(
        control_point_jitter=0.01,
        color_noise=0.01,
        slot_assignment_error_rate=0.0,
        stage="test",
    )

    noised = apply_noise(base, config, rng)

    # Check it changed compared to base
    assert not np.allclose(base.P, noised.P)
    assert not np.allclose(base.c, noised.c)


def test_sequence_hdf5_roundtrip():
    # Use empty/simple motion for speed
    def simple_motion(crf, t):
        return crf

    seq = generate_sequence(
        lambda: generate_base_character(),
        [simple_motion],
        NOISE_CLEAN,
        n_frames=2,
        fps=24,
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    try:
        seq.to_hdf5(path)
        seq2 = CRFSequence.from_hdf5(path)

        assert len(seq2.frames) == len(seq.frames)
        assert np.allclose(seq2.frames[0].P, seq.frames[0].P)
        assert np.allclose(seq2.dp_dt, seq.dp_dt)
    finally:
        os.remove(path)
