import pytest
import numpy as np
import torch
import os
import tempfile
from core.crf_tensor import CRFTensor, CRFSequence
from core.constants import (
    N_SLOTS,
    N_CTRL_PTS,
    SLOT_BG_STATIC,
    SLOT_BG_DYNAMIC,
    SLOT_BODY,
    SLOT_FACE,
    SLOT_MOUTH,
    SLOT_SECONDARY,
    SLOT_DYNAMIC,
)


def test_crf_serialize_roundtrip():
    crf = CRFTensor()
    crf.P[0, 0] = [0.5, 0.5]
    crf.c[0] = [1.0, 0.0, 0.0]
    crf.activate(0)

    data = crf.to_json()
    crf2 = CRFTensor.from_json(data)

    assert np.allclose(crf.P, crf2.P)
    assert np.allclose(crf.c, crf2.c)
    assert np.allclose(crf.alive, crf2.alive)


def test_crf_binary_roundtrip():
    crf = CRFTensor()
    crf.P[1, 1] = [0.25, 0.75]
    crf.c[1] = [0.0, 1.0, 0.0]
    crf.activate(1)

    data = crf.to_binary()
    crf2 = CRFTensor.from_binary(data)

    assert np.allclose(crf.P, crf2.P)
    assert np.allclose(crf.c, crf2.c)
    assert np.allclose(crf.alive, crf2.alive)


def test_crf_torch_roundtrip():
    crf = CRFTensor()
    crf.P[2, 2] = [0.1, 0.9]
    crf.c[2] = [0.0, 0.0, 1.0]
    crf.activate(2)

    tensors = crf.to_torch()
    crf2 = CRFTensor.from_torch(tensors)

    assert np.allclose(crf.P, crf2.P)
    assert np.allclose(crf.c, crf2.c)
    assert np.allclose(crf.alive, crf2.alive)


def test_active_slots_threshold():
    crf = CRFTensor()
    crf.activate(0)
    crf.activate(3)

    # default threshold is 0.1
    active_indices = crf.active_slots()
    assert len(active_indices) == 2
    assert 0 in active_indices
    assert 3 in active_indices
    assert 1 not in active_indices


def test_slot_block_assignment():
    crf = CRFTensor()
    assert crf.slot_block(SLOT_BG_STATIC[0]) == "bg_static"
    assert crf.slot_block(SLOT_BG_DYNAMIC[0]) == "bg_dynamic"
    assert crf.slot_block(SLOT_BODY[0]) == "body"
    assert crf.slot_block(SLOT_FACE[0]) == "face"
    assert crf.slot_block(SLOT_MOUTH[0]) == "mouth"
    assert crf.slot_block(SLOT_SECONDARY[0]) == "secondary"
    assert crf.slot_block(SLOT_DYNAMIC[0]) == "dynamic"


def test_crf_sequence_hdf5():
    crf1 = CRFTensor()
    crf1.activate(0)
    crf2 = CRFTensor()
    crf2.activate(1)

    seq = CRFSequence([crf1, crf2], dp_dt=np.array([0.1, 0.2]))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    try:
        seq.to_hdf5(path)
        seq2 = CRFSequence.from_hdf5(path)

        assert len(seq2.frames) == 2
        assert np.allclose(seq2.frames[0].alive, crf1.alive)
        assert np.allclose(seq2.frames[1].alive, crf2.alive)
        assert np.allclose(seq2.dp_dt, seq.dp_dt)
    finally:
        os.remove(path)
