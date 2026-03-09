from typing import List

from .constants import (
    SLOT_BG_STATIC,
    SLOT_BG_DYNAMIC,
    SLOT_BODY,
    SLOT_FACE,
    SLOT_MOUTH,
    SLOT_SECONDARY,
    SLOT_DYNAMIC,
    DELTA_MAX,
)


def get_slot_block_name(slot_idx: int) -> str:
    """Returns the semantic block name for a given slot index."""
    if SLOT_BG_STATIC[0] <= slot_idx <= SLOT_BG_STATIC[1]:
        return "bg_static"
    elif SLOT_BG_DYNAMIC[0] <= slot_idx <= SLOT_BG_DYNAMIC[1]:
        return "bg_dynamic"
    elif SLOT_BODY[0] <= slot_idx <= SLOT_BODY[1]:
        return "body"
    elif SLOT_FACE[0] <= slot_idx <= SLOT_FACE[1]:
        return "face"
    elif SLOT_MOUTH[0] <= slot_idx <= SLOT_MOUTH[1]:
        return "mouth"
    elif SLOT_SECONDARY[0] <= slot_idx <= SLOT_SECONDARY[1]:
        return "secondary"
    elif SLOT_DYNAMIC[0] <= slot_idx <= SLOT_DYNAMIC[1]:
        return "dynamic"
    return "unknown"


def get_slots_in_block(block_name: str) -> List[int]:
    """Returns a list of slot indices for a given block name."""
    if block_name == "bg_static":
        return list(range(SLOT_BG_STATIC[0], SLOT_BG_STATIC[1] + 1))
    elif block_name == "bg_dynamic":
        return list(range(SLOT_BG_DYNAMIC[0], SLOT_BG_DYNAMIC[1] + 1))
    elif block_name == "body":
        return list(range(SLOT_BODY[0], SLOT_BODY[1] + 1))
    elif block_name == "face":
        return list(range(SLOT_FACE[0], SLOT_FACE[1] + 1))
    elif block_name == "mouth":
        return list(range(SLOT_MOUTH[0], SLOT_MOUTH[1] + 1))
    elif block_name == "secondary":
        return list(range(SLOT_SECONDARY[0], SLOT_SECONDARY[1] + 1))
    elif block_name == "dynamic":
        return list(range(SLOT_DYNAMIC[0], SLOT_DYNAMIC[1] + 1))
    return []


def get_delta_max(slot_idx: int) -> float:
    """Returns the maximum deformation (delta) allowed for a given slot."""
    name = get_slot_block_name(slot_idx)
    if name in ("bg_static", "bg_dynamic"):
        return DELTA_MAX.get("background", 0.0)
    return DELTA_MAX.get(name, 0.0)


def is_mouth_slot(slot_idx: int) -> bool:
    """Returns True if the slot index is in the mouth region."""
    return SLOT_MOUTH[0] <= slot_idx <= SLOT_MOUTH[1]


def is_audio_conditioned(slot_idx: int) -> bool:
    """Returns True if the slot is audio-conditioned (same as mouth for v1)."""
    return is_mouth_slot(slot_idx)


def z_order_from_slot(slot_idx: int) -> int:
    """Returns the z-order for a slot (lower slots render first/behind)."""
    # By architecture definition: slots are ordered sequentially from background to foreground.
    # Therefore, the slot index itself perfectly represents the z-order.
    return slot_idx
