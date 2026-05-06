"""
detection/violation_logic.py
=============================
Pure violation-checking logic.

No model dependencies — operates only on detected bounding boxes.
Easy to unit test independently of any model or video.

All boxes are (x1, y1, x2, y2) in pixels relative to the motorcycle crop.
"""

from dataclasses import dataclass, field
from utils.box_utils import overlap_ratio


@dataclass
class ViolationResult:
    violations:       list  = field(default_factory=list)
    helmet_count:     int   = 0
    face_count:       int   = 0
    unhelmeted_count: int   = 0
    helmeted_faces:   list  = field(default_factory=list)
    # helmets that are spatially near a rider (upper half of crop)
    valid_helmet_count: int = 0


def _is_rider_region(box: tuple, crop_h: int, crop_w: int = 9999,
                     threshold: float = 0.65) -> bool:
    """
    Returns True if the box looks like a helmet on a rider (not a bike part).

    Three checks:
    1. Position : centre-y must be in the top `threshold` of the crop.
                  Helmets are always in the upper 65% — reflectors/tail-lights
                  at the bottom are rejected.
    2. Min size : box must be at least 2% of crop height tall. Tiny detections
                  are noise.
    3. Max size : box must not exceed 25% of crop width. Genuine helmet boxes
                  are small — very wide boxes are body/torso false positives.
    """
    x1, y1, x2, y2 = box
    centre_y  = (y1 + y2) / 2
    box_h     = y2 - y1
    box_w     = x2 - x1

    if centre_y >= crop_h * threshold:
        return False
    if box_h < crop_h * 0.02:          # too small — noise
        return False
    if box_w > crop_w * 0.25:          # too wide — not a helmet
        return False
    return True


def check_violations(
    helmet_boxes:               list,
    face_boxes:                 list,
    crop_h:                     int   = 9999,
    crop_w:                     int   = 9999,
    face_helmet_overlap_thresh: float = 0.60,
    helmet_position_thresh:     float = 0.65,
) -> ViolationResult:
    """
    Determine which violations are present for a single motorcycle.

    Parameters
    ----------
    helmet_boxes
        Detected helmet boxes inside the motorcycle crop.
    face_boxes
        Detected face boxes inside the motorcycle crop.
    crop_h
        Height of the motorcycle crop in pixels. Used to filter out
        helmets detected in the lower portion of the frame (false positives
        from reflectors, tail-lights, etc.).
    face_helmet_overlap_thresh
        Face overlapping a helmet box by more than this is considered
        helmeted and removed from the bare-rider count. Default: 0.60
    helmet_position_thresh
        Helmets whose centre-y is below this fraction of crop_h are
        discarded as false positives. Default: 0.75 (top 75% of crop)

    Logic
    -----
    Step 4c  Filter positional false positives
        Discard any helmet box whose centre is in the bottom 25% of the
        crop — these are almost always bike parts, not rider helmets.

    Step 4d  Suppress helmeted faces
        If a face overlaps a valid helmet by > 60%, that face is helmeted.

    Step 5   No Helmet Violation
        Triggered if any_rider AND (no valid helmets OR unhelmeted face exists)

    Step 6   Triple Riding Violation
        total_riders = valid_helmet_count + face_count
        Triggered if total_riders > 2
    """
    result = ViolationResult(
        helmet_count = len(helmet_boxes),
        face_count   = len(face_boxes),
    )

    # ── Step 4c: discard helmets in bottom portion of crop ────────────────────
    # Pass crop_w so the width check inside _is_rider_region works
    valid_helmets = [
        b for b in helmet_boxes
        if _is_rider_region(b, crop_h, crop_w, helmet_position_thresh)
    ]
    result.valid_helmet_count = len(valid_helmets)

    # ── Step 4d: identify helmeted vs bare faces ──────────────────────────────
    unhelmeted_faces = []
    helmeted_indices = []

    for i, face_box in enumerate(face_boxes):
        is_helmeted = any(
            overlap_ratio(face_box, h_box) > face_helmet_overlap_thresh
            for h_box in valid_helmets
        )
        if is_helmeted:
            helmeted_indices.append(i)
        else:
            unhelmeted_faces.append(face_box)

    result.unhelmeted_count = len(unhelmeted_faces)
    result.helmeted_faces   = helmeted_indices

    # ── Step 5: No Helmet Violation ───────────────────────────────────────────
    any_rider = result.valid_helmet_count > 0 or result.face_count > 0
    no_helmet_violated = any_rider and (
        result.valid_helmet_count == 0
        or result.unhelmeted_count > 0
    )
    if no_helmet_violated:
        result.violations.append("No Helmet")

    # ── Step 6: Triple Riding Violation ───────────────────────────────────────
    total_riders = result.valid_helmet_count + result.face_count
    if total_riders > 2:
        result.violations.append("Triple Riding")

    return result