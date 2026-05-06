"""
utils/box_utils.py
==================
Pure geometry helpers for bounding box operations.
No model dependencies — easy to test independently.
"""


def overlap_ratio(box_a: tuple, box_b: tuple) -> float:
    """
    Returns what fraction of box_a's area is covered by box_b.

    This is intentionally NOT symmetric IoU.
    We use it to ask: "Is this face mostly inside this helmet box?"

    Args:
        box_a: (x1, y1, x2, y2) — the box whose coverage we measure
        box_b: (x1, y1, x2, y2) — the reference box

    Returns:
        float in [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    box_a_area = max(1, (ax2 - ax1) * (ay2 - ay1))

    return inter_area / box_a_area


def iou(box_a: tuple, box_b: tuple) -> float:
    """
    Standard Intersection-over-Union between two boxes.

    Args:
        box_a, box_b: (x1, y1, x2, y2)

    Returns:
        float in [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def pad_box(box: tuple, frame_h: int, frame_w: int, pad: float = 0.10) -> tuple:
    """
    Expand a bounding box by `pad` fraction of its own dimensions,
    clamped to stay inside the frame.

    Args:
        box:     (x1, y1, x2, y2)
        frame_h: frame height in pixels
        frame_w: frame width  in pixels
        pad:     fraction to expand on each side (default 10 %)

    Returns:
        Expanded (x1, y1, x2, y2) clamped to frame
    """
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    return (
        max(0,       x1 - pw),
        max(0,       y1 - ph),
        min(frame_w, x2 + pw),
        min(frame_h, y2 + ph),
    )


def translate_box(box: tuple, offset_x: int, offset_y: int) -> tuple:
    """
    Shift a box by (offset_x, offset_y).

    Used to convert crop-relative coordinates back to full-frame coordinates.

    Args:
        box:      (x1, y1, x2, y2) relative to crop origin
        offset_x: x-coordinate of crop origin in the full frame
        offset_y: y-coordinate of crop origin in the full frame

    Returns:
        (x1, y1, x2, y2) in full-frame coordinates
    """
    x1, y1, x2, y2 = box
    return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)