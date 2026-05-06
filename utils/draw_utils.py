"""
utils/draw_utils.py
===================
OpenCV drawing helpers for annotating frames.
"""

import cv2
import numpy as np

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
COLORS = {
    "motorcycle" : (255, 165,   0),   # orange
    "helmet"     : (  0, 210,   0),   # green
    "face"       : (  0, 200, 220),   # cyan
    "plate"      : (200,   0, 200),   # magenta
    "violation"  : (  0,   0, 230),   # red
    "no_helmet"  : (  0,  60, 230),   # red-orange
    "triple"     : (  0,   0, 180),   # deep red
}


def draw_box(
    img: np.ndarray,
    box: tuple,
    label: str,
    color: tuple,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw a labelled bounding box on `img` (in-place).

    Args:
        img:        BGR image
        box:        (x1, y1, x2, y2)
        label:      text displayed above the box
        color:      BGR tuple
        thickness:  border thickness in pixels
        font_scale: text size

    Returns:
        The same image (modified in-place, also returned for chaining)
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    label_y = max(y1, th + 6)
    cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 4, label_y), color, -1)
    cv2.putText(
        img, label,
        (x1 + 2, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def draw_violation_banner(
    img: np.ndarray,
    violations: list[str],
    helmet_count: int,
    face_count: int,
) -> np.ndarray:
    """
    Draw a solid red banner across the top of the frame listing violations.

    Args:
        img:          BGR image
        violations:   list of violation strings (may be empty)
        helmet_count: number of helmets detected on this motorcycle
        face_count:   number of faces detected on this motorcycle

    Returns:
        Annotated image (in-place)
    """
    if not violations:
        return img

    h, w = img.shape[:2]
    banner_h = 34
    cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 0), -1)

    vio_text  = "  !!  " + "  |  ".join(violations)
    info_text = f"Helmets: {helmet_count}   Faces: {face_count}"
    full_text = f"{vio_text}       {info_text}"

    cv2.putText(
        img, full_text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (0, 60, 255), 2, cv2.LINE_AA,
    )
    return img


def draw_frame_counter(img: np.ndarray, frame_id: int) -> np.ndarray:
    """Stamp frame number in the bottom-right corner."""
    h, w = img.shape[:2]
    cv2.putText(
        img, f"Frame {frame_id}",
        (w - 130, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (180, 180, 180), 1, cv2.LINE_AA,
    )
    return img