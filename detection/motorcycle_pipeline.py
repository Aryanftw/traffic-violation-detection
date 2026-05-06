"""
detection/motorcycle_pipeline.py
=================================
Orchestrates the full per-motorcycle analysis pipeline.

Given a detected motorcycle bounding box and the full frame, this module:
  1. Crops and pads the motorcycle region
  2. Runs helmet and face detection on the crop  (model_inference)
  3. Evaluates violations                         (violation_logic)
  4. Runs OCR on the license plate if violated    (ocr_utils)
  5. Draws all annotations onto a copy of the frame (draw_utils)
  6. Returns everything the main loop needs to log and save

This file coordinates but does not duplicate logic — each concern stays
in its own module.
"""

import numpy as np

from detection.model_inference import Models
from detection.violation_logic import check_violations, ViolationResult
from utils.box_utils import pad_box, translate_box
from utils.draw_utils import draw_box, draw_violation_banner, COLORS
from utils.ocr_utils import read_plate


def process_motorcycle(
    frame:      np.ndarray,
    moto_box:   tuple,
    models:     Models,
    frame_id:   int,
    moto_idx:   int,
    ocr_gpu:    bool = False,
    crop_pad:   float = 0.12,
    face_helmet_overlap: float = 0.60,
) -> dict:
    """
    Run the full violation pipeline for one detected motorcycle.

    Parameters
    ----------
    frame       : full BGR video frame
    moto_box    : (x1,y1,x2,y2) of the motorcycle in the full frame
    models      : loaded Models instance
    frame_id    : current frame number (for logging)
    moto_idx    : index of this motorcycle within the frame
    ocr_gpu     : whether to use GPU for EasyOCR
    crop_pad    : how much to pad the motorcycle crop on each side
    face_helmet_overlap : threshold for face-helmet suppression (step 4c)

    Returns
    -------
    dict with keys:
        annotated_frame   np.ndarray  — frame with boxes + violation banner drawn
        moto_crop         np.ndarray  — annotated crop of just the motorcycle
        plate_crop        np.ndarray | None
        plate_text        str
        violations        list[str]
        result            ViolationResult
        has_violation     bool
    """
    fh, fw = frame.shape[:2]

    # ── Step 2: Pad and crop the motorcycle region ────────────────────────────
    cx1, cy1, cx2, cy2 = pad_box(moto_box, fh, fw, pad=crop_pad)
    crop = frame[cy1:cy2, cx1:cx2].copy()

    if crop.size == 0:
        return _empty_result(frame)

    # ── Step 4a: Detect helmets and plates within the crop ────────────────────
    helmet_dets = models.detect_helmets(crop)
    plate_dets  = models.detect_plates(crop)

    # ── Step 4b: Detect faces within the crop ─────────────────────────────────
    face_dets = models.detect_faces(crop)

    # ── Step 4b-filter: Remove faces that are implausibly outside the rider area
    # The face model runs on the padded crop but pedestrians walking in front
    # of the motorcycle can have their face box land inside the crop region.
    # We discard any face whose centre is above the very top of the crop
    # (y_centre < crop_h * 0.10) as these are almost always background people,
    # or whose box is larger than 60% of the crop width (not a face, a false pos).
    crop_h_px, crop_w_px = crop.shape[:2]
    filtered_face_dets = []
    for fd in face_dets:
        x1, y1, x2, y2 = fd.box
        face_w     = x2 - x1
        centre_y   = (y1 + y2) / 2
        # Discard faces whose centre is in top 10% of crop (background pedestrian)
        if centre_y < crop_h_px * 0.10:
            continue
        # Discard giant face boxes (false positives covering whole image)
        face_h = y2 - y1
        if face_w > crop_w_px * 0.40:
            continue
        # Discard face boxes taller than 50% of crop (torso/body false positives)
        if face_h > crop_h_px * 0.50:
            continue
        # Discard boxes with bad aspect ratio — real faces are roughly square.
        # A box that is 2x taller than it is wide is a torso/backpack, not a face.
        aspect = face_h / face_w if face_w > 0 else 99
        if aspect > 2.0:
            continue
        filtered_face_dets.append(fd)
    face_dets = filtered_face_dets

    # ── Deduplicate stacked helmet boxes with NMS ─────────────────────────────
    # Multiple boxes stacking on the same object (reflector, tail-light) is a
    # common false positive. Apply NMS with aggressive IoU threshold to keep
    # only one box per region.
    from detection.model_inference import Models as _M
    helmet_dets = _M._nms(helmet_dets, iou_thresh=0.30)

    # Extract raw box lists for the logic layer
    helmet_boxes = [d.box for d in helmet_dets]
    face_boxes   = [d.box for d in face_dets]

    # ── Steps 4c / 5 / 6: Evaluate violations (pure logic, no CV) ────────────
    result: ViolationResult = check_violations(
        helmet_boxes               = helmet_boxes,
        face_boxes                 = face_boxes,
        crop_h                     = crop.shape[0],
        crop_w                     = crop.shape[1],   # for helmet width filter
        face_helmet_overlap_thresh = face_helmet_overlap,
    )

    # ── Step 7: OCR the plate if there is any violation ───────────────────────
    plate_text  = ""
    plate_crop  = None
    abs_plate_box = None

    if result.violations and plate_dets:
        # Take the highest-confidence plate
        best_plate = max(plate_dets, key=lambda d: d.conf)
        # Translate crop-relative box → full-frame coordinates
        abs_plate_box = translate_box(best_plate.box, cx1, cy1)
        px1, py1, px2, py2 = pad_box(abs_plate_box, fh, fw, pad=0.05)
        plate_crop = frame[py1:py2, px1:px2].copy()
        plate_text = read_plate(plate_crop, gpu=ocr_gpu)

    # ── Draw annotations onto a working copy of the frame ────────────────────
    annotated = frame.copy()

    # Motorcycle box
    moto_label = "MOTORCYCLE"
    if result.violations:
        moto_label = "⚠ " + " | ".join(result.violations)
    draw_box(annotated, moto_box, moto_label,
             COLORS["violation"] if result.violations else COLORS["motorcycle"],
             thickness=3)

    # Helmets (translate crop-relative back to frame)
    for d in helmet_dets:
        fb = translate_box(d.box, cx1, cy1)
        draw_box(annotated, fb, f"Helmet {d.conf:.2f}", COLORS["helmet"])

    # Faces — colour differently if unhelmeted
    for i, d in enumerate(face_dets):
        fb    = translate_box(d.box, cx1, cy1)
        color = COLORS["helmet"] if i in result.helmeted_faces else COLORS["face"]
        label = f"Face {'✓' if i in result.helmeted_faces else '✗'} {d.conf:.2f}"
        draw_box(annotated, fb, label, color)

    # Plate box
    if abs_plate_box:
        draw_box(annotated, abs_plate_box,
                 f"Plate: {plate_text or '?'}", COLORS["plate"])

    # Top banner if violations present
    if result.violations:
        draw_violation_banner(
            annotated,
            result.violations,
            result.valid_helmet_count,   # filtered count, not raw
            result.face_count,
        )

    # Extract annotated motorcycle crop for saving
    moto_crop = annotated[cy1:cy2, cx1:cx2].copy()

    return {
        "annotated_frame" : annotated,
        "moto_crop"       : moto_crop,
        "plate_crop"      : plate_crop,
        "plate_text"      : plate_text,
        "violations"      : result.violations,
        "result"          : result,
        "has_violation"   : len(result.violations) > 0,
    }


def _empty_result(frame: np.ndarray) -> dict:
    """Return a no-op result when the crop is invalid."""
    return {
        "annotated_frame" : frame,
        "moto_crop"       : None,
        "plate_crop"      : None,
        "plate_text"      : "",
        "violations"      : [],
        "result"          : ViolationResult(),
        "has_violation"   : False,
    }