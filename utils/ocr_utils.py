"""
utils/ocr_utils.py
==================
EasyOCR-based license plate text extraction.
Tries multiple preprocessing strategies and returns the best result.
"""

import cv2
import numpy as np
import easyocr

_reader = None


def get_reader(gpu: bool = False) -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[OCR] Initialising EasyOCR reader ...")
        _reader = easyocr.Reader(["en"], gpu=gpu)
        print("[OCR] Ready.")
    return _reader


def _preprocess_variants(crop: np.ndarray) -> list:
    """
    Return a list of preprocessed versions of the plate crop.
    EasyOCR is run on each; the result with the most characters wins.
    """
    variants = []

    # Always upscale first — small plates need this
    h, w = crop.shape[:2]
    scale = max(1, int(100 / h))           # ensure at least 100px tall
    upscaled = cv2.resize(crop, (w * scale * 2, h * scale * 2),
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Variant 1 — raw grayscale upscaled
    variants.append(gray)

    # Variant 2 — adaptive threshold (handles uneven lighting)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    variants.append(adapt)

    # Variant 3 — Otsu threshold after denoising
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(denoised, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Variant 4 — inverted Otsu (dark text on light bg OR light on dark)
    variants.append(cv2.bitwise_not(otsu))

    # Variant 5 — CLAHE (contrast limited adaptive histogram equalisation)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    variants.append(eq)

    return variants


def read_plate(crop: np.ndarray, gpu: bool = False) -> str:
    """
    Extract text from a license plate crop.
    Tries multiple preprocessing variants and returns the longest result.

    Args:
        crop : BGR image of the plate region
        gpu  : whether to use GPU for EasyOCR

    Returns:
        Plate text string, or '' if nothing found
    """
    if crop is None or crop.size == 0:
        return ""

    reader   = get_reader(gpu=gpu)
    variants = _preprocess_variants(crop)

    best = ""
    for img in variants:
        try:
            results = reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
                detail=1,
                paragraph=False,
                min_size=10,
            )
            if results:
                # Pick the reading with highest total confidence
                text = " ".join(
                    r[1] for r in sorted(results, key=lambda r: r[2], reverse=True)
                    if r[2] > 0.1           # skip very low confidence chars
                ).strip()
                if len(text) > len(best):
                    best = text
        except Exception:
            continue

    # Clean up: remove stray spaces within likely plate numbers
    best = best.upper().strip()
    return best