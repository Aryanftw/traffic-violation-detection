"""
tests/test_box_utils.py
========================
Unit tests for bounding box geometry helpers.

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.box_utils import overlap_ratio, iou, pad_box, translate_box


def test_overlap_ratio_full():
    """Box A fully inside box B → 100 % overlap."""
    a = (10, 10, 20, 20)
    b = (0,  0,  30, 30)
    assert overlap_ratio(a, b) == 1.0


def test_overlap_ratio_none():
    """Non-overlapping boxes → 0 %."""
    a = (0,  0, 10, 10)
    b = (20, 20, 30, 30)
    assert overlap_ratio(a, b) == 0.0


def test_overlap_ratio_partial():
    """50 % of a covered by b."""
    a = (0,  0, 10, 10)   # area = 100
    b = (5,  0, 15, 10)   # overlap = 5×10 = 50  → 50/100 = 0.5
    assert abs(overlap_ratio(a, b) - 0.5) < 1e-6


def test_iou_perfect():
    a = (0, 0, 10, 10)
    assert abs(iou(a, a) - 1.0) < 1e-6


def test_iou_no_overlap():
    a = (0, 0, 5, 5)
    b = (10, 10, 15, 15)
    assert iou(a, b) == 0.0


def test_pad_box_clamped():
    """Padded box must not exceed frame boundaries."""
    box    = (0, 0, 100, 100)
    result = pad_box(box, frame_h=200, frame_w=200, pad=0.50)
    assert result[0] >= 0 and result[1] >= 0
    assert result[2] <= 200 and result[3] <= 200


def test_translate_box():
    box    = (10, 20, 30, 40)
    result = translate_box(box, offset_x=5, offset_y=15)
    assert result == (15, 35, 35, 55)


if __name__ == "__main__":
    tests = [
        test_overlap_ratio_full,
        test_overlap_ratio_none,
        test_overlap_ratio_partial,
        test_iou_perfect,
        test_iou_no_overlap,
        test_pad_box_clamped,
        test_translate_box,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}  →  {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")