"""
tests/test_violation_logic.py
==============================
Unit tests for the violation rules.
No models, no images — just box coordinates and expected outcomes.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.violation_logic import check_violations


# ── helpers ───────────────────────────────────────────────────────────────────

def b(x1, y1, x2, y2):
    """Shorthand box constructor."""
    return (x1, y1, x2, y2)


# ── No Helmet tests ───────────────────────────────────────────────────────────

def test_no_violation_single_helmeted_rider():
    """One helmet, one face that overlaps it → helmeted → no violation."""
    helmets = [b(10, 10, 50, 50)]
    faces   = [b(12, 12, 48, 48)]   # almost fully inside the helmet box
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert result.violations == [], f"Expected no violations, got {result.violations}"


def test_no_helmet_no_helmets_at_all():
    """No helmets detected at all → No Helmet violation."""
    helmets = []
    faces   = [b(10, 10, 40, 40)]
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "No Helmet" in result.violations


def test_no_helmet_bare_face_present():
    """One helmet, but one face is NOT overlapping it → bare rider → violation."""
    helmets = [b(10, 10, 50, 50)]
    faces   = [b(200, 200, 240, 240)]   # completely elsewhere in the crop
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "No Helmet" in result.violations
    assert result.unhelmeted_count == 1


def test_no_helmet_multiple_faces_one_helmeted():
    """Two faces, only one helmeted → violation because one is bare."""
    helmets = [b(10, 10, 50, 50)]
    faces   = [
        b(12, 12, 48, 48),    # helmeted (overlaps helmet box heavily)
        b(100, 100, 140, 140), # bare
    ]
    result = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "No Helmet" in result.violations
    assert result.unhelmeted_count == 1
    assert len(result.helmeted_faces) == 1


# ── Triple Riding tests ───────────────────────────────────────────────────────

def test_triple_riding_three_helmeted():
    """Three helmets → three riders → Triple Riding violation."""
    helmets = [b(0,0,30,30), b(40,0,70,30), b(80,0,110,30)]
    faces   = []
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "Triple Riding" in result.violations


def test_triple_riding_two_helmets_one_extra_face():
    """Two helmets + one bare face = 3 total → Triple Riding."""
    helmets = [b(0,0,30,30), b(40,0,70,30)]
    faces   = [b(200,200,230,230)]   # bare face, no overlap with helmets
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "Triple Riding" in result.violations
    assert "No Helmet" in result.violations   # also bare


def test_no_triple_riding_two_riders():
    """Two riders exactly → no Triple Riding."""
    helmets = [b(0,0,30,30), b(40,0,70,30)]
    faces   = []
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "Triple Riding" not in result.violations


def test_no_triple_riding_one_helmet_one_face_helmeted():
    """One helmet, one face overlapping it → 1 helmet + 1 face = 2 total → no triple."""
    helmets = [b(10,10,50,50)]
    faces   = [b(12,12,48,48)]
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "Triple Riding" not in result.violations


# ── Both violations ───────────────────────────────────────────────────────────

def test_both_violations():
    """No helmets AND 3 faces → both violations fire."""
    helmets = []
    faces   = [b(0,0,20,20), b(30,0,50,20), b(60,0,80,20)]
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500)
    assert "No Helmet"     in result.violations
    assert "Triple Riding" in result.violations


# ── Overlap threshold tests ───────────────────────────────────────────────────

def test_overlap_threshold_boundary():
    """Face that overlaps at exactly the threshold is NOT suppressed."""
    # helmet box 0,0 → 100,100  (area 10000)
    # face box   0,0 →  60, 60  overlaps 3600 / 3600 = 100 % of face area
    # → well above 60 % → helmeted
    helmets = [b(0, 0, 100, 100)]
    faces   = [b(0, 0,  60,  60)]
    result  = check_violations(helmets, faces, crop_h=500, crop_w=500, face_helmet_overlap_thresh=0.60)
    assert result.unhelmeted_count == 0, "Face should be considered helmeted"
    assert "No Helmet" not in result.violations


def test_overlap_below_threshold():
    """Face that overlaps below the threshold IS counted as bare."""
    # face 0,0→20,20  area=400
    # helmet 50,50→100,100 → no overlap at all
    helmets = [b(50, 50, 100, 100)]
    faces   = [b(0, 0, 20, 20)]
    result  = check_violations(helmets, faces, face_helmet_overlap_thresh=0.60)
    assert result.unhelmeted_count == 1
    assert "No Helmet" in result.violations


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_frame():
    """Nothing detected → no violations."""
    result = check_violations([], [], crop_h=500, crop_w=500)
    assert result.violations == []
    assert result.helmet_count == 0
    assert result.face_count == 0


def test_helmets_only_no_faces():
    """Two helmets, no faces → no violation (riders are compliant)."""
    helmets = [b(0,0,30,30), b(40,0,70,30)]
    result  = check_violations(helmets, [])
    assert result.violations == []


if __name__ == "__main__":
    # Can also be run directly without pytest
    tests = [
        test_no_violation_single_helmeted_rider,
        test_no_helmet_no_helmets_at_all,
        test_no_helmet_bare_face_present,
        test_no_helmet_multiple_faces_one_helmeted,
        test_triple_riding_three_helmeted,
        test_triple_riding_two_helmets_one_extra_face,
        test_no_triple_riding_two_riders,
        test_no_triple_riding_one_helmet_one_face_helmeted,
        test_both_violations,
        test_overlap_threshold_boundary,
        test_overlap_below_threshold,
        test_empty_frame,
        test_helmets_only_no_faces,
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