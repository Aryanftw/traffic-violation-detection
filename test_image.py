"""
test_image.py
=============
Run the full violation detection pipeline on a single image.
Useful for quickly verifying your trained models work correctly
before running on video.

Usage
-----
  python test_image.py \
      --helmet_model runs/detect/helmet_model_enhanced/weights/best.pt \
      --face_model   runs/detect/face_model_enhanced/weights/best.pt \
      --image        path/to/test_image.jpg

Output
------
  - Prints detections and violations to terminal
  - Saves annotated image to output/test_result.jpg
  - Opens the result in a window (press any key to close)
"""

import argparse
import sys
import os
import cv2

# Make sure training/ is on path for arch imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from arch.custom_modules import register_custom_modules
register_custom_modules()

from detection.model_inference    import Models
from detection.motorcycle_pipeline import process_motorcycle
from utils.draw_utils              import draw_box, draw_frame_counter, COLORS
from utils.logger                  import ViolationLogger


def parse_args():
    p = argparse.ArgumentParser(description="Test violation detection on a single image")
    p.add_argument("--helmet_model", required=True,
                   help="Path to trained helmet/plate model (.pt)")
    p.add_argument("--face_model",   required=True,
                   help="Path to trained face model (.pt)")
    p.add_argument("--image",        required=True,
                   help="Path to input image (.jpg / .png)")
    p.add_argument("--output_dir",   default="output",
                   help="Where to save the annotated result")
    p.add_argument("--conf",         type=float, default=0.30,
                   help="Motorcycle detection confidence (default: 0.30)")
    p.add_argument("--conf_helmet",  type=float, default=0.15,
                   help="Helmet confidence inside crop (default: 0.15)")
    p.add_argument("--conf_face",    type=float, default=0.20,
                   help="Face confidence inside crop (default: 0.20)")
    p.add_argument("--conf_plate",   type=float, default=0.15,
                   help="Plate confidence inside crop (default: 0.15)")
    p.add_argument("--show",         action="store_true",
                   help="Display result in a window")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load image ────────────────────────────────────────────────────────
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError(f"Could not read image: {args.image}")

    print(f"\n{'='*55}")
    print(" SINGLE IMAGE TEST")
    print(f"{'='*55}")
    print(f"  Image      : {args.image}")
    print(f"  Resolution : {frame.shape[1]}×{frame.shape[0]}")
    print(f"  Confidence : {args.conf}")
    print(f"{'='*55}\n")

    # ── Load models ───────────────────────────────────────────────────────
    models = Models(
        helmet_model_path = args.helmet_model,
        face_model_path   = args.face_model,
        conf              = args.conf,
        conf_helmet       = args.conf_helmet,
        conf_face         = args.conf_face,
        conf_plate        = args.conf_plate,
    )
    logger = ViolationLogger(output_dir=args.output_dir)

    # ── Step 1: detect motorcycles ────────────────────────────────────────
    moto_dets = models.detect_motorcycles(frame)

    print(f"  Motorcycles found: {len(moto_dets)}\n")

    if not moto_dets:
        print("  No motorcycles detected. Try lowering --conf.\n")
        cv2.imwrite(os.path.join(args.output_dir, "test_result.jpg"), frame)
        if args.show:
            cv2.imshow("Result", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    annotated = frame.copy()
    total_violations = []

    # ── Run pipeline for each motorcycle ─────────────────────────────────
    for idx, moto_det in enumerate(moto_dets):
        print(f"  ── Motorcycle {idx} "
              f"(conf={moto_det.conf:.2f}  box={moto_det.box}) ──")

        output = process_motorcycle(
            frame     = annotated,
            moto_box  = moto_det.box,
            models    = models,
            frame_id  = 0,
            moto_idx  = idx,
        )

        annotated = output["annotated_frame"]
        r = output["result"]

        print(f"    Helmets detected   : {r.helmet_count}")
        print(f"    Faces detected     : {r.face_count}")
        print(f"    Unhelmeted faces   : {r.unhelmeted_count}")
        print(f"    Total riders est.  : {r.helmet_count + r.face_count}")

        if output["violations"]:
            print(f"    ⚠  VIOLATIONS      : {' | '.join(output['violations'])}")
            print(f"    Plate text         : '{output['plate_text'] or 'not found'}'")
            total_violations.extend(output["violations"])

            logger.save(
                frame_id     = 0,
                violations   = output["violations"],
                helmet_count = r.helmet_count,
                face_count   = r.face_count,
                moto_crop    = output["moto_crop"],
                plate_crop   = output["plate_crop"],
                plate_text   = output["plate_text"],
                moto_idx     = idx,
            )
        else:
            print(f"    ✓  No violations")
        print()

    # ── Save result ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "test_result.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"{'='*55}")
    print(f"  Total violations : {len(total_violations)}")
    print(f"  Result saved to  : {out_path}")
    print(f"{'='*55}\n")

    if args.show:
        cv2.imshow("Violation Detection - Test Result (press any key to close)",
                   annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()