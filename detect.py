"""
detect.py
=========
Entry point for the Two-Wheeler Traffic Violation Detector.

This script is intentionally thin. It only:
  1. Parses CLI arguments
  2. Loads models
  3. Drives the frame loop
  4. Delegates all real work to detection/ and utils/ modules

Usage
-----
  python detect.py \
      --helmet_model runs/detect/helmet_model/weights/best.pt \
      --face_model   runs/detect/face_model/weights/best.pt   \
      --source       videos/sample.mp4

  python detect.py --helmet_model <path> --face_model <path> --source 0 --show
"""

import argparse
import time

import cv2

from detection.model_inference   import Models
from detection.motorcycle_pipeline import process_motorcycle
from utils.draw_utils            import draw_box, draw_frame_counter, COLORS
from utils.logger                import ViolationLogger


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-Wheeler Traffic Rule Violation Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--helmet_model", required=True,
                   help="Path to trained helmet/plate YOLOv8 weights (.pt)")
    p.add_argument("--face_model",   required=True,
                   help="Path to trained face YOLOv8 weights (.pt)")
    p.add_argument("--source",       required=True,
                   help="Video file path or webcam index (e.g. 0)")
    p.add_argument("--output_dir",   default="output",
                   help="Directory for results")
    p.add_argument("--conf",         type=float, default=0.30,
                   help="Motorcycle detection confidence (default: 0.30)")
    p.add_argument("--conf_helmet",  type=float, default=0.15,
                   help="Helmet confidence inside crop (default: 0.15)")
    p.add_argument("--conf_face",    type=float, default=0.20,
                   help="Face confidence inside crop (default: 0.20)")
    p.add_argument("--conf_plate",   type=float, default=0.15,
                   help="Plate confidence inside crop (default: 0.15)")
    p.add_argument("--frame_skip",   type=int,   default=2,
                   help="Process every Nth frame  (1 = every frame)")
    p.add_argument("--show",         action="store_true",
                   help="Display live preview window")
    p.add_argument("--no_save_video",action="store_true",
                   help="Skip writing annotated output video")
    p.add_argument("--ocr_gpu",      action="store_true",
                   help="Use GPU for EasyOCR")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # Load both models
    models = Models(
        helmet_model_path = args.helmet_model,
        face_model_path   = args.face_model,
        conf              = args.conf,
        conf_helmet       = args.conf_helmet,
        conf_face         = args.conf_face,
        conf_plate        = args.conf_plate,
    )

    # Set up logger / output directories
    logger = ViolationLogger(output_dir=args.output_dir)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optional video writer
    writer = None
    if not args.no_save_video:
        import os
        out_path = os.path.join(args.output_dir, "annotated_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps  = fps / max(1, args.frame_skip)
        writer   = cv2.VideoWriter(out_path, fourcc, out_fps, (fw, fh))

    print(f"\n{'='*60}")
    print(" STARTING DETECTION")
    print(f"  Source     : {source}")
    print(f"  Frames     : {total}  |  FPS: {fps:.1f}")
    print(f"  Frame skip : {args.frame_skip}")
    print(f"  Output     : {args.output_dir}")
    print(f"{'='*60}\n")

    frame_id  = 0
    processed = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % args.frame_skip != 0:
            continue
        processed += 1

        # ── Step 1: Detect all motorcycles in the frame ───────────────────
        moto_dets = models.detect_motorcycles(frame)

        # Start with clean copy for this frame
        annotated = frame.copy()

        for moto_idx, moto_det in enumerate(moto_dets):

            # ── Steps 2–7: Full pipeline for this motorcycle ──────────────
            output = process_motorcycle(
                frame      = annotated,   # pass current annotated so boxes stack
                moto_box   = moto_det.box,
                models     = models,
                frame_id   = frame_id,
                moto_idx   = moto_idx,
                ocr_gpu    = args.ocr_gpu,
            )

            annotated = output["annotated_frame"]

            # ── Step 8: Save violation data ───────────────────────────────
            if output["has_violation"]:
                logger.save(
                    frame_id     = frame_id,
                    violations   = output["violations"],
                    helmet_count = output["result"].helmet_count,
                    face_count   = output["result"].face_count,
                    moto_crop    = output["moto_crop"],
                    plate_crop   = output["plate_crop"],
                    plate_text   = output["plate_text"],
                    moto_idx     = moto_idx,
                )

        draw_frame_counter(annotated, frame_id)

        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow("Traffic Violation Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f" DONE")
    print(f"  Frames processed : {processed}")
    print(f"  Time elapsed     : {elapsed:.1f}s  ({processed/elapsed:.1f} fps)")
    print(f"  Results saved to : {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()