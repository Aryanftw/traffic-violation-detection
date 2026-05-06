"""
evaluation/confusion_matrix_models.py
=======================================
Generates YOLO's built-in per-class confusion matrices for both models.

This gives you the standard object detection confusion matrix:
  rows = ground truth class
  cols = predicted class
  + a "background" row/col for missed detections and false positives

Output
------
  evaluation/outputs/helmet_model_confusion_matrix.png
  evaluation/outputs/face_model_confusion_matrix.png
  evaluation/outputs/helmet_model_metrics.csv
  evaluation/outputs/face_model_metrics.csv

Usage
-----
  cd E:\\finalyearproject
  python evaluation/confusion_matrix_models.py ^
      --helmet_model training/runs/detect/helmet_model_enhanced-2/weights/best.pt ^
      --helmet_data  datasets/helmet_detection/data.yaml ^
      --face_model   training/runs/detect/face_model_enhanced/weights/best.pt ^
      --face_data    datasets/face_detection/data.yaml
"""

import argparse
import os
import sys
import shutil
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from arch.custom_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_validation(model_path: str, data_yaml: str, label: str):
    """
    Run YOLO validation and save confusion matrix + metrics.
    """
    print(f"\n{'='*55}")
    print(f" {label}")
    print(f"{'='*55}")
    print(f"  Model : {model_path}")
    print(f"  Data  : {data_yaml}\n")

    model   = YOLO(model_path)
    metrics = model.val(
        data    = data_yaml,
        conf    = 0.25,
        iou     = 0.45,
        plots   = True,       # ← this generates the confusion matrix PNG
        verbose = True,
    )

    # ── Copy confusion matrix plot to our output folder ───────────────────
    # YOLO saves it to runs/detect/val*/confusion_matrix.png
    val_dir = Path(metrics.save_dir)
    slug    = label.lower().replace(" ", "_").replace("/", "_")

    for cm_file in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = val_dir / cm_file
        if src.exists():
            dst = OUT_DIR / f"{slug}_{cm_file}"
            shutil.copy(src, dst)
            print(f"  Saved : {dst}")

    # ── Save per-class metrics to CSV ─────────────────────────────────────
    csv_path = OUT_DIR / f"{slug}_metrics.csv"
    names    = model.names

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "mAP50", "mAP50-95"])

        ap_per_class   = metrics.box.ap
        ap50_per_class = metrics.box.ap50
        p_per_class    = metrics.box.p
        r_per_class    = metrics.box.r

        for i in range(len(names)):
            writer.writerow([
                names[i],
                f"{p_per_class[i]:.4f}",
                f"{r_per_class[i]:.4f}",
                f"{ap50_per_class[i]:.4f}",
                f"{ap_per_class[i]:.4f}",
            ])

        # Overall row
        writer.writerow([
            "OVERALL",
            f"{metrics.box.mp:.4f}",
            f"{metrics.box.mr:.4f}",
            f"{metrics.box.map50:.4f}",
            f"{metrics.box.map:.4f}",
        ])

    print(f"  Saved : {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n  {'Class':<20s} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for i in range(len(names)):
        print(f"  {names[i]:<20s} "
              f"{p_per_class[i]:>8.3f} "
              f"{r_per_class[i]:>8.3f} "
              f"{ap50_per_class[i]:>8.3f} "
              f"{ap_per_class[i]:>10.3f}")
    print(f"  {'OVERALL':<20s} "
          f"{metrics.box.mp:>8.3f} "
          f"{metrics.box.mr:>8.3f} "
          f"{metrics.box.map50:>8.3f} "
          f"{metrics.box.map:>10.3f}")

    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate YOLO confusion matrices for both models"
    )
    p.add_argument("--helmet_model", required=True)
    p.add_argument("--helmet_data",  required=True)
    p.add_argument("--face_model",   required=True)
    p.add_argument("--face_data",    required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(args.helmet_model, args.helmet_data, "Helmet Plate Model")
    run_validation(args.face_model,   args.face_data,   "Face Model")
    print(f"\n✓ All outputs saved to: {OUT_DIR.resolve()}\n")