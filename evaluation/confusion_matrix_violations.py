"""
evaluation/confusion_matrix_violations.py
==========================================
Generates a violation-level confusion matrix.

You provide a folder of test images + a CSV of ground-truth labels.
The script runs the full pipeline on each image and compares the
predicted violations against the ground truth.

Produces confusion matrices for:
  - No Helmet        (binary: violation present / not present)
  - Triple Riding    (binary: violation present / not present)
  - Any Violation    (binary: at least one violation present / not present)

Output
------
  evaluation/outputs/cm_no_helmet.png
  evaluation/outputs/cm_triple_riding.png
  evaluation/outputs/cm_any_violation.png
  evaluation/outputs/violation_results.csv
  evaluation/outputs/violation_summary.txt

Ground Truth CSV Format
-----------------------
Create a file called ground_truth.csv with these columns:

  image,no_helmet,triple_riding
  image1.jpg,1,0
  image2.jpg,0,0
  image3.jpg,1,1

  no_helmet    : 1 if any rider in the image has no helmet, else 0
  triple_riding: 1 if more than 2 riders on any motorcycle, else 0

Usage
-----
  cd E:\\finalyearproject
  python evaluation/confusion_matrix_violations.py ^
      --helmet_model training/runs/detect/helmet_model_enhanced-2/weights/best.pt ^
      --face_model   training/runs/detect/face_model_enhanced/weights/best.pt ^
      --images_dir   evaluation/test_images ^
      --ground_truth evaluation/ground_truth.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from arch.custom_modules import register_custom_modules
register_custom_modules()

from detection.model_inference    import Models
from detection.motorcycle_pipeline import process_motorcycle

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline on one image → return predicted violation flags
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image_path: str, models: Models) -> dict:
    """
    Run the full violation pipeline on one image.
    Returns dict: {no_helmet: bool, triple_riding: bool}
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [WARN] Could not read: {image_path}")
        return {"no_helmet": False, "triple_riding": False}

    moto_dets = models.detect_motorcycles(frame)

    pred_no_helmet    = False
    pred_triple_riding = False

    for idx, moto_det in enumerate(moto_dets):
        output = process_motorcycle(
            frame    = frame,
            moto_box = moto_det.box,
            models   = models,
            frame_id = 0,
            moto_idx = idx,
        )
        if "No Helmet"     in output["violations"]:
            pred_no_helmet = True
        if "Triple Riding" in output["violations"]:
            pred_triple_riding = True

    return {
        "no_helmet":     pred_no_helmet,
        "triple_riding": pred_triple_riding,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot and save a styled confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true: list,
    y_pred: list,
    label:  str,
    filename: str,
):
    """
    Plot a binary confusion matrix and save it as PNG.
    Also returns metrics dict.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {label}", fontsize=14, fontweight="bold")

    # Left: raw counts
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Violation", "Violation"]
    )
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Raw Counts")

    # Right: normalised (recall-normalised = row-normalised)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid divide by zero
    cm_norm = cm_norm / row_sums

    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=["No Violation", "Violation"]
    )
    disp2.plot(ax=axes[1], colorbar=False, cmap="Blues",
               values_format=".2f")
    axes[1].set_title("Normalised (row)")

    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved : {out_path}")

    # Compute metrics
    n = len(y_true)
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / n if n > 0 else 0.0

    return {
        "label":     label,
        "total":     n,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Violation-level confusion matrix"
    )
    p.add_argument("--helmet_model", required=True)
    p.add_argument("--face_model",   required=True)
    p.add_argument("--images_dir",   required=True,
                   help="Folder containing test images")
    p.add_argument("--ground_truth", required=True,
                   help="CSV file with columns: image,no_helmet,triple_riding")
    p.add_argument("--conf",         type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()

    models = Models(
        helmet_model_path = args.helmet_model,
        face_model_path   = args.face_model,
        conf              = args.conf,
    )

    # ── Load ground truth ─────────────────────────────────────────────────
    images_dir = Path(args.images_dir)
    gt = {}
    with open(args.ground_truth, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["image"]] = {
                "no_helmet":     int(row["no_helmet"]),
                "triple_riding": int(row["triple_riding"]),
            }

    print(f"\n  Ground truth loaded: {len(gt)} images\n")

    # ── Run pipeline on all images ────────────────────────────────────────
    results = []
    for img_name, labels in gt.items():
        img_path = str(images_dir / img_name)
        if not os.path.exists(img_path):
            print(f"  [SKIP] Not found: {img_path}")
            continue

        print(f"  Processing: {img_name}")
        pred = predict_image(img_path, models)

        results.append({
            "image":               img_name,
            "gt_no_helmet":        labels["no_helmet"],
            "gt_triple_riding":    labels["triple_riding"],
            "pred_no_helmet":      int(pred["no_helmet"]),
            "pred_triple_riding":  int(pred["triple_riding"]),
            "gt_any":              int(labels["no_helmet"] or labels["triple_riding"]),
            "pred_any":            int(pred["no_helmet"]   or pred["triple_riding"]),
        })

    if not results:
        print("No results — check your images_dir and ground_truth CSV.")
        return

    # ── Save per-image results CSV ────────────────────────────────────────
    results_csv = OUT_DIR / "violation_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved per-image results: {results_csv}")

    # ── Build confusion matrices ──────────────────────────────────────────
    gt_nh  = [r["gt_no_helmet"]     for r in results]
    pr_nh  = [r["pred_no_helmet"]   for r in results]
    gt_tr  = [r["gt_triple_riding"] for r in results]
    pr_tr  = [r["pred_triple_riding"] for r in results]
    gt_any = [r["gt_any"]           for r in results]
    pr_any = [r["pred_any"]         for r in results]

    m_nh  = save_confusion_matrix(gt_nh,  pr_nh,  "No Helmet Violation",    "cm_no_helmet.png")
    m_tr  = save_confusion_matrix(gt_tr,  pr_tr,  "Triple Riding Violation", "cm_triple_riding.png")
    m_any = save_confusion_matrix(gt_any, pr_any, "Any Violation",           "cm_any_violation.png")

    # ── Print + save summary ──────────────────────────────────────────────
    summary_lines = [
        "VIOLATION DETECTION SYSTEM — EVALUATION SUMMARY",
        "=" * 55,
        f"Total test images : {len(results)}",
        "",
    ]

    for m in [m_nh, m_tr, m_any]:
        summary_lines += [
            f"── {m['label']} ──",
            f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}",
            f"  Precision : {m['precision']:.3f}",
            f"  Recall    : {m['recall']:.3f}",
            f"  F1        : {m['f1']:.3f}",
            f"  Accuracy  : {m['accuracy']:.3f}",
            "",
        ]

    summary_lines.append("=" * 55)
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    summary_path = OUT_DIR / "violation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\n  Saved summary: {summary_path}")
    print(f"\n✓ All outputs saved to: {OUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()