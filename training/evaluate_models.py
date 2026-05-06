"""
evaluate_models.py
==================
Validates both trained enhanced models and prints a summary table.

Usage
-----
  python training/evaluate_models.py \
      --helmet_model runs/detect/helmet_model_enhanced/weights/best.pt \
      --helmet_data  data/helmet_dataset/data.yaml \
      --face_model   runs/detect/face_model_enhanced/weights/best.pt \
      --face_data    data/face_dataset/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Must register before loading any enhanced .pt file
from arch.custom_modules import register_custom_modules


def evaluate(model_path: str, data_yaml: str, label: str) -> dict:
    print(f"\n── {label} ──────────────────────────────────")
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=False)

    mp   = metrics.box.mp
    mr   = metrics.box.mr
    map5 = metrics.box.map50
    mapv = metrics.box.map

    print(f"  Precision  : {mp:.3f}")
    print(f"  Recall     : {mr:.3f}")
    print(f"  mAP@0.5    : {map5:.3f}")
    print(f"  mAP@0.5:95 : {mapv:.3f}")
    print(f"\n  Per-class mAP@0.5:")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"    [{i}] {model.names[i]:<20s}  {ap:.3f}")

    return {"precision": mp, "recall": mr, "mAP50": map5, "mAP": mapv}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate both enhanced models")
    p.add_argument("--helmet_model", required=True)
    p.add_argument("--helmet_data",  required=True)
    p.add_argument("--face_model",   required=True)
    p.add_argument("--face_data",    required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_custom_modules()

    print("\n" + "="*55)
    print(" MODEL EVALUATION SUMMARY")
    print("="*55)

    h = evaluate(args.helmet_model, args.helmet_data, "Helmet / Plate Model (Enhanced)")
    f = evaluate(args.face_model,   args.face_data,   "Face Detection Model (Enhanced)")

    print("\n" + "="*55)
    print(f"  {'Model':<28} {'mAP@0.5':>8}  {'mAP@0.5:95':>10}")
    print(f"  {'-'*28} {'-'*8}  {'-'*10}")
    print(f"  {'Helmet/Plate (Enhanced)':<28} {h['mAP50']:>8.3f}  {h['mAP']:>10.3f}")
    print(f"  {'Face (Enhanced)':<28} {f['mAP50']:>8.3f}  {f['mAP']:>10.3f}")
    print("="*55 + "\n")