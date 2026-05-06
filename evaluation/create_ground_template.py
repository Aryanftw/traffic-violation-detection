"""
evaluation/create_ground_truth_template.py
==========================================
Scans a folder of test images and generates a blank ground_truth.csv
for you to fill in manually.

Usage
-----
  cd E:\\finalyearproject
  python evaluation/create_ground_truth_template.py --images_dir evaluation/test_images

Then open ground_truth.csv in Excel or Notepad and fill in:
  no_helmet    : 1 if any rider has no helmet, else 0
  triple_riding: 1 if more than 2 riders on any motorcycle, else 0
"""

import argparse
import csv
import os
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True,
                   help="Folder of test images")
    p.add_argument("--output",     default="evaluation/ground_truth.csv",
                   help="Where to save the template CSV")
    return p.parse_args()


def main():
    args       = parse_args()
    images_dir = Path(args.images_dir)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted([
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        print(f"No images found in: {images_dir}")
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "no_helmet", "triple_riding"])
        for img in images:
            writer.writerow([img, "", ""])   # blank — fill in manually

    print(f"Template created: {out_path.resolve()}")
    print(f"Found {len(images)} images — fill in the 0/1 values then run:")
    print(f"  python evaluation/confusion_matrix_violations.py \\")
    print(f"      --helmet_model <path> --face_model <path> \\")
    print(f"      --images_dir {images_dir} \\")
    print(f"      --ground_truth {out_path}")


if __name__ == "__main__":
    main()