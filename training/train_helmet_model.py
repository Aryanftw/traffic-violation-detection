"""
train_helmet_model.py
=====================
Trains the enhanced YOLOv8 helmet/plate/motorcyclist model.

Architecture (enhanced_helmet.yaml):
  Backbone : C2f_FasterNet blocks  + ADown downsampling
  Neck     : PANet/FPN with C2f_FasterNet + ADown
  Head     : LSCD_Detect (shared weights + GroupNorm)
"""

import argparse
import multiprocessing
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Enhanced YOLOv8 – Helmet / Plate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",     required=True,
                   help="Path to data.yaml for the helmet dataset")
    p.add_argument("--arch",     default=None,
                   help="Path to the model YAML  (default: enhanced_helmet.yaml)")
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--batch",    type=int,   default=8,
                   help="Batch size — use 8 for 6 GB VRAM, 4 if still OOM")
    p.add_argument("--name",     default="helmet_model_enhanced")
    p.add_argument("--device",   default="0")
    p.add_argument("--patience", type=int,   default=10)
    p.add_argument("--workers",  type=int,   default=4,
                   help="Dataloader workers — keep at 4 on Windows to avoid "
                        "spawn overhead; set 0 to disable multiprocessing")
    return p.parse_args()


def main():
    # ── Must be called BEFORE importing YOLO ─────────────────────────────
    from arch.custom_modules import register_custom_modules
    register_custom_modules()

    from ultralytics import YOLO

    args = parse_args()

    ARCH_YAML = Path(__file__).parent / "arch" / "enhanced_helmet.yaml"
    arch_path = Path(args.arch) if args.arch else ARCH_YAML
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture YAML not found: {arch_path}")

    print(f"\n{'='*60}")
    print(" HELMET / PLATE MODEL  —  ENHANCED ARCHITECTURE")
    print(f"{'='*60}")
    print(f"  YAML       : {arch_path}")
    print(f"  Dataset    : {data_path}")
    print(f"  Epochs     : {args.epochs}  |  Batch : {args.batch}")
    print(f"  Image size : {args.imgsz}   |  Device: GPU {args.device}")
    print(f"  Run name   : {args.name}")
    print(f"{'='*60}\n")

    model = YOLO(str(arch_path))

    model.train(
        data          = str(data_path),
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        name          = args.name,
        device        = args.device,
        patience      = args.patience,
        workers       = args.workers,
        # Augmentation
        mosaic        = 1.0,
        flipud        = 0.0,
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 5.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 2.0,
        # Optimiser
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        warmup_epochs = 3,
        cos_lr        = True,
        # Loss weights
        box           = 7.5,
        cls           = 0.5,
        dfl           = 1.5,
    )

    best = Path(f"runs/detect/{args.name}/weights/best.pt")
    print(f"\n✓ Training complete.")
    print(f"  Best weights : {best.resolve()}")
    print(f"  → Use this path as --helmet_model in detect.py\n")


# ── Windows multiprocessing guard ────────────────────────────────────────────
# On Windows, dataloader workers are spawned (not forked). Each worker
# re-executes this file from the top-level. The `if __name__ == "__main__"`
# guard prevents the workers from re-running main() — without it every worker
# tries to start training again, exhausting memory and triggering the
# "paging file too small" OSError.
if __name__ == "__main__":
    # freeze_support() is required on Windows when using multiprocessing
    # inside a frozen executable; harmless in normal Python.
    multiprocessing.freeze_support()
    main()