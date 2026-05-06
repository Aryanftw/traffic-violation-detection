"""
utils/logger.py
===============
Handles all file I/O for violation records:
  - CSV log  (one row per motorcycle per frame)
  - Saving annotated motorcycle crop images
  - Saving license plate crop images
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ViolationLogger:
    """
    Writes violation data to disk.

    Directory layout created under `output_dir`:
      output_dir/
      ├── violations_log.csv
      ├── violations/          ← annotated motorcycle crop images
      └── plates/              ← extracted license plate images
    """

    CSV_FIELDS = [
        "timestamp",
        "frame_id",
        "violations",
        "plate_text",
        "helmet_count",
        "face_count",
        "violation_image",
        "plate_image",
    ]

    def __init__(self, output_dir: str = "output"):
        self.out_dir        = Path(output_dir)
        self.violations_dir = self.out_dir / "violations"
        self.plates_dir     = self.out_dir / "plates"

        for d in (self.out_dir, self.violations_dir, self.plates_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.out_dir / "violations_log.csv"
        self._init_csv()

        # Shared timestamp prefix for this run (keeps filenames grouped)
        self.run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────────────────────

    def _init_csv(self):
        """Write header row if the CSV does not already exist."""
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def _append_csv(self, row: dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)

    # ── Public API ────────────────────────────────────────────────────────────

    def save(
        self,
        frame_id:      int,
        violations:    list[str],
        helmet_count:  int,
        face_count:    int,
        moto_crop:     np.ndarray,
        plate_crop:    np.ndarray | None,
        plate_text:    str,
        moto_idx:      int = 0,
    ):
        """
        Save images and write one log row for a single motorcycle violation.

        Args:
            frame_id:     frame number in the source video
            violations:   list of violation labels, e.g. ["No Helmet"]
            helmet_count: helmets detected on this motorcycle
            face_count:   faces detected on this motorcycle
            moto_crop:    annotated motorcycle region (BGR numpy array)
            plate_crop:   plate region crop, or None if no plate found
            plate_text:   OCR result string
            moto_idx:     index of this motorcycle within the frame (0-based)
        """
        slug = f"{self.run_ts}_f{frame_id:06d}_m{moto_idx}"

        # ── Save motorcycle violation image ───────────────────────────────
        moto_path = ""
        if moto_crop is not None and moto_crop.size > 0:
            moto_path = str(self.violations_dir / f"{slug}_violation.jpg")
            cv2.imwrite(moto_path, moto_crop)

        # ── Save plate image ──────────────────────────────────────────────
        plate_path = ""
        if plate_crop is not None and plate_crop.size > 0:
            plate_path = str(self.plates_dir / f"{slug}_plate.jpg")
            cv2.imwrite(plate_path, plate_crop)

        # ── CSV row ───────────────────────────────────────────────────────
        self._append_csv({
            "timestamp":       datetime.now().isoformat(timespec="seconds"),
            "frame_id":        frame_id,
            "violations":      " | ".join(violations),
            "plate_text":      plate_text,
            "helmet_count":    helmet_count,
            "face_count":      face_count,
            "violation_image": moto_path,
            "plate_image":     plate_path,
        })

        print(
            f"  [LOG] Frame {frame_id} | Moto {moto_idx} | "
            f"{' + '.join(violations)} | "
            f"Helmets={helmet_count} Faces={face_count} | "
            f"Plate='{plate_text}'"
        )