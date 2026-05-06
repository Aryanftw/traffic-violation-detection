"""
detection/model_inference.py
=============================
Wraps both YOLOv8 models and exposes clean, typed inference functions.

All YOLO-specific code lives here. The rest of the pipeline receives plain
Python lists of (class_name, confidence, box) tuples — no YOLO objects leak out.
"""

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Detection result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    cls:  str              # lowercase class name from the model
    conf: float            # confidence score  [0.0 – 1.0]
    box:  tuple            # (x1, y1, x2, y2) in pixels


# ─────────────────────────────────────────────────────────────────────────────
# Class-name sets
# ─────────────────────────────────────────────────────────────────────────────
# Edit these if your model uses different class names.
# Run `print(model.names)` after loading to see what your weights use.

MOTO_CLASSES   = {"motorcyclist", "motorcycle", "two-wheeler", "bike"}
HELMET_CLASSES = {"helmet"}
PLATE_CLASSES  = {"license_plate", "license plate", "licenseplate", "plate"}
FACE_CLASSES   = {"face"}


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Models:
    """
    Holds both YOLOv8 models and provides clean per-task inference methods.

    Usage
    -----
    models = Models("weights/helmet.pt", "weights/face.pt", conf=0.40)

    motos   = models.detect_motorcycles(frame)
    helmets = models.detect_helmets(crop)
    faces   = models.detect_faces(crop)
    plates  = models.detect_plates(crop)
    """

    def __init__(
        self,
        helmet_model_path: str,
        face_model_path:   str,
        conf: float = 0.30,
        # Per-task thresholds — override the global conf for specific tasks.
        # Helmets and faces inside crops need lower thresholds because:
        #   - helmet confidence is naturally lower on unhelmetted/masked riders
        #   - face confidence drops for masked, rear-facing, or small faces
        # Motorcycles on the full frame stay at the global conf (less noisy).
        conf_helmet: float = 0.08,
        conf_face:   float = 0.20,
        conf_plate:  float = 0.15,
    ):
        print(f"[Models] Loading helmet model  : {helmet_model_path}")
        self.helmet_model = YOLO(helmet_model_path)

        print(f"[Models] Loading face model    : {face_model_path}")
        self.face_model   = YOLO(face_model_path)

        self.conf        = conf
        self.conf_helmet = conf_helmet
        self.conf_face   = conf_face
        self.conf_plate  = conf_plate
        print(f"[Models] Both models ready.")
        print(f"  motorcycle conf : {conf}")
        print(f"  helmet conf     : {conf_helmet}")
        print(f"  face conf       : {conf_face}")
        print(f"  plate conf      : {conf_plate}\n")

    # ── Low-level inference ───────────────────────────────────────────────────

    def _run(self, model: YOLO, img: np.ndarray,
             conf: float = None) -> list[Detection]:
        """Run a YOLO model and return plain Detection objects."""
        c = conf if conf is not None else self.conf
        results = model(img, conf=c, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls    = results.names[int(box.cls[0])].lower()
            conf_v = float(box.conf[0])
            coords = tuple(map(int, box.xyxy[0].tolist()))
            detections.append(Detection(cls=cls, conf=conf_v, box=coords))
        return detections

    @staticmethod
    def _nms(detections: list, iou_thresh: float = 0.45) -> list:
        """
        Non-maximum suppression — removes overlapping duplicate detections.
        Keeps highest-confidence box when two overlap more than iou_thresh.
        """
        if len(detections) <= 1:
            return detections
        dets = sorted(detections, key=lambda d: d.conf, reverse=True)
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            remaining = []
            for d in dets:
                ax1,ay1,ax2,ay2 = best.box
                bx1,by1,bx2,by2 = d.box
                ix1=max(ax1,bx1); iy1=max(ay1,by1)
                ix2=min(ax2,bx2); iy2=min(ay2,by2)
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
                if inter/union < iou_thresh if union > 0 else True:
                    remaining.append(d)
            dets = remaining
        return kept

    # ── Public inference helpers ──────────────────────────────────────────────

    def detect_motorcycles(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect motorcycles in the full frame.
        Applies NMS to remove duplicate/overlapping motorcycle boxes —
        the main cause of counting pedestrians inside a ghost motorcycle crop.
        """
        dets = self._run(self.helmet_model, frame, conf=self.conf)
        motos = [d for d in dets if d.cls in MOTO_CLASSES]
        return self._nms(motos, iou_thresh=0.45)

    def detect_helmets(self, crop: np.ndarray) -> list[Detection]:
        """Detect helmets in motorcycle crop at lower conf_helmet threshold."""
        dets = self._run(self.helmet_model, crop, conf=self.conf_helmet)
        return [d for d in dets if d.cls in HELMET_CLASSES]

    def detect_plates(self, crop: np.ndarray) -> list[Detection]:
        """Detect plates in motorcycle crop at lower conf_plate threshold."""
        dets = self._run(self.helmet_model, crop, conf=self.conf_plate)
        return [d for d in dets if d.cls in PLATE_CLASSES]

    def detect_faces(self, crop: np.ndarray) -> list[Detection]:
        """Detect faces in motorcycle crop at lower conf_face threshold."""
        dets = self._run(self.face_model, crop, conf=self.conf_face)
        return [d for d in dets if d.cls in FACE_CLASSES]