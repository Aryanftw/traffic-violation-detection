import sys
import cv2
from ultralytics import YOLO

# Add custom modules path
sys.path.insert(0, "training")

# Register custom modules
from arch.custom_modules import register_custom_modules
register_custom_modules()

# Load models
helmet_model = YOLO("training/runs/detect/helmet_model_enhanced-2/weights/best.pt")
face_model = YOLO("training/runs/detect/face_model_enhanced/weights/best.pt")

# Read image
img = cv2.imread("image3.jpg")

# ---------------- Helmet Model ----------------
print("--- Helmet model (conf=0.01) ---")
helmet_results = helmet_model(img, conf=0.01, verbose=False)[0]

for box in helmet_results.boxes:
    class_name = helmet_results.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    bbox = list(map(int, box.xyxy[0].tolist()))

    print(f"{class_name:<20s}  conf={confidence:.3f}  box={bbox}")

print()

# ---------------- Face Model ----------------
print("--- Face model (conf=0.01) ---")
face_results = face_model(img, conf=0.01, verbose=False)[0]

for box in face_results.boxes:
    class_name = face_results.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    bbox = list(map(int, box.xyxy[0].tolist()))

    print(f"{class_name:<20s}  conf={confidence:.3f}  box={bbox}")