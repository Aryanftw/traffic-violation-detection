# visualdebug.py
from ultralytics import YOLO
import cv2

helmet_model = YOLO("models/helmet_best.pt")
face_model   = YOLO("models/face_best.pt")
lane_model   = YOLO("models/lane_best.pt")

image_path = "image.png"
frame = cv2.imread(image_path)

# Run all 3 models
for model, name in [(helmet_model, "HELMET"), (face_model, "FACE"), (lane_model, "LANE")]:
    results = model(frame, conf=0.4)[0]
    print(f"\n{name} MODEL detections:")
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  class {cls} ({model.names[cls]}) — conf: {conf:.2f}")
    
    # Draw and save
    annotated = results.plot()
    cv2.imwrite(f"output_{name.lower()}.jpg", annotated)
    print(f"  Saved → output_{name.lower()}.jpg")