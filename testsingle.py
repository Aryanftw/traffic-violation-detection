# test_single.py
from detector import Detector
from violation_checker import check_violations
from ocr_handler import run_ocr
from utils import ensure_dirs, save_violation
import config, cv2

ensure_dirs(config.OUTPUT_DIR, config.PLATE_DIR)
detector = Detector()

image_path = "image.png"
frame = cv2.imread(image_path)
H, W = frame.shape[:2]

motorcycles = detector.get_motorcycles(frame)
print(f"Found {len(motorcycles)} motorcycles")

for i, moto in enumerate(motorcycles):
    x1, y1, x2, y2 = moto['xyxy']

    # Add padding so helmet at top of rider isn't clipped
    PAD = 40
    x1 = max(0, x1 - PAD)
    y1 = max(0, y1 - PAD)
    x2 = min(W, x2 + PAD)
    y2 = min(H, y2 + PAD)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    violations, faces, helmets = check_violations(crop, detector)
    print(f"Moto {i+1} → Violations: {violations} | Faces: {faces} | Helmets: {helmets}")

    if violations:
        plates     = detector.get_plates(crop)
        plate_crop = None
        plate_text = None

        if plates:
            px1, py1, px2, py2 = plates[0]['xyxy']
            plate_crop = crop[py1:py2, px1:px2]
             # Save raw plate crop to inspect it
            cv2.imwrite(f"output/plates/debug_plate_{i}.jpg", plate_crop)
            print(f"  Plate crop size: {plate_crop.shape}")
            plate_text = run_ocr(plate_crop)
            print(f"  Plate text: {plate_text}")

        save_violation(i, crop, plate_crop, plate_text,
                       violations, config.OUTPUT_DIR, config.PLATE_DIR)