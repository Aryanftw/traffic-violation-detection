import cv2, os
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0: return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea)

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_violation(frame_id, moto_crop, plate_crop, plate_text, violations, out_dir, plate_dir):
    base = f"frame_{frame_id:05d}"
    cv2.imwrite(f"{out_dir}/{base}_moto.jpg", moto_crop)
    if plate_crop is not None:
        cv2.imwrite(f"{plate_dir}/{base}_plate.jpg", plate_crop)
    with open(f"{out_dir}/{base}_info.txt", "w") as f:
        f.write(f"Violations: {', '.join(violations)}\n")
        f.write(f"Plate Text: {plate_text or 'NOT DETECTED'}\n")
    print(f"[SAVED] Frame {frame_id} | {violations} | Plate: {plate_text}")