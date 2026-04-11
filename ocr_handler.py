import easyocr
import cv2
import numpy as np

# Initialize once — loading model is slow, so do it at module level
reader = easyocr.Reader(['en'], gpu=True)  # set gpu=False if you hit VRAM issues

def run_ocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return None
    try:
        # EasyOCR works best with a slightly upscaled plate
        h, w = plate_crop.shape[:2]
        target_w = 300
        if w < target_w:
            scale = target_w / w
            plate_crop = cv2.resize(plate_crop, 
                          (int(w*scale), int(h*scale)), 
                          interpolation=cv2.INTER_CUBIC)

        # Preprocessing — convert to grayscale, increase contrast
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = reader.readtext(plate_crop)
        print(f"  [OCR RAW] {results}")  # debug line


        if not results:
            return None

        # Pick the result with highest confidence
        best = max(results, key=lambda x: x[2])
        text = best[1].strip().upper()
        conf = best[2]

        if conf < 0.1:  # ignore low confidence reads
            return None

        return text

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None