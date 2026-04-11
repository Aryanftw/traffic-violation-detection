import cv2, argparse
from detector import Detector
from violation_checker import check_violations
from ocr_handler import run_ocr
from utils import ensure_dirs, save_violation
import config

def process_video(video_path):
    ensure_dirs(config.OUTPUT_DIR, config.PLATE_DIR)
    detector = Detector()
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % config.FRAME_SKIP == 0:
            motorcycles = detector.get_motorcycles(frame)

            for moto in motorcycles:
                x1, y1, x2, y2 = moto['xyxy']
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                violations, _, _ = check_violations(crop, detector)

                if violations:
                    # Get plate crop
                    plates     = detector.get_plates(crop)
                    plate_crop = None
                    plate_text = None

                    if plates:
                        px1,py1,px2,py2 = plates[0]['xyxy']
                        plate_crop = crop[py1:py2, px1:px2]
                        plate_text = run_ocr(plate_crop)

                    save_violation(frame_id, crop, plate_crop, plate_text,
                                   violations, config.OUTPUT_DIR, config.PLATE_DIR)

        frame_id += 1
        print(f"\rFrame {frame_id}", end="")

    cap.release()
    print("\n[DONE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    process_video(args.video)