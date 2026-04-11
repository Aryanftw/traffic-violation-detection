# Paths
HELMET_MODEL = "models/helmet_best.pt"
FACE_MODEL   = "models/face_best.pt"
LANE_MODEL   = "models/lane_best.pt"

OUTPUT_DIR   = "output/violations"
PLATE_DIR    = "output/plates"

# Correct class indices from your trained models
HELMET_CLASS = {'helmet': 0, 'license_plate': 1, 'motorcyclist': 2}
FACE_CLASS   = {'face': 0}
LANE_CLASS   = {'front': 0, 'rear': 1}

# Thresholds
CONF_THRESHOLD    = 0.4
OVERLAP_THRESHOLD = 0.6
FRAME_SKIP        = 5

# EasyOCR
OCR_GPU = True