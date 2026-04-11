from ultralytics import YOLO
import config

class Detector:
    def __init__(self):
        self.helmet_model = YOLO(config.HELMET_MODEL)
        self.face_model   = YOLO(config.FACE_MODEL)
        self.lane_model   = YOLO(config.LANE_MODEL)

    def detect(self, model, image, conf=None):
        conf = conf or config.CONF_THRESHOLD
        results = model(image, conf=conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append({
                'cls':  int(box.cls[0]),
                'conf': float(box.conf[0]),
                'xyxy': list(map(int, box.xyxy[0].tolist()))
            })
        return detections

    def get_motorcycles(self, frame):
        dets = self.detect(self.helmet_model, frame)
        return [d for d in dets if d['cls'] == config.HELMET_CLASS['motorcyclist']]  # fixed

    def get_helmets(self, crop):
        dets = self.detect(self.helmet_model, crop)
        return [d for d in dets if d['cls'] == config.HELMET_CLASS['helmet']]

    def get_plates(self, crop):
        dets = self.detect(self.helmet_model, crop)
        return [d for d in dets if d['cls'] == config.HELMET_CLASS['license_plate']]

    def get_faces(self, crop):
        dets = self.detect(self.face_model, crop)
        return [d for d in dets if d['cls'] == config.FACE_CLASS['face']]

    def get_orientation(self, crop):
        dets = self.detect(self.lane_model, crop)
        for d in dets:
            if d['cls'] == config.LANE_CLASS['rear']:
                return 'rear'
        return 'front'