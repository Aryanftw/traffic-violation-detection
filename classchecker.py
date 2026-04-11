# checkclasses.py
from ultralytics import YOLO

helmet = YOLO("models/helmet_best.pt")
face   = YOLO("models/face_best.pt")
lane   = YOLO("models/lane_best.pt")

print("HELMET MODEL classes:", helmet.names)
print("FACE MODEL classes:  ", face.names)
print("LANE MODEL classes:  ", lane.names)