from utils import iou
import config

def check_violations(crop, detector):
    violations = []

    # 1. Orientation — wrong lane
    orientation = detector.get_orientation(crop)
    if orientation == 'rear':
        violations.append("Wrong Lane")

    # 2. Faces and helmets
    faces   = detector.get_faces(crop)
    helmets = detector.get_helmets(crop)

    face_boxes   = [f['xyxy'] for f in faces]
    helmet_boxes = [h['xyxy'] for h in helmets]

    # Reduce face count where a helmet overlaps it by > threshold
    effective_faces = 0
    for fb in face_boxes:
        covered = any(
            iou(fb, hb) > config.OVERLAP_THRESHOLD
            for hb in helmet_boxes
        )
        if not covered:
            effective_faces += 1

    # 3. No helmet violation
    if len(helmets) == 0 or effective_faces > 1:
        violations.append("No Helmet")

    # 4. Triple riding
    total_riders = len(helmets) + effective_faces
    if total_riders > 2:
        violations.append("Triple Riding")

    return violations, effective_faces, len(helmets)