def compute_behavior_risk(face_ratio, pose_ratio):
    risk = 0.0

    if face_ratio < 0.4:
        risk += 0.4
    elif face_ratio < 0.6:
        risk += 0.2

    if pose_ratio < 0.4:
        risk += 0.4
    elif pose_ratio < 0.6:
        risk += 0.2

    return float(round(min(risk, 1.0), 2))
