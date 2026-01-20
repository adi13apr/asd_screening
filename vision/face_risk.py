def compute_face_risk(cnn_probability, face_confidence):
    adjusted_risk = cnn_probability * face_confidence
    return round(adjusted_risk, 2)
