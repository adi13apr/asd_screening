def face_clinical_flag(risk):
    if risk >= 0.7:
        return "Facial attention and expression patterns consistent with ASD-trained visual model"
    elif risk >= 0.4:
        return "Mild deviations in facial attention patterns"
    else:
        return "Facial attention patterns within expected screening range"
