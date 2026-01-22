def behavior_clinical_flag(risk):
    if risk >= 0.7:
        return "Reduced visual engagement and unstable head orientation during interaction"
    elif risk >= 0.4:
        return "Mild reduction in sustained visual attention patterns"
    else:
        return "Visual engagement patterns within expected screening range"
