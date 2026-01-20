def gesture_clinical_flag(risk):
    if risk >= 0.7:
        return "Reduced diversity and repetitive motor movement patterns observed during interaction"
    elif risk >= 0.4:
        return "Mild reduction in spontaneous gestural variability"
    else:
        return "Motor gesture patterns within expected screening range"
