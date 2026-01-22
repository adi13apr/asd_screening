def eeg_clinical_flag(risk):
    if risk >= 0.7:
        return "Atypical frontal and temporal neural signal patterns observed during EEG screening"
    elif risk >= 0.4:
        return "Mild deviations in regional EEG neural activity patterns"
    else:
        return "EEG neural activity patterns within expected screening range"
