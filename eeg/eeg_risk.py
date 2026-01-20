def normalize(val, max_val):
    return min(abs(val) / max_val, 1.0)

def compute_eeg_risk(features):
    frontal_risk   = normalize(features["frontal_variance"], 50)
    temporal_risk  = normalize(features["temporal_variance"], 50)
    asymmetry_risk = normalize(features["hemispheric_asymmetry"], 10)

    eeg_risk = (
        0.4 * frontal_risk +
        0.4 * temporal_risk +
        0.2 * asymmetry_risk
    )

    return round(eeg_risk, 2)
