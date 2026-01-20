from eeg_loader import load_eeg
from eeg_features import extract_eeg_features
from eeg_risk import compute_eeg_risk
from eeg_flag import eeg_clinical_flag

def run_eeg_screening(csv_path):
    df = load_eeg(csv_path)
    features = extract_eeg_features(df)
    risk = compute_eeg_risk(features)
    flag = eeg_clinical_flag(risk)

    return {
        "raw_measurements": {
            "frontal_variance": round(features["frontal_variance"], 3),
            "temporal_variance": round(features["temporal_variance"], 3),
            "hemispheric_asymmetry": round(features["hemispheric_asymmetry"], 3)
        },
        "risk_score": risk,
        "clinical_flag": flag
    }
