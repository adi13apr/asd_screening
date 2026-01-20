from gesture_metrics import compute_gesture_metrics
from gesture_risk import compute_gesture_risk
from gesture_flag import gesture_clinical_flag

def run_gesture_screening(duration_sec=10):
    metrics = compute_gesture_metrics(duration_sec)

    if metrics is None:
        return {
            "raw_measurements": {},
            "risk_score": 0.0,
            "clinical_flag": "Gesture data could not be captured reliably"
        }

    risk = compute_gesture_risk(metrics)
    flag = gesture_clinical_flag(risk)

    return {
        "raw_measurements": metrics,
        "risk_score": risk,
        "clinical_flag": flag
    }
