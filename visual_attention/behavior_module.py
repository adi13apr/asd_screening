from attention_metrics import compute_behavior_metrics
from behavior_risk import compute_behavior_risk
from behavior_flag import behavior_clinical_flag

def run_behavior_screening(duration_sec=10):
    metrics = compute_behavior_metrics(duration_sec)

    risk = compute_behavior_risk(
        metrics["face_presence_ratio"],
        metrics["pose_stability_ratio"]
    )

    flag = behavior_clinical_flag(risk)

    return {
        "raw_measurements": metrics,
        "risk_score": risk,
        "clinical_flag": flag
    }
