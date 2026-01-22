def compute_gesture_risk(metrics):
    risk = 0.0

    # Very low motion → limited gestural engagement
    if metrics["mean_motion"] < 500:
        risk += 0.3

    # Very high repetitiveness → possible stereotypy proxy
    if metrics["repetitiveness"] < 50:
        risk += 0.3

    # Low variance → repetitive movement
    if metrics["motion_variance"] < 1000:
        risk += 0.2

    return float(round(min(risk, 1.0), 2))
