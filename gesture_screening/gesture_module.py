import cv2
import numpy as np
from gesture_screening.gesture_metrics import compute_gesture_metrics_frame

def run_gesture_screening(video_path=None, duration_sec=None):
    """
    Gesture screening.

    - video_path: process uploaded video file (API mode)
    - duration_sec: open webcam for N seconds (local dev)
    """

    if video_path:
        cap = cv2.VideoCapture(video_path)
    elif duration_sec:
        cap = cv2.VideoCapture(0)
    else:
        raise ValueError("Either video_path or duration_sec must be provided")

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    motion_scores = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_gray is None:
            from gesture_screening.gesture_utils import preprocess_frame
            prev_gray = preprocess_frame(frame)
            continue

        motion, prev_gray = compute_gesture_metrics_frame(prev_gray, frame)
        motion_scores.append(motion)

    cap.release()

    if not motion_scores:
        raise RuntimeError("No gesture frames processed")

    motion_scores = np.array(motion_scores)

    mean_motion = float(np.mean(motion_scores))
    motion_variance = float(np.var(motion_scores))
    repetitiveness = float(np.std(motion_scores))

    # Simple risk heuristic
    risk_score = float(min(1.0, mean_motion / 50.0))

    clinical_flag = (
        "Elevated repetitive motor movement detected"
        if risk_score > 0.5
        else "Motor behavior within expected range"
    )

    return {
        "raw_measurements": {
            "mean_motion": round(mean_motion, 2),
            "motion_variance": round(motion_variance, 2),
            "repetitiveness": round(repetitiveness, 2)
        },
        "risk_score": float(round(risk_score, 2)),
        "clinical_flag": clinical_flag
    }
