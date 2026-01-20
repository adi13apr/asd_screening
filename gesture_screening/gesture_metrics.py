import cv2
import time
import numpy as np
from gesture_utils import preprocess_frame, frame_motion

def compute_gesture_metrics(duration_sec=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = preprocess_frame(frame)

    motion_scores = []

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = preprocess_frame(frame)
        motion = frame_motion(prev_gray, gray)
        motion_scores.append(motion)
        prev_gray = gray

        cv2.imshow("Gesture Screening", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    motion_scores = np.array(motion_scores)

    return {
        "mean_motion": float(round(np.mean(motion_scores), 2)),
        "motion_variance": float(round(np.var(motion_scores), 2)),
        "repetitiveness": float(round(np.std(motion_scores), 2)),
        "duration_sec": duration_sec
    }
