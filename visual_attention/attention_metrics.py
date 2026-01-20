import cv2
import time
from visual_attention.head_pose import estimate_head_pose

def compute_behavior_metrics(duration_sec=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    total_frames = 0
    face_frames = 0
    stable_pose_frames = 0

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose_mag = estimate_head_pose(gray)

        total_frames += 1

        if pose_mag is not None:
            face_frames += 1
            if pose_mag < 150:   # dlib scale threshold
                stable_pose_frames += 1

        cv2.imshow("Behavior Screening (dlib)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {
        "face_presence_ratio": round(face_frames / total_frames, 2) if total_frames else 0.0,
        "pose_stability_ratio": round(stable_pose_frames / face_frames, 2) if face_frames else 0.0,
        "duration_sec": duration_sec
    }
