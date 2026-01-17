# vision/gesture_detection.py

import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose

def get_gesture_features(video_source=0, max_seconds=5):
    cap = cv2.VideoCapture(video_source)
    pose = mp_pose.Pose()

    wrist_positions = []
    start_time = time.time()

    while time.time() - start_time < max_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            wrist = result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            wrist_positions.append([wrist.x, wrist.y])

    cap.release()
    pose.close()

    if len(wrist_positions) < 10:
        return np.array([0.0], dtype=np.float32)

    positions = np.array(wrist_positions)

    # Velocity magnitude
    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    # Gesture variability (proxy for atypical movements)
    gesture_variability = np.std(velocity)

    return np.array([gesture_variability], dtype=np.float32)
