import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

# Landmark indices used for head pose proxy
POSE_LANDMARKS = [30, 8, 36, 45, 48, 54]

def estimate_head_pose(gray_frame):
    faces = detector(gray_frame)

    if len(faces) == 0:
        return None

    shape = predictor(gray_frame, faces[0])

    points = np.array([
        [shape.part(i).x, shape.part(i).y]
        for i in POSE_LANDMARKS
    ])

    # Variance of landmark positions as pose instability proxy
    pose_variance = np.var(points, axis=0).sum()
    return float(round(pose_variance, 2))
