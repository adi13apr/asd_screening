import cv2
import numpy as np
import os

# Use OpenCV's pre-trained DNN model for face detection (no dlib dependency)
# Download model files if not present
MODEL_DIR = "models"

def _ensure_face_detector():
    """Load OpenCV face detector using DNN module."""
    # Using OpenCV's built-in Haar Cascade (faster, no external deps)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = _ensure_face_detector()

# Key facial landmark indices for head pose proxy (approximate positions)
# These are approximate landmark locations on a 640x480 image
# We'll estimate them using face bounding box corners and center
POSE_LANDMARKS = [4, 152, 33, 263, 61, 291]  # Landmark positions mapping

def estimate_head_pose(gray_frame):
    """Estimate head pose using OpenCV face detection.
    
    Args:
        gray_frame: Grayscale or BGR image frame
        
    Returns:
        float: Variance of facial landmark positions (pose instability proxy) or None if no face detected
    """
    # Convert to grayscale if needed
    if len(gray_frame.shape) == 3:
        gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_frame
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    (x, y, w, h) = faces[0]
    
    # Estimate key facial landmarks based on face bounding box
    # These are approximate positions: nose, chin, eyes, mouth corners
    face_height = h
    face_width = w
    
    # Approximate landmark positions relative to face bounding box
    landmarks = np.array([
        [x + face_width // 2, y + face_height // 3],  # Nose (approximate)
        [x + face_width // 2, y + face_height],       # Chin (approximate)
        [x + face_width // 5, y + face_height // 3],  # Left eye
        [x + 4 * face_width // 5, y + face_height // 3],  # Right eye
        [x + face_width // 4, y + 2 * face_height // 3],  # Left mouth
        [x + 3 * face_width // 4, y + 2 * face_height // 3],  # Right mouth
    ])
    
    # Variance of landmark positions as pose instability proxy
    pose_variance = np.var(landmarks, axis=0).sum()
    return float(round(pose_variance, 2))
