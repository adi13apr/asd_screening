import cv2
import os

from vision.face_utils import detect_face
from vision.cnn_inference import predict_face
from vision.face_risk import compute_face_risk
from vision.face_flag import face_clinical_flag
from vision.image_encoder_arch import get_image_encoder

MODEL_PATH = "models/image_encoder.pt"

# Load model once (good practice)
model = get_image_encoder()


def run_face_screening(image_path=None):
    """
    Runs face screening.

    Modes:
    - image_path: process uploaded image file (API / web mode)
    - None: open webcam and capture snapshot (local dev mode)
    """

    # -----------------------------
    # Mode A: API / Web (image file)
    # -----------------------------
    if image_path:
        print(f"[Face] Processing image file: {image_path}")

        if not os.path.exists(image_path):
            raise ValueError(f"Face image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded from file")

    # -----------------------------
    # Mode B: Local Dev (webcam)
    # -----------------------------
    else:
        print("[Face] Opening webcam for face snapshot")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam for face capture")

        ret, image = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture face snapshot from webcam")

        # Optional: save snapshot for debugging
        os.makedirs("tmp_face", exist_ok=True)
        debug_path = "tmp_face/debug_face.jpg"
        cv2.imwrite(debug_path, image)
        print(f"[Face] Saved debug snapshot to {debug_path}")

    # -----------------------------
    # Face Detection
    # -----------------------------
    face_img, face_conf = detect_face(image)

    if face_img is None:
        return {
            "raw_measurements": {
                "face_detected": False
            },
            "risk_score": 0.0,
            "clinical_flag": "Face not detected reliably for screening"
        }

    # -----------------------------
    # CNN Prediction
    # -----------------------------
    cnn_prob = predict_face(model, face_img)

    # -----------------------------
    # Risk + Clinical Flag
    # -----------------------------
    risk = compute_face_risk(cnn_prob, face_conf)
    flag = face_clinical_flag(risk)

    return {
        "raw_measurements": {
            "cnn_probability": float(cnn_prob),
            "face_detection_confidence": float(face_conf)
        },
        "risk_score": float(risk),
        "clinical_flag": flag
    }
