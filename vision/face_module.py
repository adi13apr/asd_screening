import cv2
from face_utils import detect_face
from cnn_inference import load_model, predict_face
from face_risk import compute_face_risk
from face_flag import face_clinical_flag
from image_encoder_arch import get_image_encoder
MODEL_PATH = "models/image_encoder.pt"

model = get_image_encoder()

def run_face_screening(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded")

    face_img, face_conf = detect_face(image)

    if face_img is None:
        return {
            "raw_measurements": {
                "face_detected": False
            },
            "risk_score": 0.0,
            "clinical_flag": "Face not detected reliably for screening"
        }

    cnn_prob = predict_face(model, face_img)
    risk = compute_face_risk(cnn_prob, face_conf)
    flag = face_clinical_flag(risk)

    return {
        "raw_measurements": {
            "cnn_probability": float(cnn_prob),
            "face_detection_confidence": float(face_conf)
        },
        "risk_score": risk,
        "clinical_flag": flag
    }
