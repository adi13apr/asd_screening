import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, 0.0

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_img = image[y:y+h, x:x+w]

    face_area = w * h
    img_area = image.shape[0] * image.shape[1]
    confidence = round(face_area / img_area, 2)

    return face_img, confidence
