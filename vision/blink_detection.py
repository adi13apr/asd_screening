import cv2
import numpy as np
from scipy.spatial import distance
from mediapipe import solutions
import time
# Set thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

mp_face_mesh = solutions.face_mesh

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_blink_features(show=False,duration=15):
    blink_count = 0
    frame_counter = 0
    frames = 0
    start_time = time.time()


    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [263, 387, 385, 362, 380, 373]

                left_eye = []
                right_eye = []
                for idx in left_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye.append((x, y))
                for idx in right_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye.append((x, y))

                left_eye = np.array(left_eye)
                right_eye = np.array(right_eye)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < EYE_AR_THRESH:
                    frame_counter += 1
                else:
                    if frame_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_count += 1
                    frame_counter = 0

            if show:
                elapsed = int(time.time() - start_time)
                cv2.putText(
                    frame,
                    f"Observing... {elapsed}/{duration}s",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("Blink Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

            frames += 1

    cap.release()
    cv2.destroyAllWindows()
    blink_rate = blink_count / duration
    return np.array([blink_rate],dtype=np.float32)
