import cv2

def capture_face_snapshot(save_path="runtime_face.jpg"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    print("Press 's' to capture image | 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return save_path
