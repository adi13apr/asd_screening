from vision.capture_face import capture_face_snapshot
from vision.face_module import run_face_screening

if __name__ == "__main__":
    img_path = capture_face_snapshot()

    if img_path:
        output = run_face_screening(img_path)
        print("\nFace Screening Output:\n")
        print(output)
