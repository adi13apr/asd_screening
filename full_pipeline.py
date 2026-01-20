from eeg.eeg_module import run_eeg_screening
from vision.face_module import run_face_screening
from visual_attention.behavior_module import run_behavior_screening
from gesture_screening.gesture_module import run_gesture_screening
from fusion.fusion_module import run_asd_fusion
from vision.capture_face import capture_face_snapshot

def run_full_screening():
    print("\n=== ASD Screening Started ===\n")

    # 1️⃣ EEG (CSV upload)
    eeg_csv = input("Enter path to EEG CSV file: ")
    eeg_output = run_eeg_screening(eeg_csv)

    # 2️⃣ Face Image (runtime snapshot)
    print("\nCapturing face image...")
    face_image_path = capture_face_snapshot()
    face_output = run_face_screening(face_image_path)

    # 3️⃣ Visual Attention & Eye Gaze (runtime video)
    print("\nRunning visual attention screening...")
    visual_attention_output = run_behavior_screening(duration_sec=10)

    # 4️⃣ Gesture / Motor Behavior (runtime video)
    print("\nRunning gesture screening...")
    gesture_output = run_gesture_screening(duration_sec=10)

    # 5️⃣ Multimodal Fusion (REAL values)
    final_output = run_asd_fusion(
        eeg_output=eeg_output,
        visual_attention_output=visual_attention_output,
        gesture_output=gesture_output,
        face_output=face_output
    )

    print("\n=== FINAL ASD SCREENING RESULT ===\n")
    print(final_output)

    return final_output


if __name__ == "__main__":
    run_full_screening()
