from eeg.eeg_module import run_eeg_screening
from vision.face_module import run_face_screening
from visual_attention.behavior_module import run_behavior_screening
from gesture_screening.gesture_module import run_gesture_screening
from fusion.fusion_module import run_asd_fusion


def run_full_screening_from_files(
    eeg_csv_path: str,
    face_image_path: str,
    video_path: str
):
    print("\n=== ASD Screening (API) Started ===\n")

    print(">>> EEG START")
    eeg_output = run_eeg_screening(eeg_csv_path)
    print(">>> EEG DONE")

    print(">>> FACE START")
    face_output = run_face_screening(face_image_path)
    print(">>> FACE DONE")

    print(">>> VISUAL START")
    visual_attention_output = run_behavior_screening(video_path=video_path)
    print(">>> VISUAL DONE")

    print(">>> GESTURE START")
    gesture_output = run_gesture_screening(video_path=video_path)
    print(">>> GESTURE DONE")

    print(">>> FUSION START")

    # 5️⃣ Fusion
    final_output = run_asd_fusion(
        eeg_output=eeg_output,
        visual_attention_output=visual_attention_output,
        gesture_output=gesture_output,
        face_output=face_output
    )

    print("\n=== FINAL ASD SCREENING RESULT (API) ===\n")
    print(final_output)

    return final_output
