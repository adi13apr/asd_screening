from fusion.fusion_module import run_asd_fusion

# Mock inputs (replace with real module outputs)
eeg_output = {
    "risk_score": 0.65,
    "clinical_flag": "Atypical frontal and temporal EEG activity"
}

visual_attention_output = {
    "risk_score": 0.7,
    "clinical_flag": "Reduced facial and eye-region engagement"
}

gesture_output = {
    "risk_score": 0.5,
    "clinical_flag": "Mild reduction in gestural variability"
}

face_output = {
    "risk_score": 0.3,
    "clinical_flag": "Facial patterns within expected screening range"
}

if __name__ == "__main__":
    result = run_asd_fusion(
        eeg_output,
        visual_attention_output,
        gesture_output,
        face_output
    )

    print("\nFinal ASD Screening Output:\n")
    print(result)
