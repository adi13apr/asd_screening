def fusion_clinical_flag(score):
    if score >= 0.7:
        return (
            "Elevated multimodal screening risk observed. "
            "Recommend comprehensive clinical evaluation and developmental assessment."
        ), "Elevated"
    elif score >= 0.4:
        return (
            "Moderate screening risk with converging behavioral and neural signals. "
            "Clinical monitoring and follow-up recommended."
        ), "Moderate"
    else:
        return (
            "Low screening risk based on current multimodal signals. "
            "Routine developmental monitoring advised."
        ), "Low"
