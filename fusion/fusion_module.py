from fusion.fusion_logic import compute_fused_risk
from fusion.fusion_flag import fusion_clinical_flag

def run_asd_fusion(
    eeg_output,
    visual_attention_output,
    gesture_output,
    face_output
):
    scores = {
        "eeg": eeg_output["risk_score"],
        "visual_attention": visual_attention_output["risk_score"],
        "gesture": gesture_output["risk_score"],
        "face": face_output["risk_score"]
    }

    fused_score = compute_fused_risk(scores)
    recommendation, level = fusion_clinical_flag(fused_score)

    return {
        "asd_risk_score": fused_score,
        "risk_level": level,
        "evidence_summary": {
            "eeg": eeg_output["clinical_flag"],
            "visual_attention": visual_attention_output["clinical_flag"],
            "gesture": gesture_output["clinical_flag"],
            "face": face_output["clinical_flag"]
        },
        "clinical_recommendation": recommendation
    }
