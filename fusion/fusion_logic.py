from fusion.fusion_weights import FUSION_WEIGHTS

def compute_fused_risk(scores):
    """
    scores = {
        "eeg": 0.0–1.0,
        "visual_attention": 0.0–1.0,
        "gesture": 0.0–1.0,
        "face": 0.0–1.0
    }
    """

    fused_score = 0.0

    for key, weight in FUSION_WEIGHTS.items():
        fused_score += scores.get(key, 0.0) * weight

    return round(float(fused_score), 2)
