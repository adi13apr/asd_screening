import cv2
import time

from visual_attention.attention_metrics import compute_behavior_metrics


def run_behavior_screening(video_path=None, duration_sec=None):
    """
    Runs visual attention screening.

    Modes:
    - video_path: process uploaded video file (API / web mode)
    - duration_sec: open webcam for N seconds (local dev mode)
    """

    if video_path:
        print(f"[VisualAttention] Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

    elif duration_sec:
        print(f"[VisualAttention] Opening webcam for {duration_sec} seconds")
        cap = cv2.VideoCapture(0)
        start_time = time.time()

    else:
        raise ValueError("Either video_path or duration_sec must be provided")

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    frame_count = 0
    all_metrics = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # -----------------------------
        # Compute visual attention metrics
        # -----------------------------
        metrics = compute_behavior_metrics(frame)
        all_metrics.append(metrics)

        # -----------------------------
        # Webcam mode: stop after duration
        # -----------------------------
        if duration_sec:
            if time.time() - start_time > duration_sec:
                break

    cap.release()

    if not all_metrics:
        raise RuntimeError("No frames processed for visual attention")

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    avg_gaze_score = sum(m["gaze_score"] for m in all_metrics) / len(all_metrics)
    avg_blink_rate = sum(m["blink_rate"] for m in all_metrics) / len(all_metrics)
    avg_joint_attention = sum(m["joint_attention"] for m in all_metrics) / len(all_metrics)

    # Simple risk heuristic (example)
    risk_score = float(
        max(0.0, min(1.0, 1.0 - avg_joint_attention))
    )

    clinical_flag = (
        "Reduced visual attention / joint attention patterns detected"
        if risk_score > 0.5
        else "Visual attention patterns within expected range"
    )

    return {
        "raw_measurements": {
            "avg_gaze_score": float(avg_gaze_score),
            "avg_blink_rate": float(avg_blink_rate),
            "avg_joint_attention": float(avg_joint_attention),
            "frames_analyzed": frame_count
        },
        "risk_score": float(risk_score),
        "clinical_flag": clinical_flag
    }
