import cv2
import numpy as np
from visual_attention.head_pose import estimate_head_pose


def compute_behavior_metrics(frame):
    """
    Compute visual attention metrics for a SINGLE FRAME.
    Used in API / video file mode.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pose_mag = estimate_head_pose(gray)

    # -----------------------------
    # Force pose_mag to scalar
    # -----------------------------
    if pose_mag is not None:
        if hasattr(pose_mag, "item"):
            pose_mag = float(pose_mag.item())
        elif isinstance(pose_mag, (list, tuple, np.ndarray)):
            pose_mag = float(pose_mag[0])

    # -----------------------------
    # Derive simple frame metrics
    # -----------------------------
    gaze_score = 1.0 if pose_mag is not None else 0.0
    blink_rate = 0.0  # placeholder (implement if you have blink logic)
    joint_attention = 1.0 if pose_mag is not None and pose_mag < 150 else 0.0

    return {
        "gaze_score": float(gaze_score),
        "blink_rate": float(blink_rate),
        "joint_attention": float(joint_attention)
    }
