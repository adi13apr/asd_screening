import numpy as np
from gesture_screening.gesture_utils import preprocess_frame, frame_motion

def compute_gesture_metrics_frame(prev_gray, frame):
    """
    Compute gesture motion metrics for a SINGLE FRAME.
    Used for API / uploaded video processing.
    """

    gray = preprocess_frame(frame)
    motion = frame_motion(prev_gray, gray)

    # Force scalar
    if hasattr(motion, "item"):
        motion = float(motion.item())
    elif isinstance(motion, (list, tuple, np.ndarray)):
        motion = float(motion[0])

    return motion, gray
