import numpy as np
from filters.overlay_utils import overlay_filter


# Offsets for each mustache type
MUSTACHE_OFFSETS = {
    "mustache1": -0.02,
    "mustache2": -0.02,
    "mustache3": -0.02,
    "mustache4": -0.03,
    "mustache5": -0.02,
}


def apply_mustache_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Mustache filter with TRUE FIXED SIZE (pixel-based).
    Size will never change regardless of distance from camera.
    """


    if filter_img is None:
        return frame


    h, w, _ = frame.shape


    # --- Key landmarks ---
    nose = landmarks.landmark[1]
    lip = landmarks.landmark[13]


    nose_pos = (int(nose.x * w), int(nose.y * h))
    lip_pos = (int(lip.x * w), int(lip.y * h))


    # --- Face angle (still rotates properly) ---
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]
    left_pos = (int(left.x * w), int(left.y * h))
    right_pos = (int(right.x * w), int(right.y * h))


    dx = right_pos[0] - left_pos[0]
    dy = right_pos[1] - left_pos[1]
    angle = -np.degrees(np.arctan2(dy, dx))


    # -----------------------------------------
    # TRUE FIXED SIZE MUSTACHE (never changes)
    # -----------------------------------------
    FIXED_WIDTH = int(w * 0.20)   # 20% of the screen width (constant)
    # You can also use a constant value like: FIXED_WIDTH = 150


    filter_w = FIXED_WIDTH
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


    # --- Offset for each mustache type ---
    vertical = MUSTACHE_OFFSETS.get(filter_key, 0.10)
    offset_y = int(filter_h * vertical)


    # Position between nose + lip
    center_x = (nose_pos[0] + lip_pos[0]) // 2
    center_y = (nose_pos[1] + lip_pos[1]) // 2 + offset_y


    # Bounding box
    x1 = center_x - filter_w // 2
    y1 = center_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h


    # Save history for compatibility
    face_history[face_index] = {
        "left": left_pos,
        "right": right_pos,
        "lip": lip_pos,
        "angle": angle,
        "scale": 1.0,
    }


    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)
