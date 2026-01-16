import numpy as np
from filters.overlay_utils import overlay_filter


# ----------------------------------------------------------
# SETTINGS FOR EACH CAT FILTER (adjust any value freely)
# ----------------------------------------------------------
CAT_SETTINGS = {
    "cat1": {"scale": 1.30, "offset_x": 0.00, "offset_y": 0.06},
    "cat2": {"scale": 1.42, "offset_x": 0.03, "offset_y": 0.10},
    "cat3": {"scale": 1.42, "offset_x": 0.03, "offset_y": 0.10},
}


def apply_cat_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Cat filter (no smoothing, per-filter scale, per-filter offset)
    IMPORTANT: Must accept 6 arguments for compatibility.
    """


    if filter_img is None:
        return frame


    h, w, _ = frame.shape


    # Stable landmarks
    top_head = landmarks.landmark[10]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]


    top_x, top_y = int(top_head.x * w), int(top_head.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)


    # Face width & angle
    dx = right_x - left_x
    dy = right_y - left_y
    face_width = np.hypot(dx, dy)
    angle = -np.degrees(np.arctan2(dy, dx))


    # Use settings for this cat filter
    cfg = CAT_SETTINGS.get(filter_key, {"scale": 1.30, "offset_x": 0.00, "offset_y": 0.12})


    scale_factor = cfg["scale"]


    filter_w = int(face_width * scale_factor)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


    # Apply offsets
    offset_x = int(filter_w * cfg["offset_x"])
    offset_y = int(filter_h * cfg["offset_y"])


    center_x = top_x + offset_x
    center_y = top_y - offset_y


    # Bounding box
    x1 = center_x - filter_w // 2
    y1 = center_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h


    # Save info to history (not used but required)
    face_history[face_index] = {
        "angle": angle,
        "scale": scale_factor
    }


    # Apply overlay
    return overlay_filter(frame, filter_img, int(x1), int(y1), int(x2), int(y2), angle)
