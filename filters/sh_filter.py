import numpy as np
from filters.overlay_utils import overlay_filter



# SETTINGS FOR EACH SHARK FILTER (adjust scale & offsets)
SHARK_SETTINGS = {
    "sh1": {"scale": 3.0, "offset_x": 0.00, "offset_y": -0.20},
    "sh2": {"scale": 2.5, "offset_x": 0.00, "offset_y": -0.18},
    "sh3": {"scale": 3.0, "offset_x": 0.00, "offset_y": -0.45},
    "sh4": {"scale": 3.0, "offset_x": -0.08, "offset_y": -0.15},
    "sh5": {"scale": 3.0, "offset_x": 0.08, "offset_y": -0.15},
    "sh6": {"scale": 3.0, "offset_x": 0.00, "offset_y": -0.45},
    "sh7": {"scale": 2.8, "offset_x": 0.00, "offset_y": -0.20},
    "sh8": {"scale": 3.2, "offset_x": 0.00, "offset_y": -0.30},
}


def apply_sh_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Shark filter (no smoothing, per-filter scale & offset)
    IMPORTANT: Accepts 6 arguments for compatibility with main code.
    """


    if filter_img is None:
        return frame


    h, w, _ = frame.shape


   
    # RAW LANDMARKS
    nose = landmarks.landmark[1]
    left = landmarks.landmark[234]
    right = landmarks.landmark[454]


    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_x, left_y = int(left.x * w), int(left.y * h)
    right_x, right_y = int(right.x * w), int(right.y * h)


    # FACE WIDTH & ANGLE
    dx = right_x - left_x
    dy = right_y - left_y
    face_width = np.hypot(dx, dy)
    angle = -np.degrees(np.arctan2(dy, dx))



    # SETTINGS

    cfg = SHARK_SETTINGS.get(filter_key, {"scale": 3.2, "offset_x": 0.00, "offset_y": -0.20})
    scale_factor = cfg["scale"]
    offset_x = int(face_width * cfg["offset_x"])
    offset_y = int(face_width * cfg["offset_y"])


  
    # FILTER SIZE
    filter_w = int(face_width * scale_factor)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


   
    # POSITIONING
    center_x = nose_x + offset_x
    center_y = nose_y + offset_y


    # BOUNDING BOX

    x1 = center_x - filter_w // 2
    y1 = center_y - filter_h // 2
    x2 = x1 + filter_w
    y2 = y1 + filter_h


    
    # HISTORY (required by main code)
    face_history[face_index] = {
        "angle": angle,
        "scale": scale_factor
    }


    # APPLY FILTER
    return overlay_filter(frame, filter_img, int(x1), int(y1), int(x2), int(y2), angle)
