import numpy as np
from filters.overlay_utils import overlay_filter


# Base scales and vertical offsets for each heart glasses variant
V_GLASSES_SCALES = {
    "v_glasses1": 1.6, "v_glasses2": 1.5, "v_glasses3": 1.5,
    "v_glasses4": 1.5, "v_glasses5": 1.6, "v_glasses6": 1.5,
    "v_glasses7": 1.7, "v_glasses8": 1.5,
}
V_GLASSES_VERTICAL_OFFSETS = {
    "v_glasses1": -0.05, "v_glasses2": -0.05, "v_glasses3": -0.05,
    "v_glasses4": -0.05, "v_glasses5": -0.05, "v_glasses6": -0.05,
    "v_glasses7": -0.05, "v_glasses8": -0.05,
}


def apply_heart_glasses_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Heart glasses filter with NO SMOOTHING.
    Size, angle, and position are directly calculated from landmarks.
    """


    if filter_img is None:
        return frame


    h, w, _ = frame.shape


    # --- 1. Raw landmarks ---
    left_eye_outer = landmarks.landmark[33]
    right_eye_outer = landmarks.landmark[263]
    nose_bridge = landmarks.landmark[168]
    chin = landmarks.landmark[152]


    lx, ly = int(left_eye_outer.x * w), int(left_eye_outer.y * h)
    rx, ry = int(right_eye_outer.x * w), int(right_eye_outer.y * h)
    nx, ny = int(nose_bridge.x * w), int(nose_bridge.y * h)
    cx, cy = int(chin.x * w), int(chin.y * h)


    # --- 2. Face width & height ---
    face_width = np.linalg.norm(np.array([rx, ry]) - np.array([lx, ly]))
    face_height = np.linalg.norm(np.array([cx, cy]) - np.array([nx, ny]))


    # --- 3. Base scale & vertical offset ---
    base_scale = V_GLASSES_SCALES.get(filter_key, 2.5)
    offset_factor = V_GLASSES_VERTICAL_OFFSETS.get(filter_key, -0.05)


    # --- 4. Filter size ---
    filter_w = int(face_width * base_scale)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


    # --- 5. Angle ---
    dx = rx - lx
    dy = ry - ly
    angle = -np.degrees(np.arctan2(dy, dx))


    # --- 6. Center position ---
    center_x = int((lx + rx) / 2)
    center_y = int((ly + ry) / 2 + face_height * offset_factor)


    # --- 7. Bounding box with padding ---
    padding_factor = 1.3
    box_w = int(filter_w * padding_factor)
    box_h = int(filter_h * padding_factor)


    x1 = center_x - box_w // 2
    y1 = center_y - box_h // 2
    x2 = x1 + box_w
    y2 = y1 + box_h


    # --- 8. Save history (optional, for compatibility) ---
    face_history[face_index] = {
        "center": np.array([center_x, center_y]),
        "scale": float(filter_w),
        "angle": angle
    }


    # --- 9. Apply filter ---
    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)


