import cv2
import numpy as np
from filters.overlay_utils import overlay_filter


# Offsets and base scales for each birthday glasses variant
B_GLASSES_SCALES = {
    "b_glasses1": 1.5, "b_glasses2": 1.5, "b_glasses3": 1.8,
    "b_glasses4": 1.8, "b_glasses5": 1.7, "b_glasses6": 1.5,
    "b_glasses7": 1.5, "b_glasses8": 1.7, "b_glasses9": 1.6,
    "b_glasses10": 1.9, "b_glasses11": 1.5, "b_glasses12": 1.5,
    "b_glasses13": 1.5, "b_glasses14": 1.55, "b_glasses15": 1.5,
    "b_glasses16": 1.5, "b_glasses17": 1.5
}


B_GLASSES_VERTICAL_OFFSETS = {
    "b_glasses1": -0.10, "b_glasses2": -0.10, "b_glasses3": -0.16,
    "b_glasses4": -0.18, "b_glasses5": -0.15, "b_glasses6": -0.10,
    "b_glasses7": -0.15, "b_glasses8": -0.20, "b_glasses9": -0.15,
    "b_glasses10": -0.19, "b_glasses11": -0.10, "b_glasses12": -0.10,
    "b_glasses13": -0.11, "b_glasses14": -0.15, "b_glasses15": -0.10,
    "b_glasses16": -0.21, "b_glasses17": -0.13
}


def apply_birthday_glasses_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Birthday glasses filter with NO SMOOTHING.
    Size, angle, and position are directly from landmarks.
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
    base_scale = B_GLASSES_SCALES.get(filter_key, 2.5)
    vertical_offset = B_GLASSES_VERTICAL_OFFSETS.get(filter_key, -0.10)


    # --- 4. Filter size ---
    filter_w = int(face_width * base_scale)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


    # --- 5. Angle ---
    dx = rx - lx
    dy = ry - ly
    angle = -np.degrees(np.arctan2(dy, dx))


    # --- 6. Center position ---
    center_x = int((lx + rx) / 2)
    center_y = int((ly + ry) / 2 + face_height * vertical_offset)


    # --- 7. Bounding box ---
    padding_factor = 1.3
    box_w = int(filter_w * padding_factor)
    box_h = int(filter_h * padding_factor)


    x1 = center_x - box_w // 2
    y1 = center_y - box_h // 2
    x2 = x1 + box_w
    y2 = y1 + box_h


    # --- 8. Save history (optional, kept for compatibility) ---
    face_history[face_index] = {
        "center": np.array([center_x, center_y]),
        "scale": float(filter_w),
        "angle": angle
    }


    # --- 9. Apply filter ---
    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)