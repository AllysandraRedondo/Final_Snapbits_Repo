import numpy as np
from filters.overlay_utils import overlay_filter



# LAG REDUCTION (Responsiveness)

smooth_factor = 0.4  # small smoothing for minor jitter; can set to 1 for instant tracking


def apply_christmas_glasses_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):
    """
    Applies Christmas glasses filter, supporting multiple faces with optional smoothing.
    The overlay sticks to the face even during fast movements.
    """
   
    # Retrieve current face's history, initialize if missing 
    history = face_history.get(face_index, {})
    previous_center = history.get("center")
    previous_scale = history.get("scale")
    previous_angle = history.get("angle")


    if filter_img is None:
        return frame


    h, w, _ = frame.shape


    # Key Landmarks & Geometry 
    left_eye_outer = landmarks.landmark[33]  
    right_eye_outer = landmarks.landmark[263]  
    nose_bridge = landmarks.landmark[168]
    chin = landmarks.landmark[152]


    lx, ly = int(left_eye_outer.x * w), int(left_eye_outer.y * h)
    rx, ry = int(right_eye_outer.x * w), int(right_eye_outer.y * h)
    nx, ny = int(nose_bridge.x * w), int(nose_bridge.y * h)
    cx, cy = int(chin.x * w), int(chin.y * h)


    face_width = np.linalg.norm(np.array([rx, ry]) - np.array([lx, ly]))
    face_height = np.linalg.norm(np.array([cx, cy]) - np.array([nx, ny]))


    #  Base scales & offsets 
    base_scales = {
        "c_glasses1": 1.6, "c_glasses2": 1.7, "c_glasses3": 1.7,
        "c_glasses4": 1.8, "c_glasses5": 1.8
    }
    vertical_offsets = {
        "c_glasses1": -0.19, "c_glasses2": -0.19, "c_glasses3": -0.19,
        "c_glasses4": -0.19, "c_glasses5": -0.18
    }


    base_scale = base_scales.get(filter_key, 2.5)
    offset_factor = vertical_offsets.get(filter_key, -0.10)


    #  Target size 
    target_width = int(face_width * base_scale)
    target_height = int(target_width * filter_img.shape[0] / filter_img.shape[1])


    #  Rotation 
    dx = rx - lx
    dy = ry - ly
    raw_angle = np.degrees(np.arctan2(dy, dx))
    corrected_angle = -raw_angle


    #  Center position 
    center_x_raw = int((lx + rx) / 2)
    center_y_raw = int((ly + ry) / 2 + face_height * offset_factor)
    target_center = np.array([center_x_raw, center_y_raw])


    #  Optional smoothing for jitter 
    if previous_center is None:
        smoothed_center = target_center
        smoothed_width = target_width
        angle = corrected_angle
    else:
        # Detect fast movement to avoid sliding
        movement = np.linalg.norm(target_center - previous_center)
        if movement > face_width * 0.05:  # >5% of face width
            smoothed_center = target_center
            smoothed_width = target_width
            angle = corrected_angle
        else:
            smoothed_center = previous_center + (target_center - previous_center) * smooth_factor
            smoothed_width = previous_scale + (target_width - previous_scale) * smooth_factor
            angle = previous_angle + (corrected_angle - previous_angle) * smooth_factor


    center_x, center_y = int(smoothed_center[0]), int(smoothed_center[1])
    smoothed_height = int(smoothed_width * filter_img.shape[0] / filter_img.shape[1])


    # --- Bounding box ---
    padding_factor = 1.3
    box_w = int(smoothed_width * padding_factor)
    box_h = int(smoothed_height * padding_factor)
    x1 = center_x - box_w // 2
    y1 = center_y - box_h // 2
    x2 = x1 + box_w
    y2 = y1 + box_h


    #  Save updated history 
    face_history[face_index] = {
        "center": smoothed_center,
        "scale": float(smoothed_width),
        "angle": angle
    }


    # Apply overlay 
    frame = overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)
    return frame
