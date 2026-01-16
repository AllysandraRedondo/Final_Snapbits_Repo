import cv2
import numpy as np
from filters.overlay_utils import overlay_filter


def apply_halloween_mask_filter(frame, landmarks, filter_img, filter_key, face_history, face_index):


    h, w, _ = frame.shape
    if filter_img is None:
        return frame


    # Key landmarks
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]


    # Pixel coordinates
    left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
    right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)


    # Face geometry & rotation
    face_width = np.hypot(right_x - left_x, right_y - left_y)
    dx, dy = right_x - left_x, right_y - left_y
    angle = -np.degrees(np.arctan2(dy, dx))


   
    # Base scale (kept exactly)
   
    base_scales = {
        "h_mask1": 2.2, "h_mask2": 2.3, "h_mask3": 2.0, "h_mask4": 2.0,
        "h_mask5": 2.2, "h_mask6": 2.0, "h_mask7": 3.5, "h_mask8": 2.2,
        "h_mask9": 2.0, "h_mask10": 2.0, "h_mask11": 2.53, "h_mask12": 5.30,
        "h_mask13": 3.50, "h_mask14": 3.70, "h_mask15": 3.1, "h_mask16": 3.1
    }
    base_scale = base_scales.get(filter_key, 2.0)


    # DIRECT scale (NO smoothing)
    
    filter_w = int(face_width * base_scale)
    filter_h = int(filter_w * filter_img.shape[0] / filter_img.shape[1])


    
    # Direct position (NO smoothing)
    
    eye_center_x = (left_x + right_x) // 2
    eye_center_y = (left_y + right_y) // 2
    center_x, center_y = eye_center_x, eye_center_y


    
    # Offsets (same as your code)
   
    offsets = {
        "h_mask1": (0, int(-filter_h * 0.23)), "h_mask2": (0, int(-filter_h * 0.15)),
        "h_mask3": (0, int(-filter_h * 0.05)), "h_mask4": (0, int(-filter_h * 0.15)),
        "h_mask5": (0, int(-filter_h * 0.13)), "h_mask6": (0, int(-filter_h * 0.15)),
        "h_mask7": (0, int(-filter_h * 0.02)), "h_mask8": (0, int(-filter_h * 0.21)),
        "h_mask9": (0, int(-filter_h * 0.30)), "h_mask10": (0, int(-filter_h * 0.10)),
        "h_mask11": (0, int(-filter_h * 0.09)), "h_mask12": (0, int(-filter_h * 0.15)),
        "h_mask13": (0, int(-filter_h * 0.15)), "h_mask14": (0, int(-filter_h * 0.13)),
        "h_mask15": (0, int(-filter_h * 0.10)), "h_mask16": (0, int(-filter_h * 0.06))
    }
    offset_x, offset_y = offsets.get(filter_key, (0, 0))


    center_x += offset_x
    center_y += offset_y


   
    # Final bounding box
  
    x1 = int(center_x - filter_w // 2)
    y1 = int(center_y - filter_h // 2)
    x2 = int(x1 + filter_w)
    y2 = int(y1 + filter_h)


    
    # No smoothing → do NOT save history
    


    return overlay_filter(frame, filter_img, x1, y1, x2, y2, angle)