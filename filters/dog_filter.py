import numpy as np
import math
from filters.overlay_utils import overlay_filter


DOG_PRESETS = {
    "dog1": {
        "BASE_SCALE": 1.0,
        "WIDTH_SCALE": 2.4,
        "HEIGHT_RATIO": 1.15,
        "OFFSET_X": 0.0,
        "OFFSET_Y": -0.15
    },
    "dog2": {
        "BASE_SCALE": 1.0,
        "WIDTH_SCALE": 2.4,
        "HEIGHT_RATIO": 1.05,
        "OFFSET_X": 0.0,
        "OFFSET_Y": -0.10
    },
    "dog3": {
        "BASE_SCALE": 1.0,
        "WIDTH_SCALE": 2.4,
        "HEIGHT_RATIO": 1.07,
        "OFFSET_X": 0.0,
        "OFFSET_Y": -0.25
    }
}


# ---------------- LANDMARKS ----------------
NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263




def apply_dog_filter(frame, landmarks, filter_img, filter_key, FACE_HISTORY, face_index):
    h, w = frame.shape[:2]


    # 🔹 Load preset
    preset = DOG_PRESETS.get(filter_key, DOG_PRESETS["dog1"])


    BASE_SCALE = preset["BASE_SCALE"]
    WIDTH_SCALE = preset["WIDTH_SCALE"]
    HEIGHT_RATIO = preset["HEIGHT_RATIO"]
    OFFSET_X = preset["OFFSET_X"]
    OFFSET_Y = preset["OFFSET_Y"]


    # -------- Landmarks --------
    nose = landmarks.landmark[NOSE_TIP]
    left_eye = landmarks.landmark[LEFT_EYE]
    right_eye = landmarks.landmark[RIGHT_EYE]


    nx, ny = int(nose.x * w), int(nose.y * h)
    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)


    # -------- Face scale --------
    eye_dist = np.hypot(rx - lx, ry - ly)
    width = eye_dist * WIDTH_SCALE * BASE_SCALE
    height = width * HEIGHT_RATIO


    # -------- Offsets --------
    offset_x = width * OFFSET_X
    offset_y = height * OFFSET_Y


    # -------- Head tilt (roll) --------
    angle = -math.degrees(math.atan2(ry - ly, rx - lx))


    # -------- Position --------
    x1 = int(nx - width / 2 + offset_x)
    y1 = int(ny - height / 2 + offset_y)
    x2 = int(x1 + width)
    y2 = int(y1 + height)


    # -------- Overlay --------
    frame = overlay_filter(
        frame,
        filter_img,
        x1, y1, x2, y2,
        angle=angle
    )


    return frame
