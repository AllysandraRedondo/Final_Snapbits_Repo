import cv2
import numpy as np

def overlay_filter(frame, filter_img, x1, y1, x2, y2, angle=0):
    """
    Overlay filter image with rotation and alpha transparency, ensuring the 
    rotated image is not clipped by the bounding box.
    """
    if filter_img is None:
        return frame

    frame_h, frame_w = frame.shape[:2]
    
    target_w = x2 - x1
    target_h = y2 - y1

    #Resize the filter image to the target dimensions
    filter_resized = cv2.resize(filter_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    #Handle Rotation
    if angle != 0:
        center = (filter_resized.shape[1] // 2, filter_resized.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)

        # Calculate the new, larger bounding box dimensions needed for rotation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((target_h * sin) + (target_w * cos))
        new_h = int((target_h * cos) + (target_w * sin))

        # Adjust the rotation matrix to keep the image centered
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation, expanding the canvas
        filter_rotated = cv2.warpAffine(
            filter_resized, M, (new_w, new_h), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        
        # Recalculate the top-left corner of the *new* bounding box
        x1_rot = x1 - int((new_w - target_w) / 2)
        y1_rot = y1 - int((new_h - target_h) / 2)
        x2_rot = x1_rot + new_w
        y2_rot = y1_rot + new_h
        
        icon_to_overlay = filter_rotated
    else:
        x1_rot, y1_rot, x2_rot, y2_rot = x1, y1, x2, y2
        icon_to_overlay = filter_resized

    # Calculate the intersection (clipping) with the frame boundaries
    
    # Coordinates of the intersection area on the frame
    ix1 = max(0, x1_rot)
    iy1 = max(0, y1_rot)
    ix2 = min(frame_w, x2_rot)
    iy2 = min(frame_h, y2_rot)
    
    inter_w = ix2 - ix1
    inter_h = iy2 - iy1
    
    if inter_w <= 0 or inter_h <= 0:
        return frame


    icon_slice_x1 = ix1 - x1_rot
    icon_slice_y1 = iy1 - y1_rot
    
    icon_slice = icon_to_overlay[icon_slice_y1:icon_slice_y1 + inter_h, 
    icon_slice_x1:icon_slice_x1 + inter_w]

    
    frame_area = frame[iy1:iy2, ix1:ix2]
    
    if frame_area.shape[:2] != icon_slice.shape[:2]:
        return frame 

    if icon_slice.shape[2] == 4:
        alpha_channel = icon_slice[:, :, 3] / 255.0
        bgr = icon_slice[:, :, :3]
        
        for c in range(0, 3):
            frame[iy1:iy2, ix1:ix2, c] = (alpha_channel * bgr[:, :, c] +
            (1 - alpha_channel) * frame_area[:, :, c])
    elif icon_slice.shape[2] == 3:
        frame[iy1:iy2, ix1:ix2] = icon_slice
        
    return frame