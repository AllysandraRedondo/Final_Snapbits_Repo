import os
import cv2

def read_image_from_local(path):
    """Safely read an image from the local filesystem with transparency."""
    try:
        if not os.path.exists(path):
            print(f" Local file not found: {path}")
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f" Failed to load image data from: {path}")
        return img
    except Exception as e:
        print(f" Error loading image from {path}: {e}")
        return None


FILTER_IMAGES = {
    "Christmas": [
        {"key": "c_glasses1", "src": "/static/filter_images/christmas/c_glasses1.png"},
        {"key": "c_glasses2", "src": "/static/filter_images/christmas/c_glasses2.png"},
        {"key": "c_glasses3", "src": "/static/filter_images/christmas/c_glasses3.png"},
        {"key": "c_glasses4", "src": "/static/filter_images/christmas/c_glasses4.png"},
        {"key": "c_glasses5", "src": "/static/filter_images/christmas/c_glasses5.png"},
    ],
    
    "Heartsday": [
        {"key": "v_glasses1", "src": "/static/filter_images/hearts_day/v_glasses1.png"},
        {"key": "v_glasses2", "src": "/static/filter_images/hearts_day/v_glasses2.png"},
        {"key": "v_glasses3", "src": "/static/filter_images/hearts_day/v_glasses3.png"},
        {"key": "v_glasses4", "src": "/static/filter_images/hearts_day/v_glasses4.png"},
        {"key": "v_glasses5", "src": "/static/filter_images/hearts_day/v_glasses5.png"},
        {"key": "v_glasses6", "src": "/static/filter_images/hearts_day/v_glasses6.png"},
        {"key": "v_glasses7", "src": "/static/filter_images/hearts_day/v_glasses7.png"},
    ],

    "Others": [
        {"key": "mustache1", "src": "/static/filter_images/Others/mustache1.png"},
        {"key": "mustache2", "src": "/static/filter_images/Others/mustache2.png"},
        {"key": "mustache3", "src": "/static/filter_images/Others/mustache3.png"},
        {"key": "mustache4", "src": "/static/filter_images/Others/mustache4.png"},
        {"key": "mustache5", "src": "/static/filter_images/Others/mustache5.png"},
        {"key": "cat1", "src": "/static/filter_images/Others/cat1.png"},
        {"key": "cat2", "src": "/static/filter_images/Others/cat2.png"},
        {"key": "cat3", "src": "/static/filter_images/Others/cat3.png"},
        {"key": "sh1", "src": "/static/filter_images/Others/sh1.png"},
        {"key": "sh2", "src": "/static/filter_images/Others/sh2.png"},
        {"key": "sh3", "src": "/static/filter_images/Others/sh3.png"},
        {"key": "sh4", "src": "/static/filter_images/Others/sh4.png"},
        {"key": "sh5", "src": "/static/filter_images/Others/sh5.png"},
        {"key": "sh6", "src": "/static/filter_images/Others/sh6.png"},
        {"key": "sh7", "src": "/static/filter_images/Others/sh7.png"},
        {"key": "sh8", "src": "/static/filter_images/Others/sh8.png"},
        {"key": "dog", "src": "/static/filter_images/Others/dog.png"},
        {"key": "dog2", "src": "/static/filter_images/Others/dog2.png"},
        {"key": "dog3", "src": "/static/filter_images/Others/dog3.png"}
    ],

    "Halloween": [
        {"key": "h_mask1", "src": "/static/filter_images/halloween/h_mask1.png"},
        {"key": "h_mask2", "src": "/static/filter_images/halloween/h_mask2.png"},
        {"key": "h_mask3", "src": "/static/filter_images/halloween/h_mask3.png"},
        {"key": "h_mask4", "src": "/static/filter_images/halloween/h_mask4.png"},
        {"key": "h_mask5", "src": "/static/filter_images/halloween/h_mask5.png"},
        {"key": "h_mask6", "src": "/static/filter_images/halloween/h_mask6.png"},
        {"key": "h_mask7", "src": "/static/filter_images/halloween/h_mask7.png"},
        {"key": "h_mask8", "src": "/static/filter_images/halloween/h_mask8.png"},
        {"key": "h_mask9", "src": "/static/filter_images/halloween/h_mask9.png"},
        {"key": "h_mask10", "src": "/static/filter_images/halloween/h_mask10.png"},
        {"key": "h_mask11", "src": "/static/filter_images/halloween/h_mask11.png"},
        {"key": "h_mask12", "src": "/static/filter_images/halloween/h_mask12.png"},
        {"key": "h_mask13", "src": "/static/filter_images/halloween/h_mask13.png"},
        {"key": "h_mask14", "src": "/static/filter_images/halloween/h_mask14.png"},
        {"key": "h_mask15", "src": "/static/filter_images/halloween/h_mask15.png"},
        {"key": "h_mask16", "src": "/static/filter_images/halloween/h_mask16.png"},
    ],

    "Birthday": [
        {"key": "b_glasses1", "src": "/static/filter_images/birthday/b_glasses1.png"},
        {"key": "b_glasses2", "src": "/static/filter_images/birthday/b_glasses2.png"},
        {"key": "b_glasses3", "src": "/static/filter_images/birthday/b_glasses3.png"},
        {"key": "b_glasses4", "src": "/static/filter_images/birthday/b_glasses4.png"},
        {"key": "b_glasses5", "src": "/static/filter_images/birthday/b_glasses5.png"},
        {"key": "b_glasses6", "src": "/static/filter_images/birthday/b_glasses6.png"},
        {"key": "b_glasses7", "src": "/static/filter_images/birthday/b_glasses7.png"},
        {"key": "b_glasses8", "src": "/static/filter_images/birthday/b_glasses8.png"},
        {"key": "b_glasses9", "src": "/static/filter_images/birthday/b_glasses9.png"},
        {"key": "b_glasses10", "src": "/static/filter_images/birthday/b_glasses10.png"},
        {"key": "b_glasses11", "src": "/static/filter_images/birthday/b_glasses11.png"},
        {"key": "b_glasses12", "src": "/static/filter_images/birthday/b_glasses12.png"},
        {"key": "b_glasses13", "src": "/static/filter_images/birthday/b_glasses13.png"},
        {"key": "b_glasses14", "src": "/static/filter_images/birthday/b_glasses14.png"},
        {"key": "b_glasses15", "src": "/static/filter_images/birthday/b_glasses15.png"},
        {"key": "b_glasses16", "src": "/static/filter_images/birthday/b_glasses16.png"},
        {"key": "b_glasses17", "src": "/static/filter_images/birthday/b_glasses17.png"},  
    ]

        }

print("[OK] Local filters loaded successfully.")

def get_filter_image(filter_key):
    """Return the local file path for a given filter key."""
    for category, filters in FILTER_IMAGES.items():
        for f in filters:
            if f["key"] == filter_key:
                return f["src"]
    return None