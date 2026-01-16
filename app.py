import os
import sys
import time
import base64
import threading
from io import BytesIO




import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file
from flask_mail import Mail, Message
import mysql.connector
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import requests
import webview
import qrcode
import winsound
import win32print
import win32ui
from PIL import ImageWin




# --- Import Local Filters ---
from filters.christmas_glasses_filter import apply_christmas_glasses_filter
from filters.hearts_day_filter import apply_heart_glasses_filter
from filters.filters_loader import FILTER_IMAGES, get_filter_image # ✅ NEW IMPORT
from filters.mustache_filter import  apply_mustache_filter
from filters.cat_filter import apply_cat_filter
from filters.sh_filter import apply_sh_filter
from filters.birthday_glasses_filter import apply_birthday_glasses_filter
from filters.halloween_mask_filter import apply_halloween_mask_filter
from filters.dog_filter import apply_dog_filter


BOOTH_TIMER = {
    "start": None,
    "active": False
}

BOOTH_DURATION = 10 * 60  

# --------------------- FLASK APP -------------------------------
app = Flask(__name__)

# --------------------- GLOBAL STATE ---------------------------
FACE_HISTORY = {}  
bg_index = None
current_filter_index = None
ai_background_np = None
camera_active = False
cap = None
camera_lock = threading.Lock()

# --------------------- CAMERA + MEDIAPIPE ---------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
# --------------------- FILTER IMAGES ---------------------------
def load_filter_image(path):
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f" Local file not found: {path}")
    return None

filter_images = {}
for category, filters in FILTER_IMAGES.items():
    for f in filters:
        img = load_filter_image(f["src"].lstrip("/")) 
        if img is not None:
            filter_images[f["key"]] = img


 
filters_list = [
    {"key": "c_glasses1", "func": apply_christmas_glasses_filter},
    {"key": "c_glasses2", "func": apply_christmas_glasses_filter},
    {"key": "c_glasses3", "func": apply_christmas_glasses_filter},
    {"key": "c_glasses4", "func": apply_christmas_glasses_filter},
    {"key": "c_glasses5", "func": apply_christmas_glasses_filter},

    {"key": "v_glasses1", "func": apply_heart_glasses_filter},
    {"key": "v_glasses2", "func": apply_heart_glasses_filter},
    {"key": "v_glasses3", "func": apply_heart_glasses_filter},
    {"key": "v_glasses4", "func": apply_heart_glasses_filter},
    {"key": "v_glasses5", "func": apply_heart_glasses_filter},
    {"key": "v_glasses6", "func": apply_heart_glasses_filter},
    {"key": "v_glasses7", "func": apply_heart_glasses_filter},

    {"key": "mustache1", "func": apply_mustache_filter},
    {"key": "mustache2", "func": apply_mustache_filter},
    {"key": "mustache3", "func": apply_mustache_filter},
    {"key": "mustache4", "func": apply_mustache_filter},
    {"key": "mustache5", "func": apply_mustache_filter},


    {"key": "cat1", "func": apply_cat_filter},
    {"key": "cat2", "func": apply_cat_filter},
    {"key": "cat3", "func": apply_cat_filter},


    {"key": "sh1", "func": apply_sh_filter},
    {"key": "sh2", "func": apply_sh_filter},
    {"key": "sh3", "func": apply_sh_filter},
    {"key": "sh4", "func": apply_sh_filter},
    {"key": "sh5", "func": apply_sh_filter},
    {"key": "sh6", "func": apply_sh_filter},
    {"key": "sh7", "func": apply_sh_filter},
    {"key": "sh8", "func": apply_sh_filter},

    {"key": "dog", "func": apply_dog_filter},
    {"key": "dog2", "func": apply_dog_filter},
    {"key": "dog3", "func": apply_dog_filter},

     {"key": "b_glasses1", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses2", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses3", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses4", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses5", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses6", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses7", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses8", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses9", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses10", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses11", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses12", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses13", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses14", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses15", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses16", "func": apply_birthday_glasses_filter},
    {"key": "b_glasses17", "func": apply_birthday_glasses_filter},

    {"key": "h_mask1", "func": apply_halloween_mask_filter},
    {"key": "h_mask2", "func": apply_halloween_mask_filter},
    {"key": "h_mask3", "func": apply_halloween_mask_filter},
    {"key": "h_mask4", "func": apply_halloween_mask_filter},
    {"key": "h_mask5", "func": apply_halloween_mask_filter},
    {"key": "h_mask6", "func": apply_halloween_mask_filter},
    {"key": "h_mask7", "func": apply_halloween_mask_filter},
    {"key": "h_mask8", "func": apply_halloween_mask_filter},
    {"key": "h_mask9", "func": apply_halloween_mask_filter},
    {"key": "h_mask10", "func": apply_halloween_mask_filter},
    {"key": "h_mask11", "func": apply_halloween_mask_filter},
    {"key": "h_mask12", "func": apply_halloween_mask_filter},
    {"key": "h_mask13", "func": apply_halloween_mask_filter},
    {"key": "h_mask14", "func": apply_halloween_mask_filter},
    {"key": "h_mask15", "func": apply_halloween_mask_filter},
    {"key": "h_mask16", "func": apply_halloween_mask_filter},


]



print(f"[OK] Loaded {len(filter_images)} local filters across categories.")


# --------------------- BACKGROUND COLORS ----------------------
background_colors = [
    ("Remove", None),
    ("Light Pink", (203, 192, 255)),
    ("Light Blue", (255, 230, 200)),
    ("Light Violet", (238, 130, 238)),
    ("Mint Green", (189, 252, 201)),
    ("Pastel Lavender", (230, 216, 255)),    
]

# --------------------- RVM SETUP ------------------------------
RVM_PATH = r"C:\xampp\htdocs\SnapbitsTesting4.1\RobustVideoMatting"
sys.path.append(RVM_PATH)
from model import MattingNetwork




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MattingNetwork().eval().to(device)
checkpoint = torch.load(os.path.join(RVM_PATH, "rvm_mobilenetv3.pth"), map_location=device)
model.load_state_dict(checkpoint)
to_tensor = ToTensor()
rec = [None] * 4




# --------------------- CAMERA UTILS ---------------------------
def find_working_camera():
    for i in range(5):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            cap_test.release()
            return i
    return 0




def start_camera():
    global cap, camera_active
    with camera_lock:
        if not camera_active:
            index = find_working_camera()
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_active = True
            print(f"Camera started at index {index}")




def stop_camera():
    global cap, camera_active
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
        cap = None
        camera_active = False
        print("Camera stopped.")


# --- SESSION TIMER GLOBALS (ADDED) ---
SESSION_LIMIT = 12  
WARNING_TIME = 6

session_timer = None
warning_timer = None
session_active = False

main_window = None  

# -------------------------------------


# --------------------- Utility: Auto-detect USB Camera ----------------
def find_working_camera():
    for i in range(5):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            cap_test.release()
            print(f" Using camera index: {i}")
            return i
    print(" No working camera found.")
    return 0
# --------------------- Helper: Start Camera ----------------
def start_camera():
    global cap, camera_active
    with camera_lock:
        if not camera_active:
            index = find_working_camera()
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(" Failed to open camera.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_active = True
            print(" Camera started.")


# --------------------- Helper: Stop Camera ----------------
def stop_camera():
    global cap, camera_active
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
            print(" Camera released.")
        cap = None
        camera_active = False


# ----------------------------------------------------------
# Session Timer Monitor (Background Thread)
# ----------------------------------------------------------
def monitor_session():
    global warning_sound_played, session_start_time, main_window, session_redirect_done

    while True:
        if session_start_time and not session_redirect_done:
            elapsed = time.time() - session_start_time

            
            if elapsed > WARNING_TIME and not warning_sound_played:
                try:
                    winsound.Beep(1000, 1000)
                except:
                    print("Sound error")
                warning_sound_played = True

            # Session end at 10 minutes
            if elapsed > SESSION_LIMIT:
                session_redirect_done = True  
                warning_sound_played = False
                session_start_time = None

                # --- Save photo if needed ---
                try:
                    import requests
                    requests.post("http://127.0.0.1:8000/save_photo", json={"image": ""})
                except Exception as e:
                    print(f"Failed to auto-save photo: {e}")

                # --- Redirect to numberOfCopies page ---
                if main_window:
                    main_window.load_url("http://192.168.1.10:8000/numberOfCopies")

        time.sleep(1)

# --------------------- ROUTES ---------------------------
@app.route("/")
def welcome():
    global bg_index, current_filter_index, FACE_HISTORY
    bg_index = None
    current_filter_index = None
    FACE_HISTORY = {} # Reset history on app start
    BOOTH_TIMER["start"] = None
    BOOTH_TIMER["active"] = False
    return render_template("welcome.html")

@app.route("/instructions")
def instruction():
    return render_template("instructions.html")

@app.route("/template")
def template():
    return render_template("template.html")

@app.route("/camera")
def camera():
    if not BOOTH_TIMER["active"]:
        BOOTH_TIMER["start"] = time.time()
        BOOTH_TIMER["active"] = True

        start_camera()

    template = request.args.get("template", "1")
    return render_template("camera.html", template=template)



@app.route("/cameraoption")
def cameraoption():
    template = request.args.get("template")
    if not template:
        return redirect(url_for("template"))
    return render_template("cameraoption.html", template=template)


@app.route("/photoselection")
def photoselection():
    template = request.args.get("template")
    if not template:
        return redirect(url_for("template"))
    photo_count = {"1": 1, "2": 2, "3": 4}.get(template, 4)
    return render_template("photoselection.html", template=template, photo_count=photo_count)

@app.route("/design")
def design():
    return render_template("design.html")

@app.route("/numberOfCopies")
def numberOfCopies():
    BOOTH_TIMER["start"] = None
    BOOTH_TIMER["active"] = False
    return render_template("numberOfCopies.html")


@app.route("/waiting")
def waiting():
    return render_template("waiting.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/email")
def email():
    return render_template("email.html")

@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")

# --------------------------TIMER---------------------------------
@app.route("/get_remaining_time")
def get_remaining_time():
    if not BOOTH_TIMER["active"]:
        return jsonify({
            "remaining": 0,
            "total": BOOTH_DURATION
        })

    elapsed = time.time() - BOOTH_TIMER["start"]
    remaining = max(0, int(BOOTH_DURATION - elapsed))

    return jsonify({
        "remaining": remaining,
        "total": BOOTH_DURATION
    })


# --------------------- BACKGROUND ---------------------------
@app.route('/set_background/<int:index>', methods=['POST'])
def set_background(index):
    global bg_index, ai_background_np
    if 0 <= index < len(background_colors):
        bg_index = index
        if index == 0:
            ai_background_np = None
        return jsonify({"status": "success", "background": background_colors[index][0]})
    return jsonify({"error": "Invalid background"}), 400




@app.route('/set_ai_background', methods=['POST'])
def set_ai_background():
    data = request.get_json()
    image_url = data.get('image_url')
    if image_url:
        def download():
            global ai_background_np, bg_index
            try:
                resp = requests.get(image_url, timeout=10)
                if resp.status_code == 200:
                    pil_img = Image.open(BytesIO(resp.content))
                    ai_background_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    bg_index = -1
            except:
                print("Failed to download AI background")
        threading.Thread(target=download, daemon=True).start()
        return jsonify({"status": "success"})
    return jsonify({"error": "No image URL provided"}), 400




@app.route('/clear_ai_background', methods=['POST'])
def clear_ai_background():
    global ai_background_np, bg_index
    ai_background_np = None
    bg_index = None
    return jsonify({"status": "success"})


# --------------------- STATIC BACKGROUND ----------------------
@app.route('/set_static_background', methods=['POST'])
def set_static_background():
    global ai_background_np, bg_index

    data = request.get_json()
    filename = data.get("filename", "")

    img_path = os.path.join("static", "backgrounds", filename)
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            ai_background_np = img
            bg_index = -1
            return jsonify({"status": "success", "message": f"{filename} applied"})

    return jsonify({"status": "error", "message": "Background not found"}), 404

@app.route('/remove_background', methods=['POST'])
def remove_background():
    global ai_background_np, bg_index
    ai_background_np = None
    bg_index = None
    return jsonify({"status": "removed"})


# --------------------- RVM + FILTER STREAM ------------------
def refine_mask(mask, frame, radius=8, eps=1e-4):
    mask = np.clip(mask, 0, 1).astype(np.float32)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    if frame_gray.shape != mask.shape:
        frame_gray = cv2.resize(frame_gray, (mask.shape[1], mask.shape[0]))
    try:
        mask_refined = cv2.ximgproc.guidedFilter(guide=frame_gray, src=mask, radius=radius, eps=eps)
    except:
        mask_refined = cv2.GaussianBlur(mask, (11,11),0)
    return np.clip(mask_refined,0,1)




def generate_frames():
    global bg_index, ai_background_np, current_filter_index, FACE_HISTORY, rec, cap, camera_active




    if cap is None or not cap.isOpened():
        print("Camera not active")
        return




    prev_mask_3ch = None
    DOWNSAMPLE = 0.4
    frame_count = 0




    while camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_count +=1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)




        # --- RVM Background ---
        frame_small = cv2.resize(frame, (0,0), fx=DOWNSAMPLE, fy=DOWNSAMPLE)
        src = to_tensor(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        if frame_count % 2 == 0:
            with torch.no_grad():
                fgr, pha, *rec = model(src, *rec, downsample_ratio=1.0)
            pha_np = cv2.resize(pha[0,0].cpu().numpy(), (640,480))
            pha_refined = refine_mask(pha_np, frame)
            prev_mask_3ch = cv2.merge([pha_refined]*3)
        mask_3ch = prev_mask_3ch if prev_mask_3ch is not None else np.ones_like(frame, dtype=np.float32)
        fgr_up = frame.copy()




        if bg_index == -1 and ai_background_np is not None:
            bg_resized = cv2.resize(ai_background_np, (640,480))
            frame_out = (fgr_up*mask_3ch + bg_resized*(1-mask_3ch)).astype(np.uint8)
        elif bg_index and bg_index>0:
            color = np.array(background_colors[bg_index][1], dtype=np.uint8)
            bg_color = np.ones_like(fgr_up)*color
            frame_out = (fgr_up*mask_3ch + bg_color*(1-mask_3ch)).astype(np.uint8)
        else:
            frame_out = fgr_up




        # --- Apply Filters ---
        results_face = face_mesh.process(rgb)
        if current_filter_index is not None and results_face.multi_face_landmarks:
            fdata = filters_list[current_filter_index]
            filter_key = fdata["key"]
            filter_img = filter_images.get(filter_key)
            current_faces = []
            if filter_img is not None:
                for face_index, face_landmarks in enumerate(results_face.multi_face_landmarks[:4]):
                    current_faces.append(face_index)
                    try:
                        frame_out = fdata["func"](frame_out, face_landmarks, filter_img, filter_key, FACE_HISTORY, face_index)
                    except Exception as e:
                        print(f"Error applying {filter_key} on face {face_index}: {e}")
            keys_to_delete = [idx for idx in FACE_HISTORY.keys() if idx not in current_faces]
            for idx in keys_to_delete:
                del FACE_HISTORY[idx]




        _, buffer = cv2.imencode('.jpg', frame_out)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')


@app.route('/video_feed')
def video_feed():
    if not camera_active:
        start_camera()
        if not camera_active:
            return jsonify({"error": "Camera not active"}), 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




# --------------------- FILTER API ---------------------------
@app.route("/get_effects/<category>")
def get_effects(category):
    normalized = category.strip().replace(" ", "").capitalize()
    effects = FILTER_IMAGES.get(normalized)
    if not effects:
        return jsonify({"error": f"No effects found for category: {category}"}), 404
    return jsonify({"category": normalized, "effects": effects})


@app.route("/set_effect", methods=["POST"])
def set_effect():
    data = request.get_json()
    effect_key = data.get("effect_key")


    found = False
    for i, f in enumerate(filters_list):
        if f["key"] == effect_key:
            global current_filter_index, FACE_HISTORY
            current_filter_index = i
            FACE_HISTORY = {}
            found = True
            break


    if not found:
        print(f"Effect not found: {effect_key}")


    return jsonify({"status": "ok"})




# ------------------- REMOVE EFFECT -------------------
@app.route("/remove_effect", methods=["POST"])
def remove_effect():
    global current_filter_index, FACE_HISTORY
    current_filter_index = None  # no filter active
    FACE_HISTORY = {}            # reset smoothing history
    print("Effect removed")
    return jsonify({"status": "ok"})


# ---------------------------------------------------------
#  NEW: AUTOMATIC PRINTING ROUTE
# ---------------------------------------------------------

@app.route("/print-image", methods=["POST"])
def print_image():
    try:
        data = request.get_json()
        filename = data.get("filename")
        copies = int(data.get("copies", 1))

        if not filename or not os.path.exists(filename):
            return jsonify({"success": False, "error": f"File not found: {filename}"})

      
        image = Image.open(filename)
        if image.mode != "RGB":
            image = image.convert("RGB")

    
        printer_name = win32print.GetDefaultPrinter()
        if not printer_name:
            return jsonify({"success": False, "error": "No default printer found"})

        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(printer_name)

      
        printer_width = hDC.GetDeviceCaps(110)  
        printer_height = hDC.GetDeviceCaps(111) 

        padding = 20
       
        img_w, img_h = image.size
        scale_w = (printer_width - 2*padding) / img_w
        scale_h = (printer_height - 2*padding) / img_h
        scale = min(scale_w, scale_h)

        scale *= 1.01  

        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        image = image.resize((target_w, target_h), Image.LANCZOS)

        x = (printer_width - target_w) // 2
        y = (printer_height - target_h) // 2

        for _ in range(copies):
            hDC.StartDoc("SnapBits 4R Photo")
            hDC.StartPage()
            dib = ImageWin.Dib(image)
            dib.draw(hDC.GetHandleOutput(), (x, y, x + target_w, y + target_h))
            hDC.EndPage()
            hDC.EndDoc()

        hDC.DeleteDC()

        return jsonify({"success": True, "message": f"Printed {copies} copy(ies) successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------
#  Upload Background
# ---------------------------------------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload_bg")
def upload_bg():
    return '''
    <html>
    <head>
      <title>Upload Background</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          background-color: #add8e6; /* light blue */
          text-align: center;
          padding: 40px;
          margin: 0;
        }

        h1 {
          font-family: "Brush Script MT", cursive;
          font-size: 60px;
          color: black;
          margin-bottom: 30px;
        }

        form {
          background: white;
          padding: 30px;
          border-radius: 15px;
          display: inline-block;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        input[type=file] {
          margin: 15px 0;
          font-size: 18px;
          padding: 10px;
        }

        button {
          padding: 14px 28px;
          font-size: 18px;
          background-color: #007bff;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: 0.3s;
        }

        button:hover {
          background-color: #0056b3;
        }
      </style>
    </head>
    <body>
      <h1>SnapBits</h1>
      <h2>Upload a New Background</h2>
      <form action="/upload_bg_submit" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br>
        <button type="submit">Upload</button>
      </form>
    </body>
    </html>
    '''

@app.route("/upload_bg_submit", methods=["POST"])
def upload_bg_submit():
    global ai_background_np, bg_index
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    save_path = os.path.join(UPLOAD_FOLDER, "uploaded_bg.jpg")
    file.save(save_path)
    bg_image = cv2.imread(save_path)

    if bg_image is not None:
        ai_background_np = bg_image
        bg_index = -1
        return '''
        <html>
        <head>
            <title>Upload Successful</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #add8e6;
                    text-align: center;
                    padding-top: 100px;
                }

                h3 {
                    font-size: 32px;
                    color: #000;
                    margin-bottom: 30px;
                }

                .back-btn {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 50px; /* oval shape */
                    padding: 14px 40px;
                    font-size: 18px;
                    cursor: pointer;
                    transition: 0.3s;
                    text-decoration: none;
                }

                .back-btn:hover {
                    background-color: #0056b3;
                    transform: scale(1.05);
                }
            </style>
        </head>
        <body>
            <h3>✅ Upload successful! You can close this page now.</h3>
            <a href="/upload_bg" class="back-btn">Back</a>
        </body>
        </html>
        '''
    else:
        return "Failed to process image", 500




# --------------------- QR CODE ---------------------------
@app.route("/get_qr")
def get_qr():
    upload_url = "http://192.168.1.10:8000/upload_bg"
    qr = qrcode.make(upload_url)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


# --------------------- EMAIL + FEEDBACK (unchanged) -------------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = "snapbitspicture@gmail.com"
app.config['MAIL_PASSWORD'] = "tlcuzdiuzomkibat"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)


@app.route("/submit", methods=["POST"])
def submit():
    email_addr = request.form["email"]
    photo_path = os.path.join("static", "photos", "last_photo.png")
    msg = Message(subject="SnapBits Photo", sender=app.config['MAIL_USERNAME'], recipients=[email_addr])
    msg.body = "Hi! This is your photo. Thank you for trying SnapBits."
    with app.open_resource(photo_path) as fp:
        msg.attach("photo.png", "image/png", fp.read())
    try:
        mail.send(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(" Email Error:", e)
    return redirect(url_for("thankyou"))


# --------------------- FEEDBACK -------------------------------
def get_db():
    return mysql.connector.connect(host='localhost', user='root', password='', database='snapbits_db', port=3307)


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    rating = data['rating']
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback_tbl (rating) VALUES (%s)", (rating,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Rating saved: {rating} stars")
        return jsonify({"message": "Thank you for your feedback!"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to save the rating"})


# --------------------- SAVE PHOTO -------------------------
@app.route("/save_photo", methods=["POST"])
def save_photo():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    filename = "last_photo.png"
    path = os.path.join("static", "photos", filename)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_data))
    return jsonify({"filename": filename})


# --------------------- START FLASK + WEBVIEW -----------------
def start_flask():
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)

if __name__ == "__main__":

    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    time.sleep(3)

    main_window = webview.create_window(
        "SnapBits Photo Booth",
        "http://10.131.67.127:8000",
        fullscreen=True
    )

    webview.start()
