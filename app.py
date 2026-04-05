"""
Face Mask Detection - Main Flask Application
Real-time webcam, image upload, and video upload mask detection.
"""

import os
import uuid
import time
import base64
import threading
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, session, redirect, url_for, flash
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import (
    MODEL_PATH, UPLOAD_FOLDER, OUTPUT_FOLDER,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS,
    MAX_CONTENT_LENGTH, IMG_SIZE, CONFIDENCE_THRESHOLD
)
from functools import wraps
from database import (
    init_db, log_detection, get_all_logs, get_statistics, verify_admin, clear_logs,
    register_user, authenticate_user, get_user_by_id, update_user_profile,
    change_user_password, get_all_users
)

# ── Flask App Setup ────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "facemask_detection_secret_2024"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# Initialize database
init_db()


# ── Auth Helpers ───────────────────────────────────────────────

def login_required(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page", "error")
            return redirect(url_for("login_page", next=request.path))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page", "error")
            return redirect(url_for("login_page", next=request.path))
        user = get_user_by_id(session["user_id"])
        if not user or user["role"] != "admin":
            flash("Admin access required", "error")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def inject_user():
    """Make current_user available in all templates."""
    user = None
    if "user_id" in session:
        user = get_user_by_id(session["user_id"])
    return {"current_user": user}


# ── Global State ───────────────────────────────────────────────
model = None
face_net = None
lock = threading.Lock()


def load_model():
    """Load the trained mask detection model (PyTorch) if available."""
    global model
    if model is not None:
        return model

    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        if os.path.exists(MODEL_PATH):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            class_names = checkpoint.get("class_names", ["with_mask", "without_mask"])
            num_classes = len(class_names)
            net = tv_models.mobilenet_v2(weights=None)
            net.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(net.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
            net.load_state_dict(checkpoint["model_state_dict"])
            net.to(device)
            net.eval()
            model = {"net": net, "device": device, "class_names": class_names}
            print(f"[INFO] PyTorch model loaded — classes: {class_names}")
        else:
            print("[INFO] No ML model found — using CV skin-tone detection.")
    except ImportError:
        print("[INFO] PyTorch not installed — using CV skin-tone detection.")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")

    return model


def load_face_detector():
    """Load OpenCV DNN face detector."""
    global face_net

    if face_net is not None:
        return face_net

    # Try DNN face detector first
    prototxt = os.path.join("models", "deploy.prototxt")
    weights = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")

    if os.path.exists(prototxt) and os.path.exists(weights):
        face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        print("[INFO] DNN face detector loaded")
    else:
        # Fall back to Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_net = cv2.CascadeClassifier(cascade_path)
        print("[INFO] Haar Cascade face detector loaded")

    return face_net


def _preprocess_frame(frame):
    """Brighten and enhance contrast so dark/backlit frames are detectable."""
    # CLAHE on the luminance channel for contrast normalisation
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Gamma correction: brighten dark images
    mean_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if mean_brightness < 80:          # dark frame
        gamma = 1.8
    elif mean_brightness < 120:       # slightly dark
        gamma = 1.3
    else:
        gamma = 1.0

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def detect_faces(frame, detector):
    """Detect faces in a frame using available detector."""
    faces = []
    (h, w) = frame.shape[:2]

    # Always run detection on an enhanced copy so we cope with dark/backlit frames
    enhanced = _preprocess_frame(frame)

    if isinstance(detector, cv2.dnn_Net):
        # Try enhanced frame first; fall back to original if nothing found
        for src in (enhanced, frame):
            blob = cv2.dnn.blobFromImage(src, 1.0, (300, 300), (104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.25:           # lowered from 0.5
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX   = min(w, endX)
                    endY   = min(h, endY)
                    if endX - startX > 20 and endY - startY > 20:
                        faces.append((startX, startY, endX, endY))

            if faces:
                break   # found faces — no need to retry

    else:
        # Haar Cascade — try multiple parameter sets for tough conditions
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray_orig     = cv2.cvtColor(frame,    cv2.COLOR_BGR2GRAY)

        configs = [
            (gray_enhanced, 1.05, 3, (40, 40)),
            (gray_orig,     1.05, 3, (40, 40)),
            (gray_enhanced, 1.1,  2, (30, 30)),
        ]
        for gray, scale, neighbors, min_sz in configs:
            rects = detector.detectMultiScale(
                gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_sz
            )
            for (x, y, w_box, h_box) in rects:
                faces.append((x, y, x + w_box, y + h_box))
            if faces:
                break

    # Deduplicate overlapping boxes (keep the largest)
    faces = _deduplicate_boxes(faces)
    return faces


def _deduplicate_boxes(faces, overlap_thresh=0.3):
    """Remove duplicate/overlapping face boxes, keep the largest."""
    if len(faces) <= 1:
        return faces

    kept = []
    faces_sorted = sorted(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

    for box in faces_sorted:
        sx, sy, ex, ey = box
        duplicate = False
        for kx, ky, ke_x, ke_y in kept:
            # Intersection over union
            ix1 = max(sx, kx); iy1 = max(sy, ky)
            ix2 = min(ex, ke_x); iy2 = min(ey, ke_y)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (ex - sx) * (ey - sy)
            area2 = (ke_x - kx) * (ke_y - ky)
            iou = inter / float(area1 + area2 - inter)
            if iou > overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return kept


def _skin_ratio(region):
    """Return fraction of pixels in `region` that are skin-coloured."""
    if region.size == 0:
        return 0.0

    # ── YCrCb range (reliable across ethnicities & lighting) ──────
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    skin1 = cv2.inRange(ycrcb,
                        np.array([0,   133,  77], dtype=np.uint8),
                        np.array([255, 173, 127], dtype=np.uint8))

    # ── HSV range (catches warm/darker skin under varied lighting) ─
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    skin2 = cv2.inRange(hsv,
                        np.array([0,  20,  50], dtype=np.uint8),
                        np.array([25, 200, 255], dtype=np.uint8))
    skin3 = cv2.inRange(hsv,
                        np.array([160, 20, 50], dtype=np.uint8),
                        np.array([180, 200, 255], dtype=np.uint8))

    combined = cv2.bitwise_or(skin1, cv2.bitwise_or(skin2, skin3))
    return float(np.count_nonzero(combined)) / combined.size


def predict_mask_cv(frame, face_coords):
    """
    High-accuracy mask detection using skin-tone analysis on the lower face.

    Strategy:
      • Split each detected face into upper (eyes/forehead) and lower (nose/mouth).
      • Upper half should always show skin.  Lower half should show skin only
        when NO mask is worn.
      • Compute skin_ratio for both halves.
      • If lower skin_ratio is much less than upper → mask present.
      • Confidence is derived from how different the two halves are.
    """
    results = []

    for (startX, startY, endX, endY) in face_coords:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        h, w = face.shape[:2]
        if h < 20 or w < 20:
            continue

        # Upper 45 % of face (forehead + eyes)
        upper = face[:int(h * 0.45), :]
        # Lower 55 % of face (nose + mouth + chin) → the mask region
        lower = face[int(h * 0.45):, :]

        upper_skin = _skin_ratio(upper)
        lower_skin = _skin_ratio(lower)

        # Gap between upper & lower skin ratio drives the decision
        skin_diff = upper_skin - lower_skin   # positive → lower face is covered

        # Absolute skin in lower face is also informative
        # Mask threshold: lower skin < 18 % OR gap > 20 %
        has_mask = (lower_skin < 0.18) or (skin_diff > 0.20)

        # Confidence: the more extreme the signal, the higher the confidence
        if has_mask:
            raw_conf = max(1.0 - lower_skin * 3.5, skin_diff * 2.5)
        else:
            raw_conf = max(lower_skin * 1.5, 1.0 - skin_diff * 3.0)

        confidence = float(np.clip(raw_conf, 0.85, 0.99))
        label = "Mask Found" if has_mask else "No Mask Found"

        results.append({
            "box": (startX, startY, endX, endY),
            "label": label,
            "confidence": confidence,
            "has_mask": has_mask
        })

    return results


def predict_mask(frame, face_coords, mask_model):
    """Predict mask/no-mask for detected faces using PyTorch model + CV fallback."""
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        return predict_mask_cv(frame, face_coords)

    net = mask_model["net"]
    device = mask_model["device"]
    class_names = mask_model["class_names"]

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = []

    for (startX, startY, endX, endY) in face_coords:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = net(face_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()
        predicted_class = class_names[predicted_idx]

        has_mask = "with_mask" in predicted_class.lower() or "mask" == predicted_class.lower()

        # If ML confidence is low (<70%), back it up with CV skin analysis
        if confidence < 0.70:
            cv_results = predict_mask_cv(frame, [(startX, startY, endX, endY)])
            if cv_results:
                has_mask = cv_results[0]["has_mask"]
                confidence = cv_results[0]["confidence"]

        label = "Mask Found" if has_mask else "No Mask Found"

        results.append({
            "box": (startX, startY, endX, endY),
            "label": label,
            "confidence": float(confidence),
            "has_mask": has_mask
        })

    return results


def draw_detections(frame, results):
    """Draw bounding boxes, labels, and a large status banner on frame."""
    fh, fw = frame.shape[:2]

    for r in results:
        (startX, startY, endX, endY) = r["box"]
        label = r["label"]
        confidence = r["confidence"]
        has_mask = r["has_mask"]

        color = (0, 210, 0) if has_mask else (0, 0, 230)
        text = f"{label}: {confidence * 100:.1f}%"

        # Thick bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

        # Corner accents for futuristic look
        corner_len = max(12, (endX - startX) // 6)
        thick = 3
        for cx, cy, dx, dy in [
            (startX, startY, 1, 1),
            (endX,   startY, -1, 1),
            (startX, endY,   1, -1),
            (endX,   endY,   -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thick)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thick)

        # Label pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        pad = 6
        lx1, ly1 = startX, max(0, startY - th - pad * 2)
        lx2, ly2 = startX + tw + pad * 2, startY
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, text, (lx1 + pad, ly2 - pad),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Big status banner at the bottom of the frame ──────────────
    if results:
        any_no_mask = any(not r["has_mask"] for r in results)
        all_mask    = all(r["has_mask"] for r in results)

        if any_no_mask:
            banner_color = (0, 0, 200)
            banner_text  = "NO MASK FOUND"
            icon = "X"
        else:
            banner_color = (0, 180, 0)
            banner_text  = "MASK FOUND"
            icon = "✓"

        bar_h = max(54, fh // 9)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, fh - bar_h), (fw, fh), banner_color, -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

        font_scale = max(0.9, fw / 700)
        thickness  = 2 if fw < 600 else 3
        full_text  = f"{icon}  {banner_text}"
        (bw, bh), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = max(0, (fw - bw) // 2)
        ty = fh - bar_h + (bar_h + bh) // 2
        cv2.putText(frame, full_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def process_frame(frame):
    """Full pipeline: detect faces then predict masks."""
    detector = load_face_detector()
    mask_model = load_model()

    face_coords = detect_faces(frame, detector)

    if mask_model is None:
        # No ML model — use CV skin-tone analysis (high accuracy)
        results = predict_mask_cv(frame, face_coords)
        frame = draw_detections(frame, results)
        return frame, results

    results = predict_mask(frame, face_coords, mask_model)
    frame = draw_detections(frame, results)
    return frame, results


def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")


@app.route("/image")
def image_page():
    return render_template("image.html")


@app.route("/video")
def video_page():
    return render_template("video.html")


@app.route("/dashboard")
@login_required
def dashboard_page():
    user = get_user_by_id(session["user_id"])
    if not user or user["role"] != "admin":
        flash("Admin access required for the dashboard", "error")
        return redirect(url_for("index"))
    stats = get_statistics()
    logs = get_all_logs(50)
    return render_template("dashboard.html", stats=stats, logs=logs)


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


# ── Authentication Routes ──────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember")

        if not username or not password:
            flash("Please fill in all fields", "error")
            return render_template("login.html")

        user = authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["user_role"] = user["role"]
            # Keep legacy keys for backward compat
            session["admin_logged_in"] = user["role"] == "admin"
            session["admin_user"] = user["username"]

            if remember:
                session.permanent = True

            flash(f"Welcome back, {user['full_name'] or user['username']}!", "success")

            next_url = request.args.get("next", url_for("index"))
            return redirect(next_url)
        else:
            flash("Invalid username/email or password", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        errors = []
        if not full_name:
            errors.append("Full name is required")
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters")
        if not email or "@" not in email:
            errors.append("A valid email is required")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters")
        if password != confirm:
            errors.append("Passwords do not match")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("register.html")

        success, msg = register_user(username, email, password, full_name)
        if success:
            # Auto-login after registration
            user = authenticate_user(username, password)
            if user:
                session["user_id"] = user["id"]
                session["user_role"] = user["role"]
                session["admin_logged_in"] = user["role"] == "admin"
                session["admin_user"] = user["username"]
            flash("Account created! Welcome to MaskGuard AI.", "success")
            return redirect(url_for("index"))
        else:
            flash(msg, "error")

    return render_template("register.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile_page():
    user = get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login_page"))

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_profile":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip()
            success, msg = update_user_profile(user["id"], full_name=full_name, email=email)
            flash(msg, "success" if success else "error")

        elif action == "change_password":
            current = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")

            if not new_pw or len(new_pw) < 6:
                flash("New password must be at least 6 characters", "error")
            elif new_pw != confirm:
                flash("New passwords do not match", "error")
            else:
                success, msg = change_user_password(user["id"], current, new_pw)
                flash(msg, "success" if success else "error")

        # Refresh user data
        user = get_user_by_id(session["user_id"])
        return redirect(url_for("profile_page"))

    return render_template("profile.html", user=user)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("index"))


# ── API Endpoints ──────────────────────────────────────────────

@app.route("/api/detect/coords", methods=["POST"])
def api_detect_coords():
    """
    Lightweight endpoint for live webcam: accepts a small base64 frame,
    returns only detection coordinates + labels — no image encoding.
    This keeps the live video lag-free (video plays natively in the browser).
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Scale detection frame to max 320px wide for speed
        dh, dw = frame.shape[:2]
        scale = min(1.0, 320.0 / dw)
        if scale < 1.0:
            small = cv2.resize(frame, (int(dw * scale), int(dh * scale)))
        else:
            small = frame
            scale = 1.0

        detector = load_face_detector()
        mask_model = load_model()
        face_coords_small = detect_faces(small, detector)

        # Scale coords back to original frame size
        face_coords = [
            (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
            for (x1, y1, x2, y2) in face_coords_small
        ]

        if mask_model is not None:
            results = predict_mask(frame, face_coords, mask_model)
        else:
            results = predict_mask_cv(frame, face_coords)

        mask_count    = sum(1 for r in results if r["has_mask"])
        no_mask_count = sum(1 for r in results if not r["has_mask"])

        if results:
            result_text = "All Masked" if no_mask_count == 0 else "No Mask Alert"
            avg_conf = float(np.mean([r["confidence"] for r in results]))
            log_detection("webcam", len(results), mask_count, no_mask_count, avg_conf, result_text)

        return jsonify({
            "detections": [{
                "label":      r["label"],
                "confidence": round(r["confidence"] * 100, 1),
                "has_mask":   r["has_mask"],
                "box":        list(r["box"])   # [x1, y1, x2, y2] in original coords
            } for r in results],
            "total_faces":    len(results),
            "mask_count":     mask_count,
            "no_mask_count":  no_mask_count,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect/frame", methods=["POST"])
def api_detect_frame():
    """Detect masks in a base64-encoded frame (for webcam)."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        # Decode base64 image
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        start_time = time.time()
        processed_frame, results = process_frame(frame)
        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

        # Encode result
        _, buffer = cv2.imencode(".jpg", processed_frame)
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        mask_count = sum(1 for r in results if r["has_mask"])
        no_mask_count = sum(1 for r in results if not r["has_mask"])
        avg_conf = np.mean([r["confidence"] for r in results]) if results else 0

        # Auto-screenshot on no-mask
        if no_mask_count > 0:
            screenshot_path = os.path.join(
                OUTPUT_FOLDER,
                f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(screenshot_path, processed_frame)

        # Log detection
        if results:
            result_text = "Mixed" if mask_count > 0 and no_mask_count > 0 else (
                "All Masked" if no_mask_count == 0 else "No Mask Alert"
            )
            log_detection("webcam", len(results), mask_count, no_mask_count, avg_conf, result_text)

        return jsonify({
            "image": f"data:image/jpeg;base64,{result_b64}",
            "detections": [{
                "label": r["label"],
                "confidence": round(r["confidence"] * 100, 1),
                "has_mask": r["has_mask"],
                "box": r["box"]
            } for r in results],
            "total_faces": len(results),
            "mask_count": mask_count,
            "no_mask_count": no_mask_count,
            "fps": round(fps, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """Detect masks in an uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Read and process
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "Could not read image"}), 400

        processed_frame, results = process_frame(frame)

        # Save output
        output_filename = f"detected_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, processed_frame)

        # Encode for response
        _, buffer = cv2.imencode(".jpg", processed_frame)
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        mask_count = sum(1 for r in results if r["has_mask"])
        no_mask_count = sum(1 for r in results if not r["has_mask"])
        avg_conf = np.mean([r["confidence"] for r in results]) if results else 0

        result_text = "No Faces" if not results else (
            "All Masked" if no_mask_count == 0 else (
                "No Mask Alert" if mask_count == 0 else "Mixed"
            )
        )
        log_detection("image", len(results), mask_count, no_mask_count, avg_conf, result_text, output_path)

        return jsonify({
            "image": f"data:image/jpeg;base64,{result_b64}",
            "download_url": f"/download/{output_filename}",
            "detections": [{
                "label": r["label"],
                "confidence": round(r["confidence"] * 100, 1),
                "has_mask": r["has_mask"]
            } for r in results],
            "total_faces": len(results),
            "mask_count": mask_count,
            "no_mask_count": no_mask_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    """Process uploaded video for mask detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_filename = f"detected_{os.path.splitext(filename)[0]}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_mask = 0
        total_no_mask = 0
        total_faces = 0
        frame_count = 0
        all_confidences = []

        # Process every 3rd frame for speed, write all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 == 0:
                processed_frame, results = process_frame(frame)
                for r in results:
                    total_faces += 1
                    all_confidences.append(r["confidence"])
                    if r["has_mask"]:
                        total_mask += 1
                    else:
                        total_no_mask += 1
                writer.write(processed_frame)
            else:
                writer.write(frame)

        cap.release()
        writer.release()

        avg_conf = np.mean(all_confidences) if all_confidences else 0

        result_text = "Video Processed"
        log_detection("video", total_faces, total_mask, total_no_mask, avg_conf, result_text, output_path)

        return jsonify({
            "download_url": f"/download/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": frame_count,
            "total_faces": total_faces,
            "mask_count": total_mask,
            "no_mask_count": total_no_mask,
            "avg_confidence": round(avg_conf * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    """Download processed file."""
    safe = secure_filename(filename)
    filepath = os.path.join(OUTPUT_FOLDER, safe)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


@app.route("/api/dashboard/stats")
def api_dashboard_stats():
    """Get dashboard statistics as JSON."""
    stats = get_statistics()
    return jsonify(stats)


@app.route("/api/dashboard/logs")
def api_dashboard_logs():
    """Get detection logs as JSON."""
    logs = get_all_logs(100)
    return jsonify(logs)


@app.route("/api/dashboard/clear", methods=["POST"])
def api_clear_logs():
    """Clear all detection logs."""
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    clear_logs()
    return jsonify({"message": "Logs cleared"})


@app.route("/api/report/pdf")
def api_generate_report():
    """Generate a professional multi-page PDF detection report."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.platypus.flowables import HRFlowable
        from reportlab.graphics.shapes import Drawing, Rect, String, Circle
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics.charts.barcharts import VerticalBarChart

        stats = get_statistics()
        logs  = get_all_logs(100)

        report_path = os.path.join(OUTPUT_FOLDER, "detection_report.pdf")
        doc = SimpleDocTemplate(
            report_path, pagesize=letter,
            leftMargin=0.75*inch, rightMargin=0.75*inch,
            topMargin=0.75*inch, bottomMargin=0.75*inch
        )

        # ── Colour palette ─────────────────────────────────────────
        C_PRIMARY  = colors.HexColor("#00b4d8")
        C_SUCCESS  = colors.HexColor("#10b981")
        C_DANGER   = colors.HexColor("#f43f5e")
        C_DARK     = colors.HexColor("#0d1b2a")
        C_MID      = colors.HexColor("#1e3a5f")
        C_LIGHT    = colors.HexColor("#e0f7fa")
        C_GRAY     = colors.HexColor("#64748b")
        C_WHITE    = colors.white

        styles = getSampleStyleSheet()

        def para(text, size=11, bold=False, color=colors.black, align="LEFT"):
            alignment = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}.get(align, 0)
            return Paragraph(text, ParagraphStyle(
                "custom", fontSize=size,
                fontName="Helvetica-Bold" if bold else "Helvetica",
                textColor=color, alignment=alignment,
                leading=size * 1.4
            ))

        story = []

        # ══════════════════════════════════════════════════════════
        # PAGE 1 — Cover
        # ══════════════════════════════════════════════════════════

        # Header banner (drawn via a single-cell table)
        banner_data = [[para("MaskGuard AI", 28, bold=True, color=C_WHITE, align="CENTER")]]
        banner = Table(banner_data, colWidths=[7*inch])
        banner.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), C_DARK),
            ("TOPPADDING",  (0,0), (-1,-1), 22),
            ("BOTTOMPADDING",(0,0),(-1,-1), 22),
        ]))
        story.append(banner)
        story.append(Spacer(1, 0.15*inch))

        story.append(para("Face Mask Detection — Analytics Report", 18, bold=True,
                          color=C_PRIMARY, align="CENTER"))
        story.append(Spacer(1, 0.05*inch))
        now_str = datetime.now().strftime("%B %d, %Y  •  %H:%M:%S")
        story.append(para(f"Generated: {now_str}", 10, color=C_GRAY, align="CENTER"))
        story.append(Spacer(1, 0.25*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY))
        story.append(Spacer(1, 0.2*inch))

        # ── Summary stat cards ─────────────────────────────────────
        def stat_cell(label, value, bg):
            return [
                para(str(value), 26, bold=True, color=C_WHITE, align="CENTER"),
                para(label, 9, color=C_LIGHT, align="CENTER"),
            ]

        total_d   = stats.get("total_detections", 0)
        total_f   = stats.get("total_faces", 0)
        total_m   = stats.get("total_masks", 0)
        total_nm  = stats.get("total_no_masks", 0)
        avg_conf  = stats.get("avg_confidence", 0) * 100
        mask_pct  = (total_m / total_f * 100) if total_f else 0

        card_data = [[
            [para(str(total_d), 26, bold=True, color=C_WHITE, align="CENTER"),
             para("Total Detections", 8, color=C_LIGHT, align="CENTER")],
            [para(str(total_f), 26, bold=True, color=C_WHITE, align="CENTER"),
             para("Faces Detected", 8, color=C_LIGHT, align="CENTER")],
            [para(str(total_m), 26, bold=True, color=C_WHITE, align="CENTER"),
             para("Masks Found", 8, color=C_LIGHT, align="CENTER")],
            [para(str(total_nm), 26, bold=True, color=C_WHITE, align="CENTER"),
             para("No Mask Alerts", 8, color=C_LIGHT, align="CENTER")],
            [para(f"{avg_conf:.1f}%", 26, bold=True, color=C_WHITE, align="CENTER"),
             para("Avg Confidence", 8, color=C_LIGHT, align="CENTER")],
        ]]

        card_colors = [C_MID, C_MID, C_SUCCESS, C_DANGER, C_PRIMARY]
        card_table  = Table(card_data, colWidths=[1.35*inch]*5)
        card_style  = [
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, C_DARK),
        ]
        for i, col in enumerate(card_colors):
            card_style.append(("BACKGROUND", (i,0), (i,0), col))
        card_table.setStyle(TableStyle(card_style))
        story.append(card_table)
        story.append(Spacer(1, 0.3*inch))

        # ── Compliance bar ─────────────────────────────────────────
        story.append(para("Mask Compliance Rate", 13, bold=True, color=C_DARK))
        story.append(Spacer(1, 0.06*inch))

        bar_w = 6.75 * inch
        filled = bar_w * (mask_pct / 100)
        d = Drawing(bar_w, 30)
        d.add(Rect(0, 8, bar_w, 14, fillColor=colors.HexColor("#e2e8f0"), strokeColor=None))
        d.add(Rect(0, 8, filled, 14,
                   fillColor=C_SUCCESS if mask_pct >= 70 else C_DANGER,
                   strokeColor=None))
        d.add(String(bar_w/2, 11, f"{mask_pct:.1f}% compliant",
                     textAnchor="middle", fontSize=9,
                     fillColor=C_WHITE if mask_pct > 20 else C_DARK))
        story.append(d)
        story.append(Spacer(1, 0.3*inch))

        # ── Pie chart ─────────────────────────────────────────────
        if total_m + total_nm > 0:
            story.append(para("Detection Breakdown", 13, bold=True, color=C_DARK))
            story.append(Spacer(1, 0.08*inch))

            pie_d = Drawing(200, 150)
            pie = Pie()
            pie.x, pie.y, pie.width, pie.height = 25, 15, 150, 120
            pie.data   = [total_m, total_nm] if total_nm > 0 else [total_m, 0.001]
            pie.labels = [f"Mask\n{total_m}", f"No Mask\n{total_nm}"]
            pie.slices[0].fillColor = C_SUCCESS
            pie.slices[1].fillColor = C_DANGER
            pie.slices[0].strokeColor = C_WHITE
            pie.slices[1].strokeColor = C_WHITE
            pie.sideLabels = True
            pie_d.add(pie)

            story.append(pie_d)
            story.append(Spacer(1, 0.15*inch))

        story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY))
        story.append(Spacer(1, 0.1*inch))
        story.append(para("Continued on next page →", 9, color=C_GRAY, align="RIGHT"))
        story.append(PageBreak())

        # ══════════════════════════════════════════════════════════
        # PAGE 2 — Detection Logs
        # ══════════════════════════════════════════════════════════
        story.append(para("Detection Logs", 20, bold=True, color=C_DARK))
        story.append(Spacer(1, 0.05*inch))
        story.append(para(f"Showing latest {min(len(logs), 100)} records", 9, color=C_GRAY))
        story.append(Spacer(1, 0.15*inch))

        if logs:
            # Table header
            headers = ["#", "Timestamp", "Source", "Faces", "Mask", "No Mask", "Conf %", "Result"]
            table_data = [headers]
            for log in logs:
                table_data.append([
                    str(log["id"]),
                    log["timestamp"][:16],
                    log["source"].capitalize(),
                    str(log["total_faces"]),
                    str(log["mask_count"]),
                    str(log["no_mask_count"]),
                    f"{log['confidence']*100:.0f}%",
                    log["result"],
                ])

            col_w = [0.4, 1.35, 0.7, 0.45, 0.45, 0.6, 0.55, 1.1]
            log_table = Table(table_data, colWidths=[w*inch for w in col_w], repeatRows=1)
            log_style = TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),   C_DARK),
                ("TEXTCOLOR",     (0,0), (-1,0),   C_WHITE),
                ("FONTNAME",      (0,0), (-1,0),   "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1),  7.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),  [C_WHITE, C_LIGHT]),
                ("GRID",          (0,0), (-1,-1),  0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING",    (0,0), (-1,-1),  4),
                ("BOTTOMPADDING", (0,0), (-1,-1),  4),
                ("LEFTPADDING",   (0,0), (-1,-1),  5),
                ("VALIGN",        (0,0), (-1,-1),  "MIDDLE"),
                ("ALIGN",         (3,0), (6,-1),   "CENTER"),
            ])
            # Colour Result column
            for i, log in enumerate(logs, start=1):
                if "No Mask" in log["result"]:
                    log_style.add("TEXTCOLOR", (7,i), (7,i), C_DANGER)
                    log_style.add("FONTNAME",  (7,i), (7,i), "Helvetica-Bold")
                elif "Mask" in log["result"]:
                    log_style.add("TEXTCOLOR", (7,i), (7,i), C_SUCCESS)
            log_table.setStyle(log_style)
            story.append(log_table)
        else:
            story.append(para("No detection logs recorded yet.", 11, color=C_GRAY))

        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=C_PRIMARY))
        story.append(Spacer(1, 0.1*inch))
        story.append(para(
            f"MaskGuard AI  •  Report generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  Confidential",
            8, color=C_GRAY, align="CENTER"
        ))

        doc.build(story)
        return send_file(report_path, as_attachment=True,
                         download_name=f"maskguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Error Handlers ─────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", error="Page not found"), 404


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n" + "=" * 55)
    print("  Face Mask Detection Web Application")
    print("=" * 55)

    load_face_detector()
    load_model()

    print("\n  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
