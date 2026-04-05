import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mask_detector.pth")
FACE_DETECTOR_PATH = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
DATABASE_PATH = os.path.join(BASE_DIR, "detection_logs.db")

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5
