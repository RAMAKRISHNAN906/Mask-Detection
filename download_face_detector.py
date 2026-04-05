"""
Download OpenCV DNN face detector model files.
Run this once before starting the app for better face detection accuracy.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            print(f"[OK] {filename} already exists")
            continue

        print(f"[DOWNLOADING] {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[OK] {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            print(f"  The app will fall back to Haar Cascade face detector.")

    print("\n[DONE] Face detector files ready.")


if __name__ == "__main__":
    download()
