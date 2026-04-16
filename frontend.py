"""
Streamlit frontend for your Sign Language Detection project.

How to run:
1. Create a virtualenv and install requirements:
   python -m venv .venv
   .\.venv\Scripts\activate   (Windows)
   pip install -r requirements.txt

2. Run:
   streamlit run frontend.py

Notes:
- This uses streamlit-webrtc to access the local webcam and process frames in real-time.
- Edit MODEL_PATH and LABELS_PATH below to point to your keras_model.h5 and labels.txt.

Requirements (example):
streamlit
streamlit-webrtc
opencv-python
cvzone
numpy
av

"""

import streamlit as st
import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import math
import os

st.set_page_config(page_title="Sign Language Detection", layout="wide")

# ===== User settings =====
MODEL_PATH_DEFAULT = r"C:\Users\sd407\Downloads\converted_keras (1)\keras_model.h5"
LABELS_PATH_DEFAULT = r"C:\Users\sd407\Downloads\converted_keras (1)\labels.txt"

st.sidebar.title("Settings")
model_path = st.sidebar.text_input("Path to keras model (.h5)", MODEL_PATH_DEFAULT)
labels_path = st.sidebar.text_input("Path to labels.txt", LABELS_PATH_DEFAULT)
imgSize = st.sidebar.number_input("Square image size", value=300, min_value=64, max_value=1024)
offset = st.sidebar.number_input("Crop offset (px)", value=20, min_value=0, max_value=200)

st.sidebar.write("---")
st.sidebar.markdown("If your model or labels are not found, edit the paths above.")

# ===== Load labels safely =====
if os.path.exists(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        labels = None
else:
    labels = None

if labels is None:
    # Provide a sensible fallback
    labels = ["Drink", "Eat", "Hello", "I love you", "No", "Please", "Sorry", "Thank you", "Yes"]

# ===== Load model lazily in transformer =====

class SignVideoTransformer(VideoTransformerBase):
    def _init_(self):
        self.detector = HandDetector(maxHands=1)
        # Classifier will be loaded on first frame to avoid long startup in Streamlit
        self.classifier = None
        self.model_loaded = False
        self.last_frame = None
        self.last_label = ""

    def load_model(self):
        try:
            self.classifier = Classifier(model_path, labels_path)
            self.model_loaded = True
            print("Model loaded")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.classifier = None
            self.model_loaded = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        imgOutput = img.copy()

        if not self.model_loaded:
            self.load_model()

        hands, _img = self.detector.findHands(img, draw=False)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Clamp coords
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

            if y2 > y1 and x2 > x1:
                imgCrop = img[y1:y2, x1:x2]
                if imgCrop.size != 0:
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    aspectRatio = h / w if w != 0 else 1

                    try:
                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wGap + wCal] = imgResize
                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hGap + hCal, :] = imgResize

                        # Prediction
                        if self.classifier is not None:
                            prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                            label = labels[index] if 0 <= index < len(labels) else str(index)
                        else:
                            label = "(model not loaded)"

                        # Save last frame + label for snapshot
                        self.last_frame = imgWhite.copy()
                        self.last_label = label

                        # Draw on output frame
                        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x + 250, y - offset - 10), (0, 255, 0), cv2.FILLED)
                        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 3)

                        # Small preview of processed square
                        small = cv2.resize(imgWhite, (120, 120))
                        h0, w0 = small.shape[:2]
                        imgOutput[0:h0, 0:w0] = small

                    except Exception as e:
                        print(f"Processing error: {e}")

        # Return modified frame
        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")


# ===== Streamlit UI =====
st.title("➡ Sign Language Detection — Streamlit Frontend")
cols = st.columns([3, 1])

with cols[0]:
    st.markdown("### Live webcam")
    ctx = webrtc_streamer(key="sign-lang-webrtc", video_transformer_factory=SignVideoTransformer,
                         media_stream_constraints={"video": True, "audio": False},
                         async_transform=True)

with cols[1]:
    st.markdown("### Controls & Snapshot")
    if ctx.state.playing:
        st.success("🔴 Camera active")
    else:
        st.info("⚪ Camera stopped")

    if st.button("Save Snapshot"):
        # Save the last processed square image, if available
        transformer = ctx.video_transformer
        if transformer and transformer.last_frame is not None:
            timestamp = int(time.time())
            save_dir = os.path.join(os.getcwd(), "snapshots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(save_path, transformer.last_frame)
            st.success(f"Saved snapshot: {save_path}")
        else:
            st.warning("No processed frame available yet.")

    st.markdown("---")
    st.markdown("*Last detected label:*")
    if ctx.video_transformer:
        st.write(ctx.video_transformer.last_label or "(none yet)")
    else:
        st.write("(not available)")

    st.markdown("---")
    st.info("Tips: Allow camera access. If the model fails to load, check the model path in the sidebar.")

st.markdown("---")
st.markdown("Developed for local use. If you want a version that saves all detections to a CSV or streams predictions to a React frontend, I can add that.")

# End of file