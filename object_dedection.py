import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os, gdown

# ----------------------------
# Auto-download model files if missing
# ----------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"

if not os.path.exists(PROTOTXT):
    gdown.download(PROTOTXT_URL, PROTOTXT, quiet=False)

if not os.path.exists(MODEL):
    gdown.download(MODEL_URL, MODEL, quiet=False)

# ----------------------------
# Load model
# ----------------------------
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
PERSON_CLASS_ID = 15
CONF_THRESH = 0.4

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("👥 Real-time People Detection (Phone Camera)")
st.markdown(
    """
    📱 **How to use**  
    1. Run this app (`streamlit run app.py`).  
    2. Open it in your **phone browser** using your PC's IP (e.g. `http://192.168.x.x:8501`).  
    3. Allow camera access → People will be detected live.  
    """
)

# ----------------------------
# Video Transformer
# ----------------------------
class PersonDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > CONF_THRESH:
                class_id = int(detections[0, 0, i, 1])
                if class_id == PERSON_CLASS_ID:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (sx, sy, ex, ey) = box.astype("int")
                    cv2.rectangle(img, (sx, sy), (ex, ey), (0, 255, 0), 2)
                    count += 1

        cv2.putText(img, f"People: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# WebRTC Stream
# ----------------------------
webrtc_streamer(
    key="people-detect",
    mode="recvonly",
    video_transformer_factory=PersonDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
