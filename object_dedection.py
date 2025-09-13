import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---- Load MobileNet-SSD Model ----
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# Download model files from:
# https://github.com/chuanqi305/MobileNet-SSD
# Place both files in the same folder as this app.py

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
PERSON_CLASS_ID = 15
CONF_THRESH = 0.4

st.title("👥 Real-time People Detection (Phone Camera)")
st.markdown(
    "Open this app on your **phone browser**, allow camera access, "
    "and it will detect & count people live."
)

# ---- Video Transformer ----
class PersonDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        # Prepare input for MobileNet-SSD
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

# ---- Start WebRTC Stream ----
webrtc_streamer(
    key="people-detect",
    mode="recvonly",  # recvonly = browser sends camera → server receives
    video_transformer_factory=PersonDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
