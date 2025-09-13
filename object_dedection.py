import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# ----------------------------
# Load Model
# ----------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL):
    st.error("❌ Please put MobileNetSSD_deploy.prototxt & .caffemodel in the app folder.")
    st.stop()

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

CONF_THRESH = 0.4

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📷 Live Object Detection")
st.write("👉 Click below to start your **phone camera** and detect objects in real-time.")

# ----------------------------
# Video Transformer
# ----------------------------
class ObjectDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRESH:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")

                cv2.rectangle(img, (sx, sy), (ex, ey), (0, 255, 0), 2)
                cv2.putText(img, label, (sx, sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                count += 1

        cv2.putText(img, f"Objects: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# Camera Stream
# ----------------------------
if st.button("📷 Open Camera"):
    webrtc_streamer(
        key="object-detect",
        video_transformer_factory=ObjectDetector,
        media_stream_constraints={"video": True, "audio": False},
    )
