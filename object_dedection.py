import cv2
import av
import os
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------------------
# Model files (manual download)
# ----------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL):
    st.error("❌ Please put both 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' in this folder.")
    st.stop()

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# COCO class labels for MobileNetSSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

PERSON_CLASS_ID = 15
CONF_THRESH = 0.4

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📷 Live Object & People Detection")
st.write("📱 Open your **camera** below. It will detect and count objects + people in real-time.")

# ----------------------------
# Video Transformer
# ----------------------------
class ObjectDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        people_count = 0
        total_objects = 0

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > CONF_THRESH:
                class_id = int(detections[0, 0, i, 1])
                label = CLASSES[class_id]

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                cv2.rectangle(img, (sx, sy), (ex, ey), (0, 255, 0), 2)
                cv2.putText(img, label, (sx, sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                total_objects += 1
                if class_id == PERSON_CLASS_ID:
                    people_count += 1

        # Show counts
        cv2.putText(
            img,
            f"Objects: {total_objects} | People: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# Open Camera (Bigger Screen)
# ----------------------------
webrtc_streamer(
    key="object-detect",
    video_transformer_factory=ObjectDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs={
        "style": {"width": "90%", "height": "auto"},  # bigger preview
    },
)
