import cv2
import av
import os
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------------------
# Model files
# ----------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL):
    st.error("❌ Please put both 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' in this folder.")
    st.stop()

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Class labels MobileNetSSD was trained on
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

PERSON_CLASS_ID = 15   # 'person' is ID 15
CONF_THRESH = 0.3

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📷 Live Object Detection by Raj")
st.write("📱 Open your **camera** below. It will detect and count objects + people in real-time.")

camera_size = st.radio(
    "📏 Select Camera Size:",
    ["Medium", "Large", "Full"],
    index=0,
    horizontal=True
)

if camera_size == "Medium":
    cam_style = {"width": "60%", "height": "auto"}
elif camera_size == "Large":
    cam_style = {"width": "80%", "height": "auto"}
else:
    cam_style = {"width": "100%", "height": "auto"}

# ----------------------------
# Video Transformer
# ----------------------------
class ObjectDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

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
                if class_id >= len(CLASSES):
                    continue
                label = CLASSES[class_id]

                # Bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")

                # Clip to frame size
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(w - 1, ex), min(h - 1, ey)

                # Draw rectangle in green
                cv2.rectangle(img, (sx, sy), (ex, ey), (0, 255, 0), 2)

                # Label at bottom of rectangle
                text_y = ey + 20 if ey + 20 < h else ey - 10
                if class_id == PERSON_CLASS_ID:
                    people_count += 1
                    cv2.putText(img, "Person", (sx, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(img, label, (sx, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                total_objects += 1

        # People text (singular/plural)
        if people_count == 1:
            people_text = "1 Person"
        else:
            people_text = f"{people_count} Persons"

        # Show counts at top-left
        cv2.rectangle(img, (5, 5), (340, 70), (255, 255, 255), -1)
        cv2.putText(img, f"Objects: {total_objects} | {people_text}",
                    (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# Start Streamlit WebRTC
# ----------------------------
webrtc_streamer(
    key="object-detect",
    video_transformer_factory=ObjectDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs={"style": cam_style},
)
