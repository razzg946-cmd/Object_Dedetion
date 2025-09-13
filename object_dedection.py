import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# ----------------------------
# Model files (manual)
# ----------------------------
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# Check if files exist
if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL):
    st.error("❌ Please make sure MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel are in the app folder.")
    st.stop()

# ----------------------------
# Load model
# ----------------------------
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

PERSON_CLASS_ID = 15
CONF_THRESH = 0.4

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("👥 Real-time People Detection (Phone Camera)")
st.markdown(
    """
    📱 **How to use**  
    1. Make sure model files are in the same folder as this app.  
    2. Run the app on your PC:  
       ```
       streamlit run app.py
       ```
    3. Open the URL on your phone browser using your PC's IP, e.g.  
       ```
       http://192.168.x.x:8501
       ```
    4. Allow camera access → live people detection & count will show automatically ✅
    """
)

# ----------------------------
# Video Transformer
# ----------------------------
class PersonDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        # Prepare input blob
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

        # Draw people count
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
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
