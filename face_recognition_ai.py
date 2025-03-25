import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# ===== CONFIG =====
KNOWN_FACES_DB = "faces_db.csv"
MODEL = "Facenet512"
DETECTOR = "retinaface"
THRESHOLD = 0.6
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("🌭️ Real-Time Face Recognition")

with st.sidebar:
    st.header("Settings")
    register_mode = st.checkbox("Register New Face")
    if register_mode:
        new_face_name = st.text_input("Enter Name:")

# ===== FUNCTIONS =====
def save_face_embedding(face_img, name):
    try:
        embedding = DeepFace.represent(face_img, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)[0]["embedding"]
        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)
        else:
            df = pd.DataFrame(columns=["name", "embedding"])

        df = pd.concat([df, pd.DataFrame([{"name": name, "embedding": str(embedding)}])], ignore_index=True)
        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"Saved {name}'s face!")
    except Exception as e:
        st.error(f"Error saving face: {str(e)}")

def recognize_face(face_img):
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database found", 0

        query_embedding = DeepFace.represent(face_img, model_name=MODEL, enforce_detection=False)[0]["embedding"]
        df = pd.read_csv(KNOWN_FACES_DB)
        df["embedding"] = df["embedding"].apply(eval)

        similarities = []
        for _, row in df.iterrows():
            db_embedding = np.array(row["embedding"])
            cos_sim = np.dot(query_embedding, db_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            similarities.append(cos_sim)

        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]

        if confidence > THRESHOLD:
            return df.iloc[best_match_idx]["name"], confidence
        else:
            return "Unknown", confidence
    except Exception as e:
        return f"Error: {str(e)}", 0

# ===== Video Processor for Streamlit WebRTC =====
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

        for x, y, w, h in faces:
            face_img = frm[y:y + h, x:x + w]

            if register_mode and new_face_name:
                save_face_embedding(face_img, new_face_name)
            else:
                name, confidence = recognize_face(face_img)
                cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frm, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ))
