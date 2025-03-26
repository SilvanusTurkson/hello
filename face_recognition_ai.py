import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os

# Configuration Constants
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
THRESHOLD = 0.6

def initialize_webcam():
    for index in range(5):  # Try multiple webcam indices
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
        if cam.isOpened():
            return cam
    return None

# Save face embeddings to CSV
def save_face_embedding(face_img, name):
    try:
        embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND)[0]["embedding"]

        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)
        else:
            df = pd.DataFrame(columns=["name", "embedding"])

        df = pd.concat([df, pd.DataFrame([{"name": name, "embedding": str(embedding)}])], ignore_index=True)
        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"✅ {name}'s face saved successfully!")
    except Exception as e:
        st.error(f"❌ Error saving face: {str(e)}")

# Recognize face from known embeddings
def recognize_face(face_img):
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database available", 0

        query_embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        df = pd.read_csv(KNOWN_FACES_DB)
        df["embedding"] = df["embedding"].apply(eval)

        similarities = [np.dot(query_embedding, np.array(row["embedding"])) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(row["embedding"]))) for _, row in df.iterrows()]

        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]

        if confidence > THRESHOLD:
            return df.iloc[best_match_idx]["name"], confidence
        else:
            return "Unknown", confidence
    except Exception as e:
        return f"Error: {str(e)}", 0

# Streamlit App Interface
st.set_page_config(page_title="🎭 AI Face Recognition System", layout="wide")
st.title("📸 Real-Time Face Recognition")

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode", ("Webcam", "Upload Image"))
    register_mode = st.checkbox("Register New Face")
    new_face_name = st.text_input("Enter New Face Name") if register_mode else None

# Image Upload Mode
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            faces = DeepFace.extract_faces(image, detector_backend=DETECTOR_BACKEND)
            for face in faces:
                x, y, w, h = face["facial_area"].values()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if register_mode and new_face_name:
                    save_face_embedding(face["face"], new_face_name)
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(image, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                st.image(image, caption="Processed Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Webcam Mode
else:
    FRAME_WINDOW = st.empty()
    cam = initialize_webcam()

    if not cam:
        st.error("❌ No webcam found. Ensure it's connected and accessible.")
        st.stop()

    stop_button = st.button("Stop Webcam")

    while not stop_button:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Error capturing frame from webcam")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = DeepFace.extract_faces(frame, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
            for face in faces:
                x, y, w, h = face["facial_area"].values()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if register_mode and new_face_name:
                    save_face_embedding(face["face"], new_face_name)
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            st.error(f"❌ Face detection error: {str(e)}")

        FRAME_WINDOW.image(frame)

    cam.release()
    st.warning("📸 Webcam stopped!")
