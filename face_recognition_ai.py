import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
import platform
from PIL import Image

# ===== CONFIGURATION =====
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet"  # Balanced accuracy and performance
DETECTOR_BACKEND = "opencv"  # Reliable for most devices
THRESHOLD = 0.6  # Similarity threshold
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ===== WEB CAMERA INITIALIZATION =====
def initialize_webcam():
    """Initialize the default laptop camera with Windows-specific handling"""
    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_V4L2

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return cap

    return None

# ===== FACE PROCESSING FUNCTIONS =====
def save_face_embedding(face_img, name):
    """Save face embeddings with enhanced validation"""
    try:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Please enter a valid name")

        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = face_img[:, :, :3]

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )[0]["embedding"]

        df = pd.DataFrame(columns=["name", "embedding"])
        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)

        df = pd.concat([df, pd.DataFrame([{ "name": name.strip(), "embedding": str(embedding) }])], ignore_index=True)

        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"✅ {name}'s face saved successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Error saving face: {str(e)}")
        return False

def recognize_face(face_img):
    """Recognize face with comprehensive error handling"""
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database available", 0.0

        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = face_img[:, :, :3]

        query_embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )[0]["embedding"]

        df = pd.read_csv(KNOWN_FACES_DB)
        df["embedding"] = df["embedding"].apply(eval)

        best_match = ("Unknown", 0.0)
        for _, row in df.iterrows():
            db_embedding = np.array(row["embedding"])
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            if similarity > best_match[1]:
                best_match = (row["name"], similarity)

        return best_match if best_match[1] > THRESHOLD else ("Unknown", best_match[1])

    except Exception as e:
        return f"Error: {str(e)}", 0.0

def get_facial_area(face):
    """Safely extract facial coordinates from detection result"""
    area = face["facial_area"]
    return area["x"], area["y"], area["w"], area["h"]

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Laptop Face Recognition", page_icon="💻", layout="centered")

st.title("💻 Laptop Face Recognition System")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode", ["Webcam", "Upload Image"])
    register_mode = st.checkbox("Register New Face")
    if register_mode:
        new_face_name = st.text_input("Enter Name", max_chars=50, help="Enter the name of the person to register")

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_file:
        try:
            image = np.array(Image.open(uploaded_file))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            faces = DeepFace.extract_faces(image, detector_backend=DETECTOR_BACKEND, enforce_detection=True, align=True)

            for face in faces:
                x, y, w, h = get_facial_area(face)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if register_mode and new_face_name:
                    if save_face_embedding(face["face"], new_face_name):
                        cv2.putText(image, f"Registered: {new_face_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(image, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

else:
    FRAME_WINDOW = st.empty()
    cam = initialize_webcam()

    if not cam:
        st.error("❌ Webcam not detected. Ensure it's enabled and accessible.")
        st.stop()

    stop_button = st.button("⏹️ Stop Webcam")

    while not stop_button:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Error capturing frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)

    cam.release()
    st.success("🎥 Webcam session ended")
