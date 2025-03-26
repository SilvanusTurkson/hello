import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
from PIL import Image

# ===== WINDOWS-SPECIFIC FIXES =====
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1"

# ===== CONFIGURATION =====
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.6
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def initialize_webcam():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return None

def save_face_embedding(face_img, name):
    try:
        if not name or not name.strip():
            raise ValueError("Invalid name")

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="base"
        )[0]["embedding"]

        df = pd.DataFrame(columns=["name", "embedding"])
        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)

        df = pd.concat([df, pd.DataFrame([{"name": name.strip(), "embedding": str(embedding)}])], ignore_index=True)
        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"✅ {name} registered successfully!")
        return True

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def recognize_face(face_img):
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database", 0.0

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        query_embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="base"
        )[0]["embedding"]

        df = pd.read_csv(KNOWN_FACES_DB)
        df["embedding"] = df["embedding"].apply(eval)

        best_match = ("Unknown", 0.0)
        for _, row in df.iterrows():
            db_embedding = np.array(row["embedding"])
            similarity = np.dot(query_embedding, db_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            if similarity > best_match[1]:
                best_match = (row["name"], similarity)

        return best_match if best_match[1] > THRESHOLD else ("Unknown", best_match[1])

    except Exception as e:
        return f"Error: {str(e)}", 0.0

def delete_face(name):
    if os.path.exists(KNOWN_FACES_DB):
        df = pd.read_csv(KNOWN_FACES_DB)
        df = df[df["name"] != name]
        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"✅ {name} deleted successfully!")
    else:
        st.error("❌ Database not found!")

st.set_page_config(page_title="Windows Face Recognition", page_icon="🖥️")
st.title("🖥️ Windows Face Recognition System")

with st.sidebar:
    st.header("Controls")
    action = st.radio("Choose Action", ["Add Face", "Delete Face", "View Faces", "Webcam Recognition", "Upload Image Recognition"])

if action == "Add Face":
    new_name = st.text_input("Enter Name")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file and new_name:
        image = np.array(Image.open(uploaded_file))
        save_face_embedding(image, new_name)

elif action == "Delete Face":
    if os.path.exists(KNOWN_FACES_DB):
        df = pd.read_csv(KNOWN_FACES_DB)
        name_to_delete = st.selectbox("Select Face to Delete", df["name"].unique())
        if st.button("Delete"): delete_face(name_to_delete)

elif action == "View Faces":
    if os.path.exists(KNOWN_FACES_DB):
        st.dataframe(pd.read_csv(KNOWN_FACES_DB))
    else:
        st.info("No faces registered.")

elif action == "Webcam Recognition":
    cap = initialize_webcam()
    FRAME_WINDOW = st.empty()
    stop_button = st.button("Stop Webcam")

    while not stop_button:
        ret, frame = cap.read()
        if not ret: break

        faces = DeepFace.extract_faces(frame, detector_backend=DETECTOR_BACKEND, enforce_detection=False, align=True)
        for face in faces:
            x, y, w, h = face["facial_area"].values()
            name, confidence = recognize_face(face["face"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

elif action == "Upload Image Recognition":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        name, confidence = recognize_face(image)
        st.image(image, caption=f"Detected: {name} ({confidence:.2f})")
