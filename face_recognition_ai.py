import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
from time import sleep

# ===== CONFIG =====
KNOWN_FACES_DB = "faces_db.csv"
MODEL = "Facenet512"
DETECTOR = "retinaface"
THRESHOLD = 0.6

# ===== STREAMLIT UI =====
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("🎭 Real-Time Face Recognition")

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode:", ("Webcam", "Upload Image"))
    if input_mode == "Webcam":
        detect_every_n = st.slider("Process every N frames:", 1, 10, 5)
    register_mode = st.checkbox("Register New Face")
    if register_mode:
        new_face_name = st.text_input("Enter Name:")

# ===== FUNCTIONS =====
def find_working_camera():
    """Find a working camera index."""
    for i in range(5):  # Try 5 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

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

# ===== MAIN PROCESSING =====
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            faces = DeepFace.extract_faces(image, detector_backend=DETECTOR, enforce_detection=False)
            for i, face in enumerate(faces):
                x, y, w, h = face["facial_area"].values()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if register_mode and new_face_name:
                    save_face_embedding(face["face"], new_face_name)
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(image, f"{name} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            st.image(image, caption="Processed Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:  # Webcam mode
    st.info("Starting Webcam...")

    cam_index = find_working_camera()
    if cam_index is None:
        st.error("Error: Cannot access any webcam. Check permissions or physical connection.")
        st.stop()

    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    FRAME_WINDOW = st.empty()
    stop_button = st.button("Stop Webcam")
    frame_count = 0

    while not stop_button:
        ret, frame = cam.read()
        if not ret:
            st.error("Error: Failed to capture frame. Check webcam or permissions.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

        if frame_count % detect_every_n == 0:
            try:
                faces = DeepFace.extract_faces(frame, detector_backend=DETECTOR, enforce_detection=False)
                for face in faces:
                    if face["confidence"] > 0.9:
                        x, y, w, h = face["facial_area"].values()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        if register_mode and new_face_name:
                            save_face_embedding(face["face"], new_face_name)
                        else:
                            name, confidence = recognize_face(face["face"])
                            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                st.error(f"Detection error: {str(e)}")

        FRAME_WINDOW.image(frame)
        sleep(0.01)

    cam.release()
    st.warning("Webcam stopped.")
