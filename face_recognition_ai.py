import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace

def initialize_webcam():
    """Initialize webcam by attempting multiple backends and indexes."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for i in range(5):  # Try multiple camera indexes (0-4)
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
    return None

def load_known_faces(directory):
    known_faces = {}
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    face_embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')[0]['embedding']
                    known_faces[person_name] = face_embedding
                except Exception as e:
                    st.error(f"Error processing image {image_path}: {e}")
    return known_faces

def recognize_face(frame, known_faces):
    try:
        face_embedding = DeepFace.represent(frame, model_name='Facenet')[0]['embedding']
        for person_name, known_embedding in known_faces.items():
            distance = np.linalg.norm(np.array(face_embedding) - np.array(known_embedding))
            if distance < 0.6:  # Lower threshold for better accuracy
                return person_name
    except Exception as e:
        st.error(f"Error recognizing face: {e}")
    return "Unknown"

def main():
    st.title("Real-Time Face Recognition System")

    known_faces_dir = "known_faces"
    st.text("Loading known faces...")

    if not os.path.exists(known_faces_dir):
        st.error("Known faces directory not found.")
        return

    known_faces = load_known_faces(known_faces_dir)
    st.success("Known faces loaded successfully!")

    st.text("Initializing webcam...")
    cap = initialize_webcam()

    if not cap:
        st.error("Failed to access webcam. Check permissions and hardware.")
        return

    st.success("Webcam initialized. Starting recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        person_name = recognize_face(frame, known_faces)

        cv2.putText(frame, person_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        st.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
