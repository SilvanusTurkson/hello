import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
import platform

# ===== CONFIGURATION =====
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet"  # Using Facenet for better stability
DETECTOR_BACKEND = "opencv"  # More reliable than retinaface
THRESHOLD = 0.6
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480

# ===== WEB CAMERA INITIALIZATION =====
def initialize_webcam():
    """Try multiple camera indices and different backends to find a working camera"""
    # Try different backends based on OS
    backends = [
        cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_V4L2,
        cv2.CAP_ANY
    ]
    
    # Try multiple camera indices (0-4)
    for backend in backends:
        for index in range(5):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Set resolution to improve performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_FRAME_HEIGHT)
                return cap
            cap.release()
    return None

# ===== FACE PROCESSING FUNCTIONS =====
def save_face_embedding(face_img, name):
    """Save face embeddings with enhanced error handling"""
    try:
        if not name or not isinstance(name, str):
            raise ValueError("Invalid name format")
            
        # Convert to RGB if needed
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
            face_img = face_img[:, :, :3]
            
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )[0]["embedding"]

        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)
        else:
            df = pd.DataFrame(columns=["name", "embedding"])

        df = pd.concat([df, pd.DataFrame([{
            "name": name.strip(),
            "embedding": str(embedding)
        }])], ignore_index=True)
        
        df.to_csv(KNOWN_FACES_DB, index=False)
        st.success(f"✅ {name}'s face saved successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error saving face: {str(e)}")
        return False

def recognize_face(face_img):
    """Recognize face with improved error handling"""
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database available", 0.0

        # Convert to RGB if needed
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
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
        
        similarities = []
        for _, row in df.iterrows():
            db_embedding = np.array(row["embedding"])
            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
            )
            similarities.append(similarity)

        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        if confidence > THRESHOLD:
            return df.iloc[best_match_idx]["name"], confidence
        return "Unknown", confidence
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def get_facial_area(face):
    """Safely extract facial coordinates"""
    area = face["facial_area"]
    return area["x"], area["y"], area["w"], area["h"]

# ===== STREAMLIT UI =====
st.set_page_config(
    page_title="AI Face Recognition System",
    page_icon="🤖",
    layout="centered"
)

st.title("🎭 Face Recognition System")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode", ["Webcam", "Upload Image"])
    register_mode = st.checkbox("Register New Face")
    if register_mode:
        new_face_name = st.text_input("Enter Name", max_chars=50)

# Main processing
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        try:
            image = np.array(Image.open(uploaded_file))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            faces = DeepFace.extract_faces(
                image,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True
            )
            
            for face in faces:
                x, y, w, h = get_facial_area(face)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if register_mode and new_face_name:
                    save_face_embedding(face["face"], new_face_name)
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(
                        image, f"{name} ({confidence:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                   use_column_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

else:  # Webcam mode
    FRAME_WINDOW = st.empty()
    cam = initialize_webcam()
    
    if not cam:
        st.error("""
        ❌ No webcam found. Try these solutions:
        1. Ensure your webcam is connected
        2. Grant camera permissions to the app
        3. Try a different USB port if using external camera
        4. Restart your computer
        """)
        st.stop()

    stop_button = st.button("Stop Webcam")
    
    while not stop_button:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Error capturing frame from webcam")
            break
        
        frame = cv2.cvtColor(frame, cv2.CAP_PROP_CONVERT_RGB)
        
        try:
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            
            for face in faces:
                if face["confidence"] > 0.85:  # Only high-confidence detections
                    x, y, w, h = get_facial_area(face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if register_mode and new_face_name:
                        save_face_embedding(face["face"], new_face_name)
                    else:
                        name, confidence = recognize_face(face["face"])
                        cv2.putText(
                            frame, f"{name} ({confidence:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )
        except Exception as e:
            st.warning(f"Face detection error: {str(e)}")
            continue
        
        FRAME_WINDOW.image(frame)

    cam.release()
    st.warning("📸 Webcam stopped!")

# Database info
with st.expander("Database Information"):
    if os.path.exists(KNOWN_FACES_DB):
        df = pd.read_csv(KNOWN_FACES_DB)
        st.write(f"Registered faces: {len(df)}")
        st.dataframe(df)
    else:
        st.warning("No database found")
