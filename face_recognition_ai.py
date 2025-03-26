import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
from PIL import Image
import base64

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

# ===== DATABASE MANAGEMENT =====
def load_database():
    """Load or initialize the face database"""
    if os.path.exists(KNOWN_FACES_DB):
        df = pd.read_csv(KNOWN_FACES_DB)
        df['embedding'] = df['embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df
    return pd.DataFrame(columns=["name", "embedding", "image"])

def save_database(df):
    """Save the face database"""
    df.to_csv(KNOWN_FACES_DB, index=False)

def add_face_to_db(name, embedding, image):
    """Add a new face to the database"""
    df = load_database()
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    new_entry = pd.DataFrame([{
        "name": name.strip(),
        "embedding": str(embedding),
        "image": img_str
    }])

    df = pd.concat([df, new_entry], ignore_index=True)
    save_database(df)
    return df

def delete_face_from_db(name):
    """Delete a face from the database"""
    df = load_database()
    df = df[df["name"] != name]
    save_database(df)
    return df

# ===== CAMERA FUNCTIONS =====
def initialize_webcam():
    """Initialize webcam with Windows-specific settings"""
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return None

def initialize_video(video_path):
    """Initialize video for face recognition"""
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            return cap
    return None

# ===== FACE PROCESSING =====
def extract_face_embedding(face_img):
    """Extract face embedding with error handling"""
    try:
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = face_img[:, :, :3]

        return DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="base"
        )[0]["embedding"]
    except Exception as e:
        st.error(f"Error extracting face features: {str(e)}")
        return None

def recognize_face(face_img, df):
    """Recognize face from database"""
    try:
        if df.empty:
            return "No database available", 0.0

        query_embedding = extract_face_embedding(face_img)
        if query_embedding is None:
            return "Error extracting features", 0.0

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

# ===== STREAMLIT UI =====
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🖥️",
    layout="centered"
)

st.title("🖥️ Face Recognition System")

# Initialize database
db_df = load_database()

tab1, tab2, tab3, tab4 = st.tabs(["Camera", "Upload Image", "Manage Database", "Video"])

with tab4:  # Video tab
    st.header("Video Face Recognition")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        temp_video_path = f"temp_{video_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        video_cap = initialize_video(temp_video_path)

        if not video_cap:
            st.error("Error loading video")
        else:
            frame_placeholder = st.empty()
            stop_button = st.button("Stop Video", key="stop_vid")

            while video_cap.isOpened() and not stop_button:
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    faces = DeepFace.extract_faces(
                        frame,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=False,
                        align=True
                    )

                    for face in faces:
                        if face["confidence"] > 0.85:
                            x, y, w, h = face["facial_area"].values()
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            name, confidence = recognize_face(face["face"], db_df)
                            cv2.putText(
                                frame, f"{name} ({confidence:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2
                            )
                except Exception as e:
                    st.warning(f"Face processing error: {str(e)}")

                frame_placeholder.image(frame, use_container_width=True)

            video_cap.release()
            os.remove(temp_video_path)
