import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
from PIL import Image

# ===== WINDOWS-SPECIFIC FIXES =====
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Helps with Windows camera access
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1"  # Prefer DirectShow backend

# ===== CONFIGURATION =====
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet"  # Best balance of speed and accuracy
DETECTOR_BACKEND = "opencv"  # Most reliable on Windows
THRESHOLD = 0.6  # Similarity threshold
FRAME_WIDTH = 640  # Optimal resolution
FRAME_HEIGHT = 480

# ===== WEB CAMERA INITIALIZATION =====
def initialize_webcam():
    """Initialize webcam with Windows-specific settings"""
    # Try different camera indices
    for index in [0, 1, 2]:  # Try common camera indices
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # Configure camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            
            # Verify the camera is actually working
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    
    return None

# ===== FACE PROCESSING FUNCTIONS =====
def save_face_embedding(face_img, name):
    """Save face embeddings with enhanced error handling"""
    try:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Please enter a valid name")
            
        # Ensure proper image format and color space
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
            face_img = face_img[:, :, :3]
        
        # First verify we can detect a face
        try:
            # Try with detection first for quality control
            DeepFace.extract_faces(
                face_img,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True
            )
        except:
            # If detection fails, try without enforcement but warn user
            st.warning("⚠️ Face detection uncertain - registering anyway")
            
        # Get face embedding
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,  # Changed to False to prevent errors
            align=True,
            normalization="base"
        )[0]["embedding"]

        # Update database
        df = pd.DataFrame(columns=["name", "embedding"])
        if os.path.exists(KNOWN_FACES_DB):
            df = pd.read_csv(KNOWN_FACES_DB)
            
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
    """Recognize face with comprehensive error handling"""
    try:
        if not os.path.exists(KNOWN_FACES_DB):
            return "No database available", 0.0

        # Convert to proper color format
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
            face_img = face_img[:, :, :3]

        # Get face embedding
        query_embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="base"
        )[0]["embedding"]

        # Compare with database
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
st.set_page_config(
    page_title="Windows Face Recognition",
    page_icon="🖥️",
    layout="centered"
)

st.title("🖥️ Windows Face Recognition System")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Mode", ["Webcam", "Upload Image"])
    register_mode = st.checkbox("Register New Face")
    if register_mode:
        new_face_name = st.text_input("Enter Name", max_chars=50, 
                                    help="Enter the name of the person to register")

# Main processing
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        try:
            # Read and preprocess image
            image = np.array(Image.open(uploaded_file))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces with Windows-optimized settings
            faces = DeepFace.extract_faces(
                image,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,  # Changed to False to prevent errors
                align=True
            )
            
            # Process each face
            for face in faces:
                x, y, w, h = get_facial_area(face)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if register_mode and new_face_name:
                    if save_face_embedding(face["face"], new_face_name):
                        cv2.putText(
                            image, f"Registered: {new_face_name}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )
                else:
                    name, confidence = recognize_face(face["face"])
                    cv2.putText(
                        image, f"{name} ({confidence:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
            
            # Display processed image (using use_container_width instead of use_column_width)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                   use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

else:  # Webcam mode
    FRAME_WINDOW = st.empty()
    cam = initialize_webcam()
    
    if not cam:
        st.error("""
        ❌ Webcam not detected on Windows. Please:
        1. Check Windows Camera Privacy Settings
        2. Ensure no other apps are using the camera
        3. Try updating your webcam drivers
        4. Test your camera with Windows Camera app first
        5. Restart your computer and try again
        """)
        st.stop()

    stop_button = st.button("⏹️ Stop Webcam")
    
    while not stop_button:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Error capturing frame from webcam")
            break
        
        # Convert and process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces with confidence threshold
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            
            # Process each detected face
            for face in faces:
                if face["confidence"] > 0.85:  # Only high-confidence detections
                    x, y, w, h = get_facial_area(face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if register_mode and new_face_name:
                        if save_face_embedding(face["face"], new_face_name):
                            cv2.putText(
                                frame, f"Registered: {new_face_name}",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2
                            )
                    else:
                        name, confidence = recognize_face(face["face"])
                        cv2.putText(
                            frame, f"{name} ({confidence:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )
        except Exception as e:
            st.warning(f"⚠️ Face processing error: {str(e)}")
            continue
        
        # Display the frame
        FRAME_WINDOW.image(frame, use_container_width=True)

    # Clean up
    cam.release()
    st.success("🎥 Webcam session ended")

# Database information
with st.expander("📁 Database Information"):
    if os.path.exists(KNOWN_FACES_DB):
        df = pd.read_csv(KNOWN_FACES_DB)
        st.write(f"Registered faces: {len(df)}")
        st.dataframe(df)
    else:
        st.info("No faces registered yet")
