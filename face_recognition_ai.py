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
        # Convert string representation of embedding back to list
        df['embedding'] = df['embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df
    return pd.DataFrame(columns=["name", "embedding", "image"])

def save_database(df):
    """Save the face database"""
    df.to_csv(KNOWN_FACES_DB, index=False)

def add_face_to_db(name, embedding, image):
    """Add a new face to the database"""
    df = load_database()
    
    # Convert image to base64 string for storage
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
    for index in [0, 1, 2]:  # Try common camera indices
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

# ===== FACE PROCESSING =====
def extract_face_embedding(face_img):
    """Extract face embedding with error handling"""
    try:
        # Convert to proper color format
        if len(face_img.shape) == 2:  # Grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # RGBA
            face_img = face_img[:, :, :3]
        
        # Get face embedding
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

# ===== MAIN APP =====
tab1, tab2, tab3 = st.tabs(["Camera", "Upload Image", "Manage Database"])

with tab1:  # Camera tab
    st.header("Live Camera Recognition")
    cam = initialize_webcam()
    
    if not cam:
        st.error("""
        ❌ Webcam not detected. Please:
        1. Check camera privacy settings
        2. Ensure no other apps are using the camera
        3. Test with Windows Camera app first
        """)
    else:
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Camera", key="stop_cam")
        
        while cam.isOpened() and not stop_button:
            ret, frame = cam.read()
            if not ret:
                st.error("Error capturing frame")
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
                        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        name, confidence = recognize_face(face["face"], db_df)
                        cv2.putText(
                            frame, f"{name} ({confidence:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )
            except Exception as e:
                st.warning(f"Face processing error: {str(e)}")
            
            frame_placeholder.image(frame, use_container_width=True)
            
            if stop_button:
                cam.release()
                st.success("Camera session ended")

with tab2:  # Upload Image tab
    st.header("Image Upload Recognition")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            faces = DeepFace.extract_faces(
                image,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            
            for face in faces:
                x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                name, confidence = recognize_face(face["face"], db_df)
                cv2.putText(
                    image, f"{name} ({confidence:.2f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab3:  # Database Management tab
    st.header("Face Database Management")
    
    # Add new face section
    with st.expander("➕ Register New Face"):
        col1, col2 = st.columns(2)
        
        with col1:
            register_name = st.text_input("Enter Name", key="reg_name")
            register_source = st.radio("Image Source", ["Webcam", "Upload"])
            
            if register_source == "Webcam":
                reg_cam = initialize_webcam()
                if reg_cam:
                    reg_frame_placeholder = st.empty()
                    capture_button = st.button("Capture Image")
                    
                    ret, reg_frame = reg_cam.read()
                    if ret:
                        reg_frame = cv2.cvtColor(reg_frame, cv2.COLOR_BGR2RGB)
                        reg_frame_placeholder.image(reg_frame, use_container_width=True)
                        
                        if capture_button:
                            try:
                                faces = DeepFace.extract_faces(
                                    reg_frame,
                                    detector_backend=DETECTOR_BACKEND,
                                    enforce_detection=True,
                                    align=True
                                )
                                if faces:
                                    embedding = extract_face_embedding(faces[0]["face"])
                                    if embedding is not None:
                                        add_face_to_db(register_name, embedding, reg_frame)
                                        st.success(f"Successfully registered {register_name}!")
                                        db_df = load_database()  # Refresh database
                                else:
                                    st.warning("No face detected in the captured image")
                            except Exception as e:
                                st.error(f"Error registering face: {str(e)}")
                            finally:
                                reg_cam.release()
                else:
                    st.warning("Could not initialize webcam for registration")
            
            else:  # Upload
                reg_uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
                if reg_uploaded_file and register_name:
                    reg_image = np.array(Image.open(reg_uploaded_file))
                    reg_image = cv2.cvtColor(reg_image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        faces = DeepFace.extract_faces(
                            reg_image,
                            detector_backend=DETECTOR_BACKEND,
                            enforce_detection=True,
                            align=True
                        )
                        if faces:
                            embedding = extract_face_embedding(faces[0]["face"])
                            if embedding is not None:
                                add_face_to_db(register_name, embedding, reg_image)
                                st.success(f"Successfully registered {register_name}!")
                                db_df = load_database()  # Refresh database
                        else:
                            st.warning("No face detected in the uploaded image")
                    except Exception as e:
                        st.error(f"Error registering face: {str(e)}")
    
    # Delete face section
    with st.expander("🗑️ Delete Registered Face"):
        if not db_df.empty:
            delete_name = st.selectbox("Select name to delete", db_df["name"].unique())
            if st.button("Delete Face"):
                db_df = delete_face_from_db(delete_name)
                st.success(f"Deleted {delete_name} from database")
        else:
            st.info("No faces in database to delete")
    
    # View database section
    with st.expander("👀 View Database"):
        if not db_df.empty:
            st.write(f"Total registered faces: {len(db_df)}")
            
            # Display each face in the database
            for _, row in db_df.iterrows():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Decode and display the stored image
                    try:
                        img_bytes = base64.b64decode(row["image"])
                        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img, caption=row["name"], width=150)
                    except:
                        st.warning("Could not load image")
                
                with col2:
                    st.write(f"**Name:** {row['name']}")
                    st.write(f"**Embedding:** {len(row['embedding'])} dimensions")
        else:
            st.info("No faces registered yet")
