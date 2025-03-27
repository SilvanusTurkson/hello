import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import os
from PIL import Image
import base64
import tempfile
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval  # Safer alternative to eval()

# ===== ENVIRONMENT SETUP =====
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# ===== CONFIGURATION =====
KNOWN_FACES_DB = "face_embeddings.csv"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.6
FRAME_WIDTH = 480  # Reduced for better performance
FRAME_HEIGHT = 360  # Reduced for better performance

# ===== INITIALIZATION =====
if 'evaluation_data' not in st.session_state:
    st.session_state.evaluation_data = {
        'true_labels': [],
        'predicted_labels': [],
        'confidence_scores': []
    }

# ===== DATABASE MANAGEMENT =====
def load_database():
    """Load or initialize the face database"""
    if os.path.exists(KNOWN_FACES_DB):
        try:
            df = pd.read_csv(KNOWN_FACES_DB)
            df['embedding'] = df['embedding'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            return pd.DataFrame(columns=["name", "embedding", "image"])
    return pd.DataFrame(columns=["name", "embedding", "image"])

def save_database(df):
    """Save the face database"""
    try:
        df.to_csv(KNOWN_FACES_DB, index=False)
    except Exception as e:
        st.error(f"Error saving database: {str(e)}")

def add_face_to_db(name, embedding, image):
    """Add a new face to the database"""
    try:
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
    except Exception as e:
        st.error(f"Error adding face to database: {str(e)}")
        return load_database()

def delete_face_from_db(name):
    """Delete a face from the database"""
    try:
        df = load_database()
        df = df[df["name"] != name]
        save_database(df)
        return df
    except Exception as e:
        st.error(f"Error deleting face from database: {str(e)}")
        return load_database()

# ===== CAMERA FUNCTIONS =====
def initialize_webcam():
    """Initialize webcam with Windows-specific settings"""
    for index in [0, 1, 2]:
        try:
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
        except Exception as e:
            st.warning(f"Camera initialization error for index {index}: {str(e)}")
    return None

# ===== FACE PROCESSING =====
def extract_face_embedding(face_img):
    """Extract face embedding with error handling"""
    try:
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = face_img[:, :, :3]

        result = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="base"
        )
        return result[0]["embedding"] if result else None
    except Exception as e:
        st.error(f"Error extracting face features: {str(e)}")
        return None

def recognize_face(face_img, df, true_label=None):
    """Recognize face from database with optional true label for evaluation"""
    try:
        if df.empty:
            return "No database available", 0.0

        query_embedding = extract_face_embedding(face_img)
        if query_embedding is None:
            return "Error extracting features", 0.0

        best_match = ("Unknown", 0.0)
        for _, row in df.iterrows():
            try:
                db_embedding = np.array(row["embedding"])
                similarity = np.dot(query_embedding, db_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))

                if similarity > best_match[1]:
                    best_match = (row["name"], similarity)
            except Exception as e:
                st.warning(f"Error comparing with face {row['name']}: {str(e)}")
                continue

        result = best_match if best_match[1] > THRESHOLD else ("Unknown", best_match[1])

        # Store evaluation data if true_label is provided
        if true_label is not None:
            st.session_state.evaluation_data['true_labels'].append(true_label)
            st.session_state.evaluation_data['predicted_labels'].append(result[0])
            st.session_state.evaluation_data['confidence_scores'].append(result[1])

        return result
    except Exception as e:
        st.error(f"Recognition error: {str(e)}")
        return "Error", 0.0

# ===== EVALUATION METRICS =====
def calculate_metrics(true_labels, predicted_labels):
    """Calculate performance metrics"""
    metrics = {}

    # Get unique labels (excluding 'Unknown')
    labels = [label for label in np.unique(true_labels + predicted_labels) if label != "Unknown"]

    if len(labels) > 0:
        try:
            metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
            metrics['precision'] = precision_score(true_labels, predicted_labels, average='weighted', labels=labels, zero_division=0)
            metrics['recall'] = recall_score(true_labels, predicted_labels, average='weighted', labels=labels, zero_division=0)
            metrics['f1'] = f1_score(true_labels, predicted_labels, average='weighted', labels=labels, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
            metrics['confusion_matrix'] = cm
            metrics['labels'] = labels
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return None
    else:
        metrics = None

    return metrics

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix with memory management"""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {str(e)}")
    finally:
        plt.close(fig)

# ===== MAIN APP =====
def main():
    st.set_page_config(
        page_title="Face Recognition System",
        page_icon="🖥️",
        layout="centered"
    )
    st.title("🖥️ Face Recognition System")

    # Initialize database
    db_df = load_database()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Camera", "Upload Image", "Video Processing", "Manage Database", "Performance Evaluation"])

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

                if stop_button:
                    cam.release()
                    st.success("Camera session ended")

    with tab2:  # Upload Image tab
        st.header("Image Upload Recognition")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        true_label = st.text_input("Enter true label for evaluation (optional)")

        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                faces = DeepFace.extract_faces(
                    image,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True
                )

                for face in faces:
                    if face["confidence"] > 0.85:
                        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        name, confidence = recognize_face(face["face"], db_df, true_label=true_label if true_label else None)
                        cv2.putText(
                            image, f"{name} ({confidence:.2f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2
                        )

                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with tab3:  # Video Processing tab
        st.header("Video Face Recognition")
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        true_label = st.text_input("Enter true label for evaluation (optional)")

        if video_file:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())
                video_path = tfile.name
                tfile.close()

                cap = cv2.VideoCapture(video_path)
                frame_placeholder = st.empty()
                stop_button = st.button("Stop Video Processing", key="stop_video")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
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
                                x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                name, confidence = recognize_face(face["face"], db_df, true_label=true_label if true_label else None)
                                cv2.putText(
                                    frame, f"{name} ({confidence:.2f})",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2
                                )
                    except Exception as e:
                        st.warning(f"Face processing error: {str(e)}")

                    frame_placeholder.image(frame, use_container_width=True)

                    if stop_button:
                        break

                cap.release()
                if not stop_button:
                    st.success("Video processing completed")
            except Exception as e:
                st.error(f"Video processing error: {str(e)}")
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass

    with tab4:  # Database Management tab
        st.header("Face Database Management")

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
                                            db_df = add_face_to_db(register_name, embedding, reg_frame)
                                            st.success(f"Successfully registered {register_name}!")
                                    else:
                                        st.warning("No face detected in the captured image")
                                except Exception as e:
                                    st.error(f"Error registering face: {str(e)}")
                                finally:
                                    reg_cam.release()
                    else:
                        st.warning("Could not initialize webcam for registration")

                else:
                    reg_uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
                    if reg_uploaded_file and register_name:
                        try:
                            reg_image = Image.open(reg_uploaded_file)
                            if reg_image.mode != 'RGB':
                                reg_image = reg_image.convert('RGB')
                            reg_image = np.array(reg_image)
                            reg_image = cv2.cvtColor(reg_image, cv2.COLOR_RGB2BGR)

                            faces = DeepFace.extract_faces(
                                reg_image,
                                detector_backend=DETECTOR_BACKEND,
                                enforce_detection=True,
                                align=True
                            )
                            if faces:
                                embedding = extract_face_embedding(faces[0]["face"])
                                if embedding is not None:
                                    db_df = add_face_to_db(register_name, embedding, reg_image)
                                    st.success(f"Successfully registered {register_name}!")
                            else:
                                st.warning("No face detected in the uploaded image")
                        except Exception as e:
                            st.error(f"Error registering face: {str(e)}")

        with st.expander("🗑️ Delete Registered Face"):
            if not db_df.empty:
                delete_name = st.selectbox("Select name to delete", db_df["name"].unique())
                if st.button("Delete Face"):
                    db_df = delete_face_from_db(delete_name)
                    st.success(f"Deleted {delete_name} from database")
            else:
                st.info("No faces in database to delete")

        with st.expander("👀 View Database"):
            if not db_df.empty:
                st.write(f"Total registered faces: {len(db_df)}")

                for _, row in db_df.iterrows():
                    col1, col2 = st.columns([1, 3])

                    with col1:
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

    with tab5:  # Performance Evaluation tab
        st.header("Performance Evaluation Metrics")

        if not st.session_state.evaluation_data['true_labels']:
            st.info("No evaluation data available yet. Use the 'Upload Image' or 'Video Processing' tabs with true labels to collect data.")
        else:
            # Calculate metrics
            metrics = calculate_metrics(
                st.session_state.evaluation_data['true_labels'],
                st.session_state.evaluation_data['predicted_labels']
            )

            if metrics:
                st.subheader("Classification Report")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                col2.metric("Precision", f"{metrics['precision']:.2f}")
                col3.metric("Recall", f"{metrics['recall']:.2f}")
                col4.metric("F1 Score", f"{metrics['f1']:.2f}")

                st.subheader("Confusion Matrix")
                plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'])

                st.subheader("Raw Evaluation Data")
                eval_df = pd.DataFrame({
                    'True Label': st.session_state.evaluation_data['true_labels'],
                    'Predicted Label': st.session_state.evaluation_data['predicted_labels'],
                    'Confidence': st.session_state.evaluation_data['confidence_scores']
                })
                st.dataframe(eval_df)

                # Download evaluation data
                csv = eval_df.to_csv(index=False)
                st.download_button(
                    label="Download Evaluation Data",
                    data=csv,
                    file_name='face_recognition_evaluation.csv',
                    mime='text/csv'
                )
            else:
                st.warning("Insufficient data for metrics calculation. Need at least 2 different known labels.")

            if st.button("Clear Evaluation Data"):
                st.session_state.evaluation_data = {
                    'true_labels': [],
                    'predicted_labels': [],
                    'confidence_scores': []
                }
                st.success("Evaluation data cleared!")

if __name__ == "__main__":
    main()
