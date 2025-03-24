def main():
    st.title("Live Facial Recognition System")
    
    # Input: Path to known faces directory
    known_faces_path = st.text_input("Enter the path to known faces directory:")
    start = st.button("Start Recognition")

    if start and known_faces_path:
        st.write("Starting live facial recognition...")

        cap = cv2.VideoCapture(0)  # Access webcam

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the webcam")
                break

            st.image(frame, channels="BGR")

            # Perform face recognition
            result = recognize_face(frame, known_faces_path)
            st.write(result)

            # Press 'q' to quit the recognition loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
