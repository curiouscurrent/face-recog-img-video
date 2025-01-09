import cv2
import face_recognition
import streamlit as st
import tempfile
import os
import numpy as np
from io import BytesIO

# Function to process the video
def process_video(uploaded_video, uploaded_image):
    # Load the target image (the face you want to detect)
    target_image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Use Dlib's face detector for faster detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    target_faces = face_cascade.detectMultiScale(gray_target, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the target face encoding if a face is found
    if len(target_faces) > 0:
        x, y, w, h = target_faces[0]  # Assuming only one face in the target image
        target_face_encoding = face_recognition.face_encodings(target_image, [(y, x + w, y + h, x)])[0]
    else:
        st.error("No face detected in the target image.")
        return None

    # Open the video file
    video_bytes = uploaded_video.read()
    video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(video_path.name, 'wb') as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path.name)

    # Get video properties for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary output video file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path.name, fourcc, fps, (width, height))

    # Process all frames to maintain original speed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if any detected faces match the target face
        for (x, y, w, h) in faces:
            face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
            matches = face_recognition.compare_faces([target_face_encoding], face_encoding, tolerance=0.4)
            if True in matches:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Save the frames where the target face is found
                out.write(frame)

    # Release resources
    cap.release()
    out.release()

    # Return the processed video for download
    with open(output_path.name, "rb") as f:
        video_bytes = f.read()

    # Return the path to the processed video
    return video_bytes, output_path.name

# Streamlit UI
st.title("Face Detection in Videos")
st.write("Upload a video and an image (face) to detect the face and get the output video.")

# Upload components
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
uploaded_image = st.file_uploader("Upload an image (face)", type=["jpg", "png", "jpeg"])

# Button to start processing
if st.button("Process Video"):
    if uploaded_video and uploaded_image:
        # Process the video and image
        output_video, output_file_path = process_video(uploaded_video, uploaded_image)
        
        if output_video:
            # Provide download link for the processed video
            st.success("Video processed successfully!")
            st.download_button("Download Processed Video", data=output_video, file_name="output_video.mp4", mime="video/mp4")
            
            # Clean up temporary files only after download is initiated
            os.remove(output_file_path)
            st.info("Temporary files cleaned up.")
    else:
        st.error("Please upload both a video and an image.")
