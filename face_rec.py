import cv2
import face_recognition

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the target image (the face you want to detect)
target_image = cv2.imread('/content/drive/MyDrive/human_face.png') 
gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Detect the face in the target image
target_faces = face_cascade.detectMultiScale(gray_target, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Get the target face encoding if a face is found
if len(target_faces) > 0:
    x, y, w, h = target_faces[0]  # Assuming only one face in the target image
    target_face_encoding = face_recognition.face_encodings(target_image, [(y, x + w, y + h, x)])[0]
else:
    print("No face detected in the target image.")
    target_face_encoding = None

# Proceed only if a target face encoding is available
if target_face_encoding is not None:
    # Open the video file
    video_path = '/content/drive/MyDrive/large_video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_girl.mp4', fourcc, fps, (width, height))

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

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processed and saved as 'output.mp4'")
else:
    print("Target face not found, video processing skipped.") 
