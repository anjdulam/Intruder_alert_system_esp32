import face_recognition
import cv2
import os

# Create a directory to store known faces if it doesn't exist
known_faces_dir = "known_faces"
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Initialize some variables
face_encodings = []
known_face_names = []

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the current frame
    cv2.imshow('Capture faces (Press "c" to capture)', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture face if 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Ask for the name of the person
            name = input("Enter the name of the person: ")
            known_face_names.append(name)
            # Save the captured face image
            face_image_path = os.path.join(known_faces_dir, f"{name}.jpg")
            cv2.imwrite(face_image_path, frame)

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Now, you can use the captured faces as known faces in your face recognition system.