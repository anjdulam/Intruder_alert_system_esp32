import cv2
import numpy as np
import face_recognition
import websocket
import json

# Load known faces
known_face_encodings = []
known_face_names = []

# Add known faces here
# Example:
# known_face_encodings.append(face_encoding_of_person)
# known_face_names.append("Name of Person")

# Load a sample image and learn how to recognize it
sample_image = face_recognition.load_image_file("known_faces/anjali.jpg")
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]
known_face_encodings.append(sample_face_encoding)
known_face_names.append("anjali")
# Load a sample image and learn how to recognize it
sample_image = face_recognition.load_image_file("known_faces/nayan.jpg")
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]
known_face_encodings.append(sample_face_encoding)
known_face_names.append("nayan")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)  # 0 for default webcam

websocket_server = "ws://192.168.30.243:81"
websocket_connection = websocket.create_connection(websocket_server)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the name of the matched known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                # If the face is unknown, send a message to the WebSocket server
                message = "on"
                websocket_connection.send(message)

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
websocket_connection.close()