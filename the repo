import cv2
import mediapipe as mp
import dlib
import numpy as np
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# Initialize Video Capture (replace '0' with your camera index or video file)
#here 0 uses the first webcam it has access to
cap = cv2.VideoCapture(0)
names = ["unknown1"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = face_detection.process(rgb_frame)
# This block of code is responsible for the rectangle that appears on the screen
    if results.detections:
        for i,detection in  enumerate(results.detections):
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), \
                         int(box.width * iw), int(box.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)
            name = names[i] if i < len(names) else "Unknown"
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Monitor', frame)

    if cv2.waitKey(1) == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()
