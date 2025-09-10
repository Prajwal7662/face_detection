"""
Face Detection using OpenCV
Author: Dikesh Chavhan
Date: 2025-08-27

Features:
- Real-time face detection from webcam
- Displays number of faces detected
- Colored rectangles for each detected face
- Labeled faces for professional look
- Press 'q' to quit the application

"""


import cv2
import random

face_cap = cv2.CascadeClassifier("C:/Users/Dikesh Chavhan/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_alt_tree.xml")

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Cannot open webcam.")

print("Face Detection Started. Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    col = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    for i, (x, y, w, h) in enumerate(faces, start=1):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Face {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows() 
