import cv2
import face_recognition
import numpy as np

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    face_landmarks_list = face_recognition.face_landmarks(frame)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            pts = np.array(face_landmarks[facial_feature])
            pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (255, 255, 255))

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
