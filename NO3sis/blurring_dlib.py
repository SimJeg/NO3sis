# Press 'h' for HOG features, 'c' for CNN features, 'q' to quit
# Source : https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py


import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)
face_locations = []
mode = 'hog'

while True:
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face detection processing
    face_locations = face_recognition.face_locations(frame, model=mode)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (25, 25), 25)

        # Put the blurred face region back into the frame image
        frame[top:bottom, left:right] = face_image

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('h'):
        mode = 'hog'
    if key == ord('c'):
        mode = 'cnn'
    print(mode)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
