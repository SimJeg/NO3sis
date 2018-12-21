import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('detector.xml').detectMultiScale

while True:
    _, frame = camera.read()
    faces = detector(frame, flags=cv2.CASCADE_SCALE_IMAGE)
    for x, y, w, h in faces:
        frame[y:y + w, x:x + h] = cv2.GaussianBlur(frame[y:y + w, x:x + h], (25, 25), 25)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) == ord('q'):
        break
