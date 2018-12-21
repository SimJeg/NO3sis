import cv2
import matplotlib.pyplot as plt
plt.ion()

# Initialize the the plot with the first frame from the camera
camera = cv2.VideoCapture(0)
get_frame = lambda : cv2.cvtColor(camera.read()[1], cv2.COLOR_BGR2RGB)
initial_frame = get_frame()
plot = plt.imshow(initial_frame)

# For each new frame, blur detected faces
get_face = cv2.CascadeClassifier('detector.xml').detectMultiScale
while True:
    frame = get_frame()
    faces = get_face(frame, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces):
        x, y, w, h = faces[0]
        blur = cv2.GaussianBlur(frame[y:y + w, x:x + h], (25,25), 25)
        frame[y:y + w, x:x + h] = blur
    plot.set_data(frame)
    plt.pause(1e-10)
