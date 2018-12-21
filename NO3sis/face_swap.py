import cv2
import face_recognition
import numpy as np
from skimage.draw import polygon

# Step 1 : get landmarks from source image

source = cv2.imread('face_mask.jpg')
face_landmarks = face_recognition.face_landmarks(source)[0]
point_src = face_landmarks.values()
pts_src = np.vstack([
    face_landmarks['chin'][::-1],
    face_landmarks['left_eyebrow'],
    face_landmarks['right_eyebrow'],
]).reshape((-1, 1, 2))

# Step 2 : target image

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    face_landmarks_list = face_recognition.face_landmarks(frame)
    for face_landmarks in face_landmarks_list:
        pts = np.vstack([
            face_landmarks['chin'][::-1],
            face_landmarks['left_eyebrow'],
            face_landmarks['right_eyebrow'],
        ]).reshape((-1, 1, 2))

        h, _ = cv2.findHomography(pts_src, pts)
        source_warped = cv2.warpPerspective(source, h, source.shape[:2][::-1])

        frame_swap = np.copy(frame)
        rr, cc = polygon(pts[..., 1], pts[..., 0])
        frame_swap[rr, cc] = source_warped[rr, cc]

        # Color transfer
        mask = np.zeros(frame.shape)
        mask[rr, cc] = 255
        r = cv2.boundingRect(pts)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        frame = cv2.seamlessClone(np.uint8(frame_swap), np.uint8(frame), np.uint8(mask), center, cv2.NORMAL_CLONE)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()


# Improvement : https://www.learnopencv.com/face-swap-using-opencv-c-python/
# Code : https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
