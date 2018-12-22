"""
Faceswap between webcam and a fixed image

Potential improvement : https://www.learnopencv.com/face-swap-using-opencv-c-python/
Code : https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
More info about models : https://github.com/davisking/dlib-models
"""

import cv2
import numpy as np
from face_recognition import face_landmarks
import sys

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = '../images/trump.jpg'


def get_hull(landmarcks, img):
    pts = np.vstack(landmarks.values())[:27]
    pts[:17] = pts[:17][::-1]
    pts = np.clip(pts, (0, 0), (img.shape[1] - 1, img.shape[0] - 1))
    return pts

# detector = cv2.CascadeClassifier('detector.xml').detectMultiScale

# Get landmarks from source image
source = cv2.imread(image_path)
landmarks = face_landmarks(source)[0]
pts1 = get_hull(landmarks, source)
j1, i1, w1, h1 = cv2.boundingRect(pts1)
face1 = source[i1:i1 + h1, j1:j1 + w1]
pts1 = np.float32(pts1 - np.array([j1, i1]))

video_capture = cv2.VideoCapture(0)

# Loop = 180ms per iteration, 150ms just for _raw_face_locations (seems to use only 1 CPU). Move to GPU ?
while True:
    # Get landmarks from new frame
    _, frame = video_capture.read()
    h, w, _ = frame.shape
    frame_swap = np.copy(frame)

    landmarks_list = face_landmarks(frame)

    # loc = detector(frame, flags=cv2.CASCADE_SCALE_IMAGE)
    # loc = [(r[1], r[0], r[1] + r[2], r[0] + r[3]) for r in loc]
    # landmarks_list = face_landmarks(frame, face_locations=loc)

    faces = []
    for landmarks in landmarks_list:
        pts2 = get_hull(landmarks, frame)

        # Restrict the transformation to a small box centered on face[i-1]
        j2, i2, w2, h2 = cv2.boundingRect(pts2)
        pts2_rect = pts2 - np.array([j2, i2])
        homography, _ = cv2.findHomography(pts1, pts2_rect)

        # Warp the source image
        face1_warped = cv2.warpPerspective(face1, homography, (w2, h2))
        # Fill face[i] with the warped face[i-1]
        mask = np.zeros((h2, w2, 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts2_rect, (1, 1, 1))

        # Harmonize colors (a bit long)
        face1_warped = cv2.seamlessClone(face1_warped, frame[i2:i2 + h2, j2:j2 + w2],
                                mask * 255, (w2 // 2, h2 // 2), cv2.NORMAL_CLONE)

        frame_swap[i2:i2 + h2, j2:j2 + w2] = \
            frame[i2:i2 + h2, j2:j2 + w2] * (1 - mask) + \
            face1_warped * mask

    cv2.imshow('Video', frame_swap)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
