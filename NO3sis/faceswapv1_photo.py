"""Faceswap between faces detected on webcam"""

import cv2
import numpy as np
from face_recognition import face_landmarks
from skimage.draw import polygon
import sys

video_capture = cv2.VideoCapture(0)

image = cv2.imread('../images/trump.jpg')
image = cv2.resize(image, (640, 480))

# Loop = 180ms per iteration, 150ms just for _raw_face_locations (seems to use only 1 CPU). Move to GPU ?
while True:
    # Get landmarks from new frame
    _, frame = video_capture.read()
    frame = np.hstack([frame, image])
    h, w, _ = frame.shape

    frame_swap = np.copy(frame)

    landmarks_list = face_landmarks(frame)
    pts_list = [np.vstack(landmarks.values()) for landmarks in landmarks_list]
    pts_list = [np.clip(pts, [0, 0], [w - 1, h - 1]) for pts in pts_list]

    hull_list = []
    for pts in pts_list:
        hull = pts[:27]
        hull[:17] = hull[:17][::-1]
        hull_list.append(hull)

    n_faces = len(landmarks_list)

    if n_faces > 1:
        for i in range(n_faces):
            # Find the homography face(i) > face(i-1) and replace face(i-1) by face(i)
            h, _ = cv2.findHomography(pts_list[i], pts_list[i - 1])
            frame_warped = cv2.warpPerspective(frame, h, frame.shape[:2][::-1])
            rr, cc = polygon(hull_list[i - 1][..., 1], hull_list[i - 1][..., 0])
            frame_swap[rr, cc] = frame_warped[rr, cc]

            # Harmonize lors (a bit long)
            mask = np.zeros(frame.shape, dtype='uint8')
            mask[rr, cc] = 255
            r = cv2.boundingRect(hull_list[i - 1])
            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
            tmp = cv2.seamlessClone(frame_swap, frame, mask, center, cv2.NORMAL_CLONE)
            frame_swap[rr, cc] = tmp[rr, cc]

    cv2.imshow('Video', frame_swap)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
