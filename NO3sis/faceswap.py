"""
Faceswap between webcam and a fixed image

Potential improvement : https://www.learnopencv.com/face-swap-using-opencv-c-python/
Code : https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
More info about models : https://github.com/davisking/dlib-models
"""

import cv2
import numpy as np
from face_recognition import face_landmarks

def get_hull(landmarcks, img):
    pts = np.vstack(landmarks.values())[:27]
    pts[:17] = pts[:17][::-1]
    pts = np.clip(pts, (0, 0), (img.shape[1] - 1, img.shape[0] - 1))
    return pts


video_capture = cv2.VideoCapture(0)

# Loop = 200ms per iteration, 160ms just for face_landmarks
while True:
    # Get landmarks from new frame
    _, frame = video_capture.read()
    h, w, _ = frame.shape
    frame_swap = np.copy(frame)

    landmarks_list = face_landmarks(frame)
    n_faces = len(landmarks_list)

    if n_faces > 1:

        pts_list = [get_hull(landmarks, frame) for landmarks in landmarks_list]

        for i in range(n_faces):

            # Restrict the transformation to a small box centered on the face
            j1, i1, w1, h1 = cv2.boundingRect(pts_list[i - 1])
            j2, i2, w2, h2 = cv2.boundingRect(pts_list[i])
            pts1_rect = pts_list[i - 1] - np.array([j1, i1])
            pts2_rect = pts_list[i] - np.array([j2, i2])

            face1 = frame_swap[i2:i2 + h2, j2:j2 + w2]
            homography, _ = cv2.findHomography(pts1_rect, pts2_rect)

            # Warp the source face
            face1_warped = cv2.warpPerspective(face1, homography, (w2, h2))
            # Fill face[i] with the warped face[i-1]
            mask = np.zeros((h2, w2, 3), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts2_rect, (1, 1, 1))

            # Harmonize colors
            face1_warped = cv2.seamlessClone(face1_warped, frame[i2:i2 + h2, j2:j2 + w2],
                                             mask * 255, (w2 // 2, h2 // 2), cv2.NORMAL_CLONE)

            frame_swap[i2:i2 + h2, j2:j2 + w2] = \
                frame[i2:i2 + h2, j2:j2 + w2] * (1 - mask) + \
                face1_warped * mask

        cv2.imshow('Video', frame_swap)
        if cv2.waitKey(1) == ord('q'):
            break
video_capture.release()
