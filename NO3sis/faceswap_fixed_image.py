"""
Faceswap between webcam and a fixed image

Potential improvement : https://www.learnopencv.com/face-swap-using-opencv-c-python/
Code : https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
More info about models : https://github.com/davisking/dlib-models
"""

import cv2
import numpy as np
from face_recognition import face_landmarks
from skimage.draw import polygon
import sys

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = '../images/macron.jpg'


def get_pts(landmarks, img):
    """ Get points around the face"""
    pts = np.vstack([
        landmarks['chin'][::-1],
        landmarks['left_eyebrow'],
        landmarks['right_eyebrow'],
    ])
    pts[:, 0] = np.clip(pts[:, 0], 0, img.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img.shape[0] - 1)
    pts = pts.reshape((-1, 1, 2))
    return pts


# Get landmarks from source image

source = cv2.imread(image_path)
landmarks = face_landmarks(source)[0]
pts_src = get_pts(landmarks, source)

video_capture = cv2.VideoCapture(0)

# Loop = 180ms per iteration, 150ms just for _raw_face_locations (seems to use only 1 CPU). Move to GPU ?
while True:
    # Get landmarks from new frame
    _, frame = video_capture.read()
    frame_swap = np.copy(frame)
    landmarks_list = face_landmarks(frame)

    faces = []
    for landmarks in landmarks_list:
        pts_dst = get_pts(landmarks, frame)
        # Find the homography between source and destination points
        h, _ = cv2.findHomography(pts_src, pts_dst)
        # Warp the source image
        source_warped = cv2.warpPerspective(source, h, frame.shape[:2][::-1])

        # Swap faces
        rr, cc = polygon(pts_dst[..., 1], pts_dst[..., 0])
        frame_swap[rr, cc] = source_warped[rr, cc]

        # Harmonize colors (a bit long)
        mask = np.zeros(frame.shape, dtype='uint8')
        mask[rr, cc] = 255
        r = cv2.boundingRect(pts_dst)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        tmp = cv2.seamlessClone(frame_swap, frame, mask, center, cv2.NORMAL_CLONE)
        frame_swap[rr, cc] = tmp[rr, cc]

    cv2.imshow('Video', frame_swap)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
