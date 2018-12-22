"""Faceswap between faces detected on webcam"""

import cv2
import numpy as np
from face_recognition import face_landmarks

video_capture = cv2.VideoCapture(0)

image = cv2.imread('../images/trump.jpg')
image = cv2.resize(image, (640, 480))
tri_list = np.load('triangle_list.npy')
hull_list = np.load('hull_list.npy')




# Loop = 180ms per iteration, 150ms just for _raw_face_locations (seems to use only 1 CPU). Move to GPU ?
while True:
    # Get landmarks from new frame
    _, frame = video_capture.read()
    frame = np.hstack([frame, image]).astype('uint8')
    h, w, _ = frame.shape

    frame_swap = np.copy(frame)
    landmarks_list = face_landmarks(frame)
    n_faces = len(landmarks_list)

    if n_faces > 1:

        pts_list = [np.vstack(landmarks.values()) for landmarks in landmarks_list]
        pts_list = [np.clip(pts, (0, 0), (w - 1, h - 1)) for pts in pts_list]

        # Move face[i-1] on face[i]
        for index in range(n_faces):
            for tri in tri_list:
                # Convert triangle to coordinates
                tri1 = pts_list[index - 1][tri]
                tri2 = pts_list[index][tri]

                # Restrict the transformation to a small box centered on face[i-1]
                j1, i1, w1, h1 = cv2.boundingRect(tri1)
                j2, i2, w2, h2 = cv2.boundingRect(tri2)
                tri1_rect = np.float32(tri1 - np.array([j1, i1]))
                tri2_rect = np.float32(tri2 - np.array([j2, i2]))

                face1 = frame[i1:i1 + h1, j1:j1 + w1]

                # Get affine transform and warp face[i-1]
                affine = cv2.getAffineTransform(tri1_rect, tri2_rect)
                face2 = cv2.warpAffine(face1, affine, (w2, h2), None, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)

                # Fill face[i] with the warped face[i-1]
                mask = np.zeros((h2, w2, 3), dtype=np.float32)
                cv2.fillConvexPoly(mask, np.int32(tri2_rect), (1, 1, 1))

                frame_swap[i2:i2 + h2, j2:j2 + w2] = \
                    frame_swap[i2:i2 + h2, j2:j2 + w2] * (1 - mask) + \
                    face2 * mask

            # Harmonize colors within convex hull
            hull = pts_list[index][hull_list]
            mask = np.zeros(frame.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
            r = cv2.boundingRect(hull)
            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
            tmp = cv2.seamlessClone(np.uint8(frame_swap), np.uint8(frame), np.uint8(mask), center, cv2.NORMAL_CLONE)
            cv2.fillConvexPoly(mask, np.int32(hull), (1, 1, 1))
            frame_swap = frame_swap * (1 - mask) + tmp * mask


    cv2.imshow('Video', frame_swap)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()