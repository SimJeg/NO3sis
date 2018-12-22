""" Create a triangulation of faces and save it """

import cv2
import numpy as np
from face_recognition import face_landmarks

# Get image
image = cv2.imread('../images/face.jpg')
h, w, _ = image.shape

# Get landmarks
landmarks = face_landmarks(image)[0]
pts = np.vstack(landmarks.values())
pts = np.clip(pts, [0, 0], [w - 1, h - 1])

# Get DelaunayTriangulation
subdiv = cv2.Subdiv2D((0, 0, w, h))
[subdiv.insert(tuple(p)) for p in pts]
tri_list = subdiv.getTriangleList()


# Convert each triangle to a set of indexes
pts2index = dict((str(p), i) for i, p in enumerate(pts))
tri_list = [[pts2index[str(tri[2 * i:2 * (i + 1)].astype(int))] for i in range(3)] for tri in tri_list]
tri_list = np.array(tri_list)

hull_list = cv2.convexHull(pts)
hull_list = [pts2index[str(p[0])] for p in hull_list]

# Save list
np.save('triangle_list', tri_list)
np.save('hull_list', hull_list)