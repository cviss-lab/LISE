import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from skimage.io import imread
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
import cv2

# original image
R1 = np.zeros((3, 3))
R1[0, 0] = -1;
R1[1, 1] = 1;
R1[2, 2] = -1;

T1 = np.array(([0], [0], [1]));

# augmented image

theta = 25

th_deg = random.uniform(-theta, theta);


th = np.deg2rad(th_deg)
phi_deg = random.uniform(360);

phi = np.deg2rad(phi_deg)

a = np.array(([0], [0], [1]));

b = np.array(([np.sin(np.pi + th) * np.cos(phi)], [np.sin(np.pi + th) * np.sin(phi)], [np.cos(np.pi + th)]));

v = (np.cross(a.T, b.T))
c = np.dot(np.squeeze(a), np.squeeze(b))

vx = np.array(([0, v[0, 2], v[0, 1]],
               [v[0, 2], 0, -v[0, 0]],
               [-v[0, 1], v[0, 0], 0]));
vx = np.round(vx, 4)

R2 = np.identity(3) + vx + (np.matmul(vx, vx) * (1 / (1 + c)))
T2 = np.array(([np.sin(th) * np.cos(phi)], [np.sin(th) * np.sin(phi)], [np.cos(th)]));

fig = plt.figure()
ax = fig.gca(projection='3d')
org = np.zeros((1, 3));
da = 0.1 * (org.T - T1);
db = 0.1 * (org.T - T2)
#
# T1[0, 0]
#
# ax.quiver(T1[0, 0], T1[1, 0], T1[2, 0], da[0, 0], da[1, 0], da[2, 0], length=0.05, normalize=True)
# ax.quiver(T2[0, 0], T2[1, 0], T2[2, 0], db[0, 0], db[1, 0], db[2, 0], length=0.05, normalize=True, color='g')
# ax.axis([-1.5, 1.5, -1.5, 1.5])
#
# plt.show()

img = imread('IMG_6996.JPG')

c = img.shape[1]
r = img.shape[0]

f = np.maximum(r, c) * 1.2
K = np.array([[f, 0, c / 2],
              [0, -f, r / 2],
              [0, 0, 1]])
R1 = np.linalg.inv(R1)
R2 = np.linalg.inv(R2)

P1 = np.matmul(K, np.concatenate((R1, np.matmul(-R1, T1)), axis=1))
P2 = np.matmul(K, np.concatenate((R2, np.matmul(-R2, T2)), axis=1))

H = np.matmul(P2, np.linalg.pinv(P1))

# n = 300  # size of patch

# bbox = np.array(([0, 0],
#                  [n, 0],
#                  [n, n],
#                  [0, n]))
#
# poly_pt = np.matmul(np.linalg.inv(H), np.concatenate((bbox, np.ones((4, 1))), axis=1).T)


h, w = img.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
pts_distort = cv2.perspectiveTransform(pts, H)

[xmin, ymin] = np.int32(pts_distort.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts_distort.max(axis=0).ravel() + 0.5)
t = [-xmin, -ymin]
Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

dst = cv2.warpPerspective(img, Ht.dot(H), (xmax - xmin, ymax - ymin))

pts_stack=np.concatenate(pts, axis=0)
pts_distort_stack=np.concatenate(pts_distort, axis=0)   # feed it in get_crop

# reverse= cv2.warpPerspective(img, np.linalg.inv(Ht).dot(np.linalg.inv(H)), (xmax - xmin, ymax - ymin))
plt.subplot(131), plt.imshow(img), plt.title('Input')
plt.subplot(132), plt.imshow(dst), plt.title('Output')
# plt.subplot(133), plt.imshow(reverse), plt.title('Reverse')
plt.show()

