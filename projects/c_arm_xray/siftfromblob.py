import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("./datasave/c_arm/img_01.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./datasave/c_arm/img_02.jpg", cv2.IMREAD_GRAYSCALE)

# avoid seg fault
ver = (cv2.__version__).split(".")
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()


keypoints1 = detector.detect(img1)
keypoints2 = detector.detect(img2)

sift = cv2.SIFT_create()
kp1, des1 = sift.compute(img1, keypoints1)
kp2, des2 = sift.compute(img2, keypoints2)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()
