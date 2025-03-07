import cv2
import numpy as np

im = cv2.imread("./datasave/c_arm/img_08.jpg", cv2.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(thresh, im_bw) = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY)
print(f"> (thresh: {(thresh)}")


sift = cv2.SIFT_create()
kp = sift.detect(im, None)
# imgk = cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgk = cv2.drawKeypoints(im_bw, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Gray", im)
cv2.imshow("Sift", imgk)
cv2.imshow("Binary", im_bw)
cv2.waitKey(0)



