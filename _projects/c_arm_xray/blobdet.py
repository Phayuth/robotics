# https://learnopencv.com/blob-detection-using-opencv-python-c/
import cv2
import numpy as np

im = cv2.imread("./datasave/c_arm/img_01.jpg", cv2.IMREAD_GRAYSCALE)
print(f"> im.shape: {im.shape}")

# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200

# # Filter by Area.
# params.filterByArea = True
# params.minArea = 1500

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01


# avoid seg fault
ver = (cv2.__version__).split(".")
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(im)
# print(f"> keypoints: {keypoints}")

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("og iamge", im)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

print("".center(50, "-"))
a = keypoints[0]
print(a.pt, a.angle, a.class_id, a.octave, a.response, a.size)
