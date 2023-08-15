import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera
capture = cv2.VideoCapture(4)
capture.set(3, 640)
capture.set(4, 640)

success, imgBGR = capture.read()
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
# imgColor = img[1] # from cv2 read return tuple and the image is in to 2nd index
plt.imshow(imgRGB)
plt.show()