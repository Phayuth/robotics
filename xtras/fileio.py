import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = "./datasave/mnist/zero.png"
img_mp = mpimg.imread(path)
img_tv = torchvision.io.read_image(path).permute(1, 2, 0) # torch return 3,h,w but we want h,w,3, so we permute it
img_cv = cv2.imread(path)

img_gray = img_mp[:,:,0]

print(np.shape(img_gray))
# plt.imshow(img_gray)
# plt.imshow(img)
plt.imshow(img_cv)
plt.show()