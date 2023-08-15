import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Camera
capture = cv2.VideoCapture(4)
capture.set(3, 640)
capture.set(4, 640)

# Multi Plot cv2
# grid_width = 800
# grid_height = 400
# grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
# success, frame = capture.read()
# cell_width = grid_width // 2
# cell_height = grid_height
# image1 = cv2.resize(frame, (cell_width, cell_height))
# image2 = cv2.resize(frame, (cell_width, cell_height))
# grid[0:cell_height, 0:cell_width] = image1
# grid[0:cell_height, cell_width:2*cell_width] = image2
# cv2.imshow("Grid", grid)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Plot cv2
# i = np.load("./iiii.npy")
# tt = torch.from_numpy(i) # we cannot view image in tensor with cv2imshow
# # plt.imshow(i)
# # plt.show()
# while True:
#     cv2.imshow("fdsa", tt)
#     cv2.waitKey(1)

# Plot mask
_, img = capture.read()
mask = np.load("./mask.npy")
masked_image = np.where(mask[..., None], img, 0)
plt.imshow(mask)
plt.show()
