import cv2
import numpy as np
import matplotlib.pyplot as plt

# import image
imgPath = "./test/img.jpg"
img = cv2.imread(imgPath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def harris_corner():
    gray = np.float32(imgGray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    plt.imshow(img)
    plt.show()


def hough_line():
    pass


def hough_circle():
    imgGray = cv2.medianBlur(imgGray, 5)
    circles = cv2.HoughCircles(imgGray, cv2.HOUGH_GRADIENT, 1, imgGray.shape[0] / 8, param1=100, param2=30, minRadius=50, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])


def canny_edge():
    edges = cv2.Canny(imgGray, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


# Apply Mask
# masked_image = np.where(mask[..., None], img, 0)
# plt.imshow(mask)
# plt.show()
