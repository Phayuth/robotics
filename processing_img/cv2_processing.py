import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner():
    gray = np.float32(imgGray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Threshold for an optimal value, it may vary depending on the image.
    plt.imshow(img)
    plt.show()


def hough_line():
    pass


def hough_circle():
    imgGrayBlur = cv2.medianBlur(imgGray, 5)
    circles = cv2.HoughCircles(imgGrayBlur, cv2.HOUGH_GRADIENT, 1, imgGrayBlur.shape[0] / 8, param1=100, param2=30, minRadius=50, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


def canny_edge():
    edges = cv2.Canny(imgGray, 100, 200)

    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image"),
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
    plt.show()


def apply_mask():
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[100:200, 100:200] = 1
    masked_image = np.where(mask[..., None], img, 0)
    plt.imshow(masked_image)
    plt.show()


def bin_erosion():
    binImage = np.zeros((300, 300))
    binImage[150:170, 30:40] = 1

    kernel = np.ones((6, 6), np.uint8)
    erodeImage = cv2.erode(binImage, kernel, cv2.BORDER_REFLECT)

    plt.subplot(121)
    plt.imshow(binImage, cmap="gray")
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(erodeImage, cmap="gray")
    plt.title("Erode Image"), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(str(os.path.abspath(os.getcwd())))

    imgPath = "./datasave/image/line_tracking.png"
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    harris_corner()
    hough_circle()
    canny_edge()
