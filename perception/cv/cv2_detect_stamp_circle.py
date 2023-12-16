import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def __check_state(img, threshold):
    imgravel = np.ravel(img)
    # area = img.shape[0] * img.shape[1]
    # val = np.sum(np.abs(np.diff(imgravel)))
    # thres = val / area
    imgmean = np.mean(imgravel)
    imgstd = np.std(imgravel)
    stdpermean = (imgstd/imgmean) * 100
    if stdpermean > threshold:
        return f"Stamped Occupied {stdpermean:.3f}"
    else:
        return f"Free {stdpermean:.3f}"


def detect_stamp_in_circle(imgRGB, minRadius=100, maxRadius=200, zoom=0, threshold=15):  # positive = zoomout, negative = zoomin
    imgGray = cv.cvtColor(imgRGB, cv.COLOR_RGB2GRAY)
    imgGray = cv.medianBlur(imgGray, 5)
    circles = cv.HoughCircles(imgGray, cv.HOUGH_GRADIENT, 1, imgGray.shape[0] / 8, param1=100, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    cirradiusmean = np.mean(circles[0, :, 2], dtype=np.uint16)  # exploit circle in image usually the same radius
    circles[0, :, 2] = cirradiusmean
    R = ((2*cirradiusmean) / np.sqrt(2) + zoom) / 2  # inner square in circle crop

    cirGray = []
    cirRGB = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            rowmin = center[1] - R
            rowmax = center[1] + R
            colmin = center[0] - R
            colmax = center[0] + R
            cropGray = imgGray[int(rowmin):int(rowmax), int(colmin):int(colmax)]
            cropRGB = imgRGB[int(rowmin):int(rowmax), int(colmin):int(colmax)]
            cirGray.append(cropGray)
            cirRGB.append(cropRGB)

    state = [__check_state(cG, threshold) for cG in cirGray]
    return circles, cirRGB, cirGray, state


if __name__ == "__main__":
    # Read image.
    imgBGR = cv.imread('/home/yuth/Downloads/testcir.jpg', cv.IMREAD_COLOR)
    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)

    plt.imshow(imgRGB)
    plt.show()

    circles, cirRGB, cirGray, state = detect_stamp_in_circle(imgRGB, minRadius=10, maxRadius=200, zoom=-10, threshold=15)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(imgRGB, center, 1, (0, 100, 100), 3)  # circle center
            cv.circle(imgRGB, center, radius, (255, 0, 255), 3)  # circle outline

    plt.imshow(imgRGB)
    plt.show()

    fig, axs = plt.subplots(2, 5)
    axs[0, 0].imshow(cirRGB[0])
    axs[0, 0].set_title(f'{state[0]}')

    axs[0, 1].imshow(cirRGB[1])
    axs[0, 1].set_title(f'{state[1]}')

    axs[0, 2].imshow(cirRGB[2])
    axs[0, 2].set_title(f'{state[2]}')

    axs[0, 3].imshow(cirRGB[3])
    axs[0, 3].set_title(f'{state[3]}')

    axs[0, 4].imshow(cirRGB[4])
    axs[0, 4].set_title(f'{state[4]}')

    axs[1, 0].imshow(cirRGB[5])
    axs[1, 0].set_title(f'{state[5]}')

    axs[1, 1].imshow(cirRGB[6])
    axs[1, 1].set_title(f'{state[6]}')

    plt.show()
