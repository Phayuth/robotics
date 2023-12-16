import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def map_2d_1():
    map = np.zeros([31, 31])
    map[22:28, 10:21] = 1
    map[4:11, 10:21] = 1
    return 1 - map


def map_2d_2():
    map = np.zeros([201, 101])
    map[145:175, 50:70] = 0.5
    map[50:70, 60:80] = 1
    map[25:60, 40:60] = 1
    return 1 - map


def map_2d_empty():
    map = np.zeros([300, 300])
    return 1 - map


def map_3d():
    map = np.ones((51, 51, 51))
    map[0:51, 25:51, 0:51] = 0
    return map


def map_3d_empty():
    map = np.ones((51, 51, 51))
    return map


def pmap(return_rgb=False):
    image_size = 30
    box_size = 4
    no_box = 10
    image = Image.new('RGB', (image_size, image_size))
    d = ImageDraw.Draw(image)

    # random obstacle
    for i in range(no_box):
        xy = np.random.randint(image_size, size=2)
        rgb = np.random.randint(255, size=3)
        d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(rgb[0], rgb[1], rgb[2]))
    # fixed obstacle
    d.rectangle([17, 0, 22, 15], fill=(255, 255, 255))

    imgArray = np.array(image)

    map = np.zeros((imgArray.shape[1], imgArray.shape[0]))
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            map[i][j] = (int(imgArray[i][j][0]) + int(imgArray[i][j][1]) + int(imgArray[i][j][2])) / (255 * 3)

    map = np.transpose(map)
    map = 1 - map

    if return_rgb:
        return imgArray
    else:
        return map


def bmap(return_rgb=False):
    image_size = 50
    box_size = 3
    no_box = 20
    image = Image.new('RGB', (image_size, image_size))
    d = ImageDraw.Draw(image)

    # random obstacle
    for i in range(no_box):
        xy = np.random.randint(image_size, size=2)
        rgb = np.random.randint(155, size=3)
        d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(rgb[0], rgb[1], rgb[2]))
    # fixed obstacle
    d.rectangle([11, 50, 18, 20], fill=(255, 255, 255))
    d.rectangle([11, 10, 18, 0], fill=(255, 255, 255))
    d.rectangle([32, 50, 39, 40], fill=(255, 255, 255))
    d.rectangle([32, 30, 39, 0], fill=(255, 255, 255))

    imgArray = np.array(image)

    map = np.zeros((imgArray.shape[1], imgArray.shape[0]))
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            map[i][j] = (int(imgArray[i][j][0]) + int(imgArray[i][j][1]) + int(imgArray[i][j][2])) / (255 * 3)

    map = np.transpose(map)
    map = 1 - map

    if return_rgb:
        return imgArray
    else:
        return map


if __name__ == "__main__":

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].imshow(map_2d_1())
    axs[0, 0].set_title(map_2d_1.__name__)

    axs[0, 1].imshow(map_2d_2())
    axs[0, 1].set_title(map_2d_2.__name__)

    axs[1, 0].imshow(map_2d_empty())
    axs[1, 0].set_title(map_2d_empty.__name__)

    axs[1, 1].imshow(pmap())
    axs[1, 1].set_title(pmap.__name__)

    axs[0, 2].imshow(bmap())
    axs[0, 2].set_title(bmap.__name__)

    plt.show()
