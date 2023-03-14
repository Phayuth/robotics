import numpy as np
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
    map = np.ones((51,51,51))
    map[0:51, 25:51, 0:51] = 0
    return map

def map_3d_empty():
    map = np.ones((51,51,51))
    return map

def pmap():
    image_size = 30
    box_size = 4
    no_box = 10
    image = Image.new('RGB', (image_size, image_size))
    d = ImageDraw.Draw(image)

    np.random.seed(9)  # 9
    for i in range(no_box):
        xy = np.random.randint(image_size, size=2)
        rgb = np.random.randint(255, size=3)
        d.rectangle([xy[0], xy[1], xy[0] + box_size, xy[1] + box_size], fill=(rgb[0], rgb[1], rgb[2])) # add obstacle in loop
    d.rectangle([17, 0, 22, 15], fill=(255, 255, 255)) # add obstacle manually

    imgArray = np.array(image)

    map = np.zeros((imgArray.shape[1], imgArray.shape[0]))
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            map[i][j] = (int(imgArray[i][j][0]) + int(imgArray[i][j][1]) + int(imgArray[i][j][2])) / (255 * 3)

    map = np.transpose(map)
    map = 1 - map

    return map