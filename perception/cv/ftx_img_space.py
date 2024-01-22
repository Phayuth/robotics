import numpy as np


class ImageFeature:

    def bresenham_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx / 2
        ystep = 1 if y0 < y1 else -1
        y = y0

        points = []
        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            points.append(coord)
            error -= dy
            if error < 0:
                y += ystep
                error += dx

        return points

    def circle_fit(x, y):
        A = []
        for i in range(len(x)):
            row = [x[i], y[i], 1]
            A.append(row)
        A = np.array(A)

        B = []
        for i in range(len(y)):
            row = [-x[i]**2 - y[i]**2]
            B.append(row)
        B = np.array(B)

        x = np.linalg.inv((np.transpose(A) @ A)) @ np.transpose(A) @ B  # pseudo inverse

        a = -x[0] / 2
        b = -x[1] / 2
        e = x[2]

        r = np.sqrt(a**2 + b**2 - e)
        return a, b, r


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    # import image
    imgPath = "/home/yuth/Downloads/harriscorner.jpg"
    # imgPath = "./map/mapdata/image/map1.png"
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # input_img = cv2.imread("/home/yuth/Downloads/ex2.png")

    # Bresenham
    # map = np.ones((100, 100))
    # p = ImageFeature.bresenham_line(0, 0, 99, 99)
    # plt.plot([0, 100], [0, 100])
    # for i, v in enumerate(p):
    #     plt.scatter(v[0], v[1])
    #     map[v[0], v[1]] = 2
    # plt.imshow(map.T)
    # plt.show()

    # Circle
    # theta = np.linspace(0,2*np.pi,50)
    # r = 100
    # x = np.zeros(len(theta))
    # y = np.zeros(len(theta))
    # for i in range(len(theta)):
    #     x[i] = r*np.cos(theta[i]) + np.random.normal(0,1) + 10
    #     y[i] = r*np.sin(theta[i]) + np.random.normal(0,1) + 5
    # a,b,rd = ImageFeature.circle_fit(x,y)
    # print(a,b,rd)
    # x_est = rd*np.cos(theta)
    # y_est = rd*np.sin(theta)
    # figure, axes = plt.subplots()
    # Drawing_colored_circle = plt.Circle((a,b),rd)
    # axes.set_aspect( 1 )
    # axes.add_artist( Drawing_colored_circle )
    # plt.title( 'Colored Circle' )
    # plt.scatter(x,y)
    # plt.show()