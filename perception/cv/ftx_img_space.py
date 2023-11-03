""" 
Hough Line
    Reference:
        - https://sbme-tutorials.github.io/2021/cv/notes/4_week4.html
        - https://www.youtube.com/watch?v=XRBc_xkZREg

Hough Circle
    Reference:
        - https://github.com/adityaintwala/Hough-Circle-Detection

Harris corners Detection
    Reference:
        - https://github.com/adityaintwala/Harris-Corner-Detection/blob/master/find_harris_corners.py
        - https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
"""

import numpy as np
from collections import defaultdict

class ImageFeature:

    def hough_line(binaryImage, threshold):
        numRows = binaryImage.shape[0]
        numCols = binaryImage.shape[1]
        maxDist = int(np.round(np.sqrt(numCols**2 + numRows**2)))  # max diatance is diagonal one
        thetas = np.deg2rad(np.arange(start=-90, stop=90))
        rhos = np.linspace(start=-maxDist, stop=maxDist, num=2 * maxDist)
        accumulator = np.zeros(shape=(2 * maxDist, len(thetas)))

        # update accumulator
        for rowIndex in range(numRows):
            for colIndex in range(numCols):
                if binaryImage[rowIndex, colIndex] > 0:
                    for thetaIndex in range(len(thetas)):

                        # calculate space parameter
                        r = colIndex * np.cos(thetas[thetaIndex]) + rowIndex * np.sin(thetas[thetaIndex])

                        # update count vote
                        rhoIndex = int(r) + maxDist
                        accumulator[rhoIndex, thetaIndex] += 1

        # apply threshold, minimum vote it should get to be considered as a line, it represents the minimum length of line that should be detected
        accumulatorMask = accumulator >= threshold
        rhoChosenIndex, thetaChosenIndex = np.where(accumulatorMask)

        rhosLines = rhos[rhoChosenIndex].astype(np.int64)
        thetasLines = thetas[thetaChosenIndex]
        candidatesLineParam = list(zip(rhosLines, thetasLines))

        return accumulator, candidatesLineParam  # distance in pixel, theta in rad

    def hough_circle(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):
        #image size
        img_height, img_width = edge_image.shape[:2]
        
        # R and Theta ranges
        dtheta = int(360 / num_thetas)
        
        ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
        thetas = np.arange(0, 360, step=dtheta)
        
        ## Radius ranges from r_min to r_max 
        rs = np.arange(r_min, r_max, step=delta_r)
        
        # Calculate Cos(theta) and Sin(theta) it will be required later
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        
        # Evaluate and keep ready the candidate circles dx and dy for different delta radius
        # based on the the parametric equation of circle.
        # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
        # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
        circle_candidates = []
        for r in rs:
            for t in range(num_thetas):
                #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
                #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
                #but its better to pre-calculate and use it here.
                circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
            
        # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
        # aready present in the dictionary instead of throwing exception.
        accumulator = defaultdict(int)
        
        for y in range(img_height):
            for x in range(img_width):
                if edge_image[y][x] != 0: #white pixel
                    # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                    for r, rcos_t, rsin_t in circle_candidates:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
            
        # Output image with detected lines drawn
        output_img = image.copy()
        # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
        out_circles = []
        
        # Sort the accumulator based on the votes for the candidate circles 
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, r = candidate_circle
            current_vote_percentage = votes / num_thetas
            if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
                out_circles.append((x, y, r, current_vote_percentage))
                print(x, y, r, current_vote_percentage)
        
        # Post process the results, can add more post processing later.
        if post_process :
            pixel_threshold = 5
            postprocess_circles = []
            for x, y, r, v in out_circles:
            # Exclude circles that are too close of each other
            # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
            # Remove nearby duplicate circles based on pixel_threshold
                if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                    postprocess_circles.append((x, y, r, v))
                out_circles = postprocess_circles
        
        # Draw shortlisted circles on the output image
        for x, y, r, v in out_circles:
            output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
        
        return output_img, out_circles

    def harris_corners(grayImage, k=0.04, windowSize=5, threshold=10000.0):
        cornersPixel = []
        offset = int(windowSize / 2)
        yRange = grayImage.shape[0] - offset
        xRange = grayImage.shape[1] - offset
        dy, dx = np.gradient(grayImage)
        Ixx = dx**2
        Ixy = dy * dx
        Iyy = dy**2

        for y in range(offset, yRange):
            for x in range(offset, xRange):
                # sliding window
                yStart = y - offset
                yEnd = y + offset + 1
                xStart = x - offset
                xEnd = x + offset + 1

                # The variable names are representative to the variable of the Harris corner equation
                windowIxx = Ixx[yStart:yEnd, xStart:xEnd]
                windowIxy = Ixy[yStart:yEnd, xStart:xEnd]
                windowIyy = Iyy[yStart:yEnd, xStart:xEnd]

                # Sum of squares of intensities of partial derevatives
                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()

                # Calculate determinant and trace of the matrix
                det = (Sxx*Syy) - (Sxy**2)
                trace = Sxx + Syy

                # Calculate R for Harris Corner equation
                # When |R| is small, which happens when λ1 and λ2 are small, the region is flat.
                # When R < 0, which happens when λ1 >> λ2 or vice versa, the region is edge.
                # When R is large, which happens when λ1 and λ2 are large and λ1 ∼ λ2, the region is a corner.
                R = det - k * (trace**2)
                if R > threshold:
                    cornersPixel.append((x, y))

        return cornersPixel

    def canny_edge():
        pass

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

    # Hough Line
    # binaryEdgeImage = np.zeros(shape=(150, 150))
    # binaryEdgeImage[:, :] = np.eye(150)
    # accumulator, lines = ImageFeature.hough_line(binaryEdgeImage, 150)
    # print(f"==>> accumulator: {accumulator}")
    # print(f"==>> lines: {lines}")
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(binaryEdgeImage)
    # axs[0].set_title("original")
    # axs[0].set_xlabel("cols")
    # axs[0].set_ylabel("rows")
    # axs[1].imshow(accumulator)
    # axs[1].set_title("hough space")
    # axs[1].set_xlabel("thetas")
    # axs[1].set_ylabel("rhos")
    # plt.show()

    # Hough Circle 
    # r_min = 10
    # r_max = 200
    # delta_r = 1
    # num_thetas = 100
    # bin_threshold = 0.4
    # min_edge_threshold = 100
    # max_edge_threshold = 200

    # input_img = cv2.imread("/home/yuth/Downloads/ex2.png")
        
    # #Edge detection on the input image
    # edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # #ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    # edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
    # circle_img, circles = ImageFeature.hough_circle(input_img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(input_img)
    # axs[0].set_title("original")
    # axs[1].imshow(edge_image, cmap='gray')
    # axs[1].set_title("gray")
    # axs[2].imshow(circle_img, cmap='gray')
    # axs[2].set_title("edge")
    # plt.show()


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

    # Harris corner
    corners = ImageFeature.harris_corners(imgGray, windowSize=2, threshold=1000)
    imgShow = np.zeros_like(imgGray)
    for x, y in corners:
        imgShow[y, x] = 1

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(20,20)
    axs[0].imshow(img)
    axs[0].set_title("original")
    axs[1].imshow(imgGray, cmap='gray')
    axs[1].set_title("gray")
    axs[2].imshow(imgShow, cmap='gray')
    axs[2].set_title("edge")
    plt.show()