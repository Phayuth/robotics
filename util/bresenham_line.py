import numpy as np

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


def bresenham_line_float(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx
    
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dErr = abs(y1 - y0)
    ystep = y0 > y1
    dX = x1 - x0

    err = dX >> 1
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





if __name__=="__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    import matplotlib.pyplot as plt
    import numpy as np

    map = np.ones((15,15))
    p = bresenham_line_float(0,0,8.5,6.2)
    print("==>> p: \n", p)


    plt.plot([0, 8.5], [0, 6.2])
    for i, v in enumerate(p):
        plt.scatter(v[0], v[1])
        map[v[0],v[1]] = 2
    plt.imshow(map.T)
    plt.show()