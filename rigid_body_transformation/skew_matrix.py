import numpy as np

def skew(x):
    return np.array([[      0,  -x[2,0],  x[1,0]],
                     [ x[2,0],        0, -x[0,0]],
                     [-x[1,0],   x[0,0],       0]])
                     