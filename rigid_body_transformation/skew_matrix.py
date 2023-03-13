import numpy as np

def skew(x):
    return np.array([[      0,  -x[2,0],  x[1,0]],
                     [ x[2,0],        0, -x[0,0]],
                     [-x[1,0],   x[0,0],       0]])
                     
if __name__ == "__main__":

    # Vector basis
    x = np.array([[1],
                [0],
                [0]])

    y = np.array([[0],
                [1],
                [0]])

    z = np.array([[0],
                [0],
                [1]])

    print(skew(x))
    print(skew(y))
    print(skew(z))
