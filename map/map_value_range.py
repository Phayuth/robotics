import numpy as np

def map_val(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

if __name__ == "__main__":
    in_max = np.pi
    in_min = -np.pi

    out_min = 0
    out_max = 100

    theta = np.pi

    pixel_val = map_val(theta, in_min, in_max, out_min, out_max)
    print("==>> pixel_val: \n", pixel_val)