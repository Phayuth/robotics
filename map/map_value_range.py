import numpy as np

def map_val(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def map_multi_val(in_array, in_min, in_max, out_min, out_max):
    out_array = np.zeros_like(in_array)
    for index,_ in enumerate(in_array):
        out_array[index,0] = (in_array[index,0] - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return out_array

if __name__ == "__main__":

    # SECTION - convert joint angle to pixel value (after inverse kinematic -> convert to pixel value -> planning)
    theta = np.pi
    pixel_val = map_val(theta, -np.pi, np.pi, 0, 360)
    print("==>> pixel_val: ", pixel_val)


    # SECTION - convert pixel value to joint angle (after planning -> convert to joint angle -> forward kinematic)
    pixel = 180
    theta_val = map_val(pixel, 0, 360, -np.pi, np.pi)
    print("==>> theta_val: ", theta_val)


    # SECTION - map value but the input is in array of shape (N,1)
    theta_arr = np.array([[np.pi], [-np.pi], [2]])
    pixel_val = map_multi_val(theta_arr, -np.pi, np.pi, 0, 360)
    print("==>> pixel_val: ", pixel_val)