import numpy as np

def map_val(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def map_multi_val(in_array, in_min, in_max, out_min, out_max):
    out_array = np.zeros_like(in_array)
    for index,_ in enumerate(in_array):
        out_array[index,0] = (in_array[index,0] - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return out_array

if __name__ == "__main__":

    # define min max of input output
    joint_max_val = np.pi
    joint_min_val = -np.pi
    pixel_min_val = 0
    pixel_max_val = 360

    # convert joint angle to pixel value (after inverse kinematic -> convert to pixel value -> planning)
    theta = np.pi
    pixel_val = map_val(theta, joint_min_val, joint_max_val, pixel_min_val, pixel_max_val)
    print("==>> pixel_val: ", pixel_val)

    # convert pixel value to joint angle (after planning -> convert to joint angle -> forward kinematic)
    pixel = 180
    theta_val = map_val(pixel, pixel_min_val, pixel_max_val, joint_min_val, joint_max_val)
    print("==>> theta_val: ", theta_val)

    # map value but the input is in array of shape (N,1)
    theta_arr = np.array([[np.pi], [-np.pi], [2]])
    pixel_val = map_multi_val(theta_arr, joint_min_val, joint_max_val, pixel_min_val, pixel_max_val)
    print("==>> pixel_val: ", pixel_val)